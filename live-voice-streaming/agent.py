import sys
from logger_setup import setup_logging
setup_logging()


# FIX: Import pyarrow BEFORE torch to prevent Windows DLL access violation
# when f5_tts imports datasets/Trainer later on.
import pyarrow
import torch
from faster_whisper import WhisperModel
import httpx
import os
import numpy as np
import threading
import queue
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Import the core F5 elements to build the model manually
from f5_tts.model.cfm import CFM
from f5_tts.model.backbones.dit import DiT
from f5_tts.infer.utils_infer import get_tokenizer, load_vocoder, infer_process

import soundfile as sf
from constants import PUNCTUATION_BOUNDARIES, MIN_WORDS_PER_CHUNK

# -----------------------------------------------------------------------------
# ASYNC PIPELINE QUEUES
# -----------------------------------------------------------------------------
stt_q = queue.Queue()
llm_q = queue.Queue()
tts_q = queue.Queue()
playback_q = queue.Queue()

def clear_queues():
    """Flushes all pending AI tasks if the user interrupts by speaking."""
    for q in [stt_q, llm_q, tts_q, playback_q]:
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break


chat_history = []

stt_model = None
vocoder = None
tts_model = None

def init_models():
    global stt_model, vocoder, tts_model
    
    if stt_model is not None:
        return

    print("⏳ Initializing models on GPU...", flush=True)
    stt_model = WhisperModel("large-v3", device="cuda", compute_type="int8")

    print("⏳ Initializing Native Bengali IndicF5 Architecture...", flush=True)
    vocoder = load_vocoder()

    vocab_file = os.getenv("VOCAB_FILE", "weights/vocab.txt")
    if not os.path.exists(vocab_file):
        print(f"❌ Error: Missing '{vocab_file}'!")
        exit(1)
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")

    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    dit_model = DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100)

    print("⏳ Initializing CFM...", flush=True)
    tts_model = CFM(
        transformer=dit_model,
        mel_spec_kwargs=dict(
            n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, 
            target_sample_rate=24000, mel_spec_type="vocos"
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    )
    print("⏳ Moving CFM to cuda...", flush=True)
    tts_model = tts_model.to("cuda")

    print("📦 Loading local offline weights...", flush=True)
    import safetensors.torch
    raw_state_dict = safetensors.torch.load_file("weights/model.safetensors")

    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        k = k.replace("ema_model.", "").replace("_orig_mod.", "")
        if not k.startswith("vocoder"):
            clean_state_dict[k] = v

    tts_model.load_state_dict(clean_state_dict, strict=False)
    print("✅ All AI Models Loaded Successfully!", flush=True)

# Run initialization synchronously so Uvicorn blocks until ready
init_models()



REF_AUDIO = os.getenv("REF_AUDIO", "assets/client_voice.wav")
REF_TEXT = os.getenv("REF_TEXT", "হ্যালো, কী করছ গো তুমি এখন, কী করছ গো তুমি?")

# -----------------------------------------------------------------------------
# WORKER THREADS
# -----------------------------------------------------------------------------
def stt_worker():
    while True:
        audio_array = stt_q.get()
        print("👂 [STT] Transcribing...")
        start_time = time.time()
        stt_lang = os.getenv("STT_LANGUAGE", "bn")
        segments, info = stt_model.transcribe(
            audio_array, 
            beam_size=5, 
            language=stt_lang, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        text = "".join([segment.text for segment in segments]).strip()
        stt_time = time.time() - start_time
        print(f"⏱️ [STT] Transcribed in {stt_time:.2f} seconds")
        if text:
            print(f"👤 You: {text}")
            llm_q.put(text)
        stt_q.task_done()

def llm_worker():
    global chat_history
    while True:
        user_text = llm_q.get()
        print(f"👤 [LLM] Received User Text: {user_text}")
        
        # Add user prompt to history
        chat_history.append({"role": "user", "content": f"Instruction: You must respond ONLY in clean, native Bengali script. Do not write any English or Banglish. Respond naturally to this user text: {user_text}"})
        
        print("🧠 [LLM] Thinking and streaming...")
        payload = {
            "model": "llama3:8b",
            "messages": chat_history,
            "stream": True
        }
        
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        ollama_url = f"http://{ollama_host}:{ollama_port}/api/chat"
        
        try:
            start_time = time.time()
            print("\n🤖 LLM Response: ", end="", flush=True)
            with httpx.stream("POST", ollama_url, json=payload, timeout=None) as r:
                sentence_buffer = ""
                full_response = ""
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            token = chunk["message"]["content"]
                            print(token, end="", flush=True)
                            sentence_buffer += token
                            full_response += token
                            
                            # Only chunk if the buffer ends with punctuation AND we have enough words
                            if sentence_buffer.strip() and sentence_buffer.strip()[-1] in PUNCTUATION_BOUNDARIES:
                                word_count = len(sentence_buffer.strip().split())
                                if word_count >= MIN_WORDS_PER_CHUNK:
                                    print(f"\n✂️ [LLM] Chunking at {word_count} words: '{sentence_buffer.strip()}'", flush=True)
                                    tts_q.put(sentence_buffer.strip())
                                    sentence_buffer = ""
                                else:
                                    pass
                                    # print(f"\n⏳ [LLM] Punctuation hit, but only {word_count} words. Accumulating: '{sentence_buffer.strip()}'", flush=True)
                
                # Push any remaining text that didn't end with punctuation
                if sentence_buffer.strip():
                    rem_count = len(sentence_buffer.strip().split())
                    print(f"\n🧹 [LLM] Flushing remaining {rem_count} words to TTS: '{sentence_buffer.strip()}'", flush=True)
                    tts_q.put(sentence_buffer.strip())
                print() # newline
                llm_time = time.time() - start_time
                print(f"⏱️ [LLM] Full response generated in {llm_time:.2f} seconds")
                chat_history.append({"role": "assistant", "content": full_response})
                
                # Keep history short to avoid context explosion
                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]
        except Exception as e:
            print(f"\n❌ [LLM] Error: {e}")
        
        llm_q.task_done()

def tts_worker():
    while True:
        sentence = tts_q.get()
        print(f"🗣️ [TTS] Synthesizing chunk: {sentence}")
        try:
            start_time = time.time()
            tts_nfe_step = int(os.getenv("TTS_NFE_STEP", "32"))
            tts_speed = float(os.getenv("TTS_SPEED", "1.0"))
            tts_cfg_strength = float(os.getenv("TTS_CFG_STRENGTH", "2.0"))
            tts_sway_coef = float(os.getenv("TTS_SWAY_COEF", "-1.0"))

            audio_output, sample_rate, _ = infer_process(
                ref_audio=REF_AUDIO,
                ref_text=REF_TEXT,
                gen_text=sentence,
                model_obj=tts_model,
                vocoder=vocoder,
                nfe_step=tts_nfe_step,
                speed=tts_speed,
                cfg_strength=tts_cfg_strength,
                sway_sampling_coef=tts_sway_coef
            )
            end_time = time.time()
            
            audio_data = audio_output
            if hasattr(audio_data, "cpu"):
                audio_data = audio_data.cpu().numpy()
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
                
            audio_length_ms = (len(audio_data) / sample_rate) * 1000
            print(f"⏱️ [TTS] Synthesized {audio_length_ms:.0f}ms of audio in {end_time - start_time:.2f} seconds", flush=True)
                
            playback_q.put((audio_data, sample_rate))
        except Exception as e:
            print(f"❌ [TTS] Error: {e}")
        tts_q.task_done()

def start_workers():
    # Start all worker threads
    threading.Thread(target=stt_worker, name="STT", daemon=True).start()
    threading.Thread(target=llm_worker, name="LLM", daemon=True).start()
    threading.Thread(target=tts_worker, name="TTS", daemon=True).start()
    print("✅ All worker threads started!")

if __name__ == "__main__":
    if not os.path.exists(REF_AUDIO):
        print(f"❌ Error: Could not find your reference file at {REF_AUDIO}!")
        exit(1)

    print("\n🚀 Starting Live Asynchronous Pipeline Workers...")
    start_workers()