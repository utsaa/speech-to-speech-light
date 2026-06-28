import torch
from faster_whisper import WhisperModel
import httpx
import os
import soundfile as sf
import numpy as np

# Import the core F5 elements to build the model manually
from f5_tts.model.cfm import CFM
from f5_tts.model.backbones.dit import DiT
from f5_tts.infer.utils_infer import get_tokenizer, load_checkpoint, load_vocoder, infer_process
from importlib.resources import files

# -----------------------------------------------------------------------------
# CONFIGURATION & SETUPS
# -----------------------------------------------------------------------------
print("⏳ Initializing models on GPU...")

# 1. The Ears: Faster-Whisper loaded in fast INT8 precision
stt_model = WhisperModel("large-v3", device="cuda", compute_type="int8")

# 2. The Voice: Build IndicF5 architecture manually and load local weights
print("⏳ Initializing Native Bengali IndicF5 Architecture...")

# Load Vocoder
vocoder = load_vocoder()

# Load Vocab from the offline weights folder (MUST be the IndicF5 specific vocab)
vocab_file = "weights/vocab.txt"
if not os.path.exists(vocab_file):
    print("❌ Error: Missing 'weights/vocab.txt'! You must download the vocab.txt file from the ai4bharat/IndicF5 repo on Hugging Face and place it in the 'weights' folder.")
    exit(1)
vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")

# Instantiate the Model Architecture
model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
dit_model = DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100)

tts_model = CFM(
    transformer=dit_model,
    mel_spec_kwargs=dict(
        n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, 
        target_sample_rate=24000, mel_spec_type="vocos"
    ),
    odeint_kwargs=dict(method="euler"),
    vocab_char_map=vocab_char_map,
).to("cuda")

print("📦 Loading local offline weights (Parsing PyTorch Checkpoint)...")
import safetensors.torch
raw_state_dict = safetensors.torch.load_file("weights/model.safetensors")

# The IndicF5 weights were saved wrapped in PyTorch's compiler and EMA objects.
# We manually strip the prefixes to map them cleanly onto our bare CFM model object.
clean_state_dict = {}
for k, v in raw_state_dict.items():
    k = k.replace("ema_model.", "").replace("_orig_mod.", "")
    if not k.startswith("vocoder"):  # We loaded vocoder separately
        clean_state_dict[k] = v

tts_model.load_state_dict(clean_state_dict, strict=False)

# 3. Reference voice tracks (Back to clean native Bengali script!)
REF_AUDIO = "assets/client_voice.wav"
REF_TEXT = "হ্যালো, কী করছ গো তুমি এখন, কী করছ গো তুমি?"

# -----------------------------------------------------------------------------
# CORE CONVERSATIONAL LOOP
# -----------------------------------------------------------------------------
def listen_and_respond():
    if not os.path.exists(REF_AUDIO):
        print(f"❌ Error: Could not find your reference file at {REF_AUDIO}!")
        return

    print("\n🤖 Agent is ready! Checking 'user_input.wav'...")
    if not os.path.exists("user_input.wav"):
        print("💡 Pro-Tip: Create a quick recording named 'user_input.wav' in this folder to test!")
        return
        
    # [A] SPEECH-TO-TEXT (STT)
    # Forced language='bn' so Whisper never accidentally transcribes in Urdu/Arabic script
    segments, info = stt_model.transcribe("user_input.wav", beam_size=5, language="bn")
    user_text = "".join([segment.text for segment in segments])
    print(f"👤 User Said (Script): {user_text}")
    
    # [B] LLM BRAIN (Ollama)
    print("🧠 Thinking...")
    response = httpx.post(
        "http://192.168.48.1:11434/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": (
                "Instruction: You must respond ONLY in clean, native Bengali script. Do not write any English or Banglish. "
                "Keep your response strictly to one short, simple sentence. "
                f"Respond naturally to this user text: {user_text}"
            ),
            "stream": False
        },
        timeout=None
    )
    llm_response = response.json()['response']
    print(f"🤖 Agent Response Text (Bengali): {llm_response}")
    
    # [C] ZERO-SHOT VOICE CLONING (TTS)
    print("🗣️ Synthesizing cloned native Bengali voice profile...")
    
    # Process audio generation natively with matched keyword hooks
    audio_output, sample_rate, _ = infer_process(
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        gen_text=llm_response,
        model_obj=tts_model,
        vocoder=vocoder,
        nfe_step=32
    )
    
    audio_data = audio_output
    if hasattr(audio_data, "cpu"):
        audio_data = audio_data.cpu().numpy()

    # Normalize if it's integer format
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    output_filename = "agent_response.wav"
    sf.write(output_filename, np.array(audio_data, dtype=np.float32), samplerate=sample_rate)
    print(f"🎉 Success! Real Bengali voice clone saved as '{output_filename}'")
    
    # Audio Playback
    os.system(f"aplay {output_filename} > /dev/null 2>&1 || play {output_filename} > /dev/null 2>&1")

if __name__ == "__main__":
    listen_and_respond()