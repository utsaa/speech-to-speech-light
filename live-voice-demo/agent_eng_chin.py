import torch
from faster_whisper import WhisperModel
import httpx
import os
import soundfile as sf
from f5_tts.api import F5TTS  # Natively imports F5-TTS

# -----------------------------------------------------------------------------
# CONFIGURATION & SETUPS
# -----------------------------------------------------------------------------
print("⏳ Initializing models on GPU... (First time will take a few minutes to fetch weights)")

# 1. The Ears: Faster-Whisper loaded in fast INT8 precision
stt_model = WhisperModel("large-v3", device="cuda", compute_type="int8")

# 2. The Voice: F5-TTS loads the base multilingual voice cloner model
tts_model = F5TTS(device="cuda")

# 3. Reference voice path (The biometric target profile)
REF_AUDIO = "assets/client_voice.wav"
# CRITICAL: Write EXACTLY what is spoken in the 5-second reference file
# REF_TEXT = "হ্যালো, কী করছ গো তুমি এখন, কী করছ গো তুমি?"
REF_TEXT = "হলো কি করছো গতুমি আখন কি করছো গতুমি"

# -----------------------------------------------------------------------------
# CORE CONVERSATIONAL LOOP
# -----------------------------------------------------------------------------
def listen_and_respond():
    if not os.path.exists(REF_AUDIO):
        print(f"❌ Error: Could not find your reference file at {REF_AUDIO}!")
        return

    print("\n🤖 Agent is ready! To run a quick test, ensure you have a 'user_input.wav' file.")
    print("👂 Processing user input file...")
    
    # [A] SPEECH-TO-TEXT (STT) - Translates user audio into text string
    # Replace 'user_input.wav' with your test mic recording or static audio input
    if not os.path.exists("user_input.wav"):
        print("💡 Pro-Tip: Create a quick recording named 'user_input.wav' in this folder to test!")
        return
        
    segments, info = stt_model.transcribe("user_input.wav", beam_size=5)
    user_text = "".join([segment.text for segment in segments])
    print(f"👤 User Said (Script): {user_text}")
    
    # [B] LLM BRAIN - Sends transcribed text directly to local Ollama server
    print("🧠 Thinking...")
    response = httpx.post(
        "http://192.168.48.1:11434/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": (
                "Instruction: You must respond ONLY in clean Bengali script. Do not write any English. "
                "Use standard punctuation like question marks (?) and simple periods. "
                f"Respond naturally and very briefly (1 short sentence) to this user text: {user_text}"
            ),
            "stream": False
        },
        timeout=None
    )
    llm_response = response.json()['response']
    print(f"🤖 Agent Response Text: {llm_response}")
    
    # [C] ZERO-SHOT VOICE CLONING (TTS) - Synthesizes response text into cloned voice
    print("🗣️ Synthesizing cloned voice profile...")
    
    audio, sample_rate, _ = tts_model.infer(
        ref_file=REF_AUDIO,
        ref_text=REF_TEXT,
        gen_text=llm_response,
        nfe_step=32  # 32 steps provides an ideal speed-to-clarity ratio
    )

    # Save the output file
    output_filename = "agent_response.wav"
    sf.write(output_filename, audio, sample_rate)
    print(f"🎉 Success! Cloned response saved as '{output_filename}'")
    
    # Tries to play it back natively on your Linux system speakers
    os.system(f"aplay {output_filename} > /dev/null 2>&1 || play {output_filename} > /dev/null 2>&1")

if __name__ == "__main__":
    listen_and_respond()
