import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

print("Importing torch", flush=True)
import torch
print("Importing WhisperModel", flush=True)
from faster_whisper import WhisperModel
print("Importing httpx", flush=True)
import httpx
print("Importing soundfile", flush=True)
import soundfile as sf
print("Importing numpy", flush=True)
import numpy as np
print("Importing sounddevice", flush=True)
import sounddevice as sd
print("Importing f5_tts CFM", flush=True)
from f5_tts.model.cfm import CFM
print("Importing f5_tts DiT", flush=True)
from f5_tts.model.backbones.dit import DiT
print("Importing f5_tts utils", flush=True)
from f5_tts.infer.utils_infer import get_tokenizer, load_vocoder, infer_process
print("All imports done", flush=True)
