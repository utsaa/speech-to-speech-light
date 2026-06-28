import sys

print("Importing torch", flush=True)
import torch
print("Importing WhisperModel", flush=True)
from faster_whisper import WhisperModel
print("Importing f5_tts", flush=True)
from f5_tts.model.cfm import CFM
print("Importing f5_tts utils", flush=True)
from f5_tts.infer.utils_infer import get_tokenizer, load_vocoder, infer_process
print("All imports succeeded", flush=True)
