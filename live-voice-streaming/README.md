# Speech-to-Speech Light 🎙️

A real-time, ultra-low latency voice AI pipeline built for seamless conversational experiences. It uses a modern asynchronous architecture to process Speech-to-Text (STT), Large Language Model (LLM) generation, and Text-to-Speech (TTS) concurrently.

## Features
- **Real-Time WebSockets:** Audio streams bidirectionally between the client browser and the FastAPI server.
- **Client-Side VAD:** Voice Activity Detection is processed dynamically on the server from raw PCM chunks sent by the browser.
- **Asynchronous AI Workers:** STT, LLM, and TTS models run on separate background daemon threads communicating via thread-safe queues.
- **Streaming LLM + Chunked TTS:** The LLM streams its response token-by-token. Sentences are chunked at punctuation boundaries and sent immediately to the TTS engine to minimize perceived latency.
- **Native Bengali Support:** Optimized for Bengali using an IndicF5 TTS architecture and local LLaMA-3 text generation.

## Prerequisites
- **Python 3.10+**
- **Ollama:** Running locally with the `llama3:8b` model.
- **CUDA/GPU:** Recommended for STT (Whisper) and TTS (F5-TTS) inference.

## Environment Variables
Create a `.env` file in this directory (or configure your environment) with the following variables:
- `VAD_THRESHOLD` (default: `0.015`) - Sensitivity for voice activity detection.
- `OLLAMA_HOST` (default: `localhost`)
- `OLLAMA_PORT` (default: `11434`)
- `STT_LANGUAGE` (default: `bn`)
- `REF_AUDIO` - Path to the reference audio for zero-shot voice cloning.
- `REF_TEXT` - Text corresponding to the reference audio.

## Running the Server
Start the application using Uvicorn:
```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8080 --reload
```
Navigate to `http://localhost:8080` to interact with the agent!

## Architecture
For a deep dive into the system design, thread management, and data flow, see [architecture.md](architecture.md).
