import os
from dotenv import load_dotenv

load_dotenv()
import io
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from logger_setup import setup_logging

# Set up logging immediately so we capture all AI model loading logs
setup_logging()

# Import from our refactored agent

from agent import start_workers, stt_q, playback_q, REF_AUDIO

app = FastAPI()

# Make sure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    if not os.path.exists(REF_AUDIO):
        print(f"❌ Error: Could not find your reference file at {REF_AUDIO}!")
        exit(1)

    print("\n🚀 Starting Live Asynchronous Pipeline Workers...")
    start_workers()
    
    # Start the async task that pulls from playback_q and broadcasts to all clients
    asyncio.create_task(broadcast_playback())

@app.on_event("shutdown")
async def shutdown_event():
    print("\n🛑 Shutting down workers...")
    # Push a poison pill to unblock the ThreadPoolExecutor
    playback_q.put((None, None))

# To store connected websocket clients
connected_clients = set()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    ui_vad = os.getenv("VAD_THRESHOLD", "0.015")
    html = html.replace("{{VAD_THRESHOLD}}", ui_vad)
    return html

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"🌐 [WebSocket] Client connected: {websocket.client}")
    
    recording = []
    is_speaking = False
    silence_timer = 0
    silence_timer = 0
    
    try:
        while True:
            # Receive raw float32 PCM audio bytes from browser
            data = await websocket.receive_bytes()
            
            # Convert bytes back to numpy float32 array
            indata = np.frombuffer(data, dtype=np.float32)
            
            # Calculate RMS volume for VAD
            rms = np.sqrt(np.mean(np.square(indata)))
            vad_threshold = float(os.getenv("VAD_THRESHOLD", "0.015"))
            
            if rms > vad_threshold:
                print(f"📊 [VAD] Current RMS: {rms:.4f}")
                if not is_speaking:
                    print("\n🎙️ [VAD] Speech detected. Recording...")
                    # clear_queues() # Flush any pending STT/LLM/TTS workloads
                is_speaking = True
                silence_timer = 0
                recording.append(indata.copy())
            elif is_speaking:
                recording.append(indata.copy())
                # Dynamically calculate the exact time duration of this audio chunk
                chunk_duration = len(indata) / 16000.0
                silence_timer += chunk_duration
                if silence_timer > 1.5:  # 1.5 seconds of silence stops the recording
                    print("🛑 [VAD] Silence detected. Queuing to STT.")
                    is_speaking = False
                    audio_data = np.concatenate(recording, axis=0)
                    stt_q.put(audio_data.flatten())
                    recording.clear()
                    
    except WebSocketDisconnect:
        print(f"🌐 [WebSocket] Client disconnected: {websocket.client}")
        connected_clients.remove(websocket)
    except Exception as e:
        print(f"❌ [WebSocket] Error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

async def broadcast_playback():
    """
    Continuously poll playback_q for generated audio chunks and send them
    over the active WebSockets to be played by the browser.
    """
    while True:
        # Wait asynchronously for the next item in the queue (no polling needed)
        audio_data, sample_rate = await asyncio.to_thread(playback_q.get)
        
        # Check for poison pill during shutdown
        if audio_data is None:
            break
        

        
        audio_length_ms = (len(audio_data) / sample_rate) * 1000
        print(f"🔊 [WebSocket] Broadcasted {audio_length_ms:.0f}ms of generated audio chunk.")

        # Convert float32 numpy array to bytes
        audio_bytes = audio_data.astype(np.float32).tobytes()
        
        # Send to all connected clients
        dead_clients = set()
        for client in connected_clients:
            try:
                await client.send_bytes(audio_bytes)
            except Exception as e:
                print(f"❌ [WebSocket] Broadcast failed: {e}")
                dead_clients.add(client)
                
        for c in dead_clients:
            connected_clients.remove(c)
            
        playback_q.task_done()

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    
    print(f"🚀 Starting FastAPI Server on http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=True)
