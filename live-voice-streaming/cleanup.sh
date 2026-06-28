#!/bin/bash

echo "======================================"
echo "🧹 Starting system cleanup..."
echo "======================================"

echo "[1/3] 🛑 Killing relevant processes (uvicorn, python)..."
# Kill standard server and agent processes
pkill -9 -f uvicorn
pkill -9 -f server.py
pkill -9 -f agent.py

echo "[2/3] 🗑️ Clearing logs..."
# Empty out the logs directory if it exists
if [ -d "logs" ]; then
    rm -rf logs/*
    echo "  -> Cleared logs/ directory."
else
    echo "  -> Logs directory not found, skipping."
fi

# Clear any root-level ollama.log
if [ -f "ollama.log" ]; then
    rm -f ollama.log
    echo "  -> Cleared ollama.log"
fi

echo "[3/3] 🧠 Cleaning CPU and GPU memory..."
# Clear CPU RAM Caches (works since you are running as root)
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
echo "  -> CPU PageCache, dentries, and inodes dropped."

# Since we killed the processes, GPU memory should be automatically freed by the NVIDIA driver.
# Let's show the user the current GPU state if nvidia-smi is available.
if command -v nvidia-smi &> /dev/null; then
    echo "  -> GPU memory has been freed. Current GPU Status:"
    nvidia-smi
else
    echo "  -> NVIDIA GPU tools not found. If using CPU, memory is cleared."
fi

echo "======================================"
echo "✅ Cleanup complete!"
echo "======================================"
