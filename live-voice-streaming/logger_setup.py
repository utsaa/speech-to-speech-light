import sys
import io
import os
from datetime import datetime

import threading

class TeeLogger:
    def __init__(self, stream, timestamp):
        self.stream = stream
        self.timestamp = timestamp
        self.run_dir = os.path.join("logs", f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.files = {}
        self.lock = threading.Lock()

    def get_file(self, thread_name):
        with self.lock:
            if thread_name not in self.files:
                filepath = os.path.join(self.run_dir, f"{thread_name}.log")
                self.files[thread_name] = open(filepath, "a", encoding="utf-8")
            return self.files[thread_name]

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        
        thread_name = threading.current_thread().name
        if thread_name == "MainThread":
            thread_name = "Server"
            
        f = self.get_file(thread_name)
        f.write(data)
        f.flush()

    def flush(self):
        self.stream.flush()
        with self.lock:
            for f in self.files.values():
                f.flush()

def setup_logging():
    if isinstance(sys.stdout, TeeLogger):
        return
        
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Fix Windows console encoding to support emojis and Bengali text
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

    sys.stdout = TeeLogger(sys.stdout, timestamp)
    sys.stderr = TeeLogger(sys.stderr, timestamp)
