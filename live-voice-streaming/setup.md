# Setup Instructions

## 1. Install uv

You can install `uv` using the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, ensure that `~/.local/bin` (or the directory mentioned in the installer output) is in your `PATH`.

## 2. Install FFmpeg

You can install FFmpeg directly using `apt`. If you are in an environment where `sudo` isn't required (like a root container session), you can run:

```bash
apt update
apt upgrade -y
apt install -y ffmpeg zstd
```

## 3. Install Ollama

You can install Ollama on Linux using their official installation script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Once installed, the Ollama service typically starts automatically in the background (via `systemd`). You can then download the `llama3:8b` model required by the app:

```bash
ollama serve
ollama pull llama3:8b
```

*(Note: If you are running inside a Docker container or an environment without `systemd`, you may need to manually start the server first by running `ollama serve &` before pulling).*

To verify that the model was downloaded successfully, you can query the Ollama tags endpoint:

```bash
curl http://localhost:11434/api/tags
```

This will return a JSON list of all available models, which should now include `llama3:8b`.
