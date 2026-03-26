# Vāk — Neural Voice Studio
## Chatterbox TTS Microservice Architecture

```
┌──────────────────────┐      POST /api/synthesise       ┌─────────────────────────────┐
│   Browser (HTML/JS)  │ ──────────────────────────────► │  Node.js Gateway  :3000     │
│                      │ ◄── audio/wav stream ─────────── │  (Express)                  │
└──────────────────────┘                                  └────────────┬────────────────┘
                                                                       │ POST /synthesise
                                                                       │ (internal)
                                                          ┌────────────▼────────────────┐
                                                          │  Python ML Service  :8000   │
                                                          │  (FastAPI + Chatterbox)     │
                                                          │  model warm in VRAM         │
                                                          └─────────────────────────────┘
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| CUDA-capable GPU | VRAM ≥ 6 GB recommended |
| Node.js | 18+ |
| npm | 9+ |

---

## Directory Layout

```
tts_service/
├── ml_service/
│   ├── main.py              ← FastAPI app
│   ├── requirements.txt
│   ├── .env                 ← create this (see below)
│   ├── data/
│   │   └── shweta_singh.wav ← your voice reference file
│   └── tmp/                 ← auto-created; holds generated WAVs
│
└── gateway/
    ├── server.js            ← Express app
    ├── package.json
    └── public/
        └── index.html       ← Frontend UI
```

---

## Step 1 — Set up the Python ML Service

```bash
cd tts_service/ml_service

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If chatterbox-tts is not yet on PyPI, install from source:
# git clone https://github.com/resemble-ai/chatterbox.git
# pip install -e ./chatterbox
```

### Create the `.env` file

```bash
# tts_service/ml_service/.env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Place your voice reference file

Copy your WAV cloning reference into `ml_service/data/`:

```bash
mkdir -p data
cp /path/to/your/voice.wav data/shweta_singh.wav
```

### Start the FastAPI server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Important:** Use `--workers 1` — the GPU model is a singleton. Multiple workers would each
> try to load the model and exhaust VRAM.

You should see:
```
[ML Service] Loading ChatterboxMultilingualTTS into VRAM…
[ML Service] Model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify: `curl http://localhost:8000/health`

---

## Step 2 — Set up the Node.js Gateway

```bash
cd tts_service/gateway
npm install
```

### Environment variables (optional)

```bash
# defaults are fine for local dev
export PORT=3000
export ML_SERVICE_URL=http://localhost:8000/synthesise
```

### Start the Express gateway

```bash
node server.js
# or for auto-reload during development:
npm run dev
```

You should see:
```
[Gateway] Listening on http://localhost:3000
[Gateway] ML service URL: http://localhost:8000/synthesise
```

---

## Step 3 — Open the UI

Navigate to **http://localhost:3000** in your browser.

1. Type or paste text into the textarea.
2. Adjust voice parameters in the right panel.
3. Click **Generate Audio**.
4. The `<audio>` player will appear when the WAV is ready.
5. Use **↓ Download WAV** to save the file.

---

## Running Both Servers Simultaneously

### Option A — Two terminal tabs (simplest)

**Terminal 1:**
```bash
cd tts_service/ml_service && source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Terminal 2:**
```bash
cd tts_service/gateway
node server.js
```

### Option B — `concurrently` (one command)

```bash
npm install -g concurrently
concurrently \
  "cd ml_service && .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1" \
  "cd gateway && node server.js"
```

### Option C — Docker Compose (production)

```yaml
# docker-compose.yml (skeleton — adapt to your image names)
version: "3.9"
services:
  ml_service:
    build: ./ml_service
    ports: ["8000:8000"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    env_file: ./ml_service/.env

  gateway:
    build: ./gateway
    ports: ["3000:3000"]
    environment:
      ML_SERVICE_URL: http://ml_service:8000/synthesise
    depends_on: [ml_service]
```

---

## API Reference

### ML Service (FastAPI)

#### `GET /health`
Returns `{"status": "ok", "model_loaded": true}`.

#### `POST /synthesise`

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to synthesise |
| `audio_prompt_path` | string | `"data/shweta_singh.wav"` | Path to reference WAV |
| `language_id` | string | `"hi"` | BCP-47 language code |
| `max_chars` | int | `300` | Max chars per chunk (50–800) |
| `exaggeration` | float | `0.5` | Emotional intensity (0–1) |
| `cfg_weight` | float | `0.5` | Reference adherence (0–1) |
| `temperature` | float | `0.8` | Stability / creativity (0.1–2.0) |

**Response:** `audio/wav` binary stream.

### Gateway (Express)

#### `POST /api/synthesise`
Same JSON body as ML service. Proxies the audio stream back to the client.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Lower `max_chars`; close other GPU processes |
| `503 ML service unavailable` | Start the Python server first |
| `audio_prompt_path not found` | Verify the WAV path relative to where uvicorn is launched |
| Garbled / cut-off audio | Reduce `temperature` to 0.3–0.5; reduce `exaggeration` |
| Cross-lingual accent issues | Set `cfg_weight=0.0` to free the model from English reference pacing |
