import os
import re
import uuid
import torch
import torchaudio as ta
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from huggingface_hub import login

# ---------------------------------------------------------------------------
# Startup / Shutdown lifecycle
# ---------------------------------------------------------------------------

model_store: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the TTS model into VRAM once at startup and keep it warm."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("[ML Service] Loading ChatterboxMultilingualTTS into VRAM…")
    # Import here so the process fails fast if the package is missing
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_store["tts"] = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    print("[ML Service] Model ready.")

    yield  # <-- application runs here

    # Cleanup (optional – frees VRAM on graceful shutdown)
    model_store.clear()
    torch.cuda.empty_cache()
    print("[ML Service] Model unloaded.")


app = FastAPI(title="Chatterbox TTS Microservice", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesise")
    audio_prompt_path: str = Field(
        "data/shweta_singh.wav",
        description="Path to the voice-cloning reference WAV file",
    )
    language_id: str = Field("hi", description="BCP-47 language code, e.g. 'hi', 'en'")
    max_chars: int = Field(300, ge=50, le=800)
    exaggeration: float = Field(0.5, ge=0.0, le=1.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(0.8, ge=0.1, le=2.0)

# ---------------------------------------------------------------------------
# Core chunking + generation logic (ported directly from your script)
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split text into safe-sized chunks by punctuation boundaries.
    Handles Hindi full stop '।' as well as ASCII sentence endings.
    """
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence is longer than max_chars, force-split it
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i : i + max_chars])
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def generate_audio(req: TTSRequest) -> tuple[torch.Tensor, int]:
    """Run the chunked TTS generation loop and return stitched tensor + sample rate."""
    model = model_store["tts"]
    chunks = chunk_text(req.text, req.max_chars)

    if not chunks:
        raise ValueError("No text chunks produced – check your input.")

    audio_chunks: list[torch.Tensor] = []

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}/{len(chunks)}  ({len(chunk)} chars)")
        wav = model.generate(
            chunk,
            language_id=req.language_id,
            audio_prompt_path=req.audio_prompt_path,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
        )
        audio_chunks.append(wav)

    stitched = torch.cat(audio_chunks, dim=-1)
    return stitched, model.sr

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "tts" in model_store}


@app.post("/synthesise")
async def synthesise(req: TTSRequest):
    """
    Accepts a JSON payload, runs TTS, and returns the WAV file.
    The file is written to a temp directory and served with FileResponse.
    """
    if "tts" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not os.path.isfile(req.audio_prompt_path):
        raise HTTPException(
            status_code=400,
            detail=f"audio_prompt_path not found: {req.audio_prompt_path}",
        )

    try:
        audio_tensor, sample_rate = generate_audio(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    # Save to a temporary file
    os.makedirs("tmp", exist_ok=True)
    out_path = f"tmp/{uuid.uuid4().hex}.wav"
    ta.save(out_path, audio_tensor, sample_rate)
    print(f"[ML Service] Saved → {out_path}")

    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename="synthesis.wav",
        background=None,  # FileResponse handles cleanup lazily
    )
