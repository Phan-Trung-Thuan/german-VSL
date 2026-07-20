"""
German Speech → German Text → German Sign Language Gloss → Animated GIF
Combined pipeline backend (FastAPI).

Pipeline:
  1. /api/transcribe  – Upload audio → Canary ASR → German text
  2. /api/gloss       – German text  → DGS gloss via spoken_to_signed
  3. /api/gif         – German text  → DGS gloss → pose → GIF (full pipeline)

Environment variables:
  ASR_MODEL   – override Canary model id (default: nvidia/canary-1b)
  LEXICON_DIR – path to a CSVPoseLookup lexicon directory
                (default: auto-detect dummy_lexicon in gloss_to_gif sub-repo)
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]  # …/aa/pipeline
AA_ROOT = PIPELINE_ROOT.parent                        # …/aa

FRONTEND_DIR = PIPELINE_ROOT / "frontend"

# canary_web_asr sub-repo backend lives one level up
CANARY_BACKEND = AA_ROOT / "canary_web_asr" / "canary_web_asr"

# gloss_to_gif sub-repo
GLOSS_REPO = AA_ROOT / "gloss_to_gif" / "gloss_to_gif"
DUMMY_LEXICON = GLOSS_REPO / "assets" / "dummy_lexicon"


def _ensure_gloss_repo_in_path():
    """Add gloss_to_gif to sys.path only when needed (lazy, avoids polluting test imports)."""
    if str(GLOSS_REPO) not in sys.path:
        sys.path.insert(0, str(GLOSS_REPO))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("ASR_MODEL", "nvidia/canary-1b")
MODELS_DIR = CANARY_BACKEND / "models"
LOCAL_MODEL_PATH = MODELS_DIR / f"{MODEL_ID.split('/')[-1]}.nemo"
TARGET_SAMPLE_RATE = 16_000

LEXICON_DIR = Path(os.getenv("LEXICON_DIR", str(DUMMY_LEXICON)))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DE Speech → Sign Language Pipeline",
    description="German speech → Canary ASR → German Sign Language gloss → animated GIF",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# ASR state
# ---------------------------------------------------------------------------
asr_model = None
model_device = "unknown"
loaded_model_id = None


# ---------------------------------------------------------------------------
# ASR helpers (ported from canary_web_asr/backend/main.py)
# ---------------------------------------------------------------------------

def _extract_text(item: Any) -> str:
    if hasattr(item, "text"):
        return str(item.text)
    if isinstance(item, dict) and "text" in item:
        return str(item["text"])
    return str(item)


def _load_asr_model():
    global asr_model, model_device, loaded_model_id
    if asr_model is not None:
        return asr_model
    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except Exception as exc:
        raise RuntimeError(f"Cannot import NeMo/Torch: {exc}") from exc

    if LOCAL_MODEL_PATH.exists():
        print(f"[ASR] Loading local model: {LOCAL_MODEL_PATH}")
        model = nemo_asr.models.ASRModel.restore_from(str(LOCAL_MODEL_PATH))
    else:
        print(f"[ASR] Downloading from HuggingFace: {MODEL_ID}")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)

    try:
        import torch
        if torch.cuda.is_available():
            model = model.cuda()
            model_device = torch.cuda.get_device_name(0)
        else:
            model_device = "cpu"
    except Exception:
        model_device = "cpu"

    model.eval()
    asr_model = model
    loaded_model_id = MODEL_ID
    print(f"[ASR] Ready on {model_device} — {MODEL_ID}")
    return asr_model


def _convert_to_16k_mono(input_path: Path, output_path: Path) -> float:
    try:
        audio, sr = sf.read(str(input_path), always_2d=False)
    except Exception:
        audio, sr = librosa.load(str(input_path), sr=None, mono=False)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        axis = 0 if audio.shape[0] <= 8 else 1
        audio = np.mean(audio, axis=axis)

    audio = np.nan_to_num(audio)
    duration = float(len(audio) / sr) if sr else 0.0

    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

    sf.write(str(output_path), audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return duration


def _do_transcribe(model, wav_path: str) -> str:
    common_kwargs = dict(batch_size=1)
    for kwargs in [
        dict(source_lang="de", target_lang="de", pnc="yes", **common_kwargs),
        dict(source_lang="de", target_lang="de", **common_kwargs),
        common_kwargs,
    ]:
        try:
            outputs = model.transcribe([wav_path], **kwargs)
            return _extract_text(outputs[0]) if outputs else ""
        except TypeError:
            continue
    outputs = model.transcribe([wav_path])
    return _extract_text(outputs[0]) if outputs else ""


# ---------------------------------------------------------------------------
# Gloss + GIF helpers (uses spoken_to_signed from gloss_to_gif sub-repo)
# ---------------------------------------------------------------------------

def _text_to_gloss(text: str, spoken_language: str = "de", glosser: str = "simple") -> list:
    """Convert German text → list of gloss sentences."""
    try:
        from spoken_to_signed.text_to_gloss.simple import text_to_gloss as _fn
        import importlib
        module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
        return module.text_to_gloss(text=text, language=spoken_language)
    except ImportError:
        _ensure_gloss_repo_in_path()
        from spoken_to_signed.bin import _text_to_gloss as _ttg
        return _ttg(text, spoken_language, glosser)


def _gloss_to_gif(
    text: str,
    lexicon_dir: Path,
    spoken_language: str = "de",
    signed_language: str = "sgg",
    glosser: str = "simple",
    gif_width: int = 400,
) -> tuple[str, list, Path]:
    """
    Full sub-pipeline: text → gloss → pose → GIF.
    Returns (gloss_str, sentences, gif_path).
    gif_path is a temp file — caller is responsible for cleanup.
    """
    from itertools import chain as ichain
    from pose_format.pose_visualizer import PoseVisualizer

    # Try pip-installed spoken_to_signed first; fall back to local gloss_to_gif repo
    try:
        import importlib
        module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
        sentences = module.text_to_gloss(text=text, language=spoken_language)
        from spoken_to_signed.gloss_to_pose import CSVPoseLookup, gloss_to_pose, concatenate_poses
        from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup

        gloss_tokens: list[str] = [gloss for sentence in sentences for _, gloss in sentence]
        gloss_str = " ".join(gloss_tokens)

        backup = FingerspellingPoseLookup()
        lookup = CSVPoseLookup(str(lexicon_dir), backup=backup)
        poses = [gloss_to_pose(gloss, lookup, spoken_language, signed_language) for gloss in sentences]
        if len(poses) == 1:
            result_pose = poses[0]
        else:
            from spoken_to_signed.gloss_to_pose.concatenate import concatenate_poses as _cp
            result_pose = _cp(poses, trim=False)

    except ImportError:
        # Fallback: local repo
        _ensure_gloss_repo_in_path()
        from spoken_to_signed.bin import _text_to_gloss as _ttg, _gloss_to_pose
        sentences = _ttg(text, spoken_language, glosser)
        gloss_tokens = [gloss for sentence in sentences for _, gloss in sentence]
        gloss_str = " ".join(gloss_tokens)
        fallback_result = _gloss_to_pose(sentences, str(lexicon_dir), spoken_language, signed_language)
        result_pose = fallback_result.pose

    # Step 3: pose → GIF (scale to gif_width)
    pose = result_pose
    scale = pose.header.dimensions.width / gif_width
    pose.header.dimensions.width = int(pose.header.dimensions.width / scale)
    pose.header.dimensions.height = int(pose.header.dimensions.height / scale)
    pose.body.data = pose.body.data / scale

    # Write to a temp GIF
    tmp_gif = Path(tempfile.mktemp(suffix=".gif"))
    visualizer = PoseVisualizer(pose)
    visualizer.save_gif(str(tmp_gif), visualizer.draw())

    return gloss_str, sentences, tmp_gif


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    lexicon_ok = LEXICON_DIR.exists() and (LEXICON_DIR / "index.csv").exists()
    return {
        "status": "ok",
        "asr_model": loaded_model_id or MODEL_ID,
        "asr_model_loaded": asr_model is not None,
        "device": model_device,
        "lexicon_dir": str(LEXICON_DIR),
        "lexicon_ok": lexicon_ok,
        "gloss_repo": str(GLOSS_REPO),
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Step 1: German audio → German text (Canary ASR)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    suffix = Path(file.filename).suffix.lower() or ".wav"
    if suffix not in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    try:
        model = _load_asr_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        raw_path = tmp_dir / f"input{suffix}"
        wav_path = tmp_dir / "input_16k.wav"

        raw_path.write_bytes(await file.read())

        try:
            duration = _convert_to_16k_mono(raw_path, wav_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Cannot read audio: {exc}") from exc

        t0 = time.perf_counter()
        try:
            text = _do_transcribe(model, str(wav_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Transcribe error: {exc}") from exc
        elapsed = time.perf_counter() - t0

    return {
        "text": text,
        "language": "de",
        "model": loaded_model_id or MODEL_ID,
        "duration_seconds": round(duration, 3),
        "inference_seconds": round(elapsed, 3),
        "device": model_device,
    }


@app.post("/api/gloss")
async def gloss_endpoint(body: dict):
    """Step 2: German text → DGS gloss."""
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required.")

    glosser = body.get("glosser", "simple")
    spoken_language = body.get("spoken_language", "de")

    try:
        sentences = _text_to_gloss(text, spoken_language, glosser)
        gloss_tokens = [gloss for sentence in sentences for _, gloss in sentence]
        return {
            "text": text,
            "gloss": " ".join(gloss_tokens),
            "gloss_tokens": gloss_tokens,
            "sentences": [[{"word": w, "gloss": g} for w, g in s] for s in sentences],
            "glosser": glosser,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gloss error: {exc}") from exc


@app.post("/api/gif")
async def gif_endpoint(body: dict):
    """Step 3: German text → DGS gloss → pose → GIF (base64)."""
    import base64

    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required.")

    glosser = body.get("glosser", "simple")
    spoken_language = body.get("spoken_language", "de")
    signed_language = body.get("signed_language", "sgg")
    gif_width = int(body.get("gif_width", 400))
    lexicon = Path(body.get("lexicon_dir", str(LEXICON_DIR)))

    if not lexicon.exists() or not (lexicon / "index.csv").exists():
        raise HTTPException(
            status_code=400,
            detail=f"Lexicon not found at: {lexicon}. "
                   "Provide a valid 'lexicon_dir' or set LEXICON_DIR env var.",
        )

    try:
        gloss_str, sentences, gif_path = _gloss_to_gif(
            text, lexicon, spoken_language, signed_language, glosser, gif_width
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GIF generation error: {exc}") from exc

    try:
        gif_bytes = gif_path.read_bytes()
        gif_b64 = base64.b64encode(gif_bytes).decode()
    finally:
        gif_path.unlink(missing_ok=True)

    gloss_tokens = [gloss for sentence in sentences for _, gloss in sentence]

    return {
        "text": text,
        "gloss": gloss_str,
        "gloss_tokens": gloss_tokens,
        "gif_base64": gif_b64,
        "gif_size_bytes": len(gif_bytes),
        "signed_language": signed_language,
        "glosser": glosser,
    }


@app.post("/api/pipeline")
async def full_pipeline(file: UploadFile = File(...), glosser: str = "simple",
                        signed_language: str = "sgg", gif_width: int = 400):
    """
    Full pipeline in one call:
    German audio → German text (ASR) → DGS gloss → animated GIF.
    Returns JSON with transcript, gloss tokens, and GIF as base64.
    """
    import base64

    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    suffix = Path(file.filename).suffix.lower() or ".wav"
    if suffix not in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    # ── Step 1: ASR ──────────────────────────────────────────────────────────
    try:
        model = _load_asr_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR model load error: {exc}") from exc

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        raw_path = tmp_dir / f"input{suffix}"
        wav_path = tmp_dir / "input_16k.wav"

        raw_path.write_bytes(await file.read())

        try:
            duration = _convert_to_16k_mono(raw_path, wav_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Cannot read audio: {exc}") from exc

        t_asr0 = time.perf_counter()
        try:
            text = _do_transcribe(model, str(wav_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Transcribe error: {exc}") from exc
        asr_time = time.perf_counter() - t_asr0

    if not text.strip():
        raise HTTPException(status_code=422, detail="ASR returned empty transcript.")

    # ── Step 2+3: Gloss + GIF ────────────────────────────────────────────────
    t_gloss0 = time.perf_counter()
    try:
        gloss_str, sentences, gif_path = _gloss_to_gif(
            text, LEXICON_DIR, "de", signed_language, glosser, gif_width
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gloss/GIF error: {exc}") from exc
    gloss_time = time.perf_counter() - t_gloss0

    try:
        gif_bytes = gif_path.read_bytes()
        gif_b64 = base64.b64encode(gif_bytes).decode()
    finally:
        gif_path.unlink(missing_ok=True)

    gloss_tokens = [gloss for sentence in sentences for _, gloss in sentence]

    return {
        "transcript": text,
        "gloss": gloss_str,
        "gloss_tokens": gloss_tokens,
        "gif_base64": gif_b64,
        "gif_size_bytes": len(gif_bytes),
        "asr_model": loaded_model_id or MODEL_ID,
        "device": model_device,
        "signed_language": signed_language,
        "glosser": glosser,
        "audio_duration_seconds": round(duration, 3),
        "asr_inference_seconds": round(asr_time, 3),
        "gloss_gif_seconds": round(gloss_time, 3),
    }


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"[START] ASR model: {MODEL_ID}")
    print(f"[START] Lexicon  : {LEXICON_DIR}")
    print(f"[START] Frontend : {FRONTEND_DIR}")
    uvicorn.run(app, host=host, port=port, reload=False)
