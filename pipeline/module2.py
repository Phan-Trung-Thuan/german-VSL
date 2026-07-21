# pipeline/module2.py
"""
Module 2 — Audio -> German Text  (ASR)
=======================================
Transcribes a 16-kHz mono WAV to German text using NVIDIA Canary-1B via NeMo.

The loaded model is cached in-process so repeated calls in the same Kaggle
session do not reload the 4 GB checkpoint every time.

Usage
-----
  from pipeline.module2 import module2

  o2 = module2(o1)              # o1 is an AudioResult from module1
  o2 = module2("/path/to.wav")  # or pass a raw WAV path
  print(o2.german_text)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .types import AudioResult, ASRResult

DEFAULT_MODEL = "nvidia/canary-1b"

# Module-level cache: model_id -> loaded NeMo model object
_MODEL_CACHE: dict[str, object] = {}


def module2(
    audio:     AudioResult | str | Path,
    model_id:  str  = DEFAULT_MODEL,
    use_cache: bool = True,
) -> ASRResult:
    """
    Transcribe audio to German text with NVIDIA Canary-1B (NeMo).

    Parameters
    ----------
    audio     : AudioResult from module1, OR a direct path to a WAV file
    model_id  : HuggingFace / NeMo model ID  (default: nvidia/canary-1b)
    use_cache : if True, the loaded model stays in memory between calls

    Returns
    -------
    ASRResult  with .german_text, .model_id, .elapsed_s
    """
    # Accept raw path or AudioResult
    if isinstance(audio, (str, Path)):
        wav_path   = Path(audio)
        input_desc = str(wav_path)
    else:
        wav_path   = audio.audio_path
        input_desc = f"{wav_path.name}  ({audio.duration_s:.1f}s)"

    if not wav_path.exists():
        raise FileNotFoundError(f"[module2] Audio file not found: {wav_path}")

    _print_header(f"Module 2 — Audio → German Text  (ASR)")
    print(f"  Input  : {input_desc}")

    model = _load_model(model_id, use_cache)

    # ── Transcribe ───────────────────────────────────────────────────
    t0   = time.perf_counter()
    text = _transcribe(model, wav_path)
    elapsed = time.perf_counter() - t0

    print(f"\n  Result : {text!r}")
    print(f"  Time   : {elapsed:.1f}s")

    return ASRResult(german_text=text, model_id=model_id, elapsed_s=elapsed)


# ── Private helpers ──────────────────────────────────────────────────────────

def _load_model(model_id: str, use_cache: bool) -> object:
    global _MODEL_CACHE

    if use_cache and model_id in _MODEL_CACHE:
        print(f"  Model  : {model_id}  [cached ✓]")
        return _MODEL_CACHE[model_id]

    print(f"  Model  : {model_id}  [loading …]")
    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise ImportError(
            f"[module2] NeMo not available: {e}\n"
            "  Install: pip install 'nemo_toolkit[asr] @ "
            "git+https://github.com/NVIDIA/NeMo.git'"
        )

    # Prefer a local .nemo file if it exists
    local = Path(__file__).resolve().parent.parent / "models" / f"{model_id.split('/')[-1]}.nemo"
    if local.exists():
        print(f"  Source : local  {local}")
        model = nemo_asr.models.ASRModel.restore_from(str(local))
    else:
        print(f"  Source : HuggingFace download …")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.cuda.get_device_name(0)
    else:
        device = "CPU"
    model.eval()
    print(f"  Device : {device}")

    if use_cache:
        _MODEL_CACHE[model_id] = model
    return model


def _transcribe(model: object, wav_path: Path) -> str:
    """Try Canary-specific kwargs first; fall back to generic transcribe."""
    for kwargs in [
        dict(source_lang="de", target_lang="de", pnc="yes", batch_size=1),
        dict(source_lang="de", target_lang="de", batch_size=1),
        dict(batch_size=1),
    ]:
        try:
            out  = model.transcribe([str(wav_path)], **kwargs)
            item = out[0] if out else ""
            return str(item.text) if hasattr(item, "text") else str(item)
        except TypeError:
            continue
    return ""


def _print_header(title: str) -> None:
    bar = "─" * 60
    print(f"\n┌{bar}┐\n│  {title:<58}│\n└{bar}┘")
