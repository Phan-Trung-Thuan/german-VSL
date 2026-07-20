#!/usr/bin/env python3
"""
pipeline_cli.py — Command-line runner for the full pipeline.

Usage examples:
  # Full pipeline: audio file -> GIF
  python pipeline_cli.py --audio speech.wav --gif output.gif

  # Just transcribe (step 1 only)
  python pipeline_cli.py --audio speech.wav --step asr

  # From text (skip ASR, go text -> gloss -> GIF)
  python pipeline_cli.py --text "Guten Morgen!" --gif output.gif

  # Use a different glosser
  python pipeline_cli.py --audio speech.wav --gif output.gif --glosser spacylemma

  # DGS (German Sign Language) instead of Swiss German SL
  python pipeline_cli.py --audio speech.wav --gif output.gif --signed-language gsg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────
PIPELINE_ROOT = Path(__file__).resolve().parent
AA_ROOT = PIPELINE_ROOT.parent

CANARY_ROOT = AA_ROOT / "canary_web_asr" / "canary_web_asr"
GLOSS_REPO  = AA_ROOT / "gloss_to_gif" / "gloss_to_gif"
DUMMY_LEXICON = GLOSS_REPO / "assets" / "dummy_lexicon"

if str(GLOSS_REPO) not in sys.path:
    sys.path.insert(0, str(GLOSS_REPO))

# ── ASR (Step 1) ────────────────────────────────────────────────────

def step_asr(audio_path: Path, model_id: str = "nvidia/canary-1b") -> str:
    """German audio → German text using Canary ASR."""
    import os
    import tempfile
    import numpy as np
    import soundfile as sf
    import librosa

    print(f"\n[Step 1] ASR: {audio_path} → German text")
    print(f"         Model: {model_id}")

    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise RuntimeError(
            "NeMo not installed. Run: pip install Cython packaging && pip install -U 'nemo_toolkit[asr]'"
        ) from e

    # Load model
    local_path = CANARY_ROOT / "models" / f"{model_id.split('/')[-1]}.nemo"
    if local_path.exists():
        print(f"         Loading from local: {local_path}")
        model = nemo_asr.models.ASRModel.restore_from(str(local_path))
    else:
        print(f"         Downloading from HuggingFace (first time, ~4GB)…")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

    device = "cpu"
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.cuda.get_device_name(0)
    model.eval()
    print(f"         Device: {device}")

    # Convert audio
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input_16k.wav"
        try:
            audio, sr = sf.read(str(audio_path), always_2d=False)
        except Exception:
            audio, sr = librosa.load(str(audio_path), sr=None, mono=False)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            axis = 0 if audio.shape[0] <= 8 else 1
            audio = np.mean(audio, axis=axis)
        audio = np.nan_to_num(audio)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(str(wav_path), audio, 16000, subtype="PCM_16")

        t0 = time.perf_counter()
        for kwargs in [
            dict(source_lang="de", target_lang="de", pnc="yes", batch_size=1),
            dict(source_lang="de", target_lang="de", batch_size=1),
            dict(batch_size=1),
        ]:
            try:
                out = model.transcribe([str(wav_path)], **kwargs)
                break
            except TypeError:
                continue
        else:
            out = model.transcribe([str(wav_path)])

        def _extract(item):
            if hasattr(item, "text"): return str(item.text)
            if isinstance(item, dict) and "text" in item: return str(item["text"])
            return str(item)

        text = _extract(out[0]) if out else ""
        elapsed = time.perf_counter() - t0

    print(f"         Result: \"{text}\"")
    print(f"         Inference time: {elapsed:.2f}s")
    return text


# ── Gloss + GIF (Steps 2 & 3) ───────────────────────────────────────

def step_gloss_gif(
    text: str,
    lexicon_dir: Path,
    spoken_language: str = "de",
    signed_language: str = "sgg",
    glosser: str = "simple",
    gif_path: Path | None = None,
    gif_width: int = 400,
) -> tuple[list[str], Path]:
    """German text → gloss tokens + animated GIF."""
    print(f"\n[Step 2] Gloss: \"{text}\"")
    print(f"         Language: {spoken_language} → {signed_language}")
    print(f"         Glosser: {glosser}")
    print(f"         Lexicon: {lexicon_dir}")

    from spoken_to_signed.bin import _text_to_gloss, _gloss_to_pose
    from pose_format.pose_visualizer import PoseVisualizer

    sentences = _text_to_gloss(text, spoken_language, glosser)
    tokens = [gloss for sentence in sentences for _, gloss in sentence]
    print(f"         Gloss tokens: {tokens}")

    print(f"\n[Step 3] Pose → GIF")
    result = _gloss_to_pose(sentences, str(lexicon_dir), spoken_language, signed_language)

    pose = result.pose
    scale = pose.header.dimensions.width / gif_width
    pose.header.dimensions.width  = int(pose.header.dimensions.width / scale)
    pose.header.dimensions.height = int(pose.header.dimensions.height / scale)
    pose.body.data = pose.body.data / scale

    if gif_path is None:
        gif_path = Path("output.gif")

    visualizer = PoseVisualizer(pose)
    visualizer.save_gif(str(gif_path), visualizer.draw())
    print(f"         Saved GIF: {gif_path} ({gif_path.stat().st_size // 1024} KB)")
    return tokens, gif_path


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="German Speech -> Sign Language Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=Path, help="Input audio file (wav/mp3/flac/m4a)")
    input_group.add_argument("--text",  type=str,  help="German text (skip ASR)")

    parser.add_argument("--gif",            type=Path, default=Path("output.gif"), help="Output GIF path (default: output.gif)")
    parser.add_argument("--gif-width",      type=int,  default=400)
    parser.add_argument("--step",           choices=["asr", "gloss", "all"], default="all",
                        help="Which steps to run (default: all)")
    parser.add_argument("--asr-model",      default="nvidia/canary-1b",
                        help="ASR model id (default: nvidia/canary-1b)")
    parser.add_argument("--glosser",        choices=["simple", "spacylemma", "rules"],
                        default="simple")
    parser.add_argument("--spoken-language", default="de")
    parser.add_argument("--signed-language",
                        choices=["sgg", "gsg", "bfi", "ase"], default="sgg",
                        help="Target sign language (default: sgg = Swiss German SL)")
    parser.add_argument("--lexicon",        type=Path, default=DUMMY_LEXICON,
                        help=f"Lexicon directory (default: {DUMMY_LEXICON})")

    args = parser.parse_args()

    t_start = time.perf_counter()
    text = args.text

    # Step 1: ASR
    if args.audio:
        if not args.audio.exists():
            print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
            sys.exit(1)
        text = step_asr(args.audio, args.asr_model)
        if not text.strip():
            print("ERROR: ASR returned empty transcript.", file=sys.stderr)
            sys.exit(1)
        if args.step == "asr":
            print(f"\n[OK] Transcript: {text}")
            return

    # Steps 2+3: Gloss + GIF
    if not args.lexicon.exists() or not (args.lexicon / "index.csv").exists():
        print(f"ERROR: lexicon not found at: {args.lexicon}", file=sys.stderr)
        print("  Use --lexicon to point to a valid CSVPoseLookup directory.", file=sys.stderr)
        sys.exit(1)

    tokens, gif_path = step_gloss_gif(
        text,
        args.lexicon,
        args.spoken_language,
        args.signed_language,
        args.glosser,
        args.gif,
        args.gif_width,
    )

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"[OK] Pipeline complete in {elapsed:.1f}s")
    print(f"  Transcript : {text}")
    print(f"  Gloss      : {' '.join(tokens)}")
    print(f"  GIF saved  : {gif_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
