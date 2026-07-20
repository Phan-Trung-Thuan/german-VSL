#!/usr/bin/env python3
"""
kaggle_run.py — CLI entry point optimized for Kaggle GPU environment.

Quick start:
  python kaggle_run.py --audio /kaggle/input/your-dataset/speech.wav --gif output.gif
  python kaggle_run.py --text "Guten Morgen" --gif output.gif
  python kaggle_run.py --audio speech.wav --step asr

Environment variables:
  ASR_MODEL   — NeMo model ID (default: nvidia/canary-1b)
  LEXICON_DIR — path to CSVPoseLookup lexicon (default: auto-detect dummy_lexicon)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent

# On Kaggle: /kaggle/working/german-VSL
# Lexicon bundled in repo under assets/dummy_lexicon (if present)
DUMMY_LEXICON = REPO_ROOT / "assets" / "dummy_lexicon"
LEXICON_DIR = Path(os.getenv("LEXICON_DIR", str(DUMMY_LEXICON)))
MODEL_ID = os.getenv("ASR_MODEL", "nvidia/canary-1b")
TARGET_SR = 16_000


# ── ASR ──────────────────────────────────────────────────────────────

def load_asr_model():
    import torch
    import nemo.collections.asr as nemo_asr

    model_path = REPO_ROOT / "models" / f"{MODEL_ID.split('/')[-1]}.nemo"
    if model_path.exists():
        print(f"[ASR] Loading local model: {model_path}")
        model = nemo_asr.models.ASRModel.restore_from(str(model_path))
    else:
        print(f"[ASR] Downloading from HuggingFace: {MODEL_ID}")
        print("      (this may take a few minutes on first run)")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.cuda.get_device_name(0)
    else:
        device = "cpu"

    model.eval()
    print(f"[ASR] Ready on {device}")
    return model


def transcribe(model, wav_path: str) -> str:
    for kwargs in [
        dict(source_lang="de", target_lang="de", pnc="yes", batch_size=1),
        dict(source_lang="de", target_lang="de", batch_size=1),
        dict(batch_size=1),
    ]:
        try:
            out = model.transcribe([wav_path], **kwargs)
            item = out[0] if out else ""
            return str(item.text) if hasattr(item, "text") else str(item)
        except TypeError:
            continue
    out = model.transcribe([wav_path])
    item = out[0] if out else ""
    return str(item.text) if hasattr(item, "text") else str(item)


def convert_to_16k_mono(input_path: Path, output_path: Path) -> float:
    import numpy as np
    import soundfile as sf
    import librosa

    try:
        audio, sr = sf.read(str(input_path), always_2d=False)
    except Exception:
        audio, sr = librosa.load(str(input_path), sr=None, mono=False)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        axis = 0 if audio.shape[0] <= 8 else 1
        audio = audio.mean(axis=axis)

    audio = np.nan_to_num(audio)
    duration = float(len(audio) / sr) if sr else 0.0

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    sf.write(str(output_path), audio, TARGET_SR, subtype="PCM_16")
    return duration


# ── Gloss + GIF ───────────────────────────────────────────────────────

def text_to_gloss(text: str, spoken_language: str = "de", glosser: str = "simple") -> list:
    import importlib
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    return module.text_to_gloss(text=text, language=spoken_language)


def gloss_to_gif(text: str, lexicon_dir: Path, spoken_language: str = "de",
                  signed_language: str = "sgg", glosser: str = "simple",
                  gif_width: int = 400) -> tuple[str, list, Path]:
    import tempfile
    from spoken_to_signed.gloss_to_pose import CSVPoseLookup, gloss_to_pose
    from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup
    from pose_format.pose_visualizer import PoseVisualizer

    sentences = text_to_gloss(text, spoken_language, glosser)
    gloss_tokens = [g for sentence in sentences for _, g in sentence]
    gloss_str = " ".join(gloss_tokens)

    backup = FingerspellingPoseLookup()
    lookup = CSVPoseLookup(str(lexicon_dir), backup=backup)
    poses = [gloss_to_pose(gloss, lookup, spoken_language, signed_language) for gloss in sentences]

    if len(poses) == 1:
        pose = poses[0]
    else:
        from spoken_to_signed.gloss_to_pose.concatenate import concatenate_poses
        pose = concatenate_poses(poses, trim=False)

    scale = pose.header.dimensions.width / gif_width
    pose.header.dimensions.width = int(pose.header.dimensions.width / scale)
    pose.header.dimensions.height = int(pose.header.dimensions.height / scale)
    pose.body.data = pose.body.data / scale

    tmp_gif = Path(tempfile.mktemp(suffix=".gif"))
    visualizer = PoseVisualizer(pose)
    visualizer.save_gif(str(tmp_gif), visualizer.draw())

    return gloss_str, sentences, tmp_gif


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="German Speech -> Sign Language pipeline (Kaggle GPU edition)"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", type=Path, help="Input audio file (wav/mp3/flac/m4a)")
    src.add_argument("--text",  type=str,  help="Input German text (skips ASR)")

    parser.add_argument("--gif",           type=Path, default=None,  help="Output GIF path")
    parser.add_argument("--step",          choices=["asr", "gloss", "full"], default="full",
                        help="Which step(s) to run (default: full)")
    parser.add_argument("--glosser",       default="simple",          help="Glosser to use")
    parser.add_argument("--signed-lang",   default="sgg",             help="Signed language code")
    parser.add_argument("--spoken-lang",   default="de",              help="Spoken language code")
    parser.add_argument("--gif-width",     type=int, default=400,     help="Output GIF width px")
    parser.add_argument("--model",         default=MODEL_ID,          help="NeMo ASR model ID")
    parser.add_argument("--lexicon",       type=Path, default=LEXICON_DIR, help="Lexicon directory")
    args = parser.parse_args()

    t_total = time.perf_counter()

    # Step 1: ASR
    text = args.text
    if args.audio:
        if args.step in ("asr", "full"):
            import tempfile
            model = load_asr_model()
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                wav_path = tmp_dir / "input_16k.wav"
                print(f"[ASR] Converting audio to 16kHz mono...")
                duration = convert_to_16k_mono(args.audio, wav_path)
                print(f"[ASR] Audio duration: {duration:.1f}s")
                t0 = time.perf_counter()
                text = transcribe(model, str(wav_path))
                print(f"[ASR] Transcript: {text!r}  ({time.perf_counter()-t0:.1f}s)")
        if args.step == "asr":
            print("\n[DONE] ASR only mode — exiting.")
            return

    if not text:
        print("[ERROR] No text produced. Check audio file or provide --text.")
        sys.exit(1)

    # Step 2+3: Gloss + GIF
    if args.step in ("gloss", "full"):
        lexicon = args.lexicon
        if not lexicon.exists() or not (lexicon / "index.csv").exists():
            print(f"[ERROR] Lexicon not found at: {lexicon}")
            print("        Set --lexicon or LEXICON_DIR env var to a valid directory.")
            sys.exit(1)

        print(f"\n[Gloss] Running glosser='{args.glosser}' on: {text!r}")
        t0 = time.perf_counter()
        gloss_str, sentences, gif_path = gloss_to_gif(
            text, lexicon, args.spoken_lang, args.signed_lang,
            args.glosser, args.gif_width
        )
        print(f"[Gloss] Result: {gloss_str!r}  ({time.perf_counter()-t0:.1f}s)")

        out_gif = args.gif or Path("output.gif")
        gif_path.rename(out_gif)
        size_kb = out_gif.stat().st_size / 1024
        print(f"[GIF]   Saved to: {out_gif}  ({size_kb:.1f} KB)")

    print(f"\n[DONE] Total time: {time.perf_counter()-t_total:.1f}s")


if __name__ == "__main__":
    main()
