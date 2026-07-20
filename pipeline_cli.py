#!/usr/bin/env python3
"""
pipeline_cli.py -- 4-step German Speech -> Sign Language pipeline.

Steps:
  1. ASR      : Audio file -> German text       (NeMo Canary)
  2. Translate : German text -> GVL text         (sign-language-simplified German)
  3. Gloss    : GVL text -> Gloss tokens        (spoken_to_signed)
  4. GIF      : Gloss tokens -> Animated GIF    (pose_format)

Usage examples:

  # Full pipeline (all 4 steps)
  python pipeline_cli.py --audio speech.wav --gif output.gif

  # Step 1 only: speech -> text
  python pipeline_cli.py --audio speech.wav --step asr

  # Steps 1-2: speech -> GVL text
  python pipeline_cli.py --audio speech.wav --step translate

  # Steps 1-3: speech -> gloss tokens
  python pipeline_cli.py --audio speech.wav --step gloss

  # Start from text (skip step 1)
  python pipeline_cli.py --text "Ich gehe heute in die Schule." --gif output.gif

  # Start from GVL text (skip steps 1-2)
  python pipeline_cli.py --gvl "GEHEN HEUTE SCHULE" --gif output.gif

  # Start from gloss tokens (skip steps 1-3)
  python pipeline_cli.py --gloss "GEHEN SCHULE" --gif output.gif
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
PIPELINE_ROOT = Path(__file__).resolve().parent
AA_ROOT       = PIPELINE_ROOT.parent
GLOSS_REPO    = AA_ROOT / "gloss_to_gif" / "gloss_to_gif"
DUMMY_LEXICON = GLOSS_REPO / "assets" / "dummy_lexicon"

if str(GLOSS_REPO) not in sys.path:
    sys.path.insert(0, str(GLOSS_REPO))

TARGET_SR = 16_000
MODEL_ID  = os.getenv("ASR_MODEL", "nvidia/canary-1b")

# ────────────────────────────────────────────────────────────────────
# STEP 1: ASR — Audio -> German text
# ────────────────────────────────────────────────────────────────────

def step1_asr(audio_path: Path, model_id: str = MODEL_ID) -> str:
    """Audio file -> German text using NeMo Canary ASR."""
    import tempfile
    import numpy as np
    import soundfile as sf
    import librosa

    print(f"\n[Step 1/4] ASR: {audio_path.name} -> German text")

    # Load model
    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise RuntimeError(f"NeMo/Torch not found: {e}")

    model_file = PIPELINE_ROOT / "models" / f"{model_id.split('/')[-1]}.nemo"
    if model_file.exists():
        print(f"  Loading local model: {model_file}")
        model = nemo_asr.models.ASRModel.restore_from(str(model_file))
    else:
        print(f"  Downloading model: {model_id}")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

    device = "cpu"
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.cuda.get_device_name(0)
    model.eval()
    print(f"  Device: {device}")

    # Convert audio to 16kHz mono WAV
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input_16k.wav"
        try:
            audio, sr = sf.read(str(audio_path), always_2d=False)
        except Exception:
            audio, sr = librosa.load(str(audio_path), sr=None, mono=False)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=0 if audio.shape[0] <= 8 else 1)
        audio = np.nan_to_num(audio)
        duration = len(audio) / sr if sr else 0.0

        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sf.write(str(wav_path), audio, TARGET_SR, subtype="PCM_16")
        print(f"  Audio duration: {duration:.1f}s")

        # Transcribe
        t0 = time.perf_counter()
        text = ""
        for kwargs in [
            dict(source_lang="de", target_lang="de", pnc="yes", batch_size=1),
            dict(source_lang="de", target_lang="de", batch_size=1),
            dict(batch_size=1),
        ]:
            try:
                out = model.transcribe([str(wav_path)], **kwargs)
                item = out[0] if out else ""
                text = str(item.text) if hasattr(item, "text") else str(item)
                break
            except TypeError:
                continue

    print(f"  Result : {text!r}  ({time.perf_counter()-t0:.1f}s)")
    return text


# ────────────────────────────────────────────────────────────────────
# STEP 2: TRANSLATE — German text -> GVL text
# ────────────────────────────────────────────────────────────────────

# German articles and auxiliary verbs that are dropped in DGS
_ARTICLES = {
    "der", "die", "das", "des", "dem", "den",
    "ein", "eine", "einer", "einem", "einen", "eines",
}
_AUX_VERBS = {
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "hat", "haben", "hatte", "hatten", "habe", "hast", "habt",
    "bin", "bist", "sei", "seien",
    "kann", "koennen", "muss", "muessen", "soll", "sollen",
    "darf", "duerfen", "mag", "moegen", "will", "wollen",
}
_PREPOSITIONS = {
    "in", "im", "an", "am", "auf", "zu", "zum", "zur", "von", "vom",
    "bei", "beim", "mit", "aus", "nach", "ueber", "unter", "vor",
    "hinter", "neben", "zwischen", "durch", "fuer", "gegen", "ohne",
}


def step2_translate(text: str, language: str = "de") -> str:
    """
    German text -> GVL (German Visual Language) text.

    Applies sign-language grammar rules:
      - Removes articles (der/die/das/ein...)
      - Removes auxiliary verbs (ist/sind/hat...)
      - Removes most prepositions (spatial ones kept as glosses)
      - Lemmatizes content words using simplemma
      - Uppercases all tokens (gloss convention)
      - Preserves word order (simplified DGS approximation)
    """
    import re
    import simplemma

    print(f"\n[Step 2/4] Translate: German text -> GVL text")
    print(f"  Input  : {text!r}")

    # Tokenize (split on whitespace, strip punctuation)
    raw_tokens = re.findall(r"[A-Za-z\u00c0-\u024f]+", text)

    gvl_tokens = []
    for token in raw_tokens:
        lower = token.lower()

        # Drop articles, auxiliaries, most prepositions
        if lower in _ARTICLES or lower in _AUX_VERBS or lower in _PREPOSITIONS:
            continue

        # Lemmatize content word
        try:
            lemma = simplemma.lemmatize(lower, lang="de")
        except Exception:
            lemma = lower

        gvl_tokens.append(lemma.upper())

    gvl_text = " ".join(gvl_tokens)
    print(f"  Output : {gvl_text!r}")
    return gvl_text


# ────────────────────────────────────────────────────────────────────
# STEP 3: GLOSS — GVL text -> Gloss tokens
# ────────────────────────────────────────────────────────────────────

def step3_gloss(gvl_text: str, spoken_language: str = "de",
                glosser: str = "simple") -> list[str]:
    """
    GVL text -> list of gloss tokens using spoken_to_signed.
    Returns a flat list of gloss strings, e.g. ['GEHEN', 'SCHULE', ...].
    """
    import importlib

    print(f"\n[Step 3/4] Gloss: GVL text -> gloss tokens")
    print(f"  Input   : {gvl_text!r}")

    # spoken_to_signed expects lowercase natural language;
    # pass GVL text as-is (simple glosser treats each word as a gloss)
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    sentences = module.text_to_gloss(text=gvl_text.lower(), language=spoken_language)

    tokens = [gloss for sentence in sentences for _, gloss in sentence]
    print(f"  Tokens  : {tokens}")
    return tokens


# ────────────────────────────────────────────────────────────────────
# STEP 4: GIF — Gloss tokens -> Animated GIF
# ────────────────────────────────────────────────────────────────────

def step4_gif(tokens: list[str], lexicon_dir: Path, gif_path: Path,
              spoken_language: str = "de", signed_language: str = "sgg",
              gif_width: int = 400) -> Path:
    """
    Gloss tokens -> animated GIF using pose_format + spoken_to_signed.
    """
    import tempfile
    from spoken_to_signed.gloss_to_pose import CSVPoseLookup, gloss_to_pose
    from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup
    from pose_format.pose_visualizer import PoseVisualizer

    print(f"\n[Step 4/4] GIF: gloss tokens -> animated GIF")
    print(f"  Tokens  : {tokens}")
    print(f"  Lexicon : {lexicon_dir}")

    backup = FingerspellingPoseLookup()
    lookup = CSVPoseLookup(str(lexicon_dir), backup=backup)

    # Build per-sentence structure expected by gloss_to_pose
    # Each sentence is a list of (word, gloss) pairs
    sentences = [[(t, t) for t in tokens]]
    poses = [gloss_to_pose(gloss, lookup, spoken_language, signed_language)
             for gloss in sentences]

    if len(poses) == 1:
        pose = poses[0]
    else:
        from spoken_to_signed.gloss_to_pose.concatenate import concatenate_poses
        pose = concatenate_poses(poses, trim=False)

    # Scale to gif_width
    scale = pose.header.dimensions.width / gif_width
    pose.header.dimensions.width  = int(pose.header.dimensions.width  / scale)
    pose.header.dimensions.height = int(pose.header.dimensions.height / scale)
    pose.body.data = pose.body.data / scale

    visualizer = PoseVisualizer(pose)
    visualizer.save_gif(str(gif_path), visualizer.draw())
    size_kb = gif_path.stat().st_size / 1024
    print(f"  Saved   : {gif_path}  ({size_kb:.1f} KB)")
    return gif_path


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="German Speech -> Sign Language (4-step pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input source (only one required) ──
    src = parser.add_argument_group("Input (choose one)")
    src.add_argument("--audio", type=Path, help="Step 1 input: audio file (wav/mp3/flac/m4a)")
    src.add_argument("--text",  type=str,  help="Step 2 input: German text (skip ASR)")
    src.add_argument("--gvl",   type=str,  help="Step 3 input: GVL text (skip steps 1-2)")
    src.add_argument("--gloss", type=str,  help="Step 4 input: space-separated gloss tokens (skip steps 1-3)")

    # ── Stop point ──
    parser.add_argument(
        "--step", default="gif",
        choices=["asr", "translate", "gloss", "gif"],
        help="Run up to this step (default: gif = full pipeline)"
    )

    # ── Output ──
    parser.add_argument("--gif",          type=Path, default=Path("output.gif"), help="Output GIF path")
    parser.add_argument("--gif-width",    type=int,  default=400)

    # ── Model / language options ──
    parser.add_argument("--asr-model",       default=MODEL_ID)
    parser.add_argument("--glosser",         choices=["simple", "spacylemma", "rules"], default="simple")
    parser.add_argument("--spoken-language", default="de")
    parser.add_argument("--signed-language", choices=["sgg", "gsg", "bfi", "ase"], default="sgg",
                        help="sgg=Swiss German SL, gsg=German SL, bfi=British, ase=American")
    parser.add_argument("--lexicon",         type=Path, default=DUMMY_LEXICON,
                        help=f"Lexicon dir for GIF step (default: {DUMMY_LEXICON})")

    args = parser.parse_args()

    # ── Validate inputs ──
    inputs = [args.audio, args.text, args.gvl, args.gloss]
    if not any(inputs):
        parser.error("Provide one of: --audio, --text, --gvl, --gloss")

    t_total = time.perf_counter()

    # ── State variables ──
    german_text = args.text
    gvl_text    = args.gvl
    gloss_tokens = [t.strip() for t in args.gloss.split()] if args.gloss else None

    # ── Step 1: ASR ──
    if args.audio:
        if not args.audio.exists():
            print(f"ERROR: file not found: {args.audio}", file=sys.stderr)
            sys.exit(1)
        german_text = step1_asr(args.audio, args.asr_model)
        if not german_text.strip():
            print("ERROR: ASR returned empty transcript.", file=sys.stderr)
            sys.exit(1)
    if args.step == "asr":
        print(f"\n{'='*55}")
        print(f"[Step 1 result] German text: {german_text}")
        print(f"{'='*55}")
        return

    # ── Step 2: Translate ──
    if gvl_text is None:
        if german_text is None:
            print("ERROR: --text or --audio required for this step.", file=sys.stderr)
            sys.exit(1)
        gvl_text = step2_translate(german_text, args.spoken_language)
    if args.step == "translate":
        print(f"\n{'='*55}")
        print(f"[Step 2 result] GVL text: {gvl_text}")
        print(f"{'='*55}")
        return

    # ── Step 3: Gloss ──
    if gloss_tokens is None:
        if gvl_text is None:
            print("ERROR: --gvl, --text, or --audio required for this step.", file=sys.stderr)
            sys.exit(1)
        gloss_tokens = step3_gloss(gvl_text, args.spoken_language, args.glosser)
    if args.step == "gloss":
        print(f"\n{'='*55}")
        print(f"[Step 3 result] Gloss tokens: {gloss_tokens}")
        print(f"{'='*55}")
        return

    # ── Step 4: GIF ──
    if not gloss_tokens:
        print("ERROR: no gloss tokens to render.", file=sys.stderr)
        sys.exit(1)
    if not args.lexicon.exists() or not (args.lexicon / "index.csv").exists():
        print(f"ERROR: lexicon not found at: {args.lexicon}", file=sys.stderr)
        print("  Use --lexicon to point to a valid CSVPoseLookup directory.", file=sys.stderr)
        sys.exit(1)

    step4_gif(gloss_tokens, args.lexicon, args.gif,
              args.spoken_language, args.signed_language, args.gif_width)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*55}")
    print(f"[OK] Pipeline complete in {elapsed:.1f}s")
    print(f"  German text  : {german_text or '(provided)'}")
    print(f"  GVL text     : {gvl_text or '(provided)'}")
    print(f"  Gloss tokens : {gloss_tokens}")
    print(f"  GIF saved    : {args.gif}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
