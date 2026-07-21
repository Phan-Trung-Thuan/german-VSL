#!/usr/bin/env python3
"""
pipeline_cli.py -- 4-step German Speech -> Sign Language pipeline.

Each step saves its output to a JSON state file so you can run steps
one at a time in separate Kaggle cells and pick up where you left off.

Steps:
  1. asr       : Audio file  -> German text       (NeMo Canary)
  2. translate : German text -> GVL text          (sign-language simplified German)
  3. gloss     : GVL text    -> Gloss tokens      (spoken_to_signed)
  4. gif       : Gloss tokens-> Animated GIF      (pose_format)

Usage in Kaggle cells:
  # Step 1 only -- audio -> German text
  !python pipeline_cli.py --audio German.mp3 --step asr

  # Step 2 -- reads German text from state, produces GVL text
  !python pipeline_cli.py --step translate

  # Step 3 -- reads GVL text from state, produces gloss tokens
  !python pipeline_cli.py --step gloss

  # Step 4 -- reads gloss tokens from state, produces GIF
  !python pipeline_cli.py --step gif --gif output.gif

  # Override any step input manually (injects into state, skips prior steps):
  !python pipeline_cli.py --text "Ich gehe heute in die Schule." --step translate
  !python pipeline_cli.py --gvl  "GEHEN HEUTE SCHULE"            --step gloss
  !python pipeline_cli.py --gloss "GEHEN SCHULE"                 --step gif --gif output.gif

  # Full pipeline in one shot:
  !python pipeline_cli.py --audio German.mp3 --step gif --gif output.gif

  # Reset state and start over:
  !python pipeline_cli.py --audio German.mp3 --step asr --reset

State file: pipeline_state.json  (auto-created in the script directory)
"""

from __future__ import annotations

import argparse
import json
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

TARGET_SR  = 16_000
MODEL_ID   = os.getenv("ASR_MODEL", "nvidia/canary-1b")
STATE_FILE = PIPELINE_ROOT / "pipeline_state.json"

# ── State helpers ────────────────────────────────────────────────────

def _sep(title: str = "") -> None:
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def load_state(path: Path) -> dict:
    """Read the current pipeline state from disk."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict, path: Path) -> None:
    """Write the pipeline state to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"  [State -> {path}]")


def print_state(state: dict) -> None:
    """Pretty-print the current state."""
    print("  Current state:")
    for k, v in state.items():
        display = str(v)
        if len(display) > 100:
            display = display[:97] + "..."
        print(f"    {k:<20}: {display}")

# ────────────────────────────────────────────────────────────────────
# STEP 1: ASR — Audio -> German text
# ────────────────────────────────────────────────────────────────────

def step1_asr(audio_path: Path, model_id: str = MODEL_ID) -> str:
    """Audio file -> German text using NeMo Canary ASR."""
    import tempfile
    import numpy as np
    import soundfile as sf
    import librosa

    _sep(f"Step 1 / 4 — ASR: {audio_path.name} -> German text")
    print(f"  Input   : {audio_path}")

    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise RuntimeError(f"NeMo/Torch not found: {e}")

    model_file = PIPELINE_ROOT / "models" / f"{model_id.split('/')[-1]}.nemo"
    if model_file.exists():
        print(f"  Model   : local  {model_file}")
        model = nemo_asr.models.ASRModel.restore_from(str(model_file))
    else:
        print(f"  Model   : downloading {model_id} ...")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

    device = "cpu"
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.cuda.get_device_name(0)
    model.eval()
    print(f"  Device  : {device}")

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
        print(f"  Duration: {duration:.1f}s")

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

    elapsed = time.perf_counter() - t0
    print(f"\n  Result  : {text!r}")
    print(f"  Time    : {elapsed:.1f}s")
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
    Removes articles, auxiliary verbs, prepositions; lemmatizes; uppercases.
    """
    import re
    import simplemma

    _sep("Step 2 / 4 — Translate: German text -> GVL text")
    print(f"  Input   : {text!r}")

    raw_tokens = re.findall(r"[A-Za-z\u00c0-\u024f]+", text)
    gvl_tokens = []
    for token in raw_tokens:
        lower = token.lower()
        if lower in _ARTICLES or lower in _AUX_VERBS or lower in _PREPOSITIONS:
            continue
        try:
            lemma = simplemma.lemmatize(lower, lang="de")
        except Exception:
            lemma = lower
        gvl_tokens.append(lemma.upper())

    gvl_text = " ".join(gvl_tokens)
    print(f"\n  Result  : {gvl_text!r}")
    return gvl_text


# ────────────────────────────────────────────────────────────────────
# STEP 3: GLOSS — GVL text -> Gloss tokens
# ────────────────────────────────────────────────────────────────────

def step3_gloss(gvl_text: str, spoken_language: str = "de",
                glosser: str = "simple") -> list[str]:
    """GVL text -> list of gloss tokens using spoken_to_signed."""
    import importlib

    _sep("Step 3 / 4 — Gloss: GVL text -> gloss tokens")
    print(f"  Input   : {gvl_text!r}")

    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    sentences = module.text_to_gloss(text=gvl_text.lower(), language=spoken_language)
    tokens = [gloss for sentence in sentences for _, gloss in sentence]

    print(f"\n  Result  : {tokens}")
    return tokens


# ────────────────────────────────────────────────────────────────────
# STEP 4: GIF — Gloss tokens -> Animated GIF
# ────────────────────────────────────────────────────────────────────

def step4_gif(tokens: list[str], lexicon_dir: Path, gif_path: Path,
              spoken_language: str = "de", signed_language: str = "sgg",
              gif_width: int = 400) -> Path:
    """Gloss tokens -> animated GIF using pose_format + spoken_to_signed."""
    from spoken_to_signed.gloss_to_pose import CSVPoseLookup, gloss_to_pose
    from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup
    from pose_format.pose_visualizer import PoseVisualizer

    _sep("Step 4 / 4 — GIF: gloss tokens -> animated GIF")
    print(f"  Tokens  : {tokens}")
    print(f"  Lexicon : {lexicon_dir}")
    print(f"  Output  : {gif_path}")

    backup = FingerspellingPoseLookup()
    lookup = CSVPoseLookup(str(lexicon_dir), backup=backup)

    sentences = [[(t, t) for t in tokens]]
    poses = [gloss_to_pose(gloss, lookup, spoken_language, signed_language)
             for gloss in sentences]

    if len(poses) == 1:
        pose = poses[0]
    else:
        from spoken_to_signed.gloss_to_pose.concatenate import concatenate_poses
        pose = concatenate_poses(poses, trim=False)

    scale = pose.header.dimensions.width / gif_width
    pose.header.dimensions.width  = int(pose.header.dimensions.width  / scale)
    pose.header.dimensions.height = int(pose.header.dimensions.height / scale)
    pose.body.data = pose.body.data / scale

    visualizer = PoseVisualizer(pose)
    visualizer.save_gif(str(gif_path), visualizer.draw())
    size_kb = gif_path.stat().st_size / 1024
    print(f"\n  Result  : {gif_path}  ({size_kb:.1f} KB)")
    return gif_path


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="German Speech -> Sign Language (4-step stateful pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Step to run ──
    parser.add_argument(
        "--step", default="gif",
        choices=["asr", "translate", "gloss", "gif"],
        help="Which step to execute (default: gif = full pipeline). "
             "Prior step results are read from pipeline_state.json.",
    )

    # ── Manual input overrides ──
    grp = parser.add_argument_group("Input overrides (optional -- normally read from state file)")
    grp.add_argument("--audio", type=Path, help="Audio file for Step 1 (wav/mp3/flac/m4a)")
    grp.add_argument("--text",  type=str,  help="German text  -- injects Step 1 result, skips ASR")
    grp.add_argument("--gvl",   type=str,  help="GVL text     -- injects Step 2 result, skips translate")
    grp.add_argument("--gloss", type=str,  help="Gloss tokens -- injects Step 3 result (space-separated)")

    # ── Output ──
    parser.add_argument("--gif",          type=Path, default=Path("output.gif"))
    parser.add_argument("--gif-width",    type=int,  default=400)

    # ── Model / language options ──
    parser.add_argument("--asr-model",       default=MODEL_ID)
    parser.add_argument("--glosser",         choices=["simple", "spacylemma", "rules"], default="simple")
    parser.add_argument("--spoken-language", default="de")
    parser.add_argument("--signed-language", choices=["sgg", "gsg", "bfi", "ase"], default="sgg",
                        help="sgg=Swiss German SL, gsg=German SL, bfi=British, ase=American")
    parser.add_argument("--lexicon",         type=Path, default=DUMMY_LEXICON)

    # ── State file ──
    parser.add_argument("--state", type=Path, default=STATE_FILE,
                        help=f"Path to JSON state file (default: {STATE_FILE})")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the state file before running")

    args = parser.parse_args()
    state_path = args.state
    t_total = time.perf_counter()

    # ── Load / reset state ──
    state: dict = {}
    if not args.reset and state_path.exists():
        state = load_state(state_path)
        _sep(f"State loaded from {state_path.name}")
        print_state(state)

    # ── Apply manual overrides ──
    if args.text:
        state["german_text"] = args.text
        print(f"  [Override] german_text  = {args.text!r}")
    if args.gvl:
        state["gvl_text"] = args.gvl
        print(f"  [Override] gvl_text     = {args.gvl!r}")
    if args.gloss:
        state["gloss_tokens"] = args.gloss.split()
        print(f"  [Override] gloss_tokens = {state['gloss_tokens']}")

    # ── Run steps in order, stop at --step ──────────────────────────
    STEP_ORDER = ["asr", "translate", "gloss", "gif"]

    for step in STEP_ORDER:

        # ── Step 1: ASR ──────────────────────────────────────────────
        if step == "asr":
            if "german_text" not in state:
                if not args.audio:
                    parser.error(
                        "Step 'asr' needs --audio  "
                        "(or inject German text with --text to skip ASR)."
                    )
                if not args.audio.exists():
                    print(f"ERROR: file not found: {args.audio}", file=sys.stderr)
                    sys.exit(1)
                state["audio_file"]  = str(args.audio)
                state["german_text"] = step1_asr(args.audio, args.asr_model)
                state["asr_model"]   = args.asr_model
                if not state["german_text"].strip():
                    print("ERROR: ASR returned empty transcript.", file=sys.stderr)
                    sys.exit(1)
                save_state(state, state_path)

            _sep("Result — Step 1: German text")
            print(f"  {state['german_text']!r}")
            _sep()

            if args.step == "asr":
                break

        # ── Step 2: Translate ─────────────────────────────────────────
        elif step == "translate":
            if "gvl_text" not in state:
                if "german_text" not in state:
                    parser.error(
                        "Step 'translate' needs German text from state. "
                        "Run --step asr first, or pass --text."
                    )
                state["gvl_text"] = step2_translate(
                    state["german_text"], args.spoken_language
                )
                save_state(state, state_path)

            _sep("Result — Step 2: GVL text")
            print(f"  {state['gvl_text']!r}")
            _sep()

            if args.step == "translate":
                break

        # ── Step 3: Gloss ─────────────────────────────────────────────
        elif step == "gloss":
            if "gloss_tokens" not in state:
                if "gvl_text" not in state:
                    parser.error(
                        "Step 'gloss' needs GVL text from state. "
                        "Run --step translate first, or pass --gvl."
                    )
                state["gloss_tokens"] = step3_gloss(
                    state["gvl_text"], args.spoken_language, args.glosser
                )
                save_state(state, state_path)

            _sep("Result — Step 3: Gloss tokens")
            print(f"  {state['gloss_tokens']}")
            _sep()

            if args.step == "gloss":
                break

        # ── Step 4: GIF ───────────────────────────────────────────────
        elif step == "gif":
            if "gloss_tokens" not in state:
                parser.error(
                    "Step 'gif' needs gloss tokens from state. "
                    "Run --step gloss first, or pass --gloss."
                )
            if not args.lexicon.exists() or not (args.lexicon / "index.csv").exists():
                print(f"ERROR: lexicon not found at: {args.lexicon}", file=sys.stderr)
                print("  Use --lexicon to point to a valid CSVPoseLookup directory.",
                      file=sys.stderr)
                sys.exit(1)

            gif_out = step4_gif(
                state["gloss_tokens"], args.lexicon, args.gif,
                args.spoken_language, args.signed_language, args.gif_width,
            )
            state["gif_path"] = str(gif_out)
            save_state(state, state_path)

            _sep("Result — Step 4: GIF")
            print(f"  {gif_out}")
            _sep()
            break

    elapsed = time.perf_counter() - t_total
    _sep(f"Done in {elapsed:.1f}s  |  state: {state_path.name}")
    print_state(state)
    _sep()


if __name__ == "__main__":
    main()
