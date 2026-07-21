# pipeline/module4.py
"""
Module 4 — Gloss Tokens -> Video Clips  (Phoenix14T lexicon lookup)
====================================================================
For each predicted gloss token, find the best matching training video
from the Phoenix14T dataset (via PhoenixLexicon) and extract a short clip.

Note on Phoenix14T videos
--------------------------
Phoenix14T provides *sentence-level* videos, not isolated per-sign clips.
Each returned clip is a short trim of the first sentence video that contains
the target gloss token.  Tokens absent from the training set are flagged as
missing (TokenClip.found = False).

Usage
-----
  from pipeline.module4 import module4
  from pipeline.lexicon  import PhoenixLexicon

  lex = PhoenixLexicon(csv_path, videos_dir)
  o4  = module4(o3, lex)            # o3 is a GlossResult
  o4  = module4(["JETZT", "WETTER"], lex)   # or a plain token list

  for clip in o4.clips:
      print(clip.token, clip.found, clip.clip_path)
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

from .types   import GlossResult, LookupResult, TokenClip
from .lexicon import PhoenixLexicon


def module4(
    gloss:           Union[GlossResult, list[str]],
    lexicon:         PhoenixLexicon,
    clip_duration_s: float = 2.0,
    _out_dir:        Optional[str | Path] = None,
) -> LookupResult:
    """
    Look up each gloss token in the Phoenix14T lexicon and extract a clip.

    Parameters
    ----------
    gloss           : GlossResult from module3, OR a list of uppercase token strings
    lexicon         : PhoenixLexicon instance (build once and reuse)
    clip_duration_s : how many seconds to trim from the matched video
    _out_dir        : directory for clip files  (default: system temp)

    Returns
    -------
    LookupResult  with .clips (list[TokenClip]), .found_count, .missing_tokens
    """
    tokens = gloss.gloss_tokens if isinstance(gloss, GlossResult) else list(gloss)

    _print_header("Module 4 — Gloss Tokens → Video Clips")
    print(f"  Tokens   : {tokens}")

    out_dir = Path(_out_dir) if _out_dir else Path(tempfile.mkdtemp(prefix="m4_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    clips:   list[TokenClip] = []
    missing: list[str]       = []

    for tok in tokens:
        tc = _resolve_token(tok, lexicon, out_dir, clip_duration_s)
        clips.append(tc)
        if tc.found:
            context = _format_context(tc)
            print(f"  [{tok:<20}] ✓  video: {tc.video_path.name}")
            print(f"               gloss context: {context}")
        else:
            print(f"  [{tok:<20}] ✗  not found in lexicon")
            missing.append(tok)

    found_n = sum(1 for c in clips if c.found)
    print(f"\n  Found  : {found_n} / {len(tokens)}")
    if missing:
        print(f"  Missing: {missing}")

    return LookupResult(clips=clips, found_count=found_n, missing_tokens=missing)


# ── Private helpers ──────────────────────────────────────────────────────────

def _resolve_token(
    tok:             str,
    lexicon:         PhoenixLexicon,
    out_dir:         Path,
    clip_duration_s: float,
) -> TokenClip:
    entries = lexicon.lookup(tok)
    if not entries:
        return TokenClip(token=tok, video_path=None, clip_path=None, found=False)

    # Pick the first entry that has a video file on disk
    entry = next((e for e in entries if e["video"] is not None), None)
    if entry is None:
        return TokenClip(token=tok, video_path=None, clip_path=None, found=False)

    video_path = entry["video"]
    stem       = Path(entry["name"]).name
    clip_path  = out_dir / f"{tok}_{stem}.mp4"

    ok = _extract_clip(video_path, clip_path, duration_s=clip_duration_s)
    if not ok:
        clip_path = None

    # Store the lexicon entry on the clip for later use
    tc            = TokenClip(token=tok, video_path=video_path,
                              clip_path=clip_path, found=True)
    tc._entry     = entry   # attach raw entry (gloss_seq, token_idx)
    return tc


def _format_context(tc: TokenClip) -> str:
    """Format the sentence gloss with the target token highlighted."""
    entry = getattr(tc, "_entry", None)
    if entry is None:
        return ""
    gloss_seq = entry["gloss_seq"]
    tok_idx   = entry["token_idx"]
    return " ".join(
        f"[{t}]" if i == tok_idx else t
        for i, t in enumerate(gloss_seq)
    )


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_clip(
    video_path: Path,
    out_path:   Path,
    start_s:    float = 0.0,
    duration_s: float = 2.0,
) -> bool:
    """Extract a clip with ffmpeg; copy full video if ffmpeg unavailable."""
    if not _ffmpeg_available():
        try:
            shutil.copy2(str(video_path), str(out_path))
            return True
        except Exception:
            return False
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-i",  str(video_path),
            "-t",  str(duration_s),
            "-c:v", "libx264", "-crf", "23",
            "-an",                      # no audio needed
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


def _print_header(title: str) -> None:
    bar = "─" * 60
    print(f"\n┌{bar}┐\n│  {title:<58}│\n└{bar}┘")
