# pipeline/module1.py
"""
Module 1 — Video -> Audio
=========================
Extract audio from a video file and convert it to a 16-kHz mono WAV.

Usage
-----
  from pipeline.module1 import module1

  o1 = module1("/path/to/video.mp4")
  print(o1)          # AudioResult(duration=33.1s, audio='...wav', video='....mp4')
  print(o1.audio_path)
  print(o1.duration_s)

Strategy (tried in order)
--------------------------
  1. ffmpeg subprocess  — fastest, no Python deps
  2. moviepy            — fallback if ffmpeg not on PATH
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .types import AudioResult

TARGET_SR = 16_000   # sample rate expected by NeMo Canary


def module1(
    video_path: str | Path,
    target_sr:  int = TARGET_SR,
    _out_dir:   Optional[str | Path] = None,
) -> AudioResult:
    """
    Extract audio from *video_path* and write a 16-kHz mono WAV file.

    Parameters
    ----------
    video_path : path to the input video (mp4 / avi / mkv / …)
    target_sr  : output sample rate in Hz  (default 16 000 for ASR)
    _out_dir   : directory for the output WAV  (default: system temp)

    Returns
    -------
    AudioResult
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"[module1] Video not found: {video_path}")

    _print_header(f"Module 1 — Video → Audio  |  {video_path.name}")
    print(f"  Input    : {video_path}")

    out_dir  = Path(_out_dir) if _out_dir else Path(tempfile.mkdtemp(prefix="m1_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / (video_path.stem + "_audio.wav")

    t0 = time.perf_counter()

    if _ffmpeg_available():
        _extract_via_ffmpeg(video_path, wav_path, target_sr)
    else:
        _extract_via_moviepy(video_path, wav_path, target_sr)

    duration = _wav_duration(wav_path)
    elapsed  = time.perf_counter() - t0

    print(f"  Output   : {wav_path}")
    print(f"  Duration : {duration:.1f}s   SR: {target_sr} Hz")
    print(f"  Time     : {elapsed:.2f}s")

    return AudioResult(
        audio_path=wav_path,
        duration_s=duration,
        source_video=video_path,
    )


# ── Private helpers ──────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_via_ffmpeg(
    video_path: Path,
    wav_path:   Path,
    target_sr:  int,
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i",  str(video_path),
        "-vn",                       # drop video stream
        "-ar", str(target_sr),       # sample rate
        "-ac", "1",                  # mono
        "-f",  "wav",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"[module1] ffmpeg failed:\n{result.stderr.decode(errors='replace')}"
        )


def _extract_via_moviepy(
    video_path: Path,
    wav_path:   Path,
    target_sr:  int,
) -> None:
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(
            str(wav_path),
            fps=target_sr,
            nbytes=2,
            ffmpeg_params=["-ac", "1"],
            logger=None,
        )
        clip.close()
    except ImportError:
        raise ImportError(
            "[module1] Neither ffmpeg nor moviepy is available.\n"
            "  Install ffmpeg:  apt-get install ffmpeg\n"
            "  Or moviepy  :  pip install moviepy"
        )


def _wav_duration(wav_path: Path) -> float:
    try:
        import soundfile as sf
        return sf.info(str(wav_path)).duration
    except Exception:
        return 0.0


def _print_header(title: str) -> None:
    bar = "─" * 60
    print(f"\n┌{bar}┐\n│  {title:<58}│\n└{bar}┘")
