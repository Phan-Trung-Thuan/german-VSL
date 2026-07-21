# pipeline/module5.py
"""
Module 5 — Video Clips -> Animated GIF
=======================================
Reads the video clips produced by module4, extracts frames with OpenCV,
optionally burns the gloss token label onto each frame, and saves the
result as an animated GIF using Pillow.

Dependencies
------------
  pip install opencv-python Pillow

Usage
-----
  from pipeline.module5 import module5

  o5 = module5(o4, "output.gif")       # o4 is a LookupResult
  o5 = module5([Path("a.mp4"), ...], "output.gif")  # or a list of paths
  print(o5.gif_path, o5.size_kb)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

from .types import LookupResult, GIFResult


def module5(
    lookup:              Union[LookupResult, list[Path]],
    output_gif:          Union[str, Path] = "output.gif",
    fps:                 int  = 10,
    width:               int  = 320,
    max_frames_per_clip: int  = 30,
    include_label:       bool = True,
) -> GIFResult:
    """
    Stitch per-token video clips into an animated GIF.

    Parameters
    ----------
    lookup              : LookupResult from module4, OR a list of video Paths
    output_gif          : output path for the GIF file
    fps                 : frames per second in the output GIF
    width               : resize each frame to this width (height auto-scaled)
    max_frames_per_clip : maximum frames extracted per clip  (keeps GIF small)
    include_label       : if True, burn the gloss token name onto each frame

    Returns
    -------
    GIFResult  with .gif_path, .size_kb, .frame_count, .token_count
    """
    _require("cv2",  "opencv-python")
    _require("PIL",  "Pillow")
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    _print_header("Module 5 — Video Clips → Animated GIF")

    pairs = _collect_pairs(lookup)
    if not pairs:
        raise ValueError(
            "[module5] No video clips available in LookupResult."
        )

    output_gif = Path(output_gif)
    print(f"  Clips  : {len(pairs)} token(s)")
    print(f"  Output : {output_gif}")

    font = _load_font()
    all_frames: list[Image.Image] = []

    for clip_path, label in pairs:
        if clip_path is None or not clip_path.exists():
            print(f"  [SKIP] {label} — clip file missing")
            continue

        frames = _extract_frames(
            clip_path, label, width,
            max_frames_per_clip, include_label,
            cv2=cv2, Image=Image,
            ImageDraw=ImageDraw, font=font,
        )
        all_frames.extend(frames)
        print(f"  [{label:<20}] {len(frames)} frames  ({clip_path.name})")

    if not all_frames:
        raise RuntimeError("[module5] No frames could be extracted from any clip.")

    # ── Save GIF ─────────────────────────────────────────────────────
    frame_ms = int(1000 / fps)
    all_frames[0].save(
        str(output_gif),
        save_all=True,
        append_images=all_frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
    )

    size_kb = output_gif.stat().st_size / 1024
    print(f"\n  Frames : {len(all_frames)}")
    print(f"  Size   : {size_kb:.1f} KB")
    print(f"  Result : {output_gif}  ✓")

    return GIFResult(
        gif_path=output_gif,
        size_kb=size_kb,
        frame_count=len(all_frames),
        token_count=len(pairs),
    )


# ── Private helpers ──────────────────────────────────────────────────────────

def _collect_pairs(
    lookup: Union[LookupResult, list],
) -> list[tuple[Optional[Path], str]]:
    """Return a list of (clip_path, label) from a LookupResult or path list."""
    if isinstance(lookup, LookupResult):
        return [
            (c.clip_path or c.video_path, c.token)
            for c in lookup.clips
            if c.found and (c.clip_path or c.video_path)
        ]
    return [(Path(p), Path(p).stem) for p in lookup if p is not None]


def _extract_frames(
    clip_path:  Path,
    label:      str,
    width:      int,
    max_frames: int,
    label_on:   bool,
    *,
    cv2,
    Image,
    ImageDraw,
    font,
) -> list:
    frames = []
    cap    = cv2.VideoCapture(str(clip_path))
    count  = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w    = frame.shape[:2]
        new_h   = max(1, int(h * width / w))
        frame   = cv2.resize(frame, (width, new_h))
        img     = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if label_on:
            img = _burn_label(img, label, width, ImageDraw, font)
        frames.append(img)
        count += 1
    cap.release()
    return frames


def _burn_label(img, label: str, width: int, ImageDraw, font) -> object:
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (width, 28)], fill=(0, 0, 0, 180))
    draw.text((8, 5), label, fill=(255, 220, 0), font=font)
    return img


def _load_font():
    from PIL import ImageFont
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, 18)
        except Exception:
            pass
    return ImageFont.load_default()


def _require(package: str, install: str) -> None:
    try:
        __import__(package)
    except ImportError:
        raise ImportError(
            f"[module5] '{package}' not installed. "
            f"Run:  pip install {install}"
        )


def _print_header(title: str) -> None:
    bar = "─" * 60
    print(f"\n┌{bar}┐\n│  {title:<58}│\n└{bar}┘")
