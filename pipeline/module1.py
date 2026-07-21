# pipeline/module1.py
"""
Module 1 -- Video / Audio / URL -> 16-kHz mono WAV
====================================================
Accepts any of the following as input:

  Local paths
  -----------
  Video  (mp4, avi, mkv, mov, ...)
      -> extracts audio with ffmpeg / moviepy

  Audio  (wav, mp3, flac, m4a, ogg, ...)
      -> resamples / converts with ffmpeg / librosa

  URLs  (auto-detected when input starts with http:// or https://)
  -------------------------------------------------------------------
  YouTube / yt-dlp-supported sites
      youtube.com/watch?v=...  |  youtu.be/...  |  vimeo.com/...
      -> downloaded with yt-dlp, then audio extracted

  Google Drive share links
      drive.google.com/file/d/FILE_ID/view
      -> downloaded with gdown

  Direct download links  (URL path ends with a media extension)
      https://example.com/video.mp4
      https://example.com/audio.mp3
      -> downloaded with requests / urllib

  Any other URL
      -> yt-dlp attempted as a catch-all fallback

Usage
-----
  from pipeline.module1 import module1

  o1 = module1("/path/to/video.mp4")
  o1 = module1("/path/to/audio.mp3")
  o1 = module1("https://www.youtube.com/watch?v=sRuPeaDwKsY")
  o1 = module1("https://drive.google.com/file/d/FILE_ID/view")
  o1 = module1("https://example.com/sample.mp4")

  print(o1)             # AudioResult(duration=33.1s, ...)
  print(o1.input_type)  # 'video' | 'audio' | 'youtube' | 'gdrive' | 'url'
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Optional

from .types import AudioResult

TARGET_SR = 16_000  # sample rate expected by NeMo Canary

_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus",
    ".aac", ".wma", ".aiff", ".aif",
}
_MEDIA_EXTENSIONS = _AUDIO_EXTENSIONS | {
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".ts",
}

_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}
_GDRIVE_HOSTS  = {"drive.google.com"}


def module1(
    input_path: str | Path,
    target_sr:  int = TARGET_SR,
    _out_dir:   Optional[str | Path] = None,
) -> AudioResult:
    """
    Convert a local video/audio file OR a URL to a 16-kHz mono WAV.

    Parameters
    ----------
    input_path : local path  OR  URL (YouTube / Google Drive / direct link)
    target_sr  : output sample rate in Hz  (default 16 000 for NeMo ASR)
    _out_dir   : directory for temp files and the output WAV

    Returns
    -------
    AudioResult  (.audio_path, .duration_s, .source_video, .input_type)
    """
    out_dir = Path(_out_dir) if _out_dir else Path(tempfile.mkdtemp(prefix="m1_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    src = str(input_path)

    # ── URL branch ────────────────────────────────────────────────────
    if src.startswith("http://") or src.startswith("https://"):
        local_file, input_type = _download_url(src, out_dir)
    else:
        local_file = Path(src)
        if not local_file.exists():
            raise FileNotFoundError(f"[module1] File not found: {local_file}")
        ext        = local_file.suffix.lower()
        input_type = "audio" if ext in _AUDIO_EXTENSIONS else "video"

    _print_header(f"Module 1 -- {input_type} -> 16-kHz WAV  |  {Path(local_file).name}")
    print(f"  Input    : {src}")
    print(f"  Type     : {input_type}")
    print(f"  Local    : {local_file}")

    wav_path = out_dir / (Path(local_file).stem + "_16k.wav")

    t0 = time.perf_counter()

    ext = Path(local_file).suffix.lower()
    if ext in _AUDIO_EXTENSIONS:
        _convert_audio(local_file, wav_path, target_sr)
    else:
        _extract_from_video(local_file, wav_path, target_sr)

    duration = _wav_duration(wav_path)
    elapsed  = time.perf_counter() - t0

    print(f"  Output   : {wav_path}")
    print(f"  Duration : {duration:.1f}s   SR: {target_sr} Hz")
    print(f"  Time     : {elapsed:.2f}s")

    result = AudioResult(
        audio_path=wav_path,
        duration_s=duration,
        source_video=Path(local_file),
    )
    result.input_type   = input_type
    result.source_url   = src if (src.startswith("http://") or src.startswith("https://")) else None
    return result


# ── URL dispatch ──────────────────────────────────────────────────────────────

def _download_url(url: str, out_dir: Path) -> tuple[Path, str]:
    """
    Detect URL type and dispatch to the appropriate downloader.
    Returns (local_file_path, input_type_string).
    """
    host = urllib.parse.urlparse(url).hostname or ""
    path = urllib.parse.urlparse(url).path.lower()

    if host in _YOUTUBE_HOSTS or "youtube" in host:
        return _download_youtube(url, out_dir), "youtube"

    if host in _GDRIVE_HOSTS:
        return _download_gdrive(url, out_dir), "gdrive"

    # Direct media link (URL path ends with a known extension)
    if any(path.endswith(ext) for ext in _MEDIA_EXTENSIONS):
        return _download_direct(url, out_dir), "url"

    # Unknown URL -- try yt-dlp as catch-all (handles 1000+ sites)
    print(f"  [INFO] Unknown URL host '{host}', trying yt-dlp ...")
    return _download_youtube(url, out_dir), "url"


# ── YouTube / yt-dlp ─────────────────────────────────────────────────────────

def _download_youtube(url: str, out_dir: Path) -> Path:
    """
    Download a video with yt-dlp and return the local mp4 path.
    Installs yt-dlp automatically if not found.
    """
    if not shutil.which("yt-dlp"):
        print("  [INFO] yt-dlp not found, installing ...")
        subprocess.run(
            ["pip", "install", "-q", "yt-dlp"],
            check=True, capture_output=True,
        )

    out_template = str(out_dir / "yt_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", out_template,
        "--no-playlist",
        url,
    ]
    print(f"  [yt-dlp] Downloading {url} ...")
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"[module1] yt-dlp failed:\n{result.stderr.decode(errors='replace')}"
        )

    # Find the downloaded file
    downloaded = sorted(out_dir.glob("yt_*"))
    if not downloaded:
        raise RuntimeError("[module1] yt-dlp ran but no output file found.")
    return downloaded[-1]


# ── Google Drive ─────────────────────────────────────────────────────────────

def _download_gdrive(url: str, out_dir: Path) -> Path:
    """
    Download a file from a Google Drive share link using gdown.
    Supports both /file/d/FILE_ID/view and ?id=FILE_ID formats.
    """
    try:
        import gdown
    except ImportError:
        print("  [INFO] gdown not found, installing ...")
        subprocess.run(
            ["pip", "install", "-q", "gdown"],
            check=True, capture_output=True,
        )
        import gdown

    out_path = str(out_dir / "gdrive_download")
    print(f"  [gdown] Downloading from Google Drive ...")
    downloaded = gdown.download(url, out_path, quiet=False, fuzzy=True)
    if downloaded is None:
        raise RuntimeError(
            "[module1] gdown failed to download the file.\n"
            "  Make sure the Google Drive file is set to 'Anyone with the link can view'."
        )
    # gdown may append the real extension
    local = Path(downloaded)
    if not local.exists():
        # Try with common extensions
        for ext in [".mp4", ".avi", ".mkv", ".mp3", ".wav"]:
            candidate = Path(out_path + ext)
            if candidate.exists():
                return candidate
        raise RuntimeError(f"[module1] gdown downloaded to unknown path: {downloaded}")
    return local


# ── Direct download ───────────────────────────────────────────────────────────

def _download_direct(url: str, out_dir: Path) -> Path:
    """
    Download a direct media URL with requests (or urllib as fallback).
    """
    parsed   = urllib.parse.urlparse(url)
    filename = Path(parsed.path).name or "download"
    out_path = out_dir / filename

    print(f"  [HTTP] Downloading {url} ...")
    try:
        import requests
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    except ImportError:
        # urllib fallback
        import urllib.request
        urllib.request.urlretrieve(url, str(out_path))

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"[module1] Direct download failed or empty file: {url}")
    return out_path


# ── Audio conversion ──────────────────────────────────────────────────────────

def _convert_audio(audio_path: Path, wav_path: Path, target_sr: int) -> None:
    """Resample any audio file to target_sr mono WAV via ffmpeg / librosa."""
    if _ffmpeg_available():
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", str(target_sr), "-ac", "1", "-f", "wav", str(wav_path),
        ]
        if subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0:
            return
        print("  [WARN] ffmpeg audio conversion failed, trying librosa ...")

    try:
        import librosa, soundfile as sf
        audio, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
        sf.write(str(wav_path), audio, target_sr, subtype="PCM_16")
    except ImportError as e:
        raise ImportError(
            f"[module1] Cannot convert audio: {e}\n"
            "  pip install ffmpeg-python librosa soundfile"
        )


# ── Video audio extraction ────────────────────────────────────────────────────

def _extract_from_video(video_path: Path, wav_path: Path, target_sr: int) -> None:
    """Extract audio from a video file via ffmpeg / moviepy."""
    if _ffmpeg_available():
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", str(target_sr), "-ac", "1", "-f", "wav", str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0:
            return
        raise RuntimeError(
            f"[module1] ffmpeg failed:\n{result.stderr.decode(errors='replace')}"
        )

    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(
            str(wav_path), fps=target_sr, nbytes=2,
            ffmpeg_params=["-ac", "1"], logger=None,
        )
        clip.close()
    except ImportError:
        raise ImportError(
            "[module1] Neither ffmpeg nor moviepy available.\n"
            "  apt-get install ffmpeg  OR  pip install moviepy"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _wav_duration(wav_path: Path) -> float:
    try:
        import soundfile as sf
        return sf.info(str(wav_path)).duration
    except Exception:
        return 0.0


def _print_header(title: str) -> None:
    bar = "-" * 60
    print(f"\n+{bar}+\n|  {title:<58}|\n+{bar}+")
