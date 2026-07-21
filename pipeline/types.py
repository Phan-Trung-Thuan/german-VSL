# pipeline/types.py
"""
Shared dataclasses that flow between pipeline modules.

Each module consumes one dataclass and produces the next:

  AudioResult  <-- module1
  ASRResult    <-- module2
  GlossResult  <-- module3
  LookupResult <-- module4  (contains list[TokenClip])
  GIFResult    <-- module5
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AudioResult:
    """Output of module1: extracted 16-kHz mono WAV."""
    audio_path:   Path    # path to the written WAV file
    duration_s:   float   # audio length in seconds
    source_video: Path    # original video path

    def __repr__(self):
        return (f"AudioResult(duration={self.duration_s:.1f}s, "
                f"audio='{self.audio_path.name}', "
                f"video='{self.source_video.name}')")


@dataclass
class ASRResult:
    """Output of module2: German transcript from Canary ASR."""
    german_text: str    # raw transcript
    model_id:    str    # which ASR model was used
    elapsed_s:   float  # inference wall-clock time

    def __repr__(self):
        return (f"ASRResult(text={self.german_text!r}, "
                f"model='{self.model_id}', "
                f"time={self.elapsed_s:.1f}s)")


@dataclass
class GlossResult:
    """Output of module3: list of sign-language gloss tokens."""
    gloss_tokens: list[str]   # e.g. ['JETZT', 'WETTER', 'MORGEN']
    gvl_text:     str         # intermediate GVL string before glossing
    source_text:  str         # original German text that was glossed

    def __repr__(self):
        return (f"GlossResult(tokens={self.gloss_tokens}, "
                f"source={self.source_text!r})")


@dataclass
class TokenClip:
    """One gloss token resolved to a video clip."""
    token:      str            # the gloss token (uppercase)
    video_path: Optional[Path] # matched sentence-level video from Phoenix14T
    clip_path:  Optional[Path] # trimmed clip written to disk (temp file)
    found:      bool           # False if token is absent from the lexicon


@dataclass
class LookupResult:
    """Output of module4: per-token video clip lookup."""
    clips:          list[TokenClip]
    found_count:    int
    missing_tokens: list[str]

    def __repr__(self):
        found = [c.token for c in self.clips if c.found]
        return (f"LookupResult(found={found}, "
                f"missing={self.missing_tokens})")


@dataclass
class GIFResult:
    """Output of module5: the rendered animated GIF."""
    gif_path:    Path
    size_kb:     float
    frame_count: int
    token_count: int

    def __repr__(self):
        return (f"GIFResult(gif='{self.gif_path}', "
                f"size={self.size_kb:.1f}KB, "
                f"frames={self.frame_count})")
