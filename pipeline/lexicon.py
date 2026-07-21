# pipeline/lexicon.py
"""
PhoenixLexicon
==============
Reads Phoenix14T annotation files and builds a gloss-token -> video index.

Phoenix14T annotation format (auto-detected):
  Pipe-separated  : id | name | video | start | end | speaker | orth | translation
  Comma-separated : name , signer , gloss , text

The lexicon maps every unique uppercase gloss token to the list of training
examples (video path + gloss sequence) that contain it.  module4 uses this
to look up a clip for each predicted gloss token.

Usage
-----
  from pipeline.lexicon import PhoenixLexicon

  lex = PhoenixLexicon(
      csv_path   = "/kaggle/input/.../phoenix14t.pami0.train.corpus.csv",
      videos_dir = "/kaggle/input/.../videos_phoenix/videos",
      split      = "train",
  )
  print(lex.stats())
  entries = lex.lookup("WETTER")   # list[dict]
  "WETTER" in lex                  # True / False
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


class PhoenixLexicon:
    """
    Gloss-token -> video-clip index built from Phoenix14T annotations.

    Parameters
    ----------
    csv_path            : path to the annotation CSV (pipe- or comma-separated)
    videos_dir          : root directory that contains the video files
    split               : dataset split name used to resolve video paths
                          ('train', 'test', 'dev')
    max_clips_per_token : keep at most this many video references per token
    """

    def __init__(
        self,
        csv_path:            str | Path,
        videos_dir:          str | Path,
        split:               str = "train",
        max_clips_per_token: int = 5,
    ):
        self.csv_path    = Path(csv_path)
        self.videos_dir  = Path(videos_dir)
        self.split       = split
        self.max_clips   = max_clips_per_token

        # index[token] -> list of entry dicts
        self.index: dict[str, list[dict]] = {}
        self._all_examples: list[dict]    = []

        self._build_index()

    @classmethod
    def from_dataset(
        cls,
        dataset,                        # Phoenix14TDataset instance
        videos_dir:          str | Path = "",
        split:               str        = "train",
        max_clips_per_token: int        = 5,
    ) -> "PhoenixLexicon":
        """
        Build a PhoenixLexicon directly from a Phoenix14TDataset object.

        This avoids needing a separate CSV file — the annotations are read
        straight from the dataset's in-memory list (loaded from the .gzip
        pickle files).

        Parameters
        ----------
        dataset             : an instantiated Phoenix14TDataset
        videos_dir          : root directory containing the video files
                              (if empty, uses dataset.video_dir)
        split               : split name for video path resolution
        max_clips_per_token : keep at most this many video refs per token

        Example
        -------
        lex = PhoenixLexicon.from_dataset(trainset, VIDEOS_DIR)
        """
        videos_dir = Path(videos_dir) if videos_dir else Path(dataset.video_dir)

        # Create a bare instance without calling __init__
        inst = cls.__new__(cls)
        inst.csv_path    = Path("<from_dataset>")
        inst.videos_dir  = videos_dir
        inst.split       = split
        inst.max_clips   = max_clips_per_token
        inst.index       = {}
        inst._all_examples = []

        inst._build_index_from_annotations(dataset.annotations, videos_dir, split)
        return inst


    # ── CSV loading ──────────────────────────────────────────────────

    def _load_csv(self) -> list[dict]:
        """Load CSV; auto-detect pipe-separated Phoenix format vs comma."""
        examples = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            sample    = f.read(2048)
            f.seek(0)
            delimiter = "|" if sample.count("|") > sample.count(",") else ","
            reader    = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                row   = {k.strip(): v.strip() for k, v in row.items()}
                name  = row.get("name")  or row.get("id",    "")
                gloss = row.get("gloss") or row.get("orth",  "")
                text  = row.get("text")  or row.get("translation", "")
                video = row.get("video", "")
                if gloss:
                    examples.append({
                        "name":  name,
                        "gloss": gloss,
                        "text":  text,
                        "video": video,
                    })
        return examples

    # ── Video path resolution ────────────────────────────────────────

    def _resolve_video_path(self, example: dict) -> Optional[Path]:
        """Try several path patterns to find the mp4 on disk."""
        name  = example["name"]
        video = example["video"]
        stem  = Path(name).name            # e.g. '11August_2010_...'
        sp    = self.split

        candidates = [
            self.videos_dir / video,                         # provided field
            self.videos_dir / sp / sp / f"{stem}.mp4",      # train/train/...mp4
            self.videos_dir / sp / f"{stem}.mp4",           # train/...mp4
            self.videos_dir / f"{stem}.mp4",                 # flat
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    # ── Index builder ────────────────────────────────────────────────

    def _build_index(self) -> None:
        print(f"[PhoenixLexicon] Loading {self.csv_path.name} …")
        examples           = self._load_csv()
        self._all_examples = examples
        self._index_examples(examples)

    def _build_index_from_annotations(
        self,
        annotations: list[dict],
        videos_dir:  Path,
        split:       str,
    ) -> None:
        """
        Build the index from a list of annotation dicts (from Phoenix14TDataset).

        Each annotation dict must have 'name' and 'gloss' keys.
        The 'text' key is optional.  Video paths are resolved by scanning
        the common Phoenix14T directory patterns under *videos_dir*.
        """
        print(f"[PhoenixLexicon] Building index from {len(annotations)} annotations …")
        examples = []
        for anno in annotations:
            name  = anno.get("name", "")
            gloss = anno.get("gloss", "")
            text  = anno.get("text", "")
            if gloss:
                examples.append({
                    "name":  name,
                    "gloss": gloss,
                    "text":  text,
                    "video": "",   # will be resolved by _resolve_video_path
                })
        self._all_examples = examples
        self._index_examples(examples)

    def _index_examples(self, examples: list[dict]) -> None:
        """Shared indexing logic used by both _build_index and _build_index_from_annotations."""
        video_found = 0
        for ex in examples:
            vp     = self._resolve_video_path(ex)
            tokens = ex["gloss"].upper().split()
            if vp is not None:
                video_found += 1
            for idx, tok in enumerate(tokens):
                entry = {
                    "video":     vp,
                    "gloss_seq": tokens,
                    "token_idx": idx,
                    "name":      ex["name"],
                }
                if tok not in self.index:
                    self.index[tok] = []
                if len(self.index[tok]) < self.max_clips:
                    self.index[tok].append(entry)

        print(f"  Examples loaded : {len(examples)}")
        print(f"  Videos on disk  : {video_found}")
        print(f"  Unique tokens   : {len(self.index)}")

    # ── Public API ───────────────────────────────────────────────────

    def lookup(self, token: str) -> list[dict]:
        """Return up to max_clips_per_token entries for *token*."""
        return self.index.get(token.upper(), [])

    def stats(self) -> str:
        source = str(self.csv_path) if str(self.csv_path) != "<from_dataset>" else "Phoenix14TDataset (in-memory)"
        return (
            f"PhoenixLexicon\n"
            f"  Source       : {source}\n"
            f"  Videos dir   : {self.videos_dir}\n"
            f"  Split        : {self.split}\n"
            f"  Examples     : {len(self._all_examples)}\n"
            f"  Unique tokens: {len(self.index)}"
        )

    def __contains__(self, token: str) -> bool:
        return token.upper() in self.index

    def __len__(self) -> int:
        return len(self._all_examples)

