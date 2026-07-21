# pipeline/utils.py
"""
Pipeline utilities
==================
Convenience functions for running the full pipeline and evaluating against
Phoenix14T ground-truth annotations.

Functions
---------
  display_gif(gif_path)                 — show GIF inline in Kaggle/Jupyter
  run_pipeline(video_path, lex, ...)    — run all 5 modules end-to-end
  evaluate_example(example, lex, ...)  — compute precision/recall/F1/WER
                                         vs. ground-truth gloss
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .types   import AudioResult, ASRResult, GlossResult, LookupResult, GIFResult
from .lexicon import PhoenixLexicon
from .signdict_scraper import SignDictScraper
from .module1 import module1
from .module2 import module2
from .module3 import module3
from .module4 import module4
from .module5 import module5


# ── Display ──────────────────────────────────────────────────────────────────

def display_gif(gif_path: str | Path) -> None:
    """Display an animated GIF inline in a Kaggle / Jupyter notebook cell."""
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(filename=str(gif_path)))
    except ImportError:
        print(f"[display_gif] GIF saved at: {gif_path}  (open manually)")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    video_path:  str | Path,
    lexicon:     Optional[PhoenixLexicon] = None,
    output_gif:  str | Path = "output.gif",
    asr_model:   str        = "nvidia/canary-1b",
    clip_duration_s: float  = 2.0,
    gif_fps:     int        = 10,
    gif_width:   int        = 320,
    scraper:     Optional[SignDictScraper] = None,
) -> dict:
    """
    Run all 5 modules end to end and return all intermediate results.

    Returns
    -------
    dict with keys: 'o1', 'o2', 'o3', 'o4', 'o5'
    """
    t0 = time.perf_counter()

    o1 = module1(video_path)
    o2 = module2(o1, model_id=asr_model)
    o3 = module3(o2)
    o4 = module4(o3, lexicon, clip_duration_s=clip_duration_s, scraper=scraper)
    o5 = module5(o4, output_gif, fps=gif_fps, width=gif_width)


    elapsed = time.perf_counter() - t0
    bar = "─" * 60
    print(f"\n┌{bar}┐\n│  {'Pipeline complete':58}│\n└{bar}┘")
    print(f"  German text  : {o2.german_text!r}")
    print(f"  Gloss tokens : {o3.gloss_tokens}")
    print(f"  Found        : {o4.found_count} / {len(o3.gloss_tokens)}")
    print(f"  GIF          : {o5.gif_path}  ({o5.size_kb:.1f} KB)")
    print(f"  Total time   : {elapsed:.1f}s")
    print(bar)

    return {"o1": o1, "o2": o2, "o3": o3, "o4": o4, "o5": o5}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_example(
    example:   dict,
    lexicon:   PhoenixLexicon,
    asr_model: str = "nvidia/canary-1b",
) -> dict:
    """
    Run the pipeline on one Phoenix14T example and compare to ground truth.

    The *example* dict must contain:
      'video_path' : path to the mp4
      'gloss'      : ground-truth gloss string  (e.g. 'JETZT WETTER MORGEN')
      'text'       : ground-truth German text   (optional, for reference)

    Returns a metrics dict:
      name, gt_text, pred_text, gt_gloss, pred_gloss,
      precision, recall, f1, gloss_wer
    """
    video_path = Path(example.get("video_path", ""))
    gt_gloss   = example.get("gloss", "").upper().split()
    gt_text    = example.get("text", "")

    o1 = module1(video_path)
    o2 = module2(o1, model_id=asr_model)
    o3 = module3(o2)

    pred_gloss = o3.gloss_tokens
    pred_set   = set(pred_gloss)
    gt_set     = set(gt_gloss)

    tp        = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gt_set)   if gt_set   else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "name":       example.get("name", ""),
        "gt_text":    gt_text,
        "pred_text":  o2.german_text,
        "gt_gloss":   gt_gloss,
        "pred_gloss": pred_gloss,
        "precision":  round(precision, 3),
        "recall":     round(recall, 3),
        "f1":         round(f1, 3),
        "gloss_wer":  round(_wer(gt_gloss, pred_gloss), 3),
    }


def _wer(ref: list[str], hyp: list[str]) -> float:
    """Compute Word-Error-Rate (edit distance) between two token lists."""
    import numpy as np
    r, h = list(ref), list(hyp)
    d    = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost    = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1,
                          d[i][j-1] + 1,
                          d[i-1][j-1] + cost)
    return d[len(r)][len(h)] / max(len(r), 1)
