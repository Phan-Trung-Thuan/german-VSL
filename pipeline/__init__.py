# pipeline/__init__.py
"""
pipeline/
=========
Modular research pipeline for Phoenix14T (German Sign Language dataset).

Modules
-------
  pipeline.types     — shared dataclasses (AudioResult, ASRResult, …)
  pipeline.lexicon   — PhoenixLexicon: builds gloss-to-video index
  pipeline.module1   — Video  -> Audio
  pipeline.module2   — Audio  -> German text  (NeMo Canary ASR)
  pipeline.module3   — German text -> Gloss tokens
  pipeline.module4   — Gloss tokens -> Video clips  (Phoenix14T lookup)
  pipeline.module5   — Video clips -> Animated GIF

Usage
-----
  from pipeline import module1, module2, module3, module4, module5
  from pipeline import PhoenixLexicon
  from pipeline import display_gif, run_pipeline, evaluate_example
"""

from .types   import AudioResult, ASRResult, GlossResult, TokenClip, LookupResult, GIFResult
from .lexicon import PhoenixLexicon
from .signdict_scraper import SignDictScraper
from .module1 import module1
from .module2 import module2
from .module3 import module3
from .module4 import module4
from .module5 import module5
from .utils   import display_gif, run_pipeline, evaluate_example

__all__ = [
    # result types
    "AudioResult", "ASRResult", "GlossResult",
    "TokenClip", "LookupResult", "GIFResult",
    # lexicon
    "PhoenixLexicon",
    "SignDictScraper",
    # modules
    "module1", "module2", "module3", "module4", "module5",
    # helpers
    "display_gif", "run_pipeline", "evaluate_example",
]

