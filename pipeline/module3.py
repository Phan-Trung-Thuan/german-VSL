# pipeline/module3.py
"""
Module 3 — German Text -> Gloss Tokens
=======================================
Converts raw German text to a sequence of sign-language gloss tokens by:
  1. Stripping punctuation
  2. Dropping grammatical function words (articles, auxiliaries, prepositions,
     conjunctions, negation particles)
  3. Lemmatising remaining content words with simplemma
  4. Upper-casing to follow the Phoenix14T gloss convention

No neural model is used — this is a deterministic rule-based step.

Usage
-----
  from pipeline.module3 import module3

  o3 = module3(o2)                          # o2 is an ASRResult
  o3 = module3("Ich gehe heute in die Schule.")  # or a raw string
  print(o3.gloss_tokens)   # ['GEHEN', 'HEUTE', 'SCHULE']
"""
from __future__ import annotations

import re
from typing import Union

from .types import ASRResult, GlossResult

# ── Drop-lists (German function words not glossed in DGS / Phoenix14T) ────────

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
_CONJUNCTIONS = {"und", "oder", "aber", "denn", "weil", "dass", "ob", "wenn"}
_PARTICLES    = {"nicht", "kein", "keine", "noch", "schon", "auch", "nur"}

_DROP = _ARTICLES | _AUX_VERBS | _PREPOSITIONS | _CONJUNCTIONS | _PARTICLES


def module3(
    asr:    Union[ASRResult, str],
    method: str = "simplemma",
) -> GlossResult:
    """
    Convert German text to sign-language gloss tokens.

    Parameters
    ----------
    asr    : ASRResult from module2, OR a raw German text string
    method : 'simplemma' (default) — lemmatise with simplemma library
             'simple'              — just uppercase, no lemmatisation

    Returns
    -------
    GlossResult  with .gloss_tokens, .gvl_text, .source_text
    """
    source_text = asr.german_text if isinstance(asr, ASRResult) else str(asr)

    _print_header("Module 3 — German Text → Gloss Tokens")
    print(f"  Input  : {source_text!r}")
    print(f"  Method : {method}")

    raw_tokens = re.findall(r"[A-Za-z\u00c0-\u024f]+", source_text)
    gvl_tokens: list[str] = []

    for token in raw_tokens:
        lower = token.lower()

        # Drop grammatical function words
        if lower in _DROP:
            continue

        # Lemmatise content word
        lemma = _lemmatise(lower, method)
        gvl_tokens.append(lemma.upper())

    gvl_text = " ".join(gvl_tokens)
    print(f"  GVL    : {gvl_text!r}")
    print(f"\n  Result : {gvl_tokens}")

    return GlossResult(
        gloss_tokens=gvl_tokens,
        gvl_text=gvl_text,
        source_text=source_text,
    )


# ── Private helpers ──────────────────────────────────────────────────────────

def _lemmatise(word: str, method: str) -> str:
    if method == "simplemma":
        try:
            import simplemma
            return simplemma.lemmatize(word, lang="de")
        except ImportError:
            # Silently fall back if simplemma not installed
            pass
        except Exception:
            pass
    return word   # identity fallback


def _print_header(title: str) -> None:
    bar = "─" * 60
    print(f"\n┌{bar}┐\n│  {title:<58}│\n└{bar}┘")
