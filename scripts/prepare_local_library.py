# prepare_local_library.py
"""
PHOENIX-Weather-2014T SignDict Local Vocabulary Builder
======================================================
1. Reads all 1,085 unique gloss tokens from the PHOENIX dataset pickle split.
2. Cleanses tokens (handling splits, slashes, and suffix flags like -SCHALTUNG).
3. Uses the SignDictScraper to fetch and download sign MP4 files.
4. Generates an offline JSON lookup mapping: {GLOSS: local_video_path}.
"""
from __future__ import annotations

import gzip
import json
import pickle
import time
from pathlib import Path
from pipeline import SignDictScraper


def load_raw_glosses(gz_path: str) -> set[str]:
    """Read unique uppercase gloss tokens from PHOENIX annotations."""
    print(f"Reading dataset: {gz_path} ...")
    with gzip.open(gz_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        data = list(data.values())

    unique_tokens = set()
    for item in data:
        gloss_seq = item.get("gloss", "")
        for tok in gloss_seq.upper().split():
            # Basic validation
            if tok.strip():
                unique_tokens.add(tok.strip())
    return unique_tokens


def clean_token(token: str) -> list[str]:
    """
    Clean and resolve token variation down to searchable words.
    Example:
      'ER|ES|SIE'       -> ['ER']
      'EIS-SCHALTUNG'   -> ['EIS']
      'WIND-SCHALTUNG'  -> ['WIND']
      'ZUM-BEISPIEL'    -> ['BEISPIEL']
    """
    # 1. Handle slashes/pipes
    tok = token.split("|")[0].split("/")[0].strip()
    
    # 2. Strip -SCHALTUNG suffix flags
    if tok.endswith("-SCHALTUNG"):
        tok = tok[:-10]

    # 3. Handle dash compounds
    if "-" in tok:
        parts = [p.strip() for p in tok.split("-") if p.strip()]
        # If it's a known multi-word compound like AUF-JEDEN-FALL, AUF WIEDERSEHEN
        if tok in ["AUF-JEDEN-FALL", "AUF-JEDER-FALL"]:
            return [tok.replace("-", " ")]
        # Otherwise return first noun or split elements
        if parts:
            return parts
            
    return [tok]


def main():
    gz_path = "phoenix14t.pami0.train.annotations_only.gz"
    output_dir = Path("signdict/videos")
    mapping_file = Path("signdict/signdict_mapping.json")

    # 1. Extract vocabulary
    try:
        raw_glosses = load_raw_glosses(gz_path)
    except FileNotFoundError:
        print(f"Error: {gz_path} not found. Please place it in the workspace directory.")
        return

    print(f"Total raw gloss tokens extracted: {len(raw_glosses)}")

    # 2. Init scraper (run silently)
    scraper = SignDictScraper(output_dir=output_dir, verbose=False)

    # Load existing mapping if any
    mapping = {}
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            print(f"Loaded {len(mapping)} existing entries from {mapping_file}")
        except Exception:
            mapping = {}

    # 3. Run mapping batch
    print("Starting SignDict local library preparation...")
    print("This will download signs. Progress summary will update periodically.")

    resolved_count = 0
    failed_count = 0

    # Sort to run consistently
    sorted_glosses = sorted(list(raw_glosses))

    for idx, raw_tok in enumerate(sorted_glosses):
        if raw_tok in mapping and Path(mapping[raw_tok]).exists():
            resolved_count += 1
            continue

        # Get search candidates
        search_words = clean_token(raw_tok)
        video_path = None

        for word in search_words:
            # Query / download
            path = scraper.get_sign(word, delay_s=0.2)
            if path and path.exists():
                video_path = path
                break

        if video_path:
            mapping[raw_tok] = str(video_path.resolve().relative_to(Path(".").resolve()))
            resolved_count += 1
        else:
            failed_count += 1

        # Periodically save mapping progress
        if idx % 20 == 0 or idx == len(sorted_glosses) - 1:
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            print(f"Progress: {idx+1}/{len(sorted_glosses)} processed. "
                  f"Resolved: {resolved_count}, Failed/Missing: {failed_count}")

    print("\nLocal sign vocabulary library preparation complete!")
    print(f"  Mapping saved to: {mapping_file}")
    print(f"  Total mapped signs: {resolved_count}")
    print(f"  Missing signs     : {failed_count}")


if __name__ == "__main__":
    main()
