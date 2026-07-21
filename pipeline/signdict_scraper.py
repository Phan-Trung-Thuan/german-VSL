# pipeline/signdict_scraper.py
"""
SignDict Scraper
================
Crawl signdict.org entries directly from HTML to resolve German glosses
to their corresponding sign language MP4 video URLs.

Usage
-----
  from pipeline.signdict_scraper import SignDictScraper

  scraper = SignDictScraper(output_dir="/kaggle/working/signdict_videos")
  video_path = scraper.get_sign("JA")   # Returns local path or None
"""
from __future__ import annotations

import re
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional


class SignDictScraper:
    """
    HTML-based scraper for SignDict.org (German Sign Language).
    Bypasses API restrictions by fetching search and entry detail pages.
    """
    BASE_URL = "https://signdict.org"

    def __init__(self, output_dir: str | Path = "signdict_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # In-memory mapping: gloss -> video_path
        self.downloaded_cache: dict[str, Path] = {}
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Scan the output folder for already downloaded videos."""
        for p in self.output_dir.glob("*.mp4"):
            self.downloaded_cache[p.stem.upper()] = p

    def get_sign_url(self, word: str) -> Optional[str]:
        """
        Search for a word on SignDict and extract the video URL from the detail page.
        """
        encoded_word = urllib.parse.quote(word.lower())
        search_url = f"{self.BASE_URL}/entry/{encoded_word}"

        # 1. Fetch search page
        req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8")
        except Exception as e:
            print(f"  [SignDict Scraper] Error fetching search page for '{word}': {e}")
            return None

        # Check if we were redirected directly to an entry page
        # (Entry pages contain the assets.wishlephant.com video links)
        video_url = self._extract_video_url(html)
        if video_url:
            return video_url

        # Otherwise, find detail page links from search results (e.g. /entry/377-ja)
        # We look for links matching: /entry/ID-word
        entry_paths = re.findall(r'href="(/entry/\d+-[^"]+)"', html)
        if not entry_paths:
            return None

        # Visit the first matching entry detail page
        detail_url = f"{self.BASE_URL}{entry_paths[0]}"
        detail_req = urllib.request.Request(detail_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(detail_req, timeout=10) as response:
                detail_html = response.read().decode("utf-8")
                return self._extract_video_url(detail_html)
        except Exception as e:
            print(f"  [SignDict Scraper] Error fetching detail page '{detail_url}': {e}")

        return None

    def _extract_video_url(self, html: str) -> Optional[str]:
        """Helper to parse MP4 video source URL from page HTML."""
        # Strategy A: og:video:url tag
        og_match = re.search(r'property="og:video:url"\s+content="([^"]+\.mp4[^"]*)"', html)
        if og_match:
            return og_match.group(1)

        # Strategy B: video tag src attribute
        video_match = re.search(r'<video[^>]+src="([^"]+\.mp4[^"]*)"', html)
        if video_match:
            return video_match.group(1)

        # Strategy C: general assets link
        assets_match = re.search(r'"(https://assets\.wishlephant\.com/[^"]+\.mp4)"', html)
        if assets_match:
            return assets_match.group(1)

        return None

    def get_sign(self, gloss: str, delay_s: float = 0.5) -> Optional[Path]:
        """
        Get the video path for a gloss. If not downloaded yet, search SignDict,
        download the MP4, and cache the path.
        """
        # Clean up gloss token (e.g. ER|ES|SIE -> ER)
        clean_gloss = gloss.split("|")[0].strip().upper()
        if not clean_gloss:
            return None

        if clean_gloss in self.downloaded_cache:
            return self.downloaded_cache[clean_gloss]

        print(f"  [SignDict] Searching for '{clean_gloss}'...")
        video_url = self.get_sign_url(clean_gloss)
        if not video_url:
            print(f"  [SignDict] Gloss '{clean_gloss}' not found.")
            return None

        # Build local target file path
        target_path = self.output_dir / f"{clean_gloss}.mp4"
        print(f"  [SignDict] Downloading {video_url} -> {target_path} ...")

        try:
            req = urllib.request.Request(video_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
                while True:
                    buf = response.read(16 * 1024)
                    if not buf:
                        break
                    out_file.write(buf)
            
            self.downloaded_cache[clean_gloss] = target_path
            time.sleep(delay_s)  # Respect server rate limits
            return target_path
        except Exception as e:
            print(f"  [SignDict] Failed to download video for '{clean_gloss}': {e}")
            if target_path.exists():
                target_path.unlink()
        return None
