# pipeline/signdict_scraper.py
"""
SignDict Scraper
================
Queries the official SignDict GraphQL API to resolve German glosses to
direct MP4 video download URLs, and downloads them to a local directory.

Usage
-----
  from pipeline.signdict_scraper import SignDictScraper

  scraper = SignDictScraper(output_dir="/kaggle/working/signdict_videos")

  # Resolve and download one gloss
  video_path = scraper.get_sign("REGNEN")   # returns Path or None
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

# Standard Python libraries
import urllib.request
import urllib.parse


class SignDictScraper:
    """
    GraphQL API-based downloader for SignDict.org (German Sign Language).
    """
    API_URL = "https://signdict.org/api/graphql"

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

    def query_api(self, word: str) -> Optional[str]:
        """
        Query SignDict GraphQL API for a word and return the MP4 video URL if found.
        """
        query = """
        query SearchWord($word: String!) {
          search(word: $word) {
            text
            currentVideo {
              videoUrl
            }
          }
        }
        """
        variables = {"word": word.lower()}
        payload = {"query": query, "variables": variables}
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.API_URL,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                res = json.loads(response.read().decode("utf-8"))
                results = res.get("data", {}).get("search", [])
                if not results:
                    return None
                # Pick the first result that has a video URL
                for item in results:
                    video = item.get("currentVideo")
                    if video and video.get("videoUrl"):
                        return video["videoUrl"]
        except Exception as e:
            print(f"  [SignDict API] Error querying '{word}': {e}")
        return None

    def get_sign(self, gloss: str, delay_s: float = 1.0) -> Optional[Path]:
        """
        Get the video path for a gloss. If not downloaded yet, query SignDict,
        download the MP4, and cache the path.
        """
        gloss_upper = gloss.upper()
        if gloss_upper in self.downloaded_cache:
            return self.downloaded_cache[gloss_upper]

        print(f"  [SignDict] Querying API for '{gloss_upper}'...")
        video_url = self.query_api(gloss_upper)
        if not video_url:
            print(f"  [SignDict] Gloss '{gloss_upper}' not found.")
            return None

        # Build local target file path
        target_path = self.output_dir / f"{gloss_upper}.mp4"
        print(f"  [SignDict] Downloading {video_url} -> {target_path} ...")

        try:
            # Add user agent to bypass headers restrictions
            req = urllib.request.Request(
                video_url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
                shutil_copy(response, out_file)
            
            self.downloaded_cache[gloss_upper] = target_path
            time.sleep(delay_s)  # Respect server rate limits
            return target_path
        except Exception as e:
            print(f"  [SignDict] Failed to download video for '{gloss_upper}': {e}")
            if target_path.exists():
                target_path.unlink() # Clean up broken file
        return None


def shutil_copy(src, dst):
    """Simple buffer copy since we're using urllib streams."""
    while True:
        buf = src.read(16*1024)
        if not buf:
            break
        dst.write(buf)
