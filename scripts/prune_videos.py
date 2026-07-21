# scripts/prune_videos.py
"""
Motion-Aware Video Pruner
=========================
Pre-processes MP4 videos in `signdict/videos/` by discarding static/still frames
using absolute frame differences (|ΔI|).

Output:
  Cleaned, motion-focused MP4 videos saved in `signdict/videos_pruned/`.

Usage:
  python scripts/prune_videos.py --threshold 2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install requirements: pip install opencv-python numpy tqdm")
    import sys
    sys.exit(1)


def prune_video(
    video_path: Path,
    output_dir: Path,
    motion_threshold: float = 2.0
) -> tuple[Path, int, int]:
    """
    Reads video_path, filters static frames where mean absolute pixel difference < threshold,
    and writes active motion frames to a new video file in output_dir.

    Returns:
      (output_video_path, total_original_frames, total_kept_frames)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_video_path = output_dir / video_path.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    total_frames = 0
    kept_frames = 0
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check motion relative to previous frame
        if prev_gray is not None and motion_threshold > 0:
            diff = cv2.absdiff(gray_frame, prev_gray)
            motion_score = np.mean(diff)
            if motion_score < motion_threshold:
                # Static freeze frame — skip writing
                continue

        prev_gray = gray_frame
        out_writer.write(frame)
        kept_frames += 1

    cap.release()
    out_writer.release()
    return out_video_path, total_frames, kept_frames


def main():
    parser = argparse.ArgumentParser(description="Prune static frames from videos based on motion threshold.")
    parser.add_argument("--input_dir", type=str, default="signdict/videos", help="Source videos directory")
    parser.add_argument("--output_dir", type=str, default="signdict/videos_pruned", help="Pruned videos directory")
    parser.add_argument("--threshold", type=float, default=2.0, help="Mean pixel diff threshold to keep frame (default: 2.0)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(input_dir.glob("*.mp4"))
    if not video_files:
        print(f"No MP4 files found in '{input_dir.resolve()}'")
        return

    print(f"Starting Motion-Aware Video Pruning on {len(video_files)} videos...")
    print(f"Input Directory  : {input_dir.resolve()}")
    print(f"Output Directory : {output_dir.resolve()}")
    print(f"Motion Threshold : {args.threshold}\n")

    total_orig_all = 0
    total_kept_all = 0

    for v_path in tqdm(video_files, desc="Pruning Videos"):
        try:
            _, orig_cnt, kept_cnt = prune_video(v_path, output_dir, args.threshold)
            total_orig_all += orig_cnt
            total_kept_all += kept_cnt
        except Exception as e:
            print(f"\n[Error] Failed processing {v_path.name}: {e}")

    pruned_pct = (1.0 - (total_kept_all / total_orig_all)) * 100 if total_orig_all > 0 else 0.0
    print(f"\nPruning Complete!")
    print(f"  Total Original Frames : {total_orig_all}")
    print(f"  Total Kept Frames     : {total_kept_all}")
    print(f"  Frames Pruned         : {total_orig_all - total_kept_all} ({pruned_pct:.1f}% reduction)")
    print(f"  Pruned videos saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
