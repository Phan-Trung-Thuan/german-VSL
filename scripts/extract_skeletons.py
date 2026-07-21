# scripts/extract_skeletons.py
"""
Whole-Body Pose Motion Extraction (Video Retargeting Pre-processing)
====================================================================
1. Scans all MP4 sign videos in `signdict/videos/`.
2. Uses MediaPipe Tasks (PoseLandmarker + HandLandmarker) to track Whole-Body keypoints:
   - Body Pose landmarks (33 keypoints)
   - Left & Right Hand landmarks (21 keypoints each)
3. Renders 2D skeleton stick-figures against a solid black background.
4. Exports output to `signdict/skeletons/`:
   - Skeleton render video: `{name}_skeleton.mp4`
   - Keypoint coordinates: `{name}_keypoints.json`

Requirements:
  pip install mediapipe opencv-python numpy tqdm
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

# Suppress MediaPipe and TensorFlow C++ logger spam
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Dependencies check
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install requirements: pip install mediapipe opencv-python numpy tqdm")
    sys.exit(1)


# Pose & Hand Landmarker task URLs
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Connection pairs for drawing skeletons
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (25, 27), (24, 26), (26, 28) # Legs
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17) # Palm
]


def ensure_model(model_url: str, filename: str) -> Path:
    """Download task model file if not already present."""
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / filename
    if not model_path.exists():
        print(f"Downloading {filename} from Google CDN...")
        urllib.request.urlretrieve(model_url, str(model_path))
        print(f"Downloaded {filename} ({model_path.stat().st_size / 1_000_000:.1f} MB)")
    return model_path


class WholeBodyPoseExtractor:
    """
    Tracks and extracts whole-body keypoints (pose + hands)
    and renders stick figure videos on a black background.
    """

    def __init__(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path = "signdict/skeletons",
    ):
        if input_dir is None:
            pruned = Path("signdict/videos_pruned")
            input_dir = pruned if (pruned.exists() and any(pruned.glob("*.mp4"))) else Path("signdict/videos")
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Download & initialize MediaPipe Tasks
        self.pose_model_path = ensure_model(POSE_MODEL_URL, "pose_landmarker.task")
        self.hand_model_path = ensure_model(HAND_MODEL_URL, "hand_landmarker.task")

        # Pose options
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.pose_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # Hand options
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.hand_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    def process_video(self, video_path: Path) -> tuple[Path, Path]:
        """Process a single video file, extract keypoints, and write skeleton output."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

        out_video_path = self.output_dir / f"{video_path.stem}_skeleton.mp4"
        out_json_path  = self.output_dir / f"{video_path.stem}_keypoints.json"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        frames_keypoints = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect Pose & Hands
            pose_result = self.pose_landmarker.detect(mp_image)
            hand_result = self.hand_landmarker.detect(mp_image)

            # Black background canvas
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw Pose Skeleton
            pose_pts = None
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                landmarks = pose_result.pose_landmarks[0]
                pose_pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

                # Draw pose connections (cyan lines)
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(pose_pts) and end_idx < len(pose_pts):
                        p1, p2 = pose_pts[start_idx], pose_pts[end_idx]
                        cv2.line(canvas, p1, p2, (255, 255, 0), 3)

                # Draw pose keypoints (yellow dots)
                for pt in pose_pts:
                    cv2.circle(canvas, pt, 4, (0, 255, 255), -1)

            # Draw Hand Skeletons
            hands_data = []
            if hand_result.hand_landmarks:
                for h_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    hand_pts = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
                    handedness = (
                        hand_result.handedness[h_idx][0].category_name
                        if hand_result.handedness and h_idx < len(hand_result.handedness)
                        else f"Hand_{h_idx}"
                    )
                    hands_data.append({"label": handedness, "points": self._landmarks_to_list(hand_landmarks)})

                    # Color: Red for Right, Green for Left
                    color = (0, 0, 255) if "Right" in handedness else (0, 255, 0)

                    # Draw hand connections
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(hand_pts) and end_idx < len(hand_pts):
                            cv2.line(canvas, hand_pts[start_idx], hand_pts[end_idx], color, 2)

                    # Draw hand keypoint dots
                    for pt in hand_pts:
                        cv2.circle(canvas, pt, 3, (255, 255, 255), -1)

            out_writer.write(canvas)

            # Store JSON keypoints
            frames_keypoints.append({
                "frame": frame_idx,
                "pose": self._landmarks_to_list(pose_result.pose_landmarks[0]) if (pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0) else None,
                "hands": hands_data
            })
            frame_idx += 1

        cap.release()
        out_writer.release()

        # Save coordinate data
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_name": video_path.name,
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": len(frames_keypoints),
                "frames": frames_keypoints
            }, f, indent=2)

        return out_video_path, out_json_path

    @staticmethod
    def _landmarks_to_list(landmarks) -> list[dict] | None:
        if not landmarks:
            return None
        return [
            {
                "x": round(lm.x, 5),
                "y": round(lm.y, 5),
                "z": round(lm.z, 5),
                "visibility": round(lm.visibility, 5) if hasattr(lm, "visibility") and lm.visibility is not None else 1.0
            }
            for lm in landmarks
        ]

    def run(self):
        """Batch process all MP4 videos in input_dir."""
        video_files = list(self.input_dir.glob("*.mp4"))
        if not video_files:
            print(f"No MP4 video files found in '{self.input_dir.resolve()}'")
            return

        print(f"Starting Whole-Body Skeleton Extraction on {len(video_files)} videos...")
        print(f"Input Directory : {self.input_dir.resolve()}")
        print(f"Output Directory: {self.output_dir.resolve()}\n")

        for video_path in tqdm(video_files, desc="Extracting Skeletons"):
            try:
                out_v, out_j = self.process_video(video_path)
            except Exception as e:
                print(f"\n[Error] Failed processing {video_path.name}: {e}")

        print(f"\nExtraction complete! Processed outputs saved in: {self.output_dir.resolve()}")


if __name__ == "__main__":
    extractor = WholeBodyPoseExtractor()
    extractor.run()
