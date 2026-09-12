"""CLI: run YOLO detection end-to-end over a video clip.

    python scripts/run_detection.py --config configs/detection.yaml

Prints per-frame detection counts. Serves as the Phase 1 smoke test that the
detection module runs on a real video without errors; the checked-in
``assets/sample_pedestrians_clip.mp4`` is a short (3s, downscaled) trim of a
public pedestrian-detection sample clip (intel-iot-devkit/sample-videos,
MIT-licensed) kept small enough to commit — point ``--video`` at any other
video for a real run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from ylc.config import load_detection_config  # noqa: E402
from ylc.detection import YoloDetector  # noqa: E402


def iter_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run YOLO detection over a video clip")
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--video", type=str, default=None, help="Override config's video_path")
    args = parser.parse_args(argv)

    config = load_detection_config(args.config)
    video_path = args.video or config.video_path

    detector = YoloDetector(
        weights=config.weights,
        class_filter=config.class_filter,
        confidence_threshold=config.confidence_threshold,
    )

    total_frames = 0
    total_detections = 0
    for frame in iter_frames(video_path):
        detections = detector.detect(frame)
        total_frames += 1
        total_detections += len(detections)
        if detections:
            print(f"frame {total_frames}: {len(detections)} detection(s)")

    print(f"processed {total_frames} frame(s), {total_detections} detection(s) total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
