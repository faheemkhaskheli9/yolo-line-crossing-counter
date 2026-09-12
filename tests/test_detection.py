from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conftest import FakeBox, FakeResult, FakeYoloModel  # noqa: E402
from ylc.detection import Detection, YoloDetector  # noqa: E402


def test_detect_returns_detection_above_threshold():
    result = FakeResult([FakeBox([10, 20, 30, 40], cls=0, conf=0.9)])
    model = FakeYoloModel([result])
    detector = YoloDetector(model=model)

    detections = detector.detect(frame="fake-frame")

    assert detections == [Detection(10, 20, 30, 40, 0, "person", 0.9)]


def test_detect_filters_below_confidence_threshold():
    result = FakeResult([FakeBox([0, 0, 1, 1], cls=0, conf=0.1)])
    model = FakeYoloModel([result])
    detector = YoloDetector(model=model, confidence_threshold=0.5)

    assert detector.detect(frame="fake-frame") == []


def test_detect_applies_class_filter_by_name():
    result = FakeResult(
        [
            FakeBox([0, 0, 1, 1], cls=0, conf=0.9),  # person
            FakeBox([0, 0, 1, 1], cls=2, conf=0.9),  # car
        ]
    )
    model = FakeYoloModel([result])
    detector = YoloDetector(model=model, class_filter=["person"])

    detections = detector.detect(frame="fake-frame")

    assert [d.class_name for d in detections] == ["person"]


def test_detect_applies_class_filter_by_id():
    result = FakeResult([FakeBox([0, 0, 1, 1], cls=2, conf=0.9)])
    model = FakeYoloModel([result])
    detector = YoloDetector(model=model, class_filter=[2])

    detections = detector.detect(frame="fake-frame")

    assert [d.class_id for d in detections] == [2]


def test_no_class_filter_keeps_everything():
    result = FakeResult(
        [
            FakeBox([0, 0, 1, 1], cls=0, conf=0.9),
            FakeBox([0, 0, 1, 1], cls=16, conf=0.9),
        ]
    )
    model = FakeYoloModel([result])
    detector = YoloDetector(model=model)

    assert len(detector.detect(frame="fake-frame")) == 2


def test_detect_batch_runs_once_per_frame():
    results = [FakeResult([FakeBox([0, 0, 1, 1], cls=0, conf=0.9)]) for _ in range(3)]
    model = FakeYoloModel(results)
    detector = YoloDetector(model=model)

    batches = detector.detect_batch(["frame1", "frame2", "frame3"])

    assert len(batches) == 3
    assert model.calls == 3


def test_invalid_confidence_threshold_rejected():
    with pytest.raises(ValueError):
        YoloDetector(confidence_threshold=1.5, model=FakeYoloModel([]))


@pytest.mark.integration
def test_real_yolo_detects_people_in_sample_clip(sample_video_path):
    """End-to-end smoke test: real pretrained YOLOv8n against the checked-in
    sample clip. Requires network access on first run to fetch weights;
    skipped automatically if that's unavailable."""
    try:
        detector = YoloDetector(class_filter=["person"], confidence_threshold=0.1)
        detector.model  # trigger weight load/download now, so failures are clear
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"pretrained YOLO weights unavailable: {exc}")

    cap = cv2.VideoCapture(sample_video_path)
    assert cap.isOpened()
    total_person_detections = 0
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            total_person_detections += len(detector.detect(frame))
    finally:
        cap.release()

    assert frame_count > 0
    assert total_person_detections > 0, "expected at least one person detection in the sample clip"
