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


# --- track() / ByteTrack integration (offline, fake model) -----------------


def test_track_attaches_persistent_track_id():
    result = FakeResult([FakeBox([10, 20, 30, 40], cls=0, conf=0.9, track_id=7)])
    model = FakeYoloModel([result])
    detector = YoloDetector(model=model)

    detections = detector.track(frame="frame-1")

    assert detections == [Detection(10, 20, 30, 40, 0, "person", 0.9, track_id=7)]


def test_track_id_persists_across_consecutive_frames():
    frame1 = FakeResult([FakeBox([0, 0, 10, 10], cls=0, conf=0.9, track_id=1)])
    frame2 = FakeResult([FakeBox([1, 1, 11, 11], cls=0, conf=0.9, track_id=1)])
    model = FakeYoloModel([frame1, frame2])
    detector = YoloDetector(model=model)

    ids = [d.track_id for d in detector.track(frame="f1")]
    ids += [d.track_id for d in detector.track(frame="f2")]

    assert ids == [1, 1]


def test_box_with_no_confirmed_track_id_is_untracked_not_invented():
    result = FakeResult([FakeBox([0, 0, 10, 10], cls=0, conf=0.9, track_id=None)])
    model = FakeYoloModel([result])
    detector = YoloDetector(model=model)

    detections = detector.track(frame="f1")

    assert detections[0].track_id is None


def test_active_track_ids_reflects_most_recent_frame():
    frame1 = FakeResult([FakeBox([0, 0, 10, 10], cls=0, conf=0.9, track_id=1)])
    frame2 = FakeResult([FakeBox([0, 0, 10, 10], cls=0, conf=0.9, track_id=2)])
    model = FakeYoloModel([frame1, frame2])
    detector = YoloDetector(model=model)

    detector.track(frame="f1")
    assert detector.active_track_ids == {1}

    detector.track(frame="f2")
    assert detector.active_track_ids == {2}
    assert detector.lost_track_ids == {1}  # still within the buffer


def test_lost_track_dropped_after_track_buffer_frames_without_a_match():
    frame_with_track = FakeResult([FakeBox([0, 0, 10, 10], cls=0, conf=0.9, track_id=1)])
    frame_empty = FakeResult([])
    model = FakeYoloModel([frame_with_track, frame_empty, frame_empty, frame_empty])
    detector = YoloDetector(model=model, track_buffer=2)

    detector.track(frame="f1")
    assert detector.active_track_ids == {1}

    detector.track(frame="f2")  # age 1 -- still within buffer
    assert 1 in detector.lost_track_ids

    detector.track(frame="f3")  # age 2 -- still within buffer (>2 required to drop)
    assert 1 in detector.lost_track_ids

    detector.track(frame="f4")  # age 3 -- exceeds buffer, dropped
    assert 1 not in detector.lost_track_ids
    assert 1 not in detector.active_track_ids


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


@pytest.mark.integration
def test_real_bytetrack_keeps_one_id_across_frames_in_sample_clip(sample_video_path):
    """AC: "a short sample video demonstrates the same object keeping one
    track ID across multiple frames." Requires network on first run to
    fetch weights, and the `lapx` package for ByteTrack's assignment step;
    skipped automatically if either is unavailable."""
    try:
        detector = YoloDetector(class_filter=["person"], confidence_threshold=0.1)
        detector.model
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"pretrained YOLO weights unavailable: {exc}")

    cap = cv2.VideoCapture(sample_video_path)
    assert cap.isOpened()
    id_sequence: list[set[int]] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                tracked = detector.track(frame)
            except Exception as exc:  # pragma: no cover - missing `lapx` etc.
                pytest.skip(f"ByteTrack unavailable: {exc}")
            id_sequence.append({d.track_id for d in tracked if d.track_id is not None})
    finally:
        cap.release()

    seen_more_than_once = {
        track_id
        for track_id in set().union(*id_sequence)
        if sum(track_id in frame_ids for frame_ids in id_sequence) > 1
    }
    assert seen_more_than_once, "expected at least one track ID to persist across multiple frames"
