"""YOLO object detection wrapper.

Wraps a pretrained Ultralytics YOLO model to run per-frame inference and
return a normalized ``Detection`` list (bounding box, class, confidence).
The class filter is configurable (by class name or numeric id) rather than
hardcoded, so the same detector serves "people crossing a line" and
"vehicles crossing a line" use cases from config alone.

The real model (``ultralytics.YOLO``) is only imported when actually needed
— either on first real inference, or never, if a test injects its own
``model=`` — so importing this module never requires a network fetch or a
GPU, and unit tests can inject a lightweight fake model instead of loading
real weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame.

    ``track_id`` is ``None`` for a plain (untracked) detection from
    :meth:`YoloDetector.detect`, and the persistent id ByteTrack assigned
    when the detection instead came from :meth:`YoloDetector.track`.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DetectionModel(Protocol):
    """Minimal interface a wrapped model must satisfy.

    This matches the callable interface of ``ultralytics.YOLO`` instances
    closely enough that either a real model or a test fake can be injected.
    """

    names: dict[int, str]

    def __call__(self, frame: Any, verbose: bool = False) -> Sequence[Any]: ...
    def track(
        self, frame: Any, persist: bool = True, tracker: str = "bytetrack.yaml", verbose: bool = False
    ) -> Sequence[Any]: ...


def _load_ultralytics_model(weights: str) -> DetectionModel:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "ultralytics is required for real YOLO inference (see requirements.txt); "
            "for tests, inject a fake `model=` into YoloDetector instead."
        ) from exc
    return YOLO(weights)


class YoloDetector:
    """Runs YOLO detection on frames, filtered by class and confidence."""

    def __init__(
        self,
        weights: str | Path = "yolov8n.pt",
        class_filter: Iterable[str | int] | None = None,
        confidence_threshold: float = 0.25,
        model: DetectionModel | None = None,
        track_buffer: int = 30,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")
        self.weights = str(weights)
        self.class_filter = set(class_filter) if class_filter else None
        self.confidence_threshold = confidence_threshold
        self._model = model
        # Tracking state, kept for debugging (AC: "active/lost tracks
        # inspectable"): frames-since-last-seen per track id we've ever
        # matched via `track()`. ByteTrack itself owns match/lost logic
        # internally; this is just our own visible mirror of it. Set once
        # at construction (not per `track()` call) so a later call omitting
        # it can't silently reset it back to the default.
        self._frames_since_seen: dict[int, int] = {}
        self._lost_after_frames = track_buffer

    @property
    def model(self) -> DetectionModel:
        if self._model is None:
            self._model = _load_ultralytics_model(self.weights)
        return self._model

    def _passes_filter(self, class_id: int, class_name: str) -> bool:
        if self.class_filter is None:
            return True
        return class_name in self.class_filter or class_id in self.class_filter

    def detect(self, frame: Any) -> list[Detection]:
        """Run detection on a single frame (e.g. an HxWx3 numpy array)."""
        results = self.model(frame, verbose=False)
        return self._parse_results(results)

    def detect_batch(self, frames: Sequence[Any]) -> list[list[Detection]]:
        """Run detection on a batch/sequence of frames, one result list per frame."""
        return [self.detect(frame) for frame in frames]

    def track(self, frame: Any, tracker: str = "bytetrack.yaml") -> list[Detection]:
        """Run detection *and* ByteTrack tracking on a single frame.

        ``persist=True`` tells the model to carry tracker state across
        calls on the same instance (one video's frames, in order) instead
        of resetting on every call. A box with no assigned id yet (a track
        not confirmed) gets ``track_id=None`` rather than a made-up id.
        """

        results = self.model.track(frame, persist=True, tracker=tracker, verbose=False)
        detections = self._parse_results(results, with_track_id=True)
        self._update_track_state(detections)
        return detections

    def _update_track_state(self, detections: list[Detection]) -> None:
        seen_ids = {d.track_id for d in detections if d.track_id is not None}
        for track_id in seen_ids:
            self._frames_since_seen[track_id] = 0
        for track_id in list(self._frames_since_seen):
            if track_id in seen_ids:
                continue
            self._frames_since_seen[track_id] += 1
            if self._frames_since_seen[track_id] > self._lost_after_frames:
                del self._frames_since_seen[track_id]

    @property
    def active_track_ids(self) -> set[int]:
        """Track ids seen in the most recent :meth:`track` call."""
        return {tid for tid, age in self._frames_since_seen.items() if age == 0}

    @property
    def lost_track_ids(self) -> set[int]:
        """Track ids not seen this frame but still within ``track_buffer``."""
        return {tid for tid, age in self._frames_since_seen.items() if age > 0}

    def _parse_results(
        self, results: Sequence[Any], with_track_id: bool = False
    ) -> list[Detection]:
        detections: list[Detection] = []
        names = self.model.names
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue
                class_id = int(box.cls[0])
                class_name = names.get(class_id, str(class_id))
                if not self._passes_filter(class_id, class_name):
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                # A box's track hasn't been confirmed yet -> box.id is None;
                # emit it untracked rather than inventing an id.
                box_id = getattr(box, "id", None) if with_track_id else None
                track_id = int(box_id[0]) if box_id is not None else None
                detections.append(
                    Detection(x1, y1, x2, y2, class_id, class_name, confidence, track_id)
                )
        return detections
