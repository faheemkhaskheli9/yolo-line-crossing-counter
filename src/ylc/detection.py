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
    """A single detected object in one frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float

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
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")
        self.weights = str(weights)
        self.class_filter = set(class_filter) if class_filter else None
        self.confidence_threshold = confidence_threshold
        self._model = model

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

    def _parse_results(self, results: Sequence[Any]) -> list[Detection]:
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
                detections.append(Detection(x1, y1, x2, y2, class_id, class_name, confidence))
        return detections
