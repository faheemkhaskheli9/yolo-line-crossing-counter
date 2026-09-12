"""Shared test fixtures: a fake YOLO-like model so detection logic can be
unit-tested without downloading real weights or running real inference."""

from __future__ import annotations

import numpy as np
import pytest


class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.cls = [float(cls)]
        self.conf = [float(conf)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYoloModel:
    """Stands in for ``ultralytics.YOLO``: callable, has ``.names``."""

    names = {0: "person", 1: "bicycle", 2: "car", 16: "dog"}

    def __init__(self, results_by_call):
        self._results_by_call = list(results_by_call)
        self.calls = 0

    def __call__(self, frame, verbose: bool = False):
        result = self._results_by_call[min(self.calls, len(self._results_by_call) - 1)]
        self.calls += 1
        return [result]


@pytest.fixture
def sample_video_path() -> str:
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "assets" / "sample_pedestrians_clip.mp4"
    assert path.is_file(), f"missing test fixture video: {path}"
    return str(path)
