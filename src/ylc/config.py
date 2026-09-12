"""Config-driven detection settings, loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: str = "yolov8n.pt"
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    class_filter: list[str] | None = None
    video_path: str = "assets/sample_pedestrians_clip.mp4"
    tracker: str = "bytetrack.yaml"  # Ultralytics tracker config name, e.g. "botsort.yaml"
    track_buffer: int = Field(
        default=30, ge=0, description="frames a lost track is kept before being dropped"
    )


def load_detection_config(path: str | Path) -> DetectionConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"detection config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return DetectionConfig.model_validate(raw)
