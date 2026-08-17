"""The frozen contracts between Sentinel Edge Track 1, 2, and 3."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping
import numpy as np

@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    timestamp: str
    frame: np.ndarray
    metadata: dict[str, Any]
    @classmethod
    def from_mapping(cls, packet: Mapping[str, Any]) -> "FramePacket":
        required = {"camera_id", "timestamp", "frame", "metadata"}
        missing = required.difference(packet)
        if missing: raise ValueError(f"FramePacket missing required fields: {sorted(missing)}")
        frame = packet["frame"]
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3: raise ValueError("FramePacket.frame must be a BGR numpy array shaped (H, W, 3)")
        if not isinstance(packet["metadata"], dict): raise ValueError("FramePacket.metadata must be a dict")
        return cls(str(packet["camera_id"]), str(packet["timestamp"]), frame, packet["metadata"])

@dataclass(frozen=True)
class UnifiedEvent:
    track_id: str
    identity_name: str | None
    identity_id: str | None
    match_score: float
    verified: bool
    current_camera: str
    timestamp: str
    face_crop_url: str | None
    next_camera: str | None
    eta_seconds: int | None
    transition_probability: float | None
    def to_dict(self) -> dict[str, Any]: return asdict(self)
