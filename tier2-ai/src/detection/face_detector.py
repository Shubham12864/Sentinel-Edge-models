"""YOLO face detection and per-camera ByteTrack integration."""
from __future__ import annotations
from pathlib import Path
from typing import Any

class FaceDetector:

    def __init__(self, weights_path: str | Path | None = None, confidence: float = 0.25, model_path: str | Path | None = None, device: str | int | None = None):
        default = Path(__file__).resolve().parents[2] / "models" / "yolo26n" / "yolo26 widerdataset.pt"
        if weights_path is not None and model_path is not None: raise ValueError("Specify only weights_path or model_path")
        selected = weights_path if weights_path is not None else model_path
        self.weights_path, self.confidence, self.device = Path(selected) if selected else default, confidence, device
        self.model = None
    def _load(self) -> None:
        if self.model is not None: return
        if not self.weights_path.exists(): raise FileNotFoundError(f"Face model weights not found: {self.weights_path}")
        try: from ultralytics import YOLO
        except ImportError as exc: raise RuntimeError("Face detection requires 'ultralytics'. Install requirements.txt.") from exc
        self.model = YOLO(str(self.weights_path))
    def detect_and_track(self, frame: Any):
        self._load()
        kwargs = {"tracker": "bytetrack.yaml", "persist": True, "conf": self.confidence, "verbose": False}
        if self.device is not None: kwargs["device"] = self.device
        results = self.model.track(frame, **kwargs)
        output = []
        for box in results[0].boxes:
            output.append({"track_id": int(box.id[0].item()) if box.id is not None else None, "bbox": [float(v) for v in box.xyxy[0].tolist()], "confidence": float(box.conf[0].item())})
        return output
    def detect(self, frame: Any): return self.detect_and_track(frame)

    def close(self) -> None:
        """Release this camera's tracker/model session after it becomes inactive."""
        self.model = None

PersonDetector = FaceDetector
