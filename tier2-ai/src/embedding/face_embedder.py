"""ArcFace embeddings with single-pass detection (recognition-only at runtime).

Runtime crops skip SCRFD entirely: the YOLO/ByteTrack bounding box has already
localised the face, so the region is margin-expanded, resized straight to the
recognizer's input size, and embedded.  Enrollment images receive exactly one
detector pass through the identical margin + resize preprocessing, keeping
gallery and query features comparable while roughly halving per-face cost.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np


class FaceEmbedder:

    def __init__(self, ctx_id: int = -1, embedding_dim: int = 512, margin_ratio: float = 0.25):
        self.ctx_id, self.embedding_dim, self.margin_ratio = ctx_id, embedding_dim, margin_ratio
        self.model = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            try:
                import insightface
            except ImportError as exc:
                raise RuntimeError("Face embedding requires insightface and onnxruntime.") from exc
            model = insightface.app.FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
            model.prepare(ctx_id=self.ctx_id)
            self.model = model

    @property
    def _recognizer(self):
        self._load()
        return self.model.models["recognition"]

    def _input_size(self) -> tuple[int, int]:
        shape = self._recognizer.input_shape
        return int(shape[2]), int(shape[3])

    def _expand_bbox(self, bbox, frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        mx, my = (x2 - x1) * self.margin_ratio / 2.0, (y2 - y1) * self.margin_ratio / 2.0
        xi1, yi1 = max(0, int(round(x1 - mx))), max(0, int(round(y1 - my)))
        xi2, yi2 = min(frame_width, int(round(x2 + mx))), min(frame_height, int(round(y2 + my)))
        if xi2 - xi1 < 2 or yi2 - yi1 < 2:
            return None
        return xi1, yi1, xi2, yi2

    def _aligned_input(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        region = self._expand_bbox(bbox, frame.shape[1], frame.shape[0])
        if region is None:
            return None
        crop = frame[region[1]:region[3], region[0]:region[2]]
        height, width = self._input_size()
        return cv2.resize(crop, (width, height))

    @staticmethod
    def _normalise(feature) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def embed_from_bbox(self, frame: np.ndarray | None, bbox) -> np.ndarray | None:
        """Runtime hot path: externally detected bbox -> one recognition pass."""
        if frame is None or getattr(frame, "size", 0) == 0 or bbox is None:
            return None
        aligned = self._aligned_input(frame, bbox)
        if aligned is None:
            return None
        return self._normalise(self._recognizer.get_feat(aligned))

    def embed_crop(self, face_crop: np.ndarray | None) -> np.ndarray | None:
        """Recognition-only embedding of an already-cropped face image."""
        if face_crop is None or getattr(face_crop, "size", 0) == 0:
            return None
        height, width = self._input_size()
        aligned = cv2.resize(face_crop, (width, height))
        return self._normalise(self._recognizer.get_feat(aligned))

    def embed_image(self, image: np.ndarray | None) -> np.ndarray | None:
        """Offline enrollment: exactly one SCRFD pass, then the same preprocessing."""
        if image is None or getattr(image, "size", 0) == 0:
            return None
        self._load()
        bboxes, _ = self.model.models["detection"].detect(image, max_num=1)
        if bboxes is None or len(bboxes) == 0:
            return None
        return self.embed_from_bbox(image, bboxes[0][:4])

    def get_embedding(self, face_crop_or_image: np.ndarray | None) -> np.ndarray | None:
        """Backward-compatible entry point.

        Full images (enrollment) go through single-pass detection; tight crops
        are recognised directly.  New code should call embed_from_bbox,
        embed_crop or embed_image explicitly.
        """
        if face_crop_or_image is None or getattr(face_crop_or_image, "size", 0) == 0:
            return None
        self._load()
        bboxes, _ = self.model.models["detection"].detect(face_crop_or_image, max_num=1)
        if bboxes is None or len(bboxes) == 0:
            return None
        return self.embed_from_bbox(face_crop_or_image, bboxes[0][:4])

    def embed(self, image: np.ndarray | None) -> list[float]:
        result = self.embed_image(image)
        return result.tolist() if result is not None else [0.0] * self.embedding_dim
