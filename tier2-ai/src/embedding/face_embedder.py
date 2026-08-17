"""Pretrained ArcFace embeddings supplied by InsightFace's buffalo_l pack."""
from __future__ import annotations
import numpy as np
class FaceEmbedder:
    def __init__(self, ctx_id: int = -1, embedding_dim: int = 512): self.ctx_id, self.embedding_dim, self.model = ctx_id, embedding_dim, None
    def _load(self):
        if self.model is not None: return
        try: import insightface
        except ImportError as exc: raise RuntimeError("Face embedding requires insightface and onnxruntime.") from exc
        self.model = insightface.app.FaceAnalysis(name="buffalo_l"); self.model.prepare(ctx_id=self.ctx_id)
    def get_embedding(self, face_crop: np.ndarray | None) -> np.ndarray | None:
        if face_crop is None or face_crop.size == 0: return None
        self._load(); faces = self.model.get(face_crop)
        return np.asarray(faces[0].embedding, dtype=np.float32) if faces else None
    def embed(self, image):
        result = self.get_embedding(image)
        return result.tolist() if result is not None else [0.0] * self.embedding_dim
