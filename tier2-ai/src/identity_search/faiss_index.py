"""Persistent cosine-similarity identity lookup backed by FAISS."""
from __future__ import annotations
import json
import os
from pathlib import Path
from threading import RLock
import tempfile
import numpy as np

class IdentitySearch:
    
    def __init__(self, dim: int = 512):
        self.dim, self.next_id, self.id_to_name, self.index = dim, 0, {}, None
        self._lock = RLock()
    def _ensure_index(self):
        if self.index is None:
            try: import faiss
            except ImportError as exc: raise RuntimeError("Identity search requires faiss-cpu.") from exc
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
    def _normalise(self, embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim: raise ValueError(f"Expected {self.dim}-D embedding, got {vector.shape[0]}")
        norm = np.linalg.norm(vector)
        if norm == 0: raise ValueError("Embedding must not be a zero vector")
        return vector / norm
    def add_identity(self, name: str, embedding, identity_id: str | None = None) -> str:
        with self._lock:
            self._ensure_index(); vector = self._normalise(embedding); pid = self.next_id; self.next_id += 1
            self.index.add_with_ids(vector.reshape(1, -1), np.array([pid], dtype=np.int64)); self.id_to_name[pid] = {"name": name, "identity_id": identity_id or str(pid)}
            return self.id_to_name[pid]["identity_id"]
    def search(self, embedding):
        with self._lock:
            self._ensure_index()
            if self.index.ntotal == 0: return None, None, 0.0
            scores, ids = self.index.search(self._normalise(embedding).reshape(1, -1), 1); pid = int(ids[0][0])
            if pid == -1: return None, None, 0.0
            item = self.id_to_name[pid]; return item["name"], item["identity_id"], float(scores[0][0])
    def save(self, path: str | Path) -> None:
        with self._lock:
            self._ensure_index(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            import faiss
            metadata_path = path.with_suffix(path.suffix + ".json")
            with tempfile.TemporaryDirectory(dir=path.parent) as directory:
                index_tmp = Path(directory) / path.name
                metadata_tmp = Path(directory) / metadata_path.name
                faiss.write_index(self.index, str(index_tmp))
                metadata_tmp.write_text(json.dumps({"next_id": self.next_id, "identities": self.id_to_name}), encoding="utf-8")
                os.replace(index_tmp, path)
                os.replace(metadata_tmp, metadata_path)
    def load(self, path: str | Path) -> bool:
        with self._lock:
            path = Path(path); metadata_path = path.with_suffix(path.suffix + ".json")
            if not path.exists(): return False
            if not metadata_path.exists(): raise FileNotFoundError(f"FAISS metadata missing: {metadata_path}")
            import faiss
            self.index = faiss.read_index(str(path)); data = json.loads(metadata_path.read_text(encoding="utf-8")); self.next_id = int(data["next_id"]); self.id_to_name = {int(k): v for k, v in data["identities"].items()}; return True
