"""Ensure the YOLO26n-face weights exist locally.

Order of resolution:
  1. bundled/local file (repo checkout or user-provided path)
  2. HuggingFace Hub download -> cached under ~/.cache/huggingface
Returns a filesystem path ready for YOLO(path).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

HF_REPO = "Shubham12864/YOLO26n-face"
HF_FILENAME = "yolo26_widerdataset.pt"
SHA256_PREFIX = "429066d7481ff45b"   # first 16 hex of verified weights


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ensure_yolo_weights(local_path: str | Path | None = None, quiet: bool = False) -> Path:
    """Return a local path to the YOLO26n-face weights, downloading if needed."""
    candidates: list[Path] = []
    if local_path:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"weights_path given but missing: {p}")
        return p

    repo_default = Path(__file__).resolve().parents[1] / "models" / "yolo26n"
    for name in ("yolo26 widerdataset.pt", HF_FILENAME):
        candidates.append(repo_default / name)

    for c in candidates:
        if c.exists() and _sha16(c) == SHA256_PREFIX:
            return c

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "YOLO weights not found locally and huggingface_hub is not installed. "
            "Run `pip install huggingface_hub` or place the .pt next to tier2-ai/models/yolo26n/"
        ) from exc

    downloaded = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
    got = Path(downloaded)
    digest = _sha16(got)
    if digest != SHA256_PREFIX and not quiet:
        print(f"[weights] WARNING: downloaded file hash {digest} != expected {SHA256_PREFIX}")
    elif not quiet:
        print(f"[weights] YOLO26n-face downloaded from hf.co/{HF_REPO} ({got.name})")
    return got
