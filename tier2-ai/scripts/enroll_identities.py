"""Create/update the persistent FAISS index from data/known_identities images."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.embedding.face_embedder import FaceEmbedder
from src.identity_search.faiss_index import IdentitySearch
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "known_identities")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "identity_index.faiss")
    args = parser.parse_args()
    embedder, searcher = FaceEmbedder(), IdentitySearch()
    enrolled = 0
    for path in sorted(args.input.glob("*")):
        image = cv2.imread(str(path))
        if image is None: continue
        embedding = embedder.embed_image(image)
        if embedding is not None:
            searcher.add_identity(path.stem, embedding, path.stem); enrolled += 1
    searcher.save(args.output)
    print(f"Enrolled {enrolled} identity image(s) into {args.output}")
if __name__ == "__main__": main()
