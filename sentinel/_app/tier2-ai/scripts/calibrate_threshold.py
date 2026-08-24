"""Offline FAR/FRR calibration sweep for the identity verifier.

Embeds every face image in a directory, scores all same-vs-different pairs,
sweeps candidate thresholds and reports FAR/FRR/EER plus the recommended cut
at a target FAR.  Run once after (re-)enrolling to pick MatchVerifier's
threshold with evidence instead of folklore:

    python scripts/calibrate_threshold.py --images data/known_identities
"""
from __future__ import annotations
import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.embedding.face_embedder import FaceEmbedder  # noqa: E402


def collect_embeddings(directory: Path):
    embedder = FaceEmbedder()
    vectors, labels = [], []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skip (unreadable): {path.name}")
            continue
        vector = embedder.embed_image(image)
        if vector is None:
            print(f"  skip (no face found): {path.name}")
            continue
        # label = immediate parent folder when it is not the scan root itself,
        # else filename stem (supports both <name>/x.jpg and <name>_01.jpg layouts)
        label = path.parent.name if path.parent != directory else path.stem
        vectors.append(vector)
        labels.append(label)
    return vectors, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "known_identities")
    parser.add_argument("--target-far", type=float, default=0.01,
                        help="recommended cut is the lowest threshold meeting this FAR (default 1%%)")
    args = parser.parse_args()

    if not args.images.exists():
        sys.exit(f"image directory not found: {args.images}")
    print(f"Embedding faces from {args.images} ...")
    vectors, labels = collect_embeddings(args.images)
    if len(vectors) < 2:
        sys.exit("Need at least 2 usable face images to build genuine/impostor pairs.")

    same, diff = [], []
    for (i, vi), (j, vj) in itertools.combinations(enumerate(vectors), 2):
        score = float(vi @ vj)  # embeddings are L2-normalised -> dot == cosine
        (same if labels[i] == labels[j] else diff).append(score)

    n_gen, n_imp = len(same), len(diff)
    if n_gen == 0:
        sys.exit("No genuine pairs found: give each person >=2 images "
                 "(e.g. data/known_identities/<name>/<file>.jpg).")
    print(f"\npairs: {n_gen} genuine / {n_imp} impostor")

    print(f"\n{'threshold':>9} {'FAR':>8} {'FRR':>8}")
    best_cut, eer, table = None, None, []
    step = 0.01
    t = 0.30
    while t <= 0.95 + 1e-9:
        far = sum(1 for s in diff if s >= t) / n_imp if n_imp else 0.0
        frr = sum(1 for s in same if s < t) / n_gen
        table.append((t, far, frr))
        if eer is None and abs(far - frr) <= 0.02:
            eer = t
        if best_cut is None and far <= args.target_far:
            best_cut = t
        t += step

    for thr, far, frr in table:
        bar = " <-- recommended cut" if best_cut is not None and abs(thr - best_cut) < 1e-9 else ""
        if abs((thr * 100) % 5) < 1e-9 or bar or abs(thr - 0.75) < 1e-9:
            print(f"{thr:9.2f} {far:8.4f} {frr:8.4f}{bar}")

    print(f"\nEER ~ {eer:.2f}" if eer else "\nEER not bracketed in sweep range")
    if best_cut is not None:
        print(f"Recommended threshold at FAR<={args.target_far:.2%}: {best_cut:.2f}")
        print("Wire it in: UnifiedPipeline(..., verifier=MatchVerifier(threshold=%.2f))" % best_cut)
    else:
        print("No threshold meets the target FAR — check enrollment quality/separation.")


if __name__ == "__main__":
    main()
