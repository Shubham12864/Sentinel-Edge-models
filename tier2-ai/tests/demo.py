import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import UnifiedPipeline


def ensure_identity_index() -> Path:
    index_path = ROOT / "data" / "identity_index.faiss"
    known_dir = ROOT / "data" / "known_identities"

    if index_path.exists():
        return index_path

    if not known_dir.exists():
        raise FileNotFoundError(
            f"No identity directory found at {known_dir}. "
            "Add reference face images first or create the folder."
        )

    print("FAISS index not found. Auto-enrolling known identities...")
    from src.embedding.face_embedder import FaceEmbedder
    from src.identity_search.faiss_index import IdentitySearch

    embedder = FaceEmbedder()
    searcher = IdentitySearch()

    enrolled = 0
    for image_path in sorted(known_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        embedding = embedder.embed_image(frame)
        if embedding is None:
            continue

        searcher.add_identity(image_path.stem, embedding, image_path.stem)
        enrolled += 1

    if enrolled == 0:
        print("No valid identity images were enrolled. Add face images to data/known_identities/.")
        return index_path

    searcher.save(index_path)
    print(f"Enrolled {enrolled} identity image(s) into {index_path}")
    return index_path


def draw_detections(frame, detections):
    for det in detections:
        bbox = det.get("bbox")
        track_id = det.get("track_id")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"track {track_id}" if track_id is not None else "face"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def main():
    index_path = ensure_identity_index()

    pipeline = UnifiedPipeline.create_default(
        weights_path=ROOT / "models" / "yolo26n" / "yolo26 widerdataset.pt",
        identity_index_path=index_path,
        crop_dir=ROOT / "data" / "face_crops",
    )

    camera_id = "LAPTOP_CAM"
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions or try another index.")

    print("Running live test. Press ESC to quit.")
    last_health_print = 0.0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        frame_index += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        detector = pipeline._detector_for(camera_id)
        raw_detections = detector.detect_and_track(frame)
        draw_detections(frame, raw_detections)

        packet = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "frame": frame,
            "metadata": {
                "source": "laptop_camera",
                "frame_id": frame_index
            }
        }

        events = pipeline.process_frame_packet(packet)

        if events:
            latest = events[-1]
            cv2.putText(frame, f"verified={latest.get('verified')}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if latest.get("identity_name"):
                cv2.putText(frame, f"name={latest['identity_name']}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            print("EVENT:", latest)

        if time.time() - last_health_print > 5:
            print("HEALTH:", pipeline.health())
            last_health_print = time.time()

        cv2.imshow("Tier 2 Live Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Final health:", pipeline.health())


if __name__ == "__main__":
    main()