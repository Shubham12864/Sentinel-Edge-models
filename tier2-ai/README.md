# Sentinel Edge — Track 2 AI Vision + Prediction

This repository receives Track 1 `FramePacket` objects and produces Track 3 `UnifiedEvent` objects. It intentionally contains no camera-ingestion, WebSocket, or frontend code.

## Structure

- `src/` contains the modular pipeline implementation.
- `models/` stores the supplied YOLO26n face weights.
- `config/` contains camera and zone configuration files.
- `data/` holds identity reference data and project assets.
- `tests/` contains validation for the quality gate, detection, tracking, identity search, and end-to-end pipeline flow.
- `sample_data/` contains example frames and identity samples for local validation.
- `notebooks/` is for experiments and debugging.

## Detector setup

Place the detector weights under:

- `models/yolo26n/yolo26 widerdataset.pt`

The production entry point is `UnifiedPipeline.process_frame_packet(packet)`. It filters weak face crops, detects and tracks faces, uses cached ArcFace/FAISS identity results per camera-track, saves crops, and returns only the fixed UnifiedEvent fields.

Tracking is handled directly by Ultralytics' built-in ByteTrack through `YOLO.track`; there is no separate custom tracker implementation.

For a complete local Tier 2 startup (without opening any camera), construct it once with:

```python
from src.pipeline import UnifiedPipeline

pipeline = UnifiedPipeline.create_default(
    identity_index_path="data/identity_index.faiss",
)
```

Tier 1 supplies `FramePacket` objects to `pipeline.process_frame_packet(packet)`. For a local, bounded latest-frame bridge that deliberately drops stale frames instead of building latency, use `src.runtime.frame_runtime.Tier2Runtime` with an event-sink callback supplied by the Tier 2 → Tier 3 integration layer. This repository intentionally does not implement Tier 1 camera drivers, Tier 3's HTTP receiver, WebSockets, or frontend.

To enroll reference images once, place them in `data/known_identities/` and run `python scripts/enroll_identities.py`. The resulting FAISS index is reusable across pipeline restarts.

## Getting started

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Add your YOLO26n weights under `models/yolo26n/`.
4. Configure the camera and zone profiles in `config/`.

## Build and test

Use:

```bash
make
pytest -q
```
