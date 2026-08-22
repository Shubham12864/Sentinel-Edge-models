# Sentinel Edge Model

This repository is the Tier 2 AI model and demo code for a face-recognition surveillance pipeline.

It is not a full end-user web app. It is a model architecture and working local demo that shows how the system is designed to work in real deployment.

## What this repository is

This project takes a camera frame, detects faces, tracks them, extracts embeddings, compares them with known identities, and emits a result event.

The flow is:

- Tier 1: camera or source provides frames
- Tier 2: this model pipeline processes frames
- Tier 3: downstream app or dashboard consumes events

This repo focuses on Tier 2.

## Main idea

The model works like this:

1. A frame comes in from a camera
2. YOLO detects faces
3. ByteTrack keeps track of each face over time
4. A face crop is extracted
5. ArcFace creates a 512-dimensional embedding
6. FAISS searches the known identity database
7. The system decides whether the person is verified
8. An event is returned with identity, score, camera, and timestamp

## Folder structure

```text
Sentinel Edge models/
├── yolo26n.pt
├── README.md
├── .gitignore
├── tier2-ai/
│   ├── README.md
│   ├── requirements.txt
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── notebooks/

# Sentinel Edge

Sentinel Edge is a local, end-to-end edge vision system for camera ingestion, face detection and tracking, identity matching, trajectory prediction, and operator monitoring.

The repository combines three runtime tiers behind one launcher:

```text
Camera or video source
        |
Tier 1: CameraSource -> FramePacket
        |
Tier 2: YOLO + ByteTrack + ArcFace + FAISS -> UnifiedEvent
        |
Tier 3: Command HQ API + WebSocket + operator console
```

## Features

- OpenCV camera, RTSP, device-index, and recorded-file sources
- Reconnect handling and FPS-bounded capture
- YOLO face detection with Ultralytics ByteTrack tracking
- ArcFace embeddings and FAISS identity search
- Quality filtering, verification, saved face crops, and Markov next-camera prediction
- Persistent identity gallery with multi-image enrollment and removal
- Camera registry with explicit Connect and Disconnect controls
- FastAPI Command HQ with live MJPEG feeds, event history, WebSocket updates, and a browser console
- Runtime tests for ingestion, pipeline hardening, limits, enrollment, and server behavior

## Repository layout

```text
run_sentinel.py       Start Tier 1 + Tier 2 + Tier 3 together
tier1-ingest/         Camera sources, registry, and ingestion manager
tier2-ai/              AI pipeline, models, identity data, scripts, and tests
tier3-hq/              FastAPI server, persistent gallery, and operator console
runs/                 Local recorded demo footage and inference output
```

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment from the repository root, then install the Tier 2 dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r tier2-ai\requirements.txt
```

The AI pipeline expects YOLO weights at:

```text
tier2-ai/models/yolo26n/yolo26 widerdataset.pt
```

The launcher can start without model loading by using `--no-ai`.

## Start the system

From the repository root:

```powershell
python run_sentinel.py
```

Open http://localhost:8000 for Command HQ. Cameras listed in [tier1-ingest/cameras.yaml](tier1-ingest/cameras.yaml) are loaded into the registry but are deliberately not connected automatically. Connect them from the console.

Useful launcher options:

```text
--host HOST             Bind address (default: 127.0.0.1)
--port PORT             HTTP port (default: 8000)
--file PATH             Add and connect one recorded-file camera
--camera-id ID          ID for --file (default: CAM_99)
--fps FPS               Capture rate for --file (default: 4)
--no-ai                 Run ingestion and Command HQ without loading AI models
```

For example:

```powershell
python run_sentinel.py --file runs\detect\predict-4\1.avi --camera-id CAM_99 --fps 4
```

## Identity enrollment

For command-line enrollment, place one or more images per identity in:

```text
tier2-ai/data/known_identities/<identity-name>/
```

Then run:

```powershell
python tier2-ai\scripts\enroll_identities.py
```

When Command HQ is running, identities can also be enrolled and removed from the console. The Tier 3 gallery is stored in `tier3-hq/data/gallery.json` and its exported FAISS index is stored in `tier3-hq/data/identity_index.faiss`.

## Development and tests

Run the automated tests from each tier directory so local imports resolve correctly:

```powershell
pytest -q tier1-ingest
pytest -q tier2-ai\tests
pytest -q tier3-hq
```

The interactive webcam demo is separate from the automated tests:

```powershell
python tier2-ai\tests\demo.py
```

The primary Tier 2 API is `UnifiedPipeline.process_frame_packet(packet)`. The bounded `Tier2Runtime` bridge consumes the resulting packets and forwards events to Command HQ.

## HTTP and realtime surface

Command HQ provides:

- `/` - operator console
- `/api/stats` - runtime statistics
- `/api/events` - recent events
- `/api/cameras` - camera registry and status
- `/api/feed/{camera_id}` - MJPEG camera feed
- `/api/identities` - identity gallery
- `/ws/events` - live event WebSocket

Camera and identity mutations are exposed through the corresponding `/api/cameras/*` and `/api/identities/*` endpoints.

## Data and privacy

Identity vectors, uploaded enrollment images, face crops, camera footage, and inference output are local operational data. Review [`.gitignore`](.gitignore) before adding or publishing generated biometric artifacts.

```bash
cd tier2-ai
python scripts/enroll_identities.py
```

This builds the FAISS identity index from the folder images.

## Real pipeline flow

```text
Camera / Source
    ↓
Tier 1: FramePacket creation
    ↓
Tier 2: Face detection + tracking + embedding + FAISS match
    ↓
Tier 3: Event receiver / alert dashboard / downstream app
```

## Main project details

The model code is under:

- [tier2-ai/src](tier2-ai/src)
- [tier2-ai/scripts](tier2-ai/scripts)
- [tier2-ai/tests](tier2-ai/tests)

The main processing entry point is:

- `UnifiedPipeline.process_frame_packet(...)`

This is the place where a frame becomes an AI event.

## Requirements

The project dependencies are listed in:

- [tier2-ai/requirements.txt](tier2-ai/requirements.txt)

Install them with:

```bash
cd tier2-ai
pip install -r requirements.txt
```

## Model files

The YOLO weights are expected in:

```text
tier2-ai/models/yolo26n/yolo26 widerdataset.pt
```

## Practical use

This is suitable for:

- local testing
- model validation
- edge AI experiments
- future cloud deployment
- building upstream and downstream integrations

It is not a complete final product by itself.

## License

This project is shared for learning and model experimentation. Add your own license if you plan to publish it publicly.

