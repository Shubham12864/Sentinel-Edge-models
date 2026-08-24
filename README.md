# 🛡 Sentinel Edge — Multi-Camera AI Surveillance Grid

Sentinel-Edge is an edge-native, multi-camera surveillance pipeline. It detects and tracks people across camera feeds, verifies identity against a private face database, predicts the next likely camera with an ETA — and presents everything in a live operator console.

```
SEE  →  IDENTIFY  →  PREDICT  →  ALERT
detect every frame   ArcFace + FAISS   Markov graph    live operator
                     512-D identity    next camera     console + events
                                       + ETA
```

> **Track architecture (frozen contracts)**
> `Tier 1` camera gateway → `FramePacket` → `Tier 2` AI vision & prediction → `UnifiedEvent` → `Tier 3` Command HQ
> Each tier is independently testable; the only coupling is the two data contracts.

---

## ✨ Highlights

| | |
|---|---|
| 🎥 **Any source** | RTSP / ONVIF IP cams, USB webcams, recorded footage — one interface |
| 🔌 **Self-healing ingest** | auto-reconnect, hot-plug, bounded FPS, latest-frame processing (no backlog ever builds) |
| 🎯 **Custom detector** | YOLO26n fine-tuned on WIDER FACE — among the first YOLO26-face models ([weights](https://huggingface.co/Shubham12864/YOLO26n-face)) |
| 🧠 **Recognition that survives the real world** | landmark-aligned ArcFace embeddings, FAISS search, hysteresis verification |
| 🔮 **Trajectory prediction** | online Markov transition graph learns *your* site: next camera + ETA + probability, persisted across restarts |
| 🖱 **Zero-code operations** | add cameras, connect/disconnect, enroll identities by drag-and-drop — all from the browser |
| 🔐 **Adversarially hardened** | path-traversal, forged-timestamp TTL-wipe, thread-bomb and gallery-poisoning regressions are tested |
| ⚡ **Edge-first privacy** | video never leaves the node; only structured events flow to the console |

---

## 🚀 Quick Start

```bash
# 1 · clone
git clone https://github.com/Shubham12864/Sentinel-Edge-models.git
cd Sentinel-Edge-models

# 2 · install the packaged application (Python 3.10+)
python -m pip install -e .

# 3 · optional: install local AI inference dependencies
python -m pip install -e ".[ai]"

# 4 · detect hardware and write the initial camera registry
sentinel setup

# 5 · start the grid and open the operator console
sentinel up
# → console opens at http://127.0.0.1:8000
```

Useful variants:

```bash
sentinel doctor                                      # dependency, hardware, model, and camera check
sentinel up --no-ai                                  # ingest + console only
sentinel up --file clip.mp4 --camera-id CAM_99       # connect a recorded file immediately
python run_sentinel.py --host 0.0.0.0 --port 8080    # run without the CLI wrapper
```

### First 3 minutes

1. Run `sentinel setup`, or open **Cameras** and save a camera mapping (`CAM_03`, an RTSP URL, webcam index, or `.mp4`).
2. Press **Connect** for each saved camera. Saving a mapping does not start capture.
3. Open **Identities**, drop 1–12 clear face photos, enter a name, and press **Enroll**. The gallery hot-reloads without a restart.
4. Use **Live** to view connected cameras, the tracking board, verified identities, events, and predictions.

### Installing from a wheel

Build and install the distributable package when using Sentinel Edge outside a checkout:

```bash
python -m pip install build
python -m build
python -m pip install dist/sentinel_edge-*.whl
sentinel setup
sentinel up
```

The `onvif` extra enables ONVIF discovery support. The `gpu` extra uses GPU-capable ONNX Runtime; use `all` to install both AI and ONVIF extras:

```bash
python -m pip install -e ".[gpu]"
python -m pip install -e ".[all]"
```

---

## 🏗 Architecture

```
┌────────────────────────── TIER 1 · INGEST ──────────────────────────┐
│  CameraSource threads (RTSP / USB / file)                           │
│  reconnect · hot-plug · FPS bounding · latest-frame semantics       │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼  FramePacket {camera_id, timestamp, frame, metadata}
┌────────────────────────── TIER 2 · AI CORE ─────────────────────────┐
│  quality gate → YOLO26n-face + ByteTrack → ArcFace(buffalo_l)       │
│  → FAISS identity search → verifier (hysteresis+margin)             │
│  → Markov transition predictor (persisted, Laplace-smoothed)        │
│  per-camera workers · bounded thread pool · gallery hot-reload      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼  UnifiedEvent
┌────────────────────────── TIER 3 · COMMAND HQ ──────────────────────┐
│  FastAPI + WebSocket fan-out · MJPEG live grid                      │
│  one-click enrollment · removable gallery · camera mapping CRUD     │
└─────────────────────────────────────────────────────────────────────┘
```

### Repository layout

```
├── run_sentinel.py            # single-command launcher for all three tiers
├── tier1-ingest/              # camera gateway and camera registry
├── tier2-ai/                  # AI core, models, scripts, and tests
│   ├── src/
│   │   ├── detection/         # YOLO26n-face + ByteTrack
│   │   ├── embedding/         # ArcFace embeddings
│   │   ├── identity_search/   # FAISS gallery and hot reload
│   │   ├── quality_gate/      # blur/size filtering
│   │   ├── runtime/           # device plan and bounded workers
│   │   ├── trajectory/        # transition graph and Markov predictor
│   │   └── verification/      # threshold and margin verification
│   ├── scripts/               # enrollment and threshold calibration
│   └── tests/                 # unit and adversarial regression tests
└── tier3-hq/                  # FastAPI server, console, and tests
```

The packaged launcher keeps the same layout under `sentinel/_app/`, so the `sentinel` command works after wheel installation as well as from a repository checkout.

---

## 🧪 Testing & Verification

```bash
pytest tier1-ingest/test_camera_source.py tier3-hq/test_server.py tier2-ai/tests/ -q
```

The tests cover:

- `FramePacket` and `UnifiedEvent` contract conformance
- malformed input, replayed packets, path traversal, forged timestamps, and gallery poisoning
- concurrency, worker bounds, registry pruning, and multi-camera processing
- device planning, face alignment, verifier logic, and FAISS search margins
- API validation and clean 4xx responses for hostile payloads

GPU is recommended for multi-camera live deployment. CPU is suitable for single-camera demos and development.

---

## ⚙️ Configuration

| What | Where | Notes |
|---|---|---|
| Camera mapping | `tier1-ingest/cameras.yaml` | managed by Command HQ; saved cameras are not auto-connected |
| Tier-2 allowlist | `tier2-ai/config/cameras.yaml` | defense-in-depth camera ID validation |
| Detector weights | `tier2-ai/models/yolo26n/` | local `.pt` file, or auto-downloaded from [YOLO26n-face](https://huggingface.co/Shubham12864/YOLO26n-face) |
| Verification threshold | `MatchVerifier(threshold=…)` | calibrate with `scripts/calibrate_threshold.py` |

Camera sources accept RTSP/HTTP URLs, numeric device indices, and recorded media paths. Camera IDs may contain letters, numbers, `_`, `.`, and `-` (maximum 64 characters). Runtime-added cameras are synchronized with the Tier-2 allowlist.

> **Biometrics stay local.** Face crops, uploaded enrollment images, identity vectors, and FAISS galleries are git-ignored and never leave the machine through this application.

---

## 🔌 HTTP and WebSocket surface

Tier 3 serves the browser console and exposes the following integration points:

| Endpoint | Purpose |
|---|---|
| `GET /api/stats` | health, counters, identities, and runtime metrics |
| `GET /api/events` | recent unified events |
| `WS /ws/events` | live event stream with a recent backlog |
| `GET /api/tracks` | currently active per-camera tracks |
| `GET /api/cameras` | saved mappings and connection state |
| `POST /api/cameras/save` | validate and save a camera mapping |
| `POST /api/cameras/{id}/connect` | explicitly start a saved source |
| `POST /api/cameras/{id}/disconnect` | stop capture while preserving mapping |
| `DELETE /api/cameras/{id}` | disconnect and remove a mapping |
| `POST /api/identities/upload` | quality-gated identity enrollment |
| `GET /api/identities/{id}/avatar` | first enrollment photo thumbnail |

## 🗺 Roadmap

- [ ] Liveness defense between detection and embedding
- [ ] Per-camera threshold adaptation and score calibration
- [ ] RAG advisory panel behind the event feed
- [ ] Map view with predicted-position markers
- [ ] Docker Compose deployment

---

## 👥 Team & Tracks

Built as three independent tracks around two frozen contracts:

| Track | Scope |
|---|---|
| **1 · Ingestion** | camera gateway, reconnection, and `FramePacket`s |
| **2 · AI Vision & Prediction** | detect → identify → predict |
| **3 · Orchestration & HQ** | event hub, console, and operator tooling |


## 🧰 Developer workflows

Run the focused regression suite from the repository root:

```bash
pytest tier1-ingest/test_camera_source.py tier3-hq/test_server.py tier2-ai/tests/ -q
```

For offline enrollment or threshold calibration, use the scripts in [tier2-ai/scripts](tier2-ai/scripts):

```bash
python tier2-ai/scripts/enroll_identities.py
python tier2-ai/scripts/calibrate_threshold.py
```

The main Tier-2 processing entry point is `UnifiedPipeline.process_frame_packet(...)`; it consumes a `FramePacket` and produces a `UnifiedEvent` for Tier 3.

## License

MIT. See the project metadata in `pyproject.toml`.

