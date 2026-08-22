"""Tier 3 -- Sentinel Edge Command HQ server.

FastAPI + WebSockets. Consumes Track 2 UnifiedEvents through a plain callable
(Hub.publish_event -- Tier 2's `event_sink`), fans them out to browsers over
/ws/events, serves the operator console (static/index.html), a per-camera
MJPEG live grid fed by Tier 1 snapshots, and gives operators software-style
control of the platform:

  - Identity database: one-click multi-image upload -> quality-gated ArcFace
    enrollment -> persistent FAISS gallery (survives restarts).
  - Camera mapping: add/remove/map cameras (RTSP / device / recorded file)
    against validated ids, with live stats.

No frontend build step: one dependency-free HTML file, served inline.
"""
from __future__ import annotations

import asyncio
import collections
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
TIER2 = REPO_ROOT / "tier2-ai"
CROPS_DIR = TIER2 / "data" / "face_crops"
UPLOAD_DIR = APP_DIR / "data" / "identity_uploads"
GALLERY_JSON = APP_DIR / "data" / "gallery.json"
EXPORTED_INDEX = APP_DIR / "data" / "identity_index.faiss"

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}$")
EVENT_FIELDS = {"track_id", "identity_name", "identity_id", "match_score", "verified",
                "current_camera", "timestamp", "face_crop_url", "next_camera",
                "eta_seconds", "transition_probability"}


# --------------------------------------------------------------------- store
class IdentityStore:
    """Operator-facing identity database backed by ArcFace + FAISS.

    Owns its persistence (gallery.json: names + raw vectors) so identities can
    be REMOVED (rebuild without them) -- something a bare FAISS index cannot do.
    Also exports a standard Track-2 index file for pipeline startup.
    """

    def __init__(self, index_path: Path = EXPORTED_INDEX):
        self.index_path = Path(index_path)
        self.gallery_json = GALLERY_JSON
        self._lock = threading.RLock()
        self.entries: list[dict[str, Any]] = []   # [{identity_id, name, vector}]
        self._embedder = None
        self._searcher = None
        self._load()

    def _load(self) -> None:
        import json
        if self.gallery_json.exists():
            try:
                self.entries = json.loads(self.gallery_json.read_text(encoding="utf-8"))
            except Exception:
                self.entries = []
        self._rebuild_searcher()

    def _ensure_models(self) -> None:
        if self._embedder is None:
            import sys
            if str(TIER2) not in sys.path:
                sys.path.insert(0, str(TIER2))
            from src.embedding.face_embedder import FaceEmbedder
            self._embedder = FaceEmbedder()

    def _rebuild_searcher(self) -> None:
        import sys
        if str(TIER2) not in sys.path:
            sys.path.insert(0, str(TIER2))
        from src.identity_search.faiss_index import IdentitySearch
        searcher = IdentitySearch()
        for entry in self.entries:
            searcher.add_identity(entry["name"], np.asarray(entry["vector"], dtype=np.float32),
                                  entry["identity_id"])
        self._searcher = searcher

    def search(self, embedding: np.ndarray):
        with self._lock:
            if self._searcher is None:
                return None, None, 0.0
            return self._searcher.search(embedding)

    def add_images(self, name: str, images: list[np.ndarray]) -> dict[str, Any]:
        """Quality-gated one-click enrollment: N images -> N gallery entries."""
        if not NAME_RE.fullmatch(name):
            raise ValueError("invalid identity name")
        self._ensure_models()
        added, skipped, vectors = 0, [], []
        for idx, image in enumerate(images):
            embedding = self._embedder.embed_image(image)
            if embedding is None:
                skipped.append({"image": idx, "reason": "no detectable face"})
                continue
            vectors.append(embedding)
            added += 1
        if added == 0:
            return {"added": 0, "skipped": skipped,
                    "error": "none of the images contained a usable face"}
        # coherence check: a single person's photos should correlate
        matrix = np.stack(vectors)
        sims = matrix @ matrix.T
        off_diag = sims[~np.eye(len(vectors), dtype=bool)]
        coherent = bool((off_diag > 0.25).mean() >= 0.5) if off_diag.size else True
        with self._lock:
            for vector in vectors:
                self.entries.append({"identity_id": name, "name": name,
                                     "vector": [float(x) for x in vector]})
            self._persist()
            self._rebuild_searcher()
            self.export_index()
        return {"added": added, "skipped": skipped, "coherent": coherent}

    def remove(self, identity_id: str) -> int:
        with self._lock:
            before = len(self.entries)
            self.entries = [e for e in self.entries if e["identity_id"] != identity_id]
            removed = before - len(self.entries)
            if removed:
                self._persist()
                self._rebuild_searcher()
                self.export_index()
            return removed

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            counts: dict[str, int] = {}
            for e in self.entries:
                counts[e["identity_id"]] = counts.get(e["identity_id"], 0) + 1
            return [{"identity_id": k, "name": k, "images": v} for k, v in sorted(counts.items())]

    def _persist(self) -> None:
        import json
        self.gallery_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.gallery_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries), encoding="utf-8")
        tmp.replace(self.gallery_json)

    def export_index(self) -> None:
        with self._lock:
            if self._searcher is not None:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                self._searcher.save(self.index_path)


# ----------------------------------------------------------------------- hub
class Hub:
    """In-memory state: events, sockets, camera registry, latest frames."""

    def __init__(self, max_events: int = 500):
        self.max_events = max_events
        self.events: collections.deque = collections.deque(maxlen=max_events)
        self.pipeline = None      # Tier-2 pipeline (deep health), set by launcher
        self.runtime = None       # Tier-2 runtime (worker stats), set by launcher
        self._manager = None      # Tier-1 IngestManager, set by launcher
        self.sockets: set[WebSocket] = set()
        self.cameras: dict[str, dict[str, Any]] = {}
        self.latest_frames: dict[str, tuple[int, Any]] = {}     # cid -> (seq, frame)
        self.frame_seqs: dict[str, int] = {}
        self.lock = threading.RLock()
        self.counters = {"received": 0, "broadcast": 0, "frames": 0}
        self.loop: asyncio.AbstractEventLoop | None = None      # uvicorn's loop
        self.ingest_add: Callable[[str, Any, float], dict] | None = None
        self.ingest_remove: Callable[[str], bool] | None = None

    def register_frame(self, camera_id: str, frame: np.ndarray) -> None:
        with self.lock:
            seq = self.frame_seqs.get(camera_id, 0) + 1
            self.frame_seqs[camera_id] = seq
            self.latest_frames[camera_id] = (seq, frame)
            self.counters["frames"] += 1

    def _broadcast(self, payload: dict[str, Any]) -> None:
        if self.loop is None:
            return
        async def _push():
            dead = []
            for ws in list(self.sockets):
                try:
                    await ws.send_json(payload)
                    self.counters["broadcast"] += 1
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.sockets.discard(ws)
        asyncio.run_coroutine_threadsafe(_push(), self.loop)

    def publish_event(self, event: dict[str, Any]) -> None:
        """Track-2 event_sink lands here. Never raises."""
        try:
            if not isinstance(event, dict):
                return
            clean = {k: event.get(k) for k in EVENT_FIELDS}
            clean["_received_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with self.lock:
                self.events.append(clean)
                self.counters["received"] += 1
            self._broadcast(clean)
        except Exception:
            pass


HUB = Hub()


# ---------------------------------------------------------------------- app
app = FastAPI(title="Sentinel Edge Command HQ", version="1.0")


@app.on_event("startup")
async def _capture_loop() -> None:
    HUB.loop = asyncio.get_running_loop()


@app.get("/", response_class=HTMLResponse)
async def index():
    html = APP_DIR / "static" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/api/stats")
async def stats():
    with HUB.lock:
        now = time.time()
        recent = 0
        for e in HUB.events:
            try:
                stamp = datetime.strptime(e["_received_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
                if now - stamp <= 60:
                    recent += 1
            except Exception:
                pass
        ingest = [src.stats() for src in getattr(HUB, "_manager", None) and HUB._manager._sources.values()] if getattr(HUB, "_manager", None) else []
        tier2 = HUB.pipeline.health() if HUB.pipeline is not None else None
        runtime = {k: v for k, v in (HUB.runtime.health() or {}).items()
                   if k != "pipeline"} if HUB.runtime is not None else None
    return {"cameras": len(HUB.cameras), "live_cameras": len(HUB.latest_frames),
            "events_total": HUB.counters["received"], "events_recent_60s": recent,
            "frames_ingested": HUB.counters["frames"],
            "sockets": len(HUB.sockets),
            "identities": len(IDENTITY_STORE.list()),
            "uptime_s": round(now - START_TIME, 1),
            "tier1_sources": ingest, "tier2": tier2, "tier2_runtime": runtime}


START_TIME = time.time()


@app.get("/api/events")
async def events(limit: int = 100):
    limit = max(1, min(int(limit), HUB.max_events))
    with HUB.lock:
        items = list(HUB.events)[-limit:]
    return {"events": items}


@app.post("/api/test-event")
async def test_event(event: dict[str, Any]):
    missing = EVENT_FIELDS - set(event)
    if missing:
        raise HTTPException(422, f"not a UnifiedEvent; missing {sorted(missing)}")
    HUB.publish_event(event)
    return {"ok": True}


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    HUB.sockets.add(ws)
    try:
        with HUB.lock:
            backlog = list(HUB.events)[-50:]
        for event in backlog:
            await ws.send_json(event)
        while True:
            await ws.receive_text()          # keepalive/pings from client
    except WebSocketDisconnect:
        pass
    finally:
        HUB.sockets.discard(ws)


# --------------------------------------------------------------- cameras api
@app.get("/api/cameras")
async def cameras():
    with HUB.lock:
        rows = []
        for cid, meta in sorted(HUB.cameras.items()):
            row = dict(meta)
            row["camera_id"] = cid
            row["has_live_frame"] = cid in HUB.latest_frames
            rows.append(row)
    return {"cameras": rows}


@app.post("/api/cameras")
async def add_camera(body: dict[str, Any]):
    cid = str(body.get("camera_id", "")).strip()
    source = body.get("source", "")
    fps = float(body.get("fps", 4.0))
    location = str(body.get("location", ""))[:120]
    if not CAMERA_ID_RE.fullmatch(cid):
        raise HTTPException(422, f"invalid camera_id {cid!r} (allowed: A-Z a-z 0-9 _ . - , max 64)")
    if source in ("", None):
        raise HTTPException(422, "source required (rtsp:// URL, device index, or media path)")
    if not (0.5 <= fps <= 30):
        raise HTTPException(422, "fps must be within 0.5..30")
    with HUB.lock:
        HUB.cameras[cid] = {"location": location, "fps": fps, "source": str(source),
                            "status": "mapped", "added_at":
                            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    result = {"ok": True, "camera_id": cid}
    if HUB.ingest_add is not None:
        outcome = HUB.ingest_add(cid, source, fps)
        with HUB.lock:
            HUB.cameras[cid]["status"] = "running" if outcome.get("ok") else f"error: {outcome.get('error', 'open failed')[:80]}"
        result.update(outcome)
    return result


@app.delete("/api/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    with HUB.lock:
        existed = HUB.cameras.pop(camera_id, None)
        HUB.latest_frames.pop(camera_id, None)
    if existed is None:
        raise HTTPException(404, f"unknown camera {camera_id!r}")
    if HUB.ingest_remove is not None:
        HUB.ingest_remove(camera_id)
    return {"ok": True, "removed": camera_id}


@app.get("/api/cameras/{camera_id}/snapshot")
async def snapshot(camera_id: str):
    with HUB.lock:
        item = HUB.latest_frames.get(camera_id)
    if item is None:
        raise HTTPException(404, f"no frame yet for {camera_id!r}")
    _, frame = item
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "encode failed")
    return JSONResponse(content={"jpeg_base64": __import__("base64").b64encode(buf.tobytes()).decode(),
                                 "seq": item[0]})


async def _mjpeg(camera_id: str):
    last_seq = -1
    idle = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
    while True:
        with HUB.lock:
            item = HUB.latest_frames.get(camera_id)
        if item is None or item[0] == last_seq:
            yield idle
            await asyncio.sleep(0.15)
            continue
        last_seq, frame = item
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            await asyncio.sleep(0.15)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        await asyncio.sleep(0.04)


@app.get("/api/feed/{camera_id}")
async def feed(camera_id: str):
    with HUB.lock:
        known = camera_id in HUB.cameras
    if not known:
        raise HTTPException(404, f"unknown camera {camera_id!r}")
    return StreamingResponse(_mjpeg(camera_id),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ------------------------------------------------------------- identities api
IDENTITY_STORE = IdentityStore()


@app.get("/api/identities")
async def identities():
    return {"identities": IDENTITY_STORE.list()}


@app.post("/api/identities/upload")
async def upload_identities(files: list[UploadFile] = File(...), name: str = Form(...)):
    name = name.strip()
    if not NAME_RE.fullmatch(name):
        raise HTTPException(422, f"invalid identity name {name!r}")
    if not files:
        raise HTTPException(422, "no files uploaded")
    if len(files) > 12:
        raise HTTPException(422, "max 12 images per upload")
    safe_dir = UPLOAD_DIR / name.replace(" ", "_")
    safe_dir.mkdir(parents=True, exist_ok=True)
    images, saved = [], []
    for f in files:
        raw = await f.read()
        if len(raw) > 10 * 1024 * 1024:
            skipped_note = f"{f.filename}: >10MB rejected"
            continue
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            continue
        stem = re.sub(r"[^A-Za-z0-9_.\-]", "_", Path(f.filename or "img.jpg").stem)[:60] or "img"
        out = safe_dir / f"{stem}_{int(time.time())}_{len(saved)}.jpg"
        cv2.imwrite(str(out), image)
        saved.append(out.name)
        images.append(image)
    if not images:
        raise HTTPException(422, "no decodable images in upload")
    try:
        result = IDENTITY_STORE.add_images(name, images)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result.update({"name": name, "uploaded": len(images), "stored_files": saved})
    return JSONResponse(result)


@app.delete("/api/identities/{identity_id}")
async def delete_identity(identity_id: str):
    removed = IDENTITY_STORE.remove(identity_id)
    if not removed:
        raise HTTPException(404, f"unknown identity {identity_id!r}")
    return {"ok": True, "removed_entries": removed}
