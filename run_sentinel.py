"""Sentinel Edge -- single-architecture launcher (Tier 1 + Tier 2 + Tier 3).

Wires the three tracks into one running system:

    Tier1 CameraSource threads  ->  Tier2 Tier2Runtime (latest-frame workers
    -> UnifiedPipeline.process_frame_packet)  ->  Tier3 Hub.publish_event
    (event_sink)  ->  WebSocket fan-out + operator console.

Usage:
    python run_sentinel.py                 # cameras from tier1-ingest/cameras.yaml
    python run_sentinel.py --file PATH --camera-id CAM_99
    python run_sentinel.py --no-ai         # ingest+HQ only (no YOLO/ArcFace load)

Console:      http://localhost:8000
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
import yaml

REPO = Path(__file__).resolve().parent
T1, T2, T3 = REPO / "tier1-ingest", REPO / "tier2-ai", REPO / "tier3-hq"
for p in (str(T1), str(T2), str(T3)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_cameras() -> dict[str, dict]:
    cfg = T1 / "cameras.yaml"
    if not cfg.exists():
        return {}
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    cameras = data.get("cameras") or {}
    resolved = {}
    for k, v in cameras.items():
        meta = v if isinstance(v, dict) else {"source": v}
        src = meta.get("source")
        # file paths may be written relative to tier1-ingest/, repo root, or CWD
        if isinstance(src, str) and not src.startswith(("rtsp://", "http://", "https://")) \
                and not src.isdigit() and not Path(src).exists():
            for base in (REPO, T1):
                candidate = base / src
                if candidate.exists():
                    meta = dict(meta)
                    meta["source"] = str(candidate)
                    break
        resolved[str(k)] = meta
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-ai", action="store_true", help="run ingest + HQ without loading models")
    ap.add_argument("--file", help="add one recorded-file camera (with --camera-id)")
    ap.add_argument("--camera-id", default="CAM_99")
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    # ------------------------------------------------------------ tier 3 ---
    import server as hq
    hub = hq.HUB

    # ------------------------------------------------------------ tier 2 ---
    pipeline = runtime = None
    if not args.no_ai:
        from src.pipeline import UnifiedPipeline
        weights = T2 / "models" / "yolo26n" / "yolo26 widerdataset.pt"
        hq.IDENTITY_STORE.export_index()   # ensure an index file exists before startup
        pipeline = UnifiedPipeline.create_default(
            weights_path=weights if weights.exists() else None,
            identity_index_path=hq.EXPORTED_INDEX,   # always watch, even if just-created
            crop_dir=T2 / "data" / "face_crops",
            transition_graph_path=T2 / "data" / "transition_graph.json")
        from src.runtime.frame_runtime import Tier2Runtime
        runtime = Tier2Runtime(pipeline, event_sink=hub.publish_event, max_workers=8)
        print(f"[tier2] pipeline ready (device plan: {getattr(pipeline, 'embedder', None) and 'lazy-loaded on first frame'})")
    else:
        print("[tier2] --no-ai: events will not be produced (ingest + HQ only)")

    # ------------------------------------------------------------ tier 1 ---
    from ingest_manager import IngestManager
    manager = IngestManager()
    hub.pipeline, hub.runtime, hub._manager = pipeline, runtime, manager

    def on_packet(packet: dict) -> None:
        hub.register_frame(packet["camera_id"], packet["frame"])
        if runtime is not None:
            runtime.submit(packet)

    def manager_add(cid, src, fps, location=""):
        with hub.lock:
            hub.cameras[cid] = {"location": location, "fps": fps, "source": str(src),
                                "status": "starting",
                                "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        outcome = manager.add_camera(cid, src, on_packet, fps=fps)
        with hub.lock:
            hub.cameras[cid]["status"] = "running" if outcome.get("ok") else f"error: {str(outcome.get('error', 'open failed'))[:80]}"
        return outcome

    hub.ingest_add = manager_add
    hub.ingest_remove = manager.remove_camera

    cameras = load_cameras()
    if args.file:
        cameras[args.camera_id] = {"source": args.file, "fps": args.fps}
    # NOTE: nothing auto-connects at boot. Cameras are mapped in the registry
    # and operators connect them explicitly from Command HQ ("Connect" button).
    if args.file:  # an explicit --file IS operator intent: register + connect it
        outcome = manager_add(args.camera_id, args.file, args.fps)
        print(f"[tier1] {args.camera_id}: {args.file} -> "
              + ("running" if outcome.get("ok") else str(outcome.get("error", "failed"))))
    else:
        print(f"[tier1] {len(cameras)} camera(s) in registry (not auto-connected) "
              f"-- connect them from the console")

    # ------------------------------------------------------- graceful exit -
    stopping = threading.Event()

    def shutdown(_sig=None, _frm=None):
        if stopping.is_set():
            return
        stopping.set()
        print("\n[tier1] stopping cameras...")
        manager.stop_all()
        if runtime is not None:
            runtime.stop(timeout=3)
        print("[tier3] bye")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"\n=== SENTINEL EDGE ONLINE ===  console: http://{args.host}:{args.port}\n")
    uvicorn.run(hq.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
