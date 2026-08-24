"""Sentinel CLI -- the one command users interact with.

    sentinel setup     first-run wizard: detects hardware & cameras, writes config
    sentinel up        start the whole grid and open the console
    sentinel doctor    health check: deps, models, hardware, camera reachability
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T1, T2, T3 = ROOT / "tier1-ingest", ROOT / "tier2-ai", ROOT / "tier3-hq"
for p in (str(T1), str(T2), str(T3)):
    if p not in sys.path:
        sys.path.insert(0, p)

BANNER = r"""
  ____  _____ _                __   ____  _   _
 / ___|| ____| | ___   _ _ __ / _| / ___|| | | | ___
 \___ \|  _| | |/ / | | | '__| |_ | |  _| |_| |/ _ \
  ___) | |___|   <| |_| | |  |  _|| |_| |  _  |  __/
 |____/|_____|_|\_\\__,_|_|  |_|   \____|_| |_|\___|
        edge-native AI surveillance · BYOC edition
"""


def have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# ------------------------------------------------------------------ doctor --
def cmd_doctor(args) -> int:
    from sentinel.discover import compute_plan, cpu_count, discover_webcams, gpu_info

    print(BANNER)
    ok = True
    checks = [
        ("opencv-python", "cv2"), ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"), ("pyyaml", "yaml"), ("multipart", "multipart"),
    ]
    print("dependencies:")
    for label, mod in checks:
        present = have(mod)
        ok &= present
        print(f"  [{'✓' if present else '✗'}] {label}")
    ai = have("torch") and have("ultralytics") and have("insightface") and have("faiss")
    print(f"  [{'✓' if ai else '○'}] ai extras "
          f"{'installed' if ai else 'missing (console-only mode; pip install sentinel-edge[ai])'}")
    weights = T2 / "models" / "yolo26n" / "yolo26 widerdataset.pt"
    if weights.exists():
        print("  [✓] YOLO26n-face weights found")
    else:
        print("  [!] weights missing -> will auto-download from huggingface on first use")

    print("\nhardware:")
    plan = compute_plan(ai_installed=ai)
    g = plan["gpu"]
    accel = "GPU" if plan["verdict"] == "gpu" else ("CPU" if ai else "—")
    print(f"  verdict : {plan['verdict']} ({accel}) — {plan['detail']}")
    print(f"  cores   : {cpu_count()}   ram: {plan['ram_gb'] or '?'}GB   "
          f"cuda_torch={g['cuda_torch']} onnx_gpu={g['onnx_gpu']}")

    print("\ncameras (probing indices 0..4):")
    cams = discover_webcams(max_index=5)
    if cams:
        for c in cams:
            print(f"  [✓] webcam {c['index']}: {c['width']}x{c['height']} @ {c['fps']}fps")
    else:
        print("  [ ] no local webcams found (RTSP/file sources still fine)")
    registry = _load_registry()
    if registry:
        from sentinel.discover import discover_rtsp
        urls = [m["source"] for m in registry.values()
                if isinstance(m.get("source"), str) and m["source"].startswith("rtsp://")]
        for u in urls:
            alive = discover_rtsp([u])
            print(f"  [{'✓' if alive else '✗'}] {u}")
    print(f"\nconfig : {_config_path()}")
    return 0 if ok else 1


# --------------------------------------------------------------- registry ---
def _config_path() -> Path:
    return T1 / "cameras.yaml"


def _load_registry() -> dict[str, dict]:
    import yaml
    cfg = _config_path()
    if not cfg.exists():
        return {}
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    cams = data.get("cameras") or {}
    return {str(k): (v if isinstance(v, dict) else {"source": v}) for k, v in cams.items()}


def _save_registry(registry: dict[str, dict]) -> None:
    import yaml
    cfg = _config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Tier 1 camera registry - edited by Command HQ / sentinel wizard.\n"
              "# source: rtsp:// URL | device index | media file path\n")
    payload = {"cameras": {}}
    for cid, meta in sorted(registry.items()):
        entry: dict = {"source": meta["source"], "fps": float(meta.get("fps", 4.0))}
        if meta.get("location"):
            entry["location"] = meta["location"]
        payload["cameras"][cid] = entry
    cfg.write_text(header + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# ------------------------------------------------------------------ wizard --
def cmd_setup(args) -> int:
    from sentinel.discover import compute_plan, cpu_count, discover_webcams, gpu_info, scan_onvif_subnet

    print(BANNER)
    ai_installed = have("ultralytics") and have("insightface")

    # 1. hardware -----------------------------------------------------------
    print("[1/4] detecting hardware...")
    g = gpu_info()
    plan = compute_plan(ai_installed=ai_installed)
    if g["name"]:
        print(f"      GPU   : {g['name']} ({g['vram_gb']} GB) "
              f"[torch-cuda={g['cuda_torch']} onnx-gpu={g['onnx_gpu']}]")
    else:
        print("      GPU   : none detected")
    print(f"      CPU   : {cpu_count()} cores · RAM ~{plan['ram_gb'] or '?'} GB")
    print(f"      mode  : {plan['verdict']} — {plan['detail']}")

    # 2. cameras ------------------------------------------------------------
    print("[2/4] discovering cameras...")
    registry = _load_registry()
    webcams = discover_webcams(max_index=args.max_webcams)
    for c in webcams:
        cid = f"WEBCAM_{c['index']}"
        if cid not in registry:
            fps = 6.0 if plan["verdict"] == "gpu" else (3.0 if plan["verdict"] == "cpu-strong" else 2.0)
            registry[cid] = {"source": c["index"], "fps": fps,
                             "location": "auto-detected webcam"}
            print(f"      + {cid}: {c['width']}x{c['height']} @ {fps}fps")
        else:
            print(f"      = {cid}: already mapped")
    if not webcams:
        print("      no local webcams auto-detected")
    if args.scan_onvif:
        print("      scanning LAN for ONVIF devices (port 8899)...")
        for ep in scan_onvif_subnet(subnet=args.subnet):
            print(f"      ? ONVIF service at {ep} — add via console with rtsp creds")

    # 3. write config ---------------------------------------------------------
    print("[3/4] writing camera registry...")
    _save_registry(registry)
    for cid, m in registry.items():
        print(f"      {cid} -> {m['source']} @ {m.get('fps', 4)}fps")

    # 4. summary --------------------------------------------------------------
    print("[4/4] done.\n")
    print("next steps:")
    print("  sentinel up          start everything + open console")
    if not ai_installed:
        print('  pip install "sentinel-edge[ai]"   enable recognition (YOLO+ArcFace)')
    print("  console > Cameras > Connect to bring a mapped camera online")
    return 0


# ---------------------------------------------------------------------- up --
def cmd_up(args) -> int:
    launcher = ROOT / "run_sentinel.py"

    def open_console():
        time.sleep(6)
        webbrowser.open(f"http://127.0.0.1:{args.port}")

    if args.open:
        threading.Thread(target=open_console, daemon=True).start()

    cmd = [sys.executable, str(launcher), "--port", str(args.port)]
    if args.no_ai:
        cmd.append("--no-ai")
    if args.file:
        cmd += ["--file", args.file]
    if args.camera_id:
        cmd += ["--camera-id", args.camera_id]
    try:
        proc = subprocess.run(cmd)
        return proc.returncode or 0
    except KeyboardInterrupt:
        return 0


# --------------------------------------------------------------------- main --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sentinel",
                                 description="Sentinel Edge — edge-native multi-camera AI surveillance")
    sub = ap.add_subparsers(dest="cmd")

    s_setup = sub.add_parser("setup", help="first-run wizard: detect hardware/cameras, write config")
    s_setup.add_argument("--max-webcams", type=int, default=3)
    s_setup.add_argument("--scan-onvif", action="store_true", help="probe LAN port 8899 for ONVIF cams")
    s_setup.add_argument("--subnet", default="192.168.1")
    s_setup.set_defaults(func=cmd_setup)

    s_up = sub.add_parser("up", help="run the full grid + open the console")
    s_up.add_argument("--port", type=int, default=8000)
    s_up.add_argument("--no-ai", action="store_true")
    s_up.add_argument("--file", help="connect one video file as a demo camera")
    s_up.add_argument("--camera-id", default="CAM_99")
    s_up.add_argument("--no-open", dest="open", action="store_false", help="do not open the browser")
    s_up.set_defaults(func=cmd_up, open=True)

    s_doc = sub.add_parser("doctor", help="health check: deps, hardware, cameras")
    s_doc.add_argument("--max-webcams", type=int, default=3)
    s_doc.set_defaults(func=cmd_doctor)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
