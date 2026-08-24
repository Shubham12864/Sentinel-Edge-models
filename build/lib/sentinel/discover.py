"""Sentinel Edge -- hardware discovery: cameras, GPU, compute plan.

Everything the first-run wizard and the launcher need to make intelligent
defaults on an unknown machine:
  - discover_webcams(): probe OpenCV device indices (works on Windows/Linux/macOS)
  - discover_rtsp_cameras(): probe saved RTSP URLs / optional LAN ONVIF scan
  - compute_plan(): GPU vs CPU decision across torch + onnxruntime backends
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Callable


# --------------------------------------------------------------------- webcams
def discover_webcams(max_index: int = 5, quick_open_ms: int = 700,
                     probe: Callable[[int], tuple[bool, Any]] | None = None) -> list[dict[str, Any]]:
    """Return [{'index': i, 'width': w, 'height': h, 'fps': f}, ...] for live webcams.

    Uses a background-thread timeout per index because VideoCapture(0) can hang
    for seconds on busy/locked devices.
    """
    try:
        import cv2
    except ImportError:
        return []

    def default_probe(idx: int):
        cap = cv2.VideoCapture(idx)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        info = None
        if ok:
            info = {"index": idx,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 1)}
        cap.release()
        return bool(ok), info

    probe_fn = probe or default_probe
    found: list[dict[str, Any]] = []
    for idx in range(max_index):
        result: dict[str, Any] = {}

        def target(idx=idx, result=result):
            ok, info = probe_fn(idx)
            if ok:
                result.update(info)

        thread = _TimeoutThread(target, quick_open_ms)
        thread.start(); thread.join(quick_open_ms / 1000 + 0.3)
        if result:
            found.append(result)
    return found


class _TimeoutThread:
    """Fire-and-forget worker whose join we simply time out (daemon-safe)."""

    def __init__(self, fn, timeout_ms: int):
        import threading
        self._t = threading.Thread(target=fn, daemon=True)
        self._timeout = timeout_ms / 1000

    def start(self): self._t.start()
    def join(self, s: float):
        self._t.join(s)


# ----------------------------------------------------------------------- rtsp
def discover_rtsp(rtsp_urls: list[str], timeout_s: float = 4.0) -> list[dict[str, str]]:
    """Check which configured RTSP URLs respond; returns reachable ones."""
    reachable = []
    for url in rtsp_urls:
        if _rtsp_responds(url, timeout_s):
            reachable.append({"url": url})
    return reachable


def _rtsp_responds(url: str, timeout_s: float) -> bool:
    try:
        import socket
        from urllib.parse import urlparse
        p = urlparse(url)
        host, port = p.hostname, p.port or 554
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def scan_onvif_subnet(subnet: str = "192.168.1", prefix_range: range | None = None,
                      port: int = 8899, timeout_s: float = 0.25) -> list[str]:
    """Quick sweep for ONVIF device-service ports (optional; needs nothing exotic).

    Returns 'http://ip:port/onvif/device_service' style endpoints that accepted
    a TCP connection.  Kept deliberately shallow -- real WS-Discovery is overkill.
    """
    hosts: list[str] = []
    sockets = []
    try:
        import concurrent.futures as cf
        import socket

        def probe(ip: str) -> str | None:
            try:
                with socket.create_connection((f"{ip}", port), timeout=timeout_s):
                    return ip
            except OSError:
                return None

        ips = [f"{subnet}.{i}" for i in (prefix_range or range(1, 255))]
        with cf.ThreadPoolExecutor(max_workers=64) as pool:
            for res in pool.map(probe, ips):
                if res:
                    hosts.append(res)
    except Exception:
        pass
    finally:
        for s in sockets:
            try: s.close()
            except Exception: pass
    return [f"http://{h}:{port}/onvif/device_service" for h in hosts]


# ------------------------------------------------------------------ compute
def gpu_info() -> dict[str, Any]:
    """Best-effort GPU report without importing heavy stacks unnecessarily."""
    info: dict[str, Any] = {"cuda_torch": False, "onnx_gpu": False, "name": None, "vram_gb": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_torch"] = True
            props = torch.cuda.get_device_properties(0)
            info["name"] = props.name
            info["vram_gb"] = round(props.total_memory / 1024**3, 1)
    except Exception:
        pass
    try:
        import onnxruntime as ort
        info["onnx_gpu"] = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        pass
    if info["name"] is None and shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            name, _, mem = out.stdout.partition(",")
            if name.strip():
                info["name"] = name.strip()
                try:
                    info["vram_gb"] = round(int(mem.strip().split()[0]) / 1024, 1)
                except Exception:
                    pass
        except Exception:
            pass
    return info


def cpu_count() -> int:
    import os
    return os.cpu_count() or 4


def total_ram_gb() -> float | None:
    try:
        import psutil  # optional dep
        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        try:  # windows fallback via wmic-free route
            out = subprocess.run(["systeminfo"], capture_output=True, text=True, timeout=10)
            for line in out.stdout.splitlines():
                if "Total Physical Memory" in line:
                    gb = float(line.split(":")[1].strip().split()[0].replace(",", ""))
                    return round(gb / 1024, 1)
        except Exception:
            pass
    return None


def compute_plan(ai_installed: bool) -> dict[str, Any]:
    """The wizard's recommendation: what this machine can realistically run."""
    g = gpu_info()
    cores = cpu_count()
    ram = total_ram_gb()
    has_gpu_accel = bool(g["cuda_torch"] or g["onnx_gpu"])
    if not ai_installed:
        verdict = "console-only"
        detail = "AI extras not installed -> run `pip install sentinel-edge[ai]` to enable recognition"
    elif has_gpu_accel:
        verdict = "gpu"
        fps = "8-15 fps/camera sustained"
        detail = f"{g['name'] or 'GPU'} ({g['vram_gb']}GB) · {fps}"
    elif cores >= 8:
        verdict = "cpu-strong"
        detail = f"{cores} cores · ~1-2 fps/camera · fine for 1-2 cameras at reduced FPS"
    else:
        verdict = "cpu-light"
        detail = f"{cores} cores · use 1 camera @ ~2 fps for demo mode"
    return {"verdict": verdict, "detail": detail, "cores": cores,
            "ram_gb": ram, "gpu": g}
