"""Tier 1 -- IngestManager: fleet lifecycle + live stats for every source."""
from __future__ import annotations

import threading
from typing import Any, Callable, Mapping


class IngestManager:
    """Owns CameraSources: start/stop the fleet, report health, survive churn."""

    def __init__(self):
        self._sources: dict[str, Any] = {}
        self._lock = threading.Lock()

    def add_camera(self, camera_id: str, source: Any, on_packet: Callable[[Mapping[str, Any]], None],
                   fps: float = 5.0, **kwargs) -> dict[str, Any]:
        from camera_source import CAMERA_ID_RE
        if not CAMERA_ID_RE.fullmatch(str(camera_id)):
            return {"ok": False, "error": f"invalid camera_id {camera_id!r}: must match ^[A-Za-z0-9_.\\-]{{1,64}}$"}
        with self._lock:
            existing = self._sources.get(camera_id)
            if existing is not None and existing.is_alive():
                return {"ok": False, "error": f"{camera_id} already running"}
            if existing is not None:
                existing.stop()
            try:
                src = _make_source(camera_id, source, on_packet, fps=fps, **kwargs)
                src.start()
                self._sources[camera_id] = src
            except Exception as exc:
                return {"ok": False, "error": f"failed to open {camera_id}: {type(exc).__name__}: {exc}"}
            return {"ok": True, "camera_id": camera_id}

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            src = self._sources.pop(camera_id, None)
        if src is None:
            return False
        src.stop()
        return True

    def stop_all(self) -> None:
        with self._lock:
            sources = list(self._sources.values())
            self._sources.clear()
        for src in sources:
            src.stop()

    def stats(self) -> list[dict[str, Any]]:
        with self._lock:
            sources = list(self._sources.values())
        return [src.stats() for src in sources]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for src in self._sources.values() if src.is_alive())


def _make_source(camera_id: str, source: Any, on_packet, fps: float, **kwargs):
    from camera_source import CameraSource
    return CameraSource(camera_id, source, on_packet, fps=fps, **kwargs)
