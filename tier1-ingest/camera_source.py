"""Tier 1 -- Camera gateway: any OpenCV-capturable source becomes FramePackets.

A CameraSource is one daemon thread owning exactly one capture.  It decodes
at a bounded FPS, stamps UTC ISO timestamps, wraps frames in the frozen
Track-1 contract (camera_id / timestamp / frame / metadata) and hands them
to a callback (normally Tier2Runtime.submit).  Read failures are tolerated
briefly (glitch) and then treated as a disconnect: the capture is released,
the source is re-opened after a delay, and reconnects are counted.  File
sources loop by default so recorded footage behaves like a live feed.
"""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class CameraSource(threading.Thread):
    """One capture -> many FramePackets, with reconnect + FPS bounding."""

    def __init__(self, camera_id: str, source: Any, on_packet: Callable[[Mapping[str, Any]], None],
                 fps: float = 5.0, reconnect_delay: float = 2.0, max_consecutive_failures: int = 30,
                 loop_file: bool = True, capture_factory: Callable[[], Any] | None = None):
        super().__init__(name=f"tier1-{camera_id}", daemon=True)
        if not CAMERA_ID_RE.fullmatch(str(camera_id)):
            raise ValueError(f"invalid camera_id {camera_id!r}: must match ^[A-Za-z0-9_.\\-]{{1,64}}$")
        self.camera_id, self.source = str(camera_id), source
        self.on_packet = on_packet
        self.fps = max(0.5, float(fps))
        self.reconnect_delay = float(reconnect_delay)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.loop_file = bool(loop_file)
        self._capture_factory = capture_factory or (lambda: _open_capture(source))
        self._stop = threading.Event()
        # observability
        self.frames_captured = 0
        self.frames_dropped = 0
        self.reconnects = 0
        self.stream_ended = False
        self.last_error: str | None = None

    # ------------------------------------------------------------------ api
    def stop(self) -> None:
        self._stop.set()

    def stats(self) -> dict[str, Any]:
        return {"camera_id": self.camera_id, "source": str(self.source),
                "alive": self.is_alive(), "frames_captured": self.frames_captured,
                "frames_dropped": self.frames_dropped, "reconnects": self.reconnects,
                "stream_ended": self.stream_ended, "last_error": self.last_error}

    # ----------------------------------------------------------------- loop
    def run(self) -> None:
        capture = None
        try:
            while not self._stop.is_set():
                if capture is None or not capture.isOpened():
                    capture = self._capture_factory()
                    if capture is None or not getattr(capture, "isOpened", lambda: False)():
                        self.last_error = "open failed"
                        self._sleep(self.reconnect_delay)
                        continue
                ok, frame = self._read(capture)
                if ok:
                    self._emit(frame)
                    self._pace_frame()
                    continue
                self.frames_dropped += 1
                drops = self.frames_dropped - self._last_reconnect_drop
                if drops >= self.max_consecutive_failures:
                    self._handle_disconnect(capture)
                    capture = None  # force reopen on next pass
                else:
                    self._sleep(0.05)
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass

    # ------------------------------------------------------------ internals
    def _read(self, capture: Any):
        try:
            return capture.read()
        except Exception as exc:  # defensive: some backends raise instead of (False, None)
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False, None

    def _emit(self, frame) -> None:
        packet = {
            "camera_id": self.camera_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "frame": frame,
            "metadata": {"source": str(self.source), "source_kind": _kind_of(self.source),
                         "ingest_fps": self.fps, "frame_seq": self.frames_captured},
        }
        self.frames_captured += 1
        try:
            self.on_packet(packet)
        except Exception as exc:  # downstream must never kill ingestion
            self.last_error = f"sink: {type(exc).__name__}: {exc}"

    def _pace_frame(self) -> None:
        self._sleep(1.0 / self.fps)

    @property
    def _last_reconnect_drop(self) -> int:
        """Drops already attributed to the previous reconnect window."""
        return getattr(self, "_attributed_drops", 0)

    def _handle_disconnect(self, capture: Any) -> None:
        self.reconnects += 1
        self._attributed_drops = self.frames_dropped
        try:
            capture.release()
        except Exception:
            pass
        if _kind_of(self.source) == "file":
            if self.loop_file:
                self.last_error = "file ended; looping back to frame 0"
                return  # reopen restarts from frame 0 on the next pass
            self.stream_ended = True
            self.last_error = "file stream ended"
            self._stop.set()  # a finite recording is done -- exit cleanly
            return
        self.stream_ended = False
        self.last_error = f"disconnected after {self.max_consecutive_failures} bad reads; retrying"
        self._sleep(self.reconnect_delay)

    def _sleep(self, seconds: float) -> None:
        self._stop.wait(max(0.0, seconds))


def _open_capture(source: Any):
    import cv2
    capture = cv2.VideoCapture(source)
    if _kind_of(source) == "file":
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return capture


def _kind_of(source: Any) -> str:
    if isinstance(source, str) and str(source).startswith(("rtsp://", "http://", "https://")):
        return "network"
    if isinstance(source, str):
        return "file"
    return "device"  # int webcam index
