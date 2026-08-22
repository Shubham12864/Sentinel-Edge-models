"""Bounded latest-frame execution with one worker thread per camera.

Tier 1 owns camera decoding.  This module only accepts already-created
FramePackets and deliberately retains at most one unprocessed frame per
camera, preventing stale-video backlog and unbounded memory growth.  Each
active camera gets its own worker thread, so a slow or busy camera can no
longer starve the others behind a single shared loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from typing import Any, Callable, Mapping

from ..schemas import FramePacket

@dataclass
class RuntimeMetrics:
    submitted: int = 0
    replaced_stale: int = 0
    processed: int = 0
    emitted_events: int = 0
    sink_errors: int = 0

class LatestFrameBuffer:
    """A fair, per-camera, capacity-one FramePacket buffer."""

    def __init__(self):
        self._slots: dict[str, dict[str, Any]] = {}
        self._condition = Condition()
        self.metrics = RuntimeMetrics()

    def put(self, packet: Mapping[str, Any]) -> str:
        """Store the packet as its camera's latest frame; returns camera_id."""
        validated = FramePacket.from_mapping(packet)
        with self._condition:
            slot = self._slots.get(validated.camera_id)
            if slot is None:
                slot = {"packet": None}
                self._slots[validated.camera_id] = slot
            elif slot["packet"] is not None:
                self.metrics.replaced_stale += 1
            slot["packet"] = packet
            self.metrics.submitted += 1
            self._condition.notify_all()
            return validated.camera_id

    def pop(self, camera_id: str, timeout: float | None = None) -> Mapping[str, Any] | None:
        """Wait for and take this camera's pending frame, if any."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                slot = self._slots.get(camera_id)
                if slot is not None and slot["packet"] is not None:
                    packet = slot["packet"]
                    slot["packet"] = None
                    return packet
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.1) if remaining is not None else 0.1)

    def pending_cameras(self) -> list[str]:
        with self._condition:
            return [camera_id for camera_id, slot in self._slots.items() if slot["packet"] is not None]

    def __len__(self) -> int:
        with self._condition:
            return sum(1 for slot in self._slots.values() if slot["packet"] is not None)

class Tier2Runtime:
    """Optional local worker bridge between Tier 1 FramePackets and an event sink.

    One daemon thread per active camera consumes that camera's latest frame,
    so cameras proceed independently instead of queueing behind one shared
    loop.  The sink is intentionally a callback: Tier 3 can supply an HTTP
    publisher, a local function, or a test collector without making this AI
    repository own a FastAPI server or WebSocket implementation.
    """

    POLL_SECONDS = 0.1

    def __init__(self, pipeline, event_sink: Callable[[dict[str, Any]], None] | None = None):
        self.pipeline, self.event_sink = pipeline, event_sink
        self.buffer = LatestFrameBuffer()
        self.metrics = self.buffer.metrics
        self._stop = Event()
        self._threads: dict[str, Thread] = {}
        self._workers_lock = Lock()

    def submit(self, packet: Mapping[str, Any]) -> None:
        camera_id = self.buffer.put(packet)
        self._ensure_worker(camera_id)

    def _ensure_worker(self, camera_id: str) -> None:
        with self._workers_lock:
            existing = self._threads.get(camera_id)
            if existing is not None and existing.is_alive():
                return
            if self._stop.is_set():
                raise RuntimeError("Tier2Runtime is stopped; call start() before submitting packets")
            thread = Thread(target=self._camera_loop, args=(camera_id,), name=f"tier2-worker-{camera_id}", daemon=True)
            self._threads[camera_id] = thread
        thread.start()

    def _camera_loop(self, camera_id: str) -> None:
        while not self._stop.is_set():
            packet = self.buffer.pop(camera_id, timeout=self.POLL_SECONDS)
            if packet is not None:
                self.process_packet(packet)

    def process_packet(self, packet: Mapping[str, Any]) -> list[dict[str, Any]]:
        events = self.pipeline.process_frame_packet(packet)
        self.metrics.processed += 1
        for event in events:
            if self.event_sink is not None:
                try:
                    self.event_sink(event)
                except Exception:
                    self.metrics.sink_errors += 1
            self.metrics.emitted_events += 1
        return events

    def process_once(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Legacy synchronous pull across cameras (single-consumer mode)."""
        pending = self.buffer.pending_cameras()
        if not pending:
            return []
        packet = self.buffer.pop(pending[0], timeout=timeout)
        if packet is None:
            return []
        return self.process_packet(packet)

    def start(self) -> None:
        """Resume (or begin) processing; workers spawn lazily per camera on submit."""
        self._stop.clear()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._workers_lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout)

    def health(self) -> dict[str, Any]:
        with self._workers_lock:
            active_workers = sum(1 for thread in self._threads.values() if thread.is_alive())
        return {"queued_cameras": len(self.buffer), "worker_running": active_workers > 0,
                "active_workers": active_workers,
                **self.metrics.__dict__, "pipeline": self.pipeline.health()}
