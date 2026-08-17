"""Bounded latest-frame execution for a long-running Tier 2 worker.

Tier 1 owns camera decoding.  This module only accepts already-created
FramePackets and deliberately retains at most one unprocessed frame per
camera, preventing stale-video backlog and unbounded memory growth.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Thread
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
        self._packets: dict[str, Mapping[str, Any]] = {}
        self._order: deque[str] = deque()
        self._condition = Condition()
        self.metrics = RuntimeMetrics()

    def put(self, packet: Mapping[str, Any]) -> None:
        validated = FramePacket.from_mapping(packet)
        with self._condition:
            if validated.camera_id in self._packets:
                self.metrics.replaced_stale += 1
            else:
                self._order.append(validated.camera_id)
            self._packets[validated.camera_id] = packet
            self.metrics.submitted += 1
            self._condition.notify()

    def get(self, timeout: float | None = None) -> Mapping[str, Any] | None:
        with self._condition:
            if not self._order:
                self._condition.wait(timeout)
            if not self._order:
                return None
            camera_id = self._order.popleft()
            return self._packets.pop(camera_id)

    def __len__(self) -> int:
        with self._condition:
            return len(self._packets)

class Tier2Runtime:
    """Optional local worker bridge between Tier 1 FramePackets and an event sink.

    The sink is intentionally a callback.  Tier 3 can supply an HTTP publisher,
    a local function, or a test collector without making this AI repository own
    a FastAPI server or WebSocket implementation.
    """
    def __init__(self, pipeline, event_sink: Callable[[dict[str, Any]], None] | None = None):
        self.pipeline, self.event_sink = pipeline, event_sink
        self.buffer = LatestFrameBuffer()
        self.metrics = self.buffer.metrics
        self._stop = Event()
        self._thread: Thread | None = None

    def submit(self, packet: Mapping[str, Any]) -> None:
        self.buffer.put(packet)

    def process_once(self, timeout: float | None = None) -> list[dict[str, Any]]:
        packet = self.buffer.get(timeout)
        if packet is None:
            return []
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

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._loop, name="tier2-frame-worker", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.process_once(timeout=0.1)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def health(self) -> dict[str, Any]:
        return {"queued_cameras": len(self.buffer), "worker_running": bool(self._thread and self._thread.is_alive()), **self.metrics.__dict__, "pipeline": self.pipeline.health()}
