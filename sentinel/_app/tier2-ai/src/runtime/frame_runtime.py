"""Bounded latest-frame execution with one worker thread per camera.

Tier 1 owns camera decoding.  This module only accepts already-created
FramePackets and deliberately retains at most one unprocessed frame per
camera, preventing stale-video backlog and unbounded memory growth.  Each
active camera gets its own worker thread, so a slow or busy camera can no
longer starve the others behind a single shared loop.

Resource bounds (hardened after adversarial audit):
- at most ``max_workers`` live worker threads exist at any moment; further
  cameras wait pending until a slot frees (their latest frame is retained),
- at most ``max_cameras`` distinct camera ids are accepted over the
  runtime's lifetime; extras are counted in ``metrics.rejected_cameras``,
- dead worker threads are pruned from the registry instead of accumulating,
- event-sink failures record ``metrics.last_sink_error`` beside the counter.
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
    rejected_cameras: int = 0
    last_sink_error: str | None = None

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
    loop.  Worker creation is bounded by ``max_workers`` and total accepted
    cameras by ``max_cameras`` — a hostile or buggy Track 1 flooding arbitrary
    camera ids can degrade to rejections, never to unbounded thread growth.
    The sink is intentionally a callback: Tier 3 can supply an HTTP publisher,
    a local function, or a test collector without making this AI repository
    own a FastAPI server or WebSocket implementation.
    """

    POLL_SECONDS = 0.1

    def __init__(self, pipeline, event_sink: Callable[[dict[str, Any]], None] | None = None,
                 max_workers: int = 8, max_cameras: int = 64):
        self.pipeline, self.event_sink = pipeline, event_sink
        self.max_workers, self.max_cameras = max(1, int(max_workers)), max(1, int(max_cameras))
        self.buffer = LatestFrameBuffer()
        self.metrics = self.buffer.metrics
        self._stop = Event()
        self._threads: dict[str, Thread] = {}
        self._workers_lock = Lock()
        self._registered_cameras: set[str] = set()

    # ------------------------------------------------------------ internals --
    def _prune_registry_locked(self) -> None:
        """Drop dead Thread objects so the registry cannot grow forever."""
        dead = [camera_id for camera_id, thread in self._threads.items() if not thread.is_alive()]
        for camera_id in dead:
            del self._threads[camera_id]

    def _alive_workers_locked(self) -> int:
        return sum(1 for thread in self._threads.values() if thread.is_alive())

    def _spawn_worker(self, camera_id: str) -> bool:
        """Spawn this camera's worker if allowed; returns True when spawned."""
        with self._workers_lock:
            self._prune_registry_locked()
            existing = self._threads.get(camera_id)
            if existing is not None and existing.is_alive():
                return False
            if self._stop.is_set():
                raise RuntimeError("Tier2Runtime is stopped; call start() before submitting packets")
            if self._alive_workers_locked() >= self.max_workers:
                return False  # held pending; retried on future submissions
            thread = Thread(target=self._camera_loop, args=(camera_id,), name=f"tier2-worker-{camera_id}", daemon=True)
            self._threads[camera_id] = thread
        thread.start()
        return True

    def _spawn_for_starved_cameras(self) -> None:
        """Retry workers for cameras whose latest frame waits without a worker."""
        pending = self.buffer.pending_cameras()
        for camera_id in pending:
            with self._workers_lock:
                if self._alive_workers_locked() >= self.max_workers:
                    return
                thread = self._threads.get(camera_id)
            if thread is None or not thread.is_alive():
                try:
                    self._spawn_worker(camera_id)
                except RuntimeError:
                    return  # runtime stopped mid-drain

    # ------------------------------------------------------------- public ----
    def submit(self, packet: Mapping[str, Any]) -> None:
        validated = FramePacket.from_mapping(packet)
        camera_id = validated.camera_id
        with self._workers_lock:
            if camera_id not in self._registered_cameras and len(self._registered_cameras) >= self.max_cameras:
                self.metrics.rejected_cameras += 1
                return
            self._registered_cameras.add(camera_id)
        self.buffer.put(packet)
        self._spawn_worker(camera_id)
        self._spawn_for_starved_cameras()

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
                except Exception as exc:
                    self.metrics.sink_errors += 1
                    self.metrics.last_sink_error = f"{type(exc).__name__}: {exc}"
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
        with self._workers_lock:
            self._prune_registry_locked()

    def health(self) -> dict[str, Any]:
        with self._workers_lock:
            active_workers = sum(1 for thread in self._threads.values() if thread.is_alive())
        return {"queued_cameras": len(self.buffer), "worker_running": active_workers > 0,
                "active_workers": active_workers, "max_workers": self.max_workers,
                "registered_cameras": len(self._registered_cameras), "max_cameras": self.max_cameras,
                **self.metrics.__dict__, "pipeline": self.pipeline.health()}
