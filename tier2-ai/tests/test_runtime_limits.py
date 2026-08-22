"""Level-3 tests: runtime resource bounds (thread-bomb containment)."""
import sys
import threading
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Shubham\Downloads\Sentinel Edge models\tier2-ai")

from src.runtime.frame_runtime import Tier2Runtime


class FastPipe:
    def process_frame_packet(self, packet):
        return []
    def health(self):
        return {}


def packet(cam):
    rng = np.random.default_rng(1)
    return {"camera_id": cam, "timestamp": "2026-08-22T10:00:00Z",
            "frame": rng.integers(0, 255, (32, 32, 3)).astype(np.uint8), "metadata": {}}


def test_thread_bomb_contained():
    rt = Tier2Runtime(FastPipe(), max_workers=4, max_cameras=500)
    for i in range(300):
        rt.submit(packet(f"CAM_{i:04d}"))
    time.sleep(0.5)
    live = sum(1 for t in rt._threads.values() if t.is_alive())
    assert live <= 4, f"{live} live threads > max_workers=4"
    assert len(rt._threads) <= 8, "registry grew without bound"
    assert rt.metrics.rejected_cameras == 0
    rt.stop()
    # pending cameras get workers as slots free -> everything drains eventually


def test_camera_ceiling_rejects_extras():
    rt = Tier2Runtime(FastPipe(), max_workers=2, max_cameras=3)
    for i in range(10):
        rt.submit(packet(f"CEIL_{i}"))
    assert rt.metrics.rejected_cameras == 7
    assert len(rt._registered_cameras) == 3
    rt.stop()


def test_sink_error_recorded():
    def bad_sink(event):
        raise ValueError("sink down")
    class OneEventPipe(FastPipe):
        def process_frame_packet(self, packet):
            return [{"verified": True}]
    rt = Tier2Runtime(OneEventPipe(), event_sink=bad_sink)
    rt.process_packet({"camera_id": "C", "timestamp": "2026-08-22T10:00:00Z",
                       "frame": np.zeros((4, 4, 3), np.uint8), "metadata": {}})
    assert rt.metrics.sink_errors == 1
    assert "ValueError" in (rt.metrics.last_sink_error or "")
    rt.stop()


def test_stop_prunes_registry_and_health_reports_bounds():
    rt = Tier2Runtime(FastPipe(), max_workers=3)
    for i in range(6):
        rt.submit(packet(f"H_{i}"))
    rt.stop(timeout=5)
    time.sleep(0.2)
    health = rt.health()
    assert health["active_workers"] == 0
    assert len(rt._threads) <= 3 or all(not t.is_alive() for t in rt._threads.values())
    assert health["max_workers"] == 3 and health["max_cameras"] == 64
