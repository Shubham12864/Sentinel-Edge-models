"""Tier 1 tests: contract conformance, pacing, reconnect, lifecycle."""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

T1 = Path(__file__).resolve().parents[1] / "tier1-ingest"
T2 = Path(__file__).resolve().parents[2] / "Sentinel Edge models" / "tier2-ai"
for p in (str(T1), str(T2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from camera_source import CAMERA_ID_RE, CameraSource  # noqa: E402


class FakeCapture:
    """Scriptable capture: yields N good frames then fails forever."""

    def __init__(self, good_frames=3):
        self.remaining = good_frames
        self.opened = True
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if self.remaining > 0:
            self.remaining -= 1
            return True, np.zeros((8, 8, 3), np.uint8)
        return False, None

    def release(self):
        self.released = True


def test_camera_id_regex():
    assert CAMERA_ID_RE.fullmatch("CAM_01")
    assert CAMERA_ID_RE.fullmatch("LAPTOP_CAM.a-9")
    assert not CAMERA_ID_RE.fullmatch("../evil")
    assert not CAMERA_ID_RE.fullmatch("x" * 65)


def test_packet_contract_exact_fields(tmp_path):
    got = []
    src = CameraSource("CAM_T", "x.avi", got.append, fps=50,
                       capture_factory=lambda: FakeCapture(good_frames=3))
    src.start(); src.join(timeout=5)
    src.stop()
    assert len(got) >= 3          # file sources LOOP: factory reopens -> more batches
    for packet in got:
        assert set(packet) == {"camera_id", "timestamp", "frame", "metadata"}
        assert packet["camera_id"] == "CAM_T"
        time.strptime(packet["timestamp"], "%Y-%m-%dT%H:%M:%SZ")  # ISO UTC shape
        assert packet["frame"].shape == (8, 8, 3)
        assert isinstance(packet["metadata"], dict)
    assert src.frames_captured == len(got)


def test_nonlooping_file_ends_cleanly():
    got = []
    src = CameraSource("CAM_S", "x.avi", got.append, fps=100,
                       loop_file=False, max_consecutive_failures=2,
                       capture_factory=lambda: FakeCapture(good_frames=4))
    src.start(); src.join(timeout=6)
    src.stop()
    assert len(got) == 4 and src.stream_ended is True


def test_invalid_camera_id_rejected_at_construction():
    with pytest.raises(ValueError):
        CameraSource("../evil", 0, lambda p: None)


def test_sink_exception_does_not_kill_source():
    def hostile_sink(packet):
        raise RuntimeError("track2 exploded")
    src = CameraSource("CAM_H", "x.avi", hostile_sink, fps=100,
                       max_consecutive_failures=2, loop_file=True,
                       capture_factory=lambda: FakeCapture(good_frames=4))
    src.start()
    deadline = time.time() + 6
    while src.frames_captured < 4 and time.time() < deadline:
        time.sleep(0.02)
    src.stop()
    assert src.frames_captured >= 4, "sink exception killed ingestion"
    assert src.last_error and src.last_error.startswith("sink:")


def test_reconnect_after_sustained_failure():
    made = {"n": 0}

    class Factory:
        def __call__(self):
            made["n"] += 1
            return FakeCapture(good_frames=1)

    src = CameraSource("CAM_R", "rtsp://x", lambda p: None, fps=200,
                       reconnect_delay=0.05, max_consecutive_failures=2,
                       capture_factory=Factory())
    src.start()
    deadline = time.time() + 6
    while made["n"] < 3 and time.time() < deadline:
        time.sleep(0.02)
    src.stop()
    assert made["n"] >= 3, f"no reconnects happened (opens={made['n']})"
    assert src.reconnects >= 1


def test_fps_pacing_bounds_rate():
    got = []
    src = CameraSource("CAM_F", "x.avi", got.append, fps=20,
                       capture_factory=lambda: FakeCapture(good_frames=40))
    src.start()
    time.sleep(0.5)
    src.stop()
    assert 2 <= len(got) <= 25, f"pacing broken: {len(got)} frames in 0.5s @20fps"


def test_ingest_manager_lifecycle():
    from ingest_manager import IngestManager
    mgr = IngestManager()
    got = []
    r = mgr.add_camera("CAM_A", "a.avi", got.append, fps=60,
                       capture_factory=lambda: FakeCapture(good_frames=5000))
    assert r["ok"]
    assert mgr.add_camera("CAM_A", "a.avi", got.append)["ok"] is False  # duplicate blocked
    bad = mgr.add_camera("../evil", 0, got.append)
    assert bad["ok"] is False
    stats = {s["camera_id"]: s for s in mgr.stats()}
    assert stats["CAM_A"]["alive"] is True
    assert mgr.remove_camera("CAM_A") and not mgr.remove_camera("CAM_A")
    mgr.stop_all()
