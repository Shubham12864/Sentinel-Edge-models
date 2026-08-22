"""Smoke tests for Steps 2+3: recognition-only embedder + per-camera runtime."""
import sys
import threading
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Shubham\Downloads\Sentinel Edge models\tier2-ai")

from src.embedding.face_embedder import FaceEmbedder
from src.pipeline import UnifiedPipeline
from src.runtime.device import resolve_device
from src.runtime.frame_runtime import LatestFrameBuffer, Tier2Runtime


# ------------------------------------------------------------- device plan ---
def test_resolve_device_full_cpu(monkeypatch):
    import src.runtime.device as dev
    monkeypatch.setattr(dev, "_torch_cuda", lambda: False)
    monkeypatch.setattr(dev, "_onnx_cuda", lambda: False)
    assert dev.resolve_device() == dev.DevicePlan("cpu", -1, False, False)
    forced = dev.resolve_device("cpu")            # works even without GPU
    assert forced.torch_device == "cpu" and forced.ctx_id == -1
    import pytest
    with pytest.raises(RuntimeError):             # cuda demanded but absent
        dev.resolve_device("cuda")


def test_resolve_device_partial_and_auto(monkeypatch):
    import src.runtime.device as dev
    monkeypatch.setattr(dev, "_torch_cuda", lambda: True)
    monkeypatch.setattr(dev, "_onnx_cuda", lambda: False)   # torch-only box
    plan = dev.resolve_device()
    assert (plan.torch_device, plan.ctx_id) == ("cuda", -1)
    monkeypatch.setattr(dev, "_torch_cuda", lambda: False)
    monkeypatch.setattr(dev, "_onnx_cuda", lambda: True)    # onnxruntime-gpu only
    plan = dev.resolve_device()
    assert (plan.torch_device, plan.ctx_id) == ("cpu", 0)
    assert dev.resolve_device("cpu").ctx_id == -1           # explicit override wins


# ---------------------------------------------------------------- Step 2 ---
class FakeRecognizer:
    input_shape = (1, 3, 112, 112)
    calls = 0
    def get_feat(self, img):
        FakeRecognizer.calls += 1
        assert img.shape == (112, 112, 3), img.shape
        return np.full((1, 512), 0.5, dtype=np.float32)


class BoomDetector:
    def detect(self, *a, **k):
        raise AssertionError("runtime embedding must NOT run SCRFD detection")


class FakeFaceAnalysis:
    def __init__(self, detection=None):
        self.models = {"detection": detection if detection is not None else BoomDetector(),
                       "recognition": FakeRecognizer()}


def make_embedder(detection=None):
    emb = FaceEmbedder()
    emb.model = FakeFaceAnalysis(detection)
    return emb


def test_expand_bbox_margin_and_clipping():
    emb = make_embedder()
    assert emb._expand_bbox([50, 50, 150, 150], 200, 200) == (38, 38, 162, 162)
    assert emb._expand_bbox([-20, -20, 60, 60], 200, 200)[0] == 0      # clipped left/top
    assert emb._expand_bbox([180, 180, 260, 260], 200, 200)[2] == 200  # clipped right/bottom
    assert emb._expand_bbox([0, 0, 1, 1], 200, 200) is None            # degenerate -> rejected


def test_embed_from_bbox_is_recognition_only_and_normalised():
    FakeRecognizer.calls = 0
    emb = make_embedder()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    vec = emb.embed_from_bbox(frame, [50, 50, 150, 150])
    assert vec is not None and vec.shape == (512,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    assert FakeRecognizer.calls == 1                                   # exactly one pass
    assert emb.embed_from_bbox(None, [0, 0, 9, 9]) is None
    assert emb.embed_crop(None) is None


def test_embed_image_uses_exactly_one_detection_pass():
    class Det:
        calls = 0
        def detect(self, image, max_num=1):
            Det.calls += 1
            return np.array([[30, 30, 90, 90, 0.99]]), None

    class NoFace:
        def detect(self, image, max_num=1):
            return np.zeros((0, 5)), None

    FakeRecognizer.calls = 0
    emb = make_embedder(detection=Det())
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    vec = emb.embed_image(img)
    assert vec is not None and abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    assert Det.calls == 1 and FakeRecognizer.calls == 1
    emb.model.models["detection"] = NoFace()
    assert emb.embed_image(np.zeros((10, 10, 3), dtype=np.uint8)) is None


# ---------------------------------------------------------------- buffers ---
def make_packet(camera_id, timestamp="2026-01-01T00:00:00Z", frame=None):
    return {"camera_id": camera_id, "timestamp": timestamp,
            "frame": np.zeros((4, 4, 3), np.uint8) if frame is None else frame,
            "metadata": {}}


def test_buffer_keeps_latest_per_camera():
    buf = LatestFrameBuffer()
    assert buf.put(make_packet("A")) == "A"
    assert buf.put(make_packet("B")) == "B"
    assert buf.put(make_packet("B")) == "B"          # overwrite unconsumed -> stale
    assert buf.metrics.replaced_stale == 1
    assert sorted(buf.pending_cameras()) == ["A", "B"]
    assert buf.pop("B", timeout=0.2)["camera_id"] == "B"
    assert buf.pop("B", timeout=0.05) is None
    assert buf.pop("A", timeout=0.2)["camera_id"] == "A"
    assert len(buf) == 0


# ------------------------------------------ Step 3: parallelism + coalescing ---
class GatePipeline:
    """Blocks every call inside process_frame_packet until released."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.processed_cameras = []
        self.release = threading.Event()

    def process_frame_packet(self, packet):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            assert self.release.wait(timeout=5.0), "gate never released"
        finally:
            with self.lock:
                self.active -= 1
                self.processed_cameras.append(packet["camera_id"])
        return []

    def health(self):
        return {}


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_runtime_processes_two_cameras_simultaneously():
    pipe = GatePipeline()
    rt = Tier2Runtime(pipe)
    rt.submit(make_packet("CAM-A"))
    rt.submit(make_packet("CAM-B"))
    assert wait_until(lambda: pipe.active >= 2), "second camera waited behind the first"
    assert pipe.max_active == 2                      # definitive concurrency proof
    pipe.release.set()
    assert wait_until(lambda: len(pipe.processed_cameras) == 2)
    assert sorted(pipe.processed_cameras) == ["CAM-A", "CAM-B"]
    assert rt.health()["active_workers"] == 2
    assert rt.metrics.processed == 2
    rt.stop()
    assert wait_until(lambda: rt.health()["active_workers"] == 0)


class SlowPipeline:
    def process_frame_packet(self, packet):
        time.sleep(0.05)
        return []
    def health(self):
        return {}


def test_burst_submits_coalesce_to_latest_frame():
    rt = Tier2Runtime(SlowPipeline())
    for _ in range(6):
        rt.submit(make_packet("C1"))
    rt.stop()                                        # joins workers -> counters settled
    accounted = rt.metrics.processed + rt.metrics.replaced_stale + len(rt.buffer)
    assert accounted == 6                            # capacity-one accounting holds exactly
    assert rt.metrics.submitted == 6
    assert rt.metrics.replaced_stale >= 1            # intermediate frames dropped by design


# ------------------------------------------- end-to-end pipeline under load ---
class FakeDetector:
    def detect_and_track(self, frame):
        if int(frame[0, 0, 2]) == 0:
            return []
        return [{"track_id": 7, "bbox": [40, 40, 120, 120], "confidence": 0.95}]
    def close(self):
        pass


class FakeEmbedder:
    def __init__(self):
        self.calls = 0
    def embed_from_bbox(self, frame, bbox):
        self.calls += 1
        v = np.zeros(512, dtype=np.float32)
        v[0] = 1.0
        return v


class FakeSearcher:
    def search(self, emb):
        return ("Alice", "alice-id", 0.93)


def textured_frame():
    rng = np.random.default_rng(7)                   # texture keeps Laplacian variance high
    frame = rng.integers(0, 255, size=(160, 160, 3)).astype(np.uint8)
    frame[0, 0, 2] = 255                             # detector enable-marker
    return frame


def test_pipeline_end_to_end_two_cameras_verified_events(tmp_path):
    pipe = UnifiedPipeline(detector=FakeDetector(), embedder=FakeEmbedder(),
                           searcher=FakeSearcher(), predictor=None,
                           crop_dir=tmp_path / "crops")
    events = []
    rt = Tier2Runtime(pipe, event_sink=events.append)
    for idx, cam in enumerate(("C1", "C2")):
        rt.submit({"camera_id": cam, "timestamp": f"2026-01-01T00:00:{idx:02d}Z",
                   "frame": textured_frame(), "metadata": {}})
    assert wait_until(lambda: len(events) >= 2), events
    rt.stop()
    assert len(events) == 2
    for ev in events:
        assert ev["verified"] is True
        assert ev["identity_name"] == "Alice"
        assert ev["identity_id"] == "alice-id"
        assert ev["track_id"] == "7"
    assert {ev["current_camera"] for ev in events} == {"C1", "C2"}
    health = pipe.health()
    assert health["packets_processed"] == 2
    assert health["events_emitted"] == 2
    assert health["active_tracks"] == 2
    assert len(list((tmp_path / "crops").glob("*.jpg"))) == 2   # best-crop persisted


def test_pipeline_event_throttling_still_suppresses_duplicates(tmp_path):
    pipe = UnifiedPipeline(detector=FakeDetector(), embedder=FakeEmbedder(),
                           searcher=FakeSearcher(), predictor=None,
                           crop_dir=tmp_path / "crops")
    frame = textured_frame()
    first = pipe.process_frame_packet({"camera_id": "C9",
                                       "timestamp": "2026-01-01T00:00:00Z",
                                       "frame": frame, "metadata": {}})
    assert len(first) == 1 and first[0]["verified"] is True
    second = pipe.process_frame_packet({"camera_id": "C9",
                                        "timestamp": "2026-01-01T00:00:01Z",
                                        "frame": frame, "metadata": {}})
    assert second == []                              # within 2s interval, nothing changed
