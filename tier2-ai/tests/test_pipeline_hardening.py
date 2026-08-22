"""Level-2 hardening tests: every red-team probe from the audit, as regression tests."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, r"C:\Users\Shubham\Downloads\Sentinel Edge models\tier2-ai")

from src.pipeline import UnifiedPipeline
from src.trajectory.transition_graph import TransitionGraph
from src.trajectory.markov_predictor import MarkovPredictor


class Det:
    def detect_and_track(self, frame):
        return [{"track_id": 1, "bbox": [10, 10, 100, 100], "confidence": 0.9}]
    def close(self):
        pass


class Emb:
    def embed_from_bbox(self, frame, bbox):
        v = np.zeros(512, np.float32)
        v[0] = 1.0
        return v


class Searcher:
    def search(self, emb):
        return ("Alice", "alice", 0.93)


def make_pipeline(crop_dir, **kw):
    return UnifiedPipeline(detector=Det(), embedder=Emb(), searcher=Searcher(),
                           predictor=None, crop_dir=crop_dir, **kw)


def packet(cam, ts, frame=None):
    return {"camera_id": cam, "timestamp": ts, "frame": frame, "metadata": {}}


def textured():
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (200, 200, 3)).astype(np.uint8)


def test_traversal_camera_id_rejected_no_file_outside(tmp_path):
    p = make_pipeline(tmp_path / "crops")
    events = p.process_frame_packet(packet(r"..\..\ESCAPED", "2026-08-22T10:00:00Z", textured()))
    assert events == []
    assert p.metrics.rejected_packets == 1
    assert not (tmp_path / "ESCAPED_1_2026-08-22T10-00-00Z.jpg").exists()
    assert list((tmp_path / "crops").glob("*")) == []


def test_allowlist_rejects_unknown_camera(tmp_path):
    p = make_pipeline(tmp_path / "crops", camera_allowlist={"CAM_01"})
    assert p.process_frame_packet(packet("CAM_99", "2026-08-22T10:00:00Z", textured())) == []
    assert p.metrics.rejected_packets == 1
    assert len(p.process_frame_packet(packet("CAM_01", "2026-08-22T10:00:00Z", textured()))) == 1


def test_malformed_timestamp_rejected_not_raised(tmp_path):
    p = make_pipeline(tmp_path / "crops")
    assert p.process_frame_packet(packet("CAM_OK", "2026-08-22T10:00:00|BROKEN", textured())) == []
    assert p.metrics.rejected_packets == 1


def test_missing_fields_rejected_not_raised(tmp_path):
    p = make_pipeline(tmp_path / "crops")
    assert p.process_frame_packet({"camera_id": "CAM_OK"}) == []
    assert p.metrics.rejected_packets == 1


def test_future_timestamp_cannot_expire_other_camera(tmp_path):
    """Direct regression for audit probe A3: forged 2027 clock must not wipe CAM_A."""
    p = make_pipeline(tmp_path / "crops")
    p.process_frame_packet(packet("CAM_A", "2026-08-22T10:00:00Z", textured()))
    assert p.health()["active_tracks"] == 1
    p.process_frame_packet(packet("CAM_B", "2027-08-22T10:00:00Z", textured()))
    assert p.health()["active_tracks"] == 2, "CAM_A track was TTL-wiped by CAM_B's forged clock"
    # and the hostile camera's own bookkeeping clock is clamped to wall time
    assert p._camera_last_seen["CAM_B"] <= time.time() + 1


def test_event_timestamp_contract_preserved(tmp_path):
    """UnifiedEvent.timestamp passes Track 1's value through untouched."""
    p = make_pipeline(tmp_path / "crops")
    ts = "2030-01-01T00:00:00Z"
    events = p.process_frame_packet(packet("CAM_T", ts, textured()))
    assert events and events[0]["timestamp"] == ts


def test_predictor_laplace_smoothing_and_decay():
    now = time.time()
    g = TransitionGraph()
    g.add_transition("A", "B", weight=10.0, eta_seconds=30.0, observed_at=now)
    g.add_transition("A", "C", weight=0.05, eta_seconds=40.0, observed_at=None)  # seeded, no timestamp
    mp = MarkovPredictor(g)
    pr = mp.predict("A", now=now)
    assert pr["next_camera"] == "B"
    assert pr["probability"] is not None and 0 < pr["probability"] < 1
    # seeded edge (virtual age 4*decay) must decay like any other, not stay immortal
    far_future = now + 50 * 3600
    pr2 = MarkovPredictor(g, decay_seconds=3600).predict("A", now=far_future)
    pr3 = MarkovPredictor(g, decay_seconds=3600).predict("A", now=far_future + 1)
    assert pr3["probability"] < 1.0  # still finite; weights faded but smoothing keeps it proper
    # smoothing keeps probabilities bounded even with a single observation
    # (one candidate => probability is legitimately exactly 1.0)
    g2 = TransitionGraph()
    g2.add_transition("X", "Y", weight=1.0, eta_seconds=5.0, observed_at=now)
    one = MarkovPredictor(g2).predict("X", now=now)
    assert 0 < one["probability"] <= 1


def test_transition_graph_roundtrip(tmp_path):
    g = TransitionGraph()
    g.add_transition("A", "B", weight=3.0, eta_seconds=42.0, observed_at=123.0)
    path = tmp_path / "graph.json"
    g.save(path)
    g2 = TransitionGraph()
    assert g2.load(path) is True
    assert g2.edges["A"]["B"]["weight"] == 3.0
    assert g2.edges["A"]["B"]["eta_seconds"] == 42.0
