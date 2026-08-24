"""Tier 3 tests: contract fan-in, camera CRUD validation, identity upload, MJPEG."""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

T3 = Path(__file__).resolve().parent.parent / "tier3-hq"
sys.path.insert(0, str(T3))

import server  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # isolated stores per test
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    backup = server.GALLERY_JSON.read_bytes() if server.GALLERY_JSON.exists() else None
    server.GALLERY_JSON.unlink(missing_ok=True)   # keep live enrollments out of unit tests
    store = server.IdentityStore(index_path=tmp_path / "index.faiss")
    store.gallery_json = tmp_path / "gallery.json"
    store.entries = []
    store._rebuild_searcher()
    monkeypatch.setattr(server, "IDENTITY_STORE", store)
    hub = server.Hub()
    monkeypatch.setattr(server, "HUB", hub)
    c = TestClient(server.app)
    yield c, hub, store
    if backup is not None:                        # restore the live gallery byte-for-byte
        server.GALLERY_JSON.write_bytes(backup)


def full_event(name=None):
    return {"track_id": "7", "identity_name": name, "identity_id": name,
            "match_score": 0.93 if name else 0.31, "verified": bool(name),
            "current_camera": "CAM_01", "timestamp": "2026-08-22T12:00:00Z",
            "face_crop_url": None, "next_camera": None, "eta_seconds": None,
            "transition_probability": None}


def test_event_sink_contract_and_fanout(client):
    c, hub, _ = client
    r = c.post("/api/test-event", json=full_event("Alice"))
    assert r.status_code == 200 and r.json()["ok"]
    events = c.get("/api/events").json()["events"]
    assert len(events) == 1 and events[0]["identity_name"] == "Alice"
    assert set(events[0]) == server.EVENT_FIELDS | {"_received_at"}


def test_malformed_event_rejected(client):
    c, _, _ = client
    bad = {"hello": 1}
    r = c.post("/api/test-event", json=bad)
    assert r.status_code == 422
    assert c.get("/api/events").json()["events"] == []


def test_camera_crud_validation(client):
    c, hub, _ = client
    ok = c.post("/api/cameras/save", json={"camera_id": "CAM_09", "source": "../runs/x.avi",
                                           "fps": 4, "location": "gate"})
    assert ok.status_code == 200 and ok.json()["state"] == "saved"
    rows = c.get("/api/cameras").json()["cameras"]
    row = next(r for r in rows if r["camera_id"] == "CAM_09")
    assert row["connected"] is False and row["status"] == "saved"   # saved ≠ connected
    for body in [{"camera_id": "../evil", "source": "x"},
                 {"camera_id": "", "source": "x"},
                 {"camera_id": "OK", "source": ""},
                 {"camera_id": "OK", "source": "x", "fps": 99},
                 {"camera_id": "x" * 65, "source": "y"}]:
        assert c.post("/api/cameras/save", json=body).status_code == 422, body
    # connect requires a real ingest bridge; with none set it stays 'connecting'
    assert c.post("/api/cameras/CAM_09/connect").status_code == 200
    assert c.post("/api/cameras/GHOST/connect").status_code == 404
    assert c.post("/api/cameras/CAM_09/disconnect").json()["state"] == "saved"
    assert c.delete("/api/cameras/CAM_09").status_code == 200
    assert c.delete("/api/cameras/CAM_09").status_code == 404


def test_identity_upload_enrollment_flow(client, monkeypatch):
    c, hub, store = client
    monkeypatch.setattr(store, "_ensure_models", lambda: None)   # no real models in unit test

    class FakeEmbedder:
        calls = 0
        def embed_image(self, image):
            FakeEmbedder.calls += 1
            v = np.zeros(512, np.float32)
            v[FakeEmbedder.calls % 3] = 1.0     # 2 similar + 1 different
            return v
    monkeypatch.setattr(store, "_embedder", FakeEmbedder())
    saved = {}
    monkeypatch.setattr(server.IdentityStore, "export_index", lambda self: saved.update(n=1))

    img = np.zeros((60, 60, 3), np.uint8)
    import cv2
    ok, buf = cv2.imencode(".jpg", img)
    files = [("files", ("a.jpg", buf.tobytes(), "image/jpeg")),
             ("files", ("b.jpg", buf.tobytes(), "image/jpeg")),
             ("files", ("c.txt", b"not an image", "text/plain"))]
    r = c.post("/api/identities/upload", data={"name": "Bob"}, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == 2 and body["uploaded"] == 2
    listing = c.get("/api/identities").json()["identities"]
    assert listing == [{"identity_id": "Bob", "name": "Bob", "images": 2}]

    # removal works and rebuilds
    assert c.delete("/api/identities/Bob").json()["removed_entries"] == 2
    assert c.get("/api/identities").json()["identities"] == []
    assert c.delete("/api/identities/Bob").status_code == 404


Bob_name = "Bob"


def test_bad_name_and_garbage_upload_rejected(client):
    c, hub, store = client
    r = c.post("/api/identities/upload", data={"name": "../evil"},
               files=[("files", ("a.jpg", b"x", "image/jpeg"))])
    assert r.status_code == 422
    import cv2
    ok, buf = cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))
    r = c.post("/api/identities/upload", data={"name": "Ghost"},
               files=[("files", ("a.png", b"\x00\x01\x02notpng", "image/png"))])
    assert r.status_code == 422          # nothing decodable


def test_stats_shape(client):
    c, hub, _ = client
    s = c.get("/api/stats").json()
    for key in ("cameras", "live_cameras", "events_total", "events_recent_60s",
                "frames_ingested", "sockets", "identities", "uptime_s"):
        assert key in s


def test_index_page_served(client):
    c, _, _ = client
    html = c.get("/").text
    assert "SENTINEL" in html.upper() and "Enroll" in html or "Enroll" in html
