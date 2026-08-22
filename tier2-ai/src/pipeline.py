"""Tier 2 orchestration: validated FramePackets in, UnifiedEvents out."""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
import threading
import time
import cv2
from .schemas import FramePacket, UnifiedEvent
from .quality_gate.blur_size_filter import laplacian_variance, passes_quality_gate
from .verification.match_verifier import MatchVerifier

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

@dataclass
class TrackState:
    raw_name: str | None = None
    raw_identity_id: str | None = None
    score: float = 0.0
    verified: bool = False
    best_quality: float = 0.0
    crop_url: str | None = None
    last_seen_epoch: float = 0.0
    last_event_epoch: float = 0.0

@dataclass
class PipelineMetrics:
    packets_processed: int = 0
    detections_seen: int = 0
    quality_rejections: int = 0
    events_emitted: int = 0
    processing_errors: int = 0
    expired_tracks: int = 0
    rejected_packets: int = 0

class UnifiedPipeline:
    """Long-lived, camera-safe Tier 2 pipeline.

    A detector factory is used in normal operation because persistent ByteTrack
    state is local to a camera.  Tests may inject one fixed detector for a
    single camera.
    """
    def __init__(self, detector=None, tracker=None, embedder=None, verifier=None, searcher=None,
                 predictor=None, crop_dir: str | Path | None = None,
                 detector_factory: Callable[[], Any] | None = None,
                 track_ttl_seconds: float = 300.0, event_interval_seconds: float = 2.0,
                 crop_improvement_ratio: float = 1.10, transition_window_seconds: float = 600.0,
                 camera_allowlist: set[str] | frozenset[str] | None = None,
                 transition_graph_path: str | Path | None = None):
        self.detector, self.tracker, self.embedder = detector, tracker, embedder
        self.detector_factory = detector_factory
        self.verifier, self.searcher, self.predictor = verifier or MatchVerifier(), searcher, predictor
        self.crop_dir = Path(crop_dir) if crop_dir else Path(__file__).resolve().parents[1] / "data" / "face_crops"
        self.track_ttl_seconds, self.event_interval_seconds = track_ttl_seconds, event_interval_seconds
        self.crop_improvement_ratio, self.transition_window_seconds = crop_improvement_ratio, transition_window_seconds
        self.camera_allowlist = frozenset(camera_allowlist) if camera_allowlist else None
        self.transition_graph_path = (
            Path(transition_graph_path) if transition_graph_path is not None
            else Path(__file__).resolve().parents[1] / "data" / "transition_graph.json")
        self.gallery_watch_path: str | Path | None = None  # set by create_default
        self.track_states: dict[tuple[str, int], TrackState] = {}
        self._camera_detectors: dict[str, Any] = {}
        self._camera_last_seen: dict[str, float] = {}
        self._identity_last_seen: dict[str, tuple[str, float]] = {}
        self._lock = threading.RLock()
        self.metrics = PipelineMetrics()

    @classmethod
    def create_default(cls, *, weights_path: str | Path | None = None, identity_index_path: str | Path | None = None,
                       crop_dir: str | Path | None = None, ctx_id: int | None = None, device: str | int | None = None,
                       camera_allowlist: set[str] | frozenset[str] | None = None,
                       transition_graph_path: str | Path | None = None, **kwargs) -> "UnifiedPipeline":
        """Build the complete production pipeline once, without opening any camera.

        device=None auto-detects: GPU stages run on CUDA when torch / onnxruntime
        expose it (e.g. an RTX laptop), everything falls back to CPU otherwise.
        device='cpu' forces CPU; device='cuda' requires a GPU.  An explicit
        ctx_id overrides the embedder's derived context id.
        camera_allowlist=None auto-loads camera ids from config/cameras.yaml
        when that file defines a non-empty top-level 'cameras:' key.
        The Markov transition graph is restored from data/transition_graph.json
        when present and persisted there as transitions are observed.
        """
        import yaml
        from .runtime.device import resolve_device
        from .detection.face_detector import FaceDetector
        from .embedding.face_embedder import FaceEmbedder
        from .identity_search.faiss_index import IdentitySearch
        from .trajectory.transition_graph import TransitionGraph
        from .trajectory.markov_predictor import MarkovPredictor
        plan = resolve_device(device)
        effective_ctx = plan.ctx_id if ctx_id is None else ctx_id
        searcher = IdentitySearch()
        if identity_index_path is not None:
            searcher.load(identity_index_path)
        if camera_allowlist is None:
            config_path = Path(__file__).resolve().parents[1] / "config" / "cameras.yaml"
            try:
                if config_path.exists():
                    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                    cameras = data.get("cameras") or {}
                    names = list(cameras) if isinstance(cameras, dict) else [c for c in cameras if c]
                    camera_allowlist = {str(name) for name in names} or None
            except Exception:
                camera_allowlist = None  # a broken config must never block startup
        graph_path = Path(transition_graph_path) if transition_graph_path is not None else (
            Path(__file__).resolve().parents[1] / "data" / "transition_graph.json")
        graph = TransitionGraph()
        try:
            graph.load(graph_path)
        except Exception:
            pass  # missing/corrupt history just means a fresh graph
        pipeline = cls(embedder=FaceEmbedder(ctx_id=effective_ctx), searcher=searcher,
                       predictor=MarkovPredictor(graph), crop_dir=crop_dir,
                       camera_allowlist=camera_allowlist, transition_graph_path=graph_path,
                       detector_factory=lambda: FaceDetector(weights_path=weights_path, device=plan.torch_device), **kwargs)
        pipeline.gallery_watch_path = identity_index_path
        return pipeline

    @staticmethod
    def _timestamp_epoch(timestamp: str) -> float:
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except ValueError as exc:
            raise ValueError("FramePacket.timestamp must be a valid ISO-8601 timestamp") from exc

    @staticmethod
    def _crop(frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        return frame[y1:y2, x1:x2]

    def _detector_for(self, camera_id: str):
        if self.detector_factory is None:
            return self.detector
        with self._lock:
            detector = self._camera_detectors.get(camera_id)
            if detector is None:
                detector = self.detector_factory()
                self._camera_detectors[camera_id] = detector
        return detector

    def _save_crop(self, crop, camera_id: str, timestamp: str, track_id: int) -> str | None:
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        safe_stamp = timestamp.replace(":", "-").replace("/", "-")
        path = self.crop_dir / f"{camera_id}_{track_id}_{safe_stamp}.jpg"
        return str(path) if cv2.imwrite(str(path), crop) else None

    def _cleanup(self, now: float) -> None:
        with self._lock:
            expired = [key for key, state in self.track_states.items() if now - state.last_seen_epoch > self.track_ttl_seconds]
            for key in expired:
                del self.track_states[key]
            self.metrics.expired_tracks += len(expired)
            inactive_cameras = [camera for camera, seen in self._camera_last_seen.items() if now - seen > self.track_ttl_seconds]
            for camera in inactive_cameras:
                detector = self._camera_detectors.pop(camera, None)
                if detector is not None and hasattr(detector, "close"):
                    detector.close()
                del self._camera_last_seen[camera]

    def _observe_verified_transition(self, identity_id: str | None, camera_id: str, now: float) -> None:
        if not identity_id or self.predictor is None or getattr(self.predictor, "transition_graph", None) is None:
            return
        previous = self._identity_last_seen.get(identity_id)
        if previous and previous[0] != camera_id:
            elapsed = now - previous[1]
            if 0 < elapsed <= self.transition_window_seconds:
                self.predictor.transition_graph.observe(previous[0], camera_id, elapsed, now)
                try:  # persistence is best-effort; IO trouble never breaks the hot path
                    if self.transition_graph_path is not None:
                        self.predictor.transition_graph.save(self.transition_graph_path)
                except Exception:
                    pass
        self._identity_last_seen[identity_id] = (camera_id, now)

    def process_frame_packet(self, packet: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Process the frozen Track 1 FramePacket contract and return Track 3 events.

        Safe to call from multiple per-camera worker threads concurrently.
        Hostile or malformed input is rejected, never raised: packets whose
        camera_id fails ``CAMERA_ID_RE`` (or an explicit allowlist), or whose
        timestamp cannot be parsed, are counted in ``metrics.rejected_packets``
        and yield ``[]``.  All TTL/throttle/transition bookkeeping runs on the
        local arrival clock -- packet-declared timestamps (past OR future) are
        never trusted for state expiry, so one hostile or clock-skewed camera
        cannot expire other cameras' tracks; the ``UnifiedEvent`` payload
        still carries Track 1's original timestamp verbatim.
        """
        try:
            frame_packet = FramePacket.from_mapping(packet)
        except ValueError:
            with self._lock:
                self.metrics.rejected_packets += 1
            return []
        if not CAMERA_ID_RE.fullmatch(frame_packet.camera_id) or (
                self.camera_allowlist is not None and frame_packet.camera_id not in self.camera_allowlist):
            with self._lock:
                self.metrics.rejected_packets += 1
            return []
        try:
            self._timestamp_epoch(frame_packet.timestamp)  # validate-only; payload passes through
        except ValueError:
            with self._lock:
                self.metrics.rejected_packets += 1
            return []
        now = time.time()  # arrival clock: the only clock trusted for state bookkeeping
        if self.searcher is not None and getattr(self.searcher, "reload_if_changed", None) is not None:
            try:
                self.searcher.reload_if_changed(self.gallery_watch_path)
            except Exception:
                pass  # a bad gallery file must not stop the live path
        self._cleanup(now)
        detector = self._detector_for(frame_packet.camera_id)
        if detector is None:
            return []
        try:
            detections = detector.detect_and_track(frame_packet.frame)
        except Exception as exc:
            with self._lock:
                self.metrics.processing_errors += 1
                try:
                    detector.load_failures = getattr(detector, "load_failures", 0) + 1
                    detector.last_error = f"{type(exc).__name__}: {exc}"
                except Exception:
                    pass
            return []
        camera_id = frame_packet.camera_id
        with self._lock:
            self.metrics.packets_processed += 1
            self._camera_last_seen[camera_id] = now
        events: list[dict[str, Any]] = []
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None or "bbox" not in detection:
                continue
            crop = self._crop(frame_packet.frame, detection["bbox"])
            quality = laplacian_variance(crop)  # cheap CPU work kept out of the lock
            if not passes_quality_gate(crop):
                with self._lock:
                    self.metrics.quality_rejections += 1
                continue
            key = (camera_id, int(track_id))
            with self._lock:
                self.metrics.detections_seen += 1
                is_new = key not in self.track_states
                state = self.track_states.setdefault(key, TrackState())
                state.last_seen_epoch = now
                previous_verified, previous_identity = state.verified, state.raw_identity_id
                should_reembed = is_new or (not state.verified and quality >= max(80.0, state.best_quality * self.crop_improvement_ratio))
                crop_improved = state.crop_url is None or quality >= state.best_quality * self.crop_improvement_ratio
            embedding = None
            if should_reembed and self.embedder is not None and self.searcher is not None:
                embedding = self.embedder.embed_from_bbox(frame_packet.frame, detection["bbox"])
            saved = None
            if crop_improved:
                saved = self._save_crop(crop, camera_id, frame_packet.timestamp, int(track_id))
            event_dict: dict[str, Any] | None = None
            with self._lock:
                if embedding is not None:
                    state.raw_name, state.raw_identity_id, state.score = self.searcher.search(embedding)
                state.verified = self.verifier.verify(state.score)
                identity_changed = state.verified != previous_verified or state.raw_identity_id != previous_identity
                if saved is not None:
                    state.crop_url, state.best_quality = saved, quality
                self._observe_verified_transition(state.raw_identity_id if state.verified else None, camera_id, now)
                emit = is_new or identity_changed or crop_improved or now - state.last_event_epoch >= self.event_interval_seconds
                if emit:
                    prediction = self.predictor.predict(camera_id, now=now) if self.predictor else {"next_camera": None, "eta_seconds": None, "probability": None}
                    state.last_event_epoch = now
                    event = UnifiedEvent(track_id=str(track_id), identity_name=state.raw_name if state.verified else None,
                        identity_id=state.raw_identity_id if state.verified else None, match_score=float(state.score), verified=state.verified,
                        current_camera=camera_id, timestamp=frame_packet.timestamp, face_crop_url=state.crop_url,
                        next_camera=prediction["next_camera"], eta_seconds=prediction["eta_seconds"], transition_probability=prediction["probability"])
                    event_dict = event.to_dict()
                    self.metrics.events_emitted += 1
            if event_dict is not None:
                events.append(event_dict)
        return events

    def health(self) -> dict[str, Any]:
        with self._lock:
            detector_health = {}
            for cid, det in self._camera_detectors.items():
                entry = {"loaded": getattr(det, "model", None) is not None}
                failures = getattr(det, "load_failures", 0)
                if failures:
                    entry["load_failures"] = failures
                last_err = getattr(det, "last_error", None)
                if last_err:
                    entry["last_error"] = str(last_err)[:120]
                detector_health[cid] = entry
            return {"active_cameras": len(self._camera_last_seen), "active_tracks": len(self.track_states),
                    "detectors": detector_health, **self.metrics.__dict__}

    def run(self, frame: Any):
        """Legacy scaffold entry point. Production callers use process_frame_packet."""
        if isinstance(frame, Mapping): return self.process_frame_packet(frame)
        detections = self.detector.detect(frame) if self.detector is not None else []
        return {"detections": detections, "track_ids": [], "embeddings": [], "verification": []}
