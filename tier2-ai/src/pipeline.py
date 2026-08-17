"""Tier 2 orchestration: validated FramePackets in, UnifiedEvents out."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
import time
import cv2
from .schemas import FramePacket, UnifiedEvent
from .quality_gate.blur_size_filter import laplacian_variance, passes_quality_gate
from .verification.match_verifier import MatchVerifier

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
                 crop_improvement_ratio: float = 1.10, transition_window_seconds: float = 600.0):
        self.detector, self.tracker, self.embedder = detector, tracker, embedder
        self.detector_factory = detector_factory
        self.verifier, self.searcher, self.predictor = verifier or MatchVerifier(), searcher, predictor
        self.crop_dir = Path(crop_dir) if crop_dir else Path(__file__).resolve().parents[1] / "data" / "face_crops"
        self.track_ttl_seconds, self.event_interval_seconds = track_ttl_seconds, event_interval_seconds
        self.crop_improvement_ratio, self.transition_window_seconds = crop_improvement_ratio, transition_window_seconds
        self.track_states: dict[tuple[str, int], TrackState] = {}
        self._camera_detectors: dict[str, Any] = {}
        self._camera_last_seen: dict[str, float] = {}
        self._identity_last_seen: dict[str, tuple[str, float]] = {}
        self.metrics = PipelineMetrics()

    @classmethod
    def create_default(cls, *, weights_path: str | Path | None = None, identity_index_path: str | Path | None = None,
                       crop_dir: str | Path | None = None, ctx_id: int = -1, **kwargs) -> "UnifiedPipeline":
        """Build the complete production pipeline once, without opening any camera."""
        from .detection.face_detector import FaceDetector
        from .embedding.face_embedder import FaceEmbedder
        from .identity_search.faiss_index import IdentitySearch
        from .trajectory.transition_graph import TransitionGraph
        from .trajectory.markov_predictor import MarkovPredictor
        searcher = IdentitySearch()
        if identity_index_path is not None:
            searcher.load(identity_index_path)
        graph = TransitionGraph()
        return cls(embedder=FaceEmbedder(ctx_id=ctx_id), searcher=searcher,
                   predictor=MarkovPredictor(graph), crop_dir=crop_dir,
                   detector_factory=lambda: FaceDetector(weights_path=weights_path), **kwargs)

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
        if camera_id not in self._camera_detectors:
            self._camera_detectors[camera_id] = self.detector_factory()
        return self._camera_detectors[camera_id]

    def _save_crop(self, crop, camera_id: str, timestamp: str, track_id: int) -> str | None:
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        safe_stamp = timestamp.replace(":", "-").replace("/", "-")
        path = self.crop_dir / f"{camera_id}_{track_id}_{safe_stamp}.jpg"
        return str(path) if cv2.imwrite(str(path), crop) else None

    def _cleanup(self, now: float) -> None:
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
        self._identity_last_seen[identity_id] = (camera_id, now)

    def process_frame_packet(self, packet: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Process the frozen Track 1 FramePacket contract and return Track 3 events."""
        frame_packet = FramePacket.from_mapping(packet)
        now = self._timestamp_epoch(frame_packet.timestamp)
        self._cleanup(now)
        detector = self._detector_for(frame_packet.camera_id)
        if detector is None:
            return []
        try:
            detections = detector.detect_and_track(frame_packet.frame)
        except Exception:
            self.metrics.processing_errors += 1
            return []
        self.metrics.packets_processed += 1
        self._camera_last_seen[frame_packet.camera_id] = now
        events: list[dict[str, Any]] = []
        for detection in detections:
            self.metrics.detections_seen += 1
            track_id = detection.get("track_id")
            if track_id is None or "bbox" not in detection:
                continue
            crop = self._crop(frame_packet.frame, detection["bbox"])
            if not passes_quality_gate(crop):
                self.metrics.quality_rejections += 1
                continue
            key = (frame_packet.camera_id, int(track_id))
            is_new = key not in self.track_states
            state = self.track_states.setdefault(key, TrackState())
            state.last_seen_epoch = now
            quality = laplacian_variance(crop)
            identity_changed = False
            should_reembed = is_new or (not state.verified and quality >= max(80.0, state.best_quality * self.crop_improvement_ratio))
            if should_reembed and self.embedder is not None and self.searcher is not None:
                previous_verified = state.verified
                previous_identity = state.raw_identity_id
                embedding = self.embedder.get_embedding(crop)
                if embedding is not None:
                    state.raw_name, state.raw_identity_id, state.score = self.searcher.search(embedding)
                state.verified = self.verifier.verify(state.score)
                identity_changed = state.verified != previous_verified or state.raw_identity_id != previous_identity
            crop_improved = state.crop_url is None or quality >= state.best_quality * self.crop_improvement_ratio
            if crop_improved:
                saved = self._save_crop(crop, frame_packet.camera_id, frame_packet.timestamp, int(track_id))
                if saved is not None:
                    state.crop_url, state.best_quality = saved, quality
            self._observe_verified_transition(state.raw_identity_id if state.verified else None, frame_packet.camera_id, now)
            emit = is_new or identity_changed or crop_improved or now - state.last_event_epoch >= self.event_interval_seconds
            if not emit:
                continue
            prediction = self.predictor.predict(frame_packet.camera_id, now=now) if self.predictor else {"next_camera": None, "eta_seconds": None, "probability": None}
            state.last_event_epoch = now
            event = UnifiedEvent(track_id=str(track_id), identity_name=state.raw_name if state.verified else None,
                identity_id=state.raw_identity_id if state.verified else None, match_score=float(state.score), verified=state.verified,
                current_camera=frame_packet.camera_id, timestamp=frame_packet.timestamp, face_crop_url=state.crop_url,
                next_camera=prediction["next_camera"], eta_seconds=prediction["eta_seconds"], transition_probability=prediction["probability"])
            events.append(event.to_dict())
            self.metrics.events_emitted += 1
        return events

    def health(self) -> dict[str, Any]:
        return {"active_cameras": len(self._camera_last_seen), "active_tracks": len(self.track_states), **self.metrics.__dict__}

    def run(self, frame: Any):
        """Legacy scaffold entry point. Production callers use process_frame_packet."""
        if isinstance(frame, Mapping): return self.process_frame_packet(frame)
        detections = self.detector.detect(frame) if self.detector is not None else []
        return {"detections": detections, "track_ids": [], "embeddings": [], "verification": []}
