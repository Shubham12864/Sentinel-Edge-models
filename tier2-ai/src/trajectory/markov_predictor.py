"""Markov-style next-camera predictor with temporal decay and Laplace smoothing."""
from __future__ import annotations

import math
import time


class MarkovPredictor:
    """Argmax over exponentially-decayed transition counts.

    Edges whose ``observed_at`` is unknown (hand-seeded priors) are treated as
    having a virtual age of 4 * decay_seconds so they also fade with time.
    Probabilities are Laplace-smoothed (add-alpha) so they stay proper even
    with one or two observations.
    """

    def __init__(self, transition_graph=None, decay_seconds: float = 3600.0, alpha: float = 0.5):
        self.transition_graph, self.decay_seconds = transition_graph, decay_seconds
        self.alpha = float(alpha)

    def _decayed(self, edge: dict, now: float) -> float:
        observed_at = edge.get("observed_at")
        age = 4.0 * self.decay_seconds if observed_at is None else max(0.0, now - observed_at)
        return edge["weight"] * math.exp(-age / self.decay_seconds)

    def predict(self, current_camera: str, now: float | None = None):
        edges = self.transition_graph.edges.get(current_camera) if self.transition_graph is not None else None
        if not edges:
            return {"next_camera": None, "eta_seconds": None, "probability": None}
        now = time.time() if now is None else now
        weighted = {target: self._decayed(value, now) for target, value in edges.items()}
        alpha = self.alpha
        denominator = sum(weighted.values()) + alpha * len(weighted)
        target = max(weighted, key=weighted.get)
        return {
            "next_camera": target,
            "eta_seconds": int(round(edges[target]["eta_seconds"])),
            "probability": (weighted[target] + alpha) / denominator if denominator > 0 else None,
        }
