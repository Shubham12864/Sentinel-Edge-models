"""Markov-style next-camera prediction stub."""

from __future__ import annotations


class MarkovPredictor:
    def __init__(self, transition_graph=None, decay_seconds: float = 3600.0): self.transition_graph, self.decay_seconds = transition_graph, decay_seconds
    def predict(self, current_camera: str, now: float | None = None):
        import math, time
        if self.transition_graph is None or not self.transition_graph.edges.get(current_camera): return {"next_camera": None, "eta_seconds": None, "probability": None}
        now = time.time() if now is None else now; edges = self.transition_graph.edges[current_camera]
        weighted = {target: value["weight"] * math.exp(-max(0, now - value["observed_at"]) / self.decay_seconds) if value["observed_at"] is not None else value["weight"] for target, value in edges.items()}
        target = max(weighted, key=weighted.get); total = sum(weighted.values())
        return {"next_camera": target, "eta_seconds": int(round(edges[target]["eta_seconds"])), "probability": weighted[target] / total if total else None}
