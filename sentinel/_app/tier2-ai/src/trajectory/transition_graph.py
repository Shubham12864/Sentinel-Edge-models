"""Camera transition graph representation."""

from __future__ import annotations
import json
from pathlib import Path


class TransitionGraph:
    """Observed camera transitions with count, typical travel time, and last-seen time."""
    def __init__(self): self.edges = {}
    def add_transition(self, source: str, target: str, weight: float = 1.0, eta_seconds: float = 0.0, observed_at: float | None = None):
        if not source or not target or weight < 0 or eta_seconds < 0: raise ValueError("Transition source/target must be non-empty and values non-negative")
        self.edges.setdefault(source, {})[target] = {"weight": float(weight), "eta_seconds": float(eta_seconds), "observed_at": observed_at}
    def observe(self, source: str, target: str, eta_seconds: float, observed_at: float):
        if source == target or eta_seconds < 0: return
        edge = self.edges.setdefault(source, {}).setdefault(target, {"weight": 0.0, "eta_seconds": 0.0, "observed_at": observed_at})
        old = edge["weight"]; edge["weight"] = old + 1; edge["eta_seconds"] = (edge["eta_seconds"] * old + eta_seconds) / edge["weight"]; edge["observed_at"] = observed_at
    def neighbors(self, source: str): return list(self.edges.get(source, {}).keys())
    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(self.edges), encoding="utf-8")
    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists(): return False
        self.edges = json.loads(path.read_text(encoding="utf-8")); return True
