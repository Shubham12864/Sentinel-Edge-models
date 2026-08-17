"""Verification threshold logic for identity matches."""

from __future__ import annotations


class MatchVerifier:
    """Simple threshold-based verification for an identity match."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def verify(self, similarity: float) -> bool:
        return similarity >= self.threshold
