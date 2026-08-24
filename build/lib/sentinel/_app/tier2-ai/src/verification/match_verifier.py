"""Verification threshold logic for identity matches.

Default behavior is unchanged (single 0.75 cut).  Optional calibrated mode
adds hysteresis (separate enter/exit thresholds) and a runner-up margin so a
match against an ambiguous gallery cannot be silently reported as verified.
"""
from __future__ import annotations


class MatchVerifier:
    """Threshold-based verification for an identity match."""

    def __init__(self, threshold: float = 0.75, exit_threshold: float | None = None,
                 min_margin: float = 0.0):
        """
        threshold       score required to call a candidate verified.
        exit_threshold  optional lower bar for *staying* verified once a track
                        has already been accepted (hysteresis).  None keeps
                        the classic single-threshold behaviour.
        min_margin      optional minimum top1-runnerup margin required for a
                        confident verification when used with verify_with_margin.
        """
        self.threshold = float(threshold)
        self.exit_threshold = self.threshold if exit_threshold is None else float(exit_threshold)
        self.min_margin = float(min_margin)
        self._was_verified = False

    def verify(self, similarity: float) -> bool:
        """Single-threshold check (unchanged contract) or hysteresis mode."""
        if self.exit_threshold == self.threshold:
            return similarity >= self.threshold
        if self._was_verified:
            self._was_verified = similarity >= self.exit_threshold
        else:
            self._was_verified = similarity >= self.threshold
        return self._was_verified

    def verify_with_margin(self, similarity: float, margin: float,
                           min_margin: float | None = None) -> bool:
        """Verify only when the score clears the threshold AND the runner-up
        margin shows the gallery answer is unambiguous."""
        needed = self.min_margin if min_margin is None else float(min_margin)
        return self.verify(similarity) and margin >= needed

    def reset(self) -> None:
        """Forget hysteresis state (call per new track/camera context)."""
        self._was_verified = False
