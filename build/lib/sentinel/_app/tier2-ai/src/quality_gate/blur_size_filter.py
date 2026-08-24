"""Blur and minimum-size filtering for detected person crops."""

from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(image: np.ndarray) -> float:
    """Return the Laplacian variance as a blur quality score."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def passes_quality_gate(image: np.ndarray, min_size: int = 60, min_blur_score: float = 80.0) -> bool:
    """Check a detected crop against minimum size and blur thresholds."""
    if image is None or image.size == 0:
        return False
    height, width = image.shape[:2]
    if width < min_size or height < min_size:
        return False
    return laplacian_variance(image) >= min_blur_score
