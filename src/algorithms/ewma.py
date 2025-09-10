"""EWMA-based change point detector using a simple NumPy EWMA implementation.

This implementation avoids an external dependency and supports sweeping `alpha`.
It flags a change when the current sample deviates from the EWMA mean by more
than `threshold * running_std`.
"""
from __future__ import annotations
from typing import List
import numpy as np


def detect_changepoints_ewma(
    series: np.ndarray,
    alpha: float = 0.2,
    threshold: float = 3.0,
    min_instances: int = 1,
) -> List[int]:
    """
    Detect change points using a lightweight EWMA (exponentially weighted mean/var).

    Args:
        series: 1D array-like of numeric values.
        alpha: smoothing factor (0 < alpha <= 1). Larger alpha -> more weight to recent samples.
        threshold: number of std deviations to trigger a detection.
        min_instances: minimal spacing (in samples) between detections.

    Returns:
        List[int]: indices of detected change points.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n == 0:
        return []

    # Initialize with first sample
    ew_mean = x[0]
    ew_var = 0.0  # biased EW variance
    detections: List[int] = []
    last_detect = -min_instances - 1

    for t, val in enumerate(x):
        # Stop if we encounter padding / missing values
        if np.isnan(val):
            break
        if t == 0:
            # first sample already used to init
            mean_prev = ew_mean
            continue

        # prediction residual using previous mean
        resid = val - ew_mean

        # current std (biased)
        std = float(np.sqrt(max(ew_var, 1e-12)))

        if std > 0 and abs(resid) > threshold * std and (t - last_detect) > min_instances:
            detections.append(int(t))
            last_detect = t

        # Update EW mean and EW variance using exponential smoothing
        mean_prev = ew_mean
        ew_mean = alpha * val + (1 - alpha) * ew_mean
        # squared deviation wrt previous mean (as in online variance)
        ew_var = alpha * (val - mean_prev) ** 2 + (1 - alpha) * ew_var

    return detections
