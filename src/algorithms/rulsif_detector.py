"""RuLSIF-based change point detector wrapper."""
from __future__ import annotations

from typing import List

import numpy as np
from roerich.change_point.onnr import OnlineNNRuLSIF


def _prepare_series(series) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return arr
    valid = ~np.isnan(arr)
    if valid.any():
        arr = arr[: int(np.max(np.where(valid))) + 1]
    return arr


def detect_changepoints_rulsif(
    series,
    *,
    window_size: int = 10,
    lag_size: int = 50,
    step: int = 5,
    n_epochs: int = 1,
    threshold: float = 0.1,
    min_distance: int = 25,
    alpha: float = 0.1,
) -> List[int]:
    arr = _prepare_series(series)
    if arr.size < 2 * window_size + lag_size + 1:
        return []

    detector = OnlineNNRuLSIF(window_size=window_size, lag_size=lag_size, step=step, n_epochs=n_epochs, alpha=alpha)
    scores, peaks = detector.predict(arr)
    scores = np.asarray(scores)

    changepoints: List[int] = []
    last_cp = -min_distance
    for idx in peaks:
        if idx >= len(scores):
            continue
        if scores[idx] > threshold and (idx - last_cp) >= min_distance:
            changepoints.append(int(idx))
            last_cp = int(idx)
    return changepoints
