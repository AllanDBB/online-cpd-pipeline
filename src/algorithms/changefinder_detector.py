"""ChangeFinder SDAR detector wrapper."""
from __future__ import annotations

from typing import List

import numpy as np
import changefinder


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


def detect_changepoints_changefinder(
    series,
    *,
    r: float = 0.5,
    order: int = 1,
    smooth: int = 7,
    threshold: float = 2.5,
    min_distance: int = 25,
) -> List[int]:
    arr = _prepare_series(series)
    if arr.size == 0:
        return []

    cf = changefinder.ChangeFinder(r=r, order=order, smooth=smooth)
    scores: List[float] = []
    for value in arr:
        scores.append(float(cf.update(value)))

    std = float(np.std(scores)) or 1e-6
    mean = float(np.mean(scores))

    changepoints: List[int] = []
    last_cp = -min_distance
    for idx, score in enumerate(scores):
        zscore = abs(score - mean) / std
        if zscore > threshold and (idx - last_cp) >= min_distance:
            changepoints.append(idx)
            last_cp = idx
    return changepoints
