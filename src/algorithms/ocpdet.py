"""OCPDet-inspired detectors."""
from __future__ import annotations

from typing import List

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPRegressor

from .ewma import detect_changepoints_ewma


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


def detect_changepoints_cumsum(series, *, threshold: float = 10.0, drift: float = 0.0, min_distance: int = 30) -> List[int]:
    """Classic cumulative sum change detector."""
    arr = _prepare_series(series)
    if arr.size == 0:
        return []
    mean = np.nanmean(arr)
    g_pos = 0.0
    g_neg = 0.0
    changepoints: List[int] = []
    last_cp = -min_distance

    for idx, value in enumerate(arr):
        diff = value - mean - drift
        g_pos = max(0.0, g_pos + diff)
        g_neg = min(0.0, g_neg + diff)
        if g_pos > threshold and (idx - last_cp) >= min_distance:
            changepoints.append(idx)
            last_cp = idx
            g_pos = 0.0
            g_neg = 0.0
        elif abs(g_neg) > threshold and (idx - last_cp) >= min_distance:
            changepoints.append(idx)
            last_cp = idx
            g_pos = 0.0
            g_neg = 0.0
    return changepoints


def detect_changepoints_two_sample(series, *, window_size: int = 40, step: int = 5, alpha: float = 0.01, min_distance: int = 20) -> List[int]:
    """Sliding KS two-sample detector."""
    arr = _prepare_series(series)
    if arr.size < 2 * window_size:
        return []
    changepoints: List[int] = []
    last_cp = -min_distance

    for start in range(0, arr.size - 2 * window_size, step):
        left = arr[start : start + window_size]
        right = arr[start + window_size : start + 2 * window_size]
        if np.isnan(left).any() or np.isnan(right).any():
            continue
        _, pvalue = stats.ks_2samp(left, right)
        midpoint = start + window_size
        if pvalue < alpha and (midpoint - last_cp) >= min_distance:
            changepoints.append(midpoint)
            last_cp = midpoint
    return changepoints


def detect_changepoints_neural(series, *, window_size: int = 20, step: int = 1, hidden_layer_sizes: tuple[int, ...] = (20,), threshold: float = 2.5, min_distance: int = 30, random_state: int | None = 123) -> List[int]:
    """Simple residual-based detector using an MLP auto-regressor."""
    arr = _prepare_series(series)
    if arr.size <= window_size + 1:
        return []

    X = []
    y = []
    for idx in range(0, arr.size - window_size - 1, step):
        window = arr[idx : idx + window_size]
        target = arr[idx + window_size]
        if np.isnan(window).any() or np.isnan(target):
            continue
        X.append(window)
        y.append(target)
    if len(X) < 10:
        return []
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    split = int(0.7 * len(X_arr))
    if split < 5:
        split = len(X_arr) - 5
    split = max(split, 5)

    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation="relu", solver="adam", max_iter=500, random_state=random_state)
    model.fit(X_arr[:split], y_arr[:split])
    preds = model.predict(X_arr)
    residuals = y_arr - preds
    std = float(np.std(residuals)) if residuals.size else 0.0
    if std == 0.0:
        return []

    changepoints: List[int] = []
    last_cp = -min_distance
    for idx, res in enumerate(residuals):
        score = abs(res) / std
        if score > threshold:
            cp = idx * step + window_size
            if (cp - last_cp) >= min_distance:
                changepoints.append(int(cp))
                last_cp = cp
    return changepoints


def detect_changepoints_ewma_wrapper(series, *, alpha: float = 0.1, threshold: float = 2.5, min_instances: int = 5) -> List[int]:
    return detect_changepoints_ewma(series, alpha=alpha, threshold=threshold, min_instances=min_instances)
