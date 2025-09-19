"""State-space model based change detectors."""
from __future__ import annotations

from typing import List

import numpy as np
from filterpy.kalman import KalmanFilter


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


def _kalman_detector(
    series,
    *,
    process_noise: float,
    measurement_noise: float,
    threshold: float,
    min_distance: int,
    adaptation: float = 0.0,
) -> List[int]:
    arr = _prepare_series(series)
    if arr.size == 0:
        return []

    kf = KalmanFilter(dim_x=2, dim_z=1)
    dt = 1.0
    kf.F = np.array([[1, dt], [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.Q = np.array([[process_noise, 0], [0, process_noise]])
    kf.R = np.array([[measurement_noise]])
    kf.x = np.array([[arr[0]], [0.0]])
    kf.P *= 10.0

    residuals: List[float] = []
    predictions: List[float] = []
    for value in arr:
        kf.predict()
        kf.update(np.array([[value]]))
        predictions.append(float(kf.x[0, 0]))
        residuals.append(float(value - kf.x[0, 0]))
        if adaptation > 0.0:
            kf.Q += np.eye(2) * adaptation

    resid_array = np.asarray(residuals)
    std = float(np.std(resid_array)) or 1e-6

    changepoints: List[int] = []
    last_cp = -min_distance
    for idx, resid in enumerate(resid_array):
        zscore = abs(resid) / std
        if zscore > threshold and (idx - last_cp) >= min_distance:
            changepoints.append(idx)
            last_cp = idx
    return changepoints


def detect_changepoints_ssm_canary(series, *, process_noise: float = 1e-3, measurement_noise: float = 0.5, threshold: float = 3.0, min_distance: int = 25) -> List[int]:
    return _kalman_detector(series, process_noise=process_noise, measurement_noise=measurement_noise, threshold=threshold, min_distance=min_distance)


def detect_changepoints_tagi_canary(series, *, process_noise: float = 5e-4, measurement_noise: float = 0.3, threshold: float = 2.8, min_distance: int = 20, adaptation: float = 1e-4) -> List[int]:
    return _kalman_detector(series, process_noise=process_noise, measurement_noise=measurement_noise, threshold=threshold, min_distance=min_distance, adaptation=adaptation)


def detect_changepoints_skf_canary(series, *, process_noise: float = 1e-2, measurement_noise: float = 0.7, threshold: float = 3.2, min_distance: int = 25) -> List[int]:
    return _kalman_detector(series, process_noise=process_noise, measurement_noise=measurement_noise, threshold=threshold, min_distance=min_distance)
