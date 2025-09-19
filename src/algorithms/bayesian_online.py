"""Bayesian online change point detection wrapper."""
from __future__ import annotations

from typing import List

import numpy as np
from bayesian_changepoint_detection.online_changepoint_detection import (
    StudentT,
    constant_hazard,
    online_changepoint_detection,
)


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


def detect_changepoints_cpfinder(
    series,
    *,
    hazard_lambda: float = 200.0,
    alpha: float = 0.1,
    beta: float = 0.01,
    kappa: float = 1.0,
    mu: float | None = None,
    probability_threshold: float = 0.6,
    min_distance: int = 25,
) -> List[int]:
    arr = _prepare_series(series)
    if arr.size == 0:
        return []
    mu = float(mu) if mu is not None else float(arr[0])
    student = StudentT(alpha=alpha, beta=beta, kappa=kappa, mu=mu)

    hazard_func = lambda r: constant_hazard(hazard_lambda, r)  # noqa: E731
    run_length_probs, _ = online_changepoint_detection(arr, hazard_func, student)
    cp_probs = run_length_probs[0, 1:]

    changepoints: List[int] = []
    last_cp = -min_distance
    for idx, prob in enumerate(cp_probs, start=1):
        if prob >= probability_threshold and (idx - last_cp) >= min_distance:
            changepoints.append(idx)
            last_cp = idx
    return changepoints
