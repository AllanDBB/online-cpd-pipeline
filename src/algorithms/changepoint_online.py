"""Wrappers for changepoint_online-style detectors using ruptures."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import ruptures as rpt


def _prepare_series(series) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return arr
    valid = ~np.isnan(arr)
    if valid.any():
        last_valid = int(np.max(np.where(valid)))
        arr = arr[: last_valid + 1]
    return arr


def _predict_breakpoints(detector, signal_length: int, *, pen: float | None = None, n_bkps: int | None = None) -> List[int]:
    if pen is not None:
        bkps = detector.predict(pen=pen)
    elif n_bkps is not None:
        bkps = detector.predict(n_bkps=n_bkps)
    else:
        raise ValueError("Either 'penalty' or 'n_bkps' must be provided")
    filtered = [int(b) for b in bkps if 0 < b < signal_length]
    return sorted(set(filtered))


def _run_detector(series: np.ndarray, algo: str, config: Dict[str, float | int | str]) -> List[int]:
    min_size = int(config.get("min_size", 20))
    jump = int(config.get("jump", 1))
    n_bkps = config.get("n_bkps")
    pen = config.get("penalty")
    model = config.get("model", "l2")
    length = series.size

    if algo == "pelt":
        detector = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(series)
        return _predict_breakpoints(detector, length, pen=pen, n_bkps=n_bkps)
    if algo == "binseg":
        detector = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(series)
        return _predict_breakpoints(detector, length, pen=pen, n_bkps=n_bkps)
    if algo == "window":
        width = int(config.get("width", 40))
        detector = rpt.Window(width=width, model=model, jump=jump).fit(series)
        return _predict_breakpoints(detector, length, pen=pen, n_bkps=n_bkps)
    if algo == "bottomup":
        detector = rpt.BottomUp(model=model, min_size=min_size, jump=jump).fit(series)
        return _predict_breakpoints(detector, length, pen=pen, n_bkps=n_bkps)
    raise ValueError(f"Unsupported algorithm '{algo}' for changepoint_online wrapper")


def detect_changepoints_focus(series, *, penalty: float = 20.0, min_size: int = 20, jump: int = 5, n_bkps: int | None = None) -> List[int]:
    """Heuristic wrapper approximating 'Focus' with RBF penalised segmentation."""
    arr = _prepare_series(series)
    if arr.size == 0:
        return []
    config = {"penalty": penalty, "min_size": min_size, "jump": jump, "model": "rbf", "n_bkps": n_bkps}
    return _run_detector(arr, "pelt", config)


def detect_changepoints_gaussian(series, *, penalty: float = 25.0, min_size: int = 30, jump: int = 5, n_bkps: int | None = None) -> List[int]:
    """Wrapper emulating Gaussian likelihood with l2 cost."""
    arr = _prepare_series(series)
    if arr.size == 0:
        return []
    config = {"penalty": penalty, "min_size": min_size, "jump": jump, "model": "l2", "n_bkps": n_bkps}
    return _run_detector(arr, "pelt", config)


def detect_changepoints_np_focus(series, *, penalty: float = 30.0, width: int = 40, jump: int = 5, n_bkps: int | None = None) -> List[int]:
    """Non-parametric focus approximation via window-based detector."""
    arr = _prepare_series(series)
    if arr.size == 0:
        return []
    config = {"penalty": penalty, "width": width, "jump": jump, "model": "rbf", "n_bkps": n_bkps}
    return _run_detector(arr, "window", config)


def detect_changepoints_md_focus(series, *, penalty: float = 35.0, min_size: int = 25, jump: int = 5, n_bkps: int | None = None) -> List[int]:
    """Multidimensional focus approximation using binary segmentation."""
    arr = _prepare_series(series)
    if arr.size == 0:
        return []
    config = {"penalty": penalty, "min_size": min_size, "jump": jump, "model": "rbf", "n_bkps": n_bkps}
    return _run_detector(arr, "binseg", config)
