"""Maximum Mean Discrepancy helper metrics for change-point evaluation."""
from __future__ import annotations
from typing import Sequence
import math
import numpy as np


def _gaussian_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    diff = x[:, None] - y[None, :]
    scale = -0.5 / max(bandwidth ** 2, 1e-12)
    values = np.exp(scale * diff ** 2)
    return float(np.mean(values))


def maximum_mean_discrepancy(
    real_changes: Sequence[int] | Sequence[float],
    detected_changes: Sequence[int] | Sequence[float],
    series_length: int,
    bandwidth: float | None = None,
) -> float:
    """Compute a simple Gaussian-kernel MMD between true and detected indices.

    Args:
        real_changes: iterable with true change-point positions.
        detected_changes: iterable with detected change-point positions.
        series_length: length of the series; used to normalise indices.
        bandwidth: optional kernel bandwidth in normalised units. If ``None`` we
            derive it from the pooled data (median heuristic) or fall back to
            ``0.1``.

    Returns:
        A non-negative float; ``0`` means identical sets. If one set is empty
        and the other not, returns ``1.0`` as a maximal discrepancy signal.
    """
    if series_length <= 0:
        raise ValueError("series_length must be positive")

    real = np.asarray(list(real_changes), dtype=float)
    detected = np.asarray(list(detected_changes), dtype=float)

    if real.size == 0 and detected.size == 0:
        return 0.0
    if real.size == 0 or detected.size == 0:
        return 1.0

    real_norm = real / float(series_length)
    detected_norm = detected / float(series_length)
    all_samples = np.concatenate([real_norm, detected_norm])

    if bandwidth is None:
        if all_samples.size > 1:
            pairwise_diffs = np.abs(all_samples[:, None] - all_samples[None, :])
            median = float(np.median(pairwise_diffs))
            bandwidth = median if median > 0 else 0.1
        else:
            bandwidth = 0.1
    bandwidth = float(bandwidth)

    k_xx = _gaussian_kernel(real_norm, real_norm, bandwidth)
    k_yy = _gaussian_kernel(detected_norm, detected_norm, bandwidth)
    k_xy = _gaussian_kernel(real_norm, detected_norm, bandwidth)

    mmd_squared = max(k_xx + k_yy - 2 * k_xy, 0.0)
    return float(math.sqrt(mmd_squared))
