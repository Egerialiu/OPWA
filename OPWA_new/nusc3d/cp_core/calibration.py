"""
Calibration — Conformal Prediction threshold calibration.

Ported from OPWA_v3/exp0/calibration.py, stripped of 2D-specific deps.

Core logic:
    q̂ = np.quantile(scores, 0.90)   ← the 90th percentile of nonconformity scores

Usage in 3D:
    - Replace SegFormer inference with your LiDAR / point-cloud model.
    - Replace GT loading with nuScenes lidarseg labels.
    - Call compute_q_hat(scores) directly on your score array.
"""

import numpy as np
from typing import Tuple


def compute_q_hat(
    scores: np.ndarray,
    quantile: float = 0.90,
) -> Tuple[float, float, int]:
    """Compute the conformal prediction threshold q̂.

    Args:
        scores: 1D array of nonconformity scores (e.g. 1 - softmax_prob(gt_label)).
        quantile: The quantile to use for q̂ (default 0.90 → 90th percentile).

    Returns:
        q_hat: The threshold value.
        coverage: Actual coverage fraction on this set (scores <= q_hat).
        n_valid: Number of valid elements processed.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {scores.shape}")
    n_valid = len(scores)
    if n_valid == 0:
        raise ValueError("Empty scores array — nothing to calibrate.")

    q_hat = float(np.quantile(scores, quantile))
    coverage = float((scores <= q_hat).mean())

    return q_hat, coverage, n_valid
