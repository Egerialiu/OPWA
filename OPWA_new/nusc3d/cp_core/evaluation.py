"""
Evaluation — Bin statistics + coverage/gap computation.

Ported from OPWA_v3/exp0/evaluation.py, generalized for any domain.

Core outputs per bin:
    - coverage_rate : fraction of points where score <= q̂
    - gap           : max(0, target_coverage - coverage_rate)
    - mean_score    : mean nonconformity score in the bin
    - mean_set_size : mean prediction-set size (if applicable)
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional


def compute_results(
    q_hat: float,
    cal_coverage: float,
    cal_pixel_count: int,
    bin_arrays: Dict[str, np.ndarray],
    bin_names: List[str],
    target_coverage: float = 0.90,
    alpha: float = 0.10,
    dataset_label: str = "unknown",
) -> dict:
    """Compute per-bin statistics and build the final results dict.

    Args:
        q_hat: CP threshold from calibration.
        cal_coverage: coverage on the calibration set.
        cal_pixel_count: number of valid calibration points.
        bin_arrays: dict with keys:
            pixel_counts, covered_counts, score_sums, set_size_sums
            each a numpy array of length N_BINS.
        bin_names: human-readable names for each bin (length N_BINS).
        target_coverage: target coverage (e.g., 0.90 for α=0.10).
        alpha: miscoverage level.
        dataset_label: string identifier for the dataset.

    Returns:
        results dict with per-bin stats + overall metrics.
    """
    pixel_counts = bin_arrays["pixel_counts"]
    covered_counts = bin_arrays["covered_counts"]
    score_sums = bin_arrays["score_sums"]
    set_size_sums = bin_arrays.get("set_size_sums",
                                   np.zeros_like(pixel_counts))

    if len(pixel_counts) != len(bin_names):
        raise ValueError(
            f"len(pixel_counts)={len(pixel_counts)} != len(bin_names)={len(bin_names)}"
        )
    total_valid = int(pixel_counts.sum())
    num_bins = len(bin_names)

    bins = {}
    for i in range(num_bins):
        pc = int(pixel_counts[i])
        cc = int(covered_counts[i])
        cov = float(cc / pc) if pc > 0 else 0.0
        gap = max(0.0, target_coverage - cov)
        mss = float(set_size_sums[i] / pc) if pc > 0 else 0.0
        ms = float(score_sums[i] / pc) if pc > 0 else 0.0

        bins[bin_names[i]] = {
            "count": pc,
            "covered_count": cc,
            "coverage_rate": round(cov, 6),
            "gap": round(gap, 6),
            "mean_set_size": round(mss, 6),
            "mean_score": round(ms, 6),
        }

    overall_cov = float(covered_counts.sum() / total_valid) if total_valid > 0 else 0.0
    max_gap = max(b["gap"] for b in bins.values())
    bin0_gap = bins[bin_names[0]]["gap"]
    bin0_ratio = float(bins[bin_names[0]]["count"] / total_valid) if total_valid > 0 else 0.0

    results = {
        "dataset": dataset_label,
        "alpha": alpha,
        "q_hat": round(q_hat, 6),
        "calibration_count": cal_pixel_count,
        "calibration_coverage": round(cal_coverage, 6),
        "total_test_count": total_valid,
        "overall_test_coverage": round(overall_cov, 6),
        "max_gap": round(max_gap, 6),
        "bin0_gap": round(bin0_gap, 6),
        "bin0_ratio": round(bin0_ratio, 6),
        "bins": bins,
    }

    return results


def save_results(results: dict, filepath: str) -> str:
    """Save results to JSON and print summary."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")
    return filepath


def print_summary(results: dict) -> None:
    """Print a human-readable summary of results."""
    print("\n" + "=" * 60)
    print("CONFORMAL PREDICTION EVALUATION")
    print("=" * 60)
    print(f"  Dataset:                {results['dataset']}")
    print(f"  Target coverage (1-α):  {1 - results['alpha']}")
    print(f"  q_hat:                  {results['q_hat']}")
    print(f"  Calibration coverage:   {results['calibration_coverage']}")
    print(f"  Calibration points:     {results['calibration_count']:,}")
    print(f"  Total test points:      {results['total_test_count']:,}")
    print(f"  Overall test coverage:  {results['overall_test_coverage']}")
    print(f"  Max gap:                {results['max_gap']}")
    print(f"  Bin0 gap:               {results['bin0_gap']}")
    print(f"  Bin0 ratio:             {results['bin0_ratio']:.4f}")
    print("-" * 60)
    print(f"  {'Bin':<20} {'Count':>10} {'Covered':>10} {'CovRate':>8} {'Gap':>8} {'|C|':>6} {'Score':>8}")
    print("-" * 60)
    for name, b in results["bins"].items():
        print(f"  {name:<20} {b['count']:>10,} {b['covered_count']:>10,} "
              f"{b['coverage_rate']:>8.4f} {b['gap']:>8.4f} "
              f"{b['mean_set_size']:>6.2f} {b['mean_score']:>8.4f}")
    print("=" * 60)
