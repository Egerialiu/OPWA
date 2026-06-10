import json
import os
import numpy as np

from config import (
    BIN_EDGES, BIN_NAMES, NUM_BINS,
    TARGET_COVERAGE, ALPHA, OUTPUT_DIR,
)


def compute_results(q_hat, cal_coverage, cal_pixel_count, bin_arrays):
    """Compute per-bin statistics and build the final results dict.

    Args:
        q_hat: float, CP threshold from calibration.
        cal_coverage: float, coverage on calibration set.
        cal_pixel_count: int, total valid calibration pixels.
        bin_arrays: dict with keys:
            pixel_counts, covered_counts, score_sums, set_size_sums
            each is a numpy array of length NUM_BINS.

    Returns:
        results: dict matching exp0_results.json spec.
    """
    pixel_counts = bin_arrays["pixel_counts"]
    covered_counts = bin_arrays["covered_counts"]
    score_sums = bin_arrays["score_sums"]
    set_size_sums = bin_arrays["set_size_sums"]

    total_valid = int(pixel_counts.sum())

    bins = {}
    for i in range(NUM_BINS):
        pc = int(pixel_counts[i])
        cc = int(covered_counts[i])
        cov = float(cc / pc) if pc > 0 else 0.0
        gap = max(0.0, TARGET_COVERAGE - cov)
        mss = float(set_size_sums[i] / pc) if pc > 0 else 0.0
        ms = float(score_sums[i] / pc) if pc > 0 else 0.0

        bins[BIN_NAMES[i]] = {
            "pixel_count": pc,
            "covered_count": cc,
            "coverage_rate": round(cov, 6),
            "gap": round(gap, 6),
            "mean_set_size": round(mss, 6),
            "mean_score": round(ms, 6),
        }

    # Overall test coverage
    overall_cov = float(covered_counts.sum() / total_valid) if total_valid > 0 else 0.0

    # Max gap across bins
    max_gap = max(b["gap"] for b in bins.values())

    # Bin 0 specific
    bin0 = bins[BIN_NAMES[0]]
    bin0_gap = bin0["gap"]
    bin0_pixel_ratio = float(bin0["pixel_count"] / total_valid) if total_valid > 0 else 0.0

    results = {
        "dataset": "foggy_cityscapes_beta0.02",
        "alpha": ALPHA,
        "q_hat": round(q_hat, 6),
        "calibration_pixel_count": cal_pixel_count,
        "calibration_coverage": round(cal_coverage, 6),
        "bins": bins,
        "overall_test_coverage": round(overall_cov, 6),
        "max_gap": round(max_gap, 6),
        "bin0_gap": round(bin0_gap, 6),
        "bin0_pixel_ratio": round(bin0_pixel_ratio, 6),
    }

    return results


def save_results(results, filepath=None):
    """Save results to JSON (and print summary)."""
    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, "exp0_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")
    return filepath


def print_summary(results):
    """Print a human-readable summary of the results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 0 — COVERAGE COLLAPSE EVALUATION")
    print("=" * 60)
    print(f"  Dataset:                {results['dataset']}")
    print(f"  Target coverage (1-α):  {1 - results['alpha']}")
    print(f"  q_hat:                  {results['q_hat']}")
    print(f"  Calibration coverage:   {results['calibration_coverage']}")
    print(f"  Calibration pixels:     {results['calibration_pixel_count']:,}")
    print(f"  Overall test coverage:  {results['overall_test_coverage']}")
    print(f"  Max gap:                {results['max_gap']}")
    print(f"  Bin0 gap:               {results['bin0_gap']}")
    print(f"  Bin0 pixel ratio:       {results['bin0_pixel_ratio']:.4f}")
    print("-" * 60)
    print(f"  {'Bin':<20} {'Pixels':>10} {'Covered':>10} {'CovRate':>8} {'Gap':>8} {'|C|':>6} {'Score':>8}")
    print("-" * 60)
    for name, b in results["bins"].items():
        print(f"  {name:<20} {b['pixel_count']:>10,} {b['covered_count']:>10,} "
              f"{b['coverage_rate']:>8.4f} {b['gap']:>8.4f} "
              f"{b['mean_set_size']:>6.2f} {b['mean_score']:>8.4f}")
    print("=" * 60)
