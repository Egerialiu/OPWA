import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

"""
Diagnostic Experiments Entry Point
===================================
Run both experiments A and B sequentially.

Usage:
    python OPWA_v3/dcp分层实验/run_all.py

The experiments can also be run individually:
    python OPWA_v3/dcp分层实验/experiment_A.py
    python OPWA_v3/dcp分层实验/experiment_B.py

Note: Experiment C (ACDC Night) is SKIPPED — ACDC dataset only contains 'fog' condition.
"""

import sys
import os
import json
import argparse
from datetime import datetime


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp0"))
sys.path.insert(0, os.path.dirname(__file__))  # for config_dcp imports
from config import DEVICE


def main():
    parser = argparse.ArgumentParser(description="DCP Diagnostic Experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["A", "B", "all"],
                        help="Which experiment to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate code paths without GPU")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Diagnostic Experiments — {datetime.now()}")
    print(f"Device: {DEVICE}")
    print(f"Output: /root/opwa_v3/exp1_outputs/")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run: validating imports...")
        from experiment_A import run_experiment_a
        from experiment_B import run_experiment_b
        print("  ✓ experiment_A.py imports OK")
        print("  ✓ experiment_B.py imports OK")
        print("Dry run complete.\n")
        return

    results = {}

    if args.experiment in ("A", "all"):
        print(f"\n{'=' * 60}")
        print("RUNNING EXPERIMENT A: DCP Stratification on ACDC Fog")
        print(f"{'=' * 60}")
        from experiment_A import run_experiment_a
        res_a_fixed, res_a_quantile, sky_stats, has_gap_a = run_experiment_a()
        results["A_fixed"] = res_a_fixed
        results["A_quantile"] = res_a_quantile
        results["A_sky"] = sky_stats
        results["A_has_gap"] = has_gap_a

    if args.experiment in ("B", "all"):
        print(f"\n{'=' * 60}")
        print("RUNNING EXPERIMENT B: Extreme Fog Subset")
        print(f"{'=' * 60}")
        from experiment_B import run_experiment_b
        cov_results, fog_scores = run_experiment_b()
        results["B_coverage"] = cov_results
        results["B_has_gap"] = cov_results.get("gap", 1.0) >= 0.10

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE — SUMMARY")
    print("=" * 60)

    if "A_has_gap" in results:
        fa = results["A_fixed"]
        fb = results.get("B_coverage", {})
        max_gap_fixed = max(b["gap"] for b in fa.get("bins", {}).values())

        print(f"\nExperiment A (DCP on ACDC Fog):")
        print(f"  Fixed bins max gap: {max_gap_fixed:.4f}")
        if results["A_has_gap"]:
            print(f"  ⚠️  GAP DETECTED ≥ 10pp")
        else:
            print(f"  No significant gap detected")

    if "B_has_gap" in results:
        cov = results["B_coverage"]
        print(f"\nExperiment B (Extreme Fog Top-20):")
        print(f"  Coverage: {cov.get('overall_coverage', 0):.4f}")
        print(f"  Gap: {cov.get('gap', 0):.4f}")
        print(f"  Mean score: {cov.get('mean_score', 0):.4f}")
        if results.get("B_has_gap", False):
            print(f"  ⚠️  GAP DETECTED ≥ 10pp")
        else:
            print(f"  No significant gap detected")

    print(f"\nExperiment C (ACDC Night): SKIPPED — no night data in ACDC dataset")
    print("=" * 60)

    # Determine next steps
    triggers_main = []
    triggers_all_clear = True

    if results.get("A_has_gap", False):
        triggers_main.append("DCP分层最大gap >= 10pp")
        triggers_all_clear = False
    if results.get("B_has_gap", False):
        triggers_main.append("极端雾子集gap >= 10pp")
        triggers_all_clear = False

    print(f"\nDecision:")
    if triggers_main:
        print(f"  主线触发: {' / '.join(triggers_main)}")
        print(f"  推荐: 进入主线方案（混合物理先验 PS-CP）")
    elif triggers_all_clear:
        print(f"  所有实验 gap < 10pp")
        print(f"  推荐: 执行备选方案（自适应预测集压缩）")


if __name__ == "__main__":
    main()
