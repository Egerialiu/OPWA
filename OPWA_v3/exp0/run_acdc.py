#!/usr/bin/env python3
"""
ACDC Fog experiment for Exp 0.
Same split CP pipeline as Foggy Cityscapes, but on real fog data.

Key differences from Foggy Cityscapes:
  - GT uses *_gt_labelTrainIds.png (train IDs directly, no mapping needed)
  - Resolution: 1920x1080 (slightly different from Cityscapes 2048x1024)
  - Calibration: Cityscapes val 250 (q_hat=0.5138, unchanged)
  - Dataset: 100 images only
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm
import json
import os
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/root/opwa_v3/OPWA_v3/exp0")
from config import DEVICE, BIN_EDGES, BIN_NAMES, NUM_BINS, OUTPUT_DIR, TARGET_COVERAGE, ALPHA
from model_loader import load_segformer, load_depth_anything
from transmittance import compute_transmittance

# ================================================================
# ACDC paths & config
# ================================================================
ACDC_ROOT = "/gz-data/ACDC"
ACDC_IMG_DIR = os.path.join(ACDC_ROOT, "rgb_anon_trainvaltest", "rgb_anon", "fog", "val")
ACDC_GT_DIR = os.path.join(ACDC_ROOT, "gt", "fog", "val")

Q_HAT = 0.513809  # from Cityscapes calibration, fixed
CAL_PIXEL_COUNT = 449381364  # from Cityscapes calibration
CAL_COVERAGE = 0.9

OUTPUT_PREFIX = "exp0_acdc"

# ================================================================
# Data functions
# ================================================================
def get_acdc_files():
    """Return (image_paths, gt_paths) sorted lists for ACDC fog/val."""
    img_pattern = os.path.join(ACDC_IMG_DIR, "*/*_rgb_anon.png")
    img_files = sorted(glob.glob(img_pattern))

    gt_files = []
    for img_path in img_files:
        # e.g. .../GOPR0476/GOPR0476_frame_000761_rgb_anon.png
        basename = os.path.basename(img_path)  # GOPR0476_frame_000761_rgb_anon.png
        gt_basename = basename.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
        gt_dir = os.path.join(ACDC_GT_DIR, os.path.basename(os.path.dirname(img_path)))
        gt_path = os.path.join(gt_dir, gt_basename)
        gt_files.append(gt_path)

    return img_files, gt_files

# ================================================================
# Inference
# ================================================================
def run_inference():
    """Run split CP on ACDC fog/val (100 images)."""
    print("Loading models...")
    seg_model, processor = load_segformer()
    seg_model.eval()
    depth_model = load_depth_anything()
    depth_model.eval()

    img_files, gt_files = get_acdc_files()
    print(f"ACDC Fog val: {len(img_files)} images")

    # Verify first GT
    gt_test = cv2.imread(gt_files[0], cv2.IMREAD_UNCHANGED)
    print(f"  GT shape: {gt_test.shape}, dtype: {gt_test.dtype}")
    print(f"  GT unique values: {sorted(np.unique(gt_test).tolist()[:15])}")
    print(f"  GT resolution: {gt_test.shape[1]}x{gt_test.shape[0]}")

    # Inference params: ACDC = 1920x1080
    TARGET_H, TARGET_W = 1080, 1920

    # Accumulators (fixed bins)
    bin_pixel_counts = np.zeros(NUM_BINS, dtype=np.int64)
    bin_covered_counts = np.zeros(NUM_BINS, dtype=np.int64)
    bin_score_sums = np.zeros(NUM_BINS, dtype=np.float64)
    bin_set_size_sums = np.zeros(NUM_BINS, dtype=np.float64)
    bin_empty_counts = np.zeros(NUM_BINS, dtype=np.int64)

    # t samples for quantile
    t_samples = []
    T_SUBSAMPLE = 50  # every 50th pixel

    # Per-image results for D1/D2
    per_image_results = []

    for idx, (img_path, gt_path) in enumerate(tqdm(zip(img_files, gt_files), total=len(img_files), desc="ACDC inference")):
        # --- 1. SegFormer ---
        img_rgb = Image.open(img_path).convert("RGB")
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = seg_model(**inputs).logits
        logits = F.interpolate(logits, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)
        probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # --- 2. Depth Anything ---
        img_bgr = cv2.imread(img_path)
        depth_raw = depth_model.infer_image(img_bgr)
        t_map = compute_transmittance(depth_raw)
        t_map = cv2.resize(t_map, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

        # --- 3. GT (train IDs directly) ---
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)  # (1080, 1920), uint8
        # ACDC gt_labelTrainIds has train IDs (0-18, 255=ignore)

        # --- 4. Per-pixel ---
        valid_mask = gt != 255
        gt_valid = gt[valid_mask]
        probs_valid = probs[valid_mask]
        t_valid = t_map[valid_mask]

        scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid]
        covered = scores <= Q_HAT
        set_sizes = (probs_valid >= (1.0 - Q_HAT)).sum(axis=1)
        empty_set = set_sizes == 0

        # Fixed bin assignment
        bin_indices = np.digitize(t_valid, BIN_EDGES) - 1

        for bin_i in range(NUM_BINS):
            mask = bin_indices == bin_i
            count = mask.sum()
            if count == 0:
                continue
            bin_pixel_counts[bin_i] += count
            bin_covered_counts[bin_i] += covered[mask].sum()
            bin_score_sums[bin_i] += scores[mask].sum()
            bin_set_size_sums[bin_i] += set_sizes[mask].sum()
            bin_empty_counts[bin_i] += empty_set[mask].sum()

        # t samples for quantile
        t_valid_sub = t_valid[::T_SUBSAMPLE]
        t_samples.append(t_valid_sub)

        # Per-image stats
        if idx < 20:
            per_image_results.append({
                "filename": os.path.basename(img_path),
                "t_mean": float(t_valid.mean()),
            })

        # Store first 5 for visualization
        if idx < 5:
            per_image_results[-1]["t_map"] = t_map
            per_image_results[-1]["bin_color"] = np.full((TARGET_H, TARGET_W), 255, dtype=np.uint8)
            per_image_results[-1]["bin_color"][valid_mask] = bin_indices.astype(np.uint8)
            per_image_results[-1]["img_bgr"] = img_bgr

    # ================================================================
    # Build fixed-bin results
    # ================================================================
    total_valid = int(bin_pixel_counts.sum())

    bins_fixed = {}
    for i, name in enumerate(BIN_NAMES):
        pc = int(bin_pixel_counts[i])
        cc = int(bin_covered_counts[i])
        ec = int(bin_empty_counts[i])
        cov = float(cc / pc) if pc > 0 else 0.0
        gap = max(0.0, TARGET_COVERAGE - cov)
        mss = float(bin_set_size_sums[i] / pc) if pc > 0 else 0.0
        ms = float(bin_score_sums[i] / pc) if pc > 0 else 0.0
        esr = float(ec / pc) if pc > 0 else 0.0
        bins_fixed[name] = {
            "pixel_count": pc, "covered_count": cc,
            "coverage_rate": round(cov, 6), "gap": round(gap, 6),
            "mean_set_size": round(mss, 6), "mean_score": round(ms, 6),
            "empty_set_count": ec, "empty_set_rate": round(esr, 6),
        }

    overall_cov = float(bin_covered_counts.sum() / total_valid) if total_valid > 0 else 0.0
    max_gap = max(b["gap"] for b in bins_fixed.values())
    bin0_gap = bins_fixed[BIN_NAMES[0]]["gap"]
    bin0_ratio = float(bins_fixed[BIN_NAMES[0]]["pixel_count"] / total_valid) if total_valid > 0 else 0.0

    results_fixed = {
        "dataset": "acdc_fog_val",
        "calibration_source": "cityscapes_clear_val_250",
        "alpha": ALPHA,
        "q_hat": round(Q_HAT, 6),
        "calibration_pixel_count": CAL_PIXEL_COUNT,
        "calibration_coverage": CAL_COVERAGE,
        "bins": bins_fixed,
        "overall_test_coverage": round(overall_cov, 6),
        "max_gap": round(max_gap, 6),
        "bin0_gap": round(bin0_gap, 6),
        "bin0_pixel_ratio": round(bin0_ratio, 6),
    }

    filepath = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_results.json")
    with open(filepath, "w") as f:
        json.dump(results_fixed, f, indent=2)
    print(f"\nSaved {filepath}")

    # Print fixed-bin summary
    print("\n--- ACDC Fixed Bin Summary ---")
    print(f"  {'Bin':<20} {'Pixels':>10} {'CovRate':>8} {'Gap':>8} {'|C|':>6} {'Empty':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for name, b in bins_fixed.items():
        print(f"  {name:<20} {b['pixel_count']:>10,} {b['coverage_rate']:>8.4f} {b['gap']:>8.4f} {b['mean_set_size']:>6.2f} {b['empty_set_rate']:>8.4f}")
    print(f"  Overall coverage: {overall_cov:.4f}, Max gap: {max_gap:.4f}, Bin0 ratio: {bin0_ratio:.4f}")

    # ================================================================
    # Quantile binning
    # ================================================================
    print("\n=== Quantile Binning ===")
    t_all = np.concatenate(t_samples)
    q20 = float(np.quantile(t_all, 0.20))
    q40 = float(np.quantile(t_all, 0.40))
    q60 = float(np.quantile(t_all, 0.60))
    q80 = float(np.quantile(t_all, 0.80))
    quantile_edges = [0.0, q20, q40, q60, q80, 1.01]
    quantile_names = [
        f"qtile0_t0.00_{q20:.4f}",
        f"qtile1_t{q20:.4f}_{q40:.4f}",
        f"qtile2_t{q40:.4f}_{q60:.4f}",
        f"qtile3_t{q60:.4f}_{q80:.4f}",
        f"qtile4_t{q80:.4f}_1.00",
    ]
    print(f"  q20={q20:.4f}, q40={q40:.4f}, q60={q60:.4f}, q80={q80:.4f}")

    # Re-run with quantile bins
    q_pixel_counts = np.zeros(NUM_BINS, dtype=np.int64)
    q_covered_counts = np.zeros(NUM_BINS, dtype=np.int64)
    q_score_sums = np.zeros(NUM_BINS, dtype=np.float64)
    q_set_size_sums = np.zeros(NUM_BINS, dtype=np.float64)
    q_empty_counts = np.zeros(NUM_BINS, dtype=np.int64)

    for idx, (img_path, gt_path) in enumerate(tqdm(zip(img_files, gt_files), total=len(img_files), desc="ACDC quantile")):
        img_rgb = Image.open(img_path).convert("RGB")
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = seg_model(**inputs).logits
        logits = F.interpolate(logits, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)
        probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

        img_bgr = cv2.imread(img_path)
        depth_raw = depth_model.infer_image(img_bgr)
        t_map = compute_transmittance(depth_raw)
        t_map = cv2.resize(t_map, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        valid_mask = gt != 255
        gt_valid = gt[valid_mask]
        probs_valid = probs[valid_mask]
        t_valid = t_map[valid_mask]

        scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid]
        covered = scores <= Q_HAT
        set_sizes = (probs_valid >= (1.0 - Q_HAT)).sum(axis=1)
        empty_set = set_sizes == 0

        q_bin_indices = np.digitize(t_valid, quantile_edges) - 1

        for bin_i in range(NUM_BINS):
            mask = q_bin_indices == bin_i
            count = mask.sum()
            if count == 0:
                continue
            q_pixel_counts[bin_i] += count
            q_covered_counts[bin_i] += covered[mask].sum()
            q_score_sums[bin_i] += scores[mask].sum()
            q_set_size_sums[bin_i] += set_sizes[mask].sum()
            q_empty_counts[bin_i] += empty_set[mask].sum()

    # Build quantile results
    q_total_valid = int(q_pixel_counts.sum())
    q_bins = {}
    for i in range(NUM_BINS):
        pc = int(q_pixel_counts[i])
        cc = int(q_covered_counts[i])
        ec = int(q_empty_counts[i])
        cov = float(cc / pc) if pc > 0 else 0.0
        gap = max(0.0, TARGET_COVERAGE - cov)
        mss = float(q_set_size_sums[i] / pc) if pc > 0 else 0.0
        ms = float(q_score_sums[i] / pc) if pc > 0 else 0.0
        esr = float(ec / pc) if pc > 0 else 0.0
        q_bins[quantile_names[i]] = {
            "pixel_count": pc, "covered_count": cc,
            "coverage_rate": round(cov, 6), "gap": round(gap, 6),
            "mean_set_size": round(mss, 6), "mean_score": round(ms, 6),
            "empty_set_count": ec, "empty_set_rate": round(esr, 6),
        }

    q_overall_cov = float(q_covered_counts.sum() / q_total_valid) if q_total_valid > 0 else 0.0
    q_max_gap = max(b["gap"] for b in q_bins.values())
    q_bin0_name = quantile_names[0]
    q_bin0_gap = q_bins[q_bin0_name]["gap"]
    q_bin0_ratio = float(q_bins[q_bin0_name]["pixel_count"] / q_total_valid) if q_total_valid > 0 else 0.0

    q_results = {
        "dataset": "acdc_fog_val",
        "calibration_source": "cityscapes_clear_val_250",
        "alpha": ALPHA,
        "q_hat": round(Q_HAT, 6),
        "bin_strategy": "quantile",
        "quantile_edges": [round(e, 6) for e in quantile_edges],
        "calibration_pixel_count": CAL_PIXEL_COUNT,
        "calibration_coverage": CAL_COVERAGE,
        "bins": q_bins,
        "overall_test_coverage": round(q_overall_cov, 6),
        "max_gap": round(q_max_gap, 6),
        "bin0_gap": round(q_bin0_gap, 6),
        "bin0_pixel_ratio": round(q_bin0_ratio, 6),
    }

    q_filepath = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_results_quantile.json")
    with open(q_filepath, "w") as f:
        json.dump(q_results, f, indent=2)
    print(f"\nSaved {q_filepath}")

    # Print quantile summary
    print("\n--- ACDC Quantile Bin Summary ---")
    print(f"  {'Bin':<30} {'Pixels':>10} {'CovRate':>8} {'Gap':>8} {'Empty':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for name, b in q_bins.items():
        print(f"  {name:<30} {b['pixel_count']:>10,} {b['coverage_rate']:>8.4f} {b['gap']:>8.4f} {b['empty_set_rate']:>8.4f}")
    print(f"  Max gap: {q_max_gap:.4f}, Low-bin gap: {q_bin0_gap:.4f}")

    # ================================================================
    # Coverage gap plot (fixed bins)
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_names_short = [f"t∈[{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})" for i in range(NUM_BINS)]
    cov_rates = [bins_fixed[n]["coverage_rate"] for n in BIN_NAMES]
    x = np.arange(NUM_BINS)
    bars = ax.bar(x, cov_rates, width=0.6, color="steelblue", edgecolor="navy")
    ax.axhline(y=TARGET_COVERAGE, color="red", linestyle="--", linewidth=2,
               label=f"Target 1-α = {TARGET_COVERAGE:.0%}")
    for bar, rate in zip(bars, cov_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names_short, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage Rate")
    ax.set_xlabel("Transmittance Bin")
    ax.set_title("ACDC Fog: Coverage Rate by Transmittance Bin\n(Standard Split CP, α=0.1, q_hat from Cityscapes)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_coverage_gap_plot.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  Saved {p1}")

    # Coverage gap plot (quantile bins)
    fig, ax = plt.subplots(figsize=(10, 6))
    q_bin_names_short = [f"t∈[{quantile_edges[i]:.4f},{quantile_edges[i+1]:.4f})" for i in range(NUM_BINS)]
    q_cov_rates = [q_bins[quantile_names[i]]["coverage_rate"] for i in range(NUM_BINS)]
    bars = ax.bar(x, q_cov_rates, width=0.6, color="coral", edgecolor="darkred")
    ax.axhline(y=TARGET_COVERAGE, color="red", linestyle="--", linewidth=2,
               label=f"Target 1-α = {TARGET_COVERAGE:.0%}")
    for bar, rate in zip(bars, q_cov_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(q_bin_names_short, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage Rate")
    ax.set_xlabel("Transmittance Bin (Quantile)")
    ax.set_title("ACDC Fog: Coverage Rate by Quantile Bin\n(Standard Split CP, α=0.1, q_hat from Cityscapes)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_coverage_gap_quantile_plot.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  Saved {p2}")

    # Transmittance visualization (first 5)
    fig, axes = plt.subplots(min(5, len(per_image_results)), 2, figsize=(14, 4 * min(5, len(per_image_results))))
    if min(5, len(per_image_results)) == 1:
        axes = axes.reshape(1, -1)
    for row in range(min(5, len(per_image_results))):
        d = per_image_results[row]
        img_rgb = cv2.cvtColor(d["img_bgr"], cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"ACDC Fog: {d['filename'][:50]}")
        axes[row, 0].axis("off")
        im = axes[row, 1].imshow(d["t_map"], cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title(f"t(x) map, mean_t={d['t_mean']:.4f}")
        for level in [0.2, 0.4, 0.6, 0.8]:
            axes[row, 1].contour(d["t_map"], levels=[level], colors="white", linewidths=0.5, alpha=0.6)
        axes[row, 1].axis("off")
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Transmittance t(x)")
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.suptitle("ACDC Fog: Transmittance Visualization", fontsize=14, y=1.01)
    p3 = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_transmittance_vis.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p3}")

    # ================================================================
    # Print decision-ready summary
    # ================================================================
    print("\n" + "=" * 60)
    print("ACDC FOG EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Fixed bins - bin0_gap:         {bin0_gap:.4f}")
    print(f"  Fixed bins - max_gap:           {max_gap:.4f}")
    print(f"  Quantile bins - qtile0_gap:     {q_bin0_gap:.4f}")
    print(f"  Quantile bins - max_gap:        {q_max_gap:.4f}")
    print(f"  Quantile t range qtile0:        [0.0, {q20:.4f})")
    print(f"  Overall test coverage:          {overall_cov:.4f}")
    print(f"  Total valid pixels:             {total_valid:,}")
    print("=" * 60)

    # Comparison with Foggy
    print("\nComparison with Foggy Cityscapes:")
    fc_path = os.path.join(OUTPUT_DIR, "exp0_results_quantile.json")
    if os.path.exists(fc_path):
        with open(fc_path) as f:
            fc = json.load(f)
        print(f"  Foggy CS qtile0_gap: {fc.get('bin0_gap', 'N/A')}")
        print(f"  ACDC     qtile0_gap: {q_bin0_gap:.4f}")
        larger = "ACDC" if q_bin0_gap > fc.get('bin0_gap', 0) else "Foggy"
        print(f"  → {larger} has larger gap")

    return results_fixed, q_results


if __name__ == "__main__":
    run_inference()
