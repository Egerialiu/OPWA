#!/usr/bin/env python3
"""
Diagnosis script for Exp 0.
Performs tasks 1-3 + D1 + D2 in a single pass through the test set.

Tasks:
  1. Compute empty_set_rate per bin (add to exp0_results.json)
  2. Quantile binning → exp0_results_quantile.json + quantile_edges.json
  3. D1: t vs brightness correlation → diag_transmittance.csv
  4. D2: Bin membership visualization → diag_bin_visualization.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, "/root/opwa_v3/OPWA_v3/exp0")
from config import DEVICE, BIN_EDGES, BIN_NAMES, NUM_BINS, OUTPUT_DIR, TARGET_COVERAGE, ALPHA
from model_loader import load_segformer, load_depth_anything
from data_utils import get_test_files, match_gt
from transmittance import compute_transmittance
from gt_utils import load_gt_label_ids

print("Loading models...")
seg_model, processor = load_segformer()
seg_model.eval()
depth_model = load_depth_anything()
depth_model.eval()

test_files = get_test_files()
print(f"Test set: {len(test_files)} images")

# Accumulators (same as existing + empty_set)
bin_pixel_counts = np.zeros(NUM_BINS, dtype=np.int64)
bin_covered_counts = np.zeros(NUM_BINS, dtype=np.int64)
bin_score_sums = np.zeros(NUM_BINS, dtype=np.float64)
bin_set_size_sums = np.zeros(NUM_BINS, dtype=np.float64)
bin_empty_counts = np.zeros(NUM_BINS, dtype=np.int64)  # NEW

# t_value subsampling for quantile computation
t_samples = []  # list of arrays, will concat after
T_SUBSAMPLE = 100  # take every Nth pixel

# Per-image stats for D1 (first 20 images)
per_image_d1 = []

# Per-image t_maps & images for D2 (first 5 images)
per_image_d2 = []

q_hat = 0.513809  # from calibration

for idx, img_path in enumerate(tqdm(test_files, desc="Diagnosis inference")):
    # --- 1. SegFormer ---
    img_rgb = Image.open(img_path).convert("RGB")
    inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    logits = F.interpolate(logits, size=(1024, 2048), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)
    probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # --- 2. Depth ---
    img_bgr = cv2.imread(img_path)
    depth_raw = depth_model.infer_image(img_bgr)
    t_map = compute_transmittance(depth_raw)
    t_map = cv2.resize(t_map, (2048, 1024), interpolation=cv2.INTER_LINEAR)

    # --- 3. GT ---
    gt_path = match_gt(img_path)
    gt = load_gt_label_ids(gt_path)

    # --- 4. Per-pixel ---
    valid_mask = gt != 255
    gt_valid = gt[valid_mask]
    probs_valid = probs[valid_mask]
    t_valid = t_map[valid_mask]

    scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid]
    covered = scores <= q_hat
    set_sizes = (probs_valid >= (1.0 - q_hat)).sum(axis=1)
    empty_set = set_sizes == 0

    # Bin assignment
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

    # Collect t samples for quantile (every T_SUBSAMPLE-th valid pixel)
    t_valid_sub = t_valid[::T_SUBSAMPLE]
    t_samples.append(t_valid_sub)

    # D1: per-image stats (first 20)
    if idx < 20:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(gray.mean())
        contrast_std = float(gray.std())
        bright_ratio = float((gray >= 200).sum() / gray.size)

        bin_ratios = []
        for bin_i in range(NUM_BINS):
            mask = bin_indices == bin_i
            r = float(mask.sum() / max(len(gt_valid), 1))
            bin_ratios.append(r)

        per_image_d1.append({
            "filename": os.path.basename(img_path),
            "t_mean": float(t_valid.mean()),
            "brightness_mean": brightness_mean,
            "contrast_std": contrast_std,
            "bright_ratio": bright_ratio,
            **{f"bin{i}_ratio": bin_ratios[i] for i in range(NUM_BINS)},
        })

    # D2: store first 5 images for visualization
    if idx < 5:
        bin_color = np.zeros((1024, 2048), dtype=np.uint8)  # bin index per pixel
        bin_color[valid_mask] = bin_indices.astype(np.uint8)
        per_image_d2.append({
            "img_bgr": img_bgr.copy(),
            "t_map": t_map.copy(),
            "bin_color": bin_color,
            "filename": os.path.basename(img_path),
        })

# ================================================================
# Task 1: Compute empty_set_rate and update exp0_results.json
# ================================================================
print("\n=== Task 1: Empty Set Rate ===")
total_valid = int(bin_pixel_counts.sum())

# Load existing results
results_path = os.path.join(OUTPUT_DIR, "exp0_results.json")
with open(results_path) as f:
    results = json.load(f)

# Update bins with empty_set info
for bin_i, name in enumerate(BIN_NAMES):
    pc = int(bin_pixel_counts[bin_i])
    ec = int(bin_empty_counts[bin_i])
    esr = float(ec / pc) if pc > 0 else 0.0
    results["bins"][name]["empty_set_count"] = ec
    results["bins"][name]["empty_set_rate"] = round(esr, 6)
    print(f"  {name}: empty_set_rate = {esr:.4f} ({ec}/{pc})")

# Save updated results
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Updated {results_path}")


# ================================================================
# Task 2: Quantile binning
# ================================================================
print("\n=== Task 2: Quantile Binning ===")
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

# Save quantile edges
with open(os.path.join(OUTPUT_DIR, "quantile_edges.json"), "w") as f:
    json.dump({
        "q20": round(q20, 6),
        "q40": round(q40, 6),
        "q60": round(q60, 6),
        "q80": round(q80, 6),
        "bin_edges": [round(e, 6) for e in quantile_edges],
        "bin_names": quantile_names,
    }, f, indent=2)

# Re-do inference stats with quantile binning
# We re-process the accumulated data... but we don't have per-pixel data anymore.
# We need to re-do the full bin assignment.
# Actually, the inference already ran above with FIXED bins.
# We need to reload and re-process.

# Let's reload the models and re-process with quantile bins
print("Re-running inference with quantile bins...")

# Re-initialize accumulators
q_pixel_counts = np.zeros(NUM_BINS, dtype=np.int64)
q_covered_counts = np.zeros(NUM_BINS, dtype=np.int64)
q_score_sums = np.zeros(NUM_BINS, dtype=np.float64)
q_set_size_sums = np.zeros(NUM_BINS, dtype=np.float64)
q_empty_counts = np.zeros(NUM_BINS, dtype=np.int64)

for idx, img_path in enumerate(tqdm(test_files, desc="Quantile inference")):
    # --- 1. SegFormer ---
    img_rgb = Image.open(img_path).convert("RGB")
    inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    logits = F.interpolate(logits, size=(1024, 2048), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)
    probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # --- 2. Depth ---
    img_bgr = cv2.imread(img_path)
    depth_raw = depth_model.infer_image(img_bgr)
    t_map = compute_transmittance(depth_raw)
    t_map = cv2.resize(t_map, (2048, 1024), interpolation=cv2.INTER_LINEAR)

    # --- 3. GT ---
    gt_path = match_gt(img_path)
    gt = load_gt_label_ids(gt_path)

    # --- 4. Per-pixel with QUANTILE bin edges ---
    valid_mask = gt != 255
    gt_valid = gt[valid_mask]
    probs_valid = probs[valid_mask]
    t_valid = t_map[valid_mask]

    scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid]
    covered = scores <= q_hat
    set_sizes = (probs_valid >= (1.0 - q_hat)).sum(axis=1)
    empty_set = set_sizes == 0

    # Quantile bin assignment
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
        "pixel_count": pc,
        "covered_count": cc,
        "coverage_rate": round(cov, 6),
        "gap": round(gap, 6),
        "mean_set_size": round(mss, 6),
        "mean_score": round(ms, 6),
        "empty_set_count": ec,
        "empty_set_rate": round(esr, 6),
    }

q_overall_cov = float(q_covered_counts.sum() / q_total_valid) if q_total_valid > 0 else 0.0
q_max_gap = max(b["gap"] for b in q_bins.values())
q_bin0_name = quantile_names[0]
q_bin0_gap = q_bins[q_bin0_name]["gap"]
q_bin0_pixel_ratio = float(q_bins[q_bin0_name]["pixel_count"] / q_total_valid) if q_total_valid > 0 else 0.0

q_results = {
    "dataset": "foggy_cityscapes_beta0.02",
    "alpha": ALPHA,
    "q_hat": round(q_hat, 6),
    "bin_strategy": "quantile",
    "quantile_edges": [round(e, 6) for e in quantile_edges],
    "calibration_pixel_count": results["calibration_pixel_count"],
    "calibration_coverage": results["calibration_coverage"],
    "bins": q_bins,
    "overall_test_coverage": round(q_overall_cov, 6),
    "max_gap": round(q_max_gap, 6),
    "bin0_gap": round(q_bin0_gap, 6),
    "bin0_pixel_ratio": round(q_bin0_pixel_ratio, 6),
}

with open(os.path.join(OUTPUT_DIR, "exp0_results_quantile.json"), "w") as f:
    json.dump(q_results, f, indent=2)
print(f"  Saved exp0_results_quantile.json")

# Print summary
print("\n--- Quantile Bin Summary ---")
print(f"  {'Bin':<30} {'Pixels':>10} {'CovRate':>8} {'Gap':>8} {'Empty':>8}")
print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
for name, b in q_bins.items():
    print(f"  {name:<30} {b['pixel_count']:>10,} {b['coverage_rate']:>8.4f} {b['gap']:>8.4f} {b['empty_set_rate']:>8.4f}")
print(f"  Max gap: {q_max_gap:.4f}, Low-bin gap: {q_bin0_gap:.4f}")


# ================================================================
# Task 3 (D1): Save diag_transmittance.csv
# ================================================================
print("\n=== D1: Transmittance vs Brightness Correlation ===")
import csv

csv_path = os.path.join(OUTPUT_DIR, "diag_transmittance.csv")
fieldnames = [
    "filename", "t_mean", "brightness_mean", "contrast_std", "bright_ratio",
    "bin0_ratio", "bin1_ratio", "bin2_ratio", "bin3_ratio", "bin4_ratio",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(per_image_d1)
print(f"  Saved {csv_path} ({len(per_image_d1)} images)")

# Compute correlation
t_means = np.array([d["t_mean"] for d in per_image_d1])
contrast_stds = np.array([d["contrast_std"] for d in per_image_d1])
brightness_means = np.array([d["brightness_mean"] for d in per_image_d1])
bright_ratios = np.array([d["bright_ratio"] for d in per_image_d1])

corr_t_contrast = float(np.corrcoef(t_means, contrast_stds)[0, 1])
corr_t_brightness = float(np.corrcoef(t_means, brightness_means)[0, 1])
corr_t_bright_ratio = float(np.corrcoef(t_means, bright_ratios)[0, 1])

print(f"  t_mean vs contrast_std:   r = {corr_t_contrast:.4f}")
print(f"  t_mean vs brightness:     r = {corr_t_brightness:.4f}")
print(f"  t_mean vs bright_ratio:   r = {corr_t_bright_ratio:.4f}")

if corr_t_contrast >= 0.5:
    print(f"  → Transmittance physically plausible (corr >= 0.5) ✓")
else:
    print(f"  → Transmittance correlation low! Check depth direction")


# ================================================================
# Task 4 (D2): Bin membership visualization
# ================================================================
print("\n=== D2: Bin Membership Visualization ===")

BIN_COLORS = [(1,0,0,0.6), (1,0.65,0,0.6), (1,1,0,0.6), (0,1,0,0.6), (0,0,1,0.6)]
BIN_LABELS = ["Bin0 (t<0.2)", "Bin1 (0.2-0.4)", "Bin2 (0.4-0.6)", "Bin3 (0.6-0.8)", "Bin4 (0.8-1.0)"]

fig, axes = plt.subplots(5, 3, figsize=(18, 25))

for row in range(min(5, len(per_image_d2))):
    d = per_image_d2[row]
    img_rgb = cv2.cvtColor(d["img_bgr"], cv2.COLOR_BGR2RGB)

    # Left: original foggy
    axes[row, 0].imshow(img_rgb)
    axes[row, 0].set_title(f"Foggy: {d['filename'][:50]}")
    axes[row, 0].axis("off")

    # Middle: transmittance heatmap
    im = axes[row, 1].imshow(d["t_map"], cmap="jet", vmin=0, vmax=1)
    axes[row, 1].set_title("Transmittance t(x)")
    axes[row, 1].axis("off")

    # Right: bin membership (color overlay)
    overlay = np.zeros_like(img_rgb, dtype=np.float32)
    for bin_i, color in enumerate(BIN_COLORS):
        mask = d["bin_color"] == bin_i
        for c in range(3):
            overlay[mask, c] = color[c]
    # Blend with original
    blended = img_rgb.astype(np.float32) / 255.0 * 0.5 + overlay * 0.5
    axes[row, 2].imshow(np.clip(blended, 0, 1))
    axes[row, 2].set_title("Bin Membership (colored by t range)")
    axes[row, 2].axis("off")

# Color bar for t_map
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Transmittance t(x)")
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

fig.suptitle("D2: Bin Membership Analysis — Foggy Image | t(x) | Bin Colors", fontsize=16, y=1.02)

d2_path = os.path.join(OUTPUT_DIR, "diag_bin_visualization.png")
fig.savefig(d2_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {d2_path}")

# ================================================================
# D1 scatter plot
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(t_means, contrast_stds, alpha=0.7)
axes[0].set_xlabel("Mean t (transmittance)")
axes[0].set_ylabel("Contrast (RGB std)")
axes[0].set_title(f"t vs Contrast\nr = {corr_t_contrast:.4f}")
axes[0].grid(alpha=0.3)

axes[1].scatter(t_means, brightness_means, alpha=0.7, color="orange")
axes[1].set_xlabel("Mean t (transmittance)")
axes[1].set_ylabel("Brightness (mean)")
axes[1].set_title(f"t vs Brightness\nr = {corr_t_brightness:.4f}")
axes[1].grid(alpha=0.3)

axes[2].scatter(t_means, bright_ratios, alpha=0.7, color="green")
axes[2].set_xlabel("Mean t (transmittance)")
axes[2].set_ylabel("Bright Ratio (>200)")
axes[2].set_title(f"t vs Bright Ratio\nr = {corr_t_bright_ratio:.4f}")
axes[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "diag_transmittance_correlation.png"), dpi=150)
plt.close(fig)
print("  Saved diag_transmittance_correlation.png")

print("\n=== Diagnosis Complete ===")
print(f"  Updated: exp0_results.json (with empty_set_rate)")
print(f"  Created: exp0_results_quantile.json")
print(f"  Created: quantile_edges.json")
print(f"  Created: diag_transmittance.csv")
print(f"  Created: diag_bin_visualization.png")
print(f"  Created: diag_transmittance_correlation.png")
