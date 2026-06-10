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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp0"))
sys.path.insert(0, os.path.dirname(__file__))
from model_loader import load_segformer

from config_dcp import (
    DEVICE, OUTPUT_DIR, Q_HAT, TARGET_COVERAGE, ALPHA,
    DCP_BIN_EDGES, DCP_BIN_NAMES, NUM_DCP_BINS,
    ACDC_TARGET_H, ACDC_TARGET_W,
    compute_dcp, get_acdc_fog_files,
)

# sky class in train IDs
SKY_TRAIN_ID = 10


def run_experiment_a():
    """Experiment A: DCP stratification on ACDC Fog."""
    print("=" * 60)
    print("DIAGNOSTIC EXPERIMENT A: DCP Stratification on ACDC Fog")
    print("=" * 60)

    # ---- Load model ----
    print("Loading SegFormer-B0...")
    seg_model, processor = load_segformer()
    seg_model.eval()

    img_files, gt_files = get_acdc_fog_files()
    n_images = len(img_files)
    print(f"ACDC Fog val: {n_images} images")

    # ---- Accumulators for fixed bins ----
    fixed_pixel_counts = np.zeros(NUM_DCP_BINS, dtype=np.int64)
    fixed_covered_counts = np.zeros(NUM_DCP_BINS, dtype=np.int64)
    fixed_score_sums = np.zeros(NUM_DCP_BINS, dtype=np.float64)
    fixed_empty_counts = np.zeros(NUM_DCP_BINS, dtype=np.int64)

    # ---- Accumulators for quantile (deferred, need DCP values first) ----
    all_dcp_values = []    # subsampled DCP values for quantile computation
    all_scores = []         # per-pixel scores
    all_covered = []        # per-pixel covered flags
    all_dcp_full = []       # full DCP maps (for quantile binning)
    all_valid_masks = []    # valid masks
    DCP_SUBSAMPLE = 50      # take every 50th pixel for quantile edges

    # ---- Sky stats accumulators ----
    sky_pixel_count = 0
    sky_dcp_sum = 0.0
    total_valid_pixels = 0

    # ---- Per-image DCP mean for fog scoring ----
    per_image_dcp_means = []

    # ---- Visualizations ----
    vis_data = []

    for idx, (img_path, gt_path) in enumerate(tqdm(zip(img_files, gt_files),
                                                    total=n_images, desc="ACDC DCP")):
        # ---- 1. SegFormer inference ----
        img_rgb = Image.open(img_path).convert("RGB")
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = seg_model(**inputs).logits
        logits = F.interpolate(logits, size=(ACDC_TARGET_H, ACDC_TARGET_W),
                               mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)
        probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,19)

        # ---- 2. DCP computation ----
        img_cv = cv2.imread(img_path)  # BGR
        img_rgb_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        dcp_map = compute_dcp(img_rgb_cv)  # (H,W), [0,1]

        # ---- 3. GT loading (train IDs directly) ----
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)  # (1080,1920), uint8

        # ---- 4. Per-pixel evaluation ----
        valid_mask = gt != 255
        gt_valid = gt[valid_mask]
        probs_valid = probs[valid_mask]
        dcp_valid = dcp_map[valid_mask]

        scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid.astype(np.int64)]
        covered = scores <= Q_HAT
        empty_set = (probs_valid >= (1.0 - Q_HAT)).sum(axis=1) == 0

        # Store for quantile computation
        all_scores.append(scores)
        all_covered.append(covered)
        all_dcp_full.append(dcp_map)
        all_valid_masks.append(valid_mask)
        sub_idx = np.arange(len(dcp_valid))[::DCP_SUBSAMPLE]
        all_dcp_values.append(dcp_valid[sub_idx])

        # ---- 5. Fixed bin accumulation ----
        bin_indices = np.digitize(dcp_valid, DCP_BIN_EDGES) - 1

        for bin_i in range(NUM_DCP_BINS):
            mask = bin_indices == bin_i
            count = mask.sum()
            if count == 0:
                continue
            fixed_pixel_counts[bin_i] += count
            fixed_covered_counts[bin_i] += covered[mask].sum()
            fixed_score_sums[bin_i] += scores[mask].sum()
            fixed_empty_counts[bin_i] += empty_set[mask].sum()

        # ---- 6. Sky stats ----
        sky_mask = gt_valid == SKY_TRAIN_ID
        sky_pixel_count += sky_mask.sum()
        sky_dcp_sum += dcp_valid[sky_mask].sum()

        total_valid_pixels += len(gt_valid)

        # ---- 7. Per-image stats ----
        per_image_dcp_means.append(float(dcp_valid.mean()))

        # ---- 8. Store visualization data (first 5) ----
        if idx < 5:
            vis_data.append({
                "filename": os.path.basename(img_path),
                "img_bgr": img_cv,
                "dcp_map": dcp_map,
                "dcp_mean": float(dcp_valid.mean()),
                "coverage": float(covered.mean()),
                "score_mean": float(scores.mean()),
                "bin_indices": bin_indices,
            })

    # ================================================================
    # Fixed Bin Results
    # ================================================================
    total_valid = int(fixed_pixel_counts.sum())

    fixed_bins = {}
    for i, name in enumerate(DCP_BIN_NAMES):
        pc = int(fixed_pixel_counts[i])
        cc = int(fixed_covered_counts[i])
        ec = int(fixed_empty_counts[i])
        cov = float(cc / pc) if pc > 0 else 0.0
        gap = max(0.0, TARGET_COVERAGE - cov)
        ms = float(fixed_score_sums[i] / pc) if pc > 0 else 0.0
        esr = float(ec / pc) if pc > 0 else 0.0
        fixed_bins[name] = {
            "bin_id": i,
            "dcp_range": [float(DCP_BIN_EDGES[i]), float(DCP_BIN_EDGES[i + 1])],
            "pixel_count": pc,
            "pixel_ratio": round(pc / total_valid, 6) if total_valid > 0 else 0.0,
            "coverage_rate": round(cov, 6),
            "gap": round(gap, 6),
            "mean_score": round(ms, 6),
            "empty_set_rate": round(esr, 6),
        }

    overall_cov = float(fixed_covered_counts.sum() / total_valid) if total_valid > 0 else 0.0
    overall_mean_score = float(fixed_score_sums.sum() / total_valid) if total_valid > 0 else 0.0

    results_fixed = {
        "dataset": "acdc_fog_val",
        "condition_variable": "DCP",
        "q_hat": Q_HAT,
        "target_coverage": TARGET_COVERAGE,
        "alpha": ALPHA,
        "overall_coverage": round(overall_cov, 6),
        "overall_mean_score": round(overall_mean_score, 6),
        "bin_strategy": "fixed",
        "bin_edges": DCP_BIN_EDGES,
        "bins": fixed_bins,
    }

    fp_fixed = os.path.join(OUTPUT_DIR, "diag_A_dcp_acdc_fixed.json")
    with open(fp_fixed, "w") as f:
        json.dump(results_fixed, f, indent=2)
    print(f"\nSaved {fp_fixed}")

    # Print summary
    print("\n--- Fixed Bin Summary ---")
    print(f"  {'Bin':<25} {'Pixels':>10} {'CovRate':>8} {'Gap':>8} {'MeanScore':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for name, b in fixed_bins.items():
        print(f"  {name:<25} {b['pixel_count']:>10,} {b['coverage_rate']:>8.4f} "
              f"{b['gap']:>8.4f} {b['mean_score']:>10.4f}")
    gaps_fixed = [b["gap"] for b in fixed_bins.values()]
    print(f"  Max gap: {max(gaps_fixed):.4f} (Bin {list(gaps_fixed).index(max(gaps_fixed))})")

    # ================================================================
    # Quantile Bin Results
    # ================================================================
    print("\n=== Computing quantile bins ===")
    dcp_all = np.concatenate(all_dcp_values)
    q20 = float(np.quantile(dcp_all, 0.20))
    q40 = float(np.quantile(dcp_all, 0.40))
    q60 = float(np.quantile(dcp_all, 0.60))
    q80 = float(np.quantile(dcp_all, 0.80))
    quantile_edges = [0.0, q20, q40, q60, q80, 1.0]
    quantile_names = [
        f"qtile0_dcp0.00_{q20:.4f}",
        f"qtile1_dcp{q20:.4f}_{q40:.4f}",
        f"qtile2_dcp{q40:.4f}_{q60:.4f}",
        f"qtile3_dcp{q60:.4f}_{q80:.4f}",
        f"qtile4_dcp{q80:.4f}_1.00",
    ]
    print(f"  Quantile edges: q20={q20:.4f}, q40={q40:.4f}, q60={q60:.4f}, q80={q80:.4f}")

    # Accumulate quantile bins
    q_pixel_counts = np.zeros(NUM_DCP_BINS, dtype=np.int64)
    q_covered_counts = np.zeros(NUM_DCP_BINS, dtype=np.int64)
    q_score_sums = np.zeros(NUM_DCP_BINS, dtype=np.float64)
    q_empty_counts = np.zeros(NUM_DCP_BINS, dtype=np.int64)

    for idx in range(len(all_scores)):
        dcp_map = all_dcp_full[idx]
        valid_mask = all_valid_masks[idx]
        dcp_valid = dcp_map[valid_mask]
        scores = all_scores[idx]
        covered = all_covered[idx]
        empty_set = covered == False  # reuse same computation for empty set
        # Actually compute empty_set properly
        # We need to recompute: we don't have set sizes stored. Re-approximate:
        # We stored scores and covered, but empty_set needs (probs >= 1-q_hat).sum()==0
        # But we already have scores. A pixel is covered if score <= Q_HAT.
        # Empty set: if no class has prob >= 1-Q_HAT. But we can't compute from scores alone.
        # Let's just use covered counts and skip empty_set for quantile.

        q_bin_indices = np.digitize(dcp_valid, quantile_edges) - 1
        for bin_i in range(NUM_DCP_BINS):
            mask = q_bin_indices == bin_i
            count = mask.sum()
            if count == 0:
                continue
            q_pixel_counts[bin_i] += count
            q_covered_counts[bin_i] += covered[mask].sum()
            q_score_sums[bin_i] += scores[mask].sum()

    q_total_valid = int(q_pixel_counts.sum())
    q_bins = {}
    for i in range(NUM_DCP_BINS):
        pc = int(q_pixel_counts[i])
        cc = int(q_covered_counts[i])
        cov = float(cc / pc) if pc > 0 else 0.0
        gap = max(0.0, TARGET_COVERAGE - cov)
        ms = float(q_score_sums[i] / pc) if pc > 0 else 0.0
        q_bins[quantile_names[i]] = {
            "bin_id": i,
            "dcp_range": [round(quantile_edges[i], 6), round(quantile_edges[i + 1], 6)],
            "pixel_count": pc,
            "pixel_ratio": round(pc / q_total_valid, 6) if q_total_valid > 0 else 0.0,
            "coverage_rate": round(cov, 6),
            "gap": round(gap, 6),
            "mean_score": round(ms, 6),
        }

    q_overall_cov = float(q_covered_counts.sum() / q_total_valid) if q_total_valid > 0 else 0.0
    q_overall_ms = float(q_score_sums.sum() / q_total_valid) if q_total_valid > 0 else 0.0

    results_quantile = {
        "dataset": "acdc_fog_val",
        "condition_variable": "DCP",
        "q_hat": Q_HAT,
        "target_coverage": TARGET_COVERAGE,
        "alpha": ALPHA,
        "overall_coverage": round(q_overall_cov, 6),
        "overall_mean_score": round(q_overall_ms, 6),
        "bin_strategy": "quantile",
        "quantile_edges": [round(e, 6) for e in quantile_edges],
        "bins": q_bins,
    }

    fp_quantile = os.path.join(OUTPUT_DIR, "diag_A_dcp_acdc_quantile.json")
    with open(fp_quantile, "w") as f:
        json.dump(results_quantile, f, indent=2)
    print(f"Saved {fp_quantile}")

    print("\n--- Quantile Bin Summary ---")
    print(f"  {'Bin':<30} {'Pixels':>10} {'CovRate':>8} {'Gap':>8} {'MeanScore':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for name, b in q_bins.items():
        print(f"  {name:<30} {b['pixel_count']:>10,} {b['coverage_rate']:>8.4f} "
              f"{b['gap']:>8.4f} {b['mean_score']:>10.4f}")
    gaps_q = [b["gap"] for b in q_bins.values()]
    print(f"  Max gap: {max(gaps_q):.4f} (qtile {list(gaps_q).index(max(gaps_q))})")

    # ================================================================
    # Sky Statistics
    # ================================================================
    sky_dcp_mean = float(sky_dcp_sum / sky_pixel_count) if sky_pixel_count > 0 else 0.0
    sky_pixel_ratio = float(sky_pixel_count / total_valid) if total_valid > 0 else 0.0

    sky_stats = {
        "sky_pixel_ratio": round(sky_pixel_ratio, 6),
        "sky_DCP_mean": round(sky_dcp_mean, 6),
        "sky_pixel_count": int(sky_pixel_count),
        "total_valid_pixels": total_valid,
        "diagnosis": (
            "无问题" if sky_dcp_mean < 0.05 else
            ("需在论文中说明" if sky_dcp_mean < 0.20 else
             "严重天空误判")
        ),
    }

    fp_sky = os.path.join(OUTPUT_DIR, "diag_A_sky_stats.json")
    with open(fp_sky, "w") as f:
        json.dump(sky_stats, f, indent=2)
    print(f"\nSaved {fp_sky}")
    print(f"Sky pixel ratio: {sky_pixel_ratio:.4f}")
    print(f"Sky DCP mean: {sky_dcp_mean:.4f} — {sky_stats['diagnosis']}")

    # ================================================================
    # Visualizations
    # ================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Coverage bar plot (fixed bins)
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_labels = [f"DCP∈[{DCP_BIN_EDGES[i]:.1f},{DCP_BIN_EDGES[i+1]:.1f})" for i in range(NUM_DCP_BINS)]
    cov_rates = [fixed_bins[n]["coverage_rate"] for n in DCP_BIN_NAMES]
    x = np.arange(NUM_DCP_BINS)
    bars = ax.bar(x, cov_rates, width=0.6, color="steelblue", edgecolor="navy")
    ax.axhline(y=TARGET_COVERAGE, color="red", linestyle="--", linewidth=2,
               label=f"Target = {TARGET_COVERAGE:.0%}")
    for bar, rate in zip(bars, cov_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage Rate")
    ax.set_xlabel("Dark Channel Prior Bin (DCP ↓ clear, DCP ↑ fog)")
    ax.set_title("Diagnostic A: Coverage Rate by DCP Bin (ACDC Fog, q_hat=0.5138)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "diag_A_dcp_coverage_fixed.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved {p1}")

    # Coverage bar plot (quantile bins)
    fig, ax = plt.subplots(figsize=(10, 6))
    q_labels = [f"DCP∈[{quantile_edges[i]:.3f},{quantile_edges[i+1]:.3f})" for i in range(NUM_DCP_BINS)]
    q_cov = [q_bins[quantile_names[i]]["coverage_rate"] for i in range(NUM_DCP_BINS)]
    bars = ax.bar(x, q_cov, width=0.6, color="coral", edgecolor="darkred")
    ax.axhline(y=TARGET_COVERAGE, color="red", linestyle="--", linewidth=2,
               label=f"Target = {TARGET_COVERAGE:.0%}")
    for bar, rate in zip(bars, q_cov):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage Rate")
    ax.set_xlabel("DCP Quantile Bin (← clear / foggy →)")
    ax.set_title("Diagnostic A: Coverage Rate by DCP Quantile Bin (ACDC Fog)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "diag_A_dcp_coverage_quantile.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved {p2}")

    # DCP visualization (first 5)
    n_vis = min(5, len(vis_data))
    fig, axes = plt.subplots(n_vis, 3, figsize=(18, 4 * n_vis))
    if n_vis == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_vis):
        d = vis_data[row]
        img_rgb = cv2.cvtColor(d["img_bgr"], cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"Original: {d['filename'][:40]}")
        axes[row, 0].axis("off")

        im = axes[row, 1].imshow(d["dcp_map"], cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title(f"DCP map, mean={d['dcp_mean']:.4f}")
        for level in [0.1, 0.2, 0.3, 0.5]:
            axes[row, 1].contour(d["dcp_map"], levels=[level],
                                 colors="white", linewidths=0.5, alpha=0.6)
        axes[row, 1].axis("off")

        # Bin color overlay
        bin_color = np.zeros((ACDC_TARGET_H, ACDC_TARGET_W), dtype=np.uint8)
        valid_mask_all = False  # just show DCP bins on full image
        # Use DCP map for bin assignment
        dcp_flat = d["dcp_map"].reshape(-1)
        bin_idx_all = np.digitize(dcp_flat, DCP_BIN_EDGES) - 1
        bin_color_vis = bin_idx_all.reshape(ACDC_TARGET_H, ACDC_TARGET_W).astype(np.float32)
        bin_color_vis = bin_color_vis / max(NUM_DCP_BINS - 1, 1)

        im2 = axes[row, 2].imshow(bin_color_vis, cmap="viridis", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Cov={d['coverage']:.3f}, Score={d['score_mean']:.4f}")
        axes[row, 2].axis("off")

    fig.subplots_adjust(right=0.92)
    cbar_ax1 = fig.add_axes([0.93, 0.55, 0.01, 0.35])
    cbar1 = fig.colorbar(im, cax=cbar_ax1)
    cbar1.set_label("DCP Value")
    cbar1.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    cbar_ax2 = fig.add_axes([0.93, 0.1, 0.01, 0.35])
    cbar2 = fig.colorbar(im2, cax=cbar_ax2)
    cbar2.set_label("Bin")
    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar2.set_ticklabels(["0", "1", "2", "3", "4"])

    fig.suptitle("Diagnostic A: DCP Visualization on ACDC Fog", fontsize=14, y=1.01)
    p3 = os.path.join(OUTPUT_DIR, "diag_A_dcp_vis.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p3}")

    # ================================================================
    # Summary
    # ================================================================
    max_gap_fixed = max(b["gap"] for b in fixed_bins.values())
    max_gap_q = max(b["gap"] for b in q_bins.values())
    has_gap = max_gap_fixed >= 0.10 or max_gap_q >= 0.10

    print("\n" + "=" * 60)
    print("EXPERIMENT A SUMMARY")
    print("=" * 60)
    print(f"  Fixed bins max gap: {max_gap_fixed:.4f}")
    print(f"  Quantile bins max gap: {max_gap_q:.4f}")
    print(f"  Overall coverage: {overall_cov:.4f}")
    print(f"  Overall mean score: {overall_mean_score:.4f}")
    print(f"  Sky DCP mean: {sky_dcp_mean:.4f} (threshold 0.05)")
    print(f"  Gap detected (≥10pp): {'YES ✓' if has_gap else 'NO'}")
    print("=" * 60)

    return results_fixed, results_quantile, sky_stats, has_gap


if __name__ == "__main__":
    run_experiment_a()
