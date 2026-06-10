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
    ACDC_TARGET_H, ACDC_TARGET_W,
    compute_dcp, get_acdc_image_paths, get_acdc_fog_files,
)

N_EXTREME = 20  # number of images for extreme fog subset
OUTLIER_THRESHOLD = 3.0  # z-score threshold for outlier detection in fog_score


def compute_fog_scores():
    """Compute fog_score for each ACDC Fog image.

    fog_score = DCP_mean × (1 - contrast_std_norm)

    Returns list of dicts with filename, DCP_mean, contrast_std, fog_score.
    """
    print("Computing fog scores for all ACDC Fog images...")
    img_paths = get_acdc_image_paths()
    results = []

    contrast_stds = []

    for img_path in tqdm(img_paths, desc="Fog scoring"):
        img_cv = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        dcp = compute_dcp(img_rgb)
        dcp_mean = float(dcp.mean())

        contrast_std = float(img_norm.std())
        contrast_stds.append(contrast_std)

        results.append({
            "filename": os.path.basename(img_path),
            "full_path": img_path,
            "dcp_mean": dcp_mean,
            "contrast_std": contrast_std,
        })

    # Normalize contrast_std across all images
    contrast_arr = np.array([r["contrast_std"] for r in results])
    c_min, c_max = contrast_arr.min(), contrast_arr.max()
    c_range = c_max - c_min
    if c_range > 1e-8:
        for r in results:
            r["contrast_std_norm"] = (r["contrast_std"] - c_min) / c_range
    else:
        for r in results:
            r["contrast_std_norm"] = 0.5

    # Compute fog_score
    for r in results:
        r["fog_score"] = r["dcp_mean"] * (1.0 - r["contrast_std_norm"])

    # Sort by fog_score descending
    results.sort(key=lambda x: x["fog_score"], reverse=True)

    return results


def run_experiment_b():
    """Experiment B: Extreme fog subset analysis."""
    print("=" * 60)
    print("DIAGNOSTIC EXPERIMENT B: Extreme Fog Subset (ACDC Fog)")
    print("=" * 60)

    # ---- Step 1: Compute fog scores ----
    all_scores = compute_fog_scores()
    n_total = len(all_scores)

    print(f"\nTotal images: {n_total}")
    print(f"Fog score range: {all_scores[-1]['fog_score']:.4f} - {all_scores[0]['fog_score']:.4f}")

    # Print top/bottom 5 for verification
    print("\nTop 5 foggiest:")
    for r in all_scores[:5]:
        print(f"  {r['filename']:45s} DCP={r['dcp_mean']:.4f} "
              f"contrast={r['contrast_std']:.4f} fog_score={r['fog_score']:.4f}")
    print("Bottom 5 clearest:")
    for r in all_scores[-5:]:
        print(f"  {r['filename']:45s} DCP={r['dcp_mean']:.4f} "
              f"contrast={r['contrast_std']:.4f} fog_score={r['fog_score']:.4f}")

    # ---- Step 2: Select top N extreme fog images ----
    extreme = all_scores[:N_EXTREME]
    extreme_filenames = [r["filename"] for r in extreme]
    extreme_scores = [r["fog_score"] for r in extreme]

    # Save fog score list
    fog_list = {
        "total_images": n_total,
        "n_extreme": N_EXTREME,
        "selection_method": "fog_score = DCP_mean * (1 - contrast_std_norm)",
        "top20_files": extreme_filenames,
        "fog_scores": [round(s, 6) for s in extreme_scores],
        "all_scores": [
            {
                "filename": r["filename"],
                "dcp_mean": round(r["dcp_mean"], 6),
                "contrast_std": round(r["contrast_std"], 6),
                "fog_score": round(r["fog_score"], 6),
            }
            for r in all_scores
        ],
    }
    fp_list = os.path.join(OUTPUT_DIR, "diag_B_extreme_fog_list.json")
    with open(fp_list, "w") as f:
        json.dump(fog_list, f, indent=2)
    print(f"\nSaved {fp_list}")

    # ---- Step 3: Coverage evaluation on extreme subset ----
    print(f"\nRunning SegFormer on {N_EXTREME} extreme fog images...")
    seg_model, processor = load_segformer()
    seg_model.eval()

    # Get full image/GT pair list
    all_img_files, all_gt_files = get_acdc_fog_files()

    # Build filename → path mapping
    img_map = {os.path.basename(p): p for p in all_img_files}
    gt_map = {os.path.basename(p).replace("_rgb_anon.png", "_gt_labelTrainIds.png"): p
              for p in all_gt_files}

    # Accumulators
    total_valid_pixels = 0
    total_covered = 0
    total_score_sum = 0.0
    total_empty = 0

    per_image_results = []

    for r in extreme:
        filename = r["filename"]
        img_path = img_map.get(filename)
        # Build GT path
        gt_basename = filename.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
        gt_path = gt_map.get(gt_basename)

        if img_path is None or gt_path is None:
            print(f"  WARNING: path not found for {filename}")
            continue

        # Inference
        img_rgb = Image.open(img_path).convert("RGB")
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = seg_model(**inputs).logits
        logits = F.interpolate(logits, size=(ACDC_TARGET_H, ACDC_TARGET_W),
                               mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)
        probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # GT
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        # Per-pixel
        valid_mask = gt != 255
        gt_valid = gt[valid_mask]
        probs_valid = probs[valid_mask]

        scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid.astype(np.int64)]
        covered = scores <= Q_HAT
        set_sizes = (probs_valid >= (1.0 - Q_HAT)).sum(axis=1)
        empty_set = set_sizes == 0

        n_valid = len(gt_valid)
        n_covered = int(covered.sum())
        score_mean = float(scores.mean())

        total_valid_pixels += n_valid
        total_covered += n_covered
        total_score_sum += float(scores.sum())
        total_empty += int(empty_set.sum())

        per_image_results.append({
            "filename": filename,
            "valid_pixels": n_valid,
            "covered_count": n_covered,
            "coverage_rate": round(n_covered / n_valid, 6) if n_valid > 0 else 0.0,
            "mean_score": round(score_mean, 6),
            "fog_score": r["fog_score"],
        })

    # Overall stats
    overall_cov = float(total_covered / total_valid_pixels) if total_valid_pixels > 0 else 0.0
    overall_mean_score = float(total_score_sum / total_valid_pixels) if total_valid_pixels > 0 else 0.0
    overall_empty_rate = float(total_empty / total_valid_pixels) if total_valid_pixels > 0 else 0.0
    overall_gap = max(0.0, TARGET_COVERAGE - overall_cov)

    coverage_results = {
        "n_images": N_EXTREME,
        "total_valid_pixels": total_valid_pixels,
        "overall_coverage": round(overall_cov, 6),
        "gap": round(overall_gap, 6),
        "target_coverage": TARGET_COVERAGE,
        "mean_score": round(overall_mean_score, 6),
        "empty_set_rate": round(overall_empty_rate, 6),
        "q_hat": Q_HAT,
        "per_image": per_image_results,
    }

    fp_cov = os.path.join(OUTPUT_DIR, "diag_B_extreme_fog_coverage.json")
    with open(fp_cov, "w") as f:
        json.dump(coverage_results, f, indent=2)
    print(f"Saved {fp_cov}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXTREME FOG SUBSET RESULTS")
    print("=" * 60)
    print(f"  Images: {N_EXTREME}")
    print(f"  Overall coverage: {overall_cov:.4f} (gap: {overall_gap:.4f})")
    print(f"  Mean score: {overall_mean_score:.4f}")
    print(f"  Empty set rate: {overall_empty_rate:.6f}")
    print(f"  Reference — ACDC Fog mean_score: 0.109, Foggy CS: 0.239")
    has_gap = overall_gap >= 0.10
    print(f"  Gap >= 10pp: {'YES ✓' if has_gap else 'NO'}")

    if n_total > 0:
        print(f"\n  Extreme subset fog_score: {extreme_scores[0]:.4f} - {extreme_scores[-1]:.4f}")
        print(f"  Full set fog_score range: {all_scores[0]['fog_score']:.4f} - {all_scores[-1]['fog_score']:.4f}")

    # ---- Step 4: Fog score histogram ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    fog_vals = [r["fog_score"] for r in all_scores]
    ax.hist(fog_vals, bins=20, color="steelblue", edgecolor="navy", alpha=0.7)
    ax.axvline(x=extreme_scores[-1] if extreme_scores else 0, color="red",
               linestyle="--", linewidth=2, label=f"Top-{N_EXTREME} threshold")
    ax.set_xlabel("Fog Score")
    ax.set_ylabel("Number of Images")
    ax.set_title("Diagnostic B: Fog Score Distribution (ACDC Fog)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "diag_B_fog_scores_hist.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved {p1}")

    print("=" * 60)
    return coverage_results, all_scores


if __name__ == "__main__":
    run_experiment_b()
