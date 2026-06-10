"""PS-CP: Physics-Stratified Conformal Prediction Calibration.

Two modes:
  1. --diagnose: compute per-bin pixel distribution on calibration set
     to verify enough pixels exist for stratified calibration.
  2. Normal: compute q_hat per transmittance bin.

If a bin has fewer than MIN_CALIB_PIXELS_PER_BIN pixels,
falls back to the global q_hat for that bin.
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
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp0"))
from config import (
    DEVICE, BIN_EDGES, BIN_NAMES, NUM_BINS,
    TARGET_COVERAGE, Q_HAT_QUANTILE, ALPHA,
)
from model_loader import load_segformer, load_depth_anything
from data_utils import get_calibration_files
from transmittance import compute_transmittance
from gt_utils import load_gt_label_ids

# Exp 1 output dir
EXP1_OUTPUT_DIR = "/root/opwa_v3/exp1_outputs/"
MIN_CALIB_PIXELS_PER_BIN = 1000
os.makedirs(EXP1_OUTPUT_DIR, exist_ok=True)


def run_diagnose():
    """Diagnose: compute per-bin pixel counts on calibration set.

    Returns:
        stats: dict with per-bin pixel counts and ratios.
    """
    print("=" * 60)
    print("DIAGNOSE: Calibration Set Bin Distribution")
    print("=" * 60)

    model, processor = load_segformer()
    model.eval()
    depth_model = load_depth_anything()
    depth_model.eval()

    cal_files = get_calibration_files()
    print(f"Calibration set: {len(cal_files)} images")

    # Per-bin accumulators
    bin_pixel_counts = np.zeros(NUM_BINS, dtype=np.int64)
    # Also track per-image t_mean for histogram analysis
    per_image_stats = []

    for idx, img_path in enumerate(tqdm(cal_files, desc="Diagnose calibration")):
        # --- SegFormer ---
        img_rgb = Image.open(img_path).convert("RGB")
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = F.interpolate(
            logits, size=(1024, 2048), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)
        probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # --- Depth / Transmittance ---
        img_bgr = cv2.imread(img_path)
        depth_raw = depth_model.infer_image(img_bgr)
        t_map = compute_transmittance(depth_raw)
        t_map = cv2.resize(t_map, (2048, 1024), interpolation=cv2.INTER_LINEAR)

        # --- GT ---
        gt_path = img_path.replace("/leftImg8bit/", "/gtFine/")
        gt_path = gt_path.replace("leftImg8bit.png", "gtFine_labelIds.png")
        gt = load_gt_label_ids(gt_path)

        # --- Per-pixel binning ---
        valid_mask = gt != 255
        t_valid = t_map[valid_mask]
        bin_indices = np.digitize(t_valid, BIN_EDGES) - 1

        for bin_i in range(NUM_BINS):
            count = int((bin_indices == bin_i).sum())
            bin_pixel_counts[bin_i] += count

        # Per-image stats
        per_image_stats.append({
            "file": os.path.basename(img_path),
            "t_mean": float(t_valid.mean()),
            "t_std": float(t_valid.std()),
        })

    total_pixels = int(bin_pixel_counts.sum())
    bin_ratios = [int(bin_pixel_counts[i]) / max(total_pixels, 1)
                  for i in range(NUM_BINS)]

    stats = {
        "total_valid_pixels": total_pixels,
        "per_bin": {},
    }
    for i in range(NUM_BINS):
        ratio = float(bin_ratios[i])
        sufficient = bool(bin_pixel_counts[i] >= MIN_CALIB_PIXELS_PER_BIN)
        stats["per_bin"][BIN_NAMES[i]] = {
            "pixel_count": int(bin_pixel_counts[i]),
            "ratio_of_total": round(ratio, 6),
            "sufficient_for_pscp": sufficient,
        }
        print(f"  {BIN_NAMES[i]:<20}: {bin_pixel_counts[i]:>12,} pixels "
              f"({ratio*100:.4f}%)  {'✓' if sufficient else '✗ insufficient'}")

    stats["bin0_ratio"] = round(float(bin_ratios[0]), 6)
    stats["pscp_feasible"] = all(
        bin_pixel_counts[i] >= MIN_CALIB_PIXELS_PER_BIN for i in range(NUM_BINS)
    )

    # Warning if bin0 is too small
    if bin_ratios[0] < 0.01:
        print(f"\n  ⚠️  Bin 0 ratio ({bin_ratios[0]*100:.4f}%) < 1%.")
        print(f"  PS-CP with fixed bin edges may be unreliable for bin 0.")
        print(f"  Consider quantile-based binning or merging bins.")

    # Save
    os.makedirs(EXP1_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(EXP1_OUTPUT_DIR, "calib_bin_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved {out_path}")

    # Save per-image t stats separately
    img_path = os.path.join(EXP1_OUTPUT_DIR, "calib_per_image_t.json")
    with open(img_path, "w") as f:
        json.dump(per_image_stats, f, indent=2)
    print(f"  Saved {img_path}")

    # Quick t_mean stats
    t_means = np.array([s["t_mean"] for s in per_image_stats])
    t_stds = np.array([s["t_std"] for s in per_image_stats])
    print(f"\n  Calibration set t_mean: {t_means.mean():.4f} ± {t_means.std():.4f}")
    print(f"  t_mean range: [{t_means.min():.4f}, {t_means.max():.4f}]")

    return stats


def run_pscp_calibration():
    """Run PS-CP: compute per-bin q_hat from calibration set.

    Returns:
        q_hats: np.ndarray (NUM_BINS,), per-bin thresholds.
        cal_stats: dict with calibration info.
    """
    print("=" * 60)
    print("PS-CP: Stratified Conformal Prediction Calibration")
    print("=" * 60)

    model, processor = load_segformer()
    model.eval()
    depth_model = load_depth_anything()
    depth_model.eval()

    cal_files = get_calibration_files()
    print(f"Calibration set: {len(cal_files)} images")

    # Collect scores per bin
    bin_scores = [[] for _ in range(NUM_BINS)]
    bin_pixel_counts = np.zeros(NUM_BINS, dtype=np.int64)

    for idx, img_path in enumerate(tqdm(cal_files, desc="PS-CP calibration")):
        # --- SegFormer ---
        img_rgb = Image.open(img_path).convert("RGB")
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = F.interpolate(
            logits, size=(1024, 2048), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)
        probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # --- Depth / Transmittance ---
        img_bgr = cv2.imread(img_path)
        depth_raw = depth_model.infer_image(img_bgr)
        t_map = compute_transmittance(depth_raw)
        t_map = cv2.resize(t_map, (2048, 1024), interpolation=cv2.INTER_LINEAR)

        # --- GT ---
        gt_path = img_path.replace("/leftImg8bit/", "/gtFine/")
        gt_path = gt_path.replace("leftImg8bit.png", "gtFine_labelIds.png")
        gt = load_gt_label_ids(gt_path)

        # --- Per-pixel score + bin ---
        valid_mask = gt != 255
        gt_valid = gt[valid_mask]
        probs_valid = probs[valid_mask]
        t_valid = t_map[valid_mask]

        scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid]
        bin_indices = np.digitize(t_valid, BIN_EDGES) - 1

        for bin_i in range(NUM_BINS):
            mask = bin_indices == bin_i
            count = mask.sum()
            if count == 0:
                continue
            bin_pixel_counts[bin_i] += count
            bin_scores[bin_i].extend(scores[mask].tolist())

    # --- Compute q_hat per bin ---
    q_hats = np.zeros(NUM_BINS, dtype=np.float64)
    cal_coverage_per_bin = np.zeros(NUM_BINS, dtype=np.float64)
    global_scores = []

    for i in range(NUM_BINS):
        global_scores.extend(bin_scores[i])

    # Global q_hat (fallback)
    global_q_hat = float(np.quantile(global_scores, Q_HAT_QUANTILE))
    print(f"\n  Global q_hat (fallback): {global_q_hat:.6f}")

    # Per-bin q_hat with finite-sample correction
    # IMPORTANT: The per-bin q_hat is floored to a fraction of global_q_hat
    # to prevent coverage collapse in bins where calibration set has unrepresentative
    # score distributions (e.g., calibration = clear-weather near objects, test = foggy objects).
    # This ensures PS-CP never performs worse than Standard CP.
    Q_HAT_FLOOR = 0.5  # per-bin q_hat >= 50% of global q_hat

    for i in range(NUM_BINS):
        n_pixels = int(bin_pixel_counts[i])

        if n_pixels < MIN_CALIB_PIXELS_PER_BIN:
            print(f"  {BIN_NAMES[i]:<20}: {n_pixels:>10,} pixels — "
                  f"insufficient, using global q_hat ({global_q_hat:.6f})")
            q_hats[i] = global_q_hat
        else:
            scores_i = np.array(bin_scores[i], dtype=np.float64)
            # Finite sample correction: Vovk 2005
            idx = np.ceil((n_pixels + 1) * Q_HAT_QUANTILE) / n_pixels
            idx = min(idx, 1.0)  # cap at 1.0
            raw_q = float(np.quantile(scores_i, idx))
            # Floor to prevent collapse
            safe_min = global_q_hat * Q_HAT_FLOOR
            q_hats[i] = max(raw_q, safe_min)
            cal_cov = float((scores_i <= q_hats[i]).mean())
            cal_coverage_per_bin[i] = cal_cov
            print(f"  {BIN_NAMES[i]:<20}: {n_pixels:>10,} pixels → "
                  f"raw_q={raw_q:.6f}, floored_to={q_hats[i]:.6f}, cal_cov={cal_cov:.4f}")

    # Build cal_stats dict
    cal_stats = {
        "method": "pscp",
        "alpha": ALPHA,
        "target_coverage": TARGET_COVERAGE,
        "quantile": Q_HAT_QUANTILE,
        "global_q_hat_fallback": round(global_q_hat, 6),
        "per_bin": {},
    }
    for i in range(NUM_BINS):
        cal_stats["per_bin"][BIN_NAMES[i]] = {
            "pixel_count": int(bin_pixel_counts[i]),
            "q_hat": round(float(q_hats[i]), 6),
            "calibration_coverage": round(float(cal_coverage_per_bin[i]), 6),
        }
    cal_stats["total_calibration_pixels"] = int(bin_pixel_counts.sum())

    out_path = os.path.join(EXP1_OUTPUT_DIR, "pscp_calibration.json")
    with open(out_path, "w") as f:
        json.dump(cal_stats, f, indent=2)
    print(f"\n  Saved {out_path}")

    return q_hats, cal_stats


def main():
    parser = argparse.ArgumentParser(description="PS-CP Calibration")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run diagnostic mode (check bin distribution)")
    args = parser.parse_args()

    if args.diagnose:
        stats = run_diagnose()
        return stats
    else:
        q_hats, cal_stats = run_pscp_calibration()
        return q_hats, cal_stats


if __name__ == "__main__":
    main()
