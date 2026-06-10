"""PS-CP: Physics-Stratified Conformal Prediction — Test Set Evaluation.

Evaluates per-bin coverage using per-bin q_hat thresholds
from PS-CP calibration. Generates exp1_results.json with the same
format as exp0_results.json for direct comparison.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp0"))
from config import (
    DEVICE, BIN_EDGES, BIN_NAMES, NUM_BINS,
    TARGET_COVERAGE, ALPHA,
)
from model_loader import load_segformer, load_depth_anything
from data_utils import get_test_files, match_gt
from transmittance import compute_transmittance
from gt_utils import load_gt_label_ids

EXP1_OUTPUT_DIR = "/root/opwa_v3/exp1_outputs/"
os.makedirs(EXP1_OUTPUT_DIR, exist_ok=True)


def evaluate_pscp(q_hats, max_images=None):
    """Evaluate PS-CP on Foggy Cityscapes test set.

    Args:
        q_hats: np.ndarray (NUM_BINS,), per-bin thresholds from PS-CP calibration.
        max_images: optional int limit for debugging.

    Returns:
        results: dict matching standard exp0_results.json format.
        per_image_data: list of per-image results for plotting.
    """
    print("=" * 60)
    print("PS-CP: Stratified Test Set Evaluation")
    print("=" * 60)

    model, processor = load_segformer()
    model.eval()
    depth_model = load_depth_anything()
    depth_model.eval()

    test_files = get_test_files()
    if max_images is not None:
        test_files = test_files[:max_images]
    print(f"Test set: {len(test_files)} images")

    # Per-bin accumulators
    bin_pixel_counts = np.zeros(NUM_BINS, dtype=np.int64)
    bin_covered_counts = np.zeros(NUM_BINS, dtype=np.int64)
    bin_score_sums = np.zeros(NUM_BINS, dtype=np.float64)
    bin_set_size_sums = np.zeros(NUM_BINS, dtype=np.float64)
    bin_empty_counts = np.zeros(NUM_BINS, dtype=np.int64)

    # Per-image results for visualization
    per_image_data = []

    for idx, img_path in enumerate(tqdm(test_files, desc="PS-CP test inference")):
        result = _process_one_image(
            img_path, model, processor, depth_model,
            q_hats, bin_pixel_counts, bin_covered_counts,
            bin_score_sums, bin_set_size_sums, bin_empty_counts,
        )
        per_image_data.append(result)

    # --- Build results JSON ---
    total_valid = int(bin_pixel_counts.sum())

    bins = {}
    for i in range(NUM_BINS):
        pc = int(bin_pixel_counts[i])
        cc = int(bin_covered_counts[i])
        ec = int(bin_empty_counts[i])
        cov = float(cc / pc) if pc > 0 else 0.0
        gap = max(0.0, TARGET_COVERAGE - cov)
        mss = float(bin_set_size_sums[i] / pc) if pc > 0 else 0.0
        ms = float(bin_score_sums[i] / pc) if pc > 0 else 0.0
        esr = float(ec / pc) if pc > 0 else 0.0

        bins[BIN_NAMES[i]] = {
            "pixel_count": pc,
            "covered_count": cc,
            "coverage_rate": round(cov, 6),
            "gap": round(gap, 6),
            "mean_set_size": round(mss, 6),
            "mean_score": round(ms, 6),
            "empty_set_count": ec,
            "empty_set_rate": round(esr, 6),
        }

    overall_cov = float(bin_covered_counts.sum() / total_valid) if total_valid > 0 else 0.0
    max_gap = max(b["gap"] for b in bins.values())
    bin0 = bins[BIN_NAMES[0]]
    bin0_gap = bin0["gap"]
    bin0_pixel_ratio = float(bin0["pixel_count"] / total_valid) if total_valid > 0 else 0.0

    results = {
        "dataset": "foggy_cityscapes_beta0.02",
        "method": "pscp",
        "alpha": ALPHA,
        "q_hats": [round(float(q), 6) for q in q_hats],
        "target_coverage": TARGET_COVERAGE,
        "bins": bins,
        "overall_test_coverage": round(overall_cov, 6),
        "max_gap": round(max_gap, 6),
        "bin0_gap": round(bin0_gap, 6),
        "bin0_pixel_ratio": round(bin0_pixel_ratio, 6),
    }

    # Save
    out_path = os.path.join(EXP1_OUTPUT_DIR, "exp1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Print summary
    print(f"\n  {'Bin':<20} {'Pixels':>10} {'Covered':>10} {'CovRate':>8} {'Gap':>8} {'|C|':>6}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")
    for name, b in bins.items():
        print(f"  {name:<20} {b['pixel_count']:>10,} {b['covered_count']:>10,} "
              f"{b['coverage_rate']:>8.4f} {b['gap']:>8.4f} {b['mean_set_size']:>6.2f}")
    print(f"\n  Overall test coverage:  {overall_cov:.6f}")
    print(f"  Max gap:                {max_gap:.6f}")
    print(f"  Bin0 gap:               {bin0_gap:.6f}")
    print(f"  Bin0 pixel ratio:       {bin0_pixel_ratio:.4f}")

    return results, per_image_data


def _process_one_image(img_path, model, processor, depth_model,
                       q_hats, bin_pixel_counts, bin_covered_counts,
                       bin_score_sums, bin_set_size_sums, bin_empty_counts):
    """Process a single test image, accumulate per-bin stats in-place.

    Args:
        img_path: str, path to foggy image.
        model: SegFormer model.
        processor: SegFormer image processor.
        depth_model: Depth Anything V2 model.
        q_hats: np.ndarray (NUM_BINS,), thresholds to use.

    In-place modifies the six bin_* arrays.
    Returns a dict with image-level info.
    """
    # --- 1. SegFormer inference ---
    img_rgb = Image.open(img_path).convert("RGB")
    inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = F.interpolate(
        logits, size=(1024, 2048), mode="bilinear", align_corners=False
    )
    probs = F.softmax(logits, dim=1)
    probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # --- 2. Depth / Transmittance ---
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
    bin_indices = np.digitize(t_valid, BIN_EDGES) - 1

    for bin_i in range(NUM_BINS):
        mask = bin_indices == bin_i
        count = mask.sum()
        if count == 0:
            continue

        # Use this bin's q_hat
        q = q_hats[bin_i]
        covered = scores[mask] <= q
        set_sizes = (probs_valid[mask] >= (1.0 - q)).sum(axis=1)
        empty_set = set_sizes == 0

        bin_pixel_counts[bin_i] += count
        bin_covered_counts[bin_i] += covered.sum()
        bin_score_sums[bin_i] += scores[mask].sum()
        bin_set_size_sums[bin_i] += set_sizes.sum()
        bin_empty_counts[bin_i] += empty_set.sum()

    return {
        "path": img_path,
        "t_map": t_map,
        "mean_t": float(t_valid.mean()),
        "bin0_ratio": float((bin_indices == 0).sum()) / max(len(gt_valid), 1),
    }


def compare_with_standard():
    """Load exp0_results.json and exp1_results.json, print comparison table."""
    import json
    import os

    exp0_path = "/root/opwa_v3/exp0_outputs/exp0_results.json"
    exp1_path = os.path.join(EXP1_OUTPUT_DIR, "exp1_results.json")

    if not os.path.exists(exp1_path):
        print("exp1_results.json not found. Run PS-CP evaluation first.")
        return

    with open(exp0_path) as f:
        exp0 = json.load(f)
    with open(exp1_path) as f:
        exp1 = json.load(f)

    print("\n" + "=" * 70)
    print("COMPARISON: Standard CP vs PS-CP")
    print("=" * 70)
    print(f"  {'Bin':<20} {'Std_CovRate':>12} {'Std_Gap':>10} "
          f"{'PS-CP_CovRate':>14} {'PS-CP_Gap':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*14} {'-'*12}")

    for name in BIN_NAMES:
        s = exp0["bins"].get(name, {})
        p = exp1["bins"].get(name, {})
        print(f"  {name:<20} {s.get('coverage_rate', 0):>12.4f} "
              f"{s.get('gap', 0):>10.4f} {p.get('coverage_rate', 0):>14.4f} "
              f"{p.get('gap', 0):>12.4f}")

    print(f"\n  Overall test coverage:")
    print(f"    Standard CP: {exp0.get('overall_test_coverage', 0):.4f}")
    print(f"    PS-CP:       {exp1.get('overall_test_coverage', 0):.4f}")
    print(f"  Target:          {TARGET_COVERAGE:.4f}")
    print("=" * 70)

    # Check if PS-CP improves bin0
    s_bin0 = exp0["bins"][BIN_NAMES[0]]["coverage_rate"]
    p_bin0 = exp1["bins"][BIN_NAMES[0]]["coverage_rate"]
    improvement = p_bin0 - s_bin0
    print(f"\n  Bin0 improvement: {improvement*100:.2f}pp "
          f"({s_bin0*100:.2f}% → {p_bin0*100:.2f}%)")
    if improvement > 0.05:
        print(f"  ✓ PS-CP significantly improves coverage in dense fog regions.")
    else:
        print(f"  ⚠️  PS-CP only marginally improves dense fog coverage.")
        print(f"  Consider quantile binning or alternative stratification.")


if __name__ == "__main__":
    # For standalone testing, load q_hats from calibration output
    cal_path = os.path.join(EXP1_OUTPUT_DIR, "pscp_calibration.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        q_hats = np.array([
            cal["per_bin"][name]["q_hat"] for name in BIN_NAMES
        ], dtype=np.float64)
        print(f"Loaded q_hats from {cal_path}")
        evaluate_pscp(q_hats)
    else:
        print(f"No calibration found at {cal_path}")
        print("Run pscp_calibration.py first.")
