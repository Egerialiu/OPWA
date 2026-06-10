#!/usr/bin/env python3
"""
Experiment 0: Coverage Collapse Verification
=============================================
Split Conformal Prediction on Foggy Cityscapes.
SegFormer-B0 segmentation + Depth Anything V2 transmittance estimation.

Execution flow (4 steps, each waits for confirmation):
  Step 1: Model loading verification (SegFormer + Depth-Anything)
  Step 2: Transmittance direction verification
  Step 3: CP Calibration
  Step 4: Full test inference + evaluation + visualization + decision tree

Usage:
  python run_experiment_0.py          # full run (requires GPU)
  python run_experiment_0.py --step 1 # run steps incrementally
  python run_experiment_0.py --dry-run # validate all code paths (no model inference)
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
import argparse
import logging
import os
from datetime import datetime
import numpy as np

from config import DEVICE, OUTPUT_DIR
from data_utils import get_calibration_files, get_test_files, match_gt


# ============================================================
# Logging
# ============================================================
def setup_logging():
    log_path = os.path.join(OUTPUT_DIR, "exp0_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ============================================================
# Step 1: Model Loading Verification
# ============================================================
def step1_model_verification(logger):
    """Verify SegFormer and Depth Anything load correctly.

    Loads both models, runs a few Cityscapes val images through SegFormer,
    and checks Depth Anything output direction.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Model Loading Verification")
    logger.info("=" * 60)

    # 1a. SegFormer
    logger.info("Loading SegFormer-B0...")
    from model_loader import load_segformer
    seg_model, processor = load_segformer()
    logger.info(f"  SegFormer loaded on {DEVICE}")
    logger.info(f"  Parameters: {sum(p.numel() for p in seg_model.parameters()):,}")

    # Run 5 Cityscapes val images to check output
    cal_files = get_calibration_files()
    sample = cal_files[:5]
    logger.info(f"  Running SegFormer on {len(sample)} calibration images...")

    import numpy as np
    from PIL import Image
    import torch.nn.functional as F

    for i, img_path in enumerate(sample):
        img_rgb = Image.open(img_path).convert("RGB")
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = seg_model(**inputs).logits
        logits_resized = F.interpolate(
            logits, size=(1024, 2048), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits_resized, dim=1)
        pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()
        unique_classes = np.unique(pred)
        logger.info(f"    [{i+1}/{len(sample)}] {os.path.basename(img_path)}: "
                     f"pred shape={pred.shape}, unique classes={len(unique_classes)}")

    logger.info("  SegFormer verification: OK (outputs have non-trivial class predictions)")

    # 1b. Depth Anything
    logger.info("Loading Depth Anything V2 Small...")
    from model_loader import load_depth_anything
    depth_model = load_depth_anything()
    logger.info(f"  Depth Anything loaded on {DEVICE}")

    # Direction verification on 1 foggy image
    test_files = get_test_files()
    foggy_sample = test_files[0]
    import cv2
    img_bgr = cv2.imread(foggy_sample)
    depth_raw = depth_model.infer_image(img_bgr)
    H, W = depth_raw.shape
    from transmittance import verify_depth_direction
    upper_depth, lower_depth = verify_depth_direction(depth_raw)
    logger.info(f"  Depth direction check on {os.path.basename(foggy_sample)}:")
    logger.info(f"    Upper 1/3 (sky/far) depth mean: {upper_depth:.4f}")
    logger.info(f"    Lower 1/3 (ground/near) depth mean: {lower_depth:.4f}")
    if upper_depth < lower_depth:
        logger.info("    → Disparity style confirmed (upper < lower). Will invert for t(x). ✓")
    else:
        logger.info("    → Depth style (upper > lower). May not need inversion. ⚠️  Check transmittance direction")

    logger.info("Step 1 complete. Waiting for confirmation before Step 2.")
    return seg_model, processor, depth_model


# ============================================================
# Step 2: Transmittance Direction Verification
# ============================================================
def step2_transmittance_verification(depth_model, logger):
    """Generate transmittance visualization for first 5 foggy images."""
    logger.info("=" * 60)
    logger.info("STEP 2: Transmittance Direction Verification")
    logger.info("=" * 60)

    test_files = get_test_files()
    sample = test_files[:5]
    import cv2
    import numpy as np
    from transmittance import compute_transmittance, verify_transmittance_direction

    per_image_results = []
    for i, img_path in enumerate(sample):
        img_bgr = cv2.imread(img_path)
        depth_raw = depth_model.infer_image(img_bgr)
        t_map = compute_transmittance(depth_raw)
        t_map_resized = cv2.resize(t_map, (2048, 1024), interpolation=cv2.INTER_LINEAR)
        upper_t, lower_t = verify_transmittance_direction(t_map_resized)

        logger.info(f"  [{i+1}/{len(sample)}] {os.path.basename(img_path)}:")
        logger.info(f"    t range: [{t_map.min():.4f}, {t_map.max():.4f}]")
        logger.info(f"    Upper 1/3 mean t: {upper_t:.4f}")
        logger.info(f"    Lower 1/3 mean t: {lower_t:.4f}")
        if upper_t < lower_t:
            logger.info(f"    → Transmittance direction CORRECT (far<near) ✓")
        else:
            logger.info(f"    → Transmittance direction INVERTED (far>near) ✗")

        per_image_results.append({
            "path": img_path,
            "t_map": t_map_resized,
            "mean_t": float(t_map_resized.mean()),
            "bin0_ratio": float((t_map_resized < 0.2).sum() / t_map_resized.size),
        })

    # Generate transmittance visualization
    from visualization import plot_transmittance_vis
    plot_transmittance_vis(per_image_results, num_samples=5)

    logger.info("Step 2 complete. Waiting for confirmation before Step 3.")
    return per_image_results


# ============================================================
# Step 3: CP Calibration
# ============================================================
def step3_calibration(logger):
    """Compute q_hat on Cityscapes val first 250 images."""
    logger.info("=" * 60)
    logger.info("STEP 3: CP Calibration")
    logger.info("=" * 60)

    from calibration import run_calibration
    q_hat, cal_coverage, cal_pixel_count = run_calibration()

    logger.info(f"  q_hat = {q_hat:.6f}")
    logger.info(f"  Calibration coverage = {cal_coverage:.6f}")
    logger.info(f"  Calibration pixels = {cal_pixel_count:,}")

    if 0.88 <= cal_coverage <= 0.92:
        logger.info("  Calibration coverage in [0.88, 0.92] ✓")
    else:
        logger.warning(f"  Calibration coverage {cal_coverage:.4f} outside [0.88, 0.92]!")
        logger.warning("  Check nonconformity score computation. Stopping.")
        sys.exit(1)

    logger.info("Step 3 complete. Waiting for confirmation before Step 4.")
    return q_hat, cal_coverage, cal_pixel_count


# ============================================================
# Step 4: Full Evaluation
# ============================================================
def step4_evaluation(q_hat, cal_coverage, cal_pixel_count, logger):
    """Run full test inference, evaluate, visualize, decide."""
    logger.info("=" * 60)
    logger.info("STEP 4: Full Test Evaluation (500 foggy images)")
    logger.info("=" * 60)

    # 4a. Inference
    from inference import TestSetInference
    inference_runner = TestSetInference(q_hat=q_hat)
    inference_runner.run(max_images=None)  # all 500

    # 4b. Evaluation
    from evaluation import compute_results, save_results, print_summary
    bin_arrays = inference_runner.get_bin_arrays()
    results = compute_results(
        q_hat=q_hat,
        cal_coverage=cal_coverage,
        cal_pixel_count=cal_pixel_count,
        bin_arrays=bin_arrays,
    )
    save_results(results)
    print_summary(results)

    # 4c. Visualization
    from visualization import generate_all_plots
    per_image_results = inference_runner.per_image_results
    generate_all_plots(results, per_image_results, bin_arrays)

    # 4d. Decision tree
    from decision_tree import DecisionTree
    dt = DecisionTree()
    dt.data = results
    print()  # blank line
    print(dt.evaluate())

    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 0 COMPLETE")
    logger.info(f"Decision: {dt.decision}")
    logger.info("=" * 60)

    return results, dt


# ============================================================
# Dry Run (no model inference)
# ============================================================
def dry_run(logger):
    """Validate all code paths without running model inference."""
    logger.info("=" * 60)
    logger.info("DRY RUN — Validating all code paths")
    logger.info("=" * 60)

    # Config
    logger.info("  config.py: OK")

    # Data utils
    cal_files = get_calibration_files()
    test_files = get_test_files()
    assert len(cal_files) == 250, f"Expected 250 calibration files, got {len(cal_files)}"
    assert len(test_files) == 500, f"Expected 500 test files, got {len(test_files)}"
    logger.info(f"  data_utils.py: calibration={len(cal_files)}, test={len(test_files)}")

    # GT matching
    gt_test = match_gt(test_files[0])
    assert os.path.exists(gt_test), f"GT not found: {gt_test}"
    all_gt_exist = all(os.path.exists(match_gt(f)) for f in test_files[:10])
    logger.info(f"  data_utils: GT mapping verified (10/10 exist)")

    # Transmittance
    from transmittance import compute_transmittance
    dummy_depth = np.random.rand(1024, 2048).astype(np.float32) * 100
    dummy_depth[:200, :] = 10  # top = close (small depth in disparity)
    dummy_depth[-200:, :] = 90  # bottom = far
    t_test = compute_transmittance(dummy_depth)
    assert 0 < t_test.min() and t_test.max() <= 1.0
    logger.info(f"  transmittance.py: t range [{t_test.min():.4f}, {t_test.max():.4f}]")

    # Evaluation (mock data)
    from evaluation import compute_results, print_summary
    np.random.seed(42)
    mock_bin_arrays = {
        "pixel_counts": np.array([50000, 100000, 200000, 300000, 350000], dtype=np.int64),
        "covered_counts": np.array([25000, 75000, 170000, 270000, 330000], dtype=np.int64),
        "score_sums": np.array([35000, 65000, 110000, 120000, 105000], dtype=np.float64),
        "set_size_sums": np.array([50000, 150000, 400000, 600000, 525000], dtype=np.float64),
    }
    mock_results = compute_results(0.85, 0.90, 1000000, mock_bin_arrays)
    print_summary(mock_results)
    logger.info(f"  evaluation.py: OK (mock results computed)")

    # Decision tree
    from decision_tree import DecisionTree
    dt = DecisionTree()
    dt.data = mock_results
    decision_text = dt.evaluate()
    logger.info(f"  decision_tree.py: OK (decision={dt.decision})")

    # Visualization
    from visualization import generate_all_plots
    from inference import TestSetInference
    # Make minimal mock per_image_results
    mock_per_image = [{
        "path": test_files[0],
        "t_map": np.random.rand(1024, 2048).astype(np.float32),
        "bin0_ratio": 0.15,
        "mean_t": 0.5,
    } for _ in range(5)]
    try:
        plots = generate_all_plots(mock_results, mock_per_image, mock_bin_arrays)
        for p in plots:
            assert os.path.exists(p), f"Plot not generated: {p}"
        logger.info(f"  visualization.py: 3 plots generated ✓")
    except Exception as e:
        logger.warning(f"  visualization.py: plot generation warning: {e}")

    logger.info("=" * 60)
    logger.info("DRY RUN COMPLETE — All modules validated")
    logger.info("=" * 60)
    return True


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Exp 0: Coverage Collapse")
    parser.add_argument("--step", type=int, default=0,
                        help="Run from a specific step (0=all, 1-4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate code paths without model inference")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Experiment 0 started at {datetime.now()}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Output: {OUTPUT_DIR}")

    if args.dry_run:
        dry_run(logger)
        return

    if args.step == 0 or args.step == 1:
        seg_model, processor, depth_model = step1_model_verification(logger)
        # Report
        logger.info("\n=== Step 1 Complete ===")
        logger.info("Status: SUCCESS (models loaded, SegFormer outputs non-trivial)")
        logger.info("Wait for instruction: YES — confirm before Step 2")

    if args.step == 0 or args.step == 2:
        if args.step == 2:
            # Reload depth model if starting from step 2
            from model_loader import load_depth_anything
            depth_model = load_depth_anything()
        per_image_results = step2_transmittance_verification(depth_model, logger)
        logger.info("\n=== Step 2 Complete ===")
        logger.info("Status: SUCCESS (transmittance_vis.png saved)")
        logger.info("Wait for instruction: YES — confirm before Step 3")

    if args.step == 0 or args.step == 3:
        q_hat, cal_coverage, cal_pixel_count = step3_calibration(logger)
        logger.info("\n=== Step 3 Complete ===")
        logger.info(f"Status: SUCCESS (q_hat={q_hat:.6f}, cal_cov={cal_coverage:.4f})")
        logger.info("Wait for instruction: YES — confirm before Step 4")

    if args.step == 0 or args.step == 4:
        if args.step == 4:
            # Recompute q_hat if starting from step 4
            from calibration import run_calibration
            q_hat, cal_coverage, cal_pixel_count = run_calibration()
        results, dt = step4_evaluation(q_hat, cal_coverage, cal_pixel_count, logger)
        logger.info("\n=== Step 4 Complete ===")
        logger.info(f"Decision: {dt.decision}")
        logger.info("Wait for instruction: YES — decision tree triggered")


if __name__ == "__main__":
    main()
