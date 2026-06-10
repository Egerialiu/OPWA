import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
import os
import numpy as np
import cv2

# ============================================================
# Import shared exp0 config (device, paths, etc.)
# ============================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp0"))
from config import DEVICE, BETA, LABEL2TRAIN

# ============================================================
# Fixed CP parameters (NEVER modify q_hat)
# ============================================================
Q_HAT = 0.513809
TARGET_COVERAGE = 0.90
ALPHA = 0.1

# ============================================================
# Output directory
# ============================================================
OUTPUT_DIR = "/root/opwa_v3/exp1_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ACDC data paths
# ============================================================
ACDC_ROOT = "/gz-data/ACDC"
ACDC_IMG_DIR = os.path.join(ACDC_ROOT, "rgb_anon_trainvaltest", "rgb_anon", "fog", "val")
ACDC_GT_DIR = os.path.join(ACDC_ROOT, "gt", "fog", "val")

ACDC_TARGET_H, ACDC_TARGET_W = 1080, 1920

# ============================================================
# DCP stratified bin definitions
# Note: DCP value direction is opposite to transmittance t(x).
#   DCP ↑ = more fog (dark channel is bright in foggy regions)
#   t(x) ↑ = less fog (high transmittance = clear)
# ============================================================

# Fixed bin edges based on DCP values
DCP_BIN_EDGES = [0.0, 0.10, 0.20, 0.30, 0.50, 1.0]
DCP_BIN_NAMES = [
    "bin0_dcp0.00_0.10_clear",
    "bin1_dcp0.10_0.20",
    "bin2_dcp0.20_0.30",
    "bin3_dcp0.30_0.50",
    "bin4_dcp0.50_1.00_fog",
]
NUM_DCP_BINS = len(DCP_BIN_NAMES)

# ============================================================
# DCP computation
# ============================================================
def compute_dcp(image_rgb, window_size=15):
    """Compute Dark Channel Prior for a given RGB image.

    Args:
        image_rgb: np.ndarray (H, W, 3), dtype uint8 or float.
                   If uint8, values in [0, 255]; if float, values in [0, 1].
        window_size: int, size of the erosion kernel.

    Returns:
        dcp: np.ndarray (H, W), float32 in [0, 1].
             DCP value = min(RGB) eroded. Higher = more fog.
    """
    if image_rgb.dtype == np.uint8:
        image_norm = image_rgb.astype(np.float32) / 255.0
    else:
        image_norm = image_rgb.astype(np.float32)

    min_channel = np.min(image_norm, axis=2)  # (H, W), min over RGB
    kernel = np.ones((window_size, window_size), dtype=np.uint8)
    dcp = cv2.erode(min_channel, kernel)  # morphological erosion
    return dcp  # values in [0, 1]

# ============================================================
# Utility: get ACDC image/GT file pairs
# ============================================================
def get_acdc_fog_files():
    """Return sorted lists of (image_path, gt_path) for ACDC fog val."""
    import glob
    img_pattern = os.path.join(ACDC_IMG_DIR, "*/*_rgb_anon.png")
    img_files = sorted(glob.glob(img_pattern))

    gt_files = []
    for img_path in img_files:
        basename = os.path.basename(img_path)
        gt_basename = basename.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
        subdir = os.path.basename(os.path.dirname(img_path))
        gt_path = os.path.join(ACDC_GT_DIR, subdir, gt_basename)
        gt_files.append(gt_path)

    return img_files, gt_files


def get_acdc_image_paths():
    """Return just image paths for fog scoring (without running model)."""
    import glob
    img_pattern = os.path.join(ACDC_IMG_DIR, "*/*_rgb_anon.png")
    return sorted(glob.glob(img_pattern))
