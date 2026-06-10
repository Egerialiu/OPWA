import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os

# ============================================================
# Paths
# ============================================================
CITYSCAPES_ROOT = "/gz-data/cityscapes/"
FOGGY_ROOT = "/gz-data/foggy_cityscapes/"

# Depth-Anything-V2 repo
DAV2_REPO = "/root/opwa_v3/Depth-Anything-V2/"
DAV2_CKPT = "/root/opwa_v3/checkpoints/depth_anything_v2_vits.pth"

# Output
OUTPUT_DIR = "/root/opwa_v3/exp0_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Device
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CP parameters
# ============================================================
ALPHA = 0.1
TARGET_COVERAGE = 1.0 - ALPHA  # 0.90
Q_HAT_QUANTILE = 0.90

# ============================================================
# Transmittance
# ============================================================
BETA = 3.0

# ============================================================
# Bin definitions (fixed, do not change)
# ============================================================
BIN_EDGES = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
BIN_NAMES = [
    "bin0_t0.00_0.20",  # dense fog
    "bin1_t0.20_0.40",
    "bin2_t0.40_0.60",
    "bin3_t0.60_0.80",
    "bin4_t0.80_1.00",  # clear
]
NUM_BINS = len(BIN_NAMES)

# ============================================================
# Calibration split
# ============================================================
CALIBRATION_COUNT = 250  # first N files alphabetically

# ============================================================
# Cityscapes label mapping (original label ID → train ID, 19 classes)
# ============================================================
LABEL2TRAIN = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0,
    8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
    16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18, -1: 255,
}
