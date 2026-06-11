"""
Configuration — Conformal Prediction parameters only.

Ported from OPWA_v3/exp0/config.py.
Stripped of model paths, data roots, and Depth-Anything references.
Add nuScenes-specific paths / model paths in a downstream config, not here.
"""

# ============================================================
# Conformal Prediction parameters
# ============================================================
ALPHA = 0.10
TARGET_COVERAGE = 1.0 - ALPHA  # 0.90
Q_HAT_QUANTILE = 0.90

# ============================================================
# Bin definitions (for transmittance / difficulty binning)
# ============================================================
# 2D default: transmittance bins [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
# 3D: replace these with distance bins or difficulty-score bins.
BIN_EDGES = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
BIN_NAMES = [
    "bin0_0.00_0.20",   # hardest
    "bin1_0.20_0.40",
    "bin2_0.40_0.60",
    "bin3_0.60_0.80",
    "bin4_0.80_1.00",   # easiest
]
NUM_BINS = len(BIN_NAMES)
