#!/usr/bin/env python3
"""
Domain shift analysis: MMD between clear, Foggy CS, and ACDC using SegFormer features.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import os
import sys
import glob

sys.path.insert(0, "/root/opwa_v3/OPWA_v3/exp0")
from config import DEVICE, OUTPUT_DIR

print("Loading models...")

from model_loader import load_segformer
seg_model, processor = load_segformer()
seg_model.eval()

# === Data ===
cal_files = sorted(glob.glob("/gz-data/cityscapes/leftImg8bit/val/*/*.png"))[:50]
foggy_files = sorted(glob.glob("/gz-data/foggy_cityscapes/leftImg8bit_foggy/val/*/*.png"))[:50]
acdc_files = sorted(glob.glob("/gz-data/ACDC/rgb_anon_trainvaltest/rgb_anon/fog/val/*/*_rgb_anon.png"))[:50]

print(f"Calibration (clear): {len(cal_files)}")
print(f"Foggy CS:           {len(foggy_files)}")
print(f"ACDC Fog:           {len(acdc_files)}")


def extract_features(paths, label):
    feats = []
    for p in tqdm(paths, desc=f"  {label}"):
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = seg_model(**inputs).logits  # (1, 19, H/4, W/4) = (1, 19, 256, 512)
        # Global average pool over spatial dims -> (19,) class logit means
        feat = logits.mean(dim=[2, 3]).squeeze(0).cpu().numpy()  # (19,)
        feats.append(feat)
    return np.array(feats)


def compute_mmd(x, y, sigma=1.0):
    n, m = x.shape[0], y.shape[0]
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)
    xn = np.sum(x**2, 1).reshape(-1, 1)
    yn = np.sum(y**2, 1).reshape(-1, 1)
    Kxx = np.exp(-(xn + xn.T - 2*xx) / (2*sigma**2))
    Kyy = np.exp(-(yn + yn.T - 2*yy) / (2*sigma**2))
    Kxy = np.exp(-(xn + yn.T - 2*xy) / (2*sigma**2))
    mmd = (Kxx.sum()/n**2 + Kyy.sum()/m**2 - 2*Kxy.sum()/(n*m))
    return float(np.sqrt(max(0, mmd)))


f_cal = extract_features(cal_files, "Clear")
f_fog = extract_features(foggy_files, "FoggyCS")
f_acdc = extract_features(acdc_files, "ACDC")

mmd_cf = compute_mmd(f_cal, f_fog)
mmd_ca = compute_mmd(f_cal, f_acdc)
mmd_fa = compute_mmd(f_fog, f_acdc)

results = {
    "calibration_set": "cityscapes_clear_val_50",
    "num_images_per_set": 50,
    "MMD_clear_vs_FoggyCS": round(mmd_cf, 6),
    "MMD_clear_vs_ACDC": round(mmd_ca, 6),
    "MMD_FoggyCS_vs_ACDC": round(mmd_fa, 6),
    "MMD_ratio_FoggyCS_over_ACDC": round(mmd_cf / max(mmd_ca, 1e-10), 4),
    "FoggyCS_qtile0_gap": 0.149135,
    "ACDC_qtile0_gap": 0.0,
}

print(f"\nMMD(clear, FoggyCS): {mmd_cf:.6f}")
print(f"MMD(clear, ACDC):    {mmd_ca:.6f}")
print(f"MMD(FoggyCS, ACDC):  {mmd_fa:.6f}")
print(f"Ratio (Foggy/ACDC):  {results['MMD_ratio_FoggyCS_over_ACDC']:.2f}x")

path = os.path.join(OUTPUT_DIR, "diag_domain_shift.json")
with open(path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {path}")
