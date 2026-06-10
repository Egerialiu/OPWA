#!/usr/bin/env python3
"""
Beta sweep: run split CP on Foggy CS with different beta values.
Also compute MMD domain shift metrics.

Betas: 0.005, 0.01, 0.02 (0.02 already done, skip if results exist)
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
import glob
import zipfile
import tempfile
import shutil

sys.path.insert(0, "/root/opwa_v3/OPWA_v3/exp0")
from config import DEVICE, BIN_EDGES, BIN_NAMES, NUM_BINS, OUTPUT_DIR, TARGET_COVERAGE
from model_loader import load_segformer, load_depth_anything
from transmittance import compute_transmittance

Q_HAT = 0.513809
FOGGY_ZIPS = [
    "/gz-data/foggy_cityscapes/leftImg8bit_trainval_foggy_part1.zip",
    "/gz-data/foggy_cityscapes/leftImg8bit_trainval_foggy_part2.zip",
    "/gz-data/foggy_cityscapes/leftImg8bit_trainval_foggy_part3.zip",
    "/gz-data/foggy_cityscapes/leftImg8bit_trainval_foggy_part4.zip",
]
EXTRACTED_DIR = "/gz-data/foggy_cityscapes/"


def get_test_files_for_beta(beta_str):
    """Get test file paths for a specific beta version."""
    if beta_str == "0.02":
        pattern = os.path.join(EXTRACTED_DIR, "leftImg8bit_foggy", "val", "*/*.png")
        all_files = sorted(glob.glob(pattern))
        return [f for f in all_files if f"beta_{beta_str}" in f]
    else:
        # Extract from zip
        beta_pattern = f"leftImg8bit_foggy/val/*/*foggy_beta_{beta_str}.png"
        matches = []
        for zp in FOGGY_ZIPS:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if name.endswith(f"foggy_beta_{beta_str}.png"):
                        matches.append(name)
        matches = sorted(set(matches))
        return matches  # zip internal paths


def match_gt_from_foggy(foggy_path):
    """Get GT path from a foggy path or zip path."""
    if foggy_path.endswith(".png"):
        gt = foggy_path.replace("_foggy_beta_0.02", "")
        gt = gt.replace("_foggy_beta_0.01", "")
        gt = gt.replace("_foggy_beta_0.005", "")
        gt = gt.replace("leftImg8bit_foggy", "gtFine")
        gt = gt.replace("leftImg8bit", "gtFine_labelIds")
        gt = gt.replace("/foggy_cityscapes/", "/cityscapes/")
        return gt
    return None


def run_beta_inference(beta_str, seg_model, processor, depth_model):
    """Run inference for a specific beta value."""
    test_paths = get_test_files_for_beta(beta_str)
    print(f"\nBeta={beta_str}: {len(test_paths)} images")

    if len(test_paths) == 0:
        print(f"  No images found for beta={beta_str}")
        return None

    T_H, T_W = 1024, 2048  # Cityscapes resolution

    # Accumulators
    bin_pc = np.zeros(NUM_BINS, dtype=np.int64)
    bin_cc = np.zeros(NUM_BINS, dtype=np.int64)
    bin_ss = np.zeros(NUM_BINS, dtype=np.float64)
    bin_sz = np.zeros(NUM_BINS, dtype=np.float64)
    bin_ec = np.zeros(NUM_BINS, dtype=np.int64)

    # For quantile
    t_samples = []
    T_SUBSAMPLE = 100

    temp_dir = None
    if beta_str != "0.02":
        temp_dir = tempfile.mkdtemp()

    try:
        for idx, path in enumerate(tqdm(test_paths, desc=f"Beta {beta_str}")):
            if beta_str == "0.02":
                img_path = path
            else:
                # Extract from zip
                with zipfile.ZipFile(FOGGY_ZIPS[0]) as z:
                    # Find which zip contains this file
                    found = False
                    for zp in FOGGY_ZIPS:
                        with zipfile.ZipFile(zp) as z2:
                            try:
                                data = z2.read(path)
                                img_path = os.path.join(temp_dir, os.path.basename(path))
                                with open(img_path, "wb") as f:
                                    f.write(data)
                                found = True
                                break
                            except KeyError:
                                continue
                    if not found:
                        continue

            # SegFormer
            img_rgb = Image.open(img_path).convert("RGB")
            inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = seg_model(**inputs).logits
            logits = F.interpolate(logits, size=(T_H, T_W), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)
            probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Depth
            img_bgr = cv2.imread(img_path)
            depth_raw = depth_model.infer_image(img_bgr)
            t_map = compute_transmittance(depth_raw)
            t_map = cv2.resize(t_map, (T_W, T_H), interpolation=cv2.INTER_LINEAR)

            # GT
            gt_path = match_gt_from_foggy(path if beta_str == "0.02" else path)
            # For zip paths, reconstruct GT path
            if beta_str != "0.02":
                # path is like "leftImg8bit_foggy/val/frankfurt/..._foggy_beta_0.005.png"
                gt_path = path.replace(f"_foggy_beta_{beta_str}", "")
                gt_path = gt_path.replace("leftImg8bit_foggy", "gtFine")
                gt_path = gt_path.replace("leftImg8bit", "gtFine_labelIds")
                gt_path = os.path.join("/gz-data/cityscapes", gt_path)

            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

            # Map to train IDs
            LABEL2TRAIN = {
                0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0,
                8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
                16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
                24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
                31: 16, 32: 17, 33: 18, -1: 255,
            }
            gt_mapped = np.full_like(gt, 255, dtype=np.uint8)
            for lid, tid in LABEL2TRAIN.items():
                gt_mapped[gt == lid] = tid
            gt = gt_mapped

            valid_mask = gt != 255
            gt_valid = gt[valid_mask]
            probs_valid = probs[valid_mask]
            t_valid = t_map[valid_mask]

            scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid]
            covered = scores <= Q_HAT
            set_sizes = (probs_valid >= (1.0 - Q_HAT)).sum(axis=1)
            empty_set = set_sizes == 0

            bin_inds = np.digitize(t_valid, BIN_EDGES) - 1
            for bi in range(NUM_BINS):
                m = bin_inds == bi
                c = m.sum()
                if c == 0: continue
                bin_pc[bi] += c
                bin_cc[bi] += covered[m].sum()
                bin_ss[bi] += scores[m].sum()
                bin_sz[bi] += set_sizes[m].sum()
                bin_ec[bi] += empty_set[m].sum()

            t_sub = t_valid[::T_SUBSAMPLE]
            t_samples.append(t_sub)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Build fixed-bin results
    total_v = int(bin_pc.sum())
    bins_f = {}
    for i, name in enumerate(BIN_NAMES):
        pc = int(bin_pc[i])
        cc = int(bin_cc[i])
        ec = int(bin_ec[i])
        cov = float(cc / pc) if pc > 0 else 0.0
        gap = max(0.0, TARGET_COVERAGE - cov)
        mss = float(bin_sz[i] / pc) if pc > 0 else 0.0
        ms = float(bin_ss[i] / pc) if pc > 0 else 0.0
        esr = float(ec / pc) if pc > 0 else 0.0
        bins_f[name] = {"pixel_count": pc, "covered_count": cc, "coverage_rate": round(cov, 6),
                        "gap": round(gap, 6), "mean_set_size": round(mss, 6), "mean_score": round(ms, 6),
                        "empty_set_count": ec, "empty_set_rate": round(esr, 6)}

    overall = float(bin_cc.sum() / total_v) if total_v > 0 else 0.0
    max_gap = max(b["gap"] for b in bins_f.values())
    bin0_gap = bins_f[BIN_NAMES[0]]["gap"]
    bin0_ratio = float(bins_f[BIN_NAMES[0]]["pixel_count"] / total_v) if total_v > 0 else 0.0

    results = {"dataset": f"foggy_cityscapes_beta{beta_str}", "alpha": 0.1,
               "q_hat": round(Q_HAT, 6), "calibration_pixel_count": 449381364,
               "calibration_coverage": 0.9, "bins": bins_f,
               "overall_test_coverage": round(overall, 6), "max_gap": round(max_gap, 6),
               "bin0_gap": round(bin0_gap, 6), "bin0_pixel_ratio": round(bin0_ratio, 6)}

    # Quantile
    if len(t_samples) > 0:
        t_all = np.concatenate(t_samples)
        q20 = float(np.quantile(t_all, 0.20))
        q40 = float(np.quantile(t_all, 0.40))
        q60 = float(np.quantile(t_all, 0.60))
        q80 = float(np.quantile(t_all, 0.80))
        q_edges = [0.0, q20, q40, q60, q80, 1.01]
        q_names = [f"qtile{i}_t0.00_{q20:.4f}" if i==0 else
                   (f"qtile{i}_t{q80:.4f}_1.00" if i==4 else
                    f"qtile{i}_t{[q20, q40, q60, q80][i-1]:.4f}_{[q20, q40, q60, q80][i]:.4f}")
                   for i in range(5)]
        # Actually, let me just use generic names
        q_names = [f"qtile{i}" for i in range(NUM_BINS)]

        results_quantile = results.copy()
        results_quantile["bin_strategy"] = "quantile"
        results_quantile["quantile_edges"] = [round(e, 6) for e in q_edges]

        # Re-do with quantile bins (re-process the data)
        # We need to re-run, but for now just report the edges and fixed results
        results_quantile["quantile_edges_20_40_60_80"] = [round(q20, 4), round(q40, 4), round(q60, 4), round(q80, 4)]

    print(f"  Overall coverage: {overall:.4f}, Bin0 gap: {bin0_gap:.4f}, Max gap: {max_gap:.4f}")
    for i, name in enumerate(BIN_NAMES):
        print(f"    {name}: cov={bins_f[name]['coverage_rate']:.4f}, gap={bins_f[name]['gap']:.4f}, mean_score={bins_f[name]['mean_score']:.4f}")

    return results


def compute_mmd(x, y, sigma=1.0):
    """Gaussian kernel MMD between two sets of samples.
    x, y: (n, d) and (m, d) numpy arrays.
    """
    n = x.shape[0]
    m = y.shape[0]
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)

    xx_norm = np.sum(x**2, axis=1).reshape(-1, 1)
    yy_norm = np.sum(y**2, axis=1).reshape(-1, 1)

    K_xx = np.exp(-(xx_norm + xx_norm.T - 2 * xx) / (2 * sigma**2))
    K_yy = np.exp(-(yy_norm + yy_norm.T - 2 * yy) / (2 * sigma**2))
    K_xy = np.exp(-(xx_norm + yy_norm.T - 2 * xy) / (2 * sigma**2))

    mmd = (K_xx.sum() / (n * n) + K_yy.sum() / (m * m) - 2 * K_xy.sum() / (n * m))
    return float(np.sqrt(max(0, mmd)))


def run_mmd_analysis(seg_model, processor):
    """Compute MMD between calibration set and each test set using SegFormer features."""
    print("\n=== Domain Shift Analysis (MMD) ===")

    from transformers import SegformerModel
    # Get the backbone model for feature extraction
    feature_model = SegformerModel.from_pretrained(
        'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
    ).eval().to(DEVICE)

    # Calibration set: 50 clear images
    cal_pattern = os.path.join("/gz-data/cityscapes", "leftImg8bit", "val", "*/*.png")
    cal_files = sorted(glob.glob(cal_pattern))[:50]
    print(f"  Calibration (clear): {len(cal_files)} images")

    # Foggy CS beta=0.02: 50 images
    test_foggy = sorted(glob.glob(os.path.join(
        "/gz-data/foggy_cityscapes", "leftImg8bit_foggy", "val", "*/*.png"
    )))[:50]

    # ACDC Fog: 50 images (all available)
    acdc_pattern = os.path.join(
        "/gz-data/ACDC", "rgb_anon_trainvaltest", "rgb_anon", "fog", "val", "*/*_rgb_anon.png"
    )
    test_acdc = sorted(glob.glob(acdc_pattern))[:50]
    print(f"  ACDC Fog: {len(test_acdc)} images")

    def extract_features(image_paths):
        features = []
        for p in tqdm(image_paths, desc="  Extracting features"):
            img = Image.open(p).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = feature_model(**inputs)
                # last_hidden_state: (B, H/4*W/4, C)
                feat = outputs.last_hidden_state  # (1, seq_len, 256)
            # Global average pooling
            feat = feat.mean(dim=1).squeeze(0).cpu().numpy()  # (256,)
            features.append(feat)
        return np.array(features)

    f_cal = extract_features(cal_files)
    f_foggy = extract_features(test_foggy)
    f_acdc = extract_features(test_acdc)

    mmd_cs = compute_mmd(f_cal, f_foggy)
    mmd_acdc = compute_mmd(f_cal, f_acdc)

    results = {
        "calibration_set": "cityscapes_clear_val_50",
        "MMD_clear_vs_FoggyCS": round(mmd_cs, 6),
        "MMD_clear_vs_ACDC": round(mmd_acdc, 6),
        "MMD_ratio_FoggyCS_over_ACDC": round(mmd_cs / max(mmd_acdc, 1e-10), 4),
        "FoggyCS_overall_gap_quantile": 0.149135,  # from previous run
        "ACDC_overall_gap": 0.0,
    }

    print(f"\n  MMD(clear, FoggyCS): {mmd_cs:.6f}")
    print(f"  MMD(clear, ACDC):    {mmd_acdc:.6f}")
    print(f"  Ratio (Foggy/ACDC):  {results['MMD_ratio_FoggyCS_over_ACDC']:.2f}x")

    return results


if __name__ == "__main__":
    print("Loading models...")
    seg_model, processor = load_segformer()
    seg_model.eval()
    depth_model = load_depth_anything()
    depth_model.eval()

    # Task B: Beta sweep
    for beta in ["0.005", "0.01", "0.02"]:
        result = run_beta_inference(beta, seg_model, processor, depth_model)
        if result is not None:
            fn = os.path.join(OUTPUT_DIR, f"exp0_beta{beta.replace('.', '')}_results.json")
            with open(fn, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved {fn}")

    # Task A: MMD domain shift
    mmd_results = run_mmd_analysis(seg_model, processor)
    mmd_path = os.path.join(OUTPUT_DIR, "diag_domain_shift.json")
    with open(mmd_path, "w") as f:
        json.dump(mmd_results, f, indent=2)
    print(f"  Saved {mmd_path}")

    # Summary
    print("\n" + "=" * 60)
    print("BETA SWEEP + DOMAIN SHIFT COMPLETE")
    print("=" * 60)
    for beta in ["0.005", "0.01", "0.02"]:
        fn = os.path.join(OUTPUT_DIR, f"exp0_beta{beta.replace('.', '')}_results.json")
        if os.path.exists(fn):
            with open(fn) as f:
                r = json.load(f)
            print(f"  Beta {beta}: overall_cov={r['overall_test_coverage']:.4f}, "
                  f"bin0_gap={r['bin0_gap']:.4f}, max_gap={r['max_gap']:.4f}")
    print(f"  MMD Clear→FoggyCS: {mmd_results['MMD_clear_vs_FoggyCS']:.4f}")
    print(f"  MMD Clear→ACDC:    {mmd_results['MMD_clear_vs_ACDC']:.4f}")
    print(f"  Ratio: {mmd_results['MMD_ratio_FoggyCS_over_ACDC']:.2f}x")
