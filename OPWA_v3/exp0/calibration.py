import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config import DEVICE, Q_HAT_QUANTILE, TARGET_COVERAGE
from model_loader import load_segformer
from data_utils import get_calibration_files


from gt_utils import load_gt_label_ids


def run_calibration():
    """Run Split CP calibration on Cityscapes val first 250 images.

    Returns:
        q_hat: float — the 90th percentile of pixel-level nonconformity scores.
        cal_coverage: float — the actual coverage on the calibration set.
        cal_pixel_count: int — total valid pixels processed.
    """
    model, processor = load_segformer()
    model.eval()

    cal_files = get_calibration_files()
    print(f"Calibration: {len(cal_files)} images")

    all_scores = []  # CPU list of valid pixel scores

    for img_path in tqdm(cal_files, desc="Calibration"):
        # --- SegFormer inference ---
        img_rgb = Image.open(img_path).convert("RGB")
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits  # (1, 19, H/4, W/4)

        # Resize to original (1024, 2048)
        logits = F.interpolate(
            logits, size=(1024, 2048), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)  # (1, 19, 1024, 2048)
        probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (1024, 2048, 19)

        # --- Load GT ---
        gt_path = img_path.replace("/leftImg8bit/", "/gtFine/")
        gt_path = gt_path.replace("leftImg8bit.png", "gtFine_labelIds.png")
        gt = load_gt_label_ids(gt_path)  # (1024, 2048), uint8, train IDs

        # --- Pixel-level score ---
        valid_mask = gt != 255
        gt_valid = gt[valid_mask]
        probs_valid = probs[valid_mask]  # (N, 19)

        scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid]
        all_scores.append(scores)

    # --- Compute q_hat ---
    all_scores = np.concatenate(all_scores)
    q_hat = float(np.quantile(all_scores, Q_HAT_QUANTILE))
    cal_coverage = float((all_scores <= q_hat).mean())
    cal_pixel_count = int(len(all_scores))

    print(f"  Total valid pixels: {cal_pixel_count}")
    print(f"  q_hat ({Q_HAT_QUANTILE*100:.0f}th percentile): {q_hat:.6f}")
    print(f"  Calibration set coverage: {cal_coverage:.6f} (target {TARGET_COVERAGE})")

    # Sanity check
    if not (0.88 <= cal_coverage <= 0.92):
        print(f"  ⚠️  Calibration coverage {cal_coverage:.4f} outside [0.88, 0.92]!")
        print(f"  Check nonconformity score computation.")

    return q_hat, cal_coverage, cal_pixel_count
