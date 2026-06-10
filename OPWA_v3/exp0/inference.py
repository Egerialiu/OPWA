import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
from tqdm import tqdm

from config import DEVICE, BIN_EDGES, NUM_BINS, OUTPUT_DIR
from model_loader import load_segformer, load_depth_anything
from data_utils import get_test_files, match_gt
from transmittance import compute_transmittance
from gt_utils import load_gt_label_ids


class TestSetInference:
    """Per-pixel inference on the Foggy Cityscapes test set.

    Accumulates pixel-level statistics across all 500 images,
    storing results in per-bin accumulators for later evaluation.
    """

    def __init__(self, q_hat):
        self.q_hat = q_hat

        # Per-bin accumulators (CPU numpy)
        self.bin_pixel_counts = np.zeros(NUM_BINS, dtype=np.int64)
        self.bin_covered_counts = np.zeros(NUM_BINS, dtype=np.int64)
        self.bin_score_sums = np.zeros(NUM_BINS, dtype=np.float64)
        self.bin_set_size_sums = np.zeros(NUM_BINS, dtype=np.float64)

        # Store per-image results for visualization
        self.per_image_results = []  # list of dicts

    def run(self, max_images=None):
        """Run inference on test set.

        Args:
            max_images: if set, limit the number of images (for debugging).
        """
        print("Loading models...")
        seg_model, processor = load_segformer()
        seg_model.eval()
        depth_model = load_depth_anything()
        depth_model.eval()

        test_files = get_test_files()
        if max_images is not None:
            test_files = test_files[:max_images]
        print(f"Test set: {len(test_files)} images")

        for idx, img_path in enumerate(tqdm(test_files, desc="Test inference")):
            result = self._process_one_image(
                img_path, seg_model, processor, depth_model
            )
            self.per_image_results.append(result)

        print("\nAccumulated pixel counts per bin:")
        for i in range(NUM_BINS):
            print(f"  {BIN_EDGES[i]:.2f}-{BIN_EDGES[i+1]:.2f}: {self.bin_pixel_counts[i]} pixels")

        total_valid = self.bin_pixel_counts.sum()
        print(f"  Total valid pixels: {total_valid}")

    def _process_one_image(self, img_path, seg_model, processor, depth_model):
        """Process a single foggy image and accumulate per-bin stats.

        Returns a dict with image-level results for visualization.
        """
        # --- 1. SegFormer inference ---
        img_rgb = Image.open(img_path).convert("RGB")
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = seg_model(**inputs).logits  # (1, 19, H/4, W/4)

        logits = F.interpolate(
            logits, size=(1024, 2048), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)  # (1, 19, 1024, 2048)
        probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (1024, 2048, 19)

        # --- 2. Depth Anything inference ---
        img_bgr = cv2.imread(img_path)  # BGR, (1024, 2048)
        depth_raw = depth_model.infer_image(img_bgr)  # (H, W) numpy
        t_map = compute_transmittance(depth_raw)  # (H, W)
        t_map = cv2.resize(t_map, (2048, 1024), interpolation=cv2.INTER_LINEAR)

        # --- 3. Load GT ---
        gt_path = match_gt(img_path)
        gt = load_gt_label_ids(gt_path)  # (1024, 2048), uint8, train IDs

        # --- 4. Per-pixel bin assignment & coverage check ---
        valid_mask = gt != 255
        gt_valid = gt[valid_mask]
        probs_valid = probs[valid_mask]  # (N, 19)
        t_valid = t_map[valid_mask]

        # Nonconformity score per pixel
        scores = 1.0 - probs_valid[np.arange(len(gt_valid)), gt_valid]

        # Coverage condition: GT class in C(x) = {c: softmax[c] >= 1 - q_hat}
        covered = scores <= self.q_hat

        # Set size for each pixel
        set_sizes = (probs_valid >= (1.0 - self.q_hat)).sum(axis=1)

        # Bin assignment
        bin_indices = np.digitize(t_valid, BIN_EDGES) - 1  # 0..4

        # Accumulate
        for bin_i in range(NUM_BINS):
            mask = bin_indices == bin_i
            count = mask.sum()
            if count == 0:
                continue
            self.bin_pixel_counts[bin_i] += count
            self.bin_covered_counts[bin_i] += covered[mask].sum()
            self.bin_score_sums[bin_i] += scores[mask].sum()
            self.bin_set_size_sums[bin_i] += set_sizes[mask].sum()

        # --- 5. Build per-image result ---
        img_result = {
            "path": img_path,
            "t_map": t_map,
            "bin0_ratio": float((bin_indices == 0).sum()) / max(len(gt_valid), 1),
            "mean_t": float(t_valid.mean()),
        }
        return img_result

    def get_bin_arrays(self):
        """Return bin-level arrays for evaluation."""
        return {
            "pixel_counts": self.bin_pixel_counts,
            "covered_counts": self.bin_covered_counts,
            "score_sums": self.bin_score_sums,
            "set_size_sums": self.bin_set_size_sums,
        }
