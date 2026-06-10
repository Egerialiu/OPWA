import numpy as np
import cv2

from config import LABEL2TRAIN


def load_gt_label_ids(gt_path):
    """Load Cityscapes gtFine_labelIds and map original IDs to train IDs.

    Cityscapes gtFine_labelIds.png uses original IDs (0-33), but SegFormer
    outputs 19 classes (train IDs 0-18). Pixels with no mapping become 255 (ignore).

    Args:
        gt_path: path to gtFine_labelIds.png
    Returns:
        gt_mapped: np.ndarray (H, W), uint8, mapped to train IDs (0-18, 255=ignore)
    """
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)  # (H, W), uint8
    if gt is None:
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt_mapped = np.full_like(gt, 255, dtype=np.uint8)
    for label_id, train_id in LABEL2TRAIN.items():
        gt_mapped[gt == label_id] = train_id
    return gt_mapped
