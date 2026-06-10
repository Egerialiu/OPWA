import os
import glob

from config import CITYSCAPES_ROOT, FOGGY_ROOT, CALIBRATION_COUNT


def get_calibration_files():
    """Return sorted list of Cityscapes val leftImg8bit paths, first 250.

    Cityscapes val: frankfurt(267) + lindau(59) + munster(174) = 500
    Sorted alphabetically across all cities, take first 250.
    """
    pattern = os.path.join(CITYSCAPES_ROOT, "leftImg8bit", "val", "*/*.png")
    all_files = sorted(glob.glob(pattern))
    assert len(all_files) == 500, f"Expected 500 Cityscapes val files, got {len(all_files)}"
    return all_files[:CALIBRATION_COUNT]


def get_test_files():
    """Return sorted list of Foggy Cityscapes val leftImg8bit paths, all 500."""
    pattern = os.path.join(FOGGY_ROOT, "leftImg8bit_foggy", "val", "*/*_foggy_beta_0.02.png")
    all_files = sorted(glob.glob(pattern))
    assert len(all_files) == 500, f"Expected 500 Foggy val files, got {len(all_files)}"
    return all_files


def match_gt(foggy_path):
    """Convert a Foggy Cityscapes leftImg8bit path to the corresponding
    Cityscapes gtFine_labelIds path.

    Example:
      input:  /gz-data/foggy_cityscapes/leftImg8bit_foggy/val/frankfurt/frankfurt_000000_000294_leftImg8bit_foggy_beta_0.02.png
      output: /gz-data/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png
    """
    # Remove _foggy_beta_0.02 suffix
    gt_path = foggy_path.replace("_foggy_beta_0.02", "")
    # Swap directory structure
    gt_path = gt_path.replace("leftImg8bit_foggy", "gtFine")
    gt_path = gt_path.replace("leftImg8bit", "gtFine_labelIds")
    gt_path = gt_path.replace("/foggy_cityscapes/", "/cityscapes/")
    return gt_path


def verify_gt_mapping(num_samples=5):
    """Sanity-check GT mapping for a few files."""
    test_files = get_test_files()
    sampled = test_files[:num_samples]
    for f in sampled:
        gt = match_gt(f)
        exists = os.path.exists(gt)
        print(f"  {os.path.basename(f)}")
        print(f"    -> {os.path.basename(gt)}  {'[EXISTS]' if exists else '[MISSING]'}")
    return all(os.path.exists(match_gt(f)) for f in sampled)
