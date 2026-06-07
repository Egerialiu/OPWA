"""
Dataset modules for OPWA training — adapted for actual data layout.

Data layout (Pix2Pix format at /gz-data/weathersynthetic_street_png):
  train_A/  → 400 degraded (rainy) images
  train_B/  → 400 clean (sunny) images
  test_A/   → 31  degraded test images
  test_B/   → 31  clean test images
  train_albedo/ → 400 albedo maps (unused in A1)

No segmentation labels available — perception loss uses pseudo-labels
from SegFormer's predictions on clean images.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opwa.utils.weather_classifier import classify_by_tensor


@dataclass
class WeatherDatasetConfig:
    """Configuration for weather dataset construction."""

    weather_synthetic_root: Optional[str] = None  # Pix2Pix format root (train_A/train_B)
    foggy_cityscapes_root: Optional[str] = None
    acdc_root: Optional[str] = None

    image_size: Tuple[int, int] = (512, 512)
    use_augmentation: bool = False

    # Perception loss without GT labels: generate pseudo-labels from SegFormer on clean
    use_pseudo_labels: bool = False


class WeatherDataset(Dataset):
    """
    Weather dataset for OPWA A1 — reads Pix2Pix format (train_A/train_B).

    When use_pseudo_labels=True, generates segmentation pseudo-labels by
    running frozen SegFormer on the clean image. This is necessary because
    WeatherSynthetic has no semantic labels.

    Returns:
      - degraded: (3, H, W) in [-1, 1]
      - clean:    (3, H, W) in [-1, 1]
      - weather_type: str
      - label: Optional (H, W) pseudo-label (int64) if use_pseudo_labels
    """

    def __init__(
        self,
        config: WeatherDatasetConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
        perception_model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []
        self.pseudo_labels: Optional[List[torch.Tensor]] = None
        self._perception_model = perception_model
        self._device = device or torch.device("cpu")

        self._load_pix2pix()
        if config.use_pseudo_labels and perception_model is not None:
            self._generate_pseudo_labels()
        print(f"[WeatherDataset] Loaded {len(self.samples)} {split} samples"
              f"{' (with pseudo-labels)' if self.pseudo_labels is not None else ''}")

    def _load_pix2pix(self):
        """Load Pix2Pix-format data: {root}/{split}_A/ (deg) + {root}/{split}_B/ (clean)."""
        root = self.config.weather_synthetic_root
        if root is None or not os.path.isdir(root):
            return

        deg_dir = os.path.join(root, f"{self.split}_A")
        clean_dir = os.path.join(root, f"{self.split}_B")
        if not os.path.isdir(deg_dir) or not os.path.isdir(clean_dir):
            raise FileNotFoundError(
                f"Pix2Pix dirs not found at {deg_dir} / {clean_dir}"
            )

        fnames = sorted(os.listdir(deg_dir))
        for fname in fnames:
            deg_path = os.path.join(deg_dir, fname)
            clean_path = os.path.join(clean_dir, fname)
            if os.path.isfile(deg_path) and os.path.isfile(clean_path):
                self.samples.append((deg_path, clean_path))

    def _generate_pseudo_labels(self):
        """Pre-compute pseudo-labels from frozen SegFormer on clean images."""
        import torch.nn.functional as F
        print(f"  Generating pseudo-labels from frozen perception model...")
        self.pseudo_labels = []
        self._perception_model.eval()
        self._perception_model.to(self._device)
        with torch.no_grad():
            for deg_path, clean_path in self.samples:
                clean_img = self._load_image(clean_path)
                clean_tensor = self._to_tensor(clean_img) * 2.0 - 1.0
                clean_tensor = clean_tensor.unsqueeze(0).to(self._device)
                output = self._perception_model(pixel_values=clean_tensor)
                logits = output.logits  # (1, 19, H/4, W/4)
                # Upsample to clean img spatial size
                logits = F.interpolate(
                    logits, size=clean_img.shape[:2],
                    mode="bilinear", align_corners=False,
                )
                label = logits.argmax(dim=1).squeeze(0).cpu()  # (H, W)
                self.pseudo_labels.append(label)
        print(f"  Generated {len(self.pseudo_labels)} pseudo-labels.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        deg_path, clean_path = self.samples[idx]

        deg_img = self._load_image(deg_path)
        clean_img = self._load_image(clean_path)

        target_size = self.config.image_size
        if deg_img.shape[:2] != target_size:
            deg_img = self._resize(deg_img, target_size)
            clean_img = self._resize(clean_img, target_size)

        deg_tensor = self._to_tensor(deg_img) * 2.0 - 1.0
        clean_tensor = self._to_tensor(clean_img) * 2.0 - 1.0

        weather_type = classify_by_tensor(deg_tensor, clean_tensor)

        result = {
            "degraded": deg_tensor,
            "clean": clean_tensor,
            "weather_type": weather_type,
        }

        if self.pseudo_labels is not None:
            label = self.pseudo_labels[idx]
            if label.shape[:2] != target_size:
                label = F.interpolate(
                    label.unsqueeze(0).unsqueeze(0).float(),
                    size=target_size, mode="nearest",
                ).squeeze(0).squeeze(0).long()
            result["label"] = label

        return result

    def _load_image(self, path: str) -> np.ndarray:
        """Load image as RGB numpy array (H, W, 3) in [0, 1]."""
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0

    def _resize(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        from PIL import Image as PILImage
        return np.array(
            PILImage.fromarray((img * 255).astype(np.uint8)).resize(
                (size[1], size[0]), PILImage.BILINEAR
            )
        ).astype(np.float32) / 255.0

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.transpose(2, 0, 1)).float()


def create_weather_dataloader(
    config: WeatherDatasetConfig,
    batch_size: int = 4,
    num_workers: int = 2,
    split: str = "train",
    shuffle: bool = True,
    perception_model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> DataLoader:
    dataset = WeatherDataset(config, split=split,
                             perception_model=perception_model, device=device)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
