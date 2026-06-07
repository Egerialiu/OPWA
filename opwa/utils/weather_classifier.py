"""
Weather classifier for WeatherSynthetic dataset.

Classifies images into rain/fog/night by comparing degraded vs clean brightness ratio
(brightness_ratio = degraded_mean / clean_mean in grayscale).

Thresholds:
  - rain:  ratio > 1.15   (degraded brighter than clean — rain reflections)
  - fog:   ratio 0.65-1.15 (degraded similar to clean — haze scattering)
  - night: ratio < 0.65    (degraded much darker — low light)
"""

import torch
import numpy as np

_RAIN_THRESHOLD = 1.15
_NIGHT_THRESHOLD = 0.65


def classify_weather(deg_img: np.ndarray, clean_img: np.ndarray) -> str:
    """Classify weather type from numpy arrays in [0, 1] or [0, 255] range."""
    ratio = deg_img.mean() / (clean_img.mean() + 1e-6)
    return _ratio_to_weather(ratio)


def classify_by_tensor(
    deg_tensor: torch.Tensor,
    clean_tensor: torch.Tensor,
) -> str:
    """Classify weather type from [-1, 1] tensors."""
    deg_mean = deg_tensor.mean().item()
    clean_mean = clean_tensor.mean().item()
    ratio = deg_mean / (clean_mean + 1e-6)
    return _ratio_to_weather(ratio)


def classify_from_batch(batch: dict) -> list[str]:
    """
    Classify weather types for an entire batch of [-1, 1] tensors.

    Returns list of 'rain'/'fog'/'night' strings, one per sample in batch.
    """
    deg = batch["degraded"]  # (B, 3, H, W)
    clean = batch["clean"] if "clean" in batch else None
    if clean is None:
        # Estimate: assume clean is ~0 (pixel mean of typical sunny street ~0)
        clean = torch.zeros_like(deg)

    B = deg.shape[0]
    results = []
    for i in range(B):
        d_mean = deg[i].mean().item()
        c_mean = clean[i].mean().item()
        ratio = d_mean / (c_mean + 1e-6)
        results.append(_ratio_to_weather(ratio))
    return results


def _ratio_to_weather(ratio: float) -> str:
    if ratio > _RAIN_THRESHOLD:
        return "rain"
    elif ratio >= _NIGHT_THRESHOLD:
        return "fog"
    return "night"
