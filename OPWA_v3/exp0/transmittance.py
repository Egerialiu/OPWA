import numpy as np
import cv2

from config import BETA


def compute_transmittance(depth_raw):
    """Convert Depth Anything V2 raw output to transmittance t in (0, 1].

    Depth Anything V2 Small outputs disparity-style values:
        larger value = closer to camera.
    We invert so that farther objects have larger distance values,
    then apply the Beer-Lambert model: t = exp(-BETA * d_distance).

    Args:
        depth_raw: np.ndarray (H, W), raw depth model output.
    Returns:
        t_map: np.ndarray (H, W), transmittance values in (0, 1].
    """
    d_min, d_max = depth_raw.min(), depth_raw.max()
    if d_max - d_min < 1e-6:
        return np.ones_like(depth_raw, dtype=np.float32) * 0.5

    # Normalize to [0, 1]
    d_norm = (depth_raw - d_min).astype(np.float32) / (d_max - d_min)

    # Invert: disparity -> depth style (far = large)
    d_distance = 1.0 - d_norm

    # Beer-Lambert: t = exp(-beta * distance)
    t_map = np.exp(-BETA * d_distance)
    return t_map  # far ~0.05, near ~1.0


def verify_depth_direction(depth_raw):
    """Check whether Depth-Anything output is disparity-style.

    Returns (upper_third_mean, lower_third_mean).
    Disparity-style: upper (sky/far) < lower (ground/near).
    """
    H, W = depth_raw.shape
    upper = depth_raw[:H // 3, :].mean()
    lower = depth_raw[2 * H // 3:, :].mean()
    return float(upper), float(lower)


def verify_transmittance_direction(t_map):
    """Return (upper_mean_t, lower_mean_t) for transmittance.

    Expectation: upper (far/sky) should have LOW t (heavy fog),
                 lower (near/ground) should have HIGH t (clear).
    So upper_mean_t < lower_mean_t.
    """
    H, W = t_map.shape
    upper = t_map[:H // 3, :].mean()
    lower = t_map[2 * H // 3:, :].mean()
    return float(upper), float(lower)
