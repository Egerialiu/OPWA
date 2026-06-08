"""
Physics-based branch for GPPI — domain-specific feature extractors for
weather types where learned D-Enc fails (fog, haze).

Dark Channel Prior (DCP):
  For foggy images, computes a transmission map estimate t(x):
    t(x) = 1 - min_{c∈{R,G,B}} min_{y∈Ω(x)} I_c(y) / A_c

  Where I is the degraded image, A is atmospheric light, and Ω(x) is a
  local patch. This provides a spatial distribution of fog density that
  the decoder can use for position-aware de-scattering.

Key insight: D-Enc fails on fog because fog is multiplicative scattering
(depth-dependent, non-local), which a lightweight CNN cannot learn in
500 steps. DCP provides the correct inductive bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DCPBranch(nn.Module):
    """
    Dark Channel Prior feature extractor for foggy image restoration.

    Pipeline:
      1. Compute dark channel from degraded image
      2. Estimate transmission map
      3. Extract multi-scale features via lightweight encoder

    Args:
        feat_channels: Output channel dimensions for multi-scale features
            (default: [64, 128, 256, 512], matching GPPI's injection points)
        window_size: Local patch size for min-filter (default: 15)
    """

    def __init__(
        self,
        feat_channels: Optional[List[int]] = None,
        window_size: int = 15,
    ):
        super().__init__()
        self.window_size = window_size
        feat_channels = feat_channels or [64, 128, 256, 512]
        num_scales = len(feat_channels)

        # Tiny encoder for DCP transmission map (1 input channel)
        # Much lighter than D-Enc (3→64 stem) since input is already processed
        self.stem = nn.Sequential(
            nn.Conv2d(1, feat_channels[0], kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(min(8, feat_channels[0]), feat_channels[0]),
            nn.SiLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList()
        in_ch = feat_channels[0]
        for i in range(1, num_scales):
            out_ch = feat_channels[i]
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(16, out_ch), out_ch),
                nn.SiLU(inplace=True),
            ))
            in_ch = out_ch

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_dark_channel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dark channel prior from normalized image.

        Args:
            x: (B, 3, H, W) image in [0, 1] range

        Returns:
            dark: (B, 1, H, W) dark channel in [0, 1]
        """
        # Min across RGB channels
        dark, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W)

        # Min filter over local window (morphological erosion)
        padding = self.window_size // 2
        dark = -F.max_pool2d(-dark, kernel_size=self.window_size,
                             stride=1, padding=padding)

        return dark

    def _estimate_transmission(self, dark_channel: torch.Tensor,
                               omega: float = 0.95) -> torch.Tensor:
        """
        Estimate transmission map: t(x) = 1 - ω * dark_channel(x)

        Args:
            dark_channel: (B, 1, H, W) dark channel values
            omega: DCP omega parameter, typical 0.95

        Returns:
            transmission: (B, 1, H, W) estimated transmission in [0, 1]
        """
        return 1.0 - omega * dark_channel

    def get_feature_dims(self) -> List[int]:
        """Return channel dimensions of multi-scale outputs."""
        return [64, 128, 256, 512][:4]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute multi-scale features from DCP.

        Args:
            x: (B, 3, H, W) degraded image in [-1, 1] (normalized)

        Returns:
            features: List of feature maps at decreasing scales
            global_embed: (B, feat_channels[-1]) global DCP embedding
        """
        # Normalize to [0, 1] for DCP computation
        x_01 = (x + 1.0) / 2.0

        # Compute DCP transmission map
        dark = self._compute_dark_channel(x_01)       # (B, 1, H, W)
        transmission = self._estimate_transmission(dark)  # (B, 1, H, W)

        # Multi-scale feature extraction from transmission map
        features = []
        f = self.stem(transmission)  # (B, C0, H/2, W/2)
        features.append(f)

        for block in self.down_blocks:
            f = block(f)
            features.append(f)

        # Global embedding from deepest feature
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        global_embed = pooled

        return features, global_embed
