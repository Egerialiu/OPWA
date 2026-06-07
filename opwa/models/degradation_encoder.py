"""
Degradation Encoder (D-Enc) for OPWA A1.

GPPI-equivalent MultiScaleEncoder (stride=2 stem) that extracts
multi-scale features from degraded images.

Architecture (matching GPPI's MultiScaleEncoder):
  - Input:  3 × H × W (degraded image, 512×512)
  - Stem:   3→64, 7×7 conv stride=2, GN + SiLU →  64 × 256 × 256
  - Down 1: 64→128, 3×3 conv stride=2, GN + SiLU → 128 × 128 × 128
  - Down 2: 128→256, 3×3 conv stride=2, GN + SiLU → 256 × 64 × 64
  - Down 3: 256→512, 3×3 conv stride=2, GN + SiLU → 512 × 32 × 32
  - Output: features at 4 scales [256, 128, 64, 32] spatial dims
  - Global embedding: GAP + MLP (512 → 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DegradationEncoder(nn.Module):
    """
    Light-weight multi-scale degradation encoder (GPPI-equivalent).

    Args:
        input_channels: Number of input channels (default: 3 for RGB)
        base_channels: Base channel count (default: 64, doubled per layer)
        num_scales: Number of scale levels to output (default: 4)
        embed_dim: Dimension of global embedding (default: 256)
    """

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        num_scales: int = 4,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim

        # Stem: stride=2 to match GPPI
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(inplace=True),
        )

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_scales - 1):
            out_ch = base_channels * (2 ** (i + 1))
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(min(16, out_ch), out_ch),
                    nn.SiLU(inplace=True),
                )
            )
            in_ch = out_ch

        # Global embedding MLP (from deepest feature)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embed_mlp = nn.Sequential(
            nn.Linear(in_ch, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable early training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input degraded image (B, 3, H, W), typically 512×512

        Returns:
            multi_scale_features: List of feature maps at decreasing scales:
                [(B, 64, 256, 256), (B, 128, 128, 128),
                 (B, 256, 64, 64), (B, 512, 32, 32)]
            global_embedding: (B, 256) global degradation embedding
        """
        # Stem: stride=2 → spatial /2
        f = self.stem(x)  # (B, 64, 256, 256)
        features = [f]

        # Down blocks: each stride=2 → spatial /2 per level
        for block in self.down_blocks:
            f = block(f)
            features.append(f)

        # Ensure exactly num_scales features
        features = features[:self.num_scales]

        # Global embedding from deepest feature
        pooled = self.global_pool(f)  # (B, C, 1, 1)
        pooled = pooled.flatten(1)  # (B, C)
        embedding = self.embed_mlp(pooled)  # (B, 256)

        return features, embedding

    def get_feature_dims(self) -> List[int]:
        """Return channel dimensions of each scale output."""
        dims = [64]
        for i in range(self.num_scales - 1):
            dims.append(64 * (2 ** (i + 1)))
        return dims[:self.num_scales]
