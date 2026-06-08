"""
Weather Encoder for OPWA A2 — conditional gate weather embedding.

Learns a compact 32-dim weather representation from degraded images
in a fully end-to-end manner (no weather labels needed).

Architecture:
  Input:  (B, 3, H, W)  — typically 512×512
    → Conv2d(3→16, k7, s4) → ReLU    # 512→128
    → Conv2d(16→32, k5, s4) → ReLU   # 128→32
    → Conv2d(32→64, k3, s4) → ReLU   # 32→8
    → AdaptiveAvgPool2d(1)            # 8→1
    → Linear(64 → 32)                 # (B, 32)
  Output: (B, 32) weather embedding

Params: ~150K
"""

import torch
import torch.nn as nn


class WeatherEncoder(nn.Module):
    """
    Light-weight weather encoder for conditional gate generation.

    Learns a 32-dim weather embedding from pixel-level features of the
    degraded image. Trained end-to-end with the rest of OPWA — no
    weather supervision required.

    Args:
        embed_dim: Output embedding dimension (default: 32)
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            # 3×512×512 → 16×128×128
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            # 16×128×128 → 32×32×32
            nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # 32×32×32 → 64×8×8
            nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            # 64×8×8 → 64×1×1
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize for stable early training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute weather embedding from degraded image.

        Args:
            x: Degraded image (B, 3, H, W), typically 512×512

        Returns:
            Weather embedding (B, embed_dim) in range roughly [-1, 1]
            after linear head (no output activation).
        """
        feat = self.net(x).flatten(1)   # (B, 64)
        return self.head(feat)           # (B, embed_dim)
