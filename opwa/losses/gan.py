"""
PatchGAN discriminator for OPWA A1 adversarial training.

Simplified PatchGAN — 4-layer Conv2d with LeakyReLU and GroupNorm.
Outputs a 30×30 patch-level confidence map.
Hinge loss for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.

    Maps (B, 3, H, W) → (B, 1, H//8, W//8) patch confidences.
    Each output element sees a ~70×70 receptive field.
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            # 512×512 → 256×256
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 256×256 → 128×128
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(16, ndf * 2), ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128×128 → 64×64
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(32, ndf * 4), ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 64×64 → 64×64 (stride=1)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.GroupNorm(min(64, ndf * 8), ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 64×64 → 64×64 → 1 channel
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HingeGANLoss:
    """Hinge loss for GAN training.

    D_loss = E[ReLU(1 - D(real))] + E[ReLU(1 + D(fake))]
    G_loss = -E[D(fake)]
    """

    @staticmethod
    def d_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        return F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()

    @staticmethod
    def g_loss(fake_pred: torch.Tensor) -> torch.Tensor:
        return -fake_pred.mean()
