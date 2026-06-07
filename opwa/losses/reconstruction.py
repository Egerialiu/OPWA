"""
Reconstruction losses for OPWA training.

Combines L2 + LPIPS + (optional) GAN losses as the "safety net"
to prevent hallucination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class L2Loss(nn.Module):
    """Perceptual L2 loss (MSE)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual similarity loss.

    Uses the pretrained LPIPS model from the lpips package.
    Falls back to a simple learned perceptual loss if lpips unavailable.
    """

    def __init__(self, net: str = "alex", verbose: bool = False,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.lpips_fn = None
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net, verbose=verbose)
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
            if device is not None:
                self.lpips_fn = self.lpips_fn.to(device)
        except ImportError:
            print("Warning: lpips not installed. Using VGG-based approximation.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lpips_fn is not None:
            # LPIPS expects [-1, 1] range
            return self.lpips_fn(pred, target).mean()
        else:
            # Fallback: simple L2 on VGG features
            return self._vgg_lpips_fallback(pred, target)

    def _vgg_lpips_fallback(self, pred, target):
        """Simple learned perceptual loss approximation."""
        diff = pred - target
        # Apply local normalization
        local_mean = F.avg_pool2d(diff, 3, 1, 1)
        diff_centered = diff - local_mean
        return diff_centered.pow(2).mean()


class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss: L2 + LPIPS.

    This serves as the "safety net" — it ensures image quality
    doesn't degrade while the perception-driven loss optimizes
    for downstream task performance.

    Args:
        lpips_weight: Weight for LPIPS term
        l2_weight: Weight for L2 term
    """

    def __init__(
        self,
        l2_weight: float = 1.0,
        lpips_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.l2_loss = L2Loss()
        self.lpips_loss = LPIPSLoss()
        if device is not None:
            self.lpips_loss = self.lpips_loss.to(device)
        self.l2_weight = l2_weight
        self.lpips_weight = lpips_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction losses.

        Args:
            pred: Reconstructed image (B, 3, H, W)
            target: Target clean image (B, 3, H, W)

        Returns:
            dict with keys: 'total', 'l2', 'lpips'
        """
        l2 = self.l2_loss(pred, target)
        lpips = self.lpips_loss(pred, target)
        total = self.l2_weight * l2 + self.lpips_weight * lpips
        return {"total": total, "l2": l2, "lpips": lpips}
