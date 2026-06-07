"""
Perception-driven loss for OPWA.

Computes cross-entropy loss between the frozen perception model's
predictions on the restored image vs. ground truth labels.

This is the key difference from GPPI: instead of only maximizing
image quality, we directly optimize for downstream perception performance.

Safety mechanism:
  - λ_p warmup: perception loss is only introduced after ~500 steps
    of pure reconstruction pre-training (prevents early hallucination)
  - Reconstruction loss acts as "safety net" during perception fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union


class PerceptionDrivenLoss(nn.Module):
    """
    Perception-driven loss that penalizes perception model errors
    on restored images.

    Core idea: after OPWA translates a degraded image, a frozen
    perception model (e.g., SegFormer-b0, YOLO) should perform
    nearly as well as on clean images.

    The loss is computed as the standard task loss (CE for segmentation,
    or detection loss for detection) between:
      - F_percept(OPWA(I_deg))  ← prediction on restored image
      - Y_gt                     ← ground truth labels

    Only the OPWA parameters receive gradients. The perception model
    remains frozen.

    Args:
        loss_type: Task type — 'segmentation' or 'detection'
        ignore_index: Index to ignore in CE loss (default: 255 for seg)
    """

    def __init__(
        self,
        loss_type: str = "segmentation",
        ignore_index: int = 255,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.ignore_index = ignore_index

        if loss_type == "segmentation":
            self.task_loss = nn.CrossEntropyLoss(
                ignore_index=ignore_index, reduction="mean"
            )
        else:
            # For detection, we'd use a more complex loss (e.g., from YOLO).
            # For A1, we focus on segmentation.
            self.task_loss = nn.CrossEntropyLoss(
                ignore_index=ignore_index, reduction="mean"
            )

    def forward(
        self,
        restored_image: torch.Tensor,
        perception_model: nn.Module,
        gt_labels: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute perception-driven loss.

        Args:
            restored_image: (B, 3, H, W) OPWA output image
            perception_model: Frozen perception model (e.g., SegFormer)
            gt_labels: (B, H, W) ground truth segmentation labels
                        or detection targets
            gt_boxes: Optional detection boxes

        Returns:
            dict with keys: 'total' containing perception loss value (scalar),
                            'details' for logging
        """
        # Ensure perception model is in eval mode and requires no grad
        was_training = perception_model.training
        perception_model.eval()

        with torch.no_grad():
            for p in perception_model.parameters():
                p.requires_grad_(False)

        # Forward through perception model
        if self.loss_type == "segmentation":
            outputs = perception_model(pixel_values=restored_image)
            logits = outputs.logits  # (B, num_classes, H, W)

            # Resize logits to match label spatial dimensions if needed
            if logits.shape[-2:] != gt_labels.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=gt_labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss_value = self.task_loss(logits, gt_labels)

        else:
            raise NotImplementedError(
                f"Loss type '{self.loss_type}' not implemented in A1"
            )

        # Restore training mode
        if was_training:
            perception_model.train()

        return {"total": loss_value}

    def compute_warmup_weight(
        self, step: int, warmup_start: int = 500, warmup_end: int = 1500
    ) -> float:
        """
        Compute perception loss weight based on training step.

        Warmup schedule:
          - step < warmup_start: weight = 0 (pure reconstruction)
          - warmup_start ≤ step ≤ warmup_end: linear ramp
          - step > warmup_end: weight = 1.0

        Args:
            step: Current training step
            warmup_start: Step to start perception training
            warmup_end: Step to reach full perception loss weight

        Returns:
            float: Weight multiplier for perception loss (0.0 to 1.0)
        """
        if step < warmup_start:
            return 0.0
        elif step > warmup_end:
            return 1.0
        else:
            return (step - warmup_start) / (warmup_end - warmup_start)
