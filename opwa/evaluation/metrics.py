"""
Evaluation metrics for OPWA A1.

Core metric: Recovery Rate
  r_mIoU = mIoU_OPWA / mIoU_clean * 100%
  r_mAP  = mAP_OPWA  / mAP_clean  * 100%

Analysis tools (from GPPI):
  - CKA analysis for orthogonal verification
  - Gate value distribution
  - Per-class mIoU tracking
  - Gradient cosine analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class EvaluationResults:
    """Container for evaluation results."""

    mIoU_raw: float = 0.0
    mIoU_opwa: float = 0.0
    mIoU_clean: float = 0.0
    r_mIoU: float = 0.0
    per_class_iou_raw: Dict[str, float] = field(default_factory=dict)
    per_class_iou_opwa: Dict[str, float] = field(default_factory=dict)
    per_class_iou_clean: Dict[str, float] = field(default_factory=dict)
    gate_values: List[float] = field(default_factory=list)
    cka_similarity: float = 0.0


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-class IoU.

    Args:
        pred: (H, W) predicted class indices
        target: (H, W) ground truth class indices
        num_classes: Number of classes
        ignore_index: Label index to ignore

    Returns:
        ious: (num_classes,) per-class IoU
        valid: (num_classes,) boolean mask of valid classes
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    # Ignore mask
    ignore_mask = target == ignore_index

    pred = pred[~ignore_mask]
    target = target[~ignore_mask]

    if len(pred) == 0:
        return torch.zeros(num_classes), torch.zeros(num_classes, dtype=torch.bool)

    # Per-class IoU
    ious = torch.zeros(num_classes)
    valid = torch.zeros(num_classes, dtype=torch.bool)

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union > 0:
            ious[cls] = intersection / union
            valid[cls] = True

    return ious, valid


def compute_miou(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 19,
    ignore_index: int = 255,
) -> Dict[str, torch.Tensor]:
    """
    Compute mean IoU from segmentation logits.

    Args:
        logits: (B, C, H, W) segmentation logits
        labels: (B, H, W) ground truth labels
        num_classes: Number of classes
        ignore_index: Label index to ignore

    Returns:
        dict with 'mIoU' (scalar) and 'per_class' (C,) tensors
    """
    preds = logits.argmax(dim=1)  # (B, H, W)

    total_ious = torch.zeros(num_classes)
    total_counts = torch.zeros(num_classes, dtype=torch.long)
    valid_classes = torch.zeros(num_classes, dtype=torch.bool)

    for b in range(logits.shape[0]):
        ious, valid = compute_iou(preds[b], labels[b], num_classes, ignore_index)
        total_ious += ious
        total_counts += valid.long()
        valid_classes = valid_classes | valid

    # Mean over classes that appear in evaluation set
    per_class_iou = torch.where(
        total_counts > 0,
        total_ious / total_counts.float(),
        torch.zeros(num_classes),
    )

    # Final mIoU over valid classes only
    if valid_classes.any():
        miou = per_class_iou[valid_classes].mean()
    else:
        miou = torch.tensor(0.0)

    return {"mIoU": miou, "per_class": per_class_iou}


def compute_recovery_rate(
    mIoU_opwa: float,
    mIoU_clean: float,
) -> float:
    """
    Compute recovery rate r_mIoU.

    Args:
        mIoU_opwa: mIoU after OPWA restoration
        mIoU_clean: mIoU on clean (upper bound)

    Returns:
        Recovery rate as percentage
    """
    if mIoU_clean > 0:
        return (mIoU_opwa / mIoU_clean) * 100.0
    return 0.0


class CKAAnalyzer:
    """
    Centered Kernel Alignment (CKA) analysis for measuring
    representational similarity between trunk and branch features.

    Used to verify functional orthogonality (target CKA < 0.1).
    """

    def __init__(self, kernel: str = "linear"):
        self.kernel = kernel

    def compute_cka(
        self,
        features_trunk: torch.Tensor,
        features_branch: torch.Tensor,
    ) -> float:
        """
        Compute CKA similarity between two sets of features.

        Args:
            features_trunk: (N, D1) trunk features
            features_branch: (N, D2) branch features

        Returns:
            CKA similarity value (0 = independent, 1 = identical)
        """
        # Center the features
        X = features_trunk - features_trunk.mean(dim=0, keepdim=True)
        Y = features_branch - features_branch.mean(dim=0, keepdim=True)

        # Gram matrices
        if self.kernel == "linear":
            K = X @ X.T
            L = Y @ Y.T
        else:  # RBF kernel
            K = self._rbf_kernel(X)
            L = self._rbf_kernel(Y)

        # Mean-center gram matrices
        K = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
        L = L - L.mean(dim=0, keepdim=True) - L.mean(dim=1, keepdim=True) + L.mean()

        # Compute HSIC values
        hsic_kl = (K * L).sum()
        hsic_kk = (K * K).sum()
        hsic_ll = (L * L).sum()

        if hsic_kk.item() > 0 and hsic_ll.item() > 0:
            cka = hsic_kl / torch.sqrt(hsic_kk * hsic_ll)
        else:
            cka = torch.tensor(0.0)

        return cka.item()

    def _rbf_kernel(self, X: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        pairwise_sq = torch.cdist(X, X, p=2).pow(2)
        if sigma is None:
            sigma = pairwise_sq.median()
        if sigma.item() == 0:
            sigma = torch.tensor(1.0)
        return torch.exp(-pairwise_sq / (2 * sigma))


class GradientCosineAnalyzer:
    """
    Analyzes gradient conflicts between trunk and branch by computing
    cosine similarity of their gradients.

    If cosine < -0.1, significant gradient conflict exists (undesirable).
    """

    @staticmethod
    def compute_gradient_cosine(
        trunk_grads: List[torch.Tensor],
        branch_grads: List[torch.Tensor],
    ) -> float:
        """
        Compute cosine similarity between trunk and branch gradients.

        Args:
            trunk_grads: List of gradient tensors from trunk parameters
            branch_grads: List of gradient tensors from branch parameters

        Returns:
            Average cosine similarity
        """
        # Concatenate all gradients
        trunk_flat = torch.cat([g.flatten() for g in trunk_grads])
        branch_flat = torch.cat([g.flatten() for g in branch_grads])

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            trunk_flat.unsqueeze(0),
            branch_flat.unsqueeze(0),
        )

        return cos_sim.item()


class Evaluator:
    """
    Full evaluator for OPWA A1.

    Evaluates:
      1. Recovery rate (r_mIoU, r_mAP)
      2. Per-class improvement
      3. Gate value distribution
      4. CKA orthogonality (optional)
    """

    def __init__(
        self,
        perception_model: nn.Module,
        num_classes: int = 19,
        class_names: Optional[List[str]] = None,
    ):
        self.perception_model = perception_model
        self.num_classes = num_classes
        self.class_names = class_names or [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
            "sky", "person", "rider", "car", "truck", "bus", "train",
            "motorcycle", "bicycle",
        ]
        self.perception_model.eval()
        for p in self.perception_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        opwa_enabled: bool = True,
        text_embedding: Optional[torch.Tensor] = None,
        use_clean_images: bool = False,
    ) -> Dict[str, float]:
        """
        Run full evaluation.

        Args:
            model: OPWA model
            dataloader: Evaluation dataloader (returns degraded, clean, label)
            device: Device
            opwa_enabled: If True, evaluate with OPWA. If False, evaluate raw.
            text_embedding: Optional precomputed text embedding (1, 77, 1024)
            use_clean_images: If True, use 'clean' instead of 'degraded' as input

        Returns:
            dict with evaluation metrics
        """
        model.eval()
        total_miou = 0.0
        total_per_class = torch.zeros(self.num_classes)
        total_valid = torch.zeros(self.num_classes, dtype=torch.long)
        num_batches = 0

        timestep = torch.full((1,), 1, dtype=torch.long, device=device)

        for batch in dataloader:
            if use_clean_images:
                inp = batch["clean"].to(device)
            else:
                inp = batch["degraded"].to(device)
            labels = batch["label"].to(device)
            B = inp.shape[0]

            if opwa_enabled:
                if text_embedding is not None:
                    encoder_hidden_states = text_embedding.repeat(B, 1, 1)
                else:
                    encoder_hidden_states = batch.get("text_embeddings", None)
                    if encoder_hidden_states is None:
                        encoder_hidden_states = torch.zeros(
                            B, 77, 1024, device=device
                        )
                output = model(
                    inp, timestep.expand(B), encoder_hidden_states
                )
                restored = output["reconstructed"]
            else:
                restored = inp

            # Perception model forward
            perception_out = self.perception_model(pixel_values=restored)
            logits = perception_out.logits

            # Resize to label size
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Compute mIoU
            result = compute_miou(logits, labels, self.num_classes)

            if not torch.isnan(result["mIoU"]):
                total_miou += result["mIoU"].item()

            per_class = result["per_class"]
            for b in range(B):
                for c in range(self.num_classes):
                    if per_class[c].item() > 0:
                        total_per_class[c] += per_class[c].item()
                        total_valid[c] += 1

            num_batches += 1

        # Averages
        avg_miou = total_miou / max(num_batches, 1)
        per_class_iou = torch.where(
            total_valid > 0,
            total_per_class / total_valid.float(),
            torch.zeros(self.num_classes),
        )

        return {
            "mIoU": avg_miou,
            "per_class_iou": per_class_iou.cpu().numpy(),
        }

    def evaluate_full_pipeline(
        self,
        model: nn.Module,
        raw_dataloader: torch.utils.data.DataLoader,
        clean_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> EvaluationResults:
        """Run full evaluation: raw → OPWA → clean comparison."""
        # Raw degradation
        raw_metrics = self.evaluate(model, raw_dataloader, device, opwa_enabled=False)

        # OPWA restoration
        opwa_metrics = self.evaluate(model, raw_dataloader, device, opwa_enabled=True)

        # Clean images (upper bound)
        clean_metrics = self.evaluate(None, clean_dataloader, device, opwa_enabled=False)

        results = EvaluationResults(
            mIoU_raw=raw_metrics["mIoU"],
            mIoU_opwa=opwa_metrics["mIoU"],
            mIoU_clean=clean_metrics["mIoU"],
            r_mIoU=compute_recovery_rate(opwa_metrics["mIoU"], clean_metrics["mIoU"]),
        )

        # Per-class
        for i, name in enumerate(self.class_names):
            results.per_class_iou_raw[name] = float(raw_metrics["per_class_iou"][i])
            results.per_class_iou_opwa[name] = float(opwa_metrics["per_class_iou"][i])
            results.per_class_iou_clean[name] = float(clean_metrics["per_class_iou"][i])

        return results
