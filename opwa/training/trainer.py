"""
OPWA A1 Trainer — Two-stage training pipeline.

Architecture: GPPI-style VAE Decoder skip injection + pixel-space loss.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from opwa.models import OPWA_A1
from opwa.losses import ReconstructionLoss, PerceptionDrivenLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs/opwa_a1"
    checkpoint_dir: str = "./checkpoints"
    batch_size: int = 4
    learning_rate: float = 1e-4
    lora_lr: float = 1e-4
    gate_lr: float = 5e-4
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    stage1_steps: int = 1000
    stage2_steps: int = 1000
    total_steps: int = 2000
    l2_weight: float = 1.0
    lpips_weight: float = 0.0
    percept_weight_max: float = 0.5
    gate_reg_weight: float = 0.0
    warmup_start: int = 500
    warmup_end: int = 1500
    log_interval: int = 20
    eval_interval: int = 200
    save_interval: int = 500
    max_checkpoints: int = 5
    device: str = "cuda"
    mixed_precision: str = "fp16"
    prompt: str = "a photo of a street scene, clear weather, high quality"
    track_per_class: bool = True


class OPWATrainer:
    def __init__(self, model, config, dataloader, eval_dataloader=None,
                 perception_model=None, text_embeddings=None):
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        self.perception_model = perception_model
        self.text_embeddings = text_embeddings
        self.device = torch.device(config.device)
        self.rec_loss_fn = ReconstructionLoss(l2_weight=config.l2_weight,
            lpips_weight=config.lpips_weight, device=self.device)
        self.percept_loss_fn = PerceptionDrivenLoss(loss_type="segmentation")
        self.model.to(self.device)
        if self.perception_model is not None:
            self.perception_model.to(self.device)
            self.perception_model.eval()
            for p in self.perception_model.parameters():
                p.requires_grad = False
        if text_embeddings is not None:
            self.text_embeddings = text_embeddings.to(self.device)
        self.optimizer = self._build_optimizer()
        self.global_step = 0
        self.current_stage = 1
        self.best_metric = 0.0
        self.checkpoint_paths = []

    def _build_optimizer(self) -> optim.Optimizer:
        proj_params = []
        gate_params = []
        lora_params = []
        other_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "gate" in name:
                gate_params.append(p)
            elif "lora" in name.lower():
                lora_params.append(p)
            elif "projection" in name:
                proj_params.append(p)
            else:
                other_params.append(p)
        param_groups = [
            {"params": proj_params + other_params, "lr": self.config.learning_rate},
            {"params": gate_params, "lr": self.config.gate_lr},
            {"params": lora_params, "lr": self.config.lora_lr},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        logger.info(f"Optimizer groups: proj={len(proj_params)}, gate={len(gate_params)}, "
                     f"lora={len(lora_params)}, other={len(other_params)}")
        logger.info(f"  gate_lr={self.config.gate_lr}")
        return optim.AdamW(param_groups, betas=(self.config.adam_beta1, self.config.adam_beta2),
                           weight_decay=self.config.weight_decay)

    def _get_text_embeddings(self, batch_size: int) -> torch.Tensor:
        if self.text_embeddings is not None:
            return self.text_embeddings.repeat(batch_size, 1, 1)
        return None

    def _get_timestep(self, batch_size: int) -> torch.Tensor:
        if self.model.training:
            return torch.randint(50, 201, (batch_size,), dtype=torch.long, device=self.device)
        return torch.full((batch_size,), 1, dtype=torch.long, device=self.device)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step: pixel-space L2 loss (gradient through VAE Decoder to gate)."""
        degraded = batch["degraded"].to(self.device)
        clean = batch["clean"].to(self.device)
        B = degraded.shape[0]
        timestep = self._get_timestep(B)
        encoder_hidden_states = self._get_text_embeddings(B)

        # Forward: I_rec in pixel space
        output = self.model(degraded, timestep, encoder_hidden_states, clean_image=clean)
        gate_vals = output["gate_values"]

        # Pixel-space L2 loss — gradient flows through VAE Decoder to gate ✅
        rec_loss = torch.nn.functional.mse_loss(output["reconstructed"], clean)

        # Gate regularization (mild, to allow movement)
        gate_reg = torch.zeros(1, device=self.device)

        total_loss = rec_loss + gate_reg

        # Perception-driven loss (stage 2 only)
        percept_loss_val = torch.zeros(1, device=self.device)
        lambda_p = self.percept_loss_fn.compute_warmup_weight(
            self.global_step,
            warmup_start=self.config.warmup_start,
            warmup_end=self.config.warmup_end,
        )
        if lambda_p > 0 and self.perception_model is not None and "label" in batch:
            gt_labels = batch["label"].to(self.device)
            with torch.no_grad():
                percept_out = self.percept_loss_fn(
                    output["reconstructed"], self.perception_model, gt_labels)
            percept_loss_val = percept_out["total"]
            total_loss = total_loss + lambda_p * self.config.percept_weight_max * percept_loss_val

        total_loss.backward()

        # Log gate gradient
        for name, p in self.model.named_parameters():
            if 'gate' in name and p.grad is not None:
                logger.info(f"Gate grad: {p.grad.abs().mean().item():.8f}  "
                           f"val: {torch.sigmoid(p.data).item():.6f}")

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1

        return {
            "total_loss": total_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "percept_loss": percept_loss_val.detach(),
            "gate_reg": gate_reg.detach(),
            "gate_vals": gate_vals.detach(),
            "lambda_p": torch.tensor(lambda_p, device=self.device),
            "l2_loss": rec_loss.detach(),
        }

    def train(self) -> Dict[str, List[float]]:
        logger.info(f"Starting OPWA A1 training (VAE Decoder injection + pixel L2)")
        logger.info(f"  Steps: {self.config.total_steps}, gate_lr: {self.config.gate_lr}")

        metrics = {"total_loss": [], "rec_loss": [], "percept_loss": [], "gate_vals": []}
        start_time = time.time()
        data_iter = iter(self.dataloader)

        while self.global_step < self.config.total_steps:
            self.current_stage = 2 if self.global_step >= self.config.stage1_steps else 1
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            step_metrics = self.train_step(batch)

            if self.global_step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Step {self.global_step}/{self.config.total_steps} "
                    f"Loss: {step_metrics['total_loss'].item():.4f} "
                    f"Rec(L2): {step_metrics['rec_loss'].item():.4f} "
                    f"Percept: {step_metrics['percept_loss'].item():.4f} "
                    f"λ_p: {step_metrics['lambda_p'].item():.3f} "
                    f"Gate: {step_metrics['gate_vals'].cpu().tolist()} "
                    f"({steps_per_sec:.1f} steps/s)"
                )
                for k in metrics:
                    v = step_metrics.get(k.replace("-", "_"), None)
                    if v is not None:
                        if v.numel() > 1:
                            metrics[k].append(v.cpu().tolist())
                        else:
                            metrics[k].append(v.cpu().item())

            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        logger.info(f"Training complete. Total time: {time.time() - start_time:.1f}s")
        return metrics

    def save_checkpoint(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"opwa_a1_step_{self.global_step}.pt")
        checkpoint = {
            "step": self.global_step,
            "stage": self.current_stage,
            "model_state_dict": self.model.state_dict(),
            "gate_values": self.model.gate.get_gate_values(),
        }
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_paths.append(checkpoint_path)
        while len(self.checkpoint_paths) > 3:
            old_ckpt = self.checkpoint_paths.pop(0)
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint["step"]
        self.current_stage = checkpoint["stage"]
        logger.info(f"Loaded checkpoint from step {self.global_step}: {checkpoint_path}")

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        loader = dataloader or self.eval_dataloader
        if loader is None:
            return {}
        self.model.eval()
        total_rec_loss = 0.0
        total_percept_loss = 0.0
        num_batches = 0
        for batch in loader:
            degraded = batch["degraded"].to(self.device)
            clean = batch["clean"].to(self.device)
            timestep = self._get_timestep(degraded.shape[0])
            encoder_hidden_states = self._get_text_embeddings(degraded.shape[0])
            output = self.model(degraded, timestep, encoder_hidden_states)
            reconstructed = output["reconstructed"]
            rec_losses = self.rec_loss_fn(reconstructed, clean)
            total_rec_loss += rec_losses["total"].item()
            if self.perception_model is not None and "label" in batch:
                gt_labels = batch["label"].to(self.device)
                percept_out = self.percept_loss_fn(reconstructed, self.perception_model, gt_labels)
                total_percept_loss += percept_out["total"].item()
            num_batches += 1
        self.model.train()
        return {"eval_rec_loss": total_rec_loss / max(num_batches, 1),
                "eval_percept_loss": total_percept_loss / max(num_batches, 1)}
