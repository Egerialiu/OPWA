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
from opwa.losses.gan import PatchGANDiscriminator, HingeGANLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs/opwa_a1"
    checkpoint_dir: str = "./checkpoints"
    batch_size: int = 4
    learning_rate: float = 1e-4
    lora_lr: float = 5e-6
    gate_lr: float = 5e-4
    weight_decay: float = 1e-5
    d_enc_weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    stage1_steps: int = 1000
    stage2_steps: int = 1000
    total_steps: int = 0  # 0 = auto: stage1 + stage2
    l2_weight: float = 1.0
    lpips_weight: float = 5.0
    gan_weight: float = 0.5
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
    gate_clamp_max: Optional[float] = None
    prompt: str = "a photo of a street scene, clear weather, high quality"
    track_per_class: bool = True

    # A2 Conditional Gate
    use_conditional_gate: bool = False
    weather_encoder_lr: float = 1e-4
    conditional_gate_lr: float = 5e-4


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
        # Discriminator for GAN training
        self.gan_weight = config.gan_weight
        if self.gan_weight > 0:
            self.discriminator = PatchGANDiscriminator().to(self.device)
            self.gan_loss_fn = HingeGANLoss()
            self.optimizer_D = optim.AdamW(
                self.discriminator.parameters(),
                lr=2e-4, betas=(0.5, 0.999), weight_decay=config.weight_decay,
            )
        else:
            self.discriminator = None
        self.global_step = 0
        self.current_stage = 1
        self.best_metric = 0.0
        self.checkpoint_paths = []

    def _build_optimizer(self) -> optim.Optimizer:
        proj_params = []
        gate_params = []
        lora_params = []
        d_enc_params = []
        skip_conv_params = []
        weather_enc_params = []
        cond_gate_params = []
        other_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "gate" in name and "conditional_gate" in name:
                cond_gate_params.append(p)
            elif "gate" in name:
                gate_params.append(p)
            elif "weather_encoder" in name:
                weather_enc_params.append(p)
            elif "conditional_gate" in name:
                cond_gate_params.append(p)
            elif "lora" in name.lower():
                lora_params.append(p)
            elif "projection" in name:
                proj_params.append(p)
            elif "skip_convs" in name:
                skip_conv_params.append(p)
            elif "degradation_encoder" in name:
                d_enc_params.append(p)
            else:
                other_params.append(p)
        param_groups = [
            {"params": proj_params + other_params, "lr": self.config.learning_rate},
            {"params": gate_params + skip_conv_params, "lr": self.config.gate_lr},
            {"params": lora_params, "lr": self.config.lora_lr},
            {"params": d_enc_params, "lr": self.config.lora_lr, "weight_decay": self.config.d_enc_weight_decay},
        ]
        if self.config.use_conditional_gate:
            if weather_enc_params:
                param_groups.append({"params": weather_enc_params, "lr": self.config.weather_encoder_lr})
            if cond_gate_params:
                param_groups.append({"params": cond_gate_params, "lr": self.config.conditional_gate_lr})
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        logger.info(f"Optimizer groups: proj={len(proj_params)}, gate+skip_conv={len(gate_params)+len(skip_conv_params)}, "
                     f"lora+d_enc={len(lora_params)+len(d_enc_params)}, other={len(other_params)}")
        if self.config.use_conditional_gate:
            logger.info(f"  weather_encoder={len(weather_enc_params)} (lr={self.config.weather_encoder_lr}), "
                        f"cond_gate={len(cond_gate_params)} (lr={self.config.conditional_gate_lr})")
        logger.info(f"  gate_lr={self.config.gate_lr} (shared with skip_conv), "
                     f"d_enc_lr={self.config.lora_lr} (shared with LoRA)")
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
        """Training step with L2 + LPIPS reconstruction + optional GAN adversarial loss."""
        degraded = batch["degraded"].to(self.device)
        clean = batch["clean"].to(self.device)
        B = degraded.shape[0]
        timestep = self._get_timestep(B)
        encoder_hidden_states = self._get_text_embeddings(B)

        # 1. Generator forward
        output = self.model(degraded, timestep, encoder_hidden_states, clean_image=clean)
        gate_vals = output["gate_values"]
        reconstructed = output["reconstructed"]

        # 2. Reconstruction loss (L2 + LPIPS)
        rec_losses = self.rec_loss_fn(reconstructed, clean)
        rec_loss = rec_losses["total"]
        lpips_val = rec_losses["lpips"]

        total_loss = rec_loss

        # 3. GAN loss (alternating D/G)
        gan_d_loss = torch.zeros(1, device=self.device)
        gan_g_loss = torch.zeros(1, device=self.device)
        if self.gan_weight > 0:
            # Train D
            real_pred = self.discriminator(clean)
            fake_pred = self.discriminator(reconstructed.detach())
            d_loss = self.gan_loss_fn.d_loss(real_pred, fake_pred)
            d_loss.backward()
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
            gan_d_loss = d_loss.detach()

            # Train G (forward through updated D)
            fake_pred2 = self.discriminator(reconstructed)
            g_loss = self.gan_loss_fn.g_loss(fake_pred2)
            total_loss = total_loss + self.gan_weight * g_loss
            gan_g_loss = g_loss.detach()

        # Gate regularization
        gate_reg = torch.zeros(1, device=self.device)
        total_loss = total_loss + gate_reg

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
                    reconstructed, self.perception_model, gt_labels)
            percept_loss_val = percept_out["total"]
            total_loss = total_loss + lambda_p * self.config.percept_weight_max * percept_loss_val

        total_loss.backward()

        # Zero D grads from G loss backward through D
        if self.gan_weight > 0:
            self.discriminator.zero_grad()

        # Log gate gradient (every log_interval)
        if self.global_step % self.config.log_interval == 0:
            for name, p in self.model.named_parameters():
                if 'gate' in name and p.grad is not None:
                    if self.config.use_conditional_gate:
                        logger.info(f"CondGate grad: {p.grad.abs().mean().item():.8f}")
                    else:
                        logger.info(f"Gate grad: {p.grad.abs().mean().item():.8f}  "
                                   f"val: {torch.sigmoid(p.data).mean().item():.6f}")

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1

        return {
            "total_loss": total_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "l2_loss": rec_losses["l2"].detach(),
            "lpips_loss": lpips_val.detach(),
            "gan_d_loss": gan_d_loss,
            "gan_g_loss": gan_g_loss,
            "percept_loss": percept_loss_val.detach(),
            "gate_reg": gate_reg.detach(),
            "gate_vals": gate_vals.detach(),
            "lambda_p": torch.tensor(lambda_p, device=self.device),
        }

    def train(self) -> Dict[str, List[float]]:
        logger.info(f"Starting OPWA A1 training (VAE Decoder injection + pixel L2)")
        logger.info(f"  Steps: {self.config.total_steps}, gate_lr: {self.config.gate_lr}")

        metrics = {"total_loss": [], "rec_loss": [], "l2_loss": [], "lpips_loss": [],
                    "gan_d_loss": [], "gan_g_loss": [],
                    "percept_loss": [], "gate_vals": []}
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
                gan_info = ""
                if self.gan_weight > 0:
                    gan_info = f"D:{step_metrics['gan_d_loss'].item():.4f} G:{step_metrics['gan_g_loss'].item():.4f} "
                gate_vals = step_metrics['gate_vals']
                if self.config.use_conditional_gate:
                    # Conditional gate: (4, B) → show per-layer mean across batch
                    gate_mean = gate_vals.mean(dim=1).cpu().tolist()
                    gate_info = f"Gate(μ): {[f'{v:.3f}' for v in gate_mean]} "
                else:
                    gate_info = f"Gate: {gate_vals.cpu().tolist()} "
                logger.info(
                    f"Step {self.global_step}/{self.config.total_steps} "
                    f"Loss: {step_metrics['total_loss'].item():.4f} "
                    f"L2: {step_metrics['l2_loss'].item():.4f} "
                    f"LPIPS: {step_metrics['lpips_loss'].item():.4f} "
                    + gan_info +
                    f"Percept: {step_metrics['percept_loss'].item():.4f} "
                    f"λ_p: {step_metrics['lambda_p'].item():.3f} "
                    + gate_info +
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
        # Only save trainable params (~4MB), skip frozen UNet/VAE (3.8GB)
        trainable_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        trainable_state = {
            k: v.cpu() for k, v in self.model.state_dict().items() if k in trainable_names
        }
        checkpoint = {
            "step": self.global_step,
            "stage": self.current_stage,
            "trainable_state_dict": trainable_state,
            "gate_values": self.model.gate.get_gate_values() if hasattr(self.model.gate, 'get_gate_values') else None,
        }
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_paths.append(checkpoint_path)
        while len(self.checkpoint_paths) > 3:
            old_ckpt = self.checkpoint_paths.pop(0)
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
        size_mb = os.path.getsize(checkpoint_path) / 1e6
        logger.info(f"Checkpoint saved ({size_mb:.0f}MB): {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("trainable_state_dict", checkpoint.get("model_state_dict"))
        self.model.load_state_dict(state, strict=False)
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
