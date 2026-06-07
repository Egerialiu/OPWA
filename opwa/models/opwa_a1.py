"""
OPWA A1 - Minimum Viable Version (VAE Decoder injection, GPPI-compatible).

Architecture:
  I_deg → VAE Encoder (frozen, in graph) → z_deg
       → UNet (frozen base + LoRA) → z_out
       → VAE Decoder (frozen, in graph):
           Each up_block receives: sample += proj[D-Enc_feat] × σ(gate[i])
       → I_rec (reconstructed)

Loss = L2(I_rec, I_clean) in pixel space
  Gradient: pixel_loss → I_rec → conv_out → up_block[3] → ... → up_block[0] → gate
  Path length: 2-8 conv layers (NOT 50-80 like UNet injection) ✅
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from diffusers import UNet2DConditionModel, AutoencoderKL

from .degradation_encoder import DegradationEncoder
from .gate import StaticGate
from .lora import add_lora_to_unet
from diffusers import LCMScheduler


class BranchProjection(nn.Module):
    """Projects a D-Enc feature map to match a VAE Decoder up_block's channel dim."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class OPWA_A1(nn.Module):
    """
    OPWA A1 — Multi-Scale Weather Adapter with VAE Decoder injection.

    GPPI-consistent design: branch features (D-Enc) are projected and gated,
    then injected directly into VAE Decoder up_blocks (NOT UNet up_blocks).

    Gradient from pixel-space loss reaches gate after 2-8 layers (VAE Decoder)
    instead of 50-80 layers (UNet).
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler=None,
        d_enc_channels: Optional[List[int]] = None,
        skip_channels: Optional[List[int]] = None,
        gate_init: Optional[List[float]] = None,
        enable_lora: bool = True,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler or LCMScheduler.from_pretrained(
            "stabilityai/sd-turbo", subfolder="scheduler"
        )

        # VAE is frozen but forward is IN the gradient graph
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # D-Enc channels: 4 scales [64, 128, 256, 512]
        d_enc_channels = d_enc_channels or [64, 128, 256, 512]
        # VAE Decoder up_block channels:
        #   up_block[0]: 512 → upsample conv_in(4→512)→mid(512)
        #   up_block[1]: 512 → upsample
        #   up_block[2]: 512→256 → upsample
        #   up_block[3]: 256→128 → upsample → conv_out(128→3)
        skip_channels = skip_channels or [512, 512, 512, 256]
        num_scales = len(d_enc_channels)

        self.degradation_encoder = DegradationEncoder(
            input_channels=3,
            base_channels=d_enc_channels[0],
            num_scales=num_scales,
            embed_dim=256,
        )

        self.projections = nn.ModuleList([
            BranchProjection(d_enc_channels[i], skip_channels[i])
            for i in range(num_scales)
        ])

        self.gate = StaticGate(num_scales=num_scales, init_values=gate_init)

        # Freeze D-Enc (stop-gradient)
        self.degradation_encoder.requires_grad_(False)

        # Freeze UNet base, inject LoRA
        for p in self.unet.parameters():
            p.requires_grad = False
        self._lora_enabled = enable_lora
        if enable_lora:
            add_lora_to_unet(self.unet, rank=lora_rank)

        # Injection into VAE Decoder up_blocks
        self._inject_features: List[Optional[torch.Tensor]] = [None] * num_scales
        self._hooks_registered = False

    def register_vae_hooks(self):
        """
        Register forward pre-hooks on VAE Decoder up_blocks.

        Each VAE Decoder up_block receives input=sample (hidden state).
        We inject our gated branch projection BEFORE the up_block processes it:
          sample = sample + proj[D-Enc_feat] × σ(gate)
        """
        if self._hooks_registered:
            return

        num_inject = len(self.projections)

        for up_idx, up_block in enumerate(self.vae.decoder.up_blocks):
            if up_idx >= num_inject:
                continue

            def make_hook(uidx: int):
                def hook(module, args):
                    inject = self._inject_features[uidx]
                    if inject is None:
                        return args
                    sample = args[0] if args else None
                    if sample is None:
                        return args
                    # Interpolate to match sample spatial size
                    if inject.shape[-2:] != sample.shape[-2:]:
                        inject = F.interpolate(inject, size=sample.shape[-2:],
                                               mode="bilinear", align_corners=False)
                    # Add to sample before up_block processes it
                    modified_sample = sample + inject
                    return (modified_sample,) + args[1:]
                return hook

            up_block.register_forward_pre_hook(make_hook(up_idx))

        self._hooks_registered = True

    def _compute_branch_and_inject(self, degraded_image: torch.Tensor):
        """Compute D-Enc features → project → gate → store for VAE hooks."""
        branch_feats, _ = self.degradation_encoder(degraded_image)
        branch_feats = [f.detach() for f in branch_feats]
        gate_vals = self.gate()
        for i, (feat, proj, g) in enumerate(zip(branch_feats, self.projections, gate_vals)):
            self._inject_features[i] = proj(feat) * g
        self.register_vae_hooks()
        return gate_vals, branch_feats

    def forward(
        self,
        degraded_image: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        clean_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass:
          1. VAE encode (in graph) → z_deg
          2. UNet (LoRA) → z_out  (frozen base, LoRA adapts for weather removal)
          3. D-Enc features → project → × gate → inject into VAE Decoder up_blocks
          4. VAE decode with injection → I_rec

        Training (clean_image given):
          Loss = L2(I_rec, I_clean) + LPIPS(I_rec, I_clean)
          Grad path: L2 → I_rec → conv_out → up_blocks → injection → gate
          Only 2-8 conv layers to reach gate (not 50-80) ✅

        Inference:
          Same forward, but no clean_image provided (no loss computed).
          Returns reconstructed image.
        """
        # 1. Branch features (D-Enc → proj → × gate)
        gate_vals, _ = self._compute_branch_and_inject(degraded_image)

        # 2. VAE encode (in graph, grad flows through frozen weights)
        #    Must NOT use @torch.no_grad() — grad needs to flow to VAE decoder
        encoded = self.vae.encode(degraded_image)
        moments = encoded.latent_dist
        z_deg = moments.sample() * self.vae.config.scaling_factor  # (B, 4, 64, 64)

        # 3. UNet translation (add noise, denoise for gradient amplification)
        alpha_prod = self.scheduler.alphas_cumprod.to(z_deg.device)[timestep.long()]
        while alpha_prod.dim() < 4:
            alpha_prod = alpha_prod.unsqueeze(-1)
        beta_prod = 1.0 - alpha_prod
        noise = torch.randn_like(z_deg)
        noisy_z = alpha_prod.sqrt() * z_deg + beta_prod.sqrt() * noise

        unet_out = self.unet(sample=noisy_z, timestep=timestep,
                             encoder_hidden_states=encoder_hidden_states)
        z_out = (noisy_z - beta_prod.sqrt() * unet_out.sample) / alpha_prod.sqrt()

        # 4. VAE decode (in graph, grad to injection features)
        z_out = z_out / self.vae.config.scaling_factor
        decoded = self.vae.decode(z_out).sample  # (B, 3, 512, 512)

        # Clean up injection features
        for i in range(len(self._inject_features)):
            self._inject_features[i] = None

        if clean_image is not None:
            return {
                "reconstructed": decoded,
                "clean_image": clean_image,
                "gate_values": gate_vals,
            }

        return {"reconstructed": decoded, "gate_values": gate_vals}

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        for p in self.degradation_encoder.parameters():
            p.requires_grad = False
        params.extend(list(self.projections.parameters()))
        params.extend(list(self.gate.parameters()))
        for name, p in self.unet.named_parameters():
            if 'lora' in name.lower():
                p.requires_grad = True
                params.append(p)
            else:
                p.requires_grad = False
        return params

    def get_total_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path="stabilityai/sd-turbo", **kwargs):
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae")
        return cls(unet=unet, vae=vae, **kwargs)
