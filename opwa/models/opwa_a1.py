"""
OPWA A1 - VAE Decoder injection with GPPI-compatible skip connections.

Architecture:
  I_deg → VAE Encoder (frozen, in graph) → z_deg, skip_acts
       → UNet (frozen base + LoRA) → z_out
       → VAE Decoder (frozen, in graph):
           Each up_block receives: sample += trunk + branch
             trunk = skip_conv(encoder_skip_act_rev) × γ (γ=1)
             branch = proj(D-Enc_feat) × σ(gate)
       → I_rec (reconstructed)

Loss = L2 + LPIPS×5 in pixel space
  Grad path: pixel_loss → I_rec → conv_out → up_blocks → injection → gate
  Path length: 2-8 conv layers (NOT 50-80 like UNet injection) ✅
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from diffusers import UNet2DConditionModel, AutoencoderKL

from .degradation_encoder import DegradationEncoder
from .gate import StaticGate
from .weather_encoder import WeatherEncoder
from .conditional_gate import ConditionalGate
from .lora import add_lora_to_unet, add_lora_to_vae
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


class SkipConv(nn.Module):
    """1×1 conv to project VAE encoder skip activation to decoder channel dim."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class OPWA_A1(nn.Module):
    """
    OPWA A1 — Multi-Scale Weather Adapter with VAE Decoder injection.

    GPPI-consistent: skip_conv trunk + gated branch injected into decoder up_blocks.
    Trunk provides clean skip connection from VAE encoder. Branch provides
    degradation-aware features from D-Enc (or noise probe).
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
        vae_lora_rank: int = 4,
        noise_probe: bool = False,
        train_d_enc: bool = False,
        use_conditional_gate: bool = False,
        weather_embed_dim: int = 32,
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler or LCMScheduler.from_pretrained(
            "stabilityai/sd-turbo", subfolder="scheduler"
        )
        self.noise_probe = noise_probe
        self.train_d_enc = train_d_enc
        self.use_conditional_gate = use_conditional_gate
        self.gate_clamp_max = None  # set via set_gate_clamp()

        # VAE is frozen but forward is IN the gradient graph
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # Channel configs
        d_enc_channels = d_enc_channels or [64, 128, 256, 512]
        skip_channels = skip_channels or [512, 512, 512, 256]
        num_scales = len(d_enc_channels)

        self.degradation_encoder = DegradationEncoder(
            input_channels=3,
            base_channels=d_enc_channels[0],
            num_scales=num_scales,
            embed_dim=256,
            dropout=0.0,  # overridden by set_d_enc_dropout()
        )

        self.projections = nn.ModuleList([
            BranchProjection(d_enc_channels[i], skip_channels[i])
            for i in range(num_scales)
        ])

        self.gate = StaticGate(num_scales=num_scales, init_values=gate_init)

        # Conditional weather gate (A2)
        self.weather_encoder: Optional[WeatherEncoder] = None
        self.conditional_gate: Optional[ConditionalGate] = None
        if use_conditional_gate:
            self.weather_encoder = WeatherEncoder(embed_dim=weather_embed_dim)
            self.conditional_gate = ConditionalGate(
                num_layers=num_scales, embed_dim=weather_embed_dim,
            )

        # Skip convs: project encoder skip acts (reversed, deepest first) → decoder channels
        enc_skip_channels = list(vae.config.block_out_channels)[:num_scales]
        dec_up_channels = []
        for up_block in vae.decoder.up_blocks[:num_scales]:
            dec_up_channels.append(up_block.resnets[0].in_channels)
        self.skip_convs = nn.ModuleList([
            SkipConv(enc_skip_channels[::-1][i], dec_up_channels[i])
            for i in range(num_scales)
        ])

        # Conditional D-Enc freeze/unfreeze
        if not train_d_enc:
            self.degradation_encoder.requires_grad_(False)

        # Freeze UNet base, inject LoRA
        for p in self.unet.parameters():
            p.requires_grad = False
        self._lora_enabled = enable_lora
        if enable_lora:
            add_lora_to_unet(self.unet, rank=lora_rank)

        # VAE LoRA
        self._vae_lora_rank = vae_lora_rank
        if vae_lora_rank > 0:
            add_lora_to_vae(self.vae, rank=vae_lora_rank)

        # Inject state: store trunk and branch separately for per-feature interpolation
        self._trunk_features: List[Optional[torch.Tensor]] = [None] * num_scales
        self._branch_features: List[Optional[torch.Tensor]] = [None] * num_scales
        self._hooks_registered = False

        # Encoder skip-activation capture
        self._encoder_skip_acts: List[Optional[torch.Tensor]] = [None] * num_scales
        self._encoder_hooks_registered = False

    def set_gate_clamp(self, max_val: float):
        """Clamp gate values to [0, max_val] to prevent branch > trunk."""
        self.gate_clamp_max = max_val

    def set_d_enc_dropout(self, p: float):
        """Set dropout on all D-Enc layers post-hoc."""
        for m in self.degradation_encoder.modules():
            if isinstance(m, nn.Dropout2d):
                m.p = p

    def register_encoder_hooks(self):
        """Capture VAE encoder down_block outputs for skip_conv trunk."""
        if self._encoder_hooks_registered:
            return
        num_blocks = min(len(self.vae.encoder.down_blocks), len(self._encoder_skip_acts))
        for i in range(num_blocks):

            def make_hook(idx):
                def hook(module, input, output):
                    self._encoder_skip_acts[idx] = output
                return hook

            self.vae.encoder.down_blocks[i].register_forward_hook(make_hook(i))
        self._encoder_hooks_registered = True

    def register_vae_hooks(self):
        """
        Register forward pre-hooks on VAE Decoder up_blocks.

        GPPI injection:
          sample = sample + trunk + branch
          trunk = skip_conv(encoder_skip_act) × γ (γ=1)
          branch = proj(D-Enc_feat) × σ(gate)
        """
        if self._hooks_registered:
            return

        num_inject = len(self.projections)

        for up_idx, up_block in enumerate(self.vae.decoder.up_blocks):
            if up_idx >= num_inject:
                continue

            def make_hook(uidx: int):
                def hook(module, args):
                    sample = args[0]
                    if sample is None:
                        return args

                    trunk = self._trunk_features[uidx]
                    branch = self._branch_features[uidx]

                    inject = torch.zeros_like(sample)
                    if trunk is not None:
                        if trunk.shape[-2:] != sample.shape[-2:]:
                            trunk = F.interpolate(trunk, size=sample.shape[-2:],
                                                  mode="bilinear", align_corners=False)
                        inject = inject + trunk
                    if branch is not None:
                        if branch.shape[-2:] != sample.shape[-2:]:
                            branch = F.interpolate(branch, size=sample.shape[-2:],
                                                   mode="bilinear", align_corners=False)
                        inject = inject + branch

                    modified_sample = sample + inject
                    return (modified_sample,) + args[1:]
                return hook

            up_block.register_forward_pre_hook(make_hook(up_idx))

        self._hooks_registered = True

    def _compute_branch(self, degraded_image: torch.Tensor) -> List[torch.Tensor]:
        """Compute branch features: D-Enc or random noise."""
        if self.noise_probe:
            B, _, H, W = degraded_image.shape
            device = degraded_image.device
            feats = []
            for i, dc in enumerate(self.degradation_encoder.get_feature_dims()):
                h = H // (2 ** (i + 1))
                w = W // (2 ** (i + 1))
                feats.append(torch.randn(B, dc, h, w, device=device))
            return feats
        else:
            branch_feats, _ = self.degradation_encoder(degraded_image)
            return branch_feats

    def _compute_and_store_injection(self, degraded_image: torch.Tensor):
        """
        Compute trunk + branch features and store for decoder hooks.

        Two modes:
          A1 (StaticGate):
            gate = σ(learned_param)          — scalar per layer, shared across batch
            branch = proj(D-Enc_feat) × gate
          A2 (ConditionalGate):
            weather_embed = WeatherEncoder(I_deg)     — (B, 32) per image
            gate_i = MLP(weather_embed, i/(N-1))      — (B,) per layer, per sample
            branch = proj(D-Enc_feat) × gate_i
        """
        branch_feats = self._compute_branch(degraded_image)

        if self.use_conditional_gate and self.weather_encoder is not None and self.conditional_gate is not None:
            # A2: per-sample, per-layer conditional gate
            weather_embed = self.weather_encoder(degraded_image)  # (B, 32)
            gate_vals = self.conditional_gate(weather_embed)      # (4, B)
        else:
            # A1: static scalar gate shared across batch
            gate_vals = self.gate()  # (4,)

        # Trunk from encoder skip acts (reversed: deepest first → up_blocks order)
        encoder_acts_rev = self._encoder_skip_acts[::-1]

        for i in range(len(self.projections)):
            # Branch: proj(D-Enc_feat) × gate
            if self.use_conditional_gate:
                # Conditional: (B,) per-sample gate → broadcast to (B, 1, 1, 1)
                g = gate_vals[i]  # (B,)
                if self.gate_clamp_max is not None:
                    g = torch.clamp(g, max=self.gate_clamp_max)
                g = g.view(-1, 1, 1, 1)
            else:
                # Static: scalar gate, shared across batch
                g = gate_vals[i]
                if self.gate_clamp_max is not None:
                    g = torch.clamp(g, max=self.gate_clamp_max)
            branch = self.projections[i](branch_feats[i]) * g

            # Trunk: skip_conv(encoder_skip_act) × γ (γ=1, no gamma param yet)
            enc_act = encoder_acts_rev[i]
            if enc_act is not None:
                trunk = self.skip_convs[i](enc_act.detach())
            else:
                trunk = None

            self._trunk_features[i] = trunk
            self._branch_features[i] = branch

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
          1. VAE encode — captures encoder skip_acts for skip_conv trunk
          2. D-Enc (or noise) → proj → × gate → branch
          3. trunk + branch stored for decoder hook injection
          4. UNet translation
          5. VAE decode — hooks inject trunk+branch into each up_block
        """
        # 1. Register encoder hooks & reset skip_acts
        if not self._encoder_hooks_registered:
            self.register_encoder_hooks()
        for i in range(len(self._encoder_skip_acts)):
            self._encoder_skip_acts[i] = None

        # 2. VAE encode (hooks fire → capture skip_acts, in grad graph)
        encoded = self.vae.encode(degraded_image)
        moments = encoded.latent_dist
        z_deg = moments.sample() * self.vae.config.scaling_factor  # (B, 4, 64, 64)

        # 3. Compute injection features (trunk + branch) from captured skip_acts
        gate_vals, _ = self._compute_and_store_injection(degraded_image)

        # 4. UNet translation (add noise, denoise for gradient amplification)
        alpha_prod = self.scheduler.alphas_cumprod.to(z_deg.device)[timestep.long()]
        while alpha_prod.dim() < 4:
            alpha_prod = alpha_prod.unsqueeze(-1)
        beta_prod = 1.0 - alpha_prod
        noise = torch.randn_like(z_deg)
        noisy_z = alpha_prod.sqrt() * z_deg + beta_prod.sqrt() * noise

        unet_out = self.unet(sample=noisy_z, timestep=timestep,
                             encoder_hidden_states=encoder_hidden_states)
        z_out = (noisy_z - beta_prod.sqrt() * unet_out.sample) / alpha_prod.sqrt()

        # 5. VAE decode (hooks inject trunk+branch into up_blocks)
        z_out = z_out / self.vae.config.scaling_factor
        decoded = self.vae.decode(z_out).sample  # (B, 3, 512, 512)

        # 6. Clean up stored features
        for i in range(len(self._trunk_features)):
            self._trunk_features[i] = None
            self._branch_features[i] = None
        for i in range(len(self._encoder_skip_acts)):
            self._encoder_skip_acts[i] = None

        if clean_image is not None:
            return {
                "reconstructed": decoded,
                "clean_image": clean_image,
                "gate_values": gate_vals,
            }

        return {"reconstructed": decoded, "gate_values": gate_vals}

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        # Projections, gate, skip_convs
        params.extend(list(self.projections.parameters()))
        params.extend(list(self.gate.parameters()))
        params.extend(list(self.skip_convs.parameters()))
        # UNet LoRA params
        for name, p in self.unet.named_parameters():
            if 'lora' in name.lower():
                p.requires_grad = True
                params.append(p)
            else:
                p.requires_grad = False
        # VAE LoRA params
        for name, p in self.vae.named_parameters():
            if 'vae_lora' in name.lower():
                p.requires_grad = True
                params.append(p)
            else:
                p.requires_grad = False
        # Weather Encoder + Conditional Gate (A2)
        if self.use_conditional_gate:
            if self.weather_encoder is not None:
                for p in self.weather_encoder.parameters():
                    p.requires_grad = True
                    params.append(p)
            if self.conditional_gate is not None:
                for p in self.conditional_gate.parameters():
                    p.requires_grad = True
                    params.append(p)

        # D-Enc (conditional)
        if self.train_d_enc:
            for p in self.degradation_encoder.parameters():
                p.requires_grad = True
                params.append(p)
        else:
            for p in self.degradation_encoder.parameters():
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
