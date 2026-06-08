"""
LoRA (Low-Rank Adaptation) integration for OPWA UNet fine-tuning.

Implements LoRA layers that can be injected into SD-Turbo UNet blocks,
keeping base weights frozen. Per GPPI Principle 5: only LoRA + Gate
parameters need to be trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL
from typing import Dict, List, Optional, Tuple


class LoRALinear(nn.Module):
    """Low-Rank Linear layer for LoRA on linear/conv weights."""

    def __init__(
        self,
        original_weight: nn.Parameter,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        is_conv: bool = False,
        kernel_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.is_conv = is_conv
        self.original_weight = original_weight
        self.original_weight.requires_grad = False

        if is_conv:
            # LoRA for Conv2d: decompose into two convs
            self.lora_down = nn.Conv2d(
                in_features, rank, kernel_size=kernel_size,
                padding=kernel_size[0] // 2 if kernel_size else 0,
                bias=False,
            )
            self.lora_up = nn.Conv2d(
                rank, out_features, kernel_size=1, bias=False,
            )
        else:
            # LoRA for Linear
            self.lora_down = nn.Linear(in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, out_features, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=0.01)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_conv:
            base = F.conv2d(
                x, self.original_weight,
                stride=1,
                padding=self.lora_down.padding,
            )
            lora_out = self.lora_up(self.lora_down(x)) * self.scaling
            return base + lora_out

        base = F.linear(x, self.original_weight)
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling
        return base + lora_out


def add_lora_to_unet(
    unet: UNet2DConditionModel,
    target_modules: Optional[List[str]] = None,
    rank: int = 4,
    alpha: float = 1.0,
) -> UNet2DConditionModel:
    """
    Inject LoRA layers into UNet attention + FFN linear layers via forward hooks.

    Uses forward_hook on each target nn.Linear: hook computes LoRA(x) = (x@A^T)@B^T * scaling
    and adds it to the original output. LoRA parameters are registered on the UNet
    module tree with names containing 'lora', so OPWA_A1.get_trainable_parameters()
    and the trainer optimizer auto-detect them.

    Args:
        unet: Pretrained UNet2DConditionModel
        target_modules: Layer name substrings to target (default: to_q/to_k/to_v/proj/FFN)
        rank: LoRA rank
        alpha: LoRA scaling factor

    Returns:
        UNet with LoRA injected
    """
    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out", "ff"]

    scaling = alpha / rank
    hooks = []

    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "time_emb" in name:
            continue
        # Match: to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2
        layer_name = name.split(".")[-1]
        matched = False
        for t in target_modules:
            if t == "to_out" and "to_out" in name and layer_name == "0":
                matched = True
                break
            if t == "ff" and ("ff.net.0.proj" in name or "ff.net.2" in name):
                matched = True
                break
            if t in ["to_q", "to_k", "to_v"] and layer_name == t:
                matched = True
                break
        if not matched:
            continue

        in_f, out_f = module.in_features, module.out_features
        safe_name = name.replace(".", "_")

        # Register LoRA A (down) and B (up) as named params on UNet
        lora_down = nn.Parameter(torch.zeros(rank, in_f))
        lora_up = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(lora_down, a=0.01)
        # lora_up stays zero-init => LoRA starts as no-op

        unet.register_parameter(f"lora_down_{safe_name}", lora_down)
        unet.register_parameter(f"lora_up_{safe_name}", lora_up)

        # Hook: compute LoRA and add to output
        def make_hook(down: nn.Parameter, up: nn.Parameter, s: float):
            def hook(_module, _input, output):
                x = _input[0]
                lora_out = (x @ down.T) @ up.T * s
                return output + lora_out
            return hook

        hook = module.register_forward_hook(make_hook(lora_down, lora_up, scaling))
        hooks.append(hook)

    # Freeze UNet base, keep LoRA trainable
    for p in unet.parameters():
        p.requires_grad = False
    for name, p in unet.named_parameters():
        if "lora_down_" in name or "lora_up_" in name:
            p.requires_grad = True

    return unet


class LoRAWrapper(nn.Module):
    """
    Wrapper that adds LoRA to a subset of UNet linear layers.

    Usage:
        unet = ... # pretrained UNet
        lora_wrapper = LoRAWrapper(unet, rank=4)
        lora_wrapper.unwrap()  # restore original forward
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        alpha: float = 1.0,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.unet = unet
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q", "k", "v", "out", "ff"]

        # Freeze UNet base
        for p in self.unet.parameters():
            p.requires_grad = False

        # Register LoRA parameters
        self.lora_ups = nn.ParameterDict()
        self.lora_downs = nn.ParameterDict()

        self._inject_lora()

    def _inject_lora(self):
        """Find target linear layers and register LoRA weights."""
        for name, module in self.unet.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            layer_name = name.split('.')[-1]
            if not any(t in layer_name for t in self.target_modules):
                continue
            if 'time_emb' in name or 'norm' in name:
                continue

            in_f, out_f = module.in_features, module.out_features
            safe_name = name.replace('.', '_')

            # Register LoRA A and B
            down = nn.Parameter(torch.zeros(rank, in_f))
            up = nn.Parameter(torch.zeros(out_f, rank))

            nn.init.kaiming_uniform_(down, a=0.01)

            self.lora_downs[safe_name] = down
            self.lora_ups[safe_name] = up

            # Store original weight reference for forward
            setattr(self, f'_lora_weight_{safe_name}', module.weight)
            setattr(self, f'_lora_scale_{safe_name}', self.alpha / self.rank)

    def forward(self, *args, **kwargs):
        """
        Forward through UNet with LoRA applied.

        We monkey-patch the forward of target Linear layers.
        This is done via _apply_lora context.
        """
        return self._forward_with_lora(*args, **kwargs)

    def _forward_with_lora(self, sample, timestep, encoder_hidden_states=None, **kwargs):
        """UNet forward with LoRA weight updates applied."""
        original_forwards = {}

        try:
            # Monkey-patch target linear layers
            for name, module in self.unet.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                safe_name = name.replace('.', '_')
                if safe_name not in self.lora_downs:
                    continue

                original_forwards[name] = module.forward
                down = self.lora_downs[safe_name]
                up = self.lora_ups[safe_name]
                scale = getattr(self, f'_lora_scale_{safe_name}')

                def make_lora_forward(orig_fwd, lora_down, lora_up, scaling):
                    def lora_forward(x):
                        base = orig_fwd(x)
                        # Apply LoRA: x @ A^T @ B^T * scaling
                        lora_out = (x @ lora_down.T) @ lora_up.T * scaling
                        return base + lora_out
                    return lora_forward

                module.forward = make_lora_forward(
                    module.forward, down, up, scale
                )

            return self.unet(sample, timestep, encoder_hidden_states, **kwargs)

        finally:
            # Restore original forwards
            for name, module in self.unet.named_modules():
                if name in original_forwards:
                    module.forward = original_forwards[name]

    def get_lora_params(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimizer."""
        return list(self.lora_ups.values()) + list(self.lora_downs.values())

    def merge_lora_weights(self):
        """Merge LoRA weights into UNet (permanent)."""
        for name, module in self.unet.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            safe_name = name.replace('.', '_')
            if safe_name not in self.lora_downs:
                continue

            down = self.lora_downs[safe_name]
            up = self.lora_ups[safe_name]
            scale = getattr(self, f'_lora_scale_{safe_name}')

            # W' = W + B @ A * scaling
            delta = (up @ down) * scale
            with torch.no_grad():
                module.weight.data.add_(delta)

    def unwrap(self) -> UNet2DConditionModel:
        """Remove LoRA and return base UNet."""
        self.merge_lora_weights()
        return self.unet


def add_lora_to_vae(
    vae: AutoencoderKL,
    rank: int = 4,
    alpha: float = 1.0,
) -> AutoencoderKL:
    """
    Inject LoRA into ALL VAE Decoder Conv2d layers (resnets, upsamplers, mid, conv_in, conv_out).

    Uses the same hook pattern as add_lora_to_unet.
    """
    scaling = alpha / rank
    hooks = []

    targeted = 0
    for name, module in vae.decoder.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        # Skip groupnorm (not Conv2d) — already filtered by isinstance
        in_c, out_c = module.in_channels, module.out_channels
        ks = module.kernel_size
        safe_name = name.replace(".", "_")

        lora_down = nn.Parameter(torch.zeros(rank, in_c, 1, 1))
        lora_up = nn.Parameter(torch.zeros(out_c, rank, 1, 1))
        nn.init.kaiming_uniform_(lora_down, a=0.01)
        # lora_up zero-init => LoRA starts as no-op

        vae.register_parameter(f"vae_lora_down_{safe_name}", lora_down)
        vae.register_parameter(f"vae_lora_up_{safe_name}", lora_up)

        def make_hook(down: nn.Parameter, up: nn.Parameter, s: float):
            def hook(_module, _input, output):
                x = _input[0]
                lora_out = F.conv2d(F.conv2d(x, down), up) * s
                return output + lora_out
            return hook

        hook = module.register_forward_hook(make_hook(lora_down, lora_up, scaling))
        hooks.append(hook)
        targeted += 1

    # Freeze VAE base, keep LoRA trainable
    for p in vae.parameters():
        p.requires_grad = False
    for name, p in vae.named_parameters():
        if "vae_lora_down_" in name or "vae_lora_up_" in name:
            p.requires_grad = True

    return vae
