"""
Conditional Gate for OPWA A2 — per-layer, per-sample gating driven by
weather embedding.

Transforms a weather embedding + normalized layer index into a
per-layer gate value via a tiny MLP:

    g_i(x) = sigmoid(MLP(concat(e_weather, i / (num_layers-1))))

Each layer queries the MLP independently with its own layer index,
so the gate MLP learns a weather-dependent allocation strategy
across the 4 decoder up_blocks.

Architecture:
    Input:  (B, embed_dim) weather embedding
      → for each layer i:
          concat(embed, i/(num_layers-1))  # (B, embed_dim+1)
          → Linear(embed_dim+1, 64) → GELU → Linear(64, 1) → sigmoid
    Output: (num_layers, B) gate values in (0, 1)

Params: ~3K
"""

import torch
import torch.nn as nn
from typing import Optional


class ConditionalGate(nn.Module):
    """
    Weather-conditional gate with per-layer query.

    For each scale layer i, computes:
        g_i = sigmoid(MLP([weather_embed, i/(num_layers-1)]))

    This lets the MLP learn a weather-dependent allocation curve
    across layers (e.g. "fog → shallow high, deep low").

    Args:
        num_layers: Number of scale levels (default: 4)
        embed_dim: Weather embedding dimension (default: 32)
        hidden_dim: MLP hidden dimension (default: 64)
    """

    def __init__(
        self,
        num_layers: int = 4,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Init bias of final layer so gate starts ~0.27 (sigmoid(-1.0))
        # This ensures branch starts weak and grows selectively.
        nn.init.constant_(self.mlp[-1].bias, -1.0)

    def forward(self, weather_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional gate values per layer.

        Args:
            weather_embed: Weather embedding (B, embed_dim)

        Returns:
            gate_values: (num_layers, B) tensor in (0, 1)
        """
        B = weather_embed.shape[0]
        device = weather_embed.device
        gates = []

        for i in range(self.num_layers):
            # Normalized layer index [0, 1]
            idx = torch.full((B, 1), float(i) / max(self.num_layers - 1, 1),
                             device=device)
            x = torch.cat([weather_embed, idx], dim=1)  # (B, embed_dim+1)
            g = torch.sigmoid(self.mlp(x)).squeeze(-1)   # (B,)
            gates.append(g)

        return torch.stack(gates, dim=0)  # (num_layers, B)

    def get_mean_gate_values(self) -> list:
        """
        Get mean gate values across batch for logging.

        Returns:
            List of num_layers floats, each averaged over the batch dimension.
        """
        return []

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}, embed_dim={self.embed_dim}"
