"""
Gate module for OPWA A1.

A1 uses static scalar gates (4 learnable parameters, one per scale level).
This matches GPPI's architecture where gate values are global scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class StaticGate(nn.Module):
    """
    Static learnable gate with per-scale scalar parameters.

    Based on GPPI's observation (Principle 4): gate values follow
    "deep ≈ 0.45, shallow ≈ 0.49–0.67" pattern.

    The 4 bias values are initialized to produce gate values around
    [0.55, 0.50, 0.47, 0.45] after sigmoid (shallow→deep decreasing),
    matching GPPI's learned pattern.

    Args:
        num_scales: Number of scale levels (default: 4)
        init_values: Optional custom initialization values (pre-sigmoid)
    """

    def __init__(
        self,
        num_scales: int = 4,
        init_values: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_scales = num_scales

        if init_values is not None:
            assert len(init_values) == num_scales
            bias = torch.tensor(init_values, dtype=torch.float32)
        else:
            # Initialize to produce ~[0.55, 0.50, 0.47, 0.45] after sigmoid
            # (shallow→deep decreasing, per GPPI prior)
            bias = torch.tensor([0.2, 0.0, -0.12, -0.2], dtype=torch.float32)

        self.gate_params = nn.Parameter(bias)

    def forward(self) -> torch.Tensor:
        """
        Returns gate values after sigmoid.

        Returns:
            gate_values: (4,) tensor in range (0, 1)
        """
        return torch.sigmoid(self.gate_params)

    def get_gate_values(self) -> List[float]:
        """Get current gate values as float list for logging."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_params).tolist()

    def extra_repr(self) -> str:
        with torch.no_grad():
            vals = torch.sigmoid(self.gate_params)
        return f"num_scales={self.num_scales}, current_values={vals.tolist()}"
