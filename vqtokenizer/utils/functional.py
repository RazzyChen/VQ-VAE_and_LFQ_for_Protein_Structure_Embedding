"""
vqtokenizer/utils/functional.py
Description:
This module implements the SwiGLU activation function for use in feedforward layers of neural networks.
"""

import torch


# SwiGLU activation function
class SwiGLU(torch.nn.Module):
    """SwiGLU layer."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gates) * x
