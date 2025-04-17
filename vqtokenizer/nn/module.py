"""
vqtokenizer/nn/module.py
Description:
This module contains the implementation of TransformerEncoder and MLPDecoder
which are used for encoding and decoding, respectively.
"""

import torch
import torch.nn as nn

from .lfq_layer import LFQLayer
from .transformerblock import TransformerBlock
from .vq_layer import VQLayer


# TransformerEncoder model with three blocks
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, dim_feedforward: int, num_layers: int = 3) -> None:
        super().__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Stack of transformer blocks using nn.Sequential
        self.layers = nn.Sequential(*[TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)])

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to model dimension
        x = self.input_projection(x)

        # Pass through transformer blocks
        x = self.layers(x)

        # Final projection
        x = self.output_projection(x)

        return x


# VQ-VAE decoder
class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
