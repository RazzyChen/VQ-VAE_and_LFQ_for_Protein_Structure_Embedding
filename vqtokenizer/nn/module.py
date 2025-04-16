"""
vqtokenizer/nn/module.py
Description:
This module contains the implementation of TransformerEncoder, VQLayer, and MLPDecoder
which are used for encoding, vector quantization, and decoding, respectively.
"""

import torch
import torch.nn as nn

from .transformerblock import TransformerBlock


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


# VQ embedding layer
class VQLayer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize codebook - use uniform distribution instead of normal distribution
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # Initialize codebook usage count
        self.register_buffer("usage_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.zeros(num_embeddings, embedding_dim))

        self.ema_decay = 0.99

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute Euclidean distance between latent variables and codebook vectors
        flat_z = z.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_z, self.codebook.weight.t())
        )

        # Find the index of the nearest codebook vector
        encoding_indices = torch.argmin(distances, dim=1)

        # Update usage count (for monitoring only)
        if self.training:
            self.usage_count.index_add_(0, encoding_indices, torch.ones_like(encoding_indices, dtype=torch.float))

        # Get quantized vectors from codebook
        quantized = self.codebook(encoding_indices).view(z.shape)

        # Straight-through estimator
        # Use quantized value in forward, use original z's gradient in backward
        z_q = z + (quantized - z).detach()

        # Compute loss: commitment loss + codebook loss
        q_loss = torch.mean((quantized.detach() - z) ** 2)  # Make codebook vectors close to encoder output
        e_loss = torch.mean((quantized - z.detach()) ** 2)  # Make encoder output close to codebook vectors
        vq_loss = q_loss + self.commitment_cost * e_loss

        # Update codebook with exponential moving average (EMA)
        if self.training:
            with torch.no_grad():
                encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

                # EMA update for codebook
                dw = encodings.t() @ flat_z
                n_i = encodings.sum(0)

                # Update EMA accumulators
                self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * n_i
                self.ema_w = self.ema_decay * self.ema_w + (1 - self.ema_decay) * dw

                # Apply EMA update
                n_i = torch.max(n_i, torch.ones_like(n_i))
                self.codebook.weight.data = self.ema_w / n_i.unsqueeze(1)

        return z_q, vq_loss, encoding_indices


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
