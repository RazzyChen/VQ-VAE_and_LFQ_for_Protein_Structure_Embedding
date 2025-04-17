import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LFQLayer(nn.Module):
    """
    Lookup-Free Quantization Layer (LFQ) following https://arxiv.org/abs/2310.05737
    - Codebook is all possible {-1, 1}^D combinations (not learnable)
    - Forward: direct sign quantization, binary encoding for indices
    - Loss: entropy aux loss + commitment loss
    """

    def __init__(
        self,
        embedding_dim: int,
        temperature: float = 1.0,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.1,
        diversity_gamma: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.diversity_gamma = diversity_gamma
        # codebook_size = 2^embedding_dim
        self.codebook_size = 2**embedding_dim
        # mask for binary encoding
        self.register_buffer("mask", 2 ** torch.arange(embedding_dim - 1, -1, -1))
        # codebook: all possible {-1, 1}^D
        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = bits * 2 - 1  # {-1, 1}
        self.register_buffer("codebook", codebook)

    def forward(self, z: torch.Tensor):
        # z: [batch, embedding_dim]
        # Quantize: sign(z) → {-1, 1}
        quantized = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))
        # Indices: binary encoding
        indices = ((quantized > 0).int() * self.mask.int()).sum(dim=-1)
        # Commitment loss
        commit_loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
        # Entropy loss (per-sample and batch)
        # Compute similarity to codebook
        sim = torch.matmul(z, self.codebook.t())  # [batch, codebook_size]
        prob = F.softmax(sim / self.temperature, dim=-1)  # [batch, codebook_size]
        per_sample_entropy = (-prob * (prob.clamp_min(1e-8)).log()).sum(dim=-1).mean()
        avg_prob = prob.mean(dim=0)
        batch_entropy = (-avg_prob * (avg_prob.clamp_min(1e-8)).log()).sum()
        entropy_aux_loss = per_sample_entropy - self.diversity_gamma * batch_entropy
        # 总损失
        loss = commit_loss + self.entropy_loss_weight * entropy_aux_loss
        return quantized, loss, indices
