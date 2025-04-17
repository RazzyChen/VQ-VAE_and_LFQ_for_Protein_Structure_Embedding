import torch
import torch.nn as nn


class LFQLayer(nn.Module):
    """
    Lookup-Free Quantization Layer (LFQ) using Gumbel-Softmax for differentiable quantization.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, temperature: float = 1.0, commitment_cost: float = 0.25
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.commitment_cost = commitment_cost
        self.codebook = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.to_logits = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, z: torch.Tensor, hard: bool = False):
        # z: [batch, embedding_dim]
        logits = self.to_logits(z)  # [batch, num_embeddings]
        soft_one_hot = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
        z_q = torch.matmul(soft_one_hot, self.codebook)  # [batch, embedding_dim]
        vq_loss = self.commitment_cost * torch.mean((z.detach() - z_q) ** 2)
        indices = torch.argmax(soft_one_hot, dim=-1)
        return z_q, vq_loss, indices
