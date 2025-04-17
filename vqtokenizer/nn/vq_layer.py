import torch
import torch.nn as nn


class VQLayer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer("usage_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.zeros(num_embeddings, embedding_dim))
        self.ema_decay = 0.99

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_z = z.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_z, self.codebook.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        if self.training:
            self.usage_count.index_add_(0, encoding_indices, torch.ones_like(encoding_indices, dtype=torch.float))
        quantized = self.codebook(encoding_indices).view(z.shape)
        z_q = z + (quantized - z).detach()
        q_loss = torch.mean((quantized.detach() - z) ** 2)
        e_loss = torch.mean((quantized - z.detach()) ** 2)
        vq_loss = q_loss + self.commitment_cost * e_loss
        if self.training:
            with torch.no_grad():
                encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                dw = encodings.t() @ flat_z
                n_i = encodings.sum(0)
                self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * n_i
                self.ema_w = self.ema_decay * self.ema_w + (1 - self.ema_decay) * dw
                n_i = torch.max(n_i, torch.ones_like(n_i))
                self.codebook.weight.data = self.ema_w / n_i.unsqueeze(1)
        return z_q, vq_loss, encoding_indices
