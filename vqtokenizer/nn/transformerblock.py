"""
vqtokenizer/nn/transformerblock.py
Description:
This module implements the TransformerBlock, which includes multi-head self-attention and a FFN with SwiGLU activation.
"""

import torch
import torch.nn as nn

from ..utils.functional import SwiGLU


class TransformerBlock(nn.Module):
    """
    TransformerBlock: A single transformer block with pre-normalization and multi-layer FFN using SwiGLU.
    Args:
        d_model (int): Model dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout rate.
    Input:
        x (Tensor): [batch_size, seq_len, d_model]
    Output:
        x (Tensor): [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Pre-normalization for attention layer
        self.norm = nn.RMSNorm(d_model)

        # Feedforward network with SwiGLU and pre-normalization
        self.feed_forward = nn.Sequential(
            nn.RMSNorm(d_model),  # Pre-normalization
            nn.Linear(d_model, dim_feedforward * 2),  # 2x for SwiGLU
            SwiGLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.RMSNorm(d_model),  # Pre-normalization
            nn.Linear(d_model, dim_feedforward * 2),  # 2x for SwiGLU
            SwiGLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.RMSNorm(d_model),  # Pre-normalization
            nn.Linear(d_model, dim_feedforward * 2),  # 2x for SwiGLU
            SwiGLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization
        attn_input = self.norm(x)

        # Self-attention layer
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input)
        x = x + attn_output  # Residual connection

        # Feedforward network (with pre-normalization)
        x = x + self.feed_forward(x)  # Residual connection

        return x
