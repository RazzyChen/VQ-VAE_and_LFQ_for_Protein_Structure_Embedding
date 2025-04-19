"""
vqtokenizer/nn/lfq_backbone.py
Description:
LFQTokenizer backbone using LFQLayer only.
"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.nn.functional import mse_loss

from .lfq_layer import LFQLayer
from .module import MLPDecoder, TransformerEncoder


class LFQTokenizer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        nhead: int = 4,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = TransformerEncoder(
            input_dim=input_dim, d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2, num_layers=3
        )
        self.projection = nn.Linear(hidden_dim, latent_dim)
        self.lfq_layer = LFQLayer(embedding_dim=latent_dim, temperature=temperature, commitment_cost=commitment_cost)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, input_dim)
        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_e = self.encoder(x)
        z_e = self.projection(z_e)
        z_q, vq_loss, indices = self.lfq_layer(z_e)
        reconstructed = self.decoder(z_q)
        return reconstructed, vq_loss, indices

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x = batch
        reconstructed, vq_loss, indices = self(x)
        recon_loss = mse_loss(reconstructed, x)
        total_loss = recon_loss + vq_loss
        self.log("train_recon_loss", recon_loss, prog_bar=True)
        self.log("train_vq_loss", vq_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x = batch
        reconstructed, vq_loss, indices = self(x)
        recon_loss = mse_loss(reconstructed, x)
        total_loss = recon_loss + vq_loss
        self.log("val_recon_loss", recon_loss)
        self.log("val_vq_loss", vq_loss)
        self.log("val_total_loss", total_loss)
        unique_indices = torch.unique(indices)
        self.log("val_unique_codes", len(unique_indices))
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_total_loss", "frequency": 1},
        }

    def encode(self, x: Tensor) -> Tensor:
        z_e = self.encoder(x)
        z_e = self.projection(z_e)
        _, _, indices = self.lfq_layer(z_e)
        return indices

    def decode(self, indices: Tensor) -> Tensor:
        quantized = torch.matmul(
            torch.nn.functional.one_hot(indices, num_classes=self.lfq_layer.num_embeddings).float(),
            self.lfq_layer.codebook,
        )
        return self.decoder(quantized)
