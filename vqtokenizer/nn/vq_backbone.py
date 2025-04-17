"""
vqtokenizer/nn/vq_backbone.py
Description:
VQTokenizer backbone using VQLayer only.
"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.nn.functional import mse_loss

from .module import MLPDecoder, TransformerEncoder
from .vq_layer import VQLayer


class VQTokenizer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_embeddings: int,
        nhead: int = 4,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = TransformerEncoder(
            input_dim=input_dim, d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2, num_layers=3
        )
        self.projection = nn.Linear(hidden_dim, latent_dim)
        self.vq_layer = VQLayer(num_embeddings, latent_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, input_dim)
        self.learning_rate = learning_rate
        self.codebook_usage_hist: Tensor | None = None


    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        z_e = self.encoder(x.unsqueeze(1))
        z_e = self.projection(z_e.squeeze(1))
        z_q, vq_loss, indices = self.vq_layer(z_e)
        reconstructed = self.decoder(z_q)
        return reconstructed, vq_loss, indices

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:

        x = batch
        reconstructed, vq_loss, indices = self(x)
        recon_loss = mse_loss(reconstructed, x)
        total_loss = recon_loss + vq_loss
        with torch.no_grad():
            codebook_usage = (self.vq_layer.usage_count > 0).float().mean().item() * 100
            if self.codebook_usage_hist is None:
                self.codebook_usage_hist = self.vq_layer.usage_count.clone().cpu()
            else:
                self.codebook_usage_hist += self.vq_layer.usage_count.clone().cpu()
            if batch_idx == 0:
                self.vq_layer.usage_count.zero_()
        self.log("train_recon_loss", recon_loss, prog_bar=True)
        self.log("train_vq_loss", vq_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)
        self.log("codebook_usage_percent", codebook_usage, prog_bar=True)
        return total_loss

    # ...existing code from backbone.py...
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # ...existing code from backbone.py...
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

    # ...existing code from backbone.py...
    def on_train_epoch_end(self) -> None:
        # ...existing code from backbone.py...
        if self.codebook_usage_hist is not None:
            try:
                self.logger.experiment.log(
                    {"codebook_usage_histogram": wandb.Histogram(self.codebook_usage_hist.numpy())}
                )
                total_usage = self.codebook_usage_hist.sum().item()
                if total_usage > 0:
                    normalized_usage = self.codebook_usage_hist / total_usage
                    active_codebook_percent = (normalized_usage > 0.00001).float().mean().item() * 100
                    self.log("active_codebook_percent", active_codebook_percent)
                    sorted_usage, _ = torch.sort(normalized_usage)
                    n = len(sorted_usage)
                    indices = torch.arange(1, n + 1, device=sorted_usage.device)
                    gini = (2 * (indices * sorted_usage).sum() / (n * sorted_usage.sum())) - (n + 1) / n
                    self.log("codebook_gini_coefficient", gini.item())
                self.codebook_usage_hist = None
            except Exception as e:
                print(f"Failed to log histogram: {e}")

    # ...existing code from backbone.py...
    def configure_optimizers(self):
        # ...existing code from backbone.py...
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_total_loss", "frequency": 1},
        }

    # ...existing code from backbone.py...
    def encode(self, x: Tensor) -> Tensor:
        z_e = self.encoder(x.unsqueeze(1)).squeeze(1)
        z_e = self.projection(z_e)
        _, _, indices = self.vq_layer(z_e)
        return indices

    def decode(self, indices: Tensor) -> Tensor:
        quantized = self.vq_layer.codebook(indices)
        return self.decoder(quantized)
