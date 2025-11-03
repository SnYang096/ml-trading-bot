"""Unified autoencoder architectures and trainers for dimensionality workflows."""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


class UnifiedAutoencoder(nn.Module):
    """Flexible autoencoder supporting multiple architectures."""

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 8,
        architecture: str = "standard",
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.architecture = architecture
        self.is_vae = False  # Will be set to True for VAE architecture

        if architecture == "standard":
            self._build_standard_architecture(dropout_rate)
        elif architecture == "deep":
            self._build_deep_architecture(dropout_rate)
        elif architecture == "production":
            self._build_production_architecture(dropout_rate)
        elif architecture == "vae":
            self._build_vae_architecture(dropout_rate)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _build_standard_architecture(self, dropout_rate: float) -> None:
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.encoding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.input_dim),
        )

    def _build_deep_architecture(self, dropout_rate: float) -> None:
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, self.encoding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.input_dim),
        )

    def _build_production_architecture(self, dropout_rate: float) -> None:
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, self.encoding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.input_dim),
        )

    def _build_vae_architecture(self, dropout_rate: float) -> None:
        """Build Variational Autoencoder (VAE) architecture."""
        # Encoder: maps input to mean and log-variance
        self.encoder_base = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.encoder_mu = nn.Linear(32, self.encoding_dim)
        self.encoder_logvar = nn.Linear(32, self.encoding_dim)

        # Decoder: maps latent code to reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.input_dim),
        )
        self.is_vae = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_vae:
            # VAE: encode to mu and logvar, sample z, decode
            h = self.encoder_base(x)
            mu = self.encoder_mu(h)
            logvar = self.encoder_logvar(h)
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            decoded = self.decoder(z)
            return decoded, z
        else:
            # Standard AE
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_vae:
            # VAE: return mean (mu) as deterministic encoding
            h = self.encoder_base(x)
            mu = self.encoder_mu(h)
            return mu
        else:
            return self.encoder(x)

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded)

    def get_compression_ratio(self) -> float:
        return self.input_dim / self.encoding_dim

    def get_model_info(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "architecture": self.architecture,
            "compression_ratio": self.get_compression_ratio(),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }


class AutoencoderTrainer:
    """Utility class for training UnifiedAutoencoder instances."""

    def __init__(
        self,
        autoencoder: UnifiedAutoencoder,
        device: str = "auto",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        kl_weight: float = 1e-3,
        task_weight: float = 0.0,
        task_head: nn.Module | None = None,
    ) -> None:
        self.autoencoder = autoencoder
        self.kl_weight = kl_weight
        self.task_weight = task_weight
        self.task_head = task_head

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.autoencoder.to(self.device)
        if self.task_head is not None:
            self.task_head.to(self.device)

        params = list(self.autoencoder.parameters())
        if self.task_head is not None:
            params.extend(list(self.task_head.parameters()))

        self.optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=20,
            factor=0.5,
        )

        self.criterion = nn.MSELoss()
        self.task_criterion = None  # Will be set based on task type

    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 300,
        batch_size: int = 256,
        verbose: bool = True,
        y_train: np.ndarray | None = None,
    ) -> list:
        self.autoencoder.train()
        if self.task_head is not None:
            self.task_head.train()

        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = None
        if y_train is not None:
            y_tensor = torch.FloatTensor(y_train).to(self.device)

        losses = []
        for epoch in range(epochs):
            # Batch training for efficiency
            epoch_losses = []
            for i in range(0, len(X_train), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                self.optimizer.zero_grad()

                if self.autoencoder.is_vae:
                    # VAE: compute reconstruction + KL divergence
                    recon, z = self.autoencoder(batch_X)
                    recon_loss = self.criterion(recon, batch_X)
                    
                    # KL divergence loss
                    h = self.autoencoder.encoder_base(batch_X)
                    mu = self.autoencoder.encoder_mu(h)
                    logvar = self.autoencoder.encoder_logvar(h)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    
                    loss = recon_loss + self.kl_weight * kl_loss
                    
                    # Task loss if enabled
                    if self.task_weight > 0 and self.task_head is not None and y_tensor is not None:
                        batch_y = y_tensor[i:i+batch_size]
                        task_pred = self.task_head(z)
                        if self.task_criterion is None:
                            # Auto-detect task type: treat as classification if values are integers in {0,1,2}
                            unique_vals = torch.unique(batch_y)
                            is_small_cardinality = unique_vals.numel() <= 3
                            # Check integer-like: max fractional part close to 0
                            frac_max = torch.max(torch.abs(batch_y - torch.round(batch_y)))
                            is_integer_like = frac_max.item() < 1e-6
                            if is_small_cardinality and is_integer_like:
                                # Classification
                                self.task_criterion = nn.CrossEntropyLoss()
                                task_pred = task_pred.reshape(-1, task_pred.shape[-1])
                                batch_y = batch_y.view(-1).long()
                            else:
                                # Regression
                                self.task_criterion = nn.MSELoss()
                                task_pred = task_pred.squeeze()
                        else:
                            # Ensure correct shapes/dtypes for existing criterion
                            if isinstance(self.task_criterion, nn.CrossEntropyLoss):
                                task_pred = task_pred.reshape(-1, task_pred.shape[-1])
                                batch_y = batch_y.view(-1).long()
                        task_loss = self.task_criterion(task_pred, batch_y)
                        loss = loss + self.task_weight * task_loss
                else:
                    # Standard AE: reconstruction only
                    recon, z = self.autoencoder(batch_X)
                    loss = self.criterion(recon, batch_X)
                    
                    # Task loss if enabled
                    if self.task_weight > 0 and self.task_head is not None and y_tensor is not None:
                        batch_y = y_tensor[i:i+batch_size]
                        task_pred = self.task_head(z)
                        if self.task_criterion is None:
                            unique_vals = torch.unique(batch_y)
                            is_small_cardinality = unique_vals.numel() <= 3
                            frac_max = torch.max(torch.abs(batch_y - torch.round(batch_y)))
                            is_integer_like = frac_max.item() < 1e-6
                            if is_small_cardinality and is_integer_like:
                                self.task_criterion = nn.CrossEntropyLoss()
                                task_pred = task_pred.reshape(-1, task_pred.shape[-1])
                                batch_y = batch_y.view(-1).long()
                            else:
                                self.task_criterion = nn.MSELoss()
                                task_pred = task_pred.squeeze()
                        else:
                            if isinstance(self.task_criterion, nn.CrossEntropyLoss):
                                task_pred = task_pred.reshape(-1, task_pred.shape[-1])
                                batch_y = batch_y.view(-1).long()
                        task_loss = self.task_criterion(task_pred, batch_y)
                        loss = loss + self.task_weight * task_loss

                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            self.scheduler.step(avg_loss)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}")

        return losses

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.autoencoder.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            encoded = self.autoencoder.encode(X_tensor)
            return encoded.cpu().numpy()

    def save_model(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.autoencoder.state_dict(),
                "model_info": self.autoencoder.get_model_info(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


__all__ = ["UnifiedAutoencoder", "AutoencoderTrainer"]
