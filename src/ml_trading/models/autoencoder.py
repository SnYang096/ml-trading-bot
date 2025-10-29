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

        if architecture == "standard":
            self._build_standard_architecture(dropout_rate)
        elif architecture == "deep":
            self._build_deep_architecture(dropout_rate)
        elif architecture == "production":
            self._build_production_architecture(dropout_rate)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
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
    ) -> None:
        self.autoencoder = autoencoder

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.autoencoder.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=20,
            factor=0.5,
        )

        self.criterion = nn.MSELoss()

    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 300,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> list:
        self.autoencoder.train()

        X_tensor = torch.FloatTensor(X_train).to(self.device)

        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            reconstructed, _ = self.autoencoder(X_tensor)
            loss = self.criterion(reconstructed, X_tensor)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

            losses.append(loss.item())

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {loss.item():.6f}")

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
