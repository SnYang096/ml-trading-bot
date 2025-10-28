#!/usr/bin/env python3
"""
统一的Autoencoder类
整合所有降维训练中的Autoencoder实现
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class UnifiedAutoencoder(nn.Module):
    """
    统一的Autoencoder类，支持不同的网络架构
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 8,
        architecture: str = "standard",
        dropout_rate: float = 0.2,
    ):
        """
        初始化Autoencoder

        Args:
            input_dim: 输入特征维度
            encoding_dim: 编码维度
            architecture: 网络架构类型 ('standard', 'deep', 'production')
            dropout_rate: Dropout率
        """
        super(UnifiedAutoencoder, self).__init__()

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

    def _build_standard_architecture(self, dropout_rate: float):
        """标准架构"""
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

    def _build_deep_architecture(self, dropout_rate: float):
        """深度架构"""
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

    def _build_production_architecture(self, dropout_rate: float):
        """生产级架构"""
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
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            (reconstructed, encoded): 重构结果和编码结果
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码"""
        return self.encoder(x)

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """仅解码"""
        return self.decoder(encoded)

    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        return self.input_dim / self.encoding_dim

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "architecture": self.architecture,
            "compression_ratio": self.get_compression_ratio(),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }


class AutoencoderTrainer:
    """Autoencoder训练器"""

    def __init__(
        self,
        autoencoder: UnifiedAutoencoder,
        device: str = "auto",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        """
        初始化训练器

        Args:
            autoencoder: Autoencoder模型
            device: 设备 ('auto', 'cpu', 'cuda')
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.autoencoder = autoencoder

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.autoencoder.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=20, factor=0.5
        )

        self.criterion = nn.MSELoss()

    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 300,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> list:
        """
        训练Autoencoder

        Args:
            X_train: 训练数据
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否显示训练过程

        Returns:
            losses: 训练损失列表
        """
        self.autoencoder.train()

        # 转换为张量
        X_tensor = torch.FloatTensor(X_train).to(self.device)

        losses = []

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            reconstructed, encoded = self.autoencoder(X_tensor)
            loss = self.criterion(reconstructed, X_tensor)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

            losses.append(loss.item())

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {loss.item():.6f}")

        return losses

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据为编码表示

        Args:
            X: 输入数据

        Returns:
            encoded: 编码后的数据
        """
        self.autoencoder.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            encoded = self.autoencoder.encode(X_tensor)
            return encoded.cpu().numpy()

    def save_model(self, path: str):
        """保存模型"""
        torch.save(
            {
                "model_state_dict": self.autoencoder.state_dict(),
                "model_info": self.autoencoder.get_model_info(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
