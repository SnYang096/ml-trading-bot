"""Deep Learning Sequence Feature Extraction with Mamba/Transformer.

支持 FP16 + FlashAttention / Mamba 提取序列特征
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List, Literal
import warnings

warnings.filterwarnings("ignore")

# Check for deep learning dependencies
DL_BACKEND = None
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
    print("✓ PyTorch available")

    # Try Mamba (preferred for efficiency)
    try:
        from mamba_ssm import Mamba

        DL_BACKEND = "mamba"
        print("✓ Mamba available (O(n) complexity)")
    except ImportError:
        print("⚠️  Mamba not available, will use Transformer")

        # Try FlashAttention (preferred for Transformer)
        try:
            from flash_attn import flash_attn_qkvpacked_func

            DL_BACKEND = "flash_attention"
            FLASH_ATTN_AVAILABLE = True
            print("✓ FlashAttention available (2-4x speedup)")
        except ImportError:
            print("⚠️  FlashAttention not available, using standard Transformer")
            DL_BACKEND = "transformer"
            FLASH_ATTN_AVAILABLE = False
            flash_attn_qkvpacked_func = None  # Define as None for safe checking

except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not installed")
    print("   Install with: pip install torch")


class MambaSequenceEncoder(nn.Module):
    """Mamba-based sequence encoder (O(n) complexity, memory efficient)."""

    def __init__(self, input_dim=5, d_model=64, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Mamba blocks
        self.mamba1 = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba2 = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            features: (batch, d_model)
        """
        # Input projection
        x = self.input_proj(x)  # (B, L, d_model)

        # Mamba block 1
        x = x + self.mamba1(self.norm1(x))

        # Mamba block 2
        x = x + self.mamba2(self.norm2(x))

        # Global average pooling
        features = x.mean(dim=1)  # (B, d_model)

        return features


class FlashAttentionEncoder(nn.Module):
    """Flash Attention-based Transformer encoder (2-4x speedup)."""

    def __init__(self, input_dim=5, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learned)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.01)

        # QKV projections
        self.qkv_proj = nn.Linear(d_model, d_model * 3)

        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.num_layers = num_layers
        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            features: (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # (B, L, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Multiple transformer layers
        for _ in range(self.num_layers):
            # Self-attention with Flash Attention
            residual = x
            x = self.norm1(x)

            # QKV projection
            qkv = self.qkv_proj(x)  # (B, L, 3*d_model)
            qkv = qkv.reshape(
                batch_size, seq_len, 3, self.nhead, self.d_model // self.nhead
            )
            qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, 3, nhead, L, head_dim)

            # Flash attention expects (B, L, 3, nhead, head_dim)
            qkv = qkv.permute(0, 3, 1, 2, 4)  # (B, L, 3, nhead, head_dim)

            # Use flash attention if available, otherwise use standard attention
            use_flash = False
            if FLASH_ATTN_AVAILABLE and flash_attn_qkvpacked_func is not None:
                try:
                    attn_out = flash_attn_qkvpacked_func(qkv.half())  # FP16
                    attn_out = attn_out.float()  # Back to FP32
                    use_flash = True
                except Exception:
                    pass

            if not use_flash:
                # Fallback to standard attention
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                q = q.reshape(batch_size, seq_len, self.d_model)
                k = k.reshape(batch_size, seq_len, self.d_model)
                v = v.reshape(batch_size, seq_len, self.d_model)

                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (
                    self.d_model**0.5
                )
                attn_weights = torch.softmax(attn_weights, dim=-1)
                attn_out = torch.matmul(attn_weights, v)

            x = residual + attn_out

            # Feed forward
            residual = x
            x = self.norm2(x)
            x = residual + self.ffn(x)

        # Global average pooling
        features = x.mean(dim=1)  # (B, d_model)

        return features


class StandardTransformerEncoder(nn.Module):
    """Standard Transformer encoder (fallback)."""

    def __init__(self, input_dim=5, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = nn.Embedding(1000, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            features: (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        )
        x = x + self.pos_encoder(positions)

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        features = x.mean(dim=1)

        return features


class DeepLearningSequenceExtractor:
    """Extract sequence features using deep learning models."""

    def __init__(
        self,
        backend: Literal["mamba", "flash_attention", "transformer", "auto"] = "auto",
        seq_length: int = 120,
        d_model: int = 64,
        use_fp16: bool = True,
        device: Optional[str] = None,
        normalization_method: Literal[
            "global", "rolling", "ema", "adaptive"
        ] = "adaptive",
    ):
        """
        Args:
            backend: 'mamba', 'flash_attention', 'transformer', or 'auto'
            seq_length: Sequence length (default: 120 bars = 10 hours for 5min)
            d_model: Model dimension (default: 64)
            use_fp16: Use FP16 mixed precision (default: True)
            device: 'cuda', 'cpu', or None (auto-detect)
            normalization_method: Normalization strategy ('global', 'rolling', 'ema', 'adaptive')
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install with: pip install torch")

        self.seq_length = seq_length
        self.d_model = d_model
        self.use_fp16 = use_fp16
        self.normalization_method = normalization_method

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Select backend
        if backend == "auto":
            backend = DL_BACKEND or "transformer"

        self.backend = backend
        self.model = None
        self.is_fitted = False

        # Normalization parameters
        self.scaler_mean = None
        self.scaler_std = None
        self.ema_mean = None
        self.ema_var = None
        self.alpha = 0.01  # EMA smoothing parameter

        print(f"\n🔷 DeepLearningSequenceExtractor")
        print(f"   Backend: {self.backend}")
        print(f"   Device: {self.device}")
        print(f"   Sequence length: {seq_length}")
        print(f"   Output dimension: {d_model}")
        print(f"   FP16: {use_fp16}")
        print(f"   Normalization: {normalization_method}")

    def _create_model(self, input_dim: int):
        """Create the appropriate model based on backend."""
        if self.backend == "mamba":
            model = MambaSequenceEncoder(
                input_dim=input_dim,
                d_model=self.d_model,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        elif self.backend == "flash_attention":
            model = FlashAttentionEncoder(
                input_dim=input_dim,
                d_model=self.d_model,
                nhead=8,
                num_layers=2,
                dropout=0.1,
            )
        else:  # transformer
            model = StandardTransformerEncoder(
                input_dim=input_dim,
                d_model=self.d_model,
                nhead=8,
                num_layers=2,
                dropout=0.1,
            )

        model = model.to(self.device)
        model.eval()  # Inference mode

        # Use FP16 if requested and on GPU
        if self.use_fp16 and self.device.type == "cuda":
            model = model.half()
            print(f"   ✓ Model converted to FP16")

        return model

    def _prepare_sequences(self, df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """Prepare sliding window sequences with improved normalization."""
        data = df[columns].values.astype(np.float32)

        # Apply different normalization strategies
        if self.normalization_method == "global":
            # Global normalization (original method)
            if self.scaler_mean is None:
                self.scaler_mean = np.mean(data, axis=0, keepdims=True)
                self.scaler_std = np.std(data, axis=0, keepdims=True) + 1e-8
                print(f"   ✓ Fitted global scaler on {len(data)} samples")

            normalized_data = (data - self.scaler_mean) / self.scaler_std

        elif self.normalization_method == "rolling":
            # Rolling window normalization
            normalized_data = np.zeros_like(data)

            for i in range(len(data) - self.seq_length + 1):
                window_data = data[i : i + self.seq_length]
                window_mean = np.mean(window_data, axis=0, keepdims=True)
                window_std = np.std(window_data, axis=0, keepdims=True) + 1e-8

                normalized_data[i : i + self.seq_length] = (
                    window_data - window_mean
                ) / window_std

            print(f"   ✓ Applied rolling window normalization")

        elif self.normalization_method == "ema":
            # Exponential moving average normalization
            normalized_data = np.zeros_like(data)

            if self.ema_mean is None:
                self.ema_mean = np.mean(data[: self.seq_length], axis=0, keepdims=True)
                self.ema_var = np.var(data[: self.seq_length], axis=0, keepdims=True)
                print(f"   ✓ Initialized EMA normalization")

            for i in range(len(data)):
                # Update EMA statistics
                self.ema_mean = (
                    self.alpha * data[i : i + 1] + (1 - self.alpha) * self.ema_mean
                )
                self.ema_var = (
                    self.alpha * (data[i : i + 1] - self.ema_mean) ** 2
                    + (1 - self.alpha) * self.ema_var
                )
                ema_std = np.sqrt(self.ema_var) + 1e-8

                # Normalize
                normalized_data[i] = (data[i] - self.ema_mean) / ema_std

            print(f"   ✓ Applied EMA normalization")

        elif self.normalization_method == "adaptive":
            # Adaptive normalization: combine global and rolling window
            if self.scaler_mean is None:
                self.scaler_mean = np.mean(data, axis=0, keepdims=True)
                self.scaler_std = np.std(data, axis=0, keepdims=True) + 1e-8
                print(f"   ✓ Fitted adaptive scaler on {len(data)} samples")

            normalized_data = np.zeros_like(data)

            for i in range(len(data) - self.seq_length + 1):
                window_data = data[i : i + self.seq_length]
                window_mean = np.mean(window_data, axis=0, keepdims=True)
                window_std = np.std(window_data, axis=0, keepdims=True) + 1e-8

                # Calculate weights for global and local statistics
                global_weight = 0.3  # Global statistics weight
                local_weight = 0.7  # Local statistics weight

                combined_mean = (
                    global_weight * self.scaler_mean + local_weight * window_mean
                )
                combined_std = (
                    global_weight * self.scaler_std + local_weight * window_std
                )

                normalized_data[i : i + self.seq_length] = (
                    window_data - combined_mean
                ) / combined_std

            print(f"   ✓ Applied adaptive normalization")

        # Create sliding windows
        num_samples = len(normalized_data) - self.seq_length + 1
        if num_samples <= 0:
            raise ValueError(
                f"Data too short ({len(data)}) for seq_length={self.seq_length}"
            )

        sequences = []
        for i in range(num_samples):
            seq = normalized_data[i : i + self.seq_length]
            sequences.append(seq)

        return np.array(sequences, dtype=np.float32)

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """Initialize model and fit scaler."""
        if feature_columns is None:
            feature_columns = ["open", "high", "low", "close", "volume"]

        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.feature_columns = feature_columns
        input_dim = len(feature_columns)

        # Create model
        self.model = self._create_model(input_dim)

        # Fit scaler
        _ = self._prepare_sequences(df, feature_columns)

        self.is_fitted = True
        print(f"   ✓ Model initialized: {input_dim} input -> {self.d_model} output")

        return self

    def transform(self, df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
        """Extract sequence features."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Prepare sequences
        sequences = self._prepare_sequences(df, self.feature_columns)
        num_samples = len(sequences)

        # Extract features in batches
        all_features = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = sequences[i : i + batch_size]

                # Convert to tensor
                if self.use_fp16 and self.device.type == "cuda":
                    batch_tensor = torch.from_numpy(batch).half().to(self.device)
                else:
                    batch_tensor = torch.from_numpy(batch).to(self.device)

                # Extract features
                features = self.model(batch_tensor)

                # Convert back to numpy
                if self.use_fp16:
                    features = features.float()
                all_features.append(features.cpu().numpy())

        # Concatenate all batches
        dl_features = np.vstack(all_features)

        print(f"   ✓ Extracted {len(dl_features)} sequence features")

        return dl_features

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(df, feature_columns)
        features = self.transform(df, batch_size)

        # Valid indices
        valid_indices = np.arange(self.seq_length - 1, len(df))

        return features, valid_indices

    def add_to_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        batch_size: int = 64,
        prefix: str = "dl_seq",
    ) -> pd.DataFrame:
        """Add deep learning sequence features to dataframe."""
        try:
            # Extract features
            if not self.is_fitted:
                features, valid_indices = self.fit_transform(
                    df, feature_columns, batch_size
                )
            else:
                features = self.transform(df, batch_size)
                valid_indices = np.arange(self.seq_length - 1, len(df))

            # Create feature names
            feature_names = [f"{prefix}_f{i}" for i in range(self.d_model)]

            # Create DataFrame
            features_df = pd.DataFrame(
                features, columns=feature_names, index=df.index[valid_indices]
            )

            # Merge
            df_with_features = df.join(features_df, how="left")

            print(f"   ✓ Added {len(feature_names)} DL sequence features")
            print(f"   ✓ Valid samples: {len(valid_indices)} / {len(df)}")

            return df_with_features

        except Exception as e:
            print(f"   ⚠️  DL feature extraction failed: {e}")
            print(f"   ⚠️  Returning original dataframe")
            return df


def add_dl_sequence_features(
    df: pd.DataFrame,
    backend: str = "auto",
    seq_length: int = 120,
    d_model: int = 64,
    feature_columns: Optional[List[str]] = None,
    use_fp16: bool = True,
    normalization_method: str = "adaptive",
) -> pd.DataFrame:
    """Convenience function to add DL sequence features.

    Args:
        df: DataFrame with OHLCV data
        backend: 'mamba', 'flash_attention', 'transformer', or 'auto'
        seq_length: Sequence length (default: 120 = 10 hours for 5min bars)
        d_model: Output dimension (default: 64)
        feature_columns: Input columns (default: OHLCV)
        use_fp16: Use FP16 mixed precision (default: True)
        normalization_method: Normalization strategy ('global', 'rolling', 'ema', 'adaptive')

    Returns:
        DataFrame with DL sequence features added
    """
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available, skipping DL features")
        return df

    print(f"\n🔷 Extracting Deep Learning Sequence Features...")

    extractor = DeepLearningSequenceExtractor(
        backend=backend,
        seq_length=seq_length,
        d_model=d_model,
        use_fp16=use_fp16,
        normalization_method=normalization_method,
    )

    df_with_features = extractor.add_to_dataframe(df, feature_columns)

    return df_with_features


# For backward compatibility
add_transformer_features = add_dl_sequence_features
