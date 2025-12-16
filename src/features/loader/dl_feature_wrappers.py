"""
Deep learning sequence feature wrappers for config-driven pipeline.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.features.time_series.dl_sequence_features import (
    add_dl_sequence_features,
)


def compute_dl_sequence_features(
    df: pd.DataFrame,
    backend: str = "auto",
    seq_length: int = 120,
    d_model: int = 64,
    feature_columns: Optional[List[str]] = None,
    use_fp16: bool = False,
    prefix: str = "dl_seq",
    device: Optional[str] = None,  # 'cuda', 'cpu', or None (auto-detect)
) -> pd.DataFrame:
    """
    Compute leak-free deep learning sequence embeddings (Mamba/Transformer).

    Args:
        df: Input dataframe.
        backend: 'mamba', 'flash_attention', 'transformer', or 'auto'.
        seq_length: Sliding window length.
        d_model: Output embedding dimension.
        feature_columns: Columns used as model inputs (defaults to OHLCV).
        use_fp16: Whether to enable FP16 during inference (GPU only).
        prefix: Feature column prefix (default: dl_seq).

    Returns:
        DataFrame with DL sequence features appended.
    """

    # add_dl_sequence_features already copies df internally; still copy for safety
    result = add_dl_sequence_features(
        df.copy(),
        backend=backend,
        seq_length=seq_length,
        d_model=d_model,
        feature_columns=feature_columns,
        use_fp16=use_fp16,
        device=device,
    )

    # Ensure expected columns exist even if DL backend is unavailable
    expected_cols = [f"{prefix}_f{i}" for i in range(d_model)]
    for col in expected_cols:
        if col not in result.columns:
            result[col] = 0.0

    return result


def compute_dl_sequence_features_from_series(
    *,
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    backend: str = "auto",
    seq_length: int = 120,
    d_model: int = 64,
    feature_columns: Optional[List[str]] = None,
    use_fp16: bool = False,
    prefix: str = "dl_seq",
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Narrow-IO entrypoint for DL sequence embeddings (Series-in, DataFrame-out).
    Keeps pipeline from passing a wide DF into heavy DL code.
    """
    df = pd.DataFrame(
        {"open": open, "high": high, "low": low, "close": close, "volume": volume}
    )
    return compute_dl_sequence_features(
        df,
        backend=backend,
        seq_length=seq_length,
        d_model=d_model,
        feature_columns=feature_columns,
        use_fp16=use_fp16,
        prefix=prefix,
        device=device,
    )
