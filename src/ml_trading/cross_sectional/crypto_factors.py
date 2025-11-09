from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CryptoCSFactorConfig:
    """
    Configuration for crypto-specific cross-sectional factor augmentation.

    Attributes:
        return_windows: Rolling windows (in bars) for relative momentum factors.
        volatility_windows: Rolling windows for realised volatility ratios.
        include_volume: Whether to add turnover/volume share metrics.
        include_orderflow: Whether to use order flow columns (e.g. taker_buy_ratio, cvd).
        clip_share: Minimum denominator to avoid division issues when computing shares.
    """

    return_windows: Sequence[int] = (12, 36, 72)  # e.g. 1h, 3h, 6h on 5m bars
    volatility_windows: Sequence[int] = (36, 72)
    include_volume: bool = True
    include_orderflow: bool = True
    clip_share: float = 1e-9


CRYPTO_FACTOR_PREFIX = "cs_crypto_"


def add_crypto_cross_sectional_factors(
    panel: pd.DataFrame,
    config: CryptoCSFactorConfig | None = None,
) -> pd.DataFrame:
    """
    Enrich a cross-sectional panel with crypto-focused factors that rely on
    multi-asset information (relative momentum, dominance, liquidity, order-flow).

    Args:
        panel: MultiIndex DataFrame indexed by (timestamp, symbol).
        config: Factor configuration. Uses defaults tuned for crypto if None.

    Returns:
        DataFrame with additional columns appended.
    """
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("Expected panel with MultiIndex (timestamp, symbol).")

    cfg = config or CryptoCSFactorConfig()
    df = panel.copy()

    # Ensure timestamp level is position 0 for groupby convenience
    time_level = df.index.names.index(
        "timestamp") if "timestamp" in df.index.names else 0
    symbol_level = 1 if time_level == 0 else 0

    if "close" in df.columns:
        _add_relative_returns(df, time_level, symbol_level, cfg.return_windows)
        _add_relative_volatility(df, time_level, symbol_level,
                                 cfg.volatility_windows)
        _add_dominance_metrics(df, time_level, symbol_level)

    if cfg.include_volume and "volume" in df.columns:
        _add_volume_share(df, time_level, cfg.clip_share)
        if "close" in df.columns:
            _add_turnover_ratio(df, time_level, symbol_level, cfg.clip_share)

    if cfg.include_orderflow:
        _add_orderflow_spreads(df, time_level, symbol_level)

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_relative_returns(
    df: pd.DataFrame,
    time_level: int,
    symbol_level: int,
    windows: Iterable[int],
) -> None:
    close = df["close"].astype(float)
    grouped_symbol = close.groupby(level=symbol_level)
    for window in windows:
        if window <= 0:
            continue
        returns = grouped_symbol.pct_change(periods=window)
        name_base = f"{CRYPTO_FACTOR_PREFIX}return_rel_{window}"
        mean_cs = returns.groupby(level=time_level).transform("mean")
        std_cs = returns.groupby(level=time_level).transform("std")
        df[f"{name_base}_demean"] = returns - mean_cs
        df[f"{name_base}_zscore"] = np.divide(
            returns - mean_cs,
            std_cs.replace(0, np.nan),
        ).fillna(0.0)
        df[f"{name_base}_rank"] = returns.groupby(level=time_level).rank(
            pct=True)


def _add_relative_volatility(
    df: pd.DataFrame,
    time_level: int,
    symbol_level: int,
    windows: Iterable[int],
) -> None:
    close = df["close"].astype(float)
    multi_asset = df.index.get_level_values(time_level).nunique() > 1
    for window in windows:
        if window <= 1:
            continue
        if multi_asset:
            log_returns = close.groupby(level=symbol_level).apply(lambda s: np.log(s).diff())
            if isinstance(log_returns.index, pd.MultiIndex):
                log_returns = log_returns.droplevel(0)
            vol = log_returns.groupby(level=symbol_level).apply(
                lambda s: s.rolling(window=window, min_periods=window // 2).std()
            )
            if isinstance(vol.index, pd.MultiIndex):
                vol = vol.droplevel(0)
        else:
            symbol_series = close.droplevel(symbol_level) if symbol_level < close.index.nlevels else close
            log_returns = np.log(symbol_series).diff()
            vol = log_returns.rolling(window=window, min_periods=window // 2).std()
        vol_mean_cs = vol.groupby(level=time_level).transform("mean")
        vol_median_cs = vol.groupby(level=time_level).transform("median")
        prefix = f"{CRYPTO_FACTOR_PREFIX}vol_rel_{window}"
        df[prefix + "_demean"] = vol - vol_mean_cs
        df[prefix + "_ratio"] = np.divide(
            vol,
            vol_median_cs.replace(0, np.nan),
        ).fillna(0.0)
        df[prefix + "_rank"] = vol.groupby(level=time_level).rank(pct=True)


def _add_dominance_metrics(
    df: pd.DataFrame,
    time_level: int,
    symbol_level: int,
) -> None:
    # Log return dominance vs strongest asset in cross-section
    close = df["close"].astype(float)
    log_return = close.groupby(
        level=symbol_level).apply(lambda s: np.log(s).diff())
    if isinstance(log_return.index, pd.MultiIndex):
        log_return = log_return.droplevel(0)
    max_ret = log_return.groupby(level=time_level).transform("max")
    min_ret = log_return.groupby(level=time_level).transform("min")
    range_ret = (max_ret - min_ret).replace(0, np.nan)
    df[f"{CRYPTO_FACTOR_PREFIX}return_dominance"] = (log_return -
                                                     min_ret) / range_ret
    df[f"{CRYPTO_FACTOR_PREFIX}return_dominance"] = df[
        f"{CRYPTO_FACTOR_PREFIX}return_dominance"].clip(0.0, 1.0).fillna(0.0)


def _add_volume_share(
    df: pd.DataFrame,
    time_level: int,
    clip_share: float,
) -> None:
    vol = df["volume"].astype(float)
    total_vol = vol.groupby(level=time_level).transform("sum")
    denom = total_vol.where(total_vol.abs() > clip_share, np.nan)
    share = np.divide(vol, denom).fillna(0.0)
    df[f"{CRYPTO_FACTOR_PREFIX}volume_share"] = share
    df[f"{CRYPTO_FACTOR_PREFIX}volume_rank"] = vol.groupby(
        level=time_level).rank(pct=True)


def _add_turnover_ratio(
    df: pd.DataFrame,
    time_level: int,
    symbol_level: int,
    clip_share: float,
) -> None:
    turnover = df["close"].astype(float) * df["volume"].astype(float)
    trailing = (turnover.groupby(level=symbol_level).rolling(
        window=48, min_periods=12).mean().droplevel(0))
    ratio = np.divide(
        turnover,
        trailing.where(trailing.abs() > clip_share, np.nan),
    ).fillna(0.0)
    df[f"{CRYPTO_FACTOR_PREFIX}turnover_ratio"] = ratio
    df[f"{CRYPTO_FACTOR_PREFIX}turnover_rank"] = turnover.groupby(
        level=time_level).rank(pct=True)


def _add_orderflow_spreads(
    df: pd.DataFrame,
    time_level: int,
    symbol_level: int,
) -> None:
    if "taker_buy_ratio" in df.columns:
        ratio = df["taker_buy_ratio"].astype(float)
        mean_cs = ratio.groupby(level=time_level).transform("mean")
        df[f"{CRYPTO_FACTOR_PREFIX}taker_buy_spread"] = ratio - mean_cs
        df[f"{CRYPTO_FACTOR_PREFIX}taker_buy_rank"] = ratio.groupby(
            level=time_level).rank(pct=True)
    if "cvd" in df.columns:
        cvd = df["cvd"].astype(float)
        cvd_change = cvd.groupby(level=symbol_level).diff()
        cvd_mean = cvd_change.groupby(level=time_level).transform("mean")
        df[f"{CRYPTO_FACTOR_PREFIX}cvd_spread"] = (cvd_change -
                                                   cvd_mean).fillna(0.0)
        df[f"{CRYPTO_FACTOR_PREFIX}cvd_rank"] = cvd_change.groupby(
            level=time_level).rank(pct=True).fillna(0.0)
