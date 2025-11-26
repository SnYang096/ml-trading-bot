"""
SR 反转策略标签：二元标签（≥2R 成功率）

标签定义：
在 SR 区入场后，动态检查未来是否先触达 +2R 止盈 而非 -1R 止损？
（hold_bars 只是寻找上限，实际可能在更早的 K 线就满足条件）

R = 1×ATR
"""

from __future__ import annotations

from collections import defaultdict
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import compute_rr_label


@dataclass
class SRSignalConfig:
    """
    Configuration for auto-generating SR reversal signals.

    Attributes:
        min_sr_strength: Minimum SR strength score (sr_strength_max) required.
        min_support_score: Minimum SQS score for support zones.
        min_resistance_score: Minimum SQS score for resistance zones.
        tolerance_mult: Multiplier applied to ATR to determine the SR zone tolerance band.
        min_tolerance_pct: Minimum tolerance expressed as % of price to avoid zero bands when ATR is tiny.
        zone_candidates: Ordered list of columns to use as SR zone price proxies.
    """

    min_sr_strength: float = 0.0
    min_support_score: float = 0.0
    min_resistance_score: float = 0.0
    tolerance_mult: float = 1.2
    min_tolerance_pct: float = 0.003
    require_first_touch: bool = False
    max_zone_touches: Optional[int] = None
    zone_price_precision: int = 2
    priority_mode: str = "strength"  # or "distance"
    distance_epsilon: float = 1e-6
    zone_candidates: Sequence[str] = (
        "sr_zone_price",
        "nearest_sr",
        "vpvr_pvp",
        "wpt_price_reconstructed",
    )


def _ensure_atr(
    df: pd.DataFrame,
    atr_col: str,
    price_col: str,
    high_col: str,
    low_col: str,
    atr_window: int,
) -> pd.Series:
    """Return an ATR series, computing it if missing."""
    if atr_col in df.columns:
        atr_series = df[atr_col].copy()
    else:
        if not {price_col, high_col, low_col}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)

        try:
            import talib

            atr_values = talib.ATR(
                df[high_col].values,
                df[low_col].values,
                df[price_col].values,
                timeperiod=atr_window,
            )
            atr_series = pd.Series(atr_values, index=df.index)
        except ImportError:
            tr = np.maximum(
                df[high_col] - df[low_col],
                np.maximum(
                    (df[high_col] - df[price_col].shift(1)).abs(),
                    (df[low_col] - df[price_col].shift(1)).abs(),
                ),
            )
            atr_series = tr.rolling(window=atr_window, min_periods=1).mean()

    return atr_series.ffill()


def _infer_zone_price(
    df: pd.DataFrame,
    price_col: str,
    candidates: Iterable[str],
) -> pd.Series:
    """Pick the first available SR zone proxy column; fall back to price."""
    for col in candidates:
        if col in df.columns:
            series = df[col].copy()
            if series.notna().any():
                return series
    return df[price_col].copy()


def _generate_sr_reversal_signals(
    df: pd.DataFrame,
    price_col: str,
    high_col: str,
    low_col: str,
    atr_series: pd.Series,
    cfg: SRSignalConfig,
    sr_strength_col: str = "sr_strength_max",
    support_score_col: str = "sqs_hal_low",
    resistance_score_col: str = "sqs_hal_high",
) -> pd.Series:
    """Heuristically derive SR reversal signals from SR strength / proximity."""
    price = df[price_col]
    high = df[high_col]
    low = df[low_col]
    zone_price = _infer_zone_price(df, price_col, cfg.zone_candidates)

    sr_strength = df.get(sr_strength_col, pd.Series(0.0, index=df.index)).fillna(0.0)
    support_score = df.get(support_score_col, pd.Series(0.0, index=df.index)).fillna(
        0.0
    )
    resistance_score = df.get(
        resistance_score_col, pd.Series(0.0, index=df.index)
    ).fillna(0.0)

    atr = atr_series.ffill()
    tolerance = atr * cfg.tolerance_mult
    tolerance = tolerance.clip(lower=price.abs() * cfg.min_tolerance_pct).fillna(
        price.abs() * cfg.min_tolerance_pct
    )

    # Price must test the SR zone (wick enters band) and then be rejected:
    # - 多头：下影线扎进支撑带，收盘重新站回支撑价之上 → “向下被拒绝，向上走”
    # - 空头：上影线扎进阻力带，收盘重新回到阻力价之下 → “向上被拒绝，向下走”
    long_touch = (low <= zone_price + tolerance) & (price >= zone_price - tolerance)
    short_touch = (high >= zone_price - tolerance) & (price <= zone_price + tolerance)

    # 多头 SR 反转：价格测试支撑区，并在支撑附近“下探失败后向上收盘”
    long_mask = (
        (sr_strength >= cfg.min_sr_strength)
        & (support_score >= cfg.min_support_score)
        & long_touch
        & (price >= zone_price)  # Close above zone → 支撑生效，向上反转
    )

    # 空头 SR 反转：价格测试阻力区，并在阻力附近“上冲失败后向下收盘”
    short_mask = (
        (sr_strength >= cfg.min_sr_strength)
        & (resistance_score >= cfg.min_resistance_score)
        & short_touch
        & (price <= zone_price)  # Close below zone → 阻力生效，向下反转
    )

    signals_arr = np.zeros(len(df), dtype=float)
    zone_price_vals = zone_price.values
    tolerance_vals = tolerance.values
    price_vals = price.values
    sr_vals = sr_strength.values
    support_vals = support_score.values
    resistance_vals = resistance_score.values

    candidate_entries = []

    def _priority(direction: int, idx: int) -> float:
        if cfg.priority_mode == "distance":
            dist = abs(price_vals[idx] - zone_price_vals[idx])
            denom = dist + max(cfg.distance_epsilon, tolerance_vals[idx])
            return 1.0 / denom
        # strength mode
        base = sr_vals[idx]
        if direction > 0:
            base += support_vals[idx]
        else:
            base += resistance_vals[idx]
        return float(base)

    for direction, mask in ((1, long_mask), (-1, short_mask)):
        mask_indices = np.flatnonzero(mask.to_numpy())
        for idx in mask_indices:
            zone_val = zone_price_vals[idx]
            if np.isnan(zone_val):
                continue
            zone_key = round(float(zone_val), cfg.zone_price_precision)
            priority_val = _priority(direction, idx)
            candidate_entries.append((idx, direction, priority_val, zone_key))

    if not candidate_entries:
        return pd.Series(signals_arr, index=df.index, dtype=float)

    candidate_entries.sort(key=lambda item: (item[0], -item[2]))
    touch_counts: defaultdict[tuple[int, float], int] = defaultdict(int)
    limit = (
        cfg.max_zone_touches
        if cfg.max_zone_touches is not None
        else (1 if cfg.require_first_touch else None)
    )

    for idx, direction, _priority_val, zone_key in candidate_entries:
        key = (direction, zone_key)
        if limit is not None and touch_counts[key] >= limit:
            continue
        if signals_arr[idx] != 0:
            continue
        signals_arr[idx] = float(direction)
        touch_counts[key] += 1

    return pd.Series(signals_arr, index=df.index, dtype=float)


def compute_sr_reversal_label(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    rr_ratio: float = 2.0,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    auto_generate_signals: bool = True,
    sr_signal_cfg: Optional[SRSignalConfig] = None,
) -> pd.Series:
    """
    SR 反转策略标签：在 SR 区入场后，50 根 K 内是否先触达 +2R 而非 -1R。

    改进点：
    - 当缺少信号列或信号全为 0 时，自动从 SR 强度 + SQS + 价格贴近度中推导信号。
    - 统一使用 label_utils.compute_rr_label 以复用经过验证的 R/R 扫描逻辑。

    Args:
        df: Input DataFrame (OHLCV + SR 特征).
        signal_col: Existing signal column name (if present).
        price_col/high_col/low_col: Price columns.
        atr_col: ATR column; will be computed if missing.
        atr_window: ATR lookback used when计算.
        max_holding_bars: Upper bound of holding horizon (默认 50 根).
        rr_ratio: Target R/R (information only; keep for backward compat).
        stop_loss_r/take_profit_r: SL/TP multiples in ATR units.
        auto_generate_signals: Whether to derive SR signals when缺失/全0.
        sr_signal_cfg: Optional SRSignalConfig for fine-tuning auto-signal heuristics.

    Returns:
        pd.Series of binary labels (1=成功 ≥2R, 0=失败, NaN=无交易/超时).
    """

    work_df = df.copy()
    if sr_signal_cfg is None:
        sr_signal_cfg = SRSignalConfig()
    elif isinstance(sr_signal_cfg, dict):
        sr_signal_cfg = SRSignalConfig(**sr_signal_cfg)
    sr_signal_cfg = _apply_env_overrides(sr_signal_cfg)

    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    missing_signal = signal_col not in work_df.columns
    existing_signal = work_df[signal_col] if not missing_signal else None
    needs_auto_signal = (
        missing_signal
        or not existing_signal.dropna().abs().gt(0).any()
        or (existing_signal is None)
    )

    if auto_generate_signals and needs_auto_signal:
        auto_signals = _generate_sr_reversal_signals(
            work_df,
            price_col=price_col,
            high_col=high_col,
            low_col=low_col,
            atr_series=atr_series,
            cfg=sr_signal_cfg,
        )
        work_df[signal_col] = auto_signals
    elif missing_signal:
        # Cannot proceed without either existing or auto-generated signals.
        return pd.Series(np.nan, index=work_df.index)

    labels = compute_rr_label(
        work_df,
        signal_col=signal_col,
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=rr_ratio,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
    )

    # compute_rr_label sets timeout samples to 0.0. For SR reversal我们将其设为 NaN 以忽略。
    timeout_mask = labels.isna()
    return labels.where(~timeout_mask, np.nan)


def _apply_env_overrides(cfg: SRSignalConfig) -> SRSignalConfig:
    """Override SR signal config using environment variables for Optuna/CLI sweeps."""

    ENV_MAP: dict[str, tuple[str, Union[type, callable]]] = {
        "SR_SIGNAL_MIN_STRENGTH": ("min_sr_strength", float),
        "SR_SIGNAL_MIN_SUPPORT": ("min_support_score", float),
        "SR_SIGNAL_MIN_RESISTANCE": ("min_resistance_score", float),
        "SR_SIGNAL_TOLERANCE_MULT": ("tolerance_mult", float),
        "SR_SIGNAL_MIN_TOLERANCE_PCT": ("min_tolerance_pct", float),
        "SR_SIGNAL_REQUIRE_FIRST_TOUCH": (
            "require_first_touch",
            lambda v: v.lower() in {"1", "true", "yes"},
        ),
        "SR_SIGNAL_MAX_TOUCHES": (
            "max_zone_touches",
            lambda v: None if v.lower() in {"none", "-1"} else int(v),
        ),
        "SR_SIGNAL_ZONE_PRECISION": ("zone_price_precision", int),
    }

    for env_var, (field, caster) in ENV_MAP.items():
        raw = os.getenv(env_var)
        if raw is None:
            continue
        try:
            value = caster(raw)
        except Exception:
            continue
        setattr(cfg, field, value)

    return cfg
