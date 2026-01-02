"""
SR 反转策略标签：二元标签（≥2R 成功率）

标签定义：
在 SR 区入场后，动态检查未来是否先触达 +2R 止盈 而非 -1R 止损？
（hold_bars 只是寻找上限，实际可能在更早的 K 线就满足条件）

R = 1×ATR
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import compute_rr_label


@dataclass
class SRSignalConfig:
    """
    Configuration for auto-generating SR reversal signals.

    Note: This config is primarily used for diagnostic/optimization scripts.
    Label generation now uses `compute_sr_reversal_label_full_scan` which does NOT
    pre-filter signals - the model learns to filter based on features.

    Thresholds default to 0.0 (no filtering) to allow full signal generation.
    Model will learn which signals to use based on features, not pre-filtered labels.

    Attributes:
        min_sr_strength: Minimum SR strength score (sr_strength_max) required.
            Default 0.0 = no filtering (model learns from features).
        min_support_score: Minimum SQS score for support zones.
            Default 0.0 = no filtering (model learns from features).
        min_resistance_score: Minimum SQS score for resistance zones.
            Default 0.0 = no filtering (model learns from features).
        tolerance_mult: Multiplier applied to ATR to determine the SR zone tolerance band.
        min_tolerance_pct: Minimum tolerance expressed as % of price to avoid zero bands when ATR is tiny.
        zone_candidates: Ordered list of columns to use as SR zone price proxies.
        use_vpin_filter: Whether to use VPIN to filter signals (default: False for backward compatibility).
        min_vpin: Minimum VPIN value for long signals (default: None, no filter).
        max_vpin: Maximum VPIN value for short signals (default: None, no filter).
        vpin_col: VPIN column name (default: "vpin").
    """

    min_sr_strength: float = (
        0.0  # Default 0.0 = no filtering, model learns from features
    )
    min_support_score: float = (
        0.0  # Default 0.0 = no filtering, model learns from features
    )
    min_resistance_score: float = (
        0.0  # Default 0.0 = no filtering, model learns from features
    )
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
    # VPIN 过滤参数
    use_vpin_filter: bool = False
    min_vpin: Optional[float] = (
        None  # 多头信号的最小 VPIN（VPIN 高表示买压大，适合反转做多）
    )
    max_vpin: Optional[float] = (
        None  # 空头信号的最大 VPIN（VPIN 低表示卖压大，适合反转做空）
    )
    vpin_col: str = "vpin"


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

    # VPIN 过滤（可选）
    vpin_filter_long = pd.Series(True, index=df.index)
    vpin_filter_short = pd.Series(True, index=df.index)

    if cfg.use_vpin_filter and cfg.vpin_col in df.columns:
        vpin = df[cfg.vpin_col].fillna(0.5)  # 默认 0.5（中性）

        # 多头：VPIN 高表示买压大，适合在支撑位反转做多
        if cfg.min_vpin is not None:
            vpin_filter_long = vpin >= cfg.min_vpin

        # 空头：VPIN 低表示卖压大，适合在阻力位反转做空
        if cfg.max_vpin is not None:
            vpin_filter_short = vpin <= cfg.max_vpin

    # 多头 SR 反转：价格测试支撑区，并在支撑附近“下探失败后向上收盘”
    long_mask = (
        (sr_strength >= cfg.min_sr_strength)
        & (support_score >= cfg.min_support_score)
        & long_touch
        & (price >= zone_price)  # Close above zone → 支撑生效，向上反转
        & vpin_filter_long  # VPIN 过滤
    )

    # 空头 SR 反转：价格测试阻力区，并在阻力附近“上冲失败后向下收盘”
    short_mask = (
        (sr_strength >= cfg.min_sr_strength)
        & (resistance_score >= cfg.min_resistance_score)
        & short_touch
        & (price <= zone_price)  # Close below zone → 阻力生效，向下反转
        & vpin_filter_short  # VPIN 过滤
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


def _combine_long_short(
    long_labels: pd.Series, short_labels: pd.Series, mode: str = "any_success"
) -> pd.Series:
    """
    Combine long/short labels into a single binary series.

    mode:
        - "any_success": 1 if either long or short succeeds; 0 if both fail; NaN otherwise
        - "long_only": use long_labels
        - "short_only": use short_labels
    """
    if mode == "long_only":
        return long_labels
    if mode == "short_only":
        return short_labels

    # any_success
    success = (long_labels == 1) | (short_labels == 1)
    fail = (long_labels == 0) & (short_labels == 0)
    result = pd.Series(np.nan, index=long_labels.index, dtype=float)
    result.loc[success] = 1.0
    result.loc[fail] = 0.0
    return result


def compute_sr_reversal_label_full_scan(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    combine_mode: str = "any_success",
    # 可选：SR过滤参数
    dist_to_sr_col: Optional[str] = None,
    dist_atr_mult: Optional[float] = None,
    sr_mask_col: Optional[str] = None,
) -> pd.Series:
    """
    全量扫描标签生成：不预先过滤信号，对每根 K 线都假设入场，扫描 ±R/R。

    这是推荐的标签生成方式，因为：
    - 不依赖信号过滤规则（如 min_sr_strength, min_support_score 等）
    - 模型可以根据特征学习哪些信号应该使用
    - 保留更多训练样本，避免标签数据过少

    - 做多：entry=open[t+1]，TP=+take_profit_r*ATR，SL=-stop_loss_r*ATR
    - 做空：entry=open[t+1]，TP=-take_profit_r*ATR，SL=+stop_loss_r*ATR
    - 分别计算 long/short 的 2R 成功与否，再按 combine_mode 合并。
    """

    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # Long-only synthetic signals (+1 everywhere)
    work_df["__long_signal"] = 1.0
    long_labels = compute_rr_label(
        work_df,
        signal_col="__long_signal",
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
    )

    # Short-only synthetic signals (-1 everywhere)
    work_df["__short_signal"] = -1.0
    short_labels = compute_rr_label(
        work_df,
        signal_col="__short_signal",
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
    )

    combined = _combine_long_short(long_labels, short_labels, mode=combine_mode)
    timeout_mask = combined.isna()
    result = combined.where(~timeout_mask, np.nan)

    # 可选：SR过滤
    if dist_to_sr_col is not None and dist_atr_mult is not None:
        # 使用距离过滤
        if dist_to_sr_col in work_df.columns:
            price_series = work_df[price_col]
            dist_pct = work_df[dist_to_sr_col].abs()
            abs_distance = dist_pct * price_series
            dist_normalized = abs_distance / (atr_series + 1e-8)
            sr_mask = dist_normalized <= dist_atr_mult
            sr_mask = sr_mask.fillna(False)
            result = result.where(sr_mask)
    elif sr_mask_col is not None:
        # 使用预计算的SR mask
        if sr_mask_col in work_df.columns:
            sr_mask = work_df[sr_mask_col].fillna(False).astype(bool)
            result = result.where(sr_mask)

    return result


def compute_sr_reversal_rr_continuous_label(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    sr_mask_col: Optional[str] = None,  # 改为可选，None时不进行SR过滤
    dist_to_sr_col: Optional[str] = None,  # 改为可选，None时不进行SR过滤
    dist_atr_mult: Optional[float] = None,  # 改为可选，None时不进行SR过滤
    combine_mode: Optional[str] = None,  # 接受但不使用，用于回测逻辑的方向判断
) -> pd.Series:
    """
    连续 RR 标签（MFE/MAE 比例），不依赖预过滤信号：
    - 默认对每根 K 线假设做多信号（__long_signal=1）
    - 使用 compute_rr_label(..., use_continuous_label=True) 计算真实 RR
    - 可选：仅在 SR 附近保留标签，其余置为 NaN（通过 sr_mask_col 或 dist_to_sr_col）
    - 如果不提供 sr_mask_col 和 dist_to_sr_col，则使用全量样本（不进行SR过滤）
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # SR 掩码：如果提供了SR过滤参数，则进行过滤；否则使用全量样本
    if sr_mask_col is not None and sr_mask_col in work_df.columns:
        sr_mask = work_df[sr_mask_col].fillna(False).astype(bool)
    elif (
        dist_to_sr_col is not None
        and dist_atr_mult is not None
        and dist_to_sr_col in work_df.columns
    ):
        # 使用距离过滤（需要单位转换）
        price_series = work_df[price_col]
        dist_pct = work_df[dist_to_sr_col].abs()
        abs_distance = dist_pct * price_series
        dist_normalized = abs_distance / (atr_series + 1e-8)
        sr_mask = dist_normalized <= dist_atr_mult
        sr_mask = sr_mask.fillna(False)
    else:
        # 不进行SR过滤，使用全量样本
        sr_mask = pd.Series(True, index=work_df.index)

    # 全量假设做多信号
    work_df["__long_signal"] = 1.0

    rr_series = compute_rr_label(
        work_df,
        signal_col="__long_signal",
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=True,
    )

    rr_series = rr_series.where(sr_mask)
    rr_series.name = "rr_label"
    return rr_series


def compute_sr_reversal_rr_continuous_label_with_weights(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    sr_mask_col: str = "is_near_sr",
    dist_to_sr_col: str = "dist_to_nearest_sr",
    dist_atr_mult: float = 1.5,
    combine_mode: Optional[str] = None,  # 接受但不使用，用于回测逻辑的方向判断
    # 样本权重参数
    compute_weights: bool = False,
    weight_strategy: str = "uniform",
    weight_config: Optional[Dict] = None,
    return_weights: bool = False,
) -> pd.Series | Tuple[pd.Series, pd.Series]:
    """
    计算连续 RR 标签（回归任务），可选地计算样本权重。

    这是 `compute_sr_reversal_rr_continuous_label` 的增强版本，支持样本权重。
    适用于回归模型（如 sr_reversal_rr_reg），通过 top_quantile 选择交易信号。

    Args:
        df: DataFrame with features
        price_col: 价格列名
        high_col: 最高价列名
        low_col: 最低价列名
        atr_col: ATR 列名
        atr_window: ATR 窗口
        max_holding_bars: 最大持仓周期
        stop_loss_r: 止损 R 倍数
        take_profit_r: 止盈 R 倍数
        sr_mask_col: SR 掩码列名（如果提供，只在 SR 附近保留标签）
        dist_to_sr_col: 距离 SR 的列名（如果提供，用于计算 SR 掩码）
        dist_atr_mult: 距离 SR 的 ATR 倍数阈值
        compute_weights: 是否计算样本权重
        weight_strategy: 权重策略名称
        weight_config: 权重策略配置
        return_weights: 是否返回权重（如果 True，返回 (labels, weights) 元组）

    Returns:
        如果 return_weights=False: 返回标签 Series
        如果 return_weights=True: 返回 (labels, weights) 元组
    """
    # 计算连续 RR 标签
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # SR 掩码：优先使用显式布尔列，其次 dist_to_sr 与 ATR 比较
    if sr_mask_col in work_df.columns:
        sr_mask = work_df[sr_mask_col].fillna(False).astype(bool)
    elif dist_to_sr_col in work_df.columns:
        # dist_to_nearest_sr 是相对百分比，需要转换为绝对价格距离后再与 ATR 比较
        price_series = work_df[price_col]
        dist_pct = work_df[dist_to_sr_col].abs()
        # 将百分比距离转换为绝对价格距离
        abs_distance = dist_pct * price_series
        # 计算归一化距离（单位：ATR）
        dist_normalized = abs_distance / (atr_series + 1e-8)
        # 判断是否在SR附近
        sr_mask = dist_normalized <= dist_atr_mult
        sr_mask = sr_mask.fillna(False)
    else:
        sr_mask = pd.Series(True, index=work_df.index)

    # 全量假设做多信号
    work_df["__long_signal"] = 1.0

    rr_series = compute_rr_label(
        work_df,
        signal_col="__long_signal",
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=True,  # 使用连续标签（回归任务）
    )

    rr_series = rr_series.where(sr_mask)
    rr_series.name = "rr_label"

    # 计算样本权重（如果需要）
    if compute_weights:
        # 对于 result_based_rr 策略，可以直接使用 rr_series 作为未来 RR
        # 这样避免重复计算，提高效率
        if weight_strategy == "result_based_rr" and weight_config is not None:
            weight_config = weight_config.copy()
            if "rr_col" not in weight_config:
                # 将计算好的 RR 值存储到 DataFrame 中，供权重函数使用
                work_df["rr_label"] = rr_series
                weight_config["rr_col"] = "rr_label"  # 告诉权重函数直接使用这个列

        weights = compute_sr_reversal_sample_weights(
            work_df,
            rr_series,  # 对于回归任务，使用 RR 值本身作为"标签"（用于确定有效样本）
            weight_strategy=weight_strategy,
            weight_config=weight_config,
            atr_col=atr_col,
            atr_window=atr_window,
            price_col=price_col,
            high_col=high_col,
            low_col=low_col,
            max_holding_bars=max_holding_bars,
            stop_loss_r=stop_loss_r,
            take_profit_r=take_profit_r,
        )
        # 将权重存储到原始 DataFrame 中（如果调用者需要）
        df["sample_weight"] = weights
        if return_weights:
            return rr_series, weights
        else:
            return rr_series
    else:
        if return_weights:
            # 即使不计算权重，也返回统一权重
            weights = pd.Series(1.0, index=rr_series.index)
            return rr_series, weights
        else:
            return rr_series


def compute_sr_reversal_label(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    rr_ratio: Optional[float] = None,
    entry_offset: int = 1,
    entry_price_col: str = "open",
    auto_generate_signals: bool = True,
    sr_signal_cfg: Optional[SRSignalConfig] = None,
) -> pd.Series:
    """
    Backward-compatible SR Reversal label generator (signal-based).

    This function exists for older tests/diagnostics that define a sparse `signal`
    series (+1 long, -1 short) and want a binary RR label at the signal timestamp:
    TP-first => 1.0, SL-first => 0.0, timeout => NaN.

    New training should prefer `compute_sr_reversal_label_full_scan`, which does not
    depend on a precomputed signal and generates labels for every bar.
    """
    work_df = df.copy()

    # Backward-compat: some older callers provide rr_ratio instead of take_profit_r.
    # rr_ratio is interpreted as TP/SL ratio in R-multiples.
    if rr_ratio is not None:
        try:
            take_profit_r = float(rr_ratio) * float(stop_loss_r)
        except Exception:
            pass

    # Ensure ATR exists (required by RR simulation)
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    if auto_generate_signals:
        if {price_col, high_col, low_col}.issubset(work_df.columns):
            cfg = sr_signal_cfg if sr_signal_cfg is not None else SRSignalConfig()
            signals = _generate_sr_reversal_signals(
                work_df,
                price_col=price_col,
                high_col=high_col,
                low_col=low_col,
                atr_series=atr_series,
                cfg=cfg,
            )
            work_df[signal_col] = signals

    # Delegate to the unified RR label generator
    return compute_rr_label(
        work_df,
        signal_col=signal_col,
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=False,
        entry_price_col=entry_price_col,
        entry_offset=entry_offset,
    )


def compute_sr_reversal_sample_weights(
    df: pd.DataFrame,
    labels: pd.Series,
    weight_strategy: str = "uniform",
    weight_config: Optional[Dict] = None,
    atr_col: str = "atr",
    atr_window: int = 14,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    # 以下参数用于 result_based_rr 策略（如果未在 weight_config 中提供）
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
) -> pd.Series:
    """
    计算 SR 反转策略的样本权重，支持多种权重策略。

    权重策略（weight_strategy）：
    1. "uniform": 统一权重 1.0（默认）
    2. "sr_strength": 基于 SR 强度分级加权
    3. "triple_resonance": VPIN + CVD + SR 三重共振加权
    4. "cvd_confirmation": CVD 确认加权
    5. "distance_based": 基于距离 SR 的远近加权
    6. "result_based_rr": 基于未来 RR 的结果驱动加权（高级策略）
    7. "composite": 组合多种策略的加权

    Args:
        df: DataFrame with features and labels
        labels: Label series (用于确定有效样本)
        weight_strategy: 权重策略名称
        weight_config: 权重策略的配置字典
        atr_col: ATR 列名
        atr_window: ATR 窗口
        price_col: 价格列名
        high_col: 最高价列名
        low_col: 最低价列名

    Returns:
        Series with sample weights (same index as labels)
    """
    if weight_config is None:
        weight_config = {}

    # 初始化权重为 1.0
    weights = pd.Series(1.0, index=labels.index)

    # 只对有效标签计算权重
    valid_mask = labels.notna()
    if not valid_mask.any():
        return weights

    # 确保 ATR 存在
    atr_series = _ensure_atr(df, atr_col, price_col, high_col, low_col, atr_window)

    if weight_strategy == "uniform":
        # 统一权重，无需修改
        pass

    elif weight_strategy == "sr_strength":
        # 策略：基于 SR 强度分级加权
        # 强度越高，权重越大
        sr_strength_col = weight_config.get("sr_strength_col", "sr_strength_max")
        strength_thresholds = weight_config.get("strength_thresholds", [0.3, 0.5, 0.7])
        strength_weights = weight_config.get("strength_weights", [1.0, 2.0, 3.0, 5.0])

        if sr_strength_col in df.columns:
            sr_strength = df[sr_strength_col].fillna(0.0)
            # 分级加权
            for i, threshold in enumerate(strength_thresholds):
                mask = (sr_strength >= threshold) & valid_mask
                if i < len(strength_weights):
                    weights.loc[mask] = strength_weights[i]
            # 最高级别
            if len(strength_weights) > len(strength_thresholds):
                max_threshold = max(strength_thresholds) if strength_thresholds else 0.0
                mask = (sr_strength >= max_threshold) & valid_mask
                weights.loc[mask] = strength_weights[-1]

    elif weight_strategy == "triple_resonance":
        # 策略：VPIN + CVD + SR 三重共振
        # 当三个条件同时满足时，给予高权重
        vpin_col = weight_config.get("vpin_col", "vpin")
        vpin_threshold = weight_config.get("vpin_threshold", 0.7)
        cvd_slope_col = weight_config.get("cvd_slope_col", "cvd_slope_5_f")
        cvd_slope_threshold = weight_config.get("cvd_slope_threshold", 0.0)
        sr_strength_col = weight_config.get("sr_strength_col", "sr_strength_max")
        sr_strength_threshold = weight_config.get("sr_strength_threshold", 0.5)
        resonance_weight = weight_config.get("resonance_weight", 5.0)

        # 检查三重共振条件
        vpin_high = pd.Series(False, index=df.index)
        if vpin_col in df.columns:
            vpin_high = df[vpin_col].fillna(0.5) >= vpin_threshold

        cvd_positive = pd.Series(False, index=df.index)
        if cvd_slope_col in df.columns:
            cvd_positive = df[cvd_slope_col].fillna(0.0) > cvd_slope_threshold
        elif "cvd" in df.columns:
            # 如果没有斜率特征，用 CVD 差分
            cvd_positive = df["cvd"].diff().fillna(0.0) > 0

        sr_strong = pd.Series(False, index=df.index)
        if sr_strength_col in df.columns:
            sr_strong = df[sr_strength_col].fillna(0.0) >= sr_strength_threshold

        # 三重共振
        triple_mask = valid_mask & vpin_high & cvd_positive & sr_strong
        weights.loc[triple_mask] = resonance_weight

    elif weight_strategy == "cvd_confirmation":
        # 策略：CVD 确认加权
        # CVD 斜率与反转方向一致时，给予更高权重
        cvd_slope_col = weight_config.get("cvd_slope_col", "cvd_slope_5_f")
        cvd_weight = weight_config.get("cvd_weight", 3.0)

        if cvd_slope_col in df.columns:
            cvd_slope = df[cvd_slope_col].fillna(0.0)
            # 对于多头反转（label=1），CVD 斜率应该为正
            # 对于空头反转，CVD 斜率应该为负
            # 这里简化处理：CVD 斜率绝对值大时，给予更高权重
            cvd_strong = cvd_slope.abs() > weight_config.get("cvd_slope_threshold", 0.1)
            mask = valid_mask & cvd_strong
            weights.loc[mask] = cvd_weight
        elif "cvd" in df.columns:
            # 使用 CVD 差分
            cvd_diff = df["cvd"].diff().abs()
            cvd_strong = cvd_diff > cvd_diff.quantile(0.7)
            mask = valid_mask & cvd_strong
            weights.loc[mask] = cvd_weight

    elif weight_strategy == "distance_based":
        # 策略：基于距离 SR 的远近加权
        # 距离越近，权重越高（反转更可靠）
        dist_col = weight_config.get("dist_col", "dist_to_nearest_sr")
        dist_atr_mult = weight_config.get("dist_atr_mult", 1.5)
        near_weight = weight_config.get("near_weight", 3.0)
        far_weight = weight_config.get("far_weight", 1.0)

        if dist_col in df.columns:
            dist_to_sr = df[dist_col].fillna(np.inf)
            atr = atr_series.fillna(1.0)
            near_sr = dist_to_sr <= dist_atr_mult * atr

            # 距离越近，权重越高（线性插值）
            max_dist = dist_atr_mult * atr
            dist_ratio = (max_dist - dist_to_sr.clip(upper=max_dist)) / max_dist.clip(
                lower=1e-6
            )
            distance_weights = far_weight + (
                near_weight - far_weight
            ) * dist_ratio.clip(0, 1)

            mask = valid_mask & (dist_to_sr < np.inf)
            weights.loc[mask] = distance_weights.loc[mask]
        elif "is_near_sr" in df.columns:
            # 使用布尔列
            near_sr = df["is_near_sr"].fillna(False).astype(bool)
            mask = valid_mask & near_sr
            weights.loc[mask] = near_weight

    elif weight_strategy == "result_based_rr":
        # 策略：基于未来 RR 的结果驱动加权（Results-based Importance Resampling）
        # 核心思想：用未来 RR 反向赋能，让模型更关注"逻辑成立且利润丰厚"的样本
        # 权重公式：Logic_Score * log(1 + RR)
        # - Logic_Score: 基于 VPIN/CVD/SR 的先验逻辑分
        # - log(1 + RR): 对未来收益进行对数平滑，避免极端值主导
        # - 如果 RR < 1（亏损），权重降到最低（如 0.05）

        # 配置参数
        logic_mode = weight_config.get(
            "logic_mode", "triple_resonance"
        )  # 或 "sr_strength", "cvd_only", "none"
        min_rr_threshold = weight_config.get(
            "min_rr_threshold", 1.0
        )  # RR 低于此值视为亏损（兼容旧配置）
        # 亏损/低RR样本权重：允许为 0.0（等价于“忽略这些样本”）
        loss_weight = float(weight_config.get("loss_weight", 0.05))
        normalize_weights = weight_config.get(
            "normalize_weights", True
        )  # 是否归一化权重
        rr_col = weight_config.get("rr_col", None)  # 如果提供，直接使用此列的 RR 值
        # 防止“全部样本权重为0”导致训练失败的兜底阈值
        min_total_weight = float(weight_config.get("min_total_weight", 1e-6))

        # 1. 计算或获取未来 RR
        if rr_col is not None and rr_col in df.columns:
            # 直接使用提供的 RR 列
            future_rr = df[rr_col].fillna(0.0)
        else:
            # 需要计算未来 RR（使用连续标签模式）
            # 注意：这里需要假设信号方向，对于反转策略，我们假设做多
            work_df = df.copy()
            work_df["__long_signal"] = 1.0

            # 使用 compute_rr_label 计算连续 RR
            # 优先使用 weight_config 中的参数，否则使用函数参数
            future_rr = compute_rr_label(
                work_df,
                signal_col="__long_signal",
                price_col=price_col,
                atr_col=atr_col,
                atr_window=atr_window,
                rr_ratio=weight_config.get(
                    "rr_ratio",
                    take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
                ),
                max_holding_bars=weight_config.get(
                    "max_holding_bars", max_holding_bars
                ),
                stop_loss_r=weight_config.get("stop_loss_r", stop_loss_r),
                take_profit_r=weight_config.get("take_profit_r", take_profit_r),
                use_continuous_label=True,  # 使用连续标签获取实际 RR
                entry_price_col="open",
                entry_offset=1,
            )
            future_rr = future_rr.fillna(0.0)

        # 2. 计算逻辑分数（Logic_Score）
        logic_score = pd.Series(1.0, index=df.index)  # 默认逻辑分为 1.0

        if logic_mode == "triple_resonance":
            # 三重共振逻辑分
            vpin_col = weight_config.get("vpin_col", "vpin")
            vpin_threshold = weight_config.get("vpin_threshold", 0.7)
            cvd_slope_col = weight_config.get("cvd_slope_col", "cvd_slope_5_f")
            cvd_slope_threshold = weight_config.get("cvd_slope_threshold", 0.0)
            sr_strength_col = weight_config.get("sr_strength_col", "sr_strength_max")
            sr_strength_threshold = weight_config.get("sr_strength_threshold", 0.5)
            logic_base = weight_config.get("logic_base", 1.0)  # 基础逻辑分
            logic_boost = weight_config.get("logic_boost", 1.5)  # 三重共振时的加成

            vpin_ok = pd.Series(True, index=df.index)
            if vpin_col in df.columns:
                vpin_ok = df[vpin_col].fillna(0.5) >= vpin_threshold

            cvd_ok = pd.Series(True, index=df.index)
            if cvd_slope_col in df.columns:
                cvd_ok = df[cvd_slope_col].fillna(0.0) > cvd_slope_threshold
            elif "cvd" in df.columns:
                cvd_ok = df["cvd"].diff().fillna(0.0) > 0

            sr_ok = pd.Series(True, index=df.index)
            if sr_strength_col in df.columns:
                sr_ok = df[sr_strength_col].fillna(0.0) >= sr_strength_threshold

            # 三重共振时给予逻辑分加成
            triple_mask = vpin_ok & cvd_ok & sr_ok
            logic_score.loc[triple_mask] = logic_base * logic_boost
            logic_score.loc[~triple_mask] = logic_base

        elif logic_mode == "sr_strength":
            # 基于 SR 强度的逻辑分
            sr_strength_col = weight_config.get("sr_strength_col", "sr_strength_max")
            if sr_strength_col in df.columns:
                sr_strength = df[sr_strength_col].fillna(0.0)
                # 将 SR 强度映射到逻辑分（0.5 到 2.0）
                logic_score = 0.5 + 1.5 * sr_strength.clip(0, 1)

        elif logic_mode == "cvd_only":
            # 仅基于 CVD 的逻辑分
            cvd_slope_col = weight_config.get("cvd_slope_col", "cvd_slope_5_f")
            if cvd_slope_col in df.columns:
                cvd_slope = df[cvd_slope_col].fillna(0.0)
                # CVD 斜率绝对值越大，逻辑分越高
                logic_score = 0.5 + 1.5 * (
                    cvd_slope.abs() / (cvd_slope.abs().quantile(0.95).clip(lower=0.1))
                ).clip(0, 1)

        # 3. 计算结果权重（Result_Weight）：根据RR比率分级处理
        # 分级策略：
        # - RR >= 2.0: 高权重（使用 log(1 + RR) * high_rr_boost）
        # - 1.0 <= RR < 2.0: 中等权重（使用 log(1 + RR)）
        # - RR < 1.0: 低权重（使用 loss_weight；可配置为 0.0）
        result_weight = pd.Series(loss_weight, index=df.index)

        # 获取分级参数
        high_rr_threshold = weight_config.get("high_rr_threshold", 2.0)  # 高RR阈值
        high_rr_boost = weight_config.get(
            "high_rr_boost", 2.0
        )  # 高RR加成倍数（提高默认值，确保盈利样本权重更高）
        medium_rr_threshold = weight_config.get(
            "medium_rr_threshold", 1.0
        )  # 中等RR阈值

        # 高RR样本（>= high_rr_threshold）：高权重（盈利样本优先）
        high_rr_mask = future_rr >= high_rr_threshold
        result_weight.loc[high_rr_mask] = (
            np.log1p(future_rr.loc[high_rr_mask]) * high_rr_boost
        )

        # 中等RR样本（medium_rr_threshold <= RR < high_rr_threshold）：中等权重
        medium_rr_mask = (future_rr >= medium_rr_threshold) & (
            future_rr < high_rr_threshold
        )
        result_weight.loc[medium_rr_mask] = np.log1p(future_rr.loc[medium_rr_mask])

        # 低RR样本（< medium_rr_threshold）：低权重（已在初始化时设置）

        # 4. 合成最终权重：确保盈利样本（高RR）的权重优先
        # 策略：对于高RR样本，直接使用 Result_Weight（不乘以 Logic_Score）
        # 对于其他样本，使用 Logic_Score * Result_Weight
        final_weights = pd.Series(0.0, index=df.index)

        # 高RR样本：直接使用 Result_Weight（盈利样本优先，不受逻辑分影响）
        final_weights.loc[high_rr_mask] = result_weight.loc[high_rr_mask]

        # 其他样本：使用 Logic_Score * Result_Weight
        other_mask = ~high_rr_mask
        final_weights.loc[other_mask] = (
            logic_score.loc[other_mask] * result_weight.loc[other_mask]
        )

        # 4.1 兜底：如果有效样本总权重接近 0，则回退到 uniform 权重（否则训练会报错/无意义）
        total_w = (
            float(final_weights.loc[valid_mask].sum()) if valid_mask.any() else 0.0
        )
        if total_w <= min_total_weight:
            # 回退：有效样本权重全部设为 1.0
            final_weights.loc[valid_mask] = 1.0
            normalize_weights = False

        # 5. 归一化（可选）
        if normalize_weights:
            # 归一化到均值 1.0，保持相对比例
            mean_weight = final_weights[valid_mask].mean()
            if mean_weight > 0:
                final_weights = final_weights / mean_weight

        weights.loc[valid_mask] = final_weights.loc[valid_mask]

    elif weight_strategy == "composite":
        # 策略：组合多种策略
        # 可以组合多个策略，权重相乘或相加
        sub_strategies = weight_config.get("sub_strategies", [])
        combine_mode = weight_config.get(
            "combine_mode", "multiply"
        )  # "multiply" or "add"

        if not sub_strategies:
            return weights

        # 递归调用各个子策略
        sub_weights_list = []
        for sub_strategy in sub_strategies:
            sub_name = sub_strategy.get("name", "uniform")
            sub_config = sub_strategy.get("config", {})
            sub_w = compute_sr_reversal_sample_weights(
                df,
                labels,
                sub_name,
                sub_config,
                atr_col,
                atr_window,
                price_col,
                high_col,
                low_col,
                max_holding_bars,
                stop_loss_r,
                take_profit_r,
            )
            sub_weights_list.append(sub_w)

        # 组合权重
        if combine_mode == "multiply":
            combined = pd.Series(1.0, index=labels.index)
            for sub_w in sub_weights_list:
                combined = combined * sub_w
            weights = combined
        else:  # add
            combined = pd.Series(0.0, index=labels.index)
            for sub_w in sub_weights_list:
                combined = combined + sub_w
            weights = combined

    else:
        print(
            f"   ⚠️  Unknown weight strategy: {weight_strategy}, using uniform weights"
        )

    # 统计信息
    n_valid = valid_mask.sum()
    if n_valid > 0:
        weighted_samples = (weights > 1.0).sum()
        print(f"   [样本权重] 策略: {weight_strategy}")
        print(f"   [样本权重] 总有效样本: {n_valid}")
        print(
            f"   [样本权重] 加权样本: {weighted_samples} ({100*weighted_samples/n_valid:.2f}%)"
        )
        print(f"   [样本权重] 权重范围: [{weights.min():.2f}, {weights.max():.2f}]")
        print(f"   [样本权重] 权重均值: {weights[valid_mask].mean():.2f}")

    return weights


def compute_sr_reversal_label_with_weights(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    combine_mode: str = "any_success",
    sr_mask_col: Optional[str] = None,
    dist_to_sr_col: Optional[str] = None,
    dist_atr_mult: float = 1.5,
    # 样本权重参数
    compute_weights: bool = False,
    weight_strategy: str = "uniform",
    weight_config: Optional[Dict] = None,
    return_weights: bool = False,
) -> pd.Series | Tuple[pd.Series, pd.Series]:
    """
    计算 SR 反转标签，可选地计算样本权重。

    这个函数结合了 `compute_sr_reversal_label_full_scan` 和 `compute_sr_reversal_rr_continuous_label`
    的优点：
    - 使用二元标签（更适合树模型）
    - 支持 long/short 合并
    - 可选地在 SR 附近过滤（通过 sr_mask_col 或 dist_to_sr_col）
    - 可选地计算样本权重

    Args:
        df: DataFrame with features
        price_col: 价格列名
        high_col: 最高价列名
        low_col: 最低价列名
        atr_col: ATR 列名
        atr_window: ATR 窗口
        max_holding_bars: 最大持仓周期
        stop_loss_r: 止损 R 倍数
        take_profit_r: 止盈 R 倍数
        combine_mode: long/short 合并模式 ("any_success", "long_only", "short_only")
        sr_mask_col: SR 掩码列名（如果提供，只在 SR 附近保留标签）
        dist_to_sr_col: 距离 SR 的列名（如果提供，用于计算 SR 掩码）
        dist_atr_mult: 距离 SR 的 ATR 倍数阈值
        compute_weights: 是否计算样本权重
        weight_strategy: 权重策略名称
        weight_config: 权重策略配置
        return_weights: 是否返回权重（如果 True，返回 (labels, weights) 元组）

    Returns:
        如果 return_weights=False: 返回标签 Series
        如果 return_weights=True: 返回 (labels, weights) 元组
    """
    # 计算标签（使用全量扫描方式）
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # 计算 long 和 short 标签
    work_df["__long_signal"] = 1.0
    long_labels = compute_rr_label(
        work_df,
        signal_col="__long_signal",
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
    )

    work_df["__short_signal"] = -1.0
    short_labels = compute_rr_label(
        work_df,
        signal_col="__short_signal",
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
    )

    # 合并 long/short
    combined_labels = _combine_long_short(long_labels, short_labels, mode=combine_mode)

    # 可选：在 SR 附近过滤
    if sr_mask_col is not None and sr_mask_col in work_df.columns:
        sr_mask = work_df[sr_mask_col].fillna(False).astype(bool)
        combined_labels = combined_labels.where(sr_mask)
    elif dist_to_sr_col is not None and dist_to_sr_col in work_df.columns:
        # dist_to_nearest_sr 是相对百分比，需要转换为绝对价格距离后再与 ATR 比较
        price_series = work_df[price_col]
        dist_pct = work_df[dist_to_sr_col].abs()
        # 将百分比距离转换为绝对价格距离
        abs_distance = dist_pct * price_series
        # 计算归一化距离（单位：ATR）
        dist_normalized = abs_distance / (atr_series + 1e-8)
        # 判断是否在SR附近
        sr_mask = dist_normalized <= dist_atr_mult
        sr_mask = sr_mask.fillna(False)
        combined_labels = combined_labels.where(sr_mask)

    # 计算样本权重（如果需要）
    if compute_weights:
        weights = compute_sr_reversal_sample_weights(
            work_df,
            combined_labels,
            weight_strategy=weight_strategy,
            weight_config=weight_config,
            atr_col=atr_col,
            atr_window=atr_window,
            price_col=price_col,
            high_col=high_col,
            low_col=low_col,
            max_holding_bars=max_holding_bars,
            stop_loss_r=stop_loss_r,
            take_profit_r=take_profit_r,
        )
        # 将权重存储到原始 DataFrame 中（如果调用者需要）
        df["sample_weight"] = weights
        if return_weights:
            return combined_labels, weights
        else:
            return combined_labels
    else:
        if return_weights:
            # 即使不计算权重，也返回统一权重
            weights = pd.Series(1.0, index=combined_labels.index)
            return combined_labels, weights
        else:
            return combined_labels


# NOTE: _apply_env_overrides function removed - no longer used
# Label generation now uses compute_sr_reversal_label_full_scan which does NOT
# use signal filtering. Diagnostic scripts create SRSignalConfig() directly.
# If needed in the future, environment variable support can be re-added.
