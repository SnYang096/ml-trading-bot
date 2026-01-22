from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RRExecutionReturnsConfig:
    """
    Compute counterfactual per-step returns for MEAN/TREND modes using
    an ATR-based R/R exit simulation (no vectorbt dependency).

    Output return series is aligned as:
      ret[t] = position[t] * (close[t+1]/close[t] - 1)

    where position[t] is 0 when flat and +/-1 when in a simulated trade.
    """

    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"

    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    # Use provided ATR if present, else compute simple ATR (TR rolling mean)
    atr_col: str = "atr"
    atr_window: int = 14

    # RR exit parameters
    max_holding_bars: int = 24
    stop_loss_r: float = 1.0
    take_profit_r: float = 2.0
    entry_offset: int = 1  # enter next bar
    use_time_exit: bool = True
    use_trailing_stop: bool = False
    trailing_atr_mult: float = 1.0
    use_breakeven_stop: bool = False

    # TREND-specific execution overrides (TC-friendly trailing stop)
    trend_use_time_exit: bool = False
    trend_use_trailing_stop: bool = True
    trend_trailing_atr_mult: float = 1.0
    trend_take_profit_r: float = 100.0  # effectively disable TP; exit via SL
    trend_stop_loss_r: float = 1.0
    trend_use_breakeven_stop: bool = False

    # MEAN-specific execution overrides (event-style mean)
    mean_use_time_exit: bool = True
    mean_use_trailing_stop: bool = True
    mean_trailing_atr_mult: float = 3.0
    mean_take_profit_r: float = 5.0
    mean_stop_loss_r: float = 3.0
    mean_use_breakeven_stop: bool = False

    # ET-specific execution overrides (Exhaustion Turn: trend late stage reversal)
    # ET requires faster take-profit and wider stop-loss due to reversal nature
    et_use_time_exit: bool = True
    et_use_trailing_stop: bool = True
    et_trailing_atr_mult: float = 2.0  # Tighter trailing stop for ET
    et_take_profit_r: float = (
        1.5  # Faster take-profit (from ET config: 2.0, but optimized for reversal)
    )
    et_stop_loss_r: float = (
        1.5  # Wider stop-loss (from ET config: 1.0, but optimized for reversal)
    )
    et_use_breakeven_stop: bool = False

    # How to decide direction from heads
    head_dir_score_col: str = "head_dir_score"

    # Entry gate based on heads (trade only when primitives suggest opportunity)
    head_mfe_col: str = "head_mfe_atr"
    head_mae_col: str = "head_mae_atr"
    mfe_min: float = 0.4
    eff_min: float = 1.05
    eps: float = 1e-9


def _ensure_atr(df: pd.DataFrame, *, cfg: RRExecutionReturnsConfig) -> pd.Series:
    if cfg.atr_col in df.columns:
        return pd.to_numeric(df[cfg.atr_col], errors="coerce").astype(float)
    high = pd.to_numeric(df[cfg.high_col], errors="coerce").astype(float)
    low = pd.to_numeric(df[cfg.low_col], errors="coerce").astype(float)
    close = pd.to_numeric(df[cfg.close_col], errors="coerce").astype(float)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=int(cfg.atr_window), min_periods=1).mean().astype(float)


def _compute_entry_signal_and_dir(
    df: pd.DataFrame,
    *,
    cfg: RRExecutionReturnsConfig,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (entry_signal_bool, dir_sign) arrays aligned to df rows.
    dir_sign in {-1,0,1} gives desired trade direction for that mode.
    """
    dir_score = (
        pd.to_numeric(df[cfg.head_dir_score_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    base_sign = np.sign(dir_score).astype(int)
    if str(mode).upper() == "TREND":
        sign = base_sign
    elif str(mode).upper() == "MEAN":
        sign = -base_sign
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mfe = (
        pd.to_numeric(df[cfg.head_mfe_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    mae = (
        pd.to_numeric(df[cfg.head_mae_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    eff = np.where(
        np.isfinite(mfe) & np.isfinite(mae),
        mfe / (mae + float(cfg.eps)),
        0.0,
    )
    tradable = (mfe >= float(cfg.mfe_min)) & (eff >= float(cfg.eff_min)) & (sign != 0)
    return tradable.astype(bool), sign.astype(int)


def _simulate_rr_position(
    df: pd.DataFrame,
    *,
    entry_ok: np.ndarray,
    dir_sign: np.ndarray,
    atr: np.ndarray,
    cfg: RRExecutionReturnsConfig,
) -> np.ndarray:
    """
    Single-position RR simulator (no overlapping trades).
    Produces position[t] in {-1,0,1}, where position[t] applies to next-bar return at t.
    """
    high = pd.to_numeric(df[cfg.high_col], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df[cfg.low_col], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df[cfg.close_col], errors="coerce").to_numpy(dtype=float)

    T = len(df)
    pos = np.zeros(T, dtype=int)
    # NOTE:
    # This is a single-position simulator in the sense that it prevents overlapping trades
    # by jumping the time index forward to the exit point. It must still allow *multiple*
    # sequential trades over the series.

    def _scan_exit(i: int, sign: int) -> int:
        entry_off = int(max(cfg.entry_offset, 1))
        scan_start = i + entry_off
        if scan_start >= T:
            return T - 1
        entry_price = close[scan_start - 1]  # approximate: enter at close of prior bar
        atr_i = atr[i]
        if not np.isfinite(entry_price) or not np.isfinite(atr_i) or atr_i <= 0:
            return min(scan_start + int(cfg.max_holding_bars), T) - 1

        if sign > 0:
            initial_stop = entry_price - float(cfg.stop_loss_r) * atr_i
            take_profit = entry_price + float(cfg.take_profit_r) * atr_i
            breakeven_trigger = entry_price + float(cfg.stop_loss_r) * atr_i
        else:
            initial_stop = entry_price + float(cfg.stop_loss_r) * atr_i
            take_profit = entry_price - float(cfg.take_profit_r) * atr_i
            breakeven_trigger = entry_price - float(cfg.stop_loss_r) * atr_i
        stop_loss = initial_stop
        breakeven_activated = False

        end_idx = (
            min(scan_start + int(cfg.max_holding_bars), T)
            if bool(cfg.use_time_exit)
            else T
        )

        for j in range(scan_start, end_idx):
            h = high[j]
            l = low[j]
            atr_j = atr[j]

            if bool(cfg.use_trailing_stop) and np.isfinite(atr_j) and atr_j > 0:
                if sign > 0:
                    cand = h - float(cfg.trailing_atr_mult) * atr_j
                    stop_loss = max(stop_loss, cand)
                else:
                    cand = l + float(cfg.trailing_atr_mult) * atr_j
                    stop_loss = min(stop_loss, cand)

            if sign > 0:
                if (
                    bool(cfg.use_breakeven_stop)
                    and (not breakeven_activated)
                    and (h >= breakeven_trigger)
                ):
                    stop_loss = entry_price
                    breakeven_activated = True
                # TP then SL
                if h >= take_profit:
                    return j
                if l <= stop_loss:
                    return j
            else:
                if (
                    bool(cfg.use_breakeven_stop)
                    and (not breakeven_activated)
                    and (l <= breakeven_trigger)
                ):
                    stop_loss = entry_price
                    breakeven_activated = True
                if l <= take_profit:
                    return j
                if h >= stop_loss:
                    return j

        return end_idx - 1 if end_idx > 0 else T - 1

    t = 0
    while t < T:
        if bool(entry_ok[t]) and int(dir_sign[t]) != 0:
            sign = int(dir_sign[t])
            eff_start = int(t)
            # Determine exit (index of bar where TP/SL/time exit triggers)
            exit_idx = int(_scan_exit(t, sign))
            # Activate position from eff_start until (exit_idx - 1) inclusive.
            # In slice semantics that's [eff_start : exit_idx).
            end_hold = max(eff_start + 1, exit_idx)
            pos[eff_start:end_hold] = sign
            # Jump forward to avoid overlapping trades; next opportunity starts at exit_idx.
            t = end_hold
            continue
        t += 1
    return pos


def compute_rr_execution_mode_returns(
    df: pd.DataFrame,
    *,
    cfg: RRExecutionReturnsConfig = RRExecutionReturnsConfig(),
    archetype_col: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute (ret_mean, ret_trend) per symbol using RR-based position simulation.
    Requires OHLC columns + head_* columns.

    Args:
        df: DataFrame with OHLC and head_* columns
        cfg: RR execution configuration
        archetype_col: Optional column name for archetype (e.g., 'gate_archetype').
                      If provided and contains 'ET', uses ET-specific config for ret_mean.
    """
    needed = [
        cfg.symbol_col,
        cfg.timestamp_col,
        cfg.high_col,
        cfg.low_col,
        cfg.close_col,
        cfg.head_dir_score_col,
        cfg.head_mfe_col,
        cfg.head_mae_col,
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for RR execution returns: {missing}"
        )

    out_mean = pd.Series(0.0, index=df.index, dtype=float)
    out_trend = pd.Series(0.0, index=df.index, dtype=float)

    for sym, g in df.groupby(cfg.symbol_col, sort=False):
        g = g.sort_values(cfg.timestamp_col)
        atr_s = _ensure_atr(g, cfg=cfg).to_numpy(dtype=float)
        r_next = (
            pd.to_numeric(g[cfg.close_col], errors="coerce")
            .astype(float)
            .pct_change()
            .shift(-1)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        # TREND (TC-friendly trailing stop)
        entry_ok_t, sign_t = _compute_entry_signal_and_dir(g, cfg=cfg, mode="TREND")
        cfg_trend = replace(
            cfg,
            use_time_exit=cfg.trend_use_time_exit,
            use_trailing_stop=cfg.trend_use_trailing_stop,
            trailing_atr_mult=cfg.trend_trailing_atr_mult,
            take_profit_r=cfg.trend_take_profit_r,
            stop_loss_r=cfg.trend_stop_loss_r,
            use_breakeven_stop=cfg.trend_use_breakeven_stop,
        )
        pos_t = _simulate_rr_position(
            g, entry_ok=entry_ok_t, dir_sign=sign_t, atr=atr_s, cfg=cfg_trend
        )
        ret_t = pd.Series(pos_t.astype(float) * r_next, index=g.index, dtype=float)

        # MEAN (event-style mean)
        # Check if ET archetype is present and use ET-specific config
        use_et_config = False
        if archetype_col and archetype_col in g.columns:
            # Check if any row has ET archetype
            et_mask = (
                g[archetype_col].astype(str).str.contains("ET", case=False, na=False)
            )
            use_et_config = et_mask.any()

        if use_et_config:
            # ET-specific config (faster TP, wider SL for reversal)
            entry_ok_m, sign_m = _compute_entry_signal_and_dir(g, cfg=cfg, mode="MEAN")
            cfg_mean = replace(
                cfg,
                use_time_exit=cfg.et_use_time_exit,
                use_trailing_stop=cfg.et_use_trailing_stop,
                trailing_atr_mult=cfg.et_trailing_atr_mult,
                take_profit_r=cfg.et_take_profit_r,
                stop_loss_r=cfg.et_stop_loss_r,
                use_breakeven_stop=cfg.et_use_breakeven_stop,
            )
        else:
            # Standard MEAN config (FR and other mean-reversion)
            entry_ok_m, sign_m = _compute_entry_signal_and_dir(g, cfg=cfg, mode="MEAN")
            cfg_mean = replace(
                cfg,
                use_time_exit=cfg.mean_use_time_exit,
                use_trailing_stop=cfg.mean_use_trailing_stop,
                trailing_atr_mult=cfg.mean_trailing_atr_mult,
                take_profit_r=cfg.mean_take_profit_r,
                stop_loss_r=cfg.mean_stop_loss_r,
                use_breakeven_stop=cfg.mean_use_breakeven_stop,
            )
        pos_m = _simulate_rr_position(
            g, entry_ok=entry_ok_m, dir_sign=sign_m, atr=atr_s, cfg=cfg_mean
        )
        ret_m = pd.Series(pos_m.astype(float) * r_next, index=g.index, dtype=float)

        out_mean.loc[g.index] = ret_m.values
        out_trend.loc[g.index] = ret_t.values

    return out_mean.astype(float), out_trend.astype(float)
