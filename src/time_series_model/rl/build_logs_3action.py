from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    compute_mode_3action,
)
from src.time_series_model.rl.execution_returns_rr import (
    RRExecutionReturnsConfig,
    compute_rr_execution_mode_returns,
)


@dataclass(frozen=True)
class BuildLogs3ActionConfig:
    """
    Build RL/BC-ready logs for 3-action Router from:
      - nnmultihead predictions (pred_* heads)
      - raw OHLCV (close) for per-step counterfactual returns

    Output is a single table with at least:
      symbol, timestamp, mode,
      head_dir_score, head_mfe_atr, head_mae_atr, head_t_to_mfe,
      drawdown, ret_mean, ret_trend

    Important: returns are computed WITHOUT using future information beyond next bar.
    """

    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"
    close_col: str = "close"

    # nnmultihead pred columns
    pred_dir_prob_col: str = "pred_dir_prob"
    pred_mfe_col: str = "pred_mfe_atr"
    pred_mae_col: str = "pred_mae_atr"
    pred_ttm_col: str = "pred_t_to_mfe"

    # output head columns
    head_dir_score_col: str = "head_dir_score"  # signed score in [-1,1]
    head_mfe_col: str = "head_mfe_atr"
    head_mae_col: str = "head_mae_atr"
    head_ttm_col: str = "head_t_to_mfe"

    # return columns for counterfactual eval env (next-step return per unit exposure)
    ret_mean_col: str = "ret_mean"
    ret_trend_col: str = "ret_trend"

    # market state
    drawdown_col: str = "drawdown"

    # how to compute direction proxy for ret_mean/ret_trend (uses only past prices)
    momentum_lookback: int = 5

    # how to compute ret_mean/ret_trend
    # - "momentum_proxy": past-momentum sign * next return (fallback)
    # - "rr_execution": RR/ATR single-position execution simulator (uses OHLC, no vectorbt)
    returns_source: str = "momentum_proxy"
    rr_returns_cfg: RRExecutionReturnsConfig = RRExecutionReturnsConfig()

    # If preds are in log1p space (typical), inverse-transform regression heads to ATR units / bars.
    preds_in_log1p: bool = True

    # If mode not provided, we compute it from preds using pure rules
    rule_cfg: Rule3ActionConfig = Rule3ActionConfig()


def _ensure_timestamp_column(df: pd.DataFrame, *, ts_col: str) -> pd.DataFrame:
    if ts_col in df.columns:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out[ts_col] = out.index
        return out
    # fallback: make a monotonically increasing integer timestamp
    out = df.copy()
    out[ts_col] = np.arange(len(out), dtype=int)
    return out


def _inverse_log1p(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    # Prevent overflow from extreme log1p predictions.
    v = v.clip(lower=0.0, upper=15.0)
    return np.expm1(v)


def _compute_market_drawdown(close: pd.Series) -> pd.Series:
    r = pd.to_numeric(close, errors="coerce").astype(float).pct_change().fillna(0.0)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0.0, np.nan)
    return dd.fillna(0.0).astype(float)


def _compute_mode_returns(
    close: pd.Series,
    *,
    lookback: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute per-step counterfactual returns for the two modes:
      - TREND: follow a past-only momentum proxy
      - MEAN: take the opposite sign of the same proxy

    Both returns are next-step close-to-close returns (t -> t+1).
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    r_next = (c.shift(-1) / c - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mom = (
        (c / c.shift(int(lookback)) - 1.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    s = np.sign(mom.to_numpy(dtype=float))
    ret_trend = pd.Series(s, index=c.index, dtype=float) * r_next
    ret_mean = -pd.Series(s, index=c.index, dtype=float) * r_next
    return ret_mean.astype(float), ret_trend.astype(float)


def build_logs_3action(
    preds_df: pd.DataFrame,
    *,
    raw_df: pd.DataFrame,
    cfg: BuildLogs3ActionConfig = BuildLogs3ActionConfig(),
    mode_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a single logs dataframe. Both preds_df and raw_df may contain multiple symbols,
    but computations (returns/drawdown) are strictly per-symbol to avoid leakage.
    """
    if cfg.symbol_col not in preds_df.columns:
        raise ValueError(f"preds_df missing required symbol column '{cfg.symbol_col}'")
    if cfg.symbol_col not in raw_df.columns and "_symbol" in raw_df.columns:
        raw_df = raw_df.rename(columns={"_symbol": cfg.symbol_col})
    if cfg.symbol_col not in raw_df.columns:
        raise ValueError(
            f"raw_df missing required symbol column '{cfg.symbol_col}' (or '_symbol')"
        )

    preds_df = _ensure_timestamp_column(preds_df, ts_col=cfg.timestamp_col)
    raw_df = _ensure_timestamp_column(raw_df, ts_col=cfg.timestamp_col)

    # Prepare/compute mode
    if mode_df is None:
        mode_out = compute_mode_3action(
            preds_df, cfg=cfg.rule_cfg, preds_in_log1p=cfg.preds_in_log1p
        )
        mode_df = preds_df[[cfg.symbol_col, cfg.timestamp_col]].copy()
        mode_df["mode"] = mode_out["mode"].astype(str).values
    else:
        mode_df = _ensure_timestamp_column(mode_df, ts_col=cfg.timestamp_col)
        if cfg.symbol_col not in mode_df.columns:
            raise ValueError(
                f"mode_df missing required symbol column '{cfg.symbol_col}'"
            )
        if "mode" not in mode_df.columns:
            raise ValueError("mode_df must contain 'mode' column")
        mode_df = mode_df[[cfg.symbol_col, cfg.timestamp_col, "mode"]].copy()

    # Join preds + mode + raw (symbol, timestamp)
    preds_keyed = preds_df.set_index([cfg.symbol_col, cfg.timestamp_col], drop=False)
    mode_keyed = mode_df.set_index([cfg.symbol_col, cfg.timestamp_col], drop=False)
    raw_keyed = raw_df.set_index([cfg.symbol_col, cfg.timestamp_col], drop=False)

    if cfg.close_col not in raw_keyed.columns:
        raise ValueError(f"raw_df missing close column '{cfg.close_col}'")

    raw_cols = [cfg.close_col]
    if str(cfg.returns_source).lower() in {"rr_execution", "rr"}:
        # Need OHLC (and optional atr) for RR execution simulator
        for c in [
            cfg.rr_returns_cfg.open_col,
            cfg.rr_returns_cfg.high_col,
            cfg.rr_returns_cfg.low_col,
        ]:
            if c not in raw_cols:
                raw_cols.append(c)
        if (
            cfg.rr_returns_cfg.atr_col in raw_keyed.columns
            and cfg.rr_returns_cfg.atr_col not in raw_cols
        ):
            raw_cols.append(cfg.rr_returns_cfg.atr_col)

    # Join in a way that avoids column overlap (preds outputs already include OHLC in many cases).
    joined = preds_keyed.join(mode_keyed[["mode"]], how="inner")
    need_from_raw = [c for c in raw_cols if c not in joined.columns]
    if need_from_raw:
        joined = joined.join(raw_keyed[need_from_raw], how="inner")
    # Avoid pandas ambiguity: symbol/timestamp exist as both index levels and columns (drop=False above).
    joined = joined.reset_index(drop=True).sort_values(
        [cfg.symbol_col, cfg.timestamp_col]
    )

    # Build head columns (prefer interpretable units)
    p = (
        pd.to_numeric(joined[cfg.pred_dir_prob_col], errors="coerce")
        .fillna(0.5)
        .astype(float)
    )
    joined[cfg.head_dir_score_col] = (p * 2.0 - 1.0).clip(-1.0, 1.0)

    if cfg.preds_in_log1p:
        joined[cfg.head_mfe_col] = _inverse_log1p(joined[cfg.pred_mfe_col])
        joined[cfg.head_mae_col] = _inverse_log1p(joined[cfg.pred_mae_col])
        joined[cfg.head_ttm_col] = _inverse_log1p(joined[cfg.pred_ttm_col])
    else:
        joined[cfg.head_mfe_col] = (
            pd.to_numeric(joined[cfg.pred_mfe_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        joined[cfg.head_mae_col] = (
            pd.to_numeric(joined[cfg.pred_mae_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        joined[cfg.head_ttm_col] = (
            pd.to_numeric(joined[cfg.pred_ttm_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

    # Safety clip: keep heads in reasonable numeric ranges for downstream BC/RL.
    for c in [cfg.head_mfe_col, cfg.head_mae_col, cfg.head_ttm_col]:
        joined[c] = (
            pd.to_numeric(joined[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        joined[c] = joined[c].clip(lower=0.0, upper=1e6)

    # Per-symbol: compute drawdown
    joined[cfg.drawdown_col] = joined.groupby(cfg.symbol_col, sort=False)[
        cfg.close_col
    ].transform(_compute_market_drawdown)

    src = str(cfg.returns_source).lower()
    if src in {"momentum_proxy", "momentum", "proxy"}:
        joined[cfg.ret_mean_col] = joined.groupby(cfg.symbol_col, sort=False)[
            cfg.close_col
        ].transform(
            lambda s: _compute_mode_returns(s, lookback=int(cfg.momentum_lookback))[0]
        )
        joined[cfg.ret_trend_col] = joined.groupby(cfg.symbol_col, sort=False)[
            cfg.close_col
        ].transform(
            lambda s: _compute_mode_returns(s, lookback=int(cfg.momentum_lookback))[1]
        )
    elif src in {"rr_execution", "rr"}:
        # RR execution simulator expects head_* and OHLC columns to exist
        ret_mean, ret_trend = compute_rr_execution_mode_returns(
            joined, cfg=cfg.rr_returns_cfg
        )
        joined[cfg.ret_mean_col] = ret_mean.values
        joined[cfg.ret_trend_col] = ret_trend.values
    else:
        raise ValueError(f"Unknown returns_source: {cfg.returns_source}")

    # Final column selection
    out = joined[
        [
            cfg.symbol_col,
            cfg.timestamp_col,
            "mode",
            cfg.head_dir_score_col,
            cfg.head_mfe_col,
            cfg.head_mae_col,
            cfg.head_ttm_col,
            cfg.drawdown_col,
            cfg.ret_mean_col,
            cfg.ret_trend_col,
        ]
    ].copy()

    # Ensure types
    out["mode"] = out["mode"].astype(str)
    for c in [
        cfg.head_dir_score_col,
        cfg.head_mfe_col,
        cfg.head_mae_col,
        cfg.head_ttm_col,
        cfg.drawdown_col,
        cfg.ret_mean_col,
        cfg.ret_trend_col,
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

    return out.reset_index(drop=True)
