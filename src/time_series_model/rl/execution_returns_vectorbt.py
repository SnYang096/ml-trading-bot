from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import simulate_rr_exits


@dataclass(frozen=True)
class VectorBTExecutionReturnsConfig:
    """
    Compute counterfactual per-step returns for MEAN/TREND modes using vectorbt,
    so execution details are consistent with the repository's VectorBT backtest semantics.

    Important semantics:
    - We construct two portfolios per symbol:
        TREND: regression score = head_dir_score * efficiency
        MEAN:  regression score = -TREND score
      Then we use symmetric top/bottom quantile entries (bi-directional).
    - RR exits are generated using the same `simulate_rr_exits` utility as VectorBTBacktest.
    - Returned series are portfolio per-step returns (net of fees/slippage configured here).

    This is intended for `rl build-logs-3action` returns_source='vectorbt_execution'.
    """

    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"

    # OHLC / ATR columns
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    atr_col: str = "atr"

    # Heads (must be in interpretable units already)
    head_dir_score_col: str = "head_dir_score"
    head_mfe_col: str = "head_mfe_atr"
    head_mae_col: str = "head_mae_atr"
    eps: float = 1e-9

    # Entry gating
    top_quantile: float = 0.05
    bottom_quantile: float = 0.05
    entry_mode: str = "cross"  # level|cross

    # Execution / backtest assumptions
    fee: float = 0.0004
    slippage: float = 0.0001
    initial_cash: float = 10000.0
    freq: Optional[str] = "4H"

    # RR exits
    max_holding_bars: int = 24
    stop_loss_r: float = 1.0
    take_profit_r: float = 2.0
    atr_window: int = 14
    entry_offset: int = 1
    entry_price_col: Optional[str] = "open"
    use_breakeven_stop: bool = False
    use_time_exit: bool = True
    use_trailing_stop: bool = False
    trailing_atr_mult: float = 1.0


def _infer_freq(index: pd.DatetimeIndex, *, fallback: Optional[str]) -> str:
    inf = index.inferred_freq
    if inf:
        return str(inf)
    if fallback:
        return str(fallback)
    # conservative fallback for vectorbt metrics; 4H is common in this repo
    return "4H"


def _build_entries_from_regression_score(
    score: pd.Series, *, top_q: float, bottom_q: float, entry_mode: str
) -> Tuple[pd.Series, pd.Series]:
    s = pd.to_numeric(score, errors="coerce").astype(float)
    top_q = float(min(max(top_q, 0.0), 1.0))
    bottom_q = float(min(max(bottom_q, 0.0), 1.0))

    # thresholds on the (full-series) distribution, consistent with VectorBTBacktest regression branch
    long_thr = float(s.quantile(1.0 - top_q)) if top_q > 0 else float("inf")
    short_thr = float(s.quantile(bottom_q)) if bottom_q > 0 else float("-inf")

    long_raw = s >= long_thr
    short_raw = s <= short_thr

    em = str(entry_mode).lower()
    if em == "cross":
        long_prev = long_raw.shift(1).fillna(False)
        short_prev = short_raw.shift(1).fillna(False)
        long_entries = long_raw & (~long_prev)
        short_entries = short_raw & (~short_prev)
    else:
        long_entries = long_raw
        short_entries = short_raw

    return long_entries.astype(bool), short_entries.astype(bool)


def _portfolio_returns_from_entries_rr(
    df: pd.DataFrame,
    *,
    score: pd.Series,
    cfg: VectorBTExecutionReturnsConfig,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Build a vectorbt portfolio from (score -> quantile entries) + RR exits,
    return (returns_series, summary_metrics).
    """
    try:
        import vectorbt as vbt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "vectorbt is required for returns_source='vectorbt_execution'"
        ) from exc

    # index
    if cfg.timestamp_col in df.columns:
        idx = pd.to_datetime(df[cfg.timestamp_col], utc=True, errors="coerce")
        df = df.copy()
        df.index = idx
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "vectorbt_execution requires a DatetimeIndex or timestamp column."
        )
    df = df.sort_index()
    # Align score to the same index (avoid boolean indexer mismatch)
    score = pd.Series(
        pd.to_numeric(score, errors="coerce").to_numpy(dtype=float), index=df.index
    )

    price = pd.to_numeric(df[cfg.close_col], errors="coerce").astype(float)
    if price.isna().all():
        # no usable price -> zero returns
        return pd.Series(0.0, index=df.index, dtype=float), {"sharpe": float("nan")}

    # entries (bi-directional)
    long_entries, short_entries = _build_entries_from_regression_score(
        score,
        top_q=cfg.top_quantile,
        bottom_q=cfg.bottom_quantile,
        entry_mode=cfg.entry_mode,
    )

    # RR exits via simulate_rr_exits (same semantics as VectorBTBacktest)
    signal_col = "signal"
    df_rr = df.copy()
    rr_signal = pd.Series(0.0, index=df_rr.index, dtype=float)
    rr_signal.loc[long_entries] = 1.0
    rr_signal.loc[short_entries] = -1.0
    df_rr[signal_col] = rr_signal

    long_exits, short_exits = simulate_rr_exits(
        df_rr,
        signal_col=signal_col,
        price_col=cfg.close_col,
        atr_col=cfg.atr_col,
        atr_window=int(cfg.atr_window),
        max_holding_bars=int(cfg.max_holding_bars),
        stop_loss_r=float(cfg.stop_loss_r),
        take_profit_r=float(cfg.take_profit_r),
        entry_price_col=cfg.entry_price_col,
        entry_offset=int(cfg.entry_offset),
        use_breakeven_stop=bool(cfg.use_breakeven_stop),
        use_time_exit=bool(cfg.use_time_exit),
        use_trailing_stop=bool(cfg.use_trailing_stop),
        trailing_atr_mult=float(cfg.trailing_atr_mult),
    )

    freq = _infer_freq(df_rr.index, fallback=cfg.freq)
    portfolio = vbt.Portfolio.from_signals(
        price,
        entries=long_entries.reindex(df_rr.index).fillna(False),
        exits=long_exits.reindex(df_rr.index).fillna(False),
        short_entries=short_entries.reindex(df_rr.index).fillna(False),
        short_exits=short_exits.reindex(df_rr.index).fillna(False),
        init_cash=float(cfg.initial_cash),
        fees=float(cfg.fee),
        slippage=float(cfg.slippage),
        freq=freq,
        size=1.0,
    )

    rets = portfolio.returns()
    rets = pd.to_numeric(rets, errors="coerce").fillna(0.0).astype(float)
    # Summary (for debug / parity checks)
    try:
        stats = portfolio.stats()
        summ = {
            "sharpe": float(stats.get("Sharpe Ratio", float("nan"))),
            "total_return_pct": float(stats.get("Total Return [%]", float("nan"))),
            "max_drawdown_pct": float(stats.get("Max Drawdown [%]", float("nan"))),
            "total_trades": float(stats.get("Total Trades", float("nan"))),
        }
    except Exception:
        summ = {"sharpe": float("nan")}

    return rets, summ


def compute_vectorbt_execution_mode_returns(
    df: pd.DataFrame,
    *,
    cfg: VectorBTExecutionReturnsConfig = VectorBTExecutionReturnsConfig(),
    out_mean_col: str = "ret_mean",
    out_trend_col: str = "ret_trend",
) -> Tuple[pd.Series, pd.Series, Dict[str, Dict[str, float]]]:
    """
    Compute (ret_mean, ret_trend) per symbol using vectorbt.

    Returns:
      (ret_mean, ret_trend, meta_by_symbol)
    """
    needed = [
        cfg.symbol_col,
        cfg.timestamp_col,
        cfg.close_col,
        cfg.high_col,
        cfg.low_col,
        cfg.head_dir_score_col,
        cfg.head_mfe_col,
        cfg.head_mae_col,
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"vectorbt_execution missing required columns: {missing}")

    out_mean = pd.Series(0.0, index=df.index, dtype=float)
    out_trend = pd.Series(0.0, index=df.index, dtype=float)
    meta: Dict[str, Dict[str, float]] = {}

    for sym, g in df.groupby(cfg.symbol_col, sort=False):
        g = g.copy()
        # score: signed direction * efficiency
        dir_score = (
            pd.to_numeric(g[cfg.head_dir_score_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        mfe = (
            pd.to_numeric(g[cfg.head_mfe_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
            .clip(lower=0.0)
        )
        mae = (
            pd.to_numeric(g[cfg.head_mae_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
            .clip(lower=0.0)
        )
        eff = mfe / (mae + float(cfg.eps))
        score_trend = (dir_score * eff).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        score_mean = -score_trend

        ret_t, meta_t = _portfolio_returns_from_entries_rr(
            g, score=score_trend, cfg=cfg
        )
        ret_m, meta_m = _portfolio_returns_from_entries_rr(g, score=score_mean, cfg=cfg)

        # align back to original group index
        # (portfolio returns are indexed by timestamp; map by timestamp)
        ts = pd.to_datetime(g[cfg.timestamp_col], utc=True, errors="coerce")
        m_map = pd.Series(ret_m.values, index=ret_m.index)
        t_map = pd.Series(ret_t.values, index=ret_t.index)
        out_mean.loc[g.index] = m_map.reindex(ts).to_numpy(dtype=float, na_value=0.0)
        out_trend.loc[g.index] = t_map.reindex(ts).to_numpy(dtype=float, na_value=0.0)

        meta[str(sym)] = {"mean": dict(meta_m), "trend": dict(meta_t)}

    out_mean.name = out_mean_col
    out_trend.name = out_trend_col
    return out_mean.astype(float), out_trend.astype(float), meta
