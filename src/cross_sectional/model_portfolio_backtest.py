from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """
    More realistic portfolio backtest for CS signals.

    Key points:
    - Uses 1-bar realized returns computed from close(t+1)/close(t)-1.
    - Rebalances every holding_period_bars at execution timestamps.
    - Uses signal at (execution_ts - execution_lag_bars) to avoid same-bar lookahead.
    - Supports long-only and market-neutral (dollar-neutral) long/short.
    - Costs apply on turnover: (fee_bps + slippage_bps) * turnover.
    - Uses non-overlapping realised returns aligned to holding_period_bars by default.
    """

    mode: str = "market_neutral"  # "long_only" | "market_neutral"
    holding_period_bars: int = 12
    execution_lag_bars: int = 1

    top_k: int = 10
    bottom_k: int = 10  # used only for market_neutral

    gross_leverage: float = 1.0
    max_weight: float = 0.10
    turnover_limit: Optional[float] = None

    fee_bps: float = 2.0
    slippage_bps: float = 0.0
    # funding/borrow applied to short exposure per bar (bps per bar)
    funding_bps_per_bar: float = 0.0
    borrow_bps_per_bar: float = 0.0
    min_assets: int = 12

    # not fully invested: keep a cash fraction unallocated
    cash_buffer: float = 0.0  # 0..1

    # equity curve construction
    equity_mode: str = "compound"  # "simple" | "compound" | "log"
    initial_capital: float = 1.0

    periods_per_year: Optional[float] = None


def _safe_sharpe(x: pd.Series, periods_per_year: Optional[float]) -> float:
    x = x.dropna()
    if x.empty:
        return float("nan")
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd <= 0:
        return float("nan")
    sharpe = mu / sd
    if periods_per_year and periods_per_year > 0:
        sharpe *= float(np.sqrt(periods_per_year))
    return float(sharpe)


def _infer_periods_per_year(index: pd.DatetimeIndex) -> Optional[float]:
    if index is None or len(index) < 3:
        return None
    diffs = index.sort_values().to_series().diff().dropna()
    if diffs.empty:
        return None
    med = diffs.median()
    seconds = med.total_seconds()
    if seconds <= 0:
        return None
    return float(365.0 * 24.0 * 3600.0 / seconds)


def _compute_one_bar_returns(
    panel: pd.DataFrame, close_col: str = "close"
) -> pd.Series:
    if close_col not in panel.columns:
        raise KeyError(f"Missing close column: {close_col}")
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel must be MultiIndex (timestamp, symbol)")

    df = panel[[close_col]].copy()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    # realized 1-bar return aligned at t: close(t+1)/close(t)-1
    nxt = df.groupby(level=1)[close_col].shift(-1)
    ret = nxt / df[close_col] - 1.0
    ret.name = "ret_1"
    return ret


def _compute_horizon_returns(
    panel: pd.DataFrame, *, close_col: str, horizon: int
) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if close_col not in panel.columns:
        raise KeyError(f"Missing close column: {close_col}")
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel must be MultiIndex (timestamp, symbol)")

    df = panel[[close_col]].copy()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    nxt = df.groupby(level=1)[close_col].shift(-horizon)
    ret = nxt / df[close_col] - 1.0
    ret.name = f"ret_{horizon}"
    return ret


def _select_weights(
    *,
    signals: pd.Series,  # indexed by (timestamp, symbol) for a single timestamp slice
    cfg: PortfolioBacktestConfig,
) -> pd.Series:
    # signals: MultiIndex slice at one timestamp
    if signals.empty:
        return pd.Series(dtype=float)
    symbols = signals.index.get_level_values(1)
    w = pd.Series(0.0, index=pd.Index(symbols, name="symbol"), dtype=float)

    s = signals.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < cfg.min_assets:
        return w

    invest_frac = float(1.0 - float(cfg.cash_buffer))
    invest_frac = max(0.0, min(1.0, invest_frac))

    if cfg.mode == "long_only":
        k = int(cfg.top_k)
        k = max(1, min(k, len(s)))
        picked = s.sort_values(ascending=False).iloc[:k]
        if picked.empty:
            return w
        base = (cfg.gross_leverage * invest_frac) / float(len(picked))
        base = min(base, cfg.max_weight)
        w.loc[picked.index.get_level_values(1)] = float(base)
        # may be under-invested due to cap; keep as-is
        return w

    if cfg.mode == "market_neutral":
        k_long = int(cfg.top_k)
        k_short = int(cfg.bottom_k)
        k_long = max(1, min(k_long, len(s)))
        k_short = max(1, min(k_short, len(s)))
        k_short = min(k_short, len(s) - 1) if len(s) > 1 else 0
        if k_short <= 0:
            return w
        ordered = s.sort_values(ascending=True)
        shorts = ordered.iloc[:k_short]
        longs = ordered.iloc[-k_long:]
        # split gross between long/short, enforce net=0
        half_gross = (cfg.gross_leverage * invest_frac) / 2.0
        wl = min(half_gross / float(len(longs)), cfg.max_weight)
        ws = min(half_gross / float(len(shorts)), cfg.max_weight)
        w.loc[longs.index.get_level_values(1)] = float(wl)
        w.loc[shorts.index.get_level_values(1)] = -float(ws)
        return w

    raise ValueError(f"Unknown mode: {cfg.mode}")


def portfolio_backtest_from_signal(
    panel: pd.DataFrame,
    *,
    signal_col: str,
    close_col: str = "close",
    cfg: PortfolioBacktestConfig = PortfolioBacktestConfig(),
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Returns:
      (timeseries_df, metrics)
    """
    if signal_col not in panel.columns:
        raise KeyError(f"Missing signal column: {signal_col}")
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel must be MultiIndex (timestamp, symbol)")

    panel = panel.copy()
    # ensure utc timestamps
    ts = pd.to_datetime(panel.index.get_level_values(0), utc=True, errors="coerce")
    panel.index = pd.MultiIndex.from_arrays(
        [ts, panel.index.get_level_values(1)], names=["timestamp", "symbol"]
    )
    panel = panel.sort_index()

    # realized return aligned to holding period (non-overlapping when rebalancing every H bars)
    H = int(cfg.holding_period_bars)
    retH = _compute_horizon_returns(panel, close_col=close_col, horizon=H)
    sig = panel[signal_col].copy()
    df = pd.concat([sig.rename("signal"), retH], axis=1).dropna(subset=["signal"])
    if df.empty:
        return pd.DataFrame(), {"error": 1.0}

    timestamps = pd.Index(
        sorted(df.index.get_level_values(0).unique()), name="timestamp"
    )
    if len(timestamps) < (cfg.execution_lag_bars + 2):
        return pd.DataFrame(), {"error": 1.0}

    L = int(cfg.execution_lag_bars)
    if H <= 0:
        raise ValueError("holding_period_bars must be positive")
    if L < 0:
        raise ValueError("execution_lag_bars must be >= 0")

    # execution indices (where we can actually trade)
    exec_idx: List[int] = []
    start_i = max(L, 0)
    i = start_i
    while i < len(timestamps):
        exec_idx.append(i)
        i += H

    prev_w: Optional[pd.Series] = None
    rows: List[Dict[str, float]] = []

    for epos in exec_idx:
        t = timestamps[epos]
        sig_pos = epos - L
        if sig_pos < 0:
            continue
        sig_ts = timestamps[sig_pos]
        g = df.xs(sig_ts, level=0, drop_level=False)["signal"]
        new_w = _select_weights(signals=g, cfg=cfg)

        turnover = 0.0
        if prev_w is not None:
            union = prev_w.index.union(new_w.index)
            prev_u = prev_w.reindex(union).fillna(0.0)
            new_u = new_w.reindex(union).fillna(0.0)
            turnover = 0.5 * float((new_u - prev_u).abs().sum())
            if (
                cfg.turnover_limit is not None
                and turnover > float(cfg.turnover_limit) > 0
            ):
                scale = float(cfg.turnover_limit) / turnover
                new_u = prev_u + (new_u - prev_u) * scale
                turnover = 0.5 * float((new_u - prev_u).abs().sum())
            new_w = new_u.reindex(new_w.index).fillna(0.0)

        bps_trade = float(cfg.fee_bps) + float(cfg.slippage_bps)
        trade_cost = bps_trade / 1e4 * turnover

        # realised holding-period return aligned at t
        try:
            r = df.xs(t, level=0, drop_level=False)[f"ret_{H}"]
        except KeyError:
            prev_w = new_w
            continue
        r = r.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty:
            prev_w = new_w
            continue

        w = new_w.reindex(r.index.get_level_values(1)).fillna(0.0)
        gross_ret = float((w.values * r.values).sum())

        short_exposure = float((-w.clip(upper=0.0)).sum())  # sum of abs short weights
        bps_funding = float(cfg.funding_bps_per_bar) + float(cfg.borrow_bps_per_bar)
        funding_cost = (bps_funding / 1e4) * short_exposure * float(H)

        net_ret = gross_ret - trade_cost - funding_cost

        rows.append(
            {
                "timestamp": pd.Timestamp(t),
                "gross_return": gross_ret,
                "net_return": net_ret,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "funding_cost": funding_cost,
                "short_exposure": short_exposure,
                "n_assets": float(len(r)),
            }
        )
        prev_w = new_w.copy()

    tsdf = pd.DataFrame(rows)
    if tsdf.empty:
        return tsdf, {"error": 1.0}
    tsdf["timestamp"] = pd.to_datetime(tsdf["timestamp"], utc=True, errors="coerce")
    tsdf = tsdf.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    # Equity curves (default: compound)
    net_r = tsdf["net_return"].fillna(0.0)
    gross_r = tsdf["gross_return"].fillna(0.0)
    init = float(cfg.initial_capital) if float(cfg.initial_capital) > 0 else 1.0
    mode = str(cfg.equity_mode).strip().lower()
    if mode == "simple":
        tsdf["net_equity"] = init * (1.0 + net_r.cumsum())
        tsdf["gross_equity"] = init * (1.0 + gross_r.cumsum())
    elif mode == "log":
        tsdf["net_equity"] = init * np.exp(
            np.log(np.maximum(1.0 + net_r, 1e-12)).cumsum()
        )
        tsdf["gross_equity"] = init * np.exp(
            np.log(np.maximum(1.0 + gross_r, 1e-12)).cumsum()
        )
    else:  # compound
        tsdf["net_equity"] = init * (1.0 + net_r).cumprod()
        tsdf["gross_equity"] = init * (1.0 + gross_r).cumprod()
    tsdf["net_cum"] = tsdf["net_equity"] / init - 1.0
    tsdf["gross_cum"] = tsdf["gross_equity"] / init - 1.0

    # Drawdown on net equity
    peak = tsdf["net_equity"].cummax()
    tsdf["drawdown"] = tsdf["net_equity"] / peak - 1.0

    ppy = cfg.periods_per_year or _infer_periods_per_year(tsdf.index)
    # If we are measuring per-holding-period returns, annualisation needs adjustment.
    if ppy and ppy > 0:
        ppy = float(ppy) / float(H)
    metrics = {
        "mode": cfg.mode,
        "holding_period_bars": float(cfg.holding_period_bars),
        "execution_lag_bars": float(cfg.execution_lag_bars),
        "top_k": float(cfg.top_k),
        "bottom_k": float(cfg.bottom_k),
        "gross_leverage": float(cfg.gross_leverage),
        "max_weight": float(cfg.max_weight),
        "turnover_limit": (
            float(cfg.turnover_limit) if cfg.turnover_limit is not None else None
        ),
        "fee_bps": float(cfg.fee_bps),
        "slippage_bps": float(cfg.slippage_bps),
        "funding_bps_per_bar": float(cfg.funding_bps_per_bar),
        "borrow_bps_per_bar": float(cfg.borrow_bps_per_bar),
        "cash_buffer": float(cfg.cash_buffer),
        "equity_mode": str(cfg.equity_mode),
        "initial_capital": float(cfg.initial_capital),
        "n_timestamps": float(len(tsdf)),
        "avg_net_return": float(tsdf["net_return"].mean()),
        "avg_gross_return": float(tsdf["gross_return"].mean()),
        "avg_turnover": float(tsdf["turnover"].mean()),
        "avg_short_exposure": (
            float(tsdf["short_exposure"].mean())
            if "short_exposure" in tsdf.columns
            else float("nan")
        ),
        "avg_funding_cost": (
            float(tsdf["funding_cost"].mean())
            if "funding_cost" in tsdf.columns
            else float("nan")
        ),
        "sharpe_net": _safe_sharpe(tsdf["net_return"], ppy),
        "sharpe_gross": _safe_sharpe(tsdf["gross_return"], ppy),
        "periods_per_year": float(ppy) if ppy else float("nan"),
        "total_return_net": float(tsdf["net_equity"].iloc[-1] / init - 1.0),
        "max_drawdown": float(tsdf["drawdown"].min()),
    }
    return tsdf, metrics
