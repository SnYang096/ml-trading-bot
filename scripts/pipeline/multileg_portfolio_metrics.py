"""Portfolio metrics for multi-leg backtests.

Each symbol runs on its own capital bucket (``pnl_per_capital`` per trade is
normalized by that strategy's gross-exposure cap).

Canonical portfolio return: **timeline equity** — sort trades by ``exit_time``,
allocate ``1 / n_symbols`` of total capital per symbol, cumsum increments.

Legacy diagnostics:
- ``return_pct_pooled`` — sum all trades (overstates ~N×).
- ``return_pct_eq_mean`` — mean of per-symbol cumulative pnl (ignores path).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _n_symbols(trades: pd.DataFrame) -> int:
    if trades.empty or "symbol" not in trades.columns:
        return 1
    return max(1, int(trades["symbol"].nunique()))


def portfolio_pnl_from_trades(trades: pd.DataFrame) -> Dict[str, Any]:
    """Pooled vs equal-weight mean of per-symbol cumulative pnl (diagnostics)."""
    if trades.empty or "pnl_per_capital" not in trades.columns:
        return {
            "n_symbols": 0,
            "sum_pnl_per_capital_pooled": 0.0,
            "portfolio_pnl_per_capital": 0.0,
            "return_pct_pooled": 0.0,
            "return_pct_eq_mean": 0.0,
            "per_symbol_pnl_per_capital": {},
        }

    pc = pd.to_numeric(trades["pnl_per_capital"], errors="coerce").fillna(0.0)
    pooled = float(pc.sum())
    if "symbol" in trades.columns:
        per_sym = trades.assign(_pc=pc).groupby("symbol", sort=True)["_pc"].sum()
        n = int(len(per_sym))
        portfolio = float(per_sym.mean()) if n else 0.0
        per_symbol = {str(k): float(v) for k, v in per_sym.items()}
    else:
        n = 1
        portfolio = pooled
        per_symbol = {}

    return {
        "n_symbols": n,
        "sum_pnl_per_capital_pooled": pooled,
        "portfolio_pnl_per_capital": portfolio,
        "return_pct_pooled": pooled * 100.0,
        "return_pct_eq_mean": portfolio * 100.0,
        "per_symbol_pnl_per_capital": per_symbol,
    }


def build_portfolio_equity_curve(
    trades: pd.DataFrame,
    *,
    time_col: str = "exit_time",
    pnl_col: str = "pnl_per_capital",
) -> pd.DataFrame:
    """Chronological portfolio equity (equal capital weight per symbol).

    Each trade contributes ``pnl_per_capital / n_symbols`` to the running total.
    ``equity`` is normalized to 1.0 = full portfolio notional at t0.
    """
    empty_cols = [
        time_col,
        "portfolio_pnl_per_capital",
        "cum_pnl_per_capital",
        "equity",
        "drawdown",
    ]
    if trades.empty or pnl_col not in trades.columns or time_col not in trades.columns:
        return pd.DataFrame(columns=empty_cols)

    n = _n_symbols(trades)
    weight = 1.0 / n
    df = trades[[time_col, pnl_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    df["portfolio_pnl_per_capital"] = (
        pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0) * weight
    )
    df = df.sort_values(time_col).reset_index(drop=True)
    df["cum_pnl_per_capital"] = df["portfolio_pnl_per_capital"].cumsum()
    df["equity"] = 1.0 + df["cum_pnl_per_capital"]
    peak = df["cum_pnl_per_capital"].cummax()
    df["drawdown"] = df["cum_pnl_per_capital"] - peak
    return df[empty_cols]


def portfolio_equity_from_trades(
    trades: pd.DataFrame,
    *,
    time_col: str = "exit_time",
    pnl_col: str = "pnl_per_capital",
) -> Dict[str, Any]:
    """Timeline portfolio return and drawdown from trade exits."""
    curve = build_portfolio_equity_curve(trades, time_col=time_col, pnl_col=pnl_col)
    n = _n_symbols(trades)
    if curve.empty:
        return {
            "n_symbols": n if not trades.empty else 0,
            "portfolio_pnl_per_capital_timeline": 0.0,
            "return_pct_timeline": 0.0,
            "max_drawdown_timeline": 0.0,
        }

    final_pc = float(curve["cum_pnl_per_capital"].iloc[-1])
    return {
        "n_symbols": n,
        "portfolio_pnl_per_capital_timeline": final_pc,
        "return_pct_timeline": final_pc * 100.0,
        "max_drawdown_timeline": float(curve["drawdown"].min()),
    }


def portfolio_metrics_from_trades(trades: pd.DataFrame) -> Dict[str, Any]:
    """Merge diagnostic means with canonical timeline portfolio metrics."""
    diag = portfolio_pnl_from_trades(trades)
    timeline = portfolio_equity_from_trades(trades)
    out = {**diag, **timeline}
    # Primary headline fields (backward-compat names in summaries).
    out["return_pct"] = timeline["return_pct_timeline"]
    out["sum_pnl_per_capital"] = timeline["portfolio_pnl_per_capital_timeline"]
    out["max_drawdown_portfolio"] = timeline["max_drawdown_timeline"]
    out["daily_sharpe"] = portfolio_daily_sharpe_from_trades(trades)
    return out


def portfolio_daily_returns_from_trades(
    trades: pd.DataFrame,
    *,
    time_col: str = "exit_time",
    pnl_col: str = "pnl_per_capital",
) -> pd.Series:
    """Daily portfolio return increments (timeline-weighted, calendar resample)."""
    curve = build_portfolio_equity_curve(trades, time_col=time_col, pnl_col=pnl_col)
    if curve.empty:
        return pd.Series(dtype=float)
    daily = (
        curve.set_index(time_col)["portfolio_pnl_per_capital"]
        .sort_index()
        .resample("1D")
        .sum()
    )
    return daily


def sharpe_from_returns(returns: pd.Series, periods_per_year: float) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def portfolio_daily_sharpe_from_trades(
    trades: pd.DataFrame,
    *,
    periods_per_year: float = 365.0,
    time_col: str = "exit_time",
    pnl_col: str = "pnl_per_capital",
) -> float:
    """Sharpe on calendar-daily portfolio returns (timeline-weighted)."""
    return sharpe_from_returns(
        portfolio_daily_returns_from_trades(trades, time_col=time_col, pnl_col=pnl_col),
        periods_per_year,
    )


def dual_add_summary_fields(
    trades: pd.DataFrame, segments: pd.DataFrame
) -> Dict[str, Any]:
    """Shared one-row summary fields for dual-add / chop-grid aligned reports."""
    agg = portfolio_metrics_from_trades(trades)
    row: Dict[str, Any] = {
        "segments": len(segments),
        "trades": len(trades),
        "n_symbols": agg["n_symbols"],
        "sum_pnl_per_capital": agg["sum_pnl_per_capital"],
        "sum_pnl_per_capital_pooled": agg["sum_pnl_per_capital_pooled"],
        "return_pct": agg["return_pct"],
        "return_pct_timeline": agg["return_pct_timeline"],
        "return_pct_eq_mean": agg["return_pct_eq_mean"],
        "return_pct_pooled": agg["return_pct_pooled"],
        "max_drawdown_portfolio": agg["max_drawdown_portfolio"],
        "daily_sharpe": agg["daily_sharpe"],
    }
    if trades.empty or segments.empty:
        return row
    row.update(
        {
            "trade_win_rate": (trades["pnl_pct"] > 0).mean(),
            "segment_win_rate": (segments["pnl_per_capital"] > 0).mean(),
            "worst_segment": segments["pnl_per_capital"].min(),
            "median_drawdown": segments["max_drawdown"].median(),
        }
    )
    return row
