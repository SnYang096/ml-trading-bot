from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LongShortBacktestConfig:
    """
    Simple cross-sectional long-short backtest on a single factor.

    - Each timestamp: rank factor cross-sectionally (pct rank).
    - Long: top quantile; Short: bottom quantile.
    - Returns: use provided target return column (e.g., future_return_12).
    - Transaction cost: apply fee_bps * turnover, where turnover is
      0.5 * sum_i |w_t(i) - w_{t-1}(i)|.
    """

    quantiles: int = 5
    fee_bps: float = 0.0
    min_assets: int = 3
    timestamp_level: int = 0
    symbol_level: int = 1


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


def long_short_backtest(
    panel: pd.DataFrame,
    *,
    factor_col: str,
    target_col: str,
    cfg: LongShortBacktestConfig = LongShortBacktestConfig(),
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Args:
        panel: MultiIndex DataFrame indexed by (timestamp, symbol) and containing
               factor_col and target_col.
    Returns:
        (timeseries_df, metrics)
    """
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel must be MultiIndex (timestamp, symbol)")
    if factor_col not in panel.columns:
        raise KeyError(f"Missing factor column: {factor_col}")
    if target_col not in panel.columns:
        raise KeyError(f"Missing target column: {target_col}")
    if cfg.quantiles < 2:
        raise ValueError("quantiles must be >= 2")

    df = panel[[factor_col, target_col]].dropna().copy()
    if df.empty:
        return pd.DataFrame(), {"error": 1.0}

    # Per timestamp backtest
    rows = []
    prev_w: Optional[pd.Series] = None

    for ts, g in df.groupby(level=cfg.timestamp_level, sort=True):
        if len(g) < cfg.min_assets:
            continue

        f = g[factor_col].astype(float)
        y = g[target_col].astype(float)
        valid = pd.concat([f, y], axis=1).dropna()
        if len(valid) < cfg.min_assets:
            continue

        # Deterministic quantile selection:
        # pick top-k and bottom-k by factor order to avoid pct-rank threshold edge cases.
        q = int(cfg.quantiles)
        n = int(len(valid))
        k = max(1, int(np.floor(n / q)))
        # ensure long/short don't overlap
        k = min(k, n // 2) if n >= 2 else 0
        if k <= 0:
            continue
        ordered = valid[factor_col].sort_values(ascending=True)
        short_syms = ordered.index[:k]
        long_syms = ordered.index[-k:]

        symbols = valid.index.get_level_values(cfg.symbol_level)
        w = pd.Series(0.0, index=symbols, dtype=float)
        # Use symbol-level index values for weight assignment.
        w.loc[long_syms.get_level_values(cfg.symbol_level)] = 1.0 / float(k)
        w.loc[short_syms.get_level_values(cfg.symbol_level)] = -1.0 / float(k)

        # Turnover vs previous weights
        turnover = 0.0
        if prev_w is not None:
            # Align union of symbols
            union = prev_w.index.union(w.index)
            w_u = w.reindex(union).fillna(0.0)
            prev_u = prev_w.reindex(union).fillna(0.0)
            turnover = 0.5 * float((w_u - prev_u).abs().sum())

        ret = float((w.values * valid[target_col].values).sum())
        fee = float(cfg.fee_bps) / 1e4 * turnover
        net = ret - fee

        rows.append(
            {
                "timestamp": ts,
                "gross_return": ret,
                "turnover": turnover,
                "fee": fee,
                "net_return": net,
                "n_assets": float(len(valid)),
                "n_long": float(k),
                "n_short": float(k),
            }
        )
        prev_w = w

    tsdf = pd.DataFrame(rows)
    if tsdf.empty:
        return tsdf, {"error": 1.0}
    tsdf["timestamp"] = pd.to_datetime(tsdf["timestamp"], utc=True, errors="coerce")
    tsdf = tsdf.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    tsdf["net_cum"] = tsdf["net_return"].fillna(0.0).cumsum()
    tsdf["gross_cum"] = tsdf["gross_return"].fillna(0.0).cumsum()

    ppy = _infer_periods_per_year(tsdf.index)
    metrics = {
        "n_timestamps": float(len(tsdf)),
        "avg_net_return": float(tsdf["net_return"].mean()),
        "avg_gross_return": float(tsdf["gross_return"].mean()),
        "avg_turnover": float(tsdf["turnover"].mean()),
        "fee_bps": float(cfg.fee_bps),
        "sharpe_net": _safe_sharpe(tsdf["net_return"], ppy),
        "sharpe_gross": _safe_sharpe(tsdf["gross_return"], ppy),
        "periods_per_year": float(ppy) if ppy else float("nan"),
    }
    return tsdf, metrics
