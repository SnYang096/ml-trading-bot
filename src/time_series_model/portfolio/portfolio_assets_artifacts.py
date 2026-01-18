from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from .portfolio_assets import (
    aggregate_from_symbol_modes,
    compute_portfolio_asset_weights,
    load_portfolio_assets_config,
    trend_zero_law_status,
)


@dataclass(frozen=True)
class PortfolioAssetsArtifacts:
    summary: Dict[str, Any]
    timeseries_tail: pd.DataFrame


def build_portfolio_assets_artifacts_from_modes(
    df: pd.DataFrame,
    *,
    portfolio_assets_yaml: str,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    mode_col: str = "mode",
    key_symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    tail_points: int = 240,
    gate_veto: bool = False,
    portfolio_drawdown: float = 0.0,
) -> PortfolioAssetsArtifacts:
    """
    Build diagnostic artifacts by aggregating per-symbol Router modes over timestamps
    and mapping to deterministic portfolio asset weights.

    This function is intentionally pure (no file I/O) so it's easy to unit test.
    """
    if df is None or len(df) == 0:
        return PortfolioAssetsArtifacts(summary={}, timeseries_tail=pd.DataFrame())
    if not portfolio_assets_yaml:
        return PortfolioAssetsArtifacts(summary={}, timeseries_tail=pd.DataFrame())
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp_col={timestamp_col}")
    if symbol_col not in df.columns:
        raise ValueError(f"Missing symbol_col={symbol_col}")
    if mode_col not in df.columns:
        raise ValueError(f"Missing mode_col={mode_col}")

    pa_cfg = load_portfolio_assets_config(portfolio_assets_yaml)

    work = df.reset_index(drop=True).copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work = (
        work.dropna(subset=[timestamp_col])
        .sort_values(timestamp_col)
        .reset_index(drop=True)
    )

    rows = []
    for ts, g in work.groupby(timestamp_col, sort=True):
        decisions = []
        for r in g.to_dict(orient="records"):
            sym = str(r.get(symbol_col, "")).strip()
            mode = str(r.get(mode_col, "NO_TRADE"))
            if not sym:
                continue
            decisions.append({"symbol": sym, "mode": mode})
        if not decisions:
            continue

        sig = aggregate_from_symbol_modes(
            decisions=decisions, key_symbols=list(key_symbols)
        )
        tz = trend_zero_law_status(
            cfg=pa_cfg,
            sig=sig,
            gate_veto=bool(gate_veto),
            portfolio_drawdown=float(portfolio_drawdown),
        )
        w = compute_portfolio_asset_weights(
            cfg=pa_cfg,
            sig=sig,
            gate_veto=bool(gate_veto),
            portfolio_drawdown=float(portfolio_drawdown),
        )
        rows.append(
            {
                "timestamp": ts,
                **sig.as_dict(),
                "trend_zero_triggered": bool(tz.get("triggered", False)),
                "trend_zero_reasons": "|".join(
                    [str(x) for x in (tz.get("reasons") or [])]
                ),
                **{f"w__{k}": float(v) for k, v in (w or {}).items()},
            }
        )

    df_pa = pd.DataFrame(rows)
    if df_pa.empty:
        return PortfolioAssetsArtifacts(summary={}, timeseries_tail=pd.DataFrame())

    weight_cols = [c for c in df_pa.columns if c.startswith("w__")]
    avg_weights = (
        df_pa[weight_cols].mean(numeric_only=True).to_dict() if weight_cols else {}
    )
    summary: Dict[str, Any] = {
        "portfolio_assets_yaml": str(portfolio_assets_yaml),
        "name": str(getattr(pa_cfg, "name", "portfolio_assets")),
        "n_timestamps": int(df_pa.shape[0]),
        "trend_zero_rate": float(df_pa["trend_zero_triggered"].mean()),
        "avg_weights": {k.replace("w__", ""): float(v) for k, v in avg_weights.items()},
        "key_symbols": list(key_symbols),
        "gate_veto": bool(gate_veto),
        "portfolio_drawdown": float(portfolio_drawdown),
    }

    tail_n = int(max(0, tail_points))
    df_tail = (
        df_pa.tail(tail_n).reset_index(drop=True) if tail_n > 0 else pd.DataFrame()
    )
    return PortfolioAssetsArtifacts(summary=summary, timeseries_tail=df_tail)
