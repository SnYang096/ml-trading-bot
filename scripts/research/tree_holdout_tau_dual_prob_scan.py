#!/usr/bin/env python3
"""Holdout τ scan for independent long/short probability columns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_execution_layer import _identify_plateau
from scripts.research.tree_holdout_tau_rr_scan import _plateau_from_rows
from src.time_series_model.strategy_config.loader import StrategyConfigLoader


def _run_side_prob_bt(
    df: pd.DataFrame,
    strategy_config: Any,
    *,
    score_col: str,
    entry_threshold: float,
    side: str,
) -> dict[str, Any] | None:
    """Binary proba: enter when P(win|side) >= threshold (long_only / short_only)."""
    work = df.copy()
    work["pred"] = pd.to_numeric(work[score_col], errors="coerce")
    bt_params = dict(strategy_config.backtest.params or {})
    bt_params["entry_mode"] = "level"
    bt_params["use_signal_direction"] = True
    bt_params["strategy_direction"] = "long_only" if side == "long" else "short_only"
    bt_params["entry_threshold"] = float(entry_threshold)

    class _Cfg:
        enabled = True
        params = bt_params

    from scripts.train_strategy_pipeline import run_vectorbt_backtest

    out = run_vectorbt_backtest(
        work,
        work["pred"].to_numpy(dtype=float),
        _Cfg(),
        task_type="binary",
        strategy_config=strategy_config,
    )
    return out


def _scan_prob_column(
    df: pd.DataFrame,
    strategy_config: Any,
    *,
    score_col: str,
    side: str,
    quantile_grid: list[float],
) -> list[dict[str, Any]]:
    work = df.copy()
    if "_symbol" not in work.columns and "symbol" in work.columns:
        work["_symbol"] = work["symbol"]
    preds = pd.to_numeric(work[score_col], errors="coerce")
    symbols = (
        sorted(work["_symbol"].dropna().unique())
        if "_symbol" in work.columns
        else [None]
    )
    rows: list[dict[str, Any]] = []
    for q in quantile_grid:
        thr = float(np.nanquantile(preds, 1.0 - q))
        sharpes: list[float] = []
        rets: list[float] = []
        trades: list[int] = []
        for sym in symbols:
            part = work if sym is None else work[work["_symbol"] == sym]
            if part.empty:
                continue
            bt = _run_side_prob_bt(
                part,
                strategy_config,
                score_col=score_col,
                entry_threshold=thr,
                side=side,
            )
            if not bt:
                continue
            sh = bt.get("sharpe")
            if sh is not None and np.isfinite(sh):
                sharpes.append(float(sh))
            tr = bt.get("total_return_pct")
            if tr is not None and np.isfinite(tr):
                rets.append(float(tr))
            tt = bt.get("total_trades")
            if tt is not None:
                trades.append(int(tt))
        row = {
            "side": side,
            "top_quantile": q,
            "entry_threshold": thr,
            "sharpe": float(np.mean(sharpes)) if sharpes else float("nan"),
            "total_return_pct": float(np.mean(rets)) if rets else float("nan"),
            "total_trades": int(np.sum(trades)),
        }
        rows.append(row)
    return rows


def run_dual_tau_scan(
    *,
    config: str | Path,
    predictions: str | Path,
    output_dir: str | Path,
    quantile_grid: str = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
    start_date: str | None = "2025-10-01",
    end_date: str | None = "2026-04-01",
) -> dict[str, Any]:
    cfg_dir = Path(config)
    if not cfg_dir.is_absolute():
        cfg_dir = (PROJECT_ROOT / cfg_dir).resolve()
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(predictions)
    if not pred_path.is_absolute():
        pred_path = (PROJECT_ROOT / pred_path).resolve()
    raw = pd.read_parquet(pred_path)
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = raw.set_index("timestamp").sort_index()
    else:
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw = raw.sort_index()
    if start_date:
        raw = raw[raw.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        raw = raw[raw.index <= pd.Timestamp(end_date, tz="UTC")]
    df = raw.copy()
    if "atr" not in df.columns:
        from src.time_series_model.pipeline.training.label_utils import _ensure_atr

        atr = _ensure_atr(
            df.reset_index(),
            atr_col="atr",
            price_col="close",
            high_col="high",
            low_col="low",
            atr_window=14,
        )
        df["atr"] = atr.values
    if "signal" not in df.columns:
        df["signal"] = 0.0
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "holdout"].copy()
    if "score_long" not in df.columns or "score_short" not in df.columns:
        raise ValueError("predictions need score_long and score_short")

    strategy_config = StrategyConfigLoader(cfg_dir).load()
    q_grid = [float(x) for x in quantile_grid.split(",") if x.strip()]

    long_rows = _scan_prob_column(
        df, strategy_config, score_col="score_long", side="long", quantile_grid=q_grid
    )
    short_rows = _scan_prob_column(
        df, strategy_config, score_col="score_short", side="short", quantile_grid=q_grid
    )
    long_plateau = _plateau_from_rows(long_rows, "top_quantile")
    short_plateau = _plateau_from_rows(short_rows, "top_quantile")

    def _pick(rows: list[dict], plateau: dict) -> dict:
        rec = plateau.get("recommended") or plateau.get("best") or {}
        q = rec.get("top_quantile")
        row = next(
            (r for r in rows if r.get("top_quantile") == q), rows[0] if rows else {}
        )
        return {"top_quantile": q, "entry_threshold": row.get("entry_threshold")}

    result = {
        "long_scan": long_rows,
        "short_scan": short_rows,
        "long_plateau": long_plateau,
        "short_plateau": short_plateau,
        "recommended": {
            "long": _pick(long_rows, long_plateau),
            "short": _pick(short_rows, short_plateau),
        },
    }

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        raise TypeError(type(obj))

    out_json = out_dir / "tau_scan_holdout_dual_prob.json"
    out_json.write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"Wrote {out_json}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--quantile-grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50"
    )
    args = ap.parse_args()
    run_dual_tau_scan(
        config=args.config,
        predictions=args.predictions,
        output_dir=args.output_dir,
        quantile_grid=args.quantile_grid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
