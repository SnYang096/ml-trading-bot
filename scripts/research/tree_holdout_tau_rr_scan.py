#!/usr/bin/env python3
"""Holdout τ (top_quantile / pred threshold) scan + RR vectorbt for tree regression."""

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
from scripts.train_strategy_pipeline import run_vectorbt_backtest
from src.time_series_model.pipeline.training.label_utils import _ensure_atr
from src.time_series_model.strategy_config.loader import StrategyConfigLoader


def _prepare_df(
    predictions_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(predictions_path)
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "holdout"].copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    if "atr" not in df.columns:
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
    return df


def _predict_segment(
    *,
    artifact_dir: Path,
    config_dir: Path,
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    data_path: str,
    feature_store_layer: str,
) -> pd.DataFrame:
    from scripts.train_strategy_pipeline import (
        _ensure_ticks_configured,
        generate_predictions,
        run_feature_pipeline,
    )
    from src.data_tools.data_handler import DataHandler
    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
    from src.time_series_model.strategies.models.model_artifact import ModelArtifact

    artifact = ModelArtifact.load(artifact_dir)
    strategy_config = StrategyConfigLoader(config_dir).load()
    feature_loader = StrategyFeatureLoader()
    data_handler = DataHandler(data_path=data_path)

    parts: list[pd.DataFrame] = []
    for sym in symbols:
        df_raw = data_handler.load_ohlcv(symbol=sym, timeframe=timeframe)
        df_raw = df_raw.loc[start_date:end_date]
        if df_raw.empty:
            continue
        df_raw = df_raw.copy()
        df_raw["_symbol"] = sym
        df_raw["symbol"] = sym
        start_ts = str(df_raw.index.min())
        end_ts = str(df_raw.index.max())
        requested = strategy_config.features.requested_features or []
        _ensure_ticks_configured(
            feature_loader,
            sym,
            data_path,
            start_ts,
            end_ts,
            requested,
        )
        df_feat = run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_config.features,
            fit=False,
            feature_store_dir="feature_store",
            feature_store_layer=feature_store_layer,
            feature_store_symbol=sym,
            feature_store_timeframe=timeframe,
        )
        model_obj = artifact.model
        models = model_obj if isinstance(model_obj, list) else [model_obj]
        preds = generate_predictions(
            models,
            model_type=strategy_config.model.trainer.params.get(
                "model_type", "lightgbm"
            ),
            task_type=strategy_config.model.trainer.params.get(
                "task_type", "regression"
            ),
            X=artifact.preprocessor.transform(
                df_feat, feature_cols=artifact.used_features
            ),
        )
        out = df_feat.copy()
        out["pred"] = np.asarray(preds, dtype=float)
        out["split"] = "segment"
        if "timestamp" not in out.columns:
            out = out.reset_index().rename(columns={"index": "timestamp"})
        parts.append(out)

    if not parts:
        raise ValueError(f"No rows for segment {start_date}→{end_date}")
    merged = pd.concat(parts, axis=0, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged = merged.set_index("timestamp").sort_index()
    if "atr" not in merged.columns:
        atr = _ensure_atr(
            merged.reset_index(),
            atr_col="atr",
            price_col="close",
            high_col="high",
            low_col="low",
            atr_window=14,
        )
        merged["atr"] = atr.values
    if "signal" not in merged.columns:
        merged["signal"] = 0.0
    return merged


def _run_bt(
    df: pd.DataFrame,
    strategy_config,
    *,
    top_quantile: float | None = None,
    bottom_quantile: float | None = None,
    long_entry_threshold: float | None = None,
    short_entry_threshold: float | None = None,
    entry_mode: str = "cross",
) -> dict[str, Any] | None:
    bt_params = dict(strategy_config.backtest.params or {})
    if top_quantile is not None:
        bt_params["top_quantile"] = float(top_quantile)
    if bottom_quantile is not None:
        bt_params["bottom_quantile"] = float(bottom_quantile)
    if long_entry_threshold is not None:
        bt_params["long_entry_threshold"] = float(long_entry_threshold)
    if short_entry_threshold is not None:
        bt_params["short_entry_threshold"] = float(short_entry_threshold)
    bt_params["entry_mode"] = entry_mode
    bt_params["entry_exit_conflict"] = bt_params.get(
        "entry_exit_conflict", "block_entry_on_exit"
    )
    # RR path requires use_signal_direction=True for bidirectional strategies (validation only).
    bt_params["use_signal_direction"] = True

    preds = df["pred"].to_numpy(dtype=float)
    # Absolute-threshold mode: bypass regression quantile branch via binary thresholds.
    task_type = "regression"
    if long_entry_threshold is not None or short_entry_threshold is not None:
        task_type = "binary"

    class _Cfg:
        enabled = True
        params = bt_params

    return run_vectorbt_backtest(
        df,
        preds,
        _Cfg(),
        task_type=task_type,
        strategy_config=strategy_config,
    )


def _scan_quantile(
    df: pd.DataFrame,
    strategy_config,
    grid: list[float],
    *,
    per_symbol: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    symbols = (
        sorted(df["_symbol"].dropna().unique())
        if per_symbol and "_symbol" in df.columns
        else [None]
    )
    for q in grid:
        sharpes: list[float] = []
        rets: list[float] = []
        trades: list[int] = []
        for sym in symbols:
            part = df if sym is None else df[df["_symbol"] == sym]
            if part.empty:
                continue
            bt = _run_bt(part, strategy_config, top_quantile=q, bottom_quantile=q)
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
        if not sharpes:
            rows.append(
                {
                    "scan": "top_bottom_quantile",
                    "top_quantile": q,
                    "bottom_quantile": q,
                    "sharpe": float("nan"),
                    "total_return_pct": float("nan"),
                    "total_trades": 0,
                    "pred_threshold_long": float(df["pred"].quantile(1 - q)),
                    "pred_threshold_short": float(df["pred"].quantile(q)),
                }
            )
            continue
        rows.append(
            {
                "scan": "top_bottom_quantile",
                "top_quantile": q,
                "bottom_quantile": q,
                "sharpe": float(np.mean(sharpes)),
                "total_return_pct": float(np.mean(rets)) if rets else float("nan"),
                "total_trades": int(np.sum(trades)),
                "pred_threshold_long": float(df["pred"].quantile(1 - q)),
                "pred_threshold_short": float(df["pred"].quantile(q)),
            }
        )
    return rows


def _scan_pred_threshold(
    df: pd.DataFrame,
    strategy_config,
    grid: list[float],
    *,
    per_symbol: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    symbols = (
        sorted(df["_symbol"].dropna().unique())
        if per_symbol and "_symbol" in df.columns
        else [None]
    )
    for thr in grid:
        sharpes: list[float] = []
        rets: list[float] = []
        trades: list[int] = []
        for sym in symbols:
            part = df if sym is None else df[df["_symbol"] == sym]
            if part.empty:
                continue
            bt = _run_bt(
                part,
                strategy_config,
                long_entry_threshold=thr,
                short_entry_threshold=thr,
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
        rows.append(
            {
                "scan": "pred_long_ge",
                "pred_threshold": thr,
                "sharpe": float(np.mean(sharpes)) if sharpes else float("nan"),
                "total_return_pct": float(np.mean(rets)) if rets else float("nan"),
                "total_trades": int(np.sum(trades)),
            }
        )
    return rows


def _plateau_from_rows(rows: list[dict[str, Any]], param_name: str) -> dict[str, Any]:
    valid = [
        r
        for r in rows
        if r.get("sharpe") is not None and np.isfinite(r.get("sharpe", np.nan))
    ]
    if not valid:
        return {"is_plateau": False, "reason": "no_valid_sharpe"}
    payload = []
    for r in valid:
        item = dict(r)
        item["sharpe"] = float(r["sharpe"])
        item[param_name] = r.get(param_name, r.get("top_quantile"))
        payload.append(item)
    vals = [[float(x[param_name])] for x in payload]
    return _identify_plateau(
        payload,
        param_names=[param_name],
        param_values=list(zip(*vals)),
    )


def run_tau_scan(
    *,
    config: str | Path,
    output_dir: str | Path,
    predictions: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: str | list[str] = "BTCUSDT,ETHUSDT",
    timeframe: str = "120T",
    data_path: str = "data/parquet_data",
    feature_store_layer: str | None = None,
    fixed_quantile: float | None = None,
    segment_label: str = "holdout",
    quantile_grid: str = "0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30",
    pred_grid: str | None = None,
    per_symbol: bool = True,
    filter_split: str | None = "holdout",
    project_root: Path | None = None,
) -> dict[str, Path]:
    """Run holdout τ quantile scan + per-symbol RR backtest."""
    root = project_root or PROJECT_ROOT
    if not predictions and not artifact_dir:
        raise ValueError("Provide predictions or artifact_dir")
    if artifact_dir and not feature_store_layer:
        raise ValueError("artifact_dir requires feature_store_layer")

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dir = Path(config)
    if not cfg_dir.is_absolute():
        cfg_dir = (root / cfg_dir).resolve()
    strategy_config = StrategyConfigLoader(cfg_dir).load()

    if artifact_dir:
        if not start_date or not end_date:
            raise ValueError("artifact_dir requires start_date and end_date")
        if isinstance(symbols, list):
            sym_list = [str(s) for s in symbols]
        else:
            sym_list = [s.strip() for s in str(symbols).split(",") if s.strip()]
        art = Path(artifact_dir)
        if not art.is_absolute():
            art = (root / art).resolve()
        df = _predict_segment(
            artifact_dir=art,
            config_dir=cfg_dir,
            symbols=sym_list,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            data_path=data_path,
            feature_store_layer=str(feature_store_layer),
        )
        pred_path = out_dir / "predictions_segment.parquet"
        df.reset_index().to_parquet(pred_path, index=False)
    else:
        pred = Path(predictions)
        if not pred.is_absolute():
            pred = (root / pred).resolve()
        df = _prepare_df(pred, start_date=start_date, end_date=end_date)
        if filter_split and "split" in df.columns:
            df = df.reset_index()
            df = df[df["split"].astype(str).str.lower() == filter_split.lower()].copy()
            if "timestamp" in df.columns:
                df = df.set_index("timestamp").sort_index()
        pred_path = pred

    q_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    q_plateau: dict[str, Any] = {}

    if fixed_quantile is None:
        q_grid = [float(x) for x in quantile_grid.split(",") if x.strip()]
        q_rows = _scan_quantile(df, strategy_config, q_grid, per_symbol=per_symbol)
        if pred_grid:
            p_grid = [float(x) for x in pred_grid.split(",") if x.strip()]
            pred_rows = _scan_pred_threshold(
                df, strategy_config, p_grid, per_symbol=per_symbol
            )
        q_plateau = _plateau_from_rows(q_rows, "top_quantile")
        best_q = (
            q_plateau.get("recommended")
            or q_plateau.get("best")
            or max(q_rows, key=lambda r: (r.get("sharpe") or -999))
        )
        rec_q = float(best_q.get("top_quantile", 0.10))
    else:
        rec_q = float(fixed_quantile)

    final: dict[str, Any] = {
        "segment": segment_label,
        "date_range": {"start": start_date, "end": end_date},
        "quantile_scan": q_rows,
        "quantile_plateau": q_plateau,
    }
    if pred_rows:
        final["pred_threshold_scan"] = pred_rows
        final["pred_plateau"] = _plateau_from_rows(pred_rows, "pred_threshold")

    final_bt_by_symbol: dict[str, Any] = {}
    if "_symbol" in df.columns:
        for sym in sorted(df["_symbol"].dropna().unique()):
            part = df[df["_symbol"] == sym]
            bt = _run_bt(
                part, strategy_config, top_quantile=rec_q, bottom_quantile=rec_q
            )
            if bt:
                final_bt_by_symbol[str(sym)] = bt
    else:
        bt = _run_bt(df, strategy_config, top_quantile=rec_q, bottom_quantile=rec_q)
        if bt:
            final_bt_by_symbol["ALL"] = bt

    final["recommended"] = {
        "top_quantile": rec_q,
        "bottom_quantile": rec_q,
        "pred_threshold_long": float(df["pred"].quantile(1 - rec_q)),
        "pred_threshold_short": float(df["pred"].quantile(rec_q)),
    }
    final["holdout_rr_backtest"] = final_bt_by_symbol

    json_path = out_dir / "tau_scan_holdout_rr.json"
    json_path.write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")

    md_lines = [
        f"# tree {segment_label} τ / RR backtest",
        "",
        f"- segment: {segment_label}",
        f"- rows: {len(df)}",
        f"- symbols: {sorted(df['_symbol'].unique()) if '_symbol' in df.columns else ['ALL']}",
    ]
    if start_date or end_date:
        md_lines.append(f"- dates: {start_date} → {end_date}")
    md_lines.append(f"- predictions: `{pred_path}`")
    md_lines.extend(["", "## Quantile scan (top/bottom = q)", ""])
    if q_rows:
        md_lines.extend(
            [
                "| q | pred_long≥ | pred_short≤ | Sharpe | Return% | Trades |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for r in q_rows:
            md_lines.append(
                f"| {r['top_quantile']:.2f} | {r['pred_threshold_long']:.4f} | "
                f"{r['pred_threshold_short']:.4f} | {r.get('sharpe', float('nan')):.3f} | "
                f"{r.get('total_return_pct', float('nan')):.2f} | {r.get('total_trades', 0)} |"
            )
    else:
        md_lines.append(f"_Scan skipped; fixed q={rec_q:.2f}_")
    md_lines.extend(
        [
            "",
            f"**Recommended q**: {rec_q:.2f} "
            f"(long≥{final['recommended']['pred_threshold_long']:.4f}, "
            f"short≤{final['recommended']['pred_threshold_short']:.4f})",
            "",
            "## Holdout RR backtest @ recommended τ",
            "",
        ]
    )
    for sym, bt in final_bt_by_symbol.items():
        md_lines.append(
            f"- **{sym}**: Sharpe={bt.get('sharpe')}, "
            f"Return={bt.get('total_return_pct')}%, trades={bt.get('total_trades')}, "
            f"win_rate={bt.get('win_rate')}"
        )
    md_path = out_dir / "tau_scan_holdout_rr.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return {"json": json_path, "md": md_path}


def main() -> int:
    ap = argparse.ArgumentParser(description="Tree holdout τ scan + RR backtest")
    ap.add_argument("--config", required=True, help="Strategy config dir")
    ap.add_argument("--predictions", default=None, help="predictions.parquet path")
    ap.add_argument(
        "--artifact-dir", default=None, help="ModelArtifact dir for segment predict"
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--data-path", default="data/parquet_data")
    ap.add_argument("--feature-store-layer", default=None)
    ap.add_argument(
        "--fixed-quantile",
        type=float,
        default=None,
        help="Skip scan; RR backtest at this q only",
    )
    ap.add_argument("--segment-label", default="holdout")
    ap.add_argument(
        "--quantile-grid",
        default="0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30",
    )
    ap.add_argument(
        "--pred-grid", default=None, help="Optional absolute pred thresholds"
    )
    ap.add_argument("--per-symbol", action="store_true", default=True)
    ap.add_argument(
        "--no-filter-split",
        action="store_true",
        help="Do not filter predictions to split=holdout",
    )
    args = ap.parse_args()

    if not args.predictions and not args.artifact_dir:
        ap.error("Provide --predictions or --artifact-dir")
    if args.artifact_dir and not args.feature_store_layer:
        ap.error("--artifact-dir requires --feature-store-layer")

    run_tau_scan(
        config=args.config,
        output_dir=args.output_dir,
        predictions=args.predictions,
        artifact_dir=args.artifact_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols,
        timeframe=args.timeframe,
        data_path=args.data_path,
        feature_store_layer=args.feature_store_layer,
        fixed_quantile=args.fixed_quantile,
        segment_label=args.segment_label,
        quantile_grid=args.quantile_grid,
        pred_grid=args.pred_grid,
        per_symbol=args.per_symbol,
        filter_split=None if args.no_filter_split else "holdout",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
