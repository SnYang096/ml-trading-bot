#!/usr/bin/env python3
"""Phase-2 joint sweep (spacing × box_pos × regime) + 4-segment validation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sweep_chop_oos_layers import (  # noqa: E402
    DEFAULT_YAML,
    _make_engine,
    _run_portfolio,
)
from scripts.diagnose_chop_grid import (  # noqa: E402
    GridConfig,
    build_features,
    merge_chop_grid_yaml,
    prefilter_box_range,
)
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv  # noqa: E402
from src.time_series_model.grid.subbar_replay import (  # noqa: E402
    merge_signal_features_onto_execution_bars,
    timeframe_to_timedelta,
)

BACKTEST = PROJECT_ROOT / "scripts" / "chop_grid_backtest.py"
SEGMENT_PATH = PROJECT_ROOT / "config" / "market_segment.yaml"


def _load_symbol_frames(
    symbols: Sequence[str],
    data_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    feat_base: GridConfig,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame | None, pd.Timedelta | None]]:
    warmup_start = start - pd.Timedelta(days=120)
    sig_delta = timeframe_to_timedelta("120T")
    out: Dict[str, Tuple[pd.DataFrame, pd.DataFrame | None, pd.Timedelta | None]] = {}
    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
        if raw.empty:
            continue
        bars_signal = _resample_ohlcv(raw, "120T")
        df = build_features(symbol, bars_signal, feat_base)
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if df.empty:
            continue
        bars_1m = _resample_ohlcv(raw, "1min")
        df_exec = merge_signal_features_onto_execution_bars(
            bars_1m, df, signal_bar_delta=sig_delta
        )
        out[symbol] = (df, df_exec, sig_delta)
    return out


def _feat_base_from_defaults(defaults: dict) -> GridConfig:
    return GridConfig(
        box_window=int(defaults.get("box_window", 120)),
        chop_min=float(defaults.get("chop_min", 0.50)),
        exit_chop_min=float(defaults.get("exit_chop_min", 0.32)),
        min_segment_bars=int(defaults.get("min_segment_bars", 6)),
        max_segment_bars=int(defaults.get("max_segment_bars", 120)),
        grid_atr_mult=float(defaults.get("grid_atr_mult", 1.0)),
        grid_pct=float(defaults.get("grid_pct", 0.010)),
        max_levels=int(defaults.get("max_levels", 2)),
        fee_bps=float(defaults.get("fee_bps", 20.0)),
        chop_signal=str(defaults.get("chop_signal", "raw")),
        chop_ts_window=int(defaults.get("chop_ts_window", 1200)),
        chop_ts_min_periods=int(defaults.get("chop_ts_min_periods", 150)),
        compute_semantic_chop_ts_q=True,
        feature_store_dir=defaults.get("feature_store_dir"),
        feature_store_layer=defaults.get("feature_store_layer"),
        feature_store_timeframe=defaults.get("feature_store_timeframe"),
        stability_min=float(defaults.get("stability_min", 0.85)),
        width_min=float(defaults.get("width_min", 0.04)),
        width_max=float(defaults.get("width_max", 0.30)),
        touches_min=int(defaults.get("touches_min", 5)),
        prefilter_rules=tuple(
            x
            for x in (defaults.get("prefilter_rules", []) or [])
            if isinstance(x, dict)
        ),
    )


def _validate_segments(
    *,
    candidate: dict,
    out_root: Path,
    symbols: str,
    config_yaml: str,
) -> List[dict]:
    data = yaml.safe_load(SEGMENT_PATH.read_text(encoding="utf-8")) or {}
    segments = [
        s
        for s in (data.get("segments") or [])
        if s.get("id")
        in {"bear_2022", "bull_2023_2024", "recent_range_to_bear", "recent_6m_oos"}
    ]
    rows: List[dict] = []
    tag = candidate["label"]
    for seg in segments:
        seg_id = str(seg["id"])
        seg_dir = out_root / tag / seg_id
        seg_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(BACKTEST),
            "--config",
            config_yaml,
            "--symbols",
            symbols,
            "--timeframe",
            "2h",
            "--execution-timeframe",
            "1min",
            "--start",
            str(seg["start_date"]),
            "--end",
            str(seg["end_date"]),
            "--grid-atr-mult",
            str(candidate["atr_mult"]),
            "--grid-pct",
            str(candidate["min_pct"]),
            "--chop-min",
            str(candidate["chop_min"]),
            "--exit-chop-min",
            str(candidate["exit_chop_min"]),
            "--chop-signal",
            str(candidate.get("chop_signal", "raw")),
            "--box-pos-min",
            str(candidate["box_lo"]),
            "--box-pos-max",
            str(candidate["box_hi"]),
            "--max-replenish-per-level",
            str(candidate.get("max_replenish", 1)),
            "--no-maps",
            "--out-dir",
            str(seg_dir),
        ]
        if candidate.get("block_stable_box"):
            cmd.append("--exclude-box")
        else:
            cmd.append("--no-exclude-box")
        print(f"\n=== validate {tag} / {seg_id} ===")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        summary = pd.read_csv(seg_dir / "summary.csv").iloc[0].to_dict()
        rows.append(
            {
                "label": tag,
                "segment": seg_id,
                "return_pct_timeline": summary.get("return_pct_timeline"),
                "trades": summary.get("trades"),
                "segment_win_rate": summary.get("segment_win_rate"),
                "max_drawdown_portfolio": summary.get("max_drawdown_portfolio"),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--config-yaml", default=str(DEFAULT_YAML))
    ap.add_argument(
        "--out-dir",
        default="results/chop_grid/experiments/oos_phase2_20260603",
    )
    ap.add_argument("--min-trades-oos", type=int, default=45)
    ap.add_argument("--top-k-validate", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    defaults = merge_chop_grid_yaml(Path(args.config_yaml))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    feat_base = _feat_base_from_defaults(defaults)
    symbol_frames = _load_symbol_frames(symbols, data_dir, start, end, feat_base)
    max_replenish = defaults.get("max_replenish_per_level")
    if max_replenish is not None:
        max_replenish = int(max_replenish)

    spacing_grid = [
        (1.15, 0.0105),
        (1.18, 0.0108),
        (1.20, 0.0110),
        (1.22, 0.0112),
        (1.25, 0.0120),
        (1.28, 0.0120),
    ]
    box_grid = [
        (0.35, 0.65),
        (0.37, 0.63),
        (0.38, 0.62),
        (0.40, 0.60),
        (0.42, 0.58),
    ]
    regime_grid = [
        ("raw", 0.50, 0.32),
        ("raw", 0.45, 0.30),
        ("raw", 0.52, 0.33),
        ("raw", 0.55, 0.35),
    ]

    rows: List[dict] = []
    for (atr_m, min_p), (box_lo, box_hi), (sig, cm, em) in product(
        spacing_grid, box_grid, regime_grid
    ):
        label = f"s{atr_m:.2f}_p{min_p:.3f}_b{box_lo:.2f}_{box_hi:.2f}_r{cm}_{em}"
        cfg = replace(
            feat_base,
            chop_min=cm,
            exit_chop_min=em,
            chop_signal=sig,
            grid_atr_mult=atr_m,
            grid_pct=min_p,
            prefilter_rules=prefilter_box_range(box_lo, box_hi),
        )
        engine = _make_engine(
            defaults,
            chop_min=cm,
            exit_chop_min=em,
            grid_atr_mult=atr_m,
            grid_pct=min_p,
            max_levels=int(defaults.get("max_levels", 2)),
            max_replenish=max_replenish,
        )
        m = _run_portfolio(
            symbols=list(symbol_frames),
            symbol_frames=symbol_frames,
            cfg=cfg,
            engine=engine,
            block_stable_box=False,
            exec_timeframe="1min",
            agg_data_dir=None,
            parquet_data_dir=data_dir,
        )
        row = {
            "label": label,
            "atr_mult": atr_m,
            "min_pct": min_p,
            "box_lo": box_lo,
            "box_hi": box_hi,
            "chop_signal": sig,
            "chop_min": cm,
            "exit_chop_min": em,
            **m,
        }
        rows.append(row)
        print(
            f"{label}: timeline={m.get('return_pct_timeline', 0):+.3f}% "
            f"trades={m.get('n_trades', 0)}"
        )

    oos_csv = out_dir / "joint_oos_sweep.csv"
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with oos_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    eligible = [r for r in rows if int(r.get("n_trades", 0)) >= args.min_trades_oos]
    eligible.sort(key=lambda r: float(r.get("return_pct_timeline", -999)), reverse=True)
    top = eligible[: args.top_k_validate]

    validate_rows: List[dict] = []
    for cand in top:
        validate_rows.extend(
            _validate_segments(
                candidate=cand,
                out_root=out_dir / "segment_validate",
                symbols=args.symbols,
                config_yaml=args.config_yaml,
            )
        )

    if validate_rows:
        val_df = pd.DataFrame(validate_rows)
        val_df.to_csv(out_dir / "segment_validate.csv", index=False)
        robust: List[dict] = []
        for label, grp in val_df.groupby("label"):
            timeline = grp["return_pct_timeline"].astype(float)
            robust.append(
                {
                    "label": label,
                    "min_segment_timeline": float(timeline.min()),
                    "sum_segment_timeline": float(timeline.sum()),
                    "all_segments_positive": bool((timeline > 0).all()),
                    "n_segments": len(grp),
                }
            )
        robust.sort(
            key=lambda r: (r["all_segments_positive"], r["min_segment_timeline"]),
            reverse=True,
        )
        (out_dir / "robust_rank.json").write_text(
            json.dumps(robust, indent=2), encoding="utf-8"
        )

    best_oos = (
        eligible[0] if eligible else max(rows, key=lambda r: r["return_pct_timeline"])
    )
    summary = {
        "oos_window": f"{args.start} → {args.end}",
        "n_joint_cells": len(rows),
        "min_trades_filter": args.min_trades_oos,
        "best_oos": best_oos,
        "top_oos_eligible": eligible[:10],
        "top_k_validated": [c["label"] for c in top],
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {oos_csv}")
    print(
        f"Best OOS (trades>={args.min_trades_oos}): {best_oos['label']} "
        f"timeline={best_oos.get('return_pct_timeline'):+.3f}%"
    )


if __name__ == "__main__":
    main()
