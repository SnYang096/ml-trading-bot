#!/usr/bin/env python3
"""
[DEPRECATED] Physics / Regime Classifier Script

⚠️ DEPRECATED: Regime classification has been migrated to gate rules in execution_archetypes.yaml.
Physical features (path_efficiency_pct, jump_risk_pct, etc.) are now computed in FeatureStore
and checked directly by gate rules. This script is kept for backward compatibility and diagnostics only.

Classifies each timestamp into Physics/Regime categories:
- TC_REGIME: Trend Continuation regime
- TE_REGIME: Trend Expansion regime
- MEAN_REGIME: Extreme Mean Reversion regime
- NO_TRADE: No viable execution regime

Also computes Symbol × Regime frequency statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.rule.regime import (
    PhysicsRegimeConfig,
    classify_regime,
)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_pred_files(preds_path: Path) -> List[Path]:
    if preds_path.is_dir():
        files = sorted(preds_path.glob("preds_*.parquet"))
        if not files:
            files = sorted(preds_path.glob("*.parquet"))
        return files
    return [preds_path]


def _ensure_timestamp_col(df: pd.DataFrame, *, col: str = "timestamp") -> pd.DataFrame:
    if df.index.name == col:
        df = df.reset_index()
    if col in df.columns:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out[col] = out.index
        return out
    return df


def _load_feature_store(
    *,
    feature_store_root: Path,
    layer: str,
    symbol: str,
    timeframe: str,
    columns: List[str],
) -> pd.DataFrame:
    """Load required features from feature store."""
    sym_dir = feature_store_root / layer / symbol / timeframe
    if not sym_dir.exists():
        raise FileNotFoundError(f"FeatureStore path not found: {sym_dir}")
    frames = []
    for p in sorted(sym_dir.glob("*.parquet")):
        df = pd.read_parquet(p)
        if "timestamp" not in df.columns:
            if df.index.name == "timestamp":
                df = df.reset_index()
            else:
                raise ValueError(f"timestamp column missing in {p}")
        keep = ["timestamp"] + [c for c in columns if c in df.columns]
        frames.append(df[keep])
    if not frames:
        raise ValueError(f"No featurestore parquet found under {sym_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")


def _extract_symbol_from_path(path: Path) -> Optional[str]:
    """Extract symbol from preds file path (e.g., preds_BTCUSDT.parquet -> BTCUSDT)."""
    name = path.stem
    if name.startswith("preds_"):
        return name[6:]  # Remove "preds_" prefix
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Physics/Regime classifier for market execution feasibility."
    )
    p.add_argument(
        "--preds",
        required=True,
        help="Preds file (.parquet/.csv) or directory of per-symbol preds_*.parquet files",
    )
    p.add_argument(
        "--feature-store-root",
        type=Path,
        help="Feature store root directory",
    )
    p.add_argument(
        "--layer",
        default="tier0",
        help="Feature store layer (default: tier0)",
    )
    p.add_argument(
        "--timeframe",
        default="240T",
        help="Timeframe (default: 240T)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output parquet/csv file path",
    )
    p.add_argument(
        "--stats-output",
        type=Path,
        help="Optional: output Symbol × Regime frequency statistics JSON",
    )
    p.add_argument(
        "--scan-physics-score-pct",
        default=None,
        help="Comma-separated physics_score_min_pct values for scan (e.g., 0.8,0.85,0.9).",
    )
    p.add_argument(
        "--scan-output",
        type=Path,
        default=None,
        help="Output JSON path for physics_score_min_pct scan report.",
    )
    p.add_argument(
        "--scan-md-output",
        type=Path,
        default=None,
        help="Optional markdown output for scan summary table.",
    )
    p.add_argument(
        "--kpi-output",
        type=Path,
        default=None,
        help="Output JSON path for Physics KPI report.",
    )
    p.add_argument(
        "--kpi-md-output",
        type=Path,
        default=None,
        help="Optional markdown output for Physics KPI report.",
    )
    p.add_argument(
        "--kpi-mfe-quantile",
        type=float,
        default=0.9,
        help="Quantile threshold for high-MFE recall KPI.",
    )
    p.add_argument(
        "--kpi-mae-quantile",
        type=float,
        default=0.95,
        help="Quantile threshold for tail MAE safety KPI.",
    )
    return p.parse_args()


def compute_regime_statistics(
    df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    regime_col: str = "regime",
    timestamp_col: str = "timestamp",
    include_duration: bool = True,
) -> pd.DataFrame:
    """
    Compute Symbol × Regime frequency statistics.

    ⚠️ IMPORTANT: This is Step 1 - time distribution only.
    Do NOT compute Sharpe here (that's Step 2, conditional on Regime).

    Returns DataFrame with columns:
    - symbol
    - regime
    - count
    - frequency (as percentage)
    - mean_duration (optional, in bars)
    - max_duration (optional, in bars)
    """
    if symbol_col not in df.columns:
        raise ValueError(f"Symbol column '{symbol_col}' not found")
    if regime_col not in df.columns:
        raise ValueError(f"Regime column '{regime_col}' not found")

    stats = df.groupby([symbol_col, regime_col]).size().reset_index(name="count")

    total_counts = df.groupby(symbol_col).size().reset_index(name="total")
    stats = stats.merge(total_counts, on=symbol_col)
    stats["frequency"] = stats["count"] / stats["total"] * 100.0

    # Compute continuous Regime duration (Step 1 requirement)
    if include_duration and timestamp_col in df.columns:
        df_sorted = df.sort_values([symbol_col, timestamp_col])
        duration_stats = []

        for symbol in df_sorted[symbol_col].unique():
            sym_df = df_sorted[df_sorted[symbol_col] == symbol].copy()
            sym_df["regime_change"] = sym_df[regime_col] != sym_df[regime_col].shift(1)
            sym_df["regime_group"] = sym_df["regime_change"].cumsum()

            for regime in sym_df[regime_col].unique():
                regime_df = sym_df[sym_df[regime_col] == regime]
                durations = regime_df.groupby("regime_group").size()

                duration_stats.append(
                    {
                        symbol_col: symbol,
                        regime_col: regime,
                        "mean_duration": durations.mean(),
                        "max_duration": durations.max(),
                        "min_duration": durations.min(),
                    }
                )

        if duration_stats:
            duration_df = pd.DataFrame(duration_stats)
            stats = stats.merge(duration_df, on=[symbol_col, regime_col], how="left")

    return stats.sort_values([symbol_col, regime_col])


def main() -> int:
    args = parse_args()

    preds_path = Path(args.preds)
    pred_files = _collect_pred_files(preds_path)

    if not pred_files:
        print(f"Error: No preds files found in {preds_path}", file=sys.stderr)
        return 1

    all_results = []
    scan_data = []

    # Required features for Physics/Regime classification
    required_features = [
        "atr",
        "atr_percentile",
        "high",
        "low",
        "close",
    ]

    for pred_file in pred_files:
        print(f"Processing {pred_file.name}...", file=sys.stderr)

        # Load preds
        preds_df = _read_any(pred_file)
        preds_df = _ensure_timestamp_col(preds_df)

        # Extract symbol
        symbol = _extract_symbol_from_path(pred_file)
        if symbol is None and "symbol" in preds_df.columns:
            symbol = preds_df["symbol"].iloc[0]

        # Load features if feature store is provided
        if args.feature_store_root:
            try:
                features_df = _load_feature_store(
                    feature_store_root=args.feature_store_root,
                    layer=args.layer,
                    symbol=symbol or "BTCUSDT",  # Fallback
                    timeframe=args.timeframe,
                    columns=required_features,
                )
                # Merge with preds
                preds_df = preds_df.merge(
                    features_df,
                    on="timestamp",
                    how="inner",
                    suffixes=("", "_fs"),
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load features for {pred_file.name}: {e}",
                    file=sys.stderr,
                )
                continue
        else:
            # Check if features are already in preds_df
            missing = [f for f in required_features if f not in preds_df.columns]
            if missing:
                print(
                    f"Warning: Missing features {missing} for {pred_file.name}",
                    file=sys.stderr,
                )
                continue

        # Store minimal scan data if requested
        if args.scan_physics_score_pct:
            scan_cols = [
                "timestamp",
                "pred_dir_prob",
                "atr",
                "atr_percentile",
                "high",
                "low",
                "close",
            ]
            scan_df = preds_df.copy()
            for col in scan_cols:
                if col not in scan_df.columns:
                    print(
                        f"Warning: Missing {col} for scan in {pred_file.name}",
                        file=sys.stderr,
                    )
                    scan_df = None
                    break
            if scan_df is not None:
                if "symbol" not in scan_df.columns:
                    scan_df["symbol"] = symbol or "UNKNOWN"
                scan_data.append((symbol or "UNKNOWN", scan_df[scan_cols + ["symbol"]]))

        # Classify Physics/Regime
        cfg = PhysicsRegimeConfig()
        regime_df = classify_regime(preds_df, cfg=cfg)

        # Merge back
        result_df = preds_df[["timestamp"]].copy()
        if "symbol" in preds_df.columns:
            result_df["symbol"] = preds_df["symbol"]
        elif symbol:
            result_df["symbol"] = symbol

        result_df = result_df.join(regime_df)

        all_results.append(result_df)

    if not all_results:
        print("Error: No valid results generated", file=sys.stderr)
        return 1

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Ensure timestamp is datetime
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
    combined_df = combined_df.sort_values(["symbol", "timestamp"])

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        combined_df.to_parquet(output_path, index=False)
    else:
        combined_df.to_csv(output_path, index=False)

    print(f"✅ Saved regime classifications to {output_path}", file=sys.stderr)

    # Compute and save statistics
    if args.stats_output:
        stats_df = compute_regime_statistics(combined_df)

        # Format as nested dict for JSON
        stats_dict = {}
        for _, row in stats_df.iterrows():
            sym = row["symbol"]
            regime = row["regime"]
            if sym not in stats_dict:
                stats_dict[sym] = {}
            stats_dict[sym][regime] = {
                "count": int(row["count"]),
                "frequency_pct": float(row["frequency"]),
            }

        stats_output_path = Path(args.stats_output)
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_output_path, "w") as f:
            json.dump(stats_dict, f, indent=2)

        print(f"✅ Saved regime statistics to {stats_output_path}", file=sys.stderr)

        # Print summary
        print("\n=== Symbol × Regime Frequency Statistics ===", file=sys.stderr)
        for symbol in sorted(stats_dict.keys()):
            print(f"\n{symbol}:", file=sys.stderr)
            for regime in ["TC_REGIME", "TE_REGIME", "MEAN_REGIME", "NO_TRADE"]:
                if regime in stats_dict[symbol]:
                    freq = stats_dict[symbol][regime]["frequency_pct"]
                    count = stats_dict[symbol][regime]["count"]
                    print(f"  {regime}: {freq:.2f}% ({count} samples)", file=sys.stderr)

    # Optional scan over physics_score_min_pct
    if args.scan_physics_score_pct and scan_data:
        scan_values = [
            float(x.strip())
            for x in args.scan_physics_score_pct.split(",")
            if x.strip()
        ]
        scan_report = {"thresholds": {}, "decision_rule": {}}
        scan_rows = []

        for pct in scan_values:
            cfg_scan = PhysicsRegimeConfig(physics_score_min_pct=pct)
            total_counts = {
                "NO_TRADE": 0,
                "TC_REGIME": 0,
                "TE_REGIME": 0,
                "MEAN_REGIME": 0,
            }
            total_rows = 0
            per_symbol_tc_te = []

            for symbol, df_scan in scan_data:
                out_scan = classify_regime(df_scan, cfg=cfg_scan)
                counts = out_scan["regime"].value_counts()
                total_rows += len(out_scan)
                for w in total_counts:
                    total_counts[w] += int(counts.get(w, 0))

                tc_te = float(
                    counts.get("TC_REGIME", 0) + counts.get("TE_REGIME", 0)
                ) / max(len(out_scan), 1)
                per_symbol_tc_te.append(tc_te)

            row = {
                "physics_score_min_pct": pct,
                "NO_TRADE": total_counts["NO_TRADE"] / max(total_rows, 1),
                "TC_REGIME": total_counts["TC_REGIME"] / max(total_rows, 1),
                "TE_REGIME": total_counts["TE_REGIME"] / max(total_rows, 1),
                "MEAN_REGIME": total_counts["MEAN_REGIME"] / max(total_rows, 1),
                "TC_TE": (total_counts["TC_REGIME"] + total_counts["TE_REGIME"])
                / max(total_rows, 1),
                "tc_te_std": (
                    float(pd.Series(per_symbol_tc_te).std(ddof=0))
                    if per_symbol_tc_te
                    else 0.0
                ),
            }
            scan_rows.append(row)

        # Decision rule (deterministic)
        eligible = [
            r
            for r in scan_rows
            if (
                r["NO_TRADE"] >= 0.70
                and 0.02 <= r["TC_TE"] <= 0.08
                and r["TE_REGIME"] >= 0.005
            )
        ]
        recommended = (
            sorted(eligible, key=lambda r: r["physics_score_min_pct"])[-1]
            if eligible
            else None
        )

        scan_report["thresholds"] = {
            str(r["physics_score_min_pct"]): r for r in scan_rows
        }
        scan_report["decision_rule"] = {
            "no_trade_min": 0.70,
            "tc_te_range": [0.02, 0.08],
            "te_min": 0.005,
            "recommended": recommended,
        }

        if args.scan_output:
            args.scan_output.parent.mkdir(parents=True, exist_ok=True)
            args.scan_output.write_text(
                json.dumps(scan_report, indent=2), encoding="utf-8"
            )

        if args.scan_md_output:
            args.scan_md_output.parent.mkdir(parents=True, exist_ok=True)
            header = "| physics_score_min_pct | NO_TRADE | TC_REGIME | TE_REGIME | MEAN_REGIME | TC+TE | tc_te_std |\n"
            sep = "|---|---|---|---|---|---|---|\n"
            lines = [header, sep]
            for r in scan_rows:
                lines.append(
                    f"| {r['physics_score_min_pct']:.2f} | {r['NO_TRADE']:.2%} | {r['TC_REGIME']:.2%} | "
                    f"{r['TE_REGIME']:.2%} | {r['MEAN_REGIME']:.2%} | {r['TC_TE']:.2%} | {r['tc_te_std']:.2%} |\n"
                )
            if recommended:
                lines.append(
                    f"\n**Recommended:** {recommended['physics_score_min_pct']:.2f}\n"
                )
            args.scan_md_output.write_text("".join(lines), encoding="utf-8")

    # Physics KPI report (recall/safety/frequency)
    if args.kpi_output or args.kpi_md_output:

        def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        mfe_col = _pick_col(combined_df, ["head_mfe_atr", "pred_mfe_atr"])
        mae_col = _pick_col(combined_df, ["head_mae_atr", "pred_mae_atr"])

        regime_allowed = combined_df["regime"].isin(
            ["TC_REGIME", "TE_REGIME", "MEAN_REGIME"]
        )
        freq_by_symbol = {}
        freq_df = (
            combined_df.groupby(["symbol", "regime"]).size().reset_index(name="count")
        )
        for sym in freq_df["symbol"].unique():
            sdf = freq_df[freq_df["symbol"] == sym].copy()
            total = sdf["count"].sum()
            freq_by_symbol[sym] = {
                row["regime"]: float(row["count"] / total) for _, row in sdf.iterrows()
            }

        kpi = {
            "frequency_overall": combined_df["regime"]
            .value_counts(normalize=True)
            .to_dict(),
            "frequency_by_symbol": freq_by_symbol,
            "recall_high_mfe": None,
            "safety_tail_mae": None,
            "notes": {
                "allowed_regimes": ["TC_REGIME", "TE_REGIME", "MEAN_REGIME"],
                "mfe_col": mfe_col,
                "mae_col": mae_col,
                "mfe_quantile": args.kpi_mfe_quantile,
                "mae_quantile": args.kpi_mae_quantile,
            },
        }

        if mfe_col:
            mfe = pd.to_numeric(combined_df[mfe_col], errors="coerce")
            thresh = float(mfe.quantile(args.kpi_mfe_quantile))
            high_mfe = mfe >= thresh
            recall = float((regime_allowed & high_mfe).sum() / max(high_mfe.sum(), 1))
            kpi["recall_high_mfe"] = {
                "threshold": thresh,
                "high_mfe_count": int(high_mfe.sum()),
                "allowed_high_mfe_count": int((regime_allowed & high_mfe).sum()),
                "recall": recall,
            }

        if mae_col:
            mae = pd.to_numeric(combined_df[mae_col], errors="coerce")
            thresh = float(mae.quantile(args.kpi_mae_quantile))
            tail = mae >= thresh
            tail_rate_all = float(tail.mean())
            tail_rate_allowed = float(
                (tail & regime_allowed).sum() / max(regime_allowed.sum(), 1)
            )
            kpi["safety_tail_mae"] = {
                "threshold": thresh,
                "tail_rate_all": tail_rate_all,
                "tail_rate_allowed": tail_rate_allowed,
                "allowed_rows": int(regime_allowed.sum()),
            }

        if args.kpi_output:
            args.kpi_output.parent.mkdir(parents=True, exist_ok=True)
            args.kpi_output.write_text(json.dumps(kpi, indent=2), encoding="utf-8")

        if args.kpi_md_output:
            args.kpi_md_output.parent.mkdir(parents=True, exist_ok=True)
            lines = ["# Physics KPI Report\n"]
            lines.append("## Frequency (overall)\n")
            lines.append("| regime | pct |\n|---|---|\n")
            for w, v in kpi["frequency_overall"].items():
                lines.append(f"| {w} | {v:.2%} |\n")

            if kpi["recall_high_mfe"]:
                r = kpi["recall_high_mfe"]
                lines.append("\n## Recall of high-MFE\n")
                lines.append("| metric | value |\n|---|---|\n")
                lines.append(f"| mfe_threshold | {r['threshold']:.4f} |\n")
                lines.append(f"| high_mfe_count | {r['high_mfe_count']} |\n")
                lines.append(
                    f"| allowed_high_mfe_count | {r['allowed_high_mfe_count']} |\n"
                )
                lines.append(f"| recall | {r['recall']:.2%} |\n")

            if kpi["safety_tail_mae"]:
                s = kpi["safety_tail_mae"]
                lines.append("\n## Safety (tail MAE)\n")
                lines.append("| metric | value |\n|---|---|\n")
                lines.append(f"| mae_threshold | {s['threshold']:.4f} |\n")
                lines.append(f"| tail_rate_all | {s['tail_rate_all']:.2%} |\n")
                lines.append(f"| tail_rate_allowed | {s['tail_rate_allowed']:.2%} |\n")

            args.kpi_md_output.write_text("".join(lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
