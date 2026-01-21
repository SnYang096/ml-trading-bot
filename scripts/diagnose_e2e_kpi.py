#!/usr/bin/env python3
"""
E2E KPI diagnostics for Router/Gate/Execution.

Inputs:
  - logs_3action.parquet (symbol, timestamp, mode, ret_mean, ret_trend, ...)
  - optional physics_regime parquet (symbol, timestamp, regime)
Outputs:
  - JSON + Markdown report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return 0.0
    mean = x.mean()
    std = x.std(ddof=1)
    return float(mean / std * np.sqrt(6 * 365)) if std > 1e-12 else 0.0


def _pct(x: pd.Series, p: float) -> float:
    x = x.dropna()
    if x.empty:
        return float("nan")
    return float(np.percentile(x, p))


def _archetype_return(row: pd.Series, ret_mean_col: str, ret_trend_col: str) -> float:
    """
    Select ret_mean or ret_trend based on archetype.
    - TC/TE → ret_trend
    - FR/ET → ret_mean
    """
    archetype = str(row.get("gate_archetype") or row.get("archetype") or "").upper()
    if not archetype:
        return 0.0

    # TC/TE → ret_trend
    if "TC" in archetype or "TE" in archetype:
        return float(row.get(ret_trend_col, 0.0) or 0.0)

    # FR/ET → ret_mean
    if "FR" in archetype or "ET" in archetype:
        return float(row.get(ret_mean_col, 0.0) or 0.0)

    return 0.0


def _mode_return(row: pd.Series, ret_mean_col: str, ret_trend_col: str) -> float:
    """Legacy function for backward compatibility. Use _archetype_return instead."""
    # Try archetype first
    archetype = str(row.get("gate_archetype") or row.get("archetype") or "").upper()
    if archetype:
        return _archetype_return(row, ret_mean_col, ret_trend_col)

    # Fallback to mode (for backward compatibility)
    mode = str(row.get("mode") or "NO_TRADE").upper()
    if mode == "MEAN":
        return float(row.get(ret_mean_col, 0.0) or 0.0)
    if mode == "TREND":
        return float(row.get(ret_trend_col, 0.0) or 0.0)
    return 0.0


def _profit_loss_ratio(returns: pd.Series) -> float:
    """Profit/Loss Ratio = mean(positive returns) / abs(mean(negative returns))."""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    mean_profit = positive.mean()
    mean_loss = abs(negative.mean())
    if mean_loss < 1e-12:
        return float("nan")
    return float(mean_profit / mean_loss)


def _kpi_for_df(
    df: pd.DataFrame, *, ret_mean_col: str, ret_trend_col: str
) -> Dict[str, Any]:
    df = df.copy()
    df["mode"] = df["mode"].astype(str)
    trade_mask = df["mode"].str.upper().isin(["MEAN", "TREND"])

    df["ret_mode"] = df.apply(
        lambda r: _mode_return(r, ret_mean_col, ret_trend_col), axis=1
    )

    trade_returns = df.loc[trade_mask, "ret_mode"]

    out = {
        "rows": int(len(df)),
        "trade_rows": int(trade_mask.sum()),
        "trade_count": int(trade_mask.sum()),  # Alias for consistency
        "trade_rate": float(trade_mask.mean()) if len(df) > 0 else 0.0,
        "mode_distribution": df["mode"].value_counts(normalize=True).to_dict(),
        "sharpe_e2e": _sharpe(df["ret_mode"]),
        "sharpe_trades_only": _sharpe(trade_returns),
        "sharpe": _sharpe(trade_returns),  # Alias for consistency
        "ret_mean_e2e": float(df["ret_mode"].mean()) if len(df) > 0 else float("nan"),
        "ret_p10_e2e": _pct(
            df["ret_mode"], 10
        ),  # 10th percentile (90% of returns are above this)
        "ret_p50_e2e": _pct(df["ret_mode"], 50),  # 50th percentile (median)
        "ret_p90_e2e": _pct(
            df["ret_mode"], 90
        ),  # 90th percentile (10% of returns are above this)
        "win_rate": (
            float((trade_returns > 0).mean()) if len(trade_returns) > 0 else 0.0
        ),
        "profit_loss_ratio": _profit_loss_ratio(trade_returns),
    }

    # Per-mode KPIs
    per_mode = {}
    for mode in ["TREND", "MEAN"]:
        sub = df[df["mode"].str.upper() == mode]
        ret_col = ret_trend_col if mode == "TREND" else ret_mean_col
        ret = pd.to_numeric(sub.get(ret_col, pd.Series(dtype=float)), errors="coerce")
        per_mode[mode] = {
            "rows": int(len(sub)),
            "trade_count": int(len(ret)),
            "sharpe": _sharpe(ret),
            "ret_mean": float(ret.mean()) if len(ret) > 0 else float("nan"),
            "ret_p10": _pct(ret, 10),
            "ret_p50": _pct(ret, 50),
            "ret_p90": _pct(ret, 90),
            "win_rate": float((ret > 0).mean()) if len(ret) > 0 else 0.0,
            "profit_loss_ratio": _profit_loss_ratio(ret),
        }
    out["per_mode"] = per_mode
    return out


def _bucket_by_score(
    df: pd.DataFrame,
    score_col: str,
    *,
    buckets: int = 5,
    ret_mean_col: str,
    ret_trend_col: str,
) -> list[dict[str, Any]]:
    if score_col not in df.columns:
        return []
    scores = pd.to_numeric(df[score_col], errors="coerce")
    mask = scores.notna()
    if mask.sum() == 0:
        return []
    try:
        q = pd.qcut(scores[mask], q=buckets, duplicates="drop")
    except ValueError:
        return []
    out = []
    for bucket in q.cat.categories:
        idx = q == bucket
        sub = df.loc[mask].loc[idx.index[idx]]
        kpi = _kpi_for_df(sub, ret_mean_col=ret_mean_col, ret_trend_col=ret_trend_col)
        out.append(
            {
                "bucket": str(bucket),
                "rows": kpi["rows"],
                "trade_rows": kpi["trade_rows"],
                "trade_rate": kpi["trade_rate"],
                "sharpe_e2e": kpi["sharpe_e2e"],
                "sharpe_trades_only": kpi["sharpe_trades_only"],
                "ret_mean_e2e": kpi["ret_mean_e2e"],
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose E2E KPI for 3action logs.")
    p.add_argument("--logs", required=True, help="logs_3action.parquet")
    p.add_argument("--regime", default=None, help="physics_regime parquet (optional)")
    p.add_argument(
        "--gate",
        default=None,
        help="gate output parquet with archetype info (optional)",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Output JSON path (default: results/e2e_kpi/e2e_kpi_report.json)",
    )
    p.add_argument(
        "--output-md",
        default=None,
        help="Output Markdown path (default: results/e2e_kpi/e2e_kpi_report.md)",
    )
    p.add_argument("--ret-mean-col", default="ret_mean")
    p.add_argument("--ret-trend-col", default="ret_trend")
    p.add_argument("--semantic-buckets", type=int, default=5)
    p.add_argument(
        "--semantic-switch-top-quantile",
        type=float,
        default=None,
        help="If set, switch TC/TE top semantic buckets to MEAN execution (use ret_mean).",
    )
    p.add_argument(
        "--no-regime-filter",
        action="store_true",
        help="Generate comparison report without regime filtering (all trades included).",
    )
    args = p.parse_args()

    logs_df = pd.read_parquet(args.logs)
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])

    if "mode" not in logs_df.columns:
        raise KeyError("logs must include 'mode' column")

    # Load gate file if provided (contains archetype information)
    archetype_col = None
    if args.gate:
        gate_df = pd.read_parquet(args.gate)
        gate_df["timestamp"] = pd.to_datetime(gate_df["timestamp"])
        # Merge archetype from gate file - support multiple column names
        gate_arch_col = None
        if "gate_archetype" in gate_df.columns:
            gate_arch_col = "gate_archetype"
        elif "gate_arch" in gate_df.columns:
            gate_arch_col = "gate_arch"
        elif "archetype" in gate_df.columns:
            gate_arch_col = "archetype"

        if gate_arch_col:
            # Merge with suffix to avoid column name conflicts
            merged_gate = logs_df.merge(
                gate_df[["symbol", "timestamp", gate_arch_col]],
                on=["symbol", "timestamp"],
                how="left",
                suffixes=("", "_gate"),
            )
            # Use the gate column (may have _gate suffix if conflict)
            actual_gate_col = gate_arch_col
            if f"{gate_arch_col}_gate" in merged_gate.columns:
                actual_gate_col = f"{gate_arch_col}_gate"
            elif gate_arch_col not in merged_gate.columns:
                # If still not found, try to find it
                for col in merged_gate.columns:
                    if "gate" in col.lower() and "arch" in col.lower():
                        actual_gate_col = col
                        break

            logs_df = merged_gate

            # Map gate archetype to standard archetype names
            def _normalize_archetype(arch):
                if pd.isna(arch) or not arch or str(arch).strip() == "":
                    return None
                arch = str(arch).upper()
                if "TRENDCONTINUATION" in arch or arch == "TC":
                    return "TC"
                elif "TRENDEXPANSION" in arch or arch == "TE":
                    return "TE"
                elif "FAILUREREVERSION" in arch or arch == "FR":
                    return "FR"
                elif "EXHAUSTIONTURN" in arch or arch == "ET":
                    return "ET"
                return None

            logs_df["archetype"] = logs_df[actual_gate_col].apply(_normalize_archetype)
            archetype_col = "archetype"
        elif "archetype" in gate_df.columns:
            logs_df = logs_df.merge(
                gate_df[["symbol", "timestamp", "archetype"]],
                on=["symbol", "timestamp"],
                how="left",
            )
            archetype_col = "archetype"

    base = _kpi_for_df(
        logs_df, ret_mean_col=args.ret_mean_col, ret_trend_col=args.ret_trend_col
    )

    by_symbol = None
    by_regime = None
    by_regime_semantic = None
    by_symbol_regime_archetype = None
    by_archetype = None
    by_symbol_archetype = None
    switched_kpi = None
    no_regime_filter_kpi = None  # KPI without regime filtering

    # Per symbol summary
    if "symbol" in logs_df.columns:
        by_symbol = {}
        for symbol, sym_df in logs_df.groupby("symbol"):
            by_symbol[str(symbol)] = _kpi_for_df(
                sym_df, ret_mean_col=args.ret_mean_col, ret_trend_col=args.ret_trend_col
            )

    # Load regime file if provided (only if not already in logs_df)
    if args.regime:
        if "regime" not in logs_df.columns:
            regime_df = pd.read_parquet(args.regime)
            regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"])
            # Support both 'regime' and 'world' columns (backward compatibility)
            # Architecture: Router/World/Regime are the same - they classify TREND/MEAN/NO_TRADE
            if "regime" not in regime_df.columns and "world" in regime_df.columns:
                # Map world to regime: TC_WORLD/TE_WORLD -> TREND, MEAN_WORLD -> MEAN, NO_TRADE -> NO_TRADE
                def _world_to_regime(w):
                    if pd.isna(w):
                        return "NO_TRADE"
                    w = str(w).upper()
                    if "TC_WORLD" in w or "TE_WORLD" in w:
                        return "TREND"
                    elif "MEAN_WORLD" in w:
                        return "MEAN"
                    else:
                        return "NO_TRADE"

                regime_df["regime"] = regime_df["world"].apply(_world_to_regime)
            elif "regime" in regime_df.columns:
                # If already has regime column, map TC_REGIME/TE_REGIME -> TREND
                def _normalize_regime(r):
                    if pd.isna(r):
                        return "NO_TRADE"
                    r = str(r).upper()
                    if "TC_REGIME" in r or "TE_REGIME" in r:
                        return "TREND"
                    elif "MEAN_REGIME" in r:
                        return "MEAN"
                    else:
                        return r  # Keep NO_TRADE as is

                regime_df["regime"] = regime_df["regime"].apply(_normalize_regime)
            if "regime" not in regime_df.columns:
                raise KeyError(
                    "regime file must contain either 'regime' or 'world' column"
                )
            merged = logs_df.merge(
                regime_df[["symbol", "timestamp", "regime"]],
                on=["symbol", "timestamp"],
                how="inner",
            )
        else:
            # Regime already in logs_df (e.g., from gate file merge)
            merged = logs_df.copy()
            # Normalize regime if needed
            if merged["regime"].dtype == object:

                def _normalize_regime_in_place(r):
                    if pd.isna(r):
                        return "NO_TRADE"
                    r = str(r).upper()
                    if "TC_REGIME" in r or "TE_REGIME" in r:
                        return "TREND"
                    elif "MEAN_REGIME" in r:
                        return "MEAN"
                    else:
                        return r

                merged["regime"] = merged["regime"].apply(_normalize_regime_in_place)
    else:
        # No regime file provided
        merged = logs_df.copy()
        if "regime" not in merged.columns:
            merged["regime"] = "UNKNOWN"
        by_regime = {}
        for regime, wdf in merged.groupby("regime"):
            # Filter out NO_TRADE regime trades for KPI calculation
            # NO_TRADE regime should not have any trades
            if regime == "NO_TRADE":
                # Only count rows where mode is also NO_TRADE
                wdf_filtered = wdf[wdf["mode"] == "NO_TRADE"]
                if len(wdf_filtered) > 0:
                    by_regime[regime] = _kpi_for_df(
                        wdf_filtered,
                        ret_mean_col=args.ret_mean_col,
                        ret_trend_col=args.ret_trend_col,
                    )
            else:
                by_regime[regime] = _kpi_for_df(
                    wdf,
                    ret_mean_col=args.ret_mean_col,
                    ret_trend_col=args.ret_trend_col,
                )

        # Symbol × Regime × Archetype (mode) cross table
        by_symbol_regime_archetype = {}
        for (symbol, regime, mode), group_df in merged.groupby(
            ["symbol", "regime", "mode"]
        ):
            key = (str(symbol), str(regime), str(mode))
            by_symbol_regime_archetype[key] = _kpi_for_df(
                group_df,
                ret_mean_col=args.ret_mean_col,
                ret_trend_col=args.ret_trend_col,
            )

        # Infer archetype from regime if not provided by gate
        # Note: This is a simplified mapping - full archetype assignment requires gate logic
        # For accurate TC/TE/FR/ET classification, use --gate parameter with gate output file
        if archetype_col is None:

            def _infer_archetype_from_regime(row):
                regime = str(row.get("regime", "")).upper()
                mode = str(row.get("mode", "")).upper()
                # Simplified inference based on regime + mode
                # TREND regime + TREND mode -> TC (default, could be TE but need more info)
                # MEAN regime + MEAN mode -> FR
                # NO_TRADE regime -> no archetype
                if regime == "TREND" and mode == "TREND":
                    return "TC"  # Default to TC, could be TE but need gate logic to distinguish
                elif regime == "MEAN" and mode == "MEAN":
                    return "FR"  # Mean reversion -> FR
                elif mode == "TREND":
                    # If mode is TREND but regime is not TREND, still infer as TC (fallback)
                    return "TC"
                return None

            merged["archetype"] = merged.apply(_infer_archetype_from_regime, axis=1)
            logs_df = merged.copy()
            archetype_col = "archetype"

    # Generate no-regime-filter comparison if requested
    # This should be calculated BEFORE regime filtering is applied
    if args.no_regime_filter:
        # Use all trades without regime filtering (original logs_df)
        # This includes trades that would be filtered out by NO_TRADE regime
        # But still apply gate filtering if gate file is provided
        original_logs = pd.read_parquet(args.logs)
        original_logs["timestamp"] = pd.to_datetime(original_logs["timestamp"])

        # Apply gate filtering if gate file is provided (same as logs_df)
        if args.gate:
            gate_df = pd.read_parquet(args.gate)
            gate_df["timestamp"] = pd.to_datetime(gate_df["timestamp"])
            if "gate_ok" in gate_df.columns:
                original_logs = original_logs.merge(
                    gate_df[["symbol", "timestamp", "gate_ok"]],
                    on=["symbol", "timestamp"],
                    how="left",
                )
                # Apply gate_ok filter (same as logs_df) - only if column exists
                if "gate_ok" in original_logs.columns:
                    original_logs = original_logs[original_logs["gate_ok"] == True]

        # Calculate KPI without regime filtering (includes NO_TRADE regime trades)
        no_regime_filter_kpi = _kpi_for_df(
            original_logs,
            ret_mean_col=args.ret_mean_col,
            ret_trend_col=args.ret_trend_col,
        )

        # Also calculate regime-filtered KPI for proper comparison
        # (only count trades in TREND/MEAN regimes, exclude NO_TRADE regime trades)
        if args.regime:
            regime_filtered_logs = merged[
                merged["regime"].isin(
                    ["TREND", "MEAN", "TC_REGIME", "TE_REGIME", "MEAN_REGIME"]
                )
            ]
            if len(regime_filtered_logs) > 0:
                # Recalculate base KPI with regime filtering
                regime_filtered_base = _kpi_for_df(
                    regime_filtered_logs,
                    ret_mean_col=args.ret_mean_col,
                    ret_trend_col=args.ret_trend_col,
                )
                # Store for comparison
                base = regime_filtered_base  # Update base to regime-filtered version

    # By Archetype (TC/TE/FR/ET) if available
    if archetype_col and archetype_col in logs_df.columns:
        by_archetype = {}
        for arch, arch_df in logs_df.groupby(archetype_col):
            if pd.notna(arch) and arch:
                by_archetype[str(arch)] = _kpi_for_df(
                    arch_df,
                    ret_mean_col=args.ret_mean_col,
                    ret_trend_col=args.ret_trend_col,
                )

        # Symbol × Archetype cross table
        by_symbol_archetype = {}
        for (symbol, arch), group_df in logs_df.groupby(["symbol", archetype_col]):
            if pd.notna(arch) and arch:
                key = (str(symbol), str(arch))
                by_symbol_archetype[key] = _kpi_for_df(
                    group_df,
                    ret_mean_col=args.ret_mean_col,
                    ret_trend_col=args.ret_trend_col,
                )

        # semantic score buckets (TC/TE only)
        # Note: After merging, regime column is normalized to TREND/MEAN/NO_TRADE
        # But semantic scores are keyed by TC_REGIME/TE_REGIME, so we need to check original regime
        # Load regime file for semantic scores if not already loaded
        regime_df_for_semantic = None
        if args.regime and (
            "tc_semantic_score" not in merged.columns
            and "te_semantic_score" not in merged.columns
        ):
            regime_df_for_semantic = pd.read_parquet(args.regime)
            regime_df_for_semantic["timestamp"] = pd.to_datetime(
                regime_df_for_semantic["timestamp"]
            )
            # Merge semantic scores
            if (
                "tc_semantic_score" in regime_df_for_semantic.columns
                or "te_semantic_score" in regime_df_for_semantic.columns
            ):
                merged = merged.merge(
                    regime_df_for_semantic[
                        [
                            "symbol",
                            "timestamp",
                            "tc_semantic_score",
                            "te_semantic_score",
                        ]
                    ],
                    on=["symbol", "timestamp"],
                    how="left",
                )

        if (
            "tc_semantic_score" in merged.columns
            or "te_semantic_score" in merged.columns
        ):
            # Use merged dataframe which already has semantic scores
            # No need to reload regime_df
            # Keep original regime values for semantic score lookup
            if (
                regime_df_for_semantic is not None
                and "regime" in regime_df_for_semantic.columns
            ):
                # Create a mapping: normalized regime -> original regime
                regime_df_for_semantic["regime_original"] = regime_df_for_semantic[
                    "regime"
                ]

                # Normalize for consistency
                def _normalize_for_semantic(r):
                    if pd.isna(r):
                        return "NO_TRADE"
                    r = str(r).upper()
                    if "TC_REGIME" in r:
                        return "TC_REGIME"
                    elif "TE_REGIME" in r:
                        return "TE_REGIME"
                    elif "MEAN_REGIME" in r:
                        return "MEAN_REGIME"
                    else:
                        return "NO_TRADE"

                regime_df_for_semantic["regime_original"] = regime_df_for_semantic[
                    "regime_original"
                ].apply(_normalize_for_semantic)

            merged_semantic = None
            if regime_df_for_semantic is not None:
                merged_semantic = logs_df.merge(
                    regime_df_for_semantic[
                        [
                            "symbol",
                            "timestamp",
                            "regime_original",
                            "tc_semantic_score",
                            "te_semantic_score",
                        ]
                    ],
                    on=["symbol", "timestamp"],
                    how="inner",
                )
            by_regime_semantic = {}
            if merged_semantic is not None:
                for regime_orig, score_col in [
                    ("TC_REGIME", "tc_semantic_score"),
                    ("TE_REGIME", "te_semantic_score"),
                ]:
                    wdf = merged_semantic[
                        merged_semantic["regime_original"] == regime_orig
                    ]
                by_regime_semantic[regime_orig] = _bucket_by_score(
                    wdf,
                    score_col,
                    buckets=args.semantic_buckets,
                    ret_mean_col=args.ret_mean_col,
                    ret_trend_col=args.ret_trend_col,
                )

            # Optional semantic switch: top quantile TC/TE -> MEAN execution
            if args.semantic_switch_top_quantile is not None:
                switch_df = merged.copy()
                q = float(args.semantic_switch_top_quantile)
                # TC
                tc_q = pd.to_numeric(
                    switch_df["tc_semantic_score"], errors="coerce"
                ).quantile(q)
                tc_mask = (switch_df["regime"] == "TC_REGIME") & (
                    pd.to_numeric(switch_df["tc_semantic_score"], errors="coerce")
                    >= tc_q
                )
                # TE
                te_q = pd.to_numeric(
                    switch_df["te_semantic_score"], errors="coerce"
                ).quantile(q)
                te_mask = (switch_df["regime"] == "TE_REGIME") & (
                    pd.to_numeric(switch_df["te_semantic_score"], errors="coerce")
                    >= te_q
                )
                switch_mask = tc_mask | te_mask
                # Override mode to MEAN for evaluation only
                switch_df.loc[switch_mask, "mode"] = "MEAN"
                switched_kpi = _kpi_for_df(
                    switch_df,
                    ret_mean_col=args.ret_mean_col,
                    ret_trend_col=args.ret_trend_col,
                )

    report = {
        "source_logs": str(args.logs),
        "source_regime": str(args.regime) if args.regime else None,
        "source_gate": str(args.gate) if args.gate else None,
        "overall": base,
        "by_symbol": by_symbol,
        "by_regime": by_regime,
        "by_archetype": by_archetype,
        "by_regime_semantic_buckets": by_regime_semantic,
        "by_symbol_regime_archetype": {
            f"{symbol}|{regime}|{archetype}": kpi
            for (symbol, regime, archetype), kpi in (
                by_symbol_regime_archetype or {}
            ).items()
        },
        "by_symbol_archetype": {
            f"{symbol}|{archetype}": kpi
            for (symbol, archetype), kpi in (by_symbol_archetype or {}).items()
        },
        "semantic_switch_top_quantile": args.semantic_switch_top_quantile,
        "kpi_with_semantic_switch": switched_kpi,
        "no_regime_filter_kpi": no_regime_filter_kpi,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("# E2E KPI Report\n")
    lines.append(f"- logs: `{args.logs}`\n")
    if args.regime:
        lines.append(f"- regime: `{args.regime}`\n")
    if args.gate:
        lines.append(f"- gate: `{args.gate}`\n")

        # Note: Router/World/Regime are the same thing - they classify trend vs mean
        # KPI focus is recall (coverage of possible trading opportunities)

    lines.append("\n## Overall\n")
    lines.append("| metric | value |\n|---|---|\n")
    for k in [
        "rows",
        "trade_rows",
        "trade_rate",
        "sharpe_e2e",
        "sharpe_trades_only",
        "ret_mean_e2e",
        "ret_p10_e2e",
        "ret_p50_e2e",
        "ret_p90_e2e",
    ]:
        v = base.get(k, float("nan"))
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |\n")
        else:
            lines.append(f"| {k} | {v} |\n")

    lines.append("\n## Per Mode (Router Output)\n")
    lines.append(
        '⚠️ **Note**: This shows Router\'s 3-action output (TREND/MEAN/NO_TRADE). For detailed archetype analysis, see "By Archetype" section below.\n\n'
    )
    lines.append(
        "| mode | trade_count | sharpe | win_rate | profit_loss_ratio | ret_mean | ret_p50 | ret_p10 | ret_p90 |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for mode, s in base["per_mode"].items():
        lines.append(
            f"| {mode} | {s['trade_count']} | {s['sharpe']:.3f} | {s['win_rate']:.3f} | "
            f"{s['profit_loss_ratio']:.3f} | {s['ret_mean']:.4f} | {s['ret_p50']:.4f} | "
            f"{s['ret_p10']:.4f} | {s['ret_p90']:.4f} |\n"
        )

    if by_symbol:
        lines.append("\n## Per Symbol\n")
        lines.append(
            "| symbol | trade_count | sharpe | win_rate | profit_loss_ratio | ret_mean |\n"
        )
        lines.append("|---|---|---|---|---|---|\n")
        for symbol in sorted(by_symbol.keys()):
            s = by_symbol[symbol]
            lines.append(
                f"| {symbol} | {s['trade_count']} | {s['sharpe']:.3f} | {s['win_rate']:.3f} | "
                f"{s['profit_loss_ratio']:.3f} | {s['ret_mean_e2e']:.4f} |\n"
            )

    if by_regime:
        lines.append("\n## By Regime\n")
        lines.append(
            "⚠️ **Note**: NO_TRADE regime should have 0 trades. If NO_TRADE shows trades, the logs file may be from before regime filtering was applied.\n\n"
        )
        lines.append(
            "| regime | trade_count | sharpe | win_rate | profit_loss_ratio | ret_mean |\n"
        )
        lines.append("|---|---|---|---|---|---|\n")
        for regime in sorted(by_regime.keys()):
            s = by_regime[regime]
            if regime == "NO_TRADE" and s["trade_count"] > 0:
                lines.append(
                    f"| {regime} ⚠️ | {s['trade_count']} | {s['sharpe']:.3f} | {s['win_rate']:.3f} | "
                    f"{s['profit_loss_ratio']:.3f} | {s['ret_mean_e2e']:.4f} |\n"
                )
            else:
                lines.append(
                    f"| {regime} | {s['trade_count']} | {s['sharpe']:.3f} | {s['win_rate']:.3f} | "
                    f"{s['profit_loss_ratio']:.3f} | {s['ret_mean_e2e']:.4f} |\n"
                )

    if by_archetype:
        lines.append("\n## By Archetype (TC/TE/FR/ET) - Detailed Report\n")
        if not args.gate:
            lines.append(
                "⚠️ **Note**: Archetype inferred from regime+mode. For accurate TC/TE/FR/ET classification, use `--gate` parameter with gate output file.\n\n"
            )

        # Summary table
        lines.append("### Summary\n")
        lines.append(
            "| archetype | trade_count | sharpe | win_rate | profit_loss_ratio | ret_mean | ret_p10 | ret_p50 | ret_p90 |\n"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        for arch in sorted(by_archetype.keys()):
            s = by_archetype[arch]
            lines.append(
                f"| {arch} | {s['trade_count']} | {s['sharpe']:.3f} | {s['win_rate']:.3f} | "
                f"{s['profit_loss_ratio']:.3f} | {s['ret_mean_e2e']:.4f} | "
                f"{s.get('ret_p10_e2e', float('nan')):.4f} | {s.get('ret_p50_e2e', float('nan')):.4f} | {s.get('ret_p90_e2e', float('nan')):.4f} |\n"
            )

        # Detailed per-archetype breakdown
        lines.append("\n### Detailed Breakdown by Archetype\n")
        for arch in sorted(by_archetype.keys()):
            s = by_archetype[arch]
            lines.append(f"\n#### {arch}\n")
            lines.append("| metric | value |\n|---|---|\n")
            for k in [
                "trade_count",
                "sharpe",
                "win_rate",
                "profit_loss_ratio",
                "ret_mean_e2e",
                "ret_p10_e2e",
                "ret_p50_e2e",
                "ret_p90_e2e",
                "trade_rate",
            ]:
                v = s.get(k, float("nan"))
                if isinstance(v, float):
                    lines.append(f"| {k} | {v:.4f} |\n")
                else:
                    lines.append(f"| {k} | {v} |\n")

    if by_symbol_archetype:
        lines.append("\n## Symbol × Archetype\n")
        lines.append(
            "| symbol | archetype | trade_count | sharpe | win_rate | profit_loss_ratio | ret_mean |\n"
        )
        lines.append("|---|---|---|---|---|---|---|\n")
        for symbol, arch in sorted(by_symbol_archetype.keys()):
            s = by_symbol_archetype[(symbol, arch)]
            lines.append(
                f"| {symbol} | {arch} | {s['trade_count']} | {s['sharpe']:.3f} | "
                f"{s['win_rate']:.3f} | {s['profit_loss_ratio']:.3f} | {s['ret_mean_e2e']:.4f} |\n"
            )

    # No-regime-filter comparison
    if no_regime_filter_kpi:
        lines.append("\n## Comparison: With vs Without Regime Filter\n")
        lines.append(
            "This comparison shows the impact of regime filtering on overall performance.\n\n"
        )
        lines.append(
            "| metric | with_regime_filter | without_regime_filter | difference |\n"
        )
        lines.append("|---|---|---|---|\n")
        # Calculate regime-filtered metrics
        if by_regime:
            total_trades = sum(kpi.get("trade_count", 0) for kpi in by_regime.values())
            if total_trades > 0:
                regime_sharpe = (
                    sum(
                        kpi.get("sharpe", 0) * kpi.get("trade_count", 0)
                        for kpi in by_regime.values()
                    )
                    / total_trades
                )
                regime_win_rate = (
                    sum(
                        kpi.get("win_rate", 0) * kpi.get("trade_count", 0)
                        for kpi in by_regime.values()
                    )
                    / total_trades
                )
            else:
                regime_sharpe = base.get("sharpe", 0.0)
                regime_win_rate = base.get("win_rate", 0.0)
        else:
            regime_sharpe = base.get("sharpe", 0.0)
            regime_win_rate = base.get("win_rate", 0.0)

        lines.append(
            f"| trade_count | {base.get('trade_count', 0)} | {no_regime_filter_kpi.get('trade_count', 0)} | "
            f"{no_regime_filter_kpi.get('trade_count', 0) - base.get('trade_count', 0)} |\n"
        )
        lines.append(
            f"| sharpe | {regime_sharpe:.3f} | {no_regime_filter_kpi.get('sharpe', 0.0):.3f} | "
            f"{no_regime_filter_kpi.get('sharpe', 0.0) - regime_sharpe:.3f} |\n"
        )
        lines.append(
            f"| win_rate | {regime_win_rate:.3f} | {no_regime_filter_kpi.get('win_rate', 0.0):.3f} | "
            f"{no_regime_filter_kpi.get('win_rate', 0.0) - regime_win_rate:.3f} |\n"
        )
        lines.append(
            f"| ret_mean | {base.get('ret_mean_e2e', 0.0):.4f} | {no_regime_filter_kpi.get('ret_mean_e2e', 0.0):.4f} | "
            f"{no_regime_filter_kpi.get('ret_mean_e2e', 0.0) - base.get('ret_mean_e2e', 0.0):.4f} |\n"
        )

    if by_symbol_regime_archetype:
        lines.append("\n## Symbol × Regime × Mode (Legacy)\n")
        lines.append(
            "⚠️ **Note**: 'mode' is legacy Router output (TREND/MEAN/NO_TRADE). Use archetype (TC/TE/FR/ET) for detailed analysis.\n\n"
        )
        lines.append(
            "| symbol | regime | mode | trade_count | sharpe | win_rate | profit_loss_ratio | ret_mean |\n"
        )
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for symbol, regime, mode in sorted(by_symbol_regime_archetype.keys()):
            s = by_symbol_regime_archetype[(symbol, regime, mode)]
            lines.append(
                f"| {symbol} | {regime} | {mode} | {s['trade_count']} | {s['sharpe']:.3f} | "
                f"{s['win_rate']:.3f} | {s['profit_loss_ratio']:.3f} | {s['ret_mean_e2e']:.4f} |\n"
            )

    if by_regime_semantic:
        lines.append("\n## By Regime Semantic Buckets\n")
        for regime, buckets in by_regime_semantic.items():
            lines.append(f"\n### {regime}\n")
            lines.append(
                "| bucket | rows | trade_rows | trade_rate | sharpe_e2e | sharpe_trades_only | ret_mean_e2e |\n"
            )
            lines.append("|---|---|---|---|---|---|---|\n")
            for r in buckets:
                lines.append(
                    f"| {r['bucket']} | {r['rows']} | {r['trade_rows']} | {r['trade_rate']:.4f} | "
                    f"{r['sharpe_e2e']:.3f} | {r['sharpe_trades_only']:.3f} | {r['ret_mean_e2e']:.4f} |\n"
                )

    if switched_kpi:
        lines.append("\n## KPI with Semantic Switch (Top Quantile → MEAN Execution)\n")
        lines.append("| metric | value |\n|---|---|\n")
        for k in [
            "rows",
            "trade_rows",
            "trade_rate",
            "sharpe_e2e",
            "sharpe_trades_only",
            "ret_mean_e2e",
        ]:
            v = switched_kpi.get(k, float("nan"))
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |\n")
            else:
                lines.append(f"| {k} | {v} |\n")

    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"✅ Wrote: {out_json}")
    print(f"✅ Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
