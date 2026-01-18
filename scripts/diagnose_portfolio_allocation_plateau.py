#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.time_series_model.diagnostics.kpi_gate import check_kpi_gate
from src.time_series_model.portfolio.portfolio_assets_artifacts import (
    build_portfolio_assets_artifacts_from_modes,
)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _apply_sweep(cfg: Dict[str, Any], *, target: str, value: float) -> Dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    if target == "global_trend_p_trend_min":
        out["router_to_weights"]["global_trend"]["p_trend_min"] = float(value)
        return out
    if target == "global_trend_regime_entropy_max":
        out["router_to_weights"]["global_trend"]["regime_entropy_max"] = float(value)
        return out
    if target == "high_beta_confidence_min":
        out["router_to_weights"]["high_beta_overlay"]["confidence_min"] = float(value)
        return out
    if target == "high_beta_crowding_max":
        out["router_to_weights"]["high_beta_overlay"]["crowding_max"] = float(value)
        return out
    if target == "trend_zero_regime_entropy_gt":
        rules = out.get("trend_zero_law", {}).get("rules") or []
        for r in rules:
            if "regime_entropy_gt" in r:
                r["regime_entropy_gt"] = float(value)
                return out
        raise ValueError("trend_zero_regime_entropy_gt rule not found")
    if target == "trend_zero_portfolio_drawdown_gt":
        rules = out.get("trend_zero_law", {}).get("rules") or []
        for r in rules:
            if "portfolio_drawdown_gt" in r:
                r["portfolio_drawdown_gt"] = float(value)
                return out
        raise ValueError("trend_zero_portfolio_drawdown_gt rule not found")
    raise ValueError(f"Unknown sweep target: {target}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Portfolio allocation plateau sweep (proxy KPIs)."
    )
    ap.add_argument("--mode", required=True, help="mode_3action parquet/csv")
    ap.add_argument("--portfolio-assets-yaml", required=True)
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument(
        "--sweep-target",
        required=True,
        choices=[
            "global_trend_p_trend_min",
            "global_trend_regime_entropy_max",
            "high_beta_confidence_min",
            "high_beta_crowding_max",
            "trend_zero_regime_entropy_gt",
            "trend_zero_portfolio_drawdown_gt",
        ],
    )
    ap.add_argument("--grid", required=True, help="Comma-separated sweep values")
    ap.add_argument(
        "--gate-yaml",
        default="config/kpi_gates/nnmh_portfolio_allocation.yaml",
        help="KPI gate yaml for auto selection",
    )
    ap.add_argument(
        "--require-gate",
        action="store_true",
        help="Only select from gate-passing candidates.",
    )
    ap.add_argument(
        "--plateau-frac",
        type=float,
        default=0.05,
        help="Plateau cutoff as fraction of |best_score|.",
    )
    ap.add_argument(
        "--score-key",
        default="rule_pcm_sharpe_mean",
        help="Metric key to optimize (default: rule_pcm_sharpe_mean).",
    )
    ap.add_argument("--out", required=True, help="Output directory for report")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = _read_any(Path(args.mode))
    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    base_cfg = _load_yaml(args.portfolio_assets_yaml)
    gate = _load_yaml(args.gate_yaml)

    grid = [float(x) for x in str(args.grid).split(",") if x.strip()]

    rows = []
    for val in grid:
        cfg = _apply_sweep(base_cfg, target=args.sweep_target, value=val)
        cfg_path = out_dir / f"cfg_{args.sweep_target}_{val}.yaml"
        _write_yaml(cfg, cfg_path)

        dd_proxy = float(metrics.get("rule_avg_max_dd", 0.0))
        pa = build_portfolio_assets_artifacts_from_modes(
            mode,
            portfolio_assets_yaml=str(cfg_path),
            timestamp_col="timestamp",
            symbol_col="symbol",
            mode_col="mode",
            gate_veto=False,
            portfolio_drawdown=dd_proxy,
        )
        pa_summary = dict(pa.summary or {})
        metrics_local = dict(metrics)
        for k, v in (pa_summary.get("avg_weights") or {}).items():
            metrics_local[f"pa__avg_weight__{k}"] = float(v)
        metrics_local["pa__trend_zero_rate"] = float(
            pa_summary.get("trend_zero_rate", 0.0)
        )

        gate_ok = None
        gate_failures = []
        if gate:
            res = check_kpi_gate(metrics=metrics_local, gate=gate)
            gate_ok = bool(res.ok)
            gate_failures = list(res.hard_failures)

        score_key = str(args.score_key)
        score_val = float(metrics_local.get(score_key, 0.0))
        score_sharpe_minus_dd = float(
            metrics_local.get("rule_pcm_sharpe_mean", 0.0)
            - metrics_local.get("rule_pcm_avg_max_dd", 0.0)
        )

        rows.append(
            {
                "value": val,
                "score_key": score_key,
                "score_value": score_val,
                "score_sharpe_minus_dd": score_sharpe_minus_dd,
                "pa__trend_zero_rate": metrics_local.get("pa__trend_zero_rate", 0.0),
                "pa__avg_weight__GLOBAL_CASH": metrics_local.get(
                    "pa__avg_weight__GLOBAL_CASH", 0.0
                ),
                "pa__avg_weight__GLOBAL_TREND": metrics_local.get(
                    "pa__avg_weight__GLOBAL_TREND", 0.0
                ),
                "pa__avg_weight__GLOBAL_MEAN": metrics_local.get(
                    "pa__avg_weight__GLOBAL_MEAN", 0.0
                ),
                "pa__avg_weight__DEFENSIVE_MEAN": metrics_local.get(
                    "pa__avg_weight__DEFENSIVE_MEAN", 0.0
                ),
                "gate_ok": gate_ok,
                "gate_failures": ";".join(gate_failures) if gate_failures else "",
            }
        )

    df_out = pd.DataFrame(rows).sort_values("value")
    df_out.to_csv(out_dir / "plateau.csv", index=False)

    score_key = str(args.score_key)
    if score_key not in df_out.columns and "score_value" in df_out.columns:
        score_key = "score_value"
    df_sel = df_out.copy()
    if args.require_gate and "gate_ok" in df_sel.columns:
        df_sel = df_sel[df_sel["gate_ok"] == True]  # noqa: E712
    best_score = (
        float(df_sel[score_key].max())
        if len(df_sel)
        else float(df_out[score_key].max())
    )
    cutoff = best_score - abs(best_score) * float(args.plateau_frac)
    plateau = (
        df_sel[df_sel[score_key] >= cutoff]
        if len(df_sel)
        else df_out[df_out[score_key] >= cutoff]
    )
    plateau_vals = plateau["value"].tolist()
    selected_val = (
        float(plateau["value"].median())
        if plateau_vals
        else float(df_out.iloc[df_out[score_key].idxmax()]["value"])
    )

    summary = {
        "score_key": score_key,
        "best_score": best_score,
        "cutoff": cutoff,
        "plateau_vals": plateau_vals,
        "selected_value": selected_val,
        "sweep_target": args.sweep_target,
        "gate_yaml": str(args.gate_yaml),
        "require_gate": bool(args.require_gate),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        "# Portfolio Allocation Plateau\n\n"
        + df_out.to_markdown(index=False)
        + "\n\n## Summary\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(f"✅ Wrote: {out_dir / 'plateau.csv'}")
    print(f"✅ Wrote: {out_dir / 'report.md'}")
    print(f"✅ Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
