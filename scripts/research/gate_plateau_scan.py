"""Batch Gate lift plateau scan (archetypes/gate.yaml rules → lift json proposals)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.research.stat_kernels.gate_optimize import optimize_gate_rule_unified
from src.research.labels import derive_is_good_from_forward_rr
from src.research.stat_kernels.robustness import UnifiedOptimizationConfig
from src.time_series_model.archetype import load_strategy_archetype

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_label_col(df: pd.DataFrame, label_col: str) -> str:
    if label_col in df.columns:
        return label_col
    derive_is_good_from_forward_rr(df, label_col=label_col)
    return label_col


def _find_rr_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"):
        if candidate in df.columns:
            return candidate
    return None


def run_gate_plateau_batch(
    parquet_path: Path,
    strategy: str,
    *,
    out_dir: Path,
    label_col: str = "is_good",
    step: float = 0.05,
    rule_id: Optional[str] = None,
    gate_path: Optional[str] = None,
    strategies_root: str = "config/strategies",
    write_back_intervals: bool = False,
    min_lift: float = 0.10,
    skip_locked: bool = True,
) -> Dict[str, Any]:
    """Optimize gate rules with lift/plateau/robustness; write proposal json + summary md."""
    df = pd.read_parquet(parquet_path)
    label_col = _ensure_label_col(df, label_col)
    rr_col = _find_rr_col(df)

    cfg = UnifiedOptimizationConfig(
        min_lift=min_lift,
        min_pass_rate=0.20,
        max_pass_rate=0.80,
        min_plateau_width=0.05,
        max_lift_std_ratio=0.3,
        threshold_step=step,
    )

    arch = load_strategy_archetype(strategy, strategies_root, gate_path=gate_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rules: List[Any] = list(arch.gate.hard_gates) + list(arch.gate.system_safety)

    results: Dict[str, Any] = {}
    summary_lines = [
        f"# gate-plateau batch · {strategy}",
        "",
        f"parquet: `{parquet_path}`",
        "",
        "| rule_id | status | recommended | plateau | robustness |",
        "|---|---|---:|---|---:|",
    ]

    for rule in all_rules:
        rid = rule.id
        if rule_id and rid != rule_id:
            continue
        if skip_locked and (
            getattr(rule, "locked", False)
            or getattr(rule, "frozen", False)
            or getattr(rule, "promote_never_disable", False)
        ):
            results[rid] = {
                "rule_id": rid,
                "status": "skipped_locked",
                "reason": "locked/frozen/promote_never_disable — preserved as-is",
            }
            summary_lines.append(f"| {rid} | skipped_locked | — | — | — |")
            continue
        if getattr(rule, "frozen", False):
            results[rid] = {
                "rule_id": rid,
                "status": "frozen",
                "reason": "frozen rule",
            }
            continue

        opt = optimize_gate_rule_unified(
            df,
            rule,
            label_col,
            cfg,
            step,
            rr_col=rr_col,
            strategy=strategy,
        )
        clean = {
            k: v
            for k, v in opt.items()
            if k not in ("scan_results", "interval_details")
        }
        if write_back_intervals and opt.get("status") == "stable_plateau_found":
            clean["threshold_interval"] = {
                "start": opt.get("plateau_start"),
                "end": opt.get("plateau_end"),
                "method": "plateau_bounds",
            }
        results[rid] = clean

        rec = opt.get("recommended_threshold", opt.get("plateau_mid"))
        rob = (opt.get("robustness_score") or {}).get("overall_score")
        ps = opt.get("status", "unknown")
        p_start = opt.get("plateau_start")
        p_end = opt.get("plateau_end")
        plateau_s = (
            f"[{p_start:.4g},{p_end:.4g}]"
            if isinstance(p_start, (int, float)) and isinstance(p_end, (int, float))
            else "—"
        )
        rec_s = f"{rec:.4g}" if isinstance(rec, (int, float)) else "—"
        rob_s = f"{rob:.3f}" if isinstance(rob, (int, float)) else "—"
        summary_lines.append(f"| {rid} | {ps} | {rec_s} | {plateau_s} | {rob_s} |")

        rule_json = out_dir / f"{rid}.json"
        rule_json.write_text(
            json.dumps(clean, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    batch = {
        "kpi": "lift",
        "strategy": strategy,
        "parquet": str(parquet_path),
        "label_col": label_col,
        "rules": results,
    }
    batch_path = out_dir / "gate_plateau_batch.json"
    batch_path.write_text(
        json.dumps(batch, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_path = out_dir / "gate_plateau_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return batch
