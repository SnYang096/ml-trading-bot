#!/usr/bin/env python3
"""Regime watchdog — 监控 TPC bull-conditional gate (variant H) 的健康度。

Variant H gates (`gate_vol_persistence_vol_persistence_bull_only`,
`gate_tpc_vol_leverage_asymmetry_mid_bull_only`) 仅在 ema_1200_position>0.10
时启用。本脚本周度运行，统计:

    1. 当前窗口内 ema_1200_position 分布（mean/p25/p50/p75/p90）；
    2. 当前窗口内 bull side (ema_1200_position>=0.10) 与 bear side
       (ema_1200_position<=-0.10) 的占比；
    3. vol_persistence / vol_leverage_asymmetry 在 deny band 内的占比，
       分 (bull, bear) 两段统计；
    4. 实际 vol gate "触发率"（即 bull-side 且 vp/vla 同时落入 deny 带的占比）；
    5. ALERT 触发条件:
       - bull_share 与基线（calibration）偏离 > 0.10 (10pp)
       - bear-side vp/vla in band 占比异常高（说明 H 的 bull-only 设计可能漏放
         了大量本该过滤的 bear-band 样本，但这是 H 设计 by-construction，所以
         主要监控 bull-side 漂移）；
       - 月度 trigger_rate（实际命中三条件的占比）相比上次校准漂移 > 50%
         相对变化。

用途:
    - 每周 cron 运行 → JSON 报告 + 简短文本摘要；
    - 非零退出码表示 ALERT，可对接 PCM watchdog 告警；
    - 与 ``regime_drift_monitor.py``（plateau 漂移）形成 D/W 两层监控。

调用示例:
    python scripts/regime_watchdog.py \\
        --window-parquet results/<...>/recent_features.parquet \\
        --strategies tpc

退出码:
    0   正常
    1   ALERT (任一 strategy 触发监控阈值)
    3   输入/配置错误
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def _percentile(series: pd.Series, q: float) -> Optional[float]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(s) < 20:
        return None
    return float(s.quantile(q))


def _load_gate_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _extract_bull_conditional_rules(
    gate_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """提取 ``ema_1200_position`` 出现在 ``all_of`` 中（且阈值正数）的规则。"""
    rules: List[Dict[str, Any]] = []
    for section in ("system_safety", "hard_gates", "guardrails"):
        for rule in gate_cfg.get(section, []) or []:
            if not isinstance(rule, dict) or rule.get("disabled"):
                continue
            when = rule.get("when") or {}
            all_of = when.get("all_of") if isinstance(when, dict) else None
            if not isinstance(all_of, list):
                continue
            ema_threshold: Optional[float] = None
            feature_conds: Dict[str, Dict[str, float]] = {}
            for c in all_of:
                if not isinstance(c, dict):
                    continue
                for k, v in c.items():
                    if k in ("all_of", "any_of", "min_matches"):
                        continue
                    if not isinstance(v, dict):
                        continue
                    if k == "ema_1200_position":
                        if "value_gt" in v:
                            ema_threshold = float(v["value_gt"])
                    else:
                        existing = feature_conds.setdefault(str(k), {})
                        for op, val in v.items():
                            existing[op] = float(val)
            if ema_threshold is not None and ema_threshold > 0 and feature_conds:
                rules.append(
                    {
                        "id": rule.get("id"),
                        "ema_threshold": ema_threshold,
                        "feature_conds": feature_conds,
                    }
                )
    return rules


from src.research.stat_kernels.drift import (
    evaluate_ic_drift_vs_baseline,
    evaluate_psi_features,
)


def evaluate_factor_health(
    *,
    window_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
    ic_baseline: Dict[str, Any],
    psi_features: List[str],
    psi_tol: float,
    ic_flip_min_abs: float,
) -> Dict[str, Any]:
    """IC drift vs baseline JSON + PSI vs reference parquet."""
    ic_items, ic_alerts = evaluate_ic_drift_vs_baseline(
        window_df=window_df,
        ic_baseline=ic_baseline,
        ic_flip_min_abs=ic_flip_min_abs,
    )
    psi_items, psi_alerts = evaluate_psi_features(
        window_df=window_df,
        reference_df=reference_df,
        psi_features=psi_features,
        psi_tol=psi_tol,
    )
    items = ic_items + psi_items
    alerts = ic_alerts + psi_alerts
    return {"items": items, "alerts": alerts, "any_alert": bool(alerts)}


def _in_band(series: pd.Series, conds: Dict[str, float]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(True, index=s.index)
    for op, thr in conds.items():
        if op == "value_gt":
            mask &= s > thr
        elif op == "value_gte":
            mask &= s >= thr
        elif op == "value_lt":
            mask &= s < thr
        elif op == "value_lte":
            mask &= s <= thr
    return mask.fillna(False)


def evaluate_strategy(
    *,
    strategy: str,
    gate_cfg: Dict[str, Any],
    window_df: pd.DataFrame,
    baseline_bull_share: Optional[float],
    baseline_trigger_rates: Optional[Dict[str, float]],
    bull_share_tol: float,
    trigger_drift_tol_rel: float,
) -> Dict[str, Any]:
    """检查当前 window 内 H 设计的健康度。"""
    items: List[Dict[str, Any]] = []
    alerts: List[str] = []

    if "ema_1200_position" not in window_df.columns:
        return {
            "strategy": strategy,
            "skipped": "ema_1200_position missing in window",
            "items": [],
            "alerts": ["MISSING_FEATURE: ema_1200_position"],
        }

    ema = pd.to_numeric(window_df["ema_1200_position"], errors="coerce")
    bull_mask = (ema >= 0.10).fillna(False)
    bear_mask = (ema <= -0.10).fillna(False)
    n_total = int(ema.notna().sum())
    bull_share = float(bull_mask.mean()) if n_total else 0.0
    bear_share = float(bear_mask.mean()) if n_total else 0.0

    items.append(
        {
            "kind": "ema_distribution",
            "n": n_total,
            "mean": float(ema.mean()) if n_total else None,
            "p25": _percentile(ema, 0.25),
            "p50": _percentile(ema, 0.50),
            "p75": _percentile(ema, 0.75),
            "p90": _percentile(ema, 0.90),
            "bull_share": bull_share,
            "bear_share": bear_share,
            "neutral_share": 1.0 - bull_share - bear_share,
        }
    )

    if baseline_bull_share is not None:
        delta = bull_share - baseline_bull_share
        if abs(delta) > bull_share_tol:
            alerts.append(
                f"BULL_SHARE_DRIFT: {bull_share:.1%} vs baseline {baseline_bull_share:.1%}"
                f" (delta={delta:+.1%}, tol={bull_share_tol:+.1%})"
            )

    rules = _extract_bull_conditional_rules(gate_cfg)
    trigger_rates: Dict[str, float] = {}
    for rule in rules:
        rule_id = str(rule["id"])
        for feature, conds in rule["feature_conds"].items():
            if feature not in window_df.columns:
                items.append(
                    {
                        "kind": "rule_check",
                        "rule": rule_id,
                        "feature": feature,
                        "skipped": "feature missing",
                    }
                )
                continue
            in_band = _in_band(window_df[feature], conds)
            band_share = float(in_band.mean())
            bull_band_share = (
                float(in_band[bull_mask].mean()) if bull_mask.any() else 0.0
            )
            bear_band_share = (
                float(in_band[bear_mask].mean()) if bear_mask.any() else 0.0
            )
            actual_trigger = float((in_band & bull_mask).mean())
            trigger_rates[rule_id] = actual_trigger
            items.append(
                {
                    "kind": "rule_check",
                    "rule": rule_id,
                    "feature": feature,
                    "ema_threshold": rule["ema_threshold"],
                    "band_share_overall": band_share,
                    "band_share_bull": bull_band_share,
                    "band_share_bear": bear_band_share,
                    "actual_trigger_rate": actual_trigger,
                }
            )

    if baseline_trigger_rates:
        for rule_id, base in baseline_trigger_rates.items():
            cur = trigger_rates.get(rule_id)
            if cur is None or base <= 0:
                continue
            rel = (cur - base) / base
            if abs(rel) > trigger_drift_tol_rel:
                alerts.append(
                    f"TRIGGER_RATE_DRIFT: {rule_id} {cur:.2%} vs baseline {base:.2%}"
                    f" (rel={rel:+.0%}, tol=±{trigger_drift_tol_rel:.0%})"
                )

    return {
        "strategy": strategy,
        "n_window": n_total,
        "alerts": alerts,
        "any_alert": bool(alerts),
        "items": items,
        "trigger_rates": trigger_rates,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Regime watchdog (variant H)")
    p.add_argument("--strategies", default="tpc")
    p.add_argument(
        "--window-parquet",
        required=True,
        help="parquet of recent feature snapshots (last N bars).",
    )
    p.add_argument("--strategies-root", default="config/strategies")
    p.add_argument(
        "--baseline-json",
        default=None,
        help="optional path to baseline_bull_share + trigger_rates JSON "
        "(written by a prior offline calibration of the live window).",
    )
    p.add_argument(
        "--bull-share-tol",
        type=float,
        default=0.10,
        help="abs tolerance on bull_share drift vs baseline.",
    )
    p.add_argument(
        "--trigger-drift-tol-rel",
        type=float,
        default=0.50,
        help="relative tolerance on trigger_rate drift vs baseline.",
    )
    p.add_argument(
        "--out-dir",
        default="results/regime_watchdog",
    )
    p.add_argument(
        "--ic-baseline-json",
        default="config/monitoring/factor_ic_baseline_tpc_20260526.json",
        help="Factor IC baseline JSON; overridden by baseline factor_ic_baseline_ref if set.",
    )
    p.add_argument(
        "--psi-features",
        default="ema_1200_position,vol_persistence,vol_leverage_asymmetry",
        help="Comma-separated features for PSI vs reference parquet.",
    )
    p.add_argument("--psi-tol", type=float, default=0.25)
    p.add_argument(
        "--ic-flip-min-abs",
        type=float,
        default=0.02,
        help="Min |IC| on both sides to flag sign flip.",
    )
    args = p.parse_args()
    return run_watchdog(args)


def run_watchdog(args: argparse.Namespace) -> int:
    """Core logic for regime_watchdog (callable from other Python code)."""
    pq = Path(args.window_parquet)
    if not pq.is_absolute():
        pq = (PROJECT_ROOT / pq).resolve()
    if not pq.exists():
        print(f"ERROR: window parquet not found: {pq}", file=sys.stderr)
        return 3
    window_df = pd.read_parquet(pq)

    ic_baseline: Dict[str, Any] = {}
    ic_baseline_path: Optional[Path] = None
    reference_df: Optional[pd.DataFrame] = None

    baseline: Dict[str, Any] = {}
    if args.baseline_json:
        bp = Path(args.baseline_json)
        if not bp.is_absolute():
            bp = (PROJECT_ROOT / bp).resolve()
        if bp.exists():
            baseline = json.loads(bp.read_text(encoding="utf-8"))
            ref_rel = baseline.get("factor_ic_baseline_ref")
            if ref_rel:
                ic_baseline_path = Path(str(ref_rel))

    if ic_baseline_path is None and args.ic_baseline_json:
        ic_baseline_path = Path(args.ic_baseline_json)

    if ic_baseline_path is not None:
        if not ic_baseline_path.is_absolute():
            ic_baseline_path = (PROJECT_ROOT / ic_baseline_path).resolve()
        if ic_baseline_path.exists():
            ic_baseline = json.loads(ic_baseline_path.read_text(encoding="utf-8"))
            src = ic_baseline.get("source_parquet")
            if src:
                sp = Path(str(src))
                if sp.exists():
                    ref_cols = [
                        c.strip() for c in args.psi_features.split(",") if c.strip()
                    ] + [str(ic_baseline.get("target", "forward_rr"))]
                    ref_cols = list(dict.fromkeys(ref_cols))
                    try:
                        reference_df = pd.read_parquet(sp, columns=ref_cols)
                    except Exception:
                        reference_df = pd.read_parquet(sp)

    strategies_root = Path(args.strategies_root)
    if not strategies_root.is_absolute():
        strategies_root = (PROJECT_ROOT / strategies_root).resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.out_dir / ts).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    any_alert = False
    for s in [x.strip() for x in args.strategies.split(",") if x.strip()]:
        gate_yaml = strategies_root / s / "archetypes" / "gate.yaml"
        gate_cfg = _load_gate_yaml(gate_yaml)
        if not gate_cfg:
            reports.append(
                {
                    "strategy": s,
                    "skipped": f"gate.yaml not found at {gate_yaml}",
                    "alerts": ["MISSING_CONFIG"],
                    "any_alert": True,
                }
            )
            any_alert = True
            continue
        base_for_s = (baseline or {}).get(s) or {}
        result = evaluate_strategy(
            strategy=s,
            gate_cfg=gate_cfg,
            window_df=window_df,
            baseline_bull_share=base_for_s.get("bull_share"),
            baseline_trigger_rates=base_for_s.get("trigger_rates"),
            bull_share_tol=args.bull_share_tol,
            trigger_drift_tol_rel=args.trigger_drift_tol_rel,
        )
        reports.append(result)
        any_alert = any_alert or bool(result.get("any_alert"))

    factor_health: Dict[str, Any] = {}
    if ic_baseline:
        psi_feats = [x.strip() for x in args.psi_features.split(",") if x.strip()]
        factor_health = evaluate_factor_health(
            window_df=window_df,
            reference_df=reference_df,
            ic_baseline=ic_baseline,
            psi_features=psi_feats,
            psi_tol=args.psi_tol,
            ic_flip_min_abs=args.ic_flip_min_abs,
        )
        any_alert = any_alert or bool(factor_health.get("any_alert"))

    out_json = {
        "ts": ts,
        "window_parquet": str(pq),
        "ic_baseline_json": str(ic_baseline_path) if ic_baseline_path else None,
        "any_alert": any_alert,
        "factor_health": factor_health,
        "reports": reports,
    }
    (out_dir / "report.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines: List[str] = [
        f"regime_watchdog @ {ts}  window={pq.name}  alert={'YES' if any_alert else 'no'}"
    ]
    for r in reports:
        s = r.get("strategy", "?")
        if r.get("skipped"):
            lines.append(f"  [{s}] SKIPPED: {r['skipped']}")
            continue
        ema_item = next(
            (it for it in r.get("items", []) if it.get("kind") == "ema_distribution"),
            {},
        )
        lines.append(
            f"  [{s}] n={r.get('n_window')}"
            f"  ema p50={ema_item.get('p50', float('nan')):.3f}"
            f"  bull={ema_item.get('bull_share', 0):.1%}"
            f"  bear={ema_item.get('bear_share', 0):.1%}"
        )
        for rid, tr in r.get("trigger_rates", {}).items():
            lines.append(f"      rule {rid}: actual_trigger={tr:.2%}")
        for a in r.get("alerts") or []:
            lines.append(f"      ALERT: {a}")
    if factor_health:
        lines.append("  [factor_health]")
        for it in factor_health.get("items") or []:
            if it.get("kind") == "ic_drift":
                lines.append(
                    f"      IC {it['feature']}: {it['current_ic']:+.4f} "
                    f"(base {it['baseline_ic']:+.4f}, flip={it.get('sign_flip')})"
                )
            elif it.get("kind") == "psi" and it.get("psi") is not None:
                lines.append(f"      PSI {it['feature']}: {it['psi']:.3f}")
        for a in factor_health.get("alerts") or []:
            lines.append(f"      ALERT: {a}")
    print("\n".join(lines))

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    return 1 if any_alert else 0


if __name__ == "__main__":
    sys.exit(main())
