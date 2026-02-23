#!/usr/bin/env python3
"""
周频快速检查 — L2 (Prefilter) + L3 (Direction) + L4 (Gate) + 特征漂移快检

读取 training_baseline.json 作为基准，对最新数据执行快速健康检查。
适合每周一次运行，~30 秒完成。

用法:
    python scripts/local_monitor_weekly.py \
        --data data/live_latest.parquet \
        --strategy me \
        --baseline results/train_final_xxx/me/training_baseline.json

    # 指定输出路径
    python scripts/local_monitor_weekly.py \
        --data data/live_latest.parquet \
        --strategy me \
        --baseline results/train_final_xxx/me/training_baseline.json \
        --output reports/weekly_health_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ====================================================================
# Threshold configuration
# ====================================================================

# 偏离阈值 (训练基线值 × ratio)
THRESHOLDS = {
    "prefilter_pass_rate_warn": 0.30,  # 偏离 30% → 🟡
    "prefilter_pass_rate_alert": 0.50,  # 偏离 50% → 🔴
    "gate_pass_rate_warn": 0.30,
    "gate_pass_rate_alert": 0.50,
    "direction_coverage_warn": 0.05,  # 绝对值偏离 5% → 🟡
    "direction_coverage_alert": 0.10,  # 绝对值偏离 10% → 🔴
    "drift_features_warn": 0.15,  # 15% 特征漂移 → 🟡
    "drift_features_alert": 0.30,  # 30% 特征漂移 → 🔴
}


def _status(deviation: float, warn: float, alert: float) -> str:
    """根据偏离度判断状态."""
    if abs(deviation) >= alert:
        return "🔴 ALERT"
    elif abs(deviation) >= warn:
        return "🟡 WARN"
    return "🟢 OK"


# ====================================================================
# L2: Prefilter 通过率检查
# ====================================================================


def check_l2_prefilter(
    df,  # pd.DataFrame
    strategy: str,
    baseline_kpi: Dict[str, Any],
    config_root: str = "config/strategies",
) -> Dict[str, Any]:
    """检查 prefilter 规则在新数据上的通过率."""
    # 加载 prefilter rules
    rules_path = PROJECT_ROOT / config_root / strategy / "archetypes" / "prefilter.yaml"
    if not rules_path.exists():
        return {"status": "⚪ SKIP", "reason": "no prefilter.yaml"}

    try:
        data = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"status": "⚪ SKIP", "reason": "parse error"}

    rules = data.get("rules", [])
    if not rules:
        return {"status": "⚪ SKIP", "reason": "no rules defined"}

    # 评估每条规则
    mask = np.ones(len(df), dtype=bool)
    rule_results = []

    for rule in rules:
        if isinstance(rule, dict) and "feature" in rule:
            feat = rule["feature"]
            op = rule.get("operator", ">=")
            val = rule.get("value", 0)

            if feat not in df.columns:
                rule_results.append(
                    {
                        "feature": feat,
                        "status": "MISSING",
                        "pass_rate": None,
                    }
                )
                continue

            col = df[feat]
            if op == ">=":
                rule_mask = col >= val
            elif op == ">":
                rule_mask = col > val
            elif op == "<=":
                rule_mask = col <= val
            elif op == "<":
                rule_mask = col < val
            elif op == "==":
                rule_mask = col == val
            else:
                rule_mask = np.ones(len(df), dtype=bool)

            rate = float(rule_mask.mean())
            mask &= rule_mask.values
            rule_results.append(
                {
                    "feature": feat,
                    "operator": op,
                    "value": val,
                    "pass_rate": round(rate, 4),
                }
            )

    overall_pass_rate = float(mask.mean())

    # 对比 baseline
    baseline_bad_rate = baseline_kpi.get("baseline_bad_rate")
    # prefilter 通过率的 baseline (如果有的话)
    baseline_pass_rate = baseline_kpi.get("pass_rate")

    deviation = 0.0
    if baseline_pass_rate and baseline_pass_rate > 0:
        deviation = (overall_pass_rate - baseline_pass_rate) / baseline_pass_rate

    status = _status(
        deviation,
        THRESHOLDS["prefilter_pass_rate_warn"],
        THRESHOLDS["prefilter_pass_rate_alert"],
    )

    return {
        "layer": "L2_prefilter",
        "status": status,
        "current_pass_rate": round(overall_pass_rate, 4),
        "baseline_pass_rate": baseline_pass_rate,
        "deviation": round(deviation, 4) if baseline_pass_rate else None,
        "n_rules": len(rules),
        "rule_details": rule_results,
    }


# ====================================================================
# L3: Direction 覆盖率检查
# ====================================================================


def check_l3_direction(
    df,  # pd.DataFrame
    baseline_kpi: Dict[str, Any],
) -> Dict[str, Any]:
    """检查方向分配覆盖率和 short 比例."""
    if "entry_direction" not in df.columns:
        return {"status": "⚪ SKIP", "reason": "no entry_direction column"}

    n_total = len(df)
    if n_total == 0:
        return {"status": "⚪ SKIP", "reason": "empty data"}

    has_direction = (df["entry_direction"] != 0).sum()
    coverage = float(has_direction / n_total)
    short_count = (df["entry_direction"] == -1).sum()
    short_ratio = float(short_count / n_total)

    baseline_coverage = baseline_kpi.get("coverage", 1.0)
    baseline_short = baseline_kpi.get("short_ratio", 0.0)

    cov_deviation = baseline_coverage - coverage  # 正数 = 覆盖率下降
    short_deviation = abs(short_ratio - baseline_short)

    cov_status = _status(
        cov_deviation,
        THRESHOLDS["direction_coverage_warn"],
        THRESHOLDS["direction_coverage_alert"],
    )

    return {
        "layer": "L3_direction",
        "status": cov_status,
        "current_coverage": round(coverage, 4),
        "baseline_coverage": baseline_coverage,
        "coverage_deviation": round(cov_deviation, 4),
        "current_short_ratio": round(short_ratio, 4),
        "baseline_short_ratio": baseline_short,
        "short_deviation": round(short_deviation, 4),
    }


# ====================================================================
# L4: Gate 通过率检查
# ====================================================================


def check_l4_gate(
    df,  # pd.DataFrame
    strategy: str,
    baseline_kpi: Dict[str, Any],
    config_root: str = "config/strategies",
) -> Dict[str, Any]:
    """检查 gate 通过率."""
    # 加载 gate rules 并在新数据上评估
    gate_path = PROJECT_ROOT / config_root / strategy / "archetypes" / "gate.yaml"
    if not gate_path.exists():
        return {"status": "⚪ SKIP", "reason": "no gate.yaml"}

    try:
        gate_data = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"status": "⚪ SKIP", "reason": "parse error"}

    hard_gates = gate_data.get("hard_gates", [])
    if not hard_gates:
        return {"status": "⚪ SKIP", "reason": "no hard_gates"}

    # 评估 gate (简化版: 只看 deny 规则)
    allowed_mask = np.ones(len(df), dtype=bool)
    rule_results = []

    for gate in hard_gates:
        when_clause = gate.get("when", {})
        for feat, condition in when_clause.items():
            if not isinstance(condition, dict):
                continue
            for op_key, threshold in condition.items():
                if feat not in df.columns:
                    rule_results.append(
                        {
                            "feature": feat,
                            "status": "MISSING",
                        }
                    )
                    continue

                col = df[feat]
                if op_key == "value_lt":
                    deny_mask = col < threshold
                elif op_key == "value_gt":
                    deny_mask = col > threshold
                elif op_key == "value_le":
                    deny_mask = col <= threshold
                elif op_key == "value_ge":
                    deny_mask = col >= threshold
                else:
                    continue

                denied = float(deny_mask.mean())
                allowed_mask &= ~deny_mask.values
                rule_results.append(
                    {
                        "gate_id": gate.get("id", "unknown"),
                        "feature": feat,
                        "operator": op_key,
                        "threshold": threshold,
                        "deny_rate": round(denied, 4),
                    }
                )

    pass_rate = float(allowed_mask.mean())

    # 对比 baseline
    baseline_pass_rate = baseline_kpi.get("pass_rate")
    deviation = 0.0
    if baseline_pass_rate and baseline_pass_rate > 0:
        deviation = (pass_rate - baseline_pass_rate) / baseline_pass_rate

    status = _status(
        deviation, THRESHOLDS["gate_pass_rate_warn"], THRESHOLDS["gate_pass_rate_alert"]
    )

    return {
        "layer": "L4_gate",
        "status": status,
        "current_pass_rate": round(pass_rate, 4),
        "baseline_pass_rate": baseline_pass_rate,
        "deviation": round(deviation, 4) if baseline_pass_rate else None,
        "n_rules_evaluated": len(rule_results),
        "rule_details": rule_results,
    }


# ====================================================================
# 特征漂移快检 (简化版)
# ====================================================================


def check_feature_drift_quick(
    df,  # pd.DataFrame
    baseline_distributions: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """快速特征漂移检测 (只看均值偏移和 NaN 率)."""
    if not baseline_distributions:
        return {"status": "⚪ SKIP", "reason": "no baseline distributions"}

    n_checked = 0
    n_drifted = 0
    drifted_features = []

    for feat, stats in baseline_distributions.items():
        if feat not in df.columns:
            continue

        n_checked += 1
        old_mean = stats.get("mean", 0)
        old_std = stats.get("std", 1)
        old_nan = stats.get("nan_rate", 0)

        col = df[feat]
        valid = col.dropna()
        if len(valid) < 10:
            continue

        new_mean = float(valid.mean())
        new_nan = float(col.isna().mean())

        # 均值偏移 (标准化)
        mean_shift = (
            abs(new_mean - old_mean) / max(old_std, 1e-8) if old_std > 1e-8 else 0
        )
        nan_drift = abs(new_nan - old_nan)

        if mean_shift > 2.0 or nan_drift > 0.05:
            n_drifted += 1
            drifted_features.append(
                {
                    "feature": feat,
                    "mean_shift_std": round(mean_shift, 3),
                    "nan_drift": round(nan_drift, 4),
                }
            )

    drift_rate = n_drifted / max(n_checked, 1)
    status = _status(
        drift_rate,
        THRESHOLDS["drift_features_warn"],
        THRESHOLDS["drift_features_alert"],
    )

    return {
        "layer": "L1_feature_drift",
        "status": status,
        "features_checked": n_checked,
        "features_drifted": n_drifted,
        "drift_rate": round(drift_rate, 4),
        "top_drifted": sorted(
            drifted_features, key=lambda x: x["mean_shift_std"], reverse=True
        )[:10],
    }


# ====================================================================
# Main
# ====================================================================


def run_weekly_check(
    data_path: Path,
    strategy: str,
    baseline_path: Path,
    *,
    config_root: str = "config/strategies",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """执行周频快速检查."""
    import pandas as pd

    # 加载 baseline
    if not baseline_path.exists():
        print(f"❌ Baseline 不存在: {baseline_path}")
        return {"error": "baseline_not_found"}

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    layer_kpis = baseline.get("layer_kpis", {})
    distributions = baseline.get("feature_distributions", {})

    # 加载新数据
    if not data_path.exists():
        print(f"❌ 数据不存在: {data_path}")
        return {"error": "data_not_found"}

    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df)} rows from {data_path.name}")

    # ── 逐层检查 ──
    checks: List[Dict[str, Any]] = []

    # L1: 特征漂移快检
    drift = check_feature_drift_quick(df, distributions)
    checks.append(drift)

    # L2: Prefilter
    l2_kpi = layer_kpis.get("L2_prefilter", {})
    prefilter = check_l2_prefilter(df, strategy, l2_kpi, config_root)
    checks.append(prefilter)

    # L3: Direction
    l3_kpi = layer_kpis.get("L3_direction", {})
    direction = check_l3_direction(df, l3_kpi)
    checks.append(direction)

    # L4: Gate
    l4_kpi = layer_kpis.get("L4_gate", {})
    gate = check_l4_gate(df, strategy, l4_kpi, config_root)
    checks.append(gate)

    # ── 汇总 ──
    statuses = [c.get("status", "⚪ SKIP") for c in checks]
    has_alert = any("ALERT" in s for s in statuses)
    has_warn = any("WARN" in s for s in statuses)

    overall = (
        "🔴 NEEDS RETRAIN"
        if has_alert
        else "🟡 NEEDS ATTENTION" if has_warn else "🟢 HEALTHY"
    )

    report = {
        "report_type": "weekly_health_check",
        "date": str(date.today()),
        "strategy": strategy,
        "data_source": str(data_path.name),
        "data_rows": len(df),
        "baseline_version": baseline.get("version", "unknown"),
        "overall_status": overall,
        "checks": checks,
    }

    # ── 打印 ──
    print(f"\n{'='*60}")
    print(f"📊 周频健康检查 — {strategy.upper()}")
    print(f"{'='*60}")
    print(f"   数据: {data_path.name} ({len(df):,} rows)")
    print(f"   基线: {baseline.get('version', 'unknown')}")
    print(f"   状态: {overall}")
    print()

    for c in checks:
        layer = c.get("layer", c.get("status", "?"))
        status = c.get("status", "?")
        detail = ""
        if "current_pass_rate" in c:
            detail = f"pass_rate={c['current_pass_rate']:.1%}"
            if c.get("baseline_pass_rate"):
                detail += f" (baseline={c['baseline_pass_rate']:.1%})"
        elif "current_coverage" in c:
            detail = f"coverage={c['current_coverage']:.1%}, short={c.get('current_short_ratio', 0):.1%}"
        elif "drift_rate" in c:
            detail = f"drift_rate={c['drift_rate']:.1%} ({c.get('features_drifted', 0)}/{c.get('features_checked', 0)})"
        elif "reason" in c:
            detail = c["reason"]

        print(f"   {status:<14} {layer:<20} {detail}")

    # ── 保存 ──
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\n   ✅ Report saved: {output_path}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="周频快速检查 (L1-L4)")
    parser.add_argument("--data", required=True, help="新数据 parquet")
    parser.add_argument("--strategy", required=True, help="策略名 (bpc/fer/me)")
    parser.add_argument("--baseline", required=True, help="训练基线 JSON")
    parser.add_argument(
        "--config-root", default="config/strategies", help="策略配置根目录"
    )
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None

    report = run_weekly_check(
        data_path=Path(args.data),
        strategy=args.strategy,
        baseline_path=Path(args.baseline),
        config_root=args.config_root,
        output_path=output,
    )

    # Exit code based on status
    status = report.get("overall_status", "")
    if "RETRAIN" in status:
        return 2
    elif "ATTENTION" in status:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
