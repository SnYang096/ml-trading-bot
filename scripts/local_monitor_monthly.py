#!/usr/bin/env python3
"""
月频全层报告 — L1-L9 逐层检查 + 训练基线对比 + 重训建议

完整验证所有层假设，对比 training_baseline.json 基线。
适合每月运行一次，输出结构化 JSON 报告。

用法:
    python scripts/local_monitor_monthly.py \
        --data data/monthly.parquet \
        --strategy me \
        --baseline results/train_final_xxx/me/training_baseline.json \
        --output reports/monthly_full_report.json
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
# Reuse weekly checks
# ====================================================================

from scripts.local_monitor_weekly import (
    check_l2_prefilter,
    check_l3_direction,
    check_l4_gate,
    check_feature_drift_quick,
)


# ====================================================================
# L5: Evidence 有效性检查
# ====================================================================


def check_l5_evidence(
    df,  # pd.DataFrame
    strategy: str,
    baseline_kpi: Dict[str, Any],
    config_root: str = "config/strategies",
) -> Dict[str, Any]:
    """检查 evidence score 与实际收益的相关性."""
    # 加载 evidence 配置
    ev_path = PROJECT_ROOT / config_root / strategy / "archetypes" / "evidence.yaml"
    if not ev_path.exists():
        return {
            "layer": "L5_evidence",
            "status": "⚪ SKIP",
            "reason": "no evidence.yaml",
        }

    try:
        ev_data = yaml.safe_load(ev_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"layer": "L5_evidence", "status": "⚪ SKIP", "reason": "parse error"}

    evidence_list = ev_data.get("evidence", [])
    if not evidence_list:
        return {
            "layer": "L5_evidence",
            "status": "⚪ SKIP",
            "reason": "no evidence features",
        }

    # 寻找 RR 列
    rr_col = None
    for candidate in ["bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"]:
        if candidate in df.columns:
            rr_col = candidate
            break

    if rr_col is None:
        return {"layer": "L5_evidence", "status": "⚪ SKIP", "reason": "no rr column"}

    # 检查每个 evidence 特征
    feature_checks = []
    from scipy import stats as sp_stats

    for ev in evidence_list:
        feat = ev.get("feature", "")
        if feat not in df.columns:
            feature_checks.append(
                {
                    "feature": feat,
                    "status": "MISSING",
                }
            )
            continue

        valid = df[[feat, rr_col]].dropna()
        if len(valid) < 30:
            feature_checks.append(
                {
                    "feature": feat,
                    "status": "SKIP",
                    "reason": "too few samples",
                }
            )
            continue

        # Spearman 相关
        try:
            corr, p_val = sp_stats.spearmanr(valid[feat], valid[rr_col])
        except Exception:
            corr, p_val = 0.0, 1.0

        feature_checks.append(
            {
                "feature": feat,
                "spearman": round(float(corr), 4),
                "p_value": round(float(p_val), 6),
                "status": (
                    "🔴" if abs(corr) < 0.03 else "🟡" if abs(corr) < 0.05 else "🟢"
                ),
            }
        )

    # 整体判断
    valid_checks = [c for c in feature_checks if "spearman" in c]
    if not valid_checks:
        return {
            "layer": "L5_evidence",
            "status": "⚪ SKIP",
            "reason": "no valid evidence checks",
        }

    avg_corr = float(np.mean([abs(c["spearman"]) for c in valid_checks]))
    n_weak = sum(1 for c in valid_checks if abs(c["spearman"]) < 0.03)

    baseline_avg = baseline_kpi.get("avg_bad_suppression", 0)

    status = "🟢 OK"
    if n_weak > len(valid_checks) * 0.5:
        status = "🔴 ALERT"
    elif n_weak > 0:
        status = "🟡 WARN"

    return {
        "layer": "L5_evidence",
        "status": status,
        "avg_abs_spearman": round(avg_corr, 4),
        "n_weak_features": n_weak,
        "n_checked": len(valid_checks),
        "baseline_avg_bad_suppression": baseline_avg,
        "feature_details": feature_checks,
    }


# ====================================================================
# L6: Entry Filter 检查
# ====================================================================


def check_l6_entry_filter(
    df,  # pd.DataFrame
    strategy: str,
    baseline_kpi: Dict[str, Any],
    config_root: str = "config/strategies",
) -> Dict[str, Any]:
    """检查 entry filter 在新数据上的通过率和 lift."""
    ef_path = (
        PROJECT_ROOT / config_root / strategy / "archetypes" / "entry_filters.yaml"
    )
    if not ef_path.exists():
        return {
            "layer": "L6_entry_filter",
            "status": "⚪ SKIP",
            "reason": "no entry_filters.yaml",
        }

    try:
        ef_data = yaml.safe_load(ef_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {
            "layer": "L6_entry_filter",
            "status": "⚪ SKIP",
            "reason": "parse error",
        }

    filters = ef_data.get("filters", ef_data.get("conditions", []))
    if not isinstance(filters, list):
        return {"layer": "L6_entry_filter", "status": "⚪ SKIP", "reason": "no filters"}

    # RR column
    rr_col = None
    for candidate in ["bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"]:
        if candidate in df.columns:
            rr_col = candidate
            break

    filter_results = []
    for f in filters:
        conditions = f.get("conditions", [])
        if not conditions:
            continue

        mask = np.ones(len(df), dtype=bool)
        for cond in conditions:
            feat = cond.get("feature", "")
            op = cond.get("operator", ">=")
            val = cond.get("value", 0)

            if feat not in df.columns:
                mask[:] = False
                break

            col = df[feat]
            if op == ">=":
                mask &= (col >= val).values
            elif op == ">":
                mask &= (col > val).values
            elif op == "<=":
                mask &= (col <= val).values
            elif op == "<":
                mask &= (col < val).values
            elif op == "==":
                mask &= (col == val).values

        pass_rate = float(mask.mean())
        result_entry: Dict[str, Any] = {
            "name": f.get("name", "unknown"),
            "pass_rate": round(pass_rate, 4),
            "n_passed": int(mask.sum()),
        }

        # lift vs baseline
        if rr_col and mask.sum() > 10:
            mean_passed = float(df.loc[mask, rr_col].mean())
            mean_all = float(df[rr_col].mean())
            lift = mean_passed / mean_all if abs(mean_all) > 1e-8 else 0
            result_entry["mean_rr_passed"] = round(mean_passed, 4)
            result_entry["lift_vs_all"] = round(float(lift), 4)

        # 对比 baseline
        bt = f.get("backtest", {})
        if bt.get("trades"):
            result_entry["baseline_trades"] = bt["trades"]

        filter_results.append(result_entry)

    status = "🟢 OK"
    if filter_results:
        low_pass = sum(1 for r in filter_results if r["pass_rate"] < 0.3)
        if low_pass > len(filter_results) * 0.5:
            status = "🔴 ALERT"
        elif low_pass > 0:
            status = "🟡 WARN"

    return {
        "layer": "L6_entry_filter",
        "status": status,
        "n_filters": len(filter_results),
        "filter_details": filter_results,
    }


# ====================================================================
# L7: Execution 层检查 (需要 backtest)
# ====================================================================


def check_l7_execution(baseline_kpi: Dict[str, Any]) -> Dict[str, Any]:
    """L7 检查: 对比基线的 backtest 指标 (仅展示, 需重跑 backtest 才能验证)."""
    if not baseline_kpi:
        return {
            "layer": "L7_execution",
            "status": "⚪ SKIP",
            "reason": "no baseline execution KPI",
        }

    return {
        "layer": "L7_execution",
        "status": "ℹ️ INFO",
        "note": "需重跑 backtest_execution_layer.py 才能完整验证",
        "baseline_sharpe": baseline_kpi.get("sharpe_per_trade"),
        "baseline_trades": baseline_kpi.get("total_trades"),
        "baseline_win_rate": baseline_kpi.get("win_rate"),
    }


# ====================================================================
# L8: PCM 检查
# ====================================================================


def check_l8_pcm(baseline_kpi: Dict[str, Any]) -> Dict[str, Any]:
    """L8 PCM 检查: 需要多策略数据, 目前只展示基线."""
    return {
        "layer": "L8_pcm",
        "status": "⚪ SKIP",
        "note": "PCM 检查需要多策略实盘数据, 单策略无法验证",
    }


# ====================================================================
# L9: 宪法层检查
# ====================================================================


def check_l9_constitution(
    config_root: str = "config",
) -> Dict[str, Any]:
    """L9: 检查宪法配置是否存在."""
    const_path = PROJECT_ROOT / config_root / "constitution"
    if not const_path.exists():
        return {
            "layer": "L9_constitution",
            "status": "⚪ SKIP",
            "reason": "no constitution dir",
        }

    yamls = list(const_path.glob("*.yaml"))
    return {
        "layer": "L9_constitution",
        "status": "🟢 OK" if yamls else "🟡 WARN",
        "n_configs": len(yamls),
        "files": [f.name for f in yamls],
    }


# ====================================================================
# 重训建议生成
# ====================================================================


def generate_retrain_advice(checks: List[Dict[str, Any]]) -> List[str]:
    """根据检查结果生成重训建议."""
    advice: List[str] = []
    for c in checks:
        status = c.get("status", "")
        layer = c.get("layer", "")

        if "ALERT" not in status and "🔴" not in status:
            continue

        if "L1" in layer:
            advice.append("L1 特征漂移严重 → 检查数据源是否变化 (通常不需要重训)")
        elif "L2" in layer:
            advice.append(
                "L2 Prefilter 偏离 → 重跑 Step 3 (analyze_archetype_feature_stratification.py)"
            )
        elif "L3" in layer:
            advice.append(
                "L3 Direction 异常 → 重跑 Step 4 (direction_strict_validation.py)"
            )
        elif "L4" in layer:
            advice.append(
                "L4 Gate 偏离 → 重跑 Step 5 (Gate 训练 + optimize_gate_unified.py)"
            )
        elif "L5" in layer:
            advice.append(
                "L5 Evidence 弱化 → 重跑 Step 6 (optimize_evidence_plateau.py)"
            )
        elif "L6" in layer:
            advice.append(
                "L6 Entry Filter 异常 → 重跑 Step 7 (optimize_entry_filter_plateau.py)"
            )

    if not advice:
        advice.append("所有层正常，无需重训")

    return advice


# ====================================================================
# Main
# ====================================================================


def run_monthly_report(
    data_path: Path,
    strategy: str,
    baseline_path: Path,
    *,
    config_root: str = "config/strategies",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """执行月频全层报告."""
    import pandas as pd

    # 加载 baseline
    if not baseline_path.exists():
        print(f"❌ Baseline 不存在: {baseline_path}")
        return {"error": "baseline_not_found"}

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    layer_kpis = baseline.get("layer_kpis", {})
    distributions = baseline.get("feature_distributions", {})

    # 加载数据
    if not data_path.exists():
        print(f"❌ 数据不存在: {data_path}")
        return {"error": "data_not_found"}

    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df)} rows from {data_path.name}")

    # ── 逐层检查 ──
    checks: List[Dict[str, Any]] = []

    # L1: 特征漂移 (完整版)
    try:
        from scripts.local_monitor_feature_drift import analyze_drift

        drift_report = analyze_drift(distributions, data_path, top_n=15)
        drift_summary = drift_report.get("summary", {})
        checks.append(
            {
                "layer": "L1_feature_drift",
                "status": drift_summary.get("overall_status", "⚪ SKIP"),
                "features_checked": drift_summary.get("features_checked", 0),
                "features_drifted": drift_summary.get("features_drifted", 0),
                "features_severe": drift_summary.get("features_severe", 0),
                "drift_rate": drift_summary.get("drift_rate", 0),
                "top_drifted": drift_report.get("top_drifted", [])[:10],
            }
        )
    except Exception as e:
        # Fallback to quick check
        drift = check_feature_drift_quick(df, distributions)
        checks.append(drift)

    # L2: Prefilter
    checks.append(
        check_l2_prefilter(
            df, strategy, layer_kpis.get("L2_prefilter", {}), config_root
        )
    )

    # L3: Direction
    checks.append(check_l3_direction(df, layer_kpis.get("L3_direction", {})))

    # L4: Gate
    checks.append(
        check_l4_gate(df, strategy, layer_kpis.get("L4_gate", {}), config_root)
    )

    # L5: Evidence
    checks.append(
        check_l5_evidence(df, strategy, layer_kpis.get("L5_evidence", {}), config_root)
    )

    # L6: Entry Filter
    checks.append(
        check_l6_entry_filter(
            df, strategy, layer_kpis.get("L6_entry_filter", {}), config_root
        )
    )

    # L7: Execution
    checks.append(check_l7_execution(layer_kpis.get("L7_execution", {})))

    # L8: PCM
    checks.append(check_l8_pcm(layer_kpis.get("L8_pcm", {})))

    # L9: Constitution
    checks.append(check_l9_constitution())

    # ── 汇总 ──
    statuses = [c.get("status", "⚪ SKIP") for c in checks]
    n_alert = sum(1 for s in statuses if "ALERT" in s or "🔴" in s)
    n_warn = sum(1 for s in statuses if "WARN" in s or "🟡" in s)
    n_ok = sum(1 for s in statuses if "OK" in s or "🟢" in s)

    overall = (
        "🔴 NEEDS RETRAIN"
        if n_alert >= 2
        else "🟡 NEEDS ATTENTION" if n_alert >= 1 or n_warn >= 3 else "🟢 HEALTHY"
    )

    retrain_advice = generate_retrain_advice(checks)

    report = {
        "report_type": "monthly_full_report",
        "date": str(date.today()),
        "strategy": strategy,
        "data_source": str(data_path.name),
        "data_rows": len(df),
        "baseline_version": baseline.get("version", "unknown"),
        "overall_status": overall,
        "summary": {
            "n_layers_checked": len(checks),
            "n_alert": n_alert,
            "n_warn": n_warn,
            "n_ok": n_ok,
        },
        "retrain_advice": retrain_advice,
        "checks": checks,
    }

    # ── 打印 ──
    print(f"\n{'='*70}")
    print(f"📊 月频全层报告 — {strategy.upper()}")
    print(f"{'='*70}")
    print(f"   数据: {data_path.name} ({len(df):,} rows)")
    print(f"   基线: {baseline.get('version', 'unknown')}")
    print(f"   状态: {overall}")
    print(f"   (🟢 {n_ok} / 🟡 {n_warn} / 🔴 {n_alert})")
    print()

    for c in checks:
        layer = c.get("layer", "?")
        status = c.get("status", "?")
        # 构建 detail
        detail_parts: List[str] = []
        if "current_pass_rate" in c:
            detail_parts.append(f"pass={c['current_pass_rate']:.1%}")
        if "current_coverage" in c:
            detail_parts.append(f"cov={c['current_coverage']:.1%}")
        if "drift_rate" in c:
            detail_parts.append(f"drift={c['drift_rate']:.1%}")
        if "avg_abs_spearman" in c:
            detail_parts.append(f"avg_spearman={c['avg_abs_spearman']:.4f}")
        if "n_filters" in c:
            detail_parts.append(f"filters={c['n_filters']}")
        if "baseline_sharpe" in c and c["baseline_sharpe"]:
            detail_parts.append(f"baseline_sharpe={c['baseline_sharpe']:.4f}")
        if "n_configs" in c:
            detail_parts.append(f"configs={c['n_configs']}")
        if "reason" in c:
            detail_parts.append(c["reason"])
        if "note" in c:
            detail_parts.append(c["note"])

        detail = ", ".join(detail_parts) if detail_parts else ""
        print(f"   {status:<14} {layer:<22} {detail}")

    # 重训建议
    print(f"\n   📋 重训建议:")
    for a in retrain_advice:
        print(f"      → {a}")

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
    parser = argparse.ArgumentParser(description="月频全层报告 (L1-L9)")
    parser.add_argument("--data", required=True, help="月度数据 parquet")
    parser.add_argument("--strategy", required=True, help="策略名 (bpc/fer/me)")
    parser.add_argument("--baseline", required=True, help="训练基线 JSON")
    parser.add_argument(
        "--config-root", default="config/strategies", help="策略配置根目录"
    )
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None

    report = run_monthly_report(
        data_path=Path(args.data),
        strategy=args.strategy,
        baseline_path=Path(args.baseline),
        config_root=args.config_root,
        output_path=output,
    )

    status = report.get("overall_status", "")
    if "RETRAIN" in status:
        return 2
    elif "ATTENTION" in status:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
