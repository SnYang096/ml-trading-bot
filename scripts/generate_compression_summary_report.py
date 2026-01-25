#!/usr/bin/env python3
"""
生成压缩优化汇总报告

合并所有archetype的优化结果，应用优化后的规则，计算E2E KPI，生成对比报告。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_optimization_results(results_file: Path) -> Dict[str, Any]:
    """加载优化结果"""
    if not results_file.exists():
        return {}
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_all_optimization_results(output_dir: Path) -> Dict[str, Any]:
    """合并所有archetype的优化结果"""
    all_results = {}

    for arch in ["TC", "TE", "FR", "ET"]:
        result_file = output_dir / f"{arch.lower()}_optimization.json"
        if result_file.exists():
            arch_results = load_optimization_results(result_file)
            all_results.update(arch_results)
            print(f"✅ 加载 {arch} 优化结果: {len(arch_results)} 个规则")
        else:
            print(f"⚠️  {arch} 优化结果文件不存在: {result_file}")

    return all_results


def apply_optimized_rules_to_logs(
    logs_df: pd.DataFrame,
    optimization_results: Dict[str, Any],
    execution_archetypes_path: str,
) -> pd.Series:
    """应用优化后的规则到logs"""
    from src.time_series_model.nnmultihead.strategy_profile import (
        load_execution_archetypes_registry,
    )
    from scripts.apply_archetype_gate import apply_gate_rules

    arches = load_execution_archetypes_registry(execution_archetypes_path)

    # 更新archetypes配置中的阈值
    for key, result in optimization_results.items():
        arch_name = result.get("archetype")
        rule_name = result.get("rule_name")
        recommended_threshold = result.get("recommended_threshold")

        if not arch_name or not rule_name or recommended_threshold is None:
            continue

        arch = arches.get(arch_name)
        if not arch or not arch.gate_rules:
            continue

        rules = arch.gate_rules.get("rules", [])
        for rule in rules:
            if rule.get("name") == rule_name:
                if "threshold" in rule:
                    rule["threshold"] = recommended_threshold
                elif "quantile" in rule:
                    rule["quantile"] = recommended_threshold
                break

    # 应用gate规则
    gate_ok_series = pd.Series(False, index=logs_df.index)

    for idx, row in logs_df.iterrows():
        features = row.to_dict()

        # 尝试每个archetype
        for arch_name, arch in arches.items():
            if not arch.gate_rules:
                gate_ok_series.loc[idx] = True
                break

            ok, _ = apply_gate_rules(
                gate_rules=arch.gate_rules,
                features=features,
                quantiles=None,
            )

            if ok:
                gate_ok_series.loc[idx] = True
                break

    return gate_ok_series


def calculate_e2e_kpi(logs_df: pd.DataFrame, gate_ok: pd.Series) -> Dict[str, float]:
    """计算E2E KPI"""
    gated_df = logs_df[gate_ok].copy()

    if len(gated_df) == 0:
        return {
            "trade_rate": 0.0,
            "total_trades": 0,
            "sharpe_e2e": 0.0,
            "win_rate": 0.0,
            "ret_mean": 0.0,
        }

    # 计算基本指标
    total_rows = len(logs_df)
    trade_rows = len(gated_df)
    trade_rate = trade_rows / total_rows if total_rows > 0 else 0.0

    # 计算收益指标
    if "ret_mean" in gated_df.columns:
        returns = gated_df["ret_mean"].dropna()
        if len(returns) > 0:
            ret_mean = float(returns.mean())
            ret_std = float(returns.std())
            sharpe_e2e = (ret_mean / ret_std * (240**0.5)) if ret_std > 0 else 0.0

            # 胜率
            wins = (returns > 0).sum()
            win_rate = wins / len(returns) if len(returns) > 0 else 0.0
        else:
            ret_mean = 0.0
            sharpe_e2e = 0.0
            win_rate = 0.0
    else:
        ret_mean = 0.0
        sharpe_e2e = 0.0
        win_rate = 0.0

    return {
        "trade_rate": trade_rate,
        "total_trades": trade_rows,
        "sharpe_e2e": sharpe_e2e,
        "win_rate": win_rate,
        "ret_mean": ret_mean,
    }


def generate_summary_report(
    baseline_kpi: Dict[str, float],
    optimized_kpi: Dict[str, float],
    optimization_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """生成汇总报告"""

    # 计算变化
    sharpe_change = optimized_kpi["sharpe_e2e"] - baseline_kpi["sharpe_e2e"]
    trade_rate_change = optimized_kpi["trade_rate"] - baseline_kpi["trade_rate"]
    win_rate_change = optimized_kpi["win_rate"] - baseline_kpi["win_rate"]

    # 生成Markdown报告
    report = f"""# 压缩优化汇总报告

## 优化概述

本报告对比了压缩优化前后的性能指标。

### 优化统计

- **TC规则优化**: {len([k for k in optimization_results.keys() if 'TrendContinuationTC' in k or k.startswith('TC_')])} 个
- **TE规则优化**: {len([k for k in optimization_results.keys() if 'TrendExpansion' in k or k.startswith('TE_')])} 个
- **FR规则优化**: {len([k for k in optimization_results.keys() if 'FailureReversion' in k or k.startswith('FR_')])} 个
- **ET规则优化**: {len([k for k in optimization_results.keys() if 'ExhaustionTurn' in k or k.startswith('ET_')])} 个
- **总计**: {len(optimization_results)} 个规则

## KPI对比

| 指标 | Baseline | 优化后 | 变化 | 变化率 |
|------|----------|--------|------|--------|
| **Sharpe比率** | {baseline_kpi['sharpe_e2e']:.4f} | {optimized_kpi['sharpe_e2e']:.4f} | {sharpe_change:+.4f} | {sharpe_change/baseline_kpi['sharpe_e2e']*100 if baseline_kpi['sharpe_e2e'] != 0 else 0:+.2f}% |
| **交易率** | {baseline_kpi['trade_rate']:.4f} | {optimized_kpi['trade_rate']:.4f} | {trade_rate_change:+.4f} | {trade_rate_change/baseline_kpi['trade_rate']*100 if baseline_kpi['trade_rate'] != 0 else 0:+.2f}% |
| **总交易数** | {baseline_kpi['total_trades']} | {optimized_kpi['total_trades']} | {optimized_kpi['total_trades'] - baseline_kpi['total_trades']:+d} | - |
| **胜率** | {baseline_kpi['win_rate']:.4f} | {optimized_kpi['win_rate']:.4f} | {win_rate_change:+.4f} | {win_rate_change/baseline_kpi['win_rate']*100 if baseline_kpi['win_rate'] != 0 else 0:+.2f}% |
| **平均收益** | {baseline_kpi['ret_mean']:.6f} | {optimized_kpi['ret_mean']:.6f} | {optimized_kpi['ret_mean'] - baseline_kpi['ret_mean']:+.6f} | - |

## 关键发现

### Sharpe比率变化

"""

    if sharpe_change > 0:
        report += f"✅ **Sharpe比率提升**: {sharpe_change:.4f} ({sharpe_change/baseline_kpi['sharpe_e2e']*100 if baseline_kpi['sharpe_e2e'] != 0 else 0:+.2f}%)\n\n"
        report += "压缩优化成功提升了策略的Sharpe比率，说明优化后的gate规则更有效地过滤了低质量交易。\n\n"
    else:
        report += f"⚠️ **Sharpe比率下降**: {sharpe_change:.4f} ({sharpe_change/baseline_kpi['sharpe_e2e']*100 if baseline_kpi['sharpe_e2e'] != 0 else 0:+.2f}%)\n\n"
        report += "压缩优化导致Sharpe比率下降，可能需要调整压缩目标或优化参数。\n\n"

    report += f"""### 交易率变化

"""

    if trade_rate_change < 0:
        report += f"✅ **交易率压缩**: {abs(trade_rate_change):.4f} ({abs(trade_rate_change)/baseline_kpi['trade_rate']*100 if baseline_kpi['trade_rate'] != 0 else 0:.2f}%)\n\n"
        report += f"成功压缩了过度交易，从 {baseline_kpi['trade_rate']:.2%} 降低到 {optimized_kpi['trade_rate']:.2%}。\n\n"
    else:
        report += f"⚠️ **交易率未压缩**: {trade_rate_change:+.4f}\n\n"

    report += f"""## 优化规则详情

### 规则阈值变化统计

"""

    # 统计规则变化
    threshold_changes = []
    for key, result in optimization_results.items():
        current = result.get("current_threshold", 0)
        recommended = result.get("recommended_threshold", 0)
        if current is not None and recommended is not None:
            threshold_changes.append(
                {
                    "rule": result.get("rule_name", key),
                    "archetype": result.get("archetype", ""),
                    "current": current,
                    "recommended": recommended,
                    "change": recommended - current,
                }
            )

    if threshold_changes:
        report += "| 规则 | Archetype | 当前阈值 | 优化后阈值 | 变化 |\n"
        report += "|------|-----------|----------|------------|------|\n"
        for change in threshold_changes[:20]:  # 只显示前20个
            report += f"| {change['rule']} | {change['archetype']} | {change['current']:.4f} | {change['recommended']:.4f} | {change['change']:+.4f} |\n"
        if len(threshold_changes) > 20:
            report += f"\n... 还有 {len(threshold_changes) - 20} 个规则未显示\n"

    report += f"""
## 结论

"""

    if sharpe_change > 0 and abs(trade_rate_change) > 0.01:
        report += "✅ **压缩优化成功**: Sharpe比率提升，同时有效压缩了过度交易。\n"
    elif sharpe_change > 0:
        report += "✅ **Sharpe比率提升**: 优化有效，但交易率压缩不明显。\n"
    elif sharpe_change < 0:
        report += "⚠️ **需要调整**: Sharpe比率下降，建议重新评估压缩目标和优化参数。\n"
    else:
        report += "ℹ️ **无明显变化**: 优化未产生显著效果。\n"

    # 保存报告
    report_file = output_dir / "compression_summary_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 汇总报告已保存: {report_file}")

    # 保存JSON格式的KPI对比
    kpi_comparison = {
        "baseline": baseline_kpi,
        "optimized": optimized_kpi,
        "changes": {
            "sharpe_e2e": sharpe_change,
            "trade_rate": trade_rate_change,
            "win_rate": win_rate_change,
            "total_trades": optimized_kpi["total_trades"]
            - baseline_kpi["total_trades"],
        },
        "optimization_stats": {
            "total_rules_optimized": len(optimization_results),
            "tc_rules": len(
                [
                    k
                    for k in optimization_results.keys()
                    if "TrendContinuationTC" in k or k.startswith("TC_")
                ]
            ),
            "te_rules": len(
                [
                    k
                    for k in optimization_results.keys()
                    if "TrendExpansion" in k or k.startswith("TE_")
                ]
            ),
            "fr_rules": len(
                [
                    k
                    for k in optimization_results.keys()
                    if "FailureReversion" in k or k.startswith("FR_")
                ]
            ),
            "et_rules": len(
                [
                    k
                    for k in optimization_results.keys()
                    if "ExhaustionTurn" in k or k.startswith("ET_")
                ]
            ),
        },
    }

    kpi_file = output_dir / "kpi_comparison.json"
    with open(kpi_file, "w", encoding="utf-8") as f:
        json.dump(kpi_comparison, f, indent=2, default=str)

    print(f"✅ KPI对比数据已保存: {kpi_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成压缩优化汇总报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="results/compression_optimization_all",
        help="输出目录（包含所有archetype的优化结果）",
    )
    parser.add_argument(
        "--raw-logs",
        required=True,
        help="原始logs文件（parquet）",
    )
    parser.add_argument(
        "--execution-archetypes",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="execution_archetypes.yaml路径",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有优化结果
    print("📋 加载所有archetype的优化结果...")
    optimization_results = merge_all_optimization_results(output_dir)

    if len(optimization_results) == 0:
        print("❌ 未找到任何优化结果")
        return 1

    print(f"✅ 共加载 {len(optimization_results)} 个优化规则\n")

    # 加载原始logs
    print("📊 加载原始logs...")
    logs_df = pd.read_parquet(args.raw_logs)
    print(f"✅ 加载 {len(logs_df)} 行数据\n")

    # 计算baseline KPI（全松阈值，所有交易都通过）
    print("📊 计算Baseline KPI...")
    baseline_gate_ok = pd.Series(True, index=logs_df.index)  # 全松，所有都通过
    baseline_kpi = calculate_e2e_kpi(logs_df, baseline_gate_ok)
    print(
        f"✅ Baseline KPI: Sharpe={baseline_kpi['sharpe_e2e']:.4f}, Trade Rate={baseline_kpi['trade_rate']:.4f}\n"
    )

    # 应用优化后的规则
    print("📊 应用优化后的规则...")
    try:
        optimized_gate_ok = apply_optimized_rules_to_logs(
            logs_df,
            optimization_results,
            args.execution_archetypes,
        )
        optimized_kpi = calculate_e2e_kpi(logs_df, optimized_gate_ok)
        print(
            f"✅ 优化后 KPI: Sharpe={optimized_kpi['sharpe_e2e']:.4f}, Trade Rate={optimized_kpi['trade_rate']:.4f}\n"
        )
    except Exception as e:
        print(f"❌ 应用优化规则失败: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # 生成汇总报告
    print("📝 生成汇总报告...")
    generate_summary_report(
        baseline_kpi,
        optimized_kpi,
        optimization_results,
        output_dir,
    )

    print("\n✅ 汇总报告生成完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
