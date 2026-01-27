#!/usr/bin/env python3
"""
生成 FR 优化方法对比报告

对比：
1. 平坦高原搜索（Plateau Optimization）
2. Optuna 优化

综合两个结果，生成对比报告
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_plateau_results(plateau_json_path: str) -> Optional[Dict[str, Any]]:
    """加载平坦高原搜索结果"""
    if not Path(plateau_json_path).exists():
        return None

    with open(plateau_json_path) as f:
        return json.load(f)


def load_optuna_results(optuna_json_path: str) -> Optional[Dict[str, Any]]:
    """加载 Optuna 优化结果"""
    if not Path(optuna_json_path).exists():
        return None

    with open(optuna_json_path) as f:
        return json.load(f)


def generate_comparison_report(
    plateau_results: Optional[Dict[str, Any]],
    optuna_results: Optional[Dict[str, Any]],
    output_md: str,
    output_json: str,
) -> None:
    """生成对比报告"""

    lines = []
    lines.append("# FR Gate 规则优化方法对比报告\n")
    lines.append("本报告对比了两种优化方法的结果：\n")
    lines.append("1. **平坦高原搜索（Plateau Optimization）**\n")
    lines.append("2. **Optuna 贝叶斯优化**\n")
    lines.append("\n---\n")

    # 平坦高原结果
    lines.append("## 1. 平坦高原搜索结果\n")
    if plateau_results:
        # 计算总体指标
        total_rules = len(plateau_results)
        optimized_rules = sum(
            1
            for v in plateau_results.values()
            if v.get("recommended_threshold") is not None
        )

        lines.append(f"- **优化规则数**: {optimized_rules} / {total_rules}\n")

        # 计算平均指标
        robustness_scores = [
            v.get("robustness_score", 0)
            for v in plateau_results.values()
            if v.get("robustness_score") is not None
        ]
        trade_rates = [
            v.get("trade_rate", 0)
            for v in plateau_results.values()
            if v.get("trade_rate") is not None
        ]

        if robustness_scores:
            lines.append(
                f"- **平均 Robustness Score**: {sum(robustness_scores) / len(robustness_scores):.4f}\n"
            )
            lines.append(f"- **最佳 Robustness Score**: {max(robustness_scores):.4f}\n")

        if trade_rates:
            lines.append(
                f"- **平均 Trade Rate**: {sum(trade_rates) / len(trade_rates):.4f}\n"
            )

        # 显示关键规则
        lines.append("\n### 关键规则优化结果\n")
        lines.append("| 规则名称 | 推荐阈值 | Robustness Score | Trade Rate |\n")
        lines.append("|---------|---------|-----------------|-----------|\n")

        # 按优先级排序
        key_rules = [
            "fr_path_efficiency_too_high",
            "fr_price_dir_consistency_too_high",
            "fr_deviation_too_low",
            "fr_not_mean_regime_path_length_too_low",
            "fr_not_mean_regime_atr_percentile_too_low",
            "fr_not_mean_regime_jump_risk_too_high",
        ]

        for rule_name in key_rules:
            if rule_name in plateau_results:
                result = plateau_results[rule_name]
                threshold = result.get("recommended_threshold", "N/A")
                robustness = result.get("robustness_score", "N/A")
                trade_rate = result.get("trade_rate", "N/A")

                if isinstance(threshold, (int, float)):
                    threshold = f"{threshold:.4f}"
                if isinstance(robustness, (int, float)):
                    robustness = f"{robustness:.4f}"
                if isinstance(trade_rate, (int, float)):
                    trade_rate = f"{trade_rate:.4f}"

                lines.append(
                    f"| {rule_name} | {threshold} | {robustness} | {trade_rate} |\n"
                )
    else:
        lines.append("⚠️ 未找到平坦高原搜索结果\n")

    lines.append("\n---\n")

    # Optuna 结果
    lines.append("## 2. Optuna 优化结果\n")
    if optuna_results:
        best_value = optuna_results.get("best_value", "N/A")
        best_params = optuna_results.get("best_params", {})
        best_attrs = optuna_results.get("best_trial_attrs", {})
        n_trials = optuna_results.get("n_trials", "N/A")

        lines.append(f"- **最佳 Sharpe**: {best_value:.4f}\n")
        lines.append(f"- **Trade Rate**: {best_attrs.get('trade_rate', 'N/A'):.4f}\n")
        lines.append(f"- **交易数**: {best_attrs.get('n_trades', 'N/A')}\n")
        lines.append(f"- **Trial 数量**: {n_trials}\n")

        lines.append("\n### 最佳参数组合\n")
        lines.append("| 参数名称 | 最佳值 |\n")
        lines.append("|---------|--------|\n")

        for param_name, param_value in sorted(best_params.items()):
            if isinstance(param_value, (int, float)):
                lines.append(f"| {param_name} | {param_value:.4f} |\n")
            else:
                lines.append(f"| {param_name} | {param_value} |\n")

        # 显示前5个最佳 trial
        trials_summary = optuna_results.get("trials_summary", [])
        if trials_summary:
            lines.append("\n### 前5个最佳 Trial\n")
            lines.append("| Trial | Sharpe | Trade Rate | N Trades |\n")
            lines.append("|-------|--------|-----------|----------|\n")

            for trial in trials_summary[:5]:
                value = trial.get("value", "N/A")
                attrs = trial.get("attrs", {})
                trade_rate = attrs.get("trade_rate", "N/A")
                n_trades = attrs.get("n_trades", "N/A")

                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
                if isinstance(trade_rate, (int, float)):
                    trade_rate = f"{trade_rate:.4f}"

                lines.append(
                    f"| {trial.get('trial_number', 'N/A')} | {value} | {trade_rate} | {n_trades} |\n"
                )
    else:
        lines.append("⚠️ 未找到 Optuna 优化结果\n")

    lines.append("\n---\n")

    # 对比分析
    lines.append("## 3. 方法对比分析\n")

    if plateau_results and optuna_results:
        lines.append("### 优化目标对比\n")
        lines.append(
            "- **平坦高原**: 优化 Robustness Score (min Sharpe across buckets)\n"
        )
        lines.append("- **Optuna**: 优化平均 Sharpe 比率\n")
        lines.append("\n### 优化方法对比\n")
        lines.append("- **平坦高原**: 网格搜索，系统扫描所有阈值\n")
        lines.append("- **Optuna**: 贝叶斯优化，智能采样\n")
        lines.append("\n### 适用场景\n")
        lines.append("- **平坦高原**: 适合需要鲁棒性的场景（最差情况最好）\n")
        lines.append("- **Optuna**: 适合需要快速找到近似最优解的场景\n")

    lines.append("\n---\n")

    # 推荐
    lines.append("## 4. 推荐\n")

    if plateau_results and optuna_results:
        plateau_robustness_scores = [
            v.get("robustness_score", -1000)
            for v in plateau_results.values()
            if v.get("robustness_score") is not None
        ]
        plateau_best_robustness = (
            max(plateau_robustness_scores) if plateau_robustness_scores else -1000
        )
        optuna_best_sharpe = optuna_results.get("best_value", -1000)

        if plateau_best_robustness > 0 and optuna_best_sharpe > 0:
            lines.append("### 参数选择建议\n")
            lines.append("1. **如果优先考虑鲁棒性**：使用平坦高原搜索的结果\n")
            lines.append("2. **如果优先考虑平均性能**：使用 Optuna 的结果\n")
            lines.append(
                "3. **综合方案**：结合两种方法，选择在两种方法中都表现好的参数\n"
            )
        elif optuna_best_sharpe > 0:
            lines.append("✅ **推荐使用 Optuna 结果**：找到了正 Sharpe 的参数组合\n")
        elif plateau_best_robustness > 0:
            lines.append(
                "✅ **推荐使用平坦高原结果**：找到了正 Robustness Score 的参数组合\n"
            )
        else:
            lines.append("⚠️ **两种方法都未找到正 Sharpe 的参数组合**\n")
            lines.append("建议：\n")
            lines.append("1. 检查数据质量\n")
            lines.append("2. 放宽约束条件（min_trade_rate, min_trades）\n")
            lines.append("3. 考虑调整 FR 策略本身\n")

    # 保存 Markdown
    output_md_path = Path(output_md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # 保存 JSON
    comparison_json = {
        "plateau_results": plateau_results,
        "optuna_results": optuna_results,
        "summary": {
            "plateau_optimized_rules": len(plateau_results) if plateau_results else 0,
            "optuna_best_sharpe": (
                optuna_results.get("best_value") if optuna_results else None
            ),
            "optuna_n_trials": (
                optuna_results.get("n_trials") if optuna_results else None
            ),
        },
    }

    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_json, f, indent=2, ensure_ascii=False)

    print(f"✅ 对比报告已生成:")
    print(f"   Markdown: {output_md_path}")
    print(f"   JSON: {output_json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 FR 优化方法对比报告")
    parser.add_argument(
        "--plateau-results",
        default="results/fr_parameter_search_2024.json",
        help="平坦高原搜索结果 JSON 文件",
    )
    parser.add_argument(
        "--optuna-results",
        default="results/fr_optuna_optimization.json",
        help="Optuna 优化结果 JSON 文件",
    )
    parser.add_argument(
        "--output-md",
        default="results/fr_optimization_comparison_report.md",
        help="输出 Markdown 报告",
    )
    parser.add_argument(
        "--output-json",
        default="results/fr_optimization_comparison_report.json",
        help="输出 JSON 报告",
    )

    args = parser.parse_args()

    # 加载结果
    plateau_results = load_plateau_results(args.plateau_results)
    optuna_results = load_optuna_results(args.optuna_results)

    if not plateau_results and not optuna_results:
        print("❌ 未找到任何优化结果")
        return 1

    # 生成报告
    generate_comparison_report(
        plateau_results,
        optuna_results,
        args.output_md,
        args.output_json,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
