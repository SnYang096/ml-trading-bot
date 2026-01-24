#!/usr/bin/env python3
"""
对比Gate优化实验结果

加载所有实验结果，对比KPI指标，生成Markdown报告。

使用方法:
    python scripts/compare_gate_optimization_experiments.py \
        --results-file results/gate_optimization_experiments/all_experiments_results.json \
        --output-dir results/gate_optimization_experiments
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


def load_experiment_results(results_file: Path) -> Dict[str, Any]:
    """加载实验结果"""
    if not results_file.exists():
        raise FileNotFoundError(f"实验结果文件不存在: {results_file}")

    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_comparison_report(
    all_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """生成对比报告"""

    # 提取KPI数据
    kpi_data = []
    for exp_name, exp_result in all_results.items():
        if "error" in exp_result:
            continue

        kpi = exp_result.get("kpi", {})
        kpi_data.append(
            {
                "实验": exp_name,
                "交易率": kpi.get("trade_rate", 0.0),
                "胜率": kpi.get("win_rate", 0.0),
                "平均收益": kpi.get("avg_return", 0.0),
                "Sharpe比率": kpi.get("sharpe_ratio", 0.0),
                "最大回撤": kpi.get("max_drawdown", 0.0),
                "总交易数": kpi.get("total_trades", 0),
            }
        )

    if not kpi_data:
        print("⚠️  没有可用的实验结果")
        return

    # 创建DataFrame
    df_kpi = pd.DataFrame(kpi_data)

    # 生成Markdown报告
    report_lines = [
        "# Gate优化实验对比报告",
        "",
        "## 实验概述",
        "",
        "本报告对比了以下优化方法的性能：",
        "",
    ]

    # 列出所有实验
    for exp_name in all_results.keys():
        if "error" not in all_results[exp_name]:
            report_lines.append(
                f"- **{exp_name}**: {all_results[exp_name].get('experiment', exp_name)}"
            )
        else:
            report_lines.append(
                f"- **{exp_name}**: ❌ 失败 - {all_results[exp_name].get('error', 'Unknown error')}"
            )

    report_lines.extend(
        [
            "",
            "## KPI对比表",
            "",
            df_kpi.to_markdown(index=False),
            "",
            "## 详细分析",
            "",
        ]
    )

    # 找出最佳实验
    if len(df_kpi) > 0:
        best_sharpe = df_kpi.loc[df_kpi["Sharpe比率"].idxmax()]
        best_trade_rate = df_kpi.loc[df_kpi["交易率"].idxmax()]
        best_win_rate = df_kpi.loc[df_kpi["胜率"].idxmax()]
        lowest_dd = df_kpi.loc[df_kpi["最大回撤"].idxmin()]

        report_lines.extend(
            [
                "### 最佳表现",
                "",
                f"- **最高Sharpe比率**: {best_sharpe['实验']} (Sharpe={best_sharpe['Sharpe比率']:.4f})",
                f"- **最高交易率**: {best_trade_rate['实验']} (交易率={best_trade_rate['交易率']:.4f})",
                f"- **最高胜率**: {best_win_rate['实验']} (胜率={best_win_rate['胜率']:.4f})",
                f"- **最低回撤**: {lowest_dd['实验']} (回撤={lowest_dd['最大回撤']:.4f})",
                "",
                "### 推荐策略",
                "",
            ]
        )

        # 综合评分
        df_kpi["综合评分"] = (
            df_kpi["Sharpe比率"] * 0.4
            + df_kpi["交易率"] * 0.3
            + df_kpi["胜率"] * 0.2
            - df_kpi["最大回撤"] * 0.1
        )

        best_overall = df_kpi.loc[df_kpi["综合评分"].idxmax()]
        report_lines.append(
            f"**推荐使用**: {best_overall['实验']} (综合评分={best_overall['综合评分']:.4f})"
        )
        report_lines.extend(
            [
                "",
                "综合评分 = Sharpe比率×0.4 + 交易率×0.3 + 胜率×0.2 - 最大回撤×0.1",
                "",
            ]
        )

    # 添加优化规则统计
    report_lines.extend(
        [
            "## 优化规则统计",
            "",
        ]
    )

    for exp_name, exp_result in all_results.items():
        if "error" in exp_result:
            continue

        opt_results = exp_result.get("optimization_results", {})
        if opt_results:
            total_rules = sum(
                len(rules) if isinstance(rules, list) else 0
                for rules in opt_results.values()
            )
            report_lines.append(f"- **{exp_name}**: 优化了 {total_rules} 个规则")

    report_lines.append("")

    # 保存报告
    report_text = "\n".join(report_lines)
    report_file = output_dir / "comparison_report.md"
    report_file.write_text(report_text, encoding="utf-8")

    # 保存KPI数据为CSV
    csv_file = output_dir / "kpi_comparison.csv"
    df_kpi.to_csv(csv_file, index=False, encoding="utf-8")

    print(f"✅ 对比报告已生成:")
    print(f"   - Markdown: {report_file}")
    print(f"   - CSV: {csv_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对比Gate优化实验结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-file",
        required=True,
        help="实验结果JSON文件",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载实验结果
    print("📊 加载实验结果...")
    results_file = Path(args.results_file)
    all_results = load_experiment_results(results_file)

    print(f"✅ 加载了 {len(all_results)} 个实验结果")

    # 生成对比报告
    print("\n📝 生成对比报告...")
    generate_comparison_report(all_results, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
