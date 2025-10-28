#!/usr/bin/env python3
"""
Generate Rolling Dimensionality Report
生成滚动降维训练的综合报告
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime


def generate_rolling_dim_report():
    """
    生成滚动降维训练的综合报告
    """
    print("📋 Generating Rolling Dimensionality Report")
    print("=" * 50)

    # 查找所有滚动降维结果
    results_dirs = glob.glob("results/rolling_dim_*")

    if not results_dirs:
        print("❌ No rolling dimensionality results found")
        return None

    print(f"📊 Found {len(results_dirs)} result directories")

    all_reports = []

    for results_dir in results_dirs:
        symbol = results_dir.split("_")[-1]
        print(f"\n🔍 Processing {symbol}...")

        # 查找摘要报告
        summary_files = glob.glob(f"{results_dir}/summary_report.json")
        final_result_files = glob.glob(f"{results_dir}/final_results_*.json")

        symbol_report = {
            "symbol": symbol,
            "summary_reports": [],
            "final_results": [],
            "timestamp": datetime.now().isoformat(),
        }

        # 读取摘要报告
        for summary_file in summary_files:
            try:
                with open(summary_file, "r") as f:
                    summary_data = json.load(f)
                    symbol_report["summary_reports"].append(summary_data)
            except Exception as e:
                print(f"❌ Error reading {summary_file}: {e}")

        # 读取最终结果
        for result_file in final_result_files:
            try:
                with open(result_file, "r") as f:
                    result_data = json.load(f)
                    symbol_report["final_results"].append(result_data)
            except Exception as e:
                print(f"❌ Error reading {result_file}: {e}")

        all_reports.append(symbol_report)

    # 生成综合报告
    comprehensive_report = generate_comprehensive_analysis(all_reports)

    # 保存报告
    save_report(comprehensive_report)

    print("\n✅ Rolling dimensionality report generated!")
    return comprehensive_report


def generate_comprehensive_analysis(all_reports):
    """
    生成综合分析报告
    """
    print("\n📊 Generating comprehensive analysis...")

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_symbols": len(all_reports),
        "symbols": [report["symbol"] for report in all_reports],
        "summary_statistics": {},
        "performance_trends": {},
        "recommendations": [],
        "detailed_reports": all_reports,
    }

    # 提取性能统计
    performance_metrics = []

    for report in all_reports:
        for summary in report["summary_reports"]:
            if "summary_statistics" in summary:
                stats = summary["summary_statistics"]
                performance_metrics.append(
                    {
                        "symbol": report["symbol"],
                        "training_r2": stats.get("training_compressed_r2", 0),
                        "test_r2": stats.get("average_test_r2", 0),
                        "improvement": stats.get("training_improvement", 0),
                        "compression_ratio": stats.get("compression_ratio", 0),
                        "selected_factors": stats.get("selected_factors", 0),
                    }
                )

    if performance_metrics:
        df = pd.DataFrame(performance_metrics)

        analysis["summary_statistics"] = {
            "average_training_r2": df["training_r2"].mean(),
            "average_test_r2": df["test_r2"].mean(),
            "average_improvement": df["improvement"].mean(),
            "average_compression_ratio": df["compression_ratio"].mean(),
            "average_selected_factors": df["selected_factors"].mean(),
            "best_performing_symbol": (
                df.loc[df["test_r2"].idxmax(), "symbol"] if len(df) > 0 else None
            ),
            "highest_improvement_symbol": (
                df.loc[df["improvement"].idxmax(), "symbol"] if len(df) > 0 else None
            ),
        }

        # 生成建议
        analysis["recommendations"] = generate_recommendations(df)

    return analysis


def generate_recommendations(df):
    """
    基于性能数据生成建议
    """
    recommendations = []

    if len(df) == 0:
        return ["No performance data available for recommendations"]

    # 性能建议
    avg_test_r2 = df["test_r2"].mean()
    if avg_test_r2 > 0.6:
        recommendations.append("✅ Overall model performance is good (R² > 0.6)")
    elif avg_test_r2 > 0.4:
        recommendations.append("⚠️ Model performance is moderate (R² > 0.4)")
    else:
        recommendations.append("❌ Model performance needs improvement (R² < 0.4)")

    # 改进建议
    avg_improvement = df["improvement"].mean()
    if avg_improvement > 0.01:
        recommendations.append(
            "✅ Dimensionality reduction shows significant improvement"
        )
    elif avg_improvement > 0.005:
        recommendations.append("✅ Dimensionality reduction shows moderate improvement")
    else:
        recommendations.append("⚠️ Dimensionality reduction shows minimal improvement")

    # 压缩比建议
    avg_compression = df["compression_ratio"].mean()
    if avg_compression > 5:
        recommendations.append("✅ High compression ratio achieved")
    else:
        recommendations.append("⚠️ Consider increasing compression ratio")

    # 最佳实践建议
    recommendations.extend(
        [
            "🔄 Continue monitoring feature drift across all symbols",
            "📊 Implement automated retraining when drift is detected",
            "🎯 Focus on symbols with highest improvement potential",
            "⚡ Optimize feature selection thresholds for better performance",
        ]
    )

    return recommendations


def save_report(report):
    """
    保存报告到文件
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = (
        f"reports/rolling_dimensionality_comprehensive_report_{timestamp}.json"
    )

    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"💾 Comprehensive report saved to: {report_file}")

    # 生成简化的摘要报告
    summary_file = f"reports/rolling_dimensionality_summary_{timestamp}.txt"
    generate_summary_text(report, summary_file)

    print(f"💾 Summary report saved to: {summary_file}")


def generate_summary_text(report, filename):
    """
    生成文本格式的摘要报告
    """
    with open(filename, "w") as f:
        f.write("Rolling Dimensionality Training Comprehensive Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Generated: {report['timestamp']}\n")
        f.write(f"Symbols Analyzed: {report['total_symbols']}\n")
        f.write(f"Symbols: {', '.join(report['symbols'])}\n\n")

        if "summary_statistics" in report and report["summary_statistics"]:
            stats = report["summary_statistics"]
            f.write("Performance Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Training R²: {stats.get('average_training_r2', 0):.3f}\n")
            f.write(f"Average Test R²: {stats.get('average_test_r2', 0):.3f}\n")
            f.write(f"Average Improvement: {stats.get('average_improvement', 0):.3f}\n")
            f.write(
                f"Average Compression Ratio: {stats.get('average_compression_ratio', 0):.1f}x\n"
            )
            f.write(
                f"Average Selected Factors: {stats.get('average_selected_factors', 0):.0f}\n"
            )
            f.write(
                f"Best Performing Symbol: {stats.get('best_performing_symbol', 'N/A')}\n"
            )
            f.write(
                f"Highest Improvement Symbol: {stats.get('highest_improvement_symbol', 'N/A')}\n\n"
            )

        f.write("Recommendations:\n")
        f.write("-" * 30 + "\n")
        for i, rec in enumerate(report["recommendations"], 1):
            f.write(f"{i}. {rec}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Report generated by Rolling Dimensionality Training System\n")


if __name__ == "__main__":
    report = generate_rolling_dim_report()

    if report:
        print("\n📊 Report Summary:")
        print("-" * 30)
        print(f"Symbols analyzed: {report['total_symbols']}")
        print(f"Symbols: {', '.join(report['symbols'])}")

        if "summary_statistics" in report and report["summary_statistics"]:
            stats = report["summary_statistics"]
            print(f"Average Test R²: {stats.get('average_test_r2', 0):.3f}")
            print(f"Average Improvement: {stats.get('average_improvement', 0):.3f}")
            print(f"Best Symbol: {stats.get('best_performing_symbol', 'N/A')}")

        print("\n🎯 Top Recommendations:")
        for i, rec in enumerate(report["recommendations"][:3], 1):
            print(f"  {i}. {rec}")

        print("\n✅ Comprehensive report generation complete!")
    else:
        print("❌ No data available for report generation")
