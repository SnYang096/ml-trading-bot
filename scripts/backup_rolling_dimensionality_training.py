#!/usr/bin/env python3
"""
Rolling Dimensionality Training Script
实现季度数据的滚动训练和降维协同系统

功能：
1. 使用2024年数据训练，2025年数据测试
2. 基于漂移的动态降维触发
3. 降维前后效果对比
4. 反馈闭环优化
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set matplotlib backend to avoid display issues
import matplotlib

matplotlib.use("Agg")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ml_trading.models.rolling_dimensionality_engine import RollingDimensionalityEngine
from ml_trading.models.interpretable_factor_engine import create_sample_data


def create_quarterly_data_splits():
    """
    创建季度数据分割配置
    """
    quarters = {
        "2024_Q1": {"start": "2024-01-01", "end": "2024-03-31"},
        "2024_Q2": {"start": "2024-04-01", "end": "2024-06-30"},
        "2024_Q3": {"start": "2024-07-01", "end": "2024-09-30"},
        "2024_Q4": {"start": "2024-10-01", "end": "2024-12-31"},
        "2025_Q1": {"start": "2025-01-01", "end": "2025-03-31"},
        "2025_Q2": {"start": "2025-04-01", "end": "2025-06-30"},
        "2025_Q3": {"start": "2025-07-01", "end": "2025-09-30"},
        "2025_Q4": {"start": "2025-10-01", "end": "2025-12-31"},
    }
    return quarters


def run_quarterly_rolling_training(args):
    """
    运行季度滚动训练
    """
    print("🚀 Starting Quarterly Rolling Dimensionality Training")
    print("=" * 70)
    print(f"Training Period: 2024 (Full Year)")
    print(f"Testing Period: 2025 (Full Year)")
    print(f"Symbol: {args.symbol}")
    print("=" * 70)

    # 初始化引擎
    engine = RollingDimensionalityEngine(
        encoding_dim=args.encoding_dim,
        drift_threshold=args.drift_threshold,
        min_improvement=args.min_improvement,
        model_save_dir=f"models/rolling_dim_{args.symbol.replace('-', '_')}",
        results_save_dir=f"results/rolling_dim_{args.symbol.replace('-', '_')}",
    )

    # 获取季度配置
    quarters = create_quarterly_data_splits()

    all_results = []

    # 1. 使用2024年全年数据训练
    print("\n📊 Phase 1: Training on 2024 Full Year Data")
    print("-" * 50)

    train_results = engine.run_rolling_training_with_dimensionality(
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        train_start="2024-01-01",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-03-31",  # 先用Q1测试
        symbol=args.symbol,
    )

    all_results.append({"phase": "2024_full_year_training", "results": train_results})

    # 2. 逐季度测试2025年数据
    print("\n📊 Phase 2: Quarterly Testing on 2025 Data")
    print("-" * 50)

    for quarter_name, quarter_dates in quarters.items():
        if not quarter_name.startswith("2025"):
            continue

        print(f"\n🔍 Testing on {quarter_name}")
        print(f"   Period: {quarter_dates['start']} to {quarter_dates['end']}")

        # 在测试季度上运行降维和评估
        quarter_results = engine.run_rolling_training_with_dimensionality(
            train_data_path=args.train_data_path,
            test_data_path=args.test_data_path,
            train_start="2024-01-01",  # 使用2024年数据训练
            train_end="2024-12-31",
            test_start=quarter_dates["start"],
            test_end=quarter_dates["end"],
            symbol=args.symbol,
        )

        all_results.append(
            {"phase": f"quarterly_test_{quarter_name}", "results": quarter_results}
        )

        print(f"✅ {quarter_name} testing complete")

    # 3. 生成综合报告
    print("\n📋 Phase 3: Generating Comprehensive Report")
    print("-" * 50)

    summary_report = generate_comprehensive_report(all_results, args.symbol)

    # 4. 保存最终结果
    save_final_results(all_results, summary_report, args.symbol)

    print("\n" + "=" * 70)
    print("🎉 Quarterly Rolling Dimensionality Training Complete!")
    print(f"📊 Total phases completed: {len(all_results)}")
    print(f"🎯 Overall performance summary available in results/")

    return all_results, summary_report


def generate_comprehensive_report(all_results, symbol):
    """
    生成综合报告
    """
    print("📋 Generating comprehensive report...")

    # 提取关键指标
    training_results = []
    testing_results = []

    for result in all_results:
        if "training" in result["phase"]:
            training_results.append(result["results"])
        else:
            testing_results.append(result["results"])

    # 计算统计指标
    if training_results:
        train_original_r2 = np.mean(
            [r["original_performance"].get("r2", 0) for r in training_results]
        )
        train_compressed_r2 = np.mean(
            [r["compressed_performance"].get("r2", 0) for r in training_results]
        )
        train_improvement = train_compressed_r2 - train_original_r2

        summary_stats = {
            "training_period": "2024 Full Year",
            "training_original_r2": train_original_r2,
            "training_compressed_r2": train_compressed_r2,
            "training_improvement": train_improvement,
            "compression_ratio": training_results[0]["compression_ratio"],
            "selected_factors": training_results[0]["selected_factors_count"],
        }
    else:
        summary_stats = {"error": "No training results found"}

    if testing_results:
        test_r2_scores = [r["test_performance"].get("r2", 0) for r in testing_results]
        summary_stats.update(
            {
                "testing_period": "2025 Quarters",
                "average_test_r2": np.mean(test_r2_scores),
                "test_r2_std": np.std(test_r2_scores),
                "test_r2_min": np.min(test_r2_scores),
                "test_r2_max": np.max(test_r2_scores),
            }
        )

    # 生成报告内容
    report = {
        "symbol": symbol,
        "timestamp": pd.Timestamp.now().isoformat(),
        "summary_statistics": summary_stats,
        "detailed_results": all_results,
        "recommendations": generate_recommendations(summary_stats),
    }

    return report


def generate_recommendations(summary_stats):
    """
    生成建议
    """
    recommendations = []

    if "training_improvement" in summary_stats:
        improvement = summary_stats["training_improvement"]
        if improvement > 0.01:
            recommendations.append(
                "✅ Dimensionality reduction shows significant improvement (>1%)"
            )
        elif improvement > 0.005:
            recommendations.append(
                "✅ Dimensionality reduction shows moderate improvement (>0.5%)"
            )
        else:
            recommendations.append(
                "⚠️ Dimensionality reduction shows minimal improvement"
            )

    if "average_test_r2" in summary_stats:
        test_r2 = summary_stats["average_test_r2"]
        if test_r2 > 0.6:
            recommendations.append("✅ Model shows good generalization on test data")
        elif test_r2 > 0.4:
            recommendations.append("⚠️ Model shows moderate generalization")
        else:
            recommendations.append(
                "❌ Model shows poor generalization - consider feature engineering"
            )

    if "compression_ratio" in summary_stats:
        ratio = summary_stats["compression_ratio"]
        if ratio > 5:
            recommendations.append(
                "✅ High compression ratio achieved - efficient feature reduction"
            )
        else:
            recommendations.append(
                "⚠️ Low compression ratio - consider increasing encoding dimensions"
            )

    recommendations.append("🔄 Continue monitoring feature drift and retrain quarterly")
    recommendations.append(
        "📊 Track model performance across different market conditions"
    )

    return recommendations


def save_final_results(all_results, summary_report, symbol):
    """
    保存最终结果
    """
    import json

    # 保存详细结果
    results_file = f"results/rolling_dim_{symbol.replace('-', '_')}/final_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(
            {"all_results": all_results, "summary_report": summary_report},
            f,
            indent=2,
            default=str,
        )

    # 保存摘要报告
    summary_file = f"results/rolling_dim_{symbol.replace('-', '_')}/summary_report.json"
    with open(summary_file, "w") as f:
        json.dump(summary_report, f, indent=2, default=str)

    print(f"💾 Final results saved to: {results_file}")
    print(f"💾 Summary report saved to: {summary_file}")


def run_drift_triggered_training(args):
    """
    运行基于漂移触发的训练
    """
    print("🚀 Starting Drift-Triggered Training")
    print("=" * 50)

    # 初始化引擎
    engine = RollingDimensionalityEngine(
        encoding_dim=args.encoding_dim,
        drift_threshold=args.drift_threshold,
        min_improvement=args.min_improvement,
    )

    # 运行漂移触发训练
    results = engine.run_drift_triggered_training(
        data_path=args.train_data_path,
        start_date="2024-01-01",
        end_date="2025-12-31",
        symbol=args.symbol,
        rolling_window_days=30,
    )

    print("🎉 Drift-triggered training complete!")
    return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Rolling Dimensionality Training with Quarterly Data"
    )

    # Data parameters
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/train_2024.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="data/test_2025.csv",
        help="Path to test data",
    )
    parser.add_argument("--symbol", type=str, default="ETH-USD", help="Trading symbol")

    # Model parameters
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=8,
        help="Encoding dimension for dimensionality reduction",
    )
    parser.add_argument(
        "--drift-threshold", type=float, default=0.3, help="Drift detection threshold"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        help="Minimum improvement threshold",
    )

    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quarterly", "drift-triggered"],
        default="quarterly",
        help="Training mode: quarterly or drift-triggered",
    )

    args = parser.parse_args()

    if args.mode == "quarterly":
        # 运行季度滚动训练
        all_results, summary_report = run_quarterly_rolling_training(args)

        # 打印摘要
        print("\n📊 Final Summary:")
        print("-" * 30)
        if "summary_statistics" in summary_report:
            stats = summary_report["summary_statistics"]
            print(
                f"Training R²: {stats.get('training_original_r2', 0):.3f} → {stats.get('training_compressed_r2', 0):.3f}"
            )
            print(f"Improvement: {stats.get('training_improvement', 0):.3f}")
            print(f"Test R²: {stats.get('average_test_r2', 0):.3f}")
            print(f"Compression Ratio: {stats.get('compression_ratio', 0):.1f}x")

        print("\n🎯 Recommendations:")
        for rec in summary_report.get("recommendations", []):
            print(f"  {rec}")

    elif args.mode == "drift-triggered":
        # 运行漂移触发训练
        results = run_drift_triggered_training(args)
        print(f"Results: {results}")


if __name__ == "__main__":
    main()
