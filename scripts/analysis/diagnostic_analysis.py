"""深度诊断分析 - 检查胜率、特征重要性、持仓时间等."""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml_trading.data_tools.data_loader import MarketDataLoader


def analyze_prediction_distribution(
    model_path: str = "trained_model_wavelet_may_2025.pkl",
):
    """分析预测分布，检查是否有问题."""

    print("=" * 80)
    print("🔍 深度诊断分析")
    print("=" * 80)

    # Load model
    print("\n1. 加载模型...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    strategy = model_data["strategy"]

    # 检查训练数据的target分布
    print("\n2. 检查训练数据target分布...")
    engineered_data = model_data["engineered_data"]

    for timeframe, data in engineered_data.items():
        if timeframe != "5T":  # 只看5T
            continue

        # 准备target
        y_stage1, y_stage2 = strategy.pipeline.prepare_targets(data)

        print(f"\n   Timeframe: {timeframe}")
        print(f"   Total samples: {len(y_stage1)}")
        print(
            f"   Long signals (y=1): {(y_stage1 == 1).sum()} ({(y_stage1 == 1).sum()/len(y_stage1)*100:.2f}%)"
        )
        print(
            f"   Hold signals (y=0): {(y_stage1 == 0).sum()} ({(y_stage1 == 0).sum()/len(y_stage1)*100:.2f}%)"
        )
        print(
            f"   Short signals (y=-1): {(y_stage1 == -1).sum()} ({(y_stage1 == -1).sum()/len(y_stage1)*100:.2f}%)"
        )

        # 检查实际收益率分布
        print(f"\n   实际收益率统计:")
        print(f"   Mean return: {y_stage2.mean():.6f}")
        print(f"   Std return: {y_stage2.std():.6f}")
        print(
            f"   Positive returns: {(y_stage2 > 0).sum()} ({(y_stage2 > 0).sum()/len(y_stage2)*100:.2f}%)"
        )
        print(
            f"   Returns > 0.1%: {(y_stage2 > 0.001).sum()} ({(y_stage2 > 0.001).sum()/len(y_stage2)*100:.2f}%)"
        )
        print(
            f"   Returns < -0.1%: {(y_stage2 < -0.001).sum()} ({(y_stage2 < -0.001).sum()/len(y_stage2)*100:.2f}%)"
        )

        # ⚠️ 关键检查：在long signals中，有多少真的上涨了
        long_mask = y_stage1 == 1
        long_returns = y_stage2[long_mask]
        if len(long_returns) > 0:
            print(f"\n   ⚠️ 关键检查 - Long信号的实际表现:")
            print(f"   Long信号数量: {len(long_returns)}")
            print(
                f"   实际上涨次数: {(long_returns > 0).sum()} ({(long_returns > 0).sum()/len(long_returns)*100:.2f}%)"
            )
            print(f"   平均收益: {long_returns.mean():.6f}")
            print(f"   中位数收益: {long_returns.median():.6f}")

            print(
                f"\n   💡 这说明：如果完美预测所有long信号，基准胜率是 {(long_returns > 0).sum()/len(long_returns)*100:.2f}%"
            )


def analyze_feature_importance(model_path: str = "trained_model_wavelet_may_2025.pkl"):
    """分析特征重要性."""

    print("\n" + "=" * 80)
    print("📊 特征重要性分析")
    print("=" * 80)

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    strategy = model_data["strategy"]
    engineered_data = model_data["engineered_data"]

    # 只分析5T周期
    timeframe = "5T"
    data = engineered_data[timeframe]

    # 获取模型
    stage1_model = strategy.pipeline.stage1_models.get(timeframe)

    if stage1_model and hasattr(stage1_model.model, "feature_importance"):
        print(f"\n{timeframe} Stage 1 (分类) 特征重要性:")

        # 获取特征名
        feature_columns = [
            col
            for col in data.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        # 获取重要性
        importances = stage1_model.model.feature_importance(importance_type="gain")

        # 创建DataFrame
        feature_importance_df = pd.DataFrame(
            {"feature": feature_columns[: len(importances)], "importance": importances}
        ).sort_values("importance", ascending=False)

        print(f"\nTop 20 最重要特征:")
        for idx, row in feature_importance_df.head(20).iterrows():
            print(f"   {row['feature']}: {row['importance']:.2f}")

        # 保存完整列表
        feature_importance_df.to_csv("feature_importance_5T.csv", index=False)
        print(f"\n✅ 完整特征重要性已保存到: feature_importance_5T.csv")

        return feature_importance_df
    else:
        print("   ⚠️ 无法获取特征重要性")
        return None


def calculate_holding_time_stats(
    oos_results_file: str = "oos_test_results_with_timeseries_cv.json",
):
    """计算平均持仓时间."""

    print("\n" + "=" * 80)
    print("⏱️  持仓时间分析")
    print("=" * 80)

    with open(oos_results_file, "r") as f:
        results = json.load(f)

    # 定义时间框架对应的分钟数
    timeframe_minutes = {"5T": 5, "15T": 15, "45T": 45, "60T": 60, "240T": 240}

    print("\n平均持仓时间（假设每个信号持有1个周期）:")
    print("-" * 60)

    for month, month_data in results.items():
        print(f"\n{month}:")
        for tf, metrics in month_data.items():
            minutes = timeframe_minutes.get(tf, 0)
            num_signals = metrics.get("num_signals", 0)

            # 假设每个信号持有1个周期
            avg_holding_minutes = minutes

            print(f"   {tf}: 平均持仓 {avg_holding_minutes} 分钟")
            print(f"        (每周期{minutes}分钟 × 1周期)")
            print(f"        总计 {num_signals} 次交易")


def generate_comprehensive_report():
    """生成综合诊断报告."""

    print("\n" + "=" * 80)
    print("📝 生成综合诊断报告")
    print("=" * 80)

    report = []
    report.append("# 🔍 ML交易系统综合诊断报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("=" * 80 + "\n")

    # 1. 胜率分析
    report.append("\n## 1️⃣ 胜率合理性分析\n")

    with open("trained_model_wavelet_may_2025.pkl", "rb") as f:
        model_data = pickle.load(f)

    strategy = model_data["strategy"]
    engineered_data = model_data["engineered_data"]
    data = engineered_data["5T"]

    y_stage1, y_stage2 = strategy.pipeline.prepare_targets(data)

    long_mask = y_stage1 == 1
    long_returns = y_stage2[long_mask]
    baseline_winrate = (
        (long_returns > 0).sum() / len(long_returns) * 100
        if len(long_returns) > 0
        else 0
    )

    report.append(f"### 基准胜率（完美预测）\n")
    report.append(f"- 训练数据中Long信号总数: {(y_stage1 == 1).sum()}\n")
    report.append(f"- 其中实际上涨: {(long_returns > 0).sum()}\n")
    report.append(f"- **基准胜率: {baseline_winrate:.2f}%**\n")
    report.append(
        f"\n💡 如果模型能完美预测所有应该做多的时机，理论最高胜率就是 {baseline_winrate:.2f}%\n"
    )

    # OOS胜率
    with open("oos_test_results_with_timeseries_cv.json", "r") as f:
        oos_results = json.load(f)

    oos_winrates = []
    for month, data in oos_results.items():
        if "5T" in data:
            oos_winrates.append(data["5T"]["win_rate"] * 100)

    avg_oos_winrate = np.mean(oos_winrates)

    report.append(f"\n### OOS测试胜率\n")
    report.append(f"- 平均OOS胜率: {avg_oos_winrate:.2f}%\n")
    report.append(f"- 基准胜率: {baseline_winrate:.2f}%\n")
    report.append(f"- **差距: {avg_oos_winrate - baseline_winrate:.2f}%**\n")

    if avg_oos_winrate > baseline_winrate + 5:
        report.append(f"\n⚠️ **警告**: OOS胜率明显高于基准胜率，可能存在问题！\n")
    else:
        report.append(f"\n✅ **正常**: OOS胜率在合理范围内\n")

    # 2. 交易频率
    report.append(f"\n## 2️⃣ 交易频率分析\n")

    for month, data in oos_results.items():
        if "5T" in data:
            num_signals = data["5T"]["num_signals"]
            report.append(
                f"- {month}: {num_signals} 次交易 (约每 {30*24*60/num_signals:.1f} 分钟一次)\n"
            )

    # 保存报告
    report_text = "".join(report)

    with open("DIAGNOSTIC_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print("\n✅ 诊断报告已保存到: DIAGNOSTIC_REPORT.md")


def main():
    """主函数."""

    print("\n" + "🔍 开始深度诊断分析..." + "\n")

    # 1. 分析预测分布和基准胜率
    analyze_prediction_distribution()

    # 2. 分析特征重要性
    feature_importance_df = analyze_feature_importance()

    # 3. 计算持仓时间
    calculate_holding_time_stats()

    # 4. 生成综合报告
    generate_comprehensive_report()

    print("\n" + "=" * 80)
    print("✅ 诊断分析完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print("  1. DIAGNOSTIC_REPORT.md - 综合诊断报告")
    print("  2. feature_importance_5T.csv - 特征重要性")
    print("\n请查看这些文件了解详细信息。\n")


if __name__ == "__main__":
    main()
