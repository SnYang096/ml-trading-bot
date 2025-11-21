#!/usr/bin/env python3
"""
验证特征与未来收益的相关性：区分"真实 Alpha"和"数据泄漏"

这个脚本执行两个关键实验：
1. 滞后测试：检查特征滞后 1-2 根后相关性是否显著下降
2. 随机打乱测试：检查特征是否只对真实行情有相关性，对随机数据无相关性

如果特征已经正确 shift(1)，那么：
- 滞后测试：相关性应该缓慢衰减（真实信号）
- 随机打乱测试：相关性应该降至 ~0（证明不是虚假相关）
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr
from data_tools.data_utils import load_data
from time_series_model.pipeline.training.train_rank_ic_standalone import (
    split_train_test,
)
from time_series_model.pipeline.training.rank_ic_trainer import prepare_rank_ic_labels


def test_lag_correlation(labels: pd.DataFrame, features: list[str], max_lag: int = 3):
    """
    实验 1：滞后测试

    检查特征滞后 1-N 根后与 future_return 的相关性变化。
    如果相关性缓慢衰减，说明是真实信号；如果急剧下降，可能是对齐问题。
    """
    print("\n" + "=" * 80)
    print("🔬 实验 1：滞后测试（Lag Correlation Test）")
    print("=" * 80)
    print("目的：检查特征滞后后相关性是否缓慢衰减（真实信号）还是急剧下降（对齐问题）")
    print()

    results = []
    for feat in features:
        if feat not in labels.columns:
            continue

        base = labels[[feat, "future_return"]].dropna()
        if len(base) < 100:
            continue

        # 计算不同滞后的相关性
        corr_data = []
        for lag in range(max_lag + 1):
            if lag == 0:
                feat_series = base[feat]
            else:
                feat_series = base[feat].shift(lag)
                base_lag = base[[feat, "future_return"]].copy()
                base_lag[feat] = feat_series
                base_lag = base_lag.dropna()
                if len(base_lag) < 100:
                    break
                feat_series = base_lag[feat]

            mask = feat_series.notna() & base["future_return"].notna()
            if mask.sum() < 100:
                break

            corr, p_value = spearmanr(feat_series[mask], base["future_return"][mask])
            corr_data.append({"lag": lag, "correlation": corr, "p_value": p_value})

        if len(corr_data) >= 2:
            results.append({"feature": feat, "correlations": corr_data})

    # 打印结果
    print(f"{'特征':<30} {'Lag 0':>10} {'Lag 1':>10} {'Lag 2':>10} {'衰减模式':>15}")
    print("-" * 80)
    for result in results:
        feat = result["feature"]
        corrs = result["correlations"]
        lag0 = corrs[0]["correlation"] if len(corrs) > 0 else np.nan
        lag1 = corrs[1]["correlation"] if len(corrs) > 1 else np.nan
        lag2 = corrs[2]["correlation"] if len(corrs) > 2 else np.nan

        # 判断衰减模式
        if len(corrs) >= 2:
            decay = abs(lag0) - abs(lag1) if not np.isnan(lag1) else 0
            if abs(decay) < 0.01:
                pattern = "缓慢衰减 ✅"
            elif abs(decay) > 0.05:
                pattern = "急剧下降 ⚠️"
            else:
                pattern = "中等衰减"
        else:
            pattern = "N/A"

        print(f"{feat:<30} {lag0:>10.4f} {lag1:>10.4f} {lag2:>10.4f} {pattern:>15}")

    print("\n📊 解读：")
    print("  - 缓慢衰减 ✅：说明是真实信号，特征对未来收益有持续预测能力")
    print("  - 急剧下降 ⚠️：可能是时间对齐问题，需要检查 shift(1) 是否正确应用")
    print()

    return results


def test_shuffled_correlation(
    labels: pd.DataFrame, features: list[str], seed: int = 42, n_shuffle: int = 10
):
    """
    实验 2：随机打乱测试

    将 future_return 随机打乱，检查特征是否仍与之相关。
    如果相关性降至 ~0，说明特征只对真实行情有相关性，不是虚假相关。

    通过多次打乱（n_shuffle 次）并计算均值和标准差，提高结论的可靠性。
    单次打乱可能偶然出现较高相关性，多次实验可避免误判。

    Args:
        labels: 包含特征和标签的 DataFrame
        features: 要测试的特征列表
        seed: 随机种子
        n_shuffle: 打乱次数，默认 10 次
    """
    print("\n" + "=" * 80)
    print("🔬 实验 2：随机打乱测试（Shuffled Correlation Test）")
    print("=" * 80)
    print(
        "目的：检查特征是否只对真实行情有相关性，对随机数据无相关性（证明不是虚假相关）"
    )
    print(f"方法：进行 {n_shuffle} 次随机打乱，计算相关性的均值和标准差以提高稳健性")
    print()

    results = []
    for feat in features:
        if feat not in labels.columns:
            continue

        # 真实相关性
        base = labels[[feat, "future_return"]].dropna()
        if len(base) < 100:
            continue

        corr_real, p_real = spearmanr(base[feat], base["future_return"])

        # 多次打乱，计算相关性的均值和标准差
        shuffled_corrs = []
        for i in range(n_shuffle):
            labels_shuffled = labels.copy().reset_index(drop=True)
            shuffled_returns = (
                labels["future_return"]
                .sample(frac=1, random_state=seed + i)
                .reset_index(drop=True)
            )
            labels_shuffled["future_return_shuffled"] = shuffled_returns

            base_shuffled = labels_shuffled[[feat, "future_return_shuffled"]].dropna()
            if len(base_shuffled) < 100:
                continue

            corr_shuffled, _ = spearmanr(
                base_shuffled[feat], base_shuffled["future_return_shuffled"]
            )
            shuffled_corrs.append(corr_shuffled)

        if len(shuffled_corrs) == 0:
            continue

        mean_corr_shuffled = np.mean(shuffled_corrs)
        std_corr_shuffled = np.std(shuffled_corrs)

        # 计算衰减比（使用均值）
        ratio = abs(mean_corr_shuffled) / abs(corr_real) if corr_real != 0 else np.nan

        results.append(
            {
                "feature": feat,
                "corr_real": corr_real,
                "p_real": p_real,
                "corr_shuffled_mean": mean_corr_shuffled,
                "corr_shuffled_std": std_corr_shuffled,
                "corr_shuffled_values": shuffled_corrs,
                "ratio": ratio,
            }
        )

    # 打印结果
    print(
        f"{'特征':<30} {'真实相关性':>12} {'打乱后均值':>12} {'打乱后标准差':>12} {'衰减比':>10} {'结论':>15}"
    )
    print("-" * 80)
    for r in results:
        ratio = r["ratio"]
        corr_real = r["corr_real"]
        p_real = r["p_real"]

        # 改进的判断逻辑：
        # 1. 如果真实相关性很小（< 0.01）或 p 值不显著（> 0.05），即使 ratio > 0.5 也不标记为可疑
        #    因为这些特征本身就没有预测能力，打乱后相关性可能是噪声
        # 2. 只有当真实相关性显著（p < 0.05）且 ratio > 0.5 时，才标记为可疑
        if abs(corr_real) < 0.01 or p_real > 0.05:
            if ratio < 0.1:
                conclusion = "真实信号 ✅"
            else:
                conclusion = "相关性弱（非可疑）"
        elif ratio < 0.1:
            conclusion = "真实信号 ✅"
        elif ratio > 0.5:
            conclusion = "可疑 ⚠️"
        else:
            conclusion = "需进一步检查"

        print(
            f"{r['feature']:<30} {r['corr_real']:>12.4f} "
            f"{r['corr_shuffled_mean']:>12.4f} {r['corr_shuffled_std']:>12.4f} "
            f"{ratio:>10.2%} {conclusion:>15}"
        )

    print("\n📊 解读：")
    print("  - 真实信号 ✅：打乱后相关性均值降至 < 10%，说明特征只对真实行情有相关性")
    print("  - 可疑 ⚠️：打乱后相关性均值仍 > 50%，可能是虚假相关或数据泄漏")
    print("  - 标准差：如果标准差较大，说明单次打乱的结果波动较大，需要更多次实验")
    print()

    return results


def test_walk_forward_simulation(
    labels: pd.DataFrame,
    features: list[str],
    n_windows: int = 5,
    min_train_size: int = 500,
    test_window_size: int = 200,
    date_col: str = None,
):
    """
    实验 3：前向填充模拟（Walk-Forward Simulation）

    模拟滚动窗口训练，确保每次训练只用到截止某时间点的数据。
    如果模型在 walk-forward 测试中表现远差于一次性训练，可能暗示泄漏。

    Args:
        labels: 包含特征和标签的 DataFrame
        features: 要测试的特征列表
        n_windows: 测试窗口数量
        min_train_size: 最小训练集大小
        test_window_size: 每个测试窗口的大小
        date_col: 日期列名（如果提供，用于按时间排序）

    Returns:
        包含每个窗口的 walk-forward 和一次性训练结果的列表
    """
    print("\n" + "=" * 80)
    print("🔬 实验 3：前向填充模拟（Walk-Forward Simulation）")
    print("=" * 80)
    print("目的：检查模型在滚动窗口训练中的表现是否与一次性训练一致")
    print("原理：如果存在数据泄漏，walk-forward 表现会远差于一次性训练")
    print()

    # 确保数据按时间排序
    if date_col and date_col in labels.columns:
        labels = labels.sort_values(date_col).reset_index(drop=True)
    elif labels.index.name == "datetime" or isinstance(labels.index, pd.DatetimeIndex):
        labels = labels.sort_index().reset_index(drop=True)
    else:
        # 如果没有明确的日期列，假设数据已经是按时间顺序的
        labels = labels.reset_index(drop=True)

    # 过滤有效的特征
    valid_features = [f for f in features if f in labels.columns]
    if len(valid_features) == 0:
        print("⚠️  没有找到有效的特征列")
        return []

    # 准备数据：确保有必要的列
    required_cols = valid_features + ["future_return", "volatility_normalized_target"]
    missing_cols = [col for col in required_cols if col not in labels.columns]
    if missing_cols:
        print(f"⚠️  缺少必要的列: {missing_cols}")
        return []

    # 计算窗口大小
    total_size = len(labels)
    if total_size < min_train_size + test_window_size:
        print(f"⚠️  数据量不足: {total_size} < {min_train_size + test_window_size}")
        return []

    # 计算每个测试窗口的起始位置
    available_size = total_size - min_train_size
    window_step = max(1, available_size // n_windows)

    results = []

    print(f"📊 数据统计:")
    print(f"   - 总样本数: {total_size}")
    print(f"   - 最小训练集大小: {min_train_size}")
    print(f"   - 测试窗口大小: {test_window_size}")
    print(f"   - 测试窗口数量: {n_windows}")
    print(f"   - 特征数量: {len(valid_features)}")
    print()

    for window_idx in range(n_windows):
        # 计算当前窗口的边界
        train_end = min_train_size + window_idx * window_step
        test_start = train_end
        test_end = min(test_start + test_window_size, total_size)

        if test_end - test_start < 50:  # 测试集太小，跳过
            continue

        train_data = labels.iloc[:train_end].copy()
        test_data = labels.iloc[test_start:test_end].copy()

        # 移除 NaN
        train_clean = train_data[required_cols].dropna()
        test_clean = test_data[required_cols].dropna()

        if len(train_clean) < min_train_size * 0.8 or len(test_clean) < 50:
            continue

        print(f"窗口 {window_idx + 1}/{n_windows}:")
        print(f"   - 训练集: {len(train_clean)} 个样本 (0:{train_end})")
        print(f"   - 测试集: {len(test_clean)} 个样本 ({test_start}:{test_end})")

        # 方法 1: Walk-Forward（只使用到当前时间点的数据）
        # 使用简单的线性回归或相关性作为代理指标
        # 这里我们计算特征与目标的相关性作为预测能力的代理

        # 计算训练集上的特征-目标相关性（作为"模型"）
        train_corrs = {}
        for feat in valid_features:
            mask = (
                train_clean[feat].notna()
                & train_clean["volatility_normalized_target"].notna()
            )
            if mask.sum() > 50:
                corr, _ = spearmanr(
                    train_clean.loc[mask, feat],
                    train_clean.loc[mask, "volatility_normalized_target"],
                )
                train_corrs[feat] = corr

        # 在测试集上评估：使用训练集的相关性作为预测信号
        # 预测 = 特征的加权和（权重 = 训练集相关性）
        if len(train_corrs) > 0:
            # 构建预测：使用特征的相关性作为权重
            test_predictions_wf = pd.Series(0.0, index=test_clean.index)
            for feat, corr in train_corrs.items():
                mask = test_clean[feat].notna()
                if mask.sum() > 0:
                    # 标准化特征后加权
                    feat_vals = test_clean.loc[mask, feat]
                    if feat_vals.std() > 0:
                        feat_normalized = (
                            feat_vals - feat_vals.mean()
                        ) / feat_vals.std()
                        test_predictions_wf.loc[mask] += corr * feat_normalized

            # 计算 walk-forward 的 Rank IC
            mask = test_predictions_wf.notna() & test_clean["future_return"].notna()
            if mask.sum() > 50:
                rank_ic_wf, _ = spearmanr(
                    test_predictions_wf[mask], test_clean.loc[mask, "future_return"]
                )
            else:
                rank_ic_wf = np.nan
        else:
            rank_ic_wf = np.nan

        # 方法 2: 一次性训练（使用全部数据，包括未来数据 - 这是泄漏的情况）
        # 使用全部数据计算相关性
        all_data = labels[required_cols].dropna()
        all_corrs = {}
        for feat in valid_features:
            mask = (
                all_data[feat].notna()
                & all_data["volatility_normalized_target"].notna()
            )
            if mask.sum() > 50:
                corr, _ = spearmanr(
                    all_data.loc[mask, feat],
                    all_data.loc[mask, "volatility_normalized_target"],
                )
                all_corrs[feat] = corr

        # 在测试集上评估（使用全部数据的相关性）
        if len(all_corrs) > 0:
            test_predictions_all = pd.Series(0.0, index=test_clean.index)
            for feat, corr in all_corrs.items():
                mask = test_clean[feat].notna()
                if mask.sum() > 0:
                    feat_vals = test_clean.loc[mask, feat]
                    if feat_vals.std() > 0:
                        feat_normalized = (
                            feat_vals - feat_vals.mean()
                        ) / feat_vals.std()
                        test_predictions_all.loc[mask] += corr * feat_normalized

            # 计算一次性训练的 Rank IC
            mask = test_predictions_all.notna() & test_clean["future_return"].notna()
            if mask.sum() > 50:
                rank_ic_all, _ = spearmanr(
                    test_predictions_all[mask], test_clean.loc[mask, "future_return"]
                )
            else:
                rank_ic_all = np.nan
        else:
            rank_ic_all = np.nan

        # 计算性能差异
        if not np.isnan(rank_ic_wf) and not np.isnan(rank_ic_all):
            performance_gap = abs(rank_ic_all) - abs(rank_ic_wf)
            gap_ratio = (
                abs(rank_ic_all) / abs(rank_ic_wf)
                if abs(rank_ic_wf) > 0.001
                else np.nan
            )
        else:
            performance_gap = np.nan
            gap_ratio = np.nan

        results.append(
            {
                "window_idx": window_idx,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_size": len(train_clean),
                "test_size": len(test_clean),
                "rank_ic_walk_forward": rank_ic_wf,
                "rank_ic_all_data": rank_ic_all,
                "performance_gap": performance_gap,
                "gap_ratio": gap_ratio,
            }
        )

        print(f"   - Walk-Forward Rank IC: {rank_ic_wf:.4f}")
        print(f"   - 一次性训练 Rank IC: {rank_ic_all:.4f}")
        if not np.isnan(performance_gap):
            print(f"   - 性能差异: {performance_gap:.4f} (比率: {gap_ratio:.2f}x)")
            if gap_ratio > 2.0:
                print(
                    f"   ⚠️  警告: 一次性训练表现远好于 walk-forward，可能存在数据泄漏！"
                )
        print()

    # 汇总结果
    if results:
        print("=" * 80)
        print("📊 汇总结果")
        print("=" * 80)

        results_df = pd.DataFrame(results)
        avg_gap = results_df["performance_gap"].mean()
        avg_ratio = results_df["gap_ratio"].mean()
        avg_wf_ic = results_df["rank_ic_walk_forward"].mean()
        avg_all_ic = results_df["rank_ic_all_data"].mean()

        print(f"{'指标':<30} {'Walk-Forward':>15} {'一次性训练':>15} {'差异':>15}")
        print("-" * 80)
        print(
            f"{'平均 Rank IC':<30} {avg_wf_ic:>15.4f} {avg_all_ic:>15.4f} {avg_gap:>15.4f}"
        )
        print(f"{'平均性能比率':<30} {'':>15} {'':>15} {avg_ratio:>15.2f}x")
        print()

        print("📊 解读：")
        if avg_ratio > 2.0:
            print("  ⚠️  警告: 一次性训练表现显著好于 walk-forward (比率 > 2x)")
            print("      → 可能存在数据泄漏，特征可能使用了未来信息")
        elif avg_ratio > 1.5:
            print("  ⚠️  注意: 一次性训练表现略好于 walk-forward (比率 > 1.5x)")
            print("      → 建议进一步检查特征计算逻辑")
        else:
            print("  ✅ 正常: Walk-forward 和一次性训练表现接近")
            print("      → 特征计算逻辑可能是安全的")
        print()

    return results


def main():
    """主函数：运行所有验证实验"""
    import argparse

    parser = argparse.ArgumentParser(
        description="验证特征与未来收益的相关性：区分真实 Alpha 和数据泄漏"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/parquet_data",
        help="数据路径",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="交易符号",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="开始日期",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-01-01",
        help="结束日期",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="240T",
        help="时间周期",
    )
    parser.add_argument(
        "--top-factors",
        type=str,
        default="custom/leakage_shift_features.json",
        help="特征列表文件（JSON 或 YAML）",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="预测周期",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="测试集比例（用于分割数据）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/feature_correlation_validation",
        help="结果输出目录",
    )
    parser.add_argument(
        "--n-shuffle",
        type=int,
        default=10,
        help="随机打乱测试的次数",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=5,
        help="Walk-forward 测试的窗口数量",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=500,
        help="Walk-forward 测试的最小训练集大小",
    )
    parser.add_argument(
        "--test-window-size",
        type=int,
        default=200,
        help="Walk-forward 测试的每个测试窗口大小",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🔍 特征相关性验证：区分真实 Alpha 和数据泄漏")
    print("=" * 80)
    print(f"数据路径: {args.data_path}")
    print(f"交易符号: {args.symbol}")
    print(f"时间范围: {args.start_date} ~ {args.end_date}")
    print(f"时间周期: {args.timeframe}")
    print(f"特征列表: {args.top_factors}")
    print(f"预测周期: {args.horizon}")
    print()

    # 加载数据和特征
    print("📊 加载数据和特征...")
    df, feature_cols, engineer = load_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        feature_type="comprehensive",
        top_factors=args.top_factors,
        engineer=None,
        fit=True,
    )

    # 准备标签
    print("📝 准备标签...")
    labels = prepare_rank_ic_labels(
        df,
        price_col="close",
        hold_period=args.horizon,
        lookback_window=60,
        ensure_volatility=True,
    )
    labels = labels.dropna(subset=["volatility_normalized_target"])

    # 分割数据（只使用训练集进行验证）
    train_labels, _ = split_train_test(labels, test_size=args.test_size)
    print(f"✅ 使用训练集进行验证: {len(train_labels)} 个样本")
    print()

    # 选择要验证的特征（排除基础列）
    features_to_test = [
        f for f in feature_cols if f not in ["close", "_symbol", "future_return"]
    ]
    print(f"📋 将验证 {len(features_to_test)} 个特征")
    print()

    # 实验 1：滞后测试
    lag_results = test_lag_correlation(train_labels, features_to_test, max_lag=2)

    # 实验 2：随机打乱测试
    shuffle_results = test_shuffled_correlation(
        train_labels, features_to_test, seed=42, n_shuffle=args.n_shuffle
    )

    # 实验 3：前向填充模拟测试
    walk_forward_results = test_walk_forward_simulation(
        train_labels,
        features_to_test,
        n_windows=args.n_windows,
        min_train_size=args.min_train_size,
        test_window_size=args.test_window_size,
        date_col=None,  # 如果数据有日期列，可以指定
    )

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("💾 保存结果")
    print("=" * 80)

    # 保存滞后测试结果
    # 将嵌套结构转换为扁平化的 DataFrame
    lag_data = []
    for result in lag_results:
        feature = result["feature"]
        for corr_info in result["correlations"]:
            lag_data.append(
                {
                    "feature": feature,
                    "lag": corr_info["lag"],
                    "correlation": corr_info["correlation"],
                    "p_value": corr_info["p_value"],
                }
            )

    if lag_data:
        lag_df = pd.DataFrame(lag_data)
        lag_csv_path = output_dir / "lag_test_results.csv"
        lag_json_path = output_dir / "lag_test_results.json"
        lag_df.to_csv(lag_csv_path, index=False)
        lag_df.to_json(lag_json_path, orient="records", indent=2)
        print(f"✅ 滞后测试结果已保存:")
        print(f"   - CSV: {lag_csv_path}")
        print(f"   - JSON: {lag_json_path}")

    # 保存打乱测试结果
    # 将 shuffle_results 转换为 DataFrame（排除列表字段，单独保存）
    shuffle_data = []
    shuffle_detailed_data = []
    for result in shuffle_results:
        feature = result["feature"]
        # 主结果（不包含详细列表）
        shuffle_data.append(
            {
                "feature": feature,
                "corr_real": result["corr_real"],
                "p_real": result["p_real"],
                "corr_shuffled_mean": result["corr_shuffled_mean"],
                "corr_shuffled_std": result["corr_shuffled_std"],
                "ratio": result["ratio"],
            }
        )
        # 详细结果（每个打乱次数的结果）
        for idx, corr_val in enumerate(result["corr_shuffled_values"]):
            shuffle_detailed_data.append(
                {
                    "feature": feature,
                    "shuffle_idx": idx,
                    "correlation": corr_val,
                }
            )

    if shuffle_data:
        shuffle_df = pd.DataFrame(shuffle_data)
        shuffle_csv_path = output_dir / "shuffle_test_results.csv"
        shuffle_json_path = output_dir / "shuffle_test_results.json"
        shuffle_df.to_csv(shuffle_csv_path, index=False)
        shuffle_df.to_json(shuffle_json_path, orient="records", indent=2)
        print(f"✅ 打乱测试结果已保存:")
        print(f"   - CSV: {shuffle_csv_path}")
        print(f"   - JSON: {shuffle_json_path}")

        # 保存详细的打乱结果（每个打乱次数的相关性）
        if shuffle_detailed_data:
            shuffle_detailed_df = pd.DataFrame(shuffle_detailed_data)
            shuffle_detailed_csv_path = output_dir / "shuffle_test_detailed_results.csv"
            shuffle_detailed_df.to_csv(shuffle_detailed_csv_path, index=False)
            print(f"   - 详细结果 CSV: {shuffle_detailed_csv_path}")

    # 保存 Walk-Forward 测试结果
    if walk_forward_results:
        walk_forward_df = pd.DataFrame(walk_forward_results)
        walk_forward_csv_path = output_dir / "walk_forward_test_results.csv"
        walk_forward_json_path = output_dir / "walk_forward_test_results.json"
        walk_forward_df.to_csv(walk_forward_csv_path, index=False)
        walk_forward_df.to_json(walk_forward_json_path, orient="records", indent=2)
        print(f"✅ Walk-Forward 测试结果已保存:")
        print(f"   - CSV: {walk_forward_csv_path}")
        print(f"   - JSON: {walk_forward_json_path}")

    print()

    # 总结
    print("\n" + "=" * 80)
    print("📋 总结")
    print("=" * 80)
    print("✅ 如果所有实验都通过：")
    print("   - 滞后测试：相关性缓慢衰减 → 真实信号")
    print("   - 随机打乱：相关性降至 ~0 → 不是虚假相关")
    print("   - Walk-Forward：与一次性训练表现接近 → 无数据泄漏")
    print("   → 结论：特征包含真实 Alpha，不是数据泄漏")
    print()
    print("⚠️  如果实验未通过：")
    print("   - 滞后测试：相关性急剧下降 → 检查 shift(1) 是否正确应用")
    print("   - 随机打乱：相关性仍高 → 可能存在数据泄漏或虚假相关")
    print("   - Walk-Forward：表现远差于一次性训练 → 可能存在数据泄漏")
    print("   → 建议：进一步检查特征计算逻辑")
    print()


if __name__ == "__main__":
    main()
