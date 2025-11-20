#!/usr/bin/env python3
"""
分析过拟合问题：为什么 IC 高但预测不准、还亏损

检查点：
1. Fold 之间的 IC 方差（高方差 = 过拟合）
2. 最后一个 Fold 的 IC 是否异常高（时间泄漏）
3. 特征重要性分布（是否过度依赖少数特征）
4. 训练集 vs 测试集的 IC 差异
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scipy.stats import spearmanr


def analyze_overfitting(results_file: str, models_file: str = None):
    """分析过拟合问题"""

    print("=" * 80)
    print("🔍 过拟合分析")
    print("=" * 80)
    print()

    # 加载结果
    with open(results_file, "r") as f:
        results = json.load(f)

    print("📊 1. TSCV Fold 分析")
    print("-" * 80)
    cv_results = results.get("cv_results", {})
    fold_results = cv_results.get("fold_results", [])

    if fold_results:
        fold_ics = [f["rank_ic"] for f in fold_results]
        avg_ic = np.mean(fold_ics)
        std_ic = np.std(fold_ics)
        min_ic = np.min(fold_ics)
        max_ic = np.max(fold_ics)

        print(f"   Fold ICs: {[f'{ic:.4f}' for ic in fold_ics]}")
        print(f"   Average: {avg_ic:.4f}")
        print(f"   Std Dev: {std_ic:.4f}")
        print(f"   Range: [{min_ic:.4f}, {max_ic:.4f}]")
        print(f"   CV (std/mean): {abs(std_ic/avg_ic) if avg_ic != 0 else 'inf':.2f}")
        print()

        # 检查最后一个 Fold 是否异常高
        last_fold_ic = fold_ics[-1]
        other_folds_ic = fold_ics[:-1]
        other_avg = np.mean(other_folds_ic)
        other_std = np.std(other_folds_ic)

        if last_fold_ic > other_avg + 2 * other_std:
            print(f"   ⚠️  警告：最后一个 Fold 的 IC ({last_fold_ic:.4f}) 异常高！")
            print(f"      其他 Fold 平均: {other_avg:.4f} ± {other_std:.4f}")
            print(f"      可能是时间泄漏或过拟合到测试集附近的数据")
            print()

        # 检查 IC 方差
        if std_ic > 0.15:
            print(f"   ⚠️  警告：Fold 之间的 IC 方差很大 (std={std_ic:.4f})")
            print(f"      说明模型在不同时间段表现不稳定，可能存在过拟合")
            print()

    print("📊 2. OOS 测试集分析")
    print("-" * 80)
    oos_results = results.get("oos_results", {})
    oos_ic = oos_results.get("rank_ic", None)

    if oos_ic is not None:
        print(f"   OOS Rank IC: {oos_ic:.4f}")
        if avg_ic is not None:
            ic_drop = avg_ic - oos_ic
            print(f"   CV Average IC: {avg_ic:.4f}")
            print(f"   IC 下降: {ic_drop:.4f}")
            if ic_drop < -0.1:
                print(f"   ⚠️  警告：OOS IC 比 CV IC 高很多，可能存在数据泄漏")
            elif ic_drop > 0.1:
                print(f"   ⚠️  警告：OOS IC 比 CV IC 低很多，可能存在过拟合")
            print()

    # 交易表现分析
    eval_data = oos_results.get("evaluation", {})
    conf_stats = eval_data.get("confidence_statistics", {})
    hc_trades = conf_stats.get("high_confidence_trades", {})

    if hc_trades:
        print("📊 3. 交易表现分析")
        print("-" * 80)
        win_rate = hc_trades.get("win_rate", 0)
        total_return = hc_trades.get("total_return", 0)
        sharpe = hc_trades.get("sharpe_ratio", 0)
        max_dd = hc_trades.get("max_drawdown", 0)

        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Sharpe Ratio: {sharpe:.4f}")
        print(f"   Max Drawdown: {max_dd:.2%}")
        print()

        # 分析 IC 高但表现差的原因
        if oos_ic > 0.15 and win_rate < 0.3:
            print("   ⚠️  问题诊断：IC 高但交易表现差")
            print("      可能原因：")
            print("      1. IC 计算的是排序相关性，但实际交易需要方向准确")
            print("      2. 模型可能过度拟合到排序，但预测值本身不准确")
            print("      3. 可能存在数据泄漏，导致 IC 虚高")
            print("      4. 交易成本或滑点未考虑")
            print()

    # 特征重要性分析（如果有模型文件）
    if models_file and os.path.exists(models_file):
        print("📊 4. 特征重要性分析")
        print("-" * 80)
        try:
            with open(models_file, "rb") as f:
                models = pickle.load(f)

            if models and len(models) > 0:
                # 获取第一个模型的特征重要性
                model = models[0]
                if hasattr(model, "feature_importance"):
                    importances = model.feature_importance(importance_type="gain")
                    feature_names = model.feature_name()

                    # 计算特征重要性统计
                    importances_df = pd.DataFrame(
                        {"feature": feature_names, "importance": importances}
                    ).sort_values("importance", ascending=False)

                    top_10_importance = importances_df.head(10)["importance"].sum()
                    total_importance = importances_df["importance"].sum()
                    top_10_ratio = (
                        top_10_importance / total_importance
                        if total_importance > 0
                        else 0
                    )

                    print(f"   总特征数: {len(importances_df)}")
                    print(f"   Top 10 特征重要性占比: {top_10_ratio:.1%}")
                    print()
                    print("   Top 10 重要特征:")
                    for i, row in importances_df.head(10).iterrows():
                        print(f"      {row['feature']}: {row['importance']:.2f}")
                    print()

                    if top_10_ratio > 0.7:
                        print("   ⚠️  警告：模型过度依赖少数特征（Top 10 占比 > 70%）")
                        print(
                            "      这可能导致过拟合，模型可能只记住了这些特征的特定模式"
                        )
                        print()
        except Exception as e:
            print(f"   ⚠️  无法加载模型文件: {e}")
            print()

    # 总结和建议
    print("=" * 80)
    print("📋 总结和建议")
    print("=" * 80)

    issues = []
    if std_ic and std_ic > 0.15:
        issues.append("Fold 之间 IC 方差大（过拟合风险）")
    if last_fold_ic and last_fold_ic > other_avg + 2 * other_std:
        issues.append("最后一个 Fold IC 异常高（时间泄漏风险）")
    if oos_ic and avg_ic and (avg_ic - oos_ic) > 0.1:
        issues.append("OOS IC 明显低于 CV IC（过拟合）")
    if oos_ic and oos_ic > 0.15 and win_rate and win_rate < 0.3:
        issues.append("IC 高但交易表现差（预测值不准确）")

    if issues:
        print("⚠️  发现的问题：")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print()
        print("💡 建议：")
        print("   1. 增加正则化强度（降低 learning_rate，增加 min_data_in_leaf）")
        print("   2. 减少特征数量（移除不重要或高相关的特征）")
        print("   3. 增加 TSCV gap（防止时间泄漏）")
        print("   4. 检查特征是否存在数据泄漏")
        print("   5. 使用更严格的早停策略")
    else:
        print("✅ 未发现明显的过拟合问题")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分析过拟合问题")
    parser.add_argument("--results", type=str, required=True, help="结果 JSON 文件路径")
    parser.add_argument(
        "--models", type=str, default=None, help="模型 PKL 文件路径（可选）"
    )

    args = parser.parse_args()

    analyze_overfitting(args.results, args.models)
