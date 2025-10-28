#!/usr/bin/env python3
"""
分析增强模型的特征重要性
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def categorize_feature(feature_name):
    """根据特征名称分类"""
    name = feature_name.lower()

    # WPT特征
    if "_wpt_" in name:
        source = name.split("_")[0]
        return f"WPT_{source}"

    # Hurst特征
    if "hurst" in name:
        source = name.split("_")[0]
        return f"Hurst_{source}"

    # Hilbert特征
    if "hilbert" in name:
        source = name.split("_")[0]
        return f"Hilbert_{source}"

    # Spectral特征
    if "spectral" in name:
        source = name.split("_")[0]
        return f"Spectral_{source}"

    # 订单流特征
    if any(
        x in name
        for x in [
            "ofi",
            "order_flow",
            "tbr_",
            "delta_divergence",
            "liquidity",
            "pressure",
            "cvd_",
        ]
    ):
        return "OrderFlow"

    # 高级衍生特征
    if any(
        x in name
        for x in [
            "compression",
            "structure",
            "slope_consistency",
            "momentum_persistence",
            "trend_volatility",
            "bb_width",
            "range_ratio",
            "atr_percentile",
        ]
    ):
        return "Advanced_Derived"

    # 技术指标
    if any(
        x in name
        for x in ["rsi", "macd", "bb_", "atr", "sma", "ema", "roc", "momentum"]
    ):
        return "Technical"

    # 价格/成交量基础特征
    if any(x in name for x in ["returns", "price", "volume", "volatility"]):
        return "Basic_Price_Volume"

    # 时间特征
    if any(x in name for x in ["hour", "day_of"]):
        return "Temporal"

    return "Other"


def analyze_enhanced_model():
    """分析增强模型的特征重要性"""

    print("=" * 80)
    print("🔍 增强模型特征重要性分析")
    print("=" * 80)

    # 加载模型
    print("\n1. 加载增强模型...")
    model_path = Path("trained_model_enhanced_may_2025.pkl")
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    print(f"   ✓ 模型加载成功")

    # 获取5T周期的Stage1模型（信号分类）
    if "strategy" not in model_data:
        print("❌ 模型数据格式不正确")
        return

    strategy = model_data["strategy"]
    if not hasattr(strategy, "pipeline") or not hasattr(
        strategy.pipeline, "stage1_models"
    ):
        print("❌ 无法找到stage1_models")
        return

    if "5T" not in strategy.pipeline.stage1_models:
        print("❌ 没有5T周期的模型")
        print(f"可用周期: {list(strategy.pipeline.stage1_models.keys())}")
        return

    stage1_model = strategy.pipeline.stage1_models["5T"]

    # 获取特征重要性（访问内部的LightGBM Booster对象）
    print("\n2. 提取特征重要性...")
    if not hasattr(stage1_model, "model") or stage1_model.model is None:
        print("❌ 模型未训练")
        return

    lgb_model = stage1_model.model  # 获取内部的lightgbm.Booster对象
    feature_importance = lgb_model.feature_importance(importance_type="gain")
    feature_names = lgb_model.feature_name()

    # 创建DataFrame
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importance}
    ).sort_values("importance", ascending=False)

    # 添加类别
    importance_df["category"] = importance_df["feature"].apply(categorize_feature)

    print(f"   ✓ 共有 {len(feature_names)} 个特征")
    print(f"   ✓ 实际使用 {np.sum(feature_importance > 0)} 个特征")

    # 统计各类别特征数量和重要性
    print("\n3. 按类别统计...")
    category_stats = (
        importance_df.groupby("category")
        .agg({"importance": ["sum", "mean", "count"], "feature": "count"})
        .round(2)
    )

    category_stats.columns = [
        "Total_Importance",
        "Avg_Importance",
        "Count",
        "Feature_Count",
    ]
    category_stats = category_stats.sort_values("Total_Importance", ascending=False)

    print("\n" + "=" * 80)
    print("📊 各类别特征统计")
    print("=" * 80)
    print(category_stats.to_string())

    # Top 50特征
    print("\n" + "=" * 80)
    print("🏆 Top 50 重要特征")
    print("=" * 80)
    top50 = importance_df.head(50)

    for idx, row in top50.iterrows():
        print(
            f"{row.name+1:3d}. [{row['category']:20s}] {row['feature']:40s} {row['importance']:10.1f}"
        )

    # 新增特征的表现
    print("\n" + "=" * 80)
    print("✨ 新增特征表现分析")
    print("=" * 80)

    # WPT特征
    wpt_features = importance_df[importance_df["category"].str.startswith("WPT_")]
    print(f"\n【WPT特征】 共{len(wpt_features)}个")
    print(f"  总重要性: {wpt_features['importance'].sum():.1f}")
    print(f"  平均重要性: {wpt_features['importance'].mean():.1f}")
    print(f"  Top 10:")
    for idx, row in wpt_features.head(10).iterrows():
        print(f"    {row['feature']:50s} {row['importance']:8.1f}")

    # Hurst特征
    hurst_features = importance_df[importance_df["category"].str.startswith("Hurst_")]
    print(f"\n【Hurst特征】 共{len(hurst_features)}个")
    print(f"  总重要性: {hurst_features['importance'].sum():.1f}")
    print(f"  平均重要性: {hurst_features['importance'].mean():.1f}")
    print(f"  Top 10:")
    for idx, row in hurst_features.head(10).iterrows():
        print(f"    {row['feature']:50s} {row['importance']:8.1f}")

    # Hilbert特征
    hilbert_features = importance_df[
        importance_df["category"].str.startswith("Hilbert_")
    ]
    print(f"\n【Hilbert特征】 共{len(hilbert_features)}个")
    print(f"  总重要性: {hilbert_features['importance'].sum():.1f}")
    print(f"  平均重要性: {hilbert_features['importance'].mean():.1f}")
    print(f"  Top 10:")
    for idx, row in hilbert_features.head(10).iterrows():
        print(f"    {row['feature']:50s} {row['importance']:8.1f}")

    # Spectral特征
    spectral_features = importance_df[
        importance_df["category"].str.startswith("Spectral_")
    ]
    print(f"\n【Spectral特征】 共{len(spectral_features)}个")
    print(f"  总重要性: {spectral_features['importance'].sum():.1f}")
    print(f"  平均重要性: {spectral_features['importance'].mean():.1f}")
    print(f"  Top 10:")
    for idx, row in spectral_features.head(10).iterrows():
        print(f"    {row['feature']:50s} {row['importance']:8.1f}")

    # 订单流特征
    orderflow_features = importance_df[importance_df["category"] == "OrderFlow"]
    print(f"\n【订单流特征】 共{len(orderflow_features)}个")
    print(f"  总重要性: {orderflow_features['importance'].sum():.1f}")
    print(f"  平均重要性: {orderflow_features['importance'].mean():.1f}")
    print(f"  Top 10:")
    for idx, row in orderflow_features.head(10).iterrows():
        print(f"    {row['feature']:50s} {row['importance']:8.1f}")

    # 保存详细报告
    print("\n" + "=" * 80)
    print("💾 保存详细报告...")
    print("=" * 80)

    # 保存完整特征重要性
    importance_df.to_csv("enhanced_feature_importance_full.csv", index=False)
    print(f"   ✓ 完整特征重要性: enhanced_feature_importance_full.csv")

    # 保存类别统计
    category_stats.to_csv("enhanced_feature_importance_by_category.csv")
    print(f"   ✓ 类别统计: enhanced_feature_importance_by_category.csv")

    # 生成Markdown报告
    with open("增强模型特征重要性报告.md", "w", encoding="utf-8") as f:
        f.write("# 增强模型特征重要性分析报告\n\n")
        f.write(f"**分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: trained_model_enhanced_may_2025.pkl\n")
        f.write(f"**周期**: 5分钟 (5T)\n")
        f.write(f"**总特征数**: {len(feature_names)}\n\n")

        f.write("---\n\n")
        f.write("## 📊 特征类别统计\n\n")
        f.write("| 类别 | 特征数量 | 总重要性 | 平均重要性 |\n")
        f.write("|------|----------|----------|------------|\n")
        for cat, row in category_stats.iterrows():
            f.write(
                f"| {cat} | {int(row['Count'])} | {row['Total_Importance']:.1f} | {row['Avg_Importance']:.1f} |\n"
            )

        f.write("\n---\n\n")
        f.write("## 🏆 Top 50 重要特征\n\n")
        f.write("| 排名 | 类别 | 特征名 | 重要性 |\n")
        f.write("|------|------|--------|--------|\n")
        for idx, row in top50.iterrows():
            f.write(
                f"| {row.name+1} | {row['category']} | `{row['feature']}` | {row['importance']:.1f} |\n"
            )

        f.write("\n---\n\n")
        f.write("## ✨ 新增特征表现\n\n")

        # WPT
        f.write("### WPT特征\n\n")
        f.write(f"- **数量**: {len(wpt_features)}\n")
        f.write(f"- **总重要性**: {wpt_features['importance'].sum():.1f}\n")
        f.write(f"- **平均重要性**: {wpt_features['importance'].mean():.1f}\n\n")
        f.write("**Top 10**:\n\n")
        for idx, row in wpt_features.head(10).iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.1f}\n")

        # Hurst
        f.write("\n### Hurst特征\n\n")
        f.write(f"- **数量**: {len(hurst_features)}\n")
        f.write(f"- **总重要性**: {hurst_features['importance'].sum():.1f}\n")
        f.write(f"- **平均重要性**: {hurst_features['importance'].mean():.1f}\n\n")
        f.write("**Top 10**:\n\n")
        for idx, row in hurst_features.head(10).iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.1f}\n")

        # Hilbert
        f.write("\n### Hilbert特征\n\n")
        f.write(f"- **数量**: {len(hilbert_features)}\n")
        f.write(f"- **总重要性**: {hilbert_features['importance'].sum():.1f}\n")
        f.write(f"- **平均重要性**: {hilbert_features['importance'].mean():.1f}\n\n")
        f.write("**Top 10**:\n\n")
        for idx, row in hilbert_features.head(10).iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.1f}\n")

        # Spectral
        f.write("\n### Spectral特征\n\n")
        f.write(f"- **数量**: {len(spectral_features)}\n")
        f.write(f"- **总重要性**: {spectral_features['importance'].sum():.1f}\n")
        f.write(f"- **平均重要性**: {spectral_features['importance'].mean():.1f}\n\n")
        f.write("**Top 10**:\n\n")
        for idx, row in spectral_features.head(10).iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.1f}\n")

        # OrderFlow
        f.write("\n### 订单流特征\n\n")
        f.write(f"- **数量**: {len(orderflow_features)}\n")
        f.write(f"- **总重要性**: {orderflow_features['importance'].sum():.1f}\n")
        f.write(f"- **平均重要性**: {orderflow_features['importance'].mean():.1f}\n\n")
        f.write("**Top 10**:\n\n")
        for idx, row in orderflow_features.head(10).iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.1f}\n")

        f.write("\n---\n\n")
        f.write("## 💡 关键发现\n\n")

        # 计算新增特征的总贡献
        new_features = pd.concat(
            [
                wpt_features,
                hurst_features,
                hilbert_features,
                spectral_features,
                orderflow_features,
            ]
        )
        new_importance_total = new_features["importance"].sum()
        total_importance = importance_df["importance"].sum()
        new_importance_pct = (new_importance_total / total_importance) * 100

        f.write(f"1. **新增特征贡献**: {new_importance_pct:.1f}% 的总重要性\n")
        f.write(f"2. **最有价值的新增类别**: {category_stats.head(1).index[0]}\n")
        f.write(f"3. **平均重要性最高**: {category_stats['Avg_Importance'].idxmax()}\n")

        # 推荐保留的特征
        f.write("\n## 🎯 特征精简建议\n\n")
        f.write("基于重要性分析，建议保留以下特征：\n\n")

        # 按累计重要性保留Top 150特征
        importance_df_sorted = importance_df.sort_values("importance", ascending=False)
        cumsum_importance = importance_df_sorted["importance"].cumsum()
        total = importance_df_sorted["importance"].sum()

        # 找到累计90%重要性的特征数量
        idx_90 = (cumsum_importance / total >= 0.9).idxmax()
        n_features_90 = importance_df_sorted.loc[:idx_90].shape[0]

        # 找到累计95%重要性的特征数量
        idx_95 = (cumsum_importance / total >= 0.95).idxmax()
        n_features_95 = importance_df_sorted.loc[:idx_95].shape[0]

        f.write(f"- **保留90%重要性**: 需要Top {n_features_90}个特征\n")
        f.write(f"- **保留95%重要性**: 需要Top {n_features_95}个特征\n")
        f.write(f"- **建议保留**: Top 150-200个特征即可\n\n")

        f.write("---\n\n")
        f.write("*报告生成完成*\n")

    print(f"   ✓ Markdown报告: 增强模型特征重要性报告.md")

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)

    # 关键发现总结
    print("\n💡 关键发现：")
    print(f"   1. 新增特征贡献了 {new_importance_pct:.1f}% 的总重要性")
    print(f"   2. 保留90%重要性只需 {n_features_90} 个特征")
    print(f"   3. 保留95%重要性只需 {n_features_95} 个特征")
    print(f"   4. 建议精简到 150-200 个特征")


if __name__ == "__main__":
    analyze_enhanced_model()
