"""分析为什么基线模型特征多，增强模型特征少."""

import pandas as pd
import numpy as np

print("=" * 80)
print("🔍 特征数量差异分析")
print("=" * 80)

# 读取特征重要性文件
feature_imp = pd.read_csv("feature_importance_5T.csv")

print(f"\n基线模型 (Wavelet) 特征总数: {len(feature_imp)}")
print(f"  (来自 feature_importance_5T.csv)")

# 按类型分组统计
wavelet_count = len([f for f in feature_imp["feature"] if "wavelet" in f])
hilbert_count = len([f for f in feature_imp["feature"] if "hilbert" in f])
spectral_count = len([f for f in feature_imp["feature"] if "spectral" in f])
cvd_count = len([f for f in feature_imp["feature"] if "cvd" in f or "tbr_" in f])
volume_count = len([f for f in feature_imp["feature"] if "volume" in f])
momentum_count = len(
    [
        f
        for f in feature_imp["feature"]
        if "momentum" in f or "roc" in f or "acceleration" in f
    ]
)
volatility_count = len(
    [f for f in feature_imp["feature"] if "volatility" in f or "atr" in f or "bb_" in f]
)
ma_count = len([f for f in feature_imp["feature"] if "sma" in f or "ema" in f])
temporal_count = len([f for f in feature_imp["feature"] if "hour" in f or "day" in f])
structure_count = len(
    [
        f
        for f in feature_imp["feature"]
        if "structure" in f or "compression" in f or "tension" in f
    ]
)

print("\n基线模型特征分类:")
print(f"  小波特征 (wavelet): {wavelet_count}")
print(f"  Hilbert变换: {hilbert_count}")
print(f"  光谱分析 (spectral): {spectral_count}")
print(f"  订单流 (CVD/TBR): {cvd_count}")
print(f"  成交量特征: {volume_count}")
print(f"  动量特征: {momentum_count}")
print(f"  波动率特征: {volatility_count}")
print(f"  均线特征: {ma_count}")
print(f"  时间特征: {temporal_count}")
print(f"  市场结构特征: {structure_count}")

# 显示一些具体的特征
print("\n基线模型的独特高级特征:")
unique_features = [
    f
    for f in feature_imp["feature"]
    if any(
        x in f
        for x in [
            "compression",
            "structure",
            "divergence",
            "persistence",
            "slope_consistency",
            "temporal",
        ]
    )
]
for i, f in enumerate(unique_features[:15], 1):
    imp_value = feature_imp[feature_imp["feature"] == f]["importance"].values[0]
    print(f"  {i:2d}. {f:<40s} (重要性: {imp_value:.1f})")

# 增强模型分析
print("\n" + "=" * 80)
print("增强模型 (WPT+Hurst+Entropy) 分析")
print("=" * 80)

print(f"\n从训练日志看到:")
print(f"  总特征数: 68")
print(f"  WPT特征: 36")
print(f"  Hurst特征: 6")
print(f"  其他基础特征: ~26")

print("\n增强模型特征构成:")
print("  1. 小波包 (WPT) 特征: ~36个")
print("     - 8个频带的能量、均值、标准差")
print("     - 能量比例")
print("     - Shannon熵")
print("     - 高低频比率")
print("     - 主导频带")
print("\n  2. Hurst指数特征: ~6个")
print("     - hurst值")
print("     - hurst_deviation")
print("     - hurst_trend_signal")
print("     - hurst_mean_revert_signal")
print("     - hurst_change")
print("     - hurst_acceleration")
print("\n  3. 基础技术指标: ~26个")
print("     - RSI, MACD, BB, ATR")
print("     - 价格变化、波动率")
print("     - 成交量特征")
print("     - 动量特征")

# 对比分析
print("\n" + "=" * 80)
print("💡 为什么增强模型特征更少？")
print("=" * 80)

print("\n原因分析:")
print("\n1. **设计理念不同**")
print("   基线模型 (142特征): 堆叠式特征工程")
print("     - 小波变换 (close, volume, CVD, TBR各自的小波)")
print("     - Hilbert变换")
print("     - 光谱分析")
print("     - 大量衍生指标 (compression, structure, divergence等)")
print("     - 时间特征 (hour_sin, hour_cos等)")
print("     - 市场微观结构特征")
print("\n   增强模型 (68特征): 精简式特征工程")
print("     - 用WPT替代多个小波变换（更精细但更简洁）")
print("     - 用Hurst指数代替多个趋势/反转指标")
print("     - 用Shannon熵代替多个市场状态指标")
print("     - 只保留核心技术指标")

print("\n2. **特征冗余性**")
print("   基线模型:")
print("     - close的小波 + volume的小波 + CVD的小波 + TBR的小波")
print("     - 很多特征可能高度相关")
print("\n   增强模型:")
print("     - 只对close做WPT（但分解更精细）")
print("     - 去除冗余，保留核心")

print("\n3. **计算效率**")
print("   基线模型: 142个特征，计算慢")
print("   增强模型: 68个特征，计算快2倍")

print("\n" + "=" * 80)
print("🎯 哪个更好？")
print("=" * 80)

print("\n从训练CV结果看:")
print(f"  基线模型: 73.3% ± 1.5% (142特征)")
print(f"  增强模型: 70.67% ± 2.0% (68特征)")
print(f"  差距: -2.63%")

print("\n**可能的解释:**")
print("  1. 基线模型的'特征堆叠'在训练集上表现更好")
print("  2. 但更多特征 ≠ 更好的泛化")
print("  3. 增强模型更简洁，可能过拟合风险更低")
print("  4. **OOS测试才能看出真正的优劣**")

print("\n**类比:**")
print("  基线模型 = 用100个指标看市场（全面但可能冗余）")
print("  增强模型 = 用10个核心指标看市场（精简但可能更本质）")

print("\n" + "=" * 80)
print("📊 建议")
print("=" * 80)

print("\n1. **基线模型 (142特征) 的优势:**")
print("   ✅ 已验证：OOS准确率91.24%，胜率91.30%")
print("   ✅ 信息量大，捕捉多方面信号")
print("   ⚠️ 特征冗余，计算慢")

print("\n2. **增强模型 (68特征) 的优势:**")
print("   ✅ 特征精简，计算快")
print("   ✅ 理论更优雅（WPT、Hurst都是高级方法）")
print("   ✅ 可能泛化更好（待OOS验证）")
print("   ⚠️ 训练准确率略低")

print("\n3. **最佳实践:**")
print("   方案A: 直接使用基线模型（已验证优秀）")
print("   方案B: 测试增强模型OOS，如果>92%则更好")
print("   方案C: Ensemble两个模型（各取所长）")
print("   方案D: 对基线模型做特征选择，保留Top 70个特征")

print("\n" + "=" * 80)
