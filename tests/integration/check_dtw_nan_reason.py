"""
检查 DTW features 的 NaN 比例是否合理
分析 dist_to_nearest_sr 的分布，确认多少数据点在 SR 阈值范围内
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader


def check_dtw_nan_reason():
    """检查 DTW features NaN 的原因"""

    print("📊 检查 DTW features NaN 的原因...\n")

    # 加载数据
    from src.data_tools.data_handler import MarketDataLoader

    loader = MarketDataLoader()
    loader.symbol = "BTCUSDT"
    loader.timeframe = "1h"
    start_date = "2024-01-01"
    end_date = "2024-07-01"
    df = loader.load_data(start_date=start_date, end_date=end_date)

    print(f"✅ 加载数据: {len(df)} 行")
    print(f"   日期范围: {df.index[0]} 到 {df.index[-1]}\n")

    # 加载特征配置
    config_dir = "config/strategies/sr_reversal_long"
    feature_loader = StrategyFeatureLoader(config_dir)

    # 计算依赖特征（需要 dist_to_nearest_sr）
    print("📊 计算依赖特征...")
    features_config = feature_loader.get_features_config()

    # 需要先计算 sr_strength_max 来获得 dist_to_nearest_sr
    from src.features.loader.parallel_computer import ParallelFeatureComputer

    computer = ParallelFeatureComputer(
        use_monthly_cache=True,
        use_memory_cache=True,
    )

    # 计算必要的依赖特征
    required_deps = ["sr_strength_max"]  # 这会生成 dist_to_nearest_sr

    print("   计算 sr_strength_max (获取 dist_to_nearest_sr)...")
    df_features = computer.compute_features_parallel(
        df,
        features_config,
        required_deps,
        fit=True,
    )

    # 检查 dist_to_nearest_sr
    if "dist_to_nearest_sr" not in df_features.columns:
        print("   ⚠️  dist_to_nearest_sr 不存在，需要先计算 sr_strength_max 的依赖")
        print("   计算 sqs_hal_high 和 sqs_hal_low...")
        df_features = computer.compute_features_parallel(
            df_features,
            features_config,
            ["sqs_hal_high", "sqs_hal_low", "sr_strength_max"],
            fit=True,
        )

    if "dist_to_nearest_sr" not in df_features.columns:
        print("   ❌ 无法获取 dist_to_nearest_sr，跳过分析")
        return

    # 获取 ATR
    if "atr" not in df_features.columns:
        from src.features.time_series.baseline_features import BaselineFeatureEngineer

        df_features["atr"] = BaselineFeatureEngineer.compute_atr(df_features["close"])

    # 计算归一化的 SR 距离
    atr = df_features["atr"].fillna(df_features["atr"].median())
    sr_dist = df_features["dist_to_nearest_sr"].abs()
    sr_dist_normalized = sr_dist / (atr + 1e-8)

    # 分析分布
    sr_threshold = 1.5  # 配置中的阈值
    near_sr_mask = sr_dist_normalized <= sr_threshold

    print(f"\n📈 dist_to_nearest_sr 分析:")
    print(f"   总数据点: {len(df_features)}")
    print(
        f"   距离 SR <= {sr_threshold} ATR: {near_sr_mask.sum()} ({near_sr_mask.sum()/len(df_features)*100:.1f}%)"
    )
    print(
        f"   距离 SR > {sr_threshold} ATR: {(~near_sr_mask).sum()} ({(~near_sr_mask).sum()/len(df_features)*100:.1f}%)"
    )
    print(f"\n   距离分布统计 (单位: ATR):")
    print(f"     - 最小值: {sr_dist_normalized.min():.2f}")
    print(f"     - 25%分位: {sr_dist_normalized.quantile(0.25):.2f}")
    print(f"     - 中位数: {sr_dist_normalized.median():.2f}")
    print(f"     - 75%分位: {sr_dist_normalized.quantile(0.75):.2f}")
    print(f"     - 最大值: {sr_dist_normalized.max():.2f}")

    # 检查不同阈值下的覆盖率
    print(f"\n📊 不同阈值下的覆盖率:")
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        coverage = (sr_dist_normalized <= threshold).sum() / len(df_features) * 100
        print(f"   threshold={threshold:.1f} ATR: {coverage:.1f}%")

    # 计算 DTW features 的 NaN 比例（如果已计算）
    print(f"\n🔍 DTW features 分析:")
    dtw_cols = [col for col in df_features.columns if col.startswith("dtw_")]
    if dtw_cols:
        for col in dtw_cols[:5]:  # 只显示前5个
            nan_pct = df_features[col].isna().sum() / len(df_features) * 100
            print(f"   {col}: {nan_pct:.1f}% NaN")
    else:
        print("   ⚠️  DTW features 未计算（需要运行完整特征计算）")

    print(f"\n💡 结论:")
    expected_nan_pct = (1 - near_sr_mask.sum() / len(df_features)) * 100
    print(f"   预期 NaN 比例: {expected_nan_pct:.1f}%")
    print(f"   实际观察到的: 52.6%")
    print(f"   差异: {abs(expected_nan_pct - 52.6):.1f}%")

    if abs(expected_nan_pct - 52.6) < 5:
        print(f"   ✅ NaN 比例符合预期（基于 SR 距离阈值）")
    else:
        print(f"   ⚠️  NaN 比例与预期不符，可能需要检查")

    if near_sr_mask.sum() / len(df_features) < 0.3:
        print(
            f"   💡 建议: 当前阈值 {sr_threshold} ATR 覆盖率较低，可考虑提高到 2.0 或 2.5 ATR"
        )


if __name__ == "__main__":
    check_dtw_nan_reason()
