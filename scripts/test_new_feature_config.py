#!/usr/bin/env python3
"""
测试新的特征配置是否正常工作

验证：
1. 交互特征和衍生特征是否能正确加载
2. 所有特征函数是否正确映射
3. 特征计算是否有错误
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.strategy_config import StrategyConfigLoader


def create_test_data(n_samples=500):
    """创建测试数据"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

    # 生成价格数据
    price_base = 50000
    returns = np.random.randn(n_samples) * 0.01
    prices = price_base * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
            "cvd": np.random.randn(n_samples).cumsum() * 1000,
            "taker_buy_ratio": np.random.uniform(0.3, 0.7, n_samples),
        },
        index=dates,
    )

    # 计算基础指标
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["rsi"] = 50 + np.random.randn(n_samples) * 10  # 简化的 RSI

    return df


def test_feature_loading():
    """测试特征加载"""
    print("=" * 80)
    print("测试新的特征配置")
    print("=" * 80)
    print()

    # 1. 创建测试数据
    print("📊 创建测试数据...")
    df = create_test_data(500)
    print(f"   数据形状: {df.shape}")
    print(f"   数据列: {list(df.columns)}")
    print()

    # 2. 加载策略配置
    print("📋 加载 SR Reversal 策略配置...")
    strategy_dir = PROJECT_ROOT / "config" / "strategies" / "sr_reversal"
    config_loader = StrategyConfigLoader(strategy_dir)
    strategy_config = config_loader.load()
    requested_features = strategy_config.features.requested_features
    print(f"   请求的特征数量: {len(requested_features)}")
    print(f"   请求的特征: {requested_features[:10]}...")
    print()

    # 3. 初始化特征加载器
    print("🔧 初始化特征加载器...")
    feature_loader = StrategyFeatureLoader(
        feature_deps_path=str(PROJECT_ROOT / "config" / "feature_dependencies.yaml"),
        cache_dir=str(PROJECT_ROOT / "cache" / "features"),
        use_disk_cache=False,  # 测试时禁用缓存
        use_memory_cache=True,
        max_workers=2,
    )
    print("   ✅ 特征加载器初始化成功")
    print()

    # 4. 加载特征
    print("🚀 开始加载特征...")
    try:
        df_features = feature_loader.load_features_from_requested(
            df.copy(),
            requested_features,
            fit=True,
        )
        print(f"   ✅ 特征加载成功")
        print(f"   原始列数: {len(df.columns)}")
        print(f"   新增列数: {len(df_features.columns) - len(df.columns)}")
        print(f"   总列数: {len(df_features.columns)}")
        print()
    except Exception as e:
        print(f"   ❌ 特征加载失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 5. 检查新添加的特征
    print("🔍 检查新添加的特征...")
    new_cols = [c for c in df_features.columns if c not in df.columns]

    # 检查交互特征
    interaction_features = [c for c in new_cols if "_x_" in c]
    print(f"   交互特征数量: {len(interaction_features)}")
    if interaction_features:
        print(f"   交互特征示例: {interaction_features[:5]}")

    # 检查衍生特征
    derived_features = [
        "sr_strength_combined",
        "sr_distance_normalized",
        "dist_to_zz_high",
        "dist_to_zz_low",
        "dist_to_zz_high_atr",
        "dist_to_zz_low_atr",
        "cvd_slope_5",
        "atr_ratio",
        "bb_width_ratio",
        "compression_score",
        "tbr_ma_5",
        "tbr_spike",
    ]
    found_derived = [c for c in derived_features if c in df_features.columns]
    print(f"   衍生特征数量: {len(found_derived)} / {len(derived_features)}")
    if found_derived:
        print(f"   衍生特征: {found_derived}")

    # 检查缺失的衍生特征
    missing_derived = [c for c in derived_features if c not in df_features.columns]
    if missing_derived:
        print(f"   ⚠️  缺失的衍生特征: {missing_derived}")
        print(f"      可能原因: 依赖的基础特征不存在（如 zz_high_value, bb_upper 等）")
    print()

    # 6. 检查特征值
    print("📈 检查特征值...")
    sample_features = [
        "vpin",
        "vpin_x_wick_upper",
        "vpin_x_wick_lower",
        "sr_strength_combined",
        "sr_distance_normalized",
        "cvd_slope_5",
        "atr_ratio",
    ]

    for feat in sample_features:
        if feat in df_features.columns:
            values = df_features[feat].dropna()
            if len(values) > 0:
                print(f"   ✅ {feat}:")
                print(f"      有效值数量: {len(values)}")
                print(f"      范围: [{values.min():.4f}, {values.max():.4f}]")
                print(f"      均值: {values.mean():.4f}")
            else:
                print(f"   ⚠️  {feat}: 无有效值")
        else:
            print(f"   ❌ {feat}: 特征不存在")
    print()

    # 7. 检查是否有 NaN 或 Inf
    print("🔍 检查数据质量...")
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    nan_counts = df_features[numeric_cols].isna().sum()
    inf_counts = np.isinf(df_features[numeric_cols]).sum()

    high_nan_cols = nan_counts[nan_counts > len(df_features) * 0.5].index.tolist()
    high_inf_cols = inf_counts[inf_counts > 0].index.tolist()

    if high_nan_cols:
        print(f"   ⚠️  高 NaN 比例的特征 (>50%): {len(high_nan_cols)} 个")
        print(f"      示例: {high_nan_cols[:5]}")
    else:
        print(f"   ✅ 没有高 NaN 比例的特征")

    if high_inf_cols:
        print(f"   ⚠️  包含 Inf 的特征: {len(high_inf_cols)} 个")
        print(f"      示例: {high_inf_cols[:5]}")
    else:
        print(f"   ✅ 没有包含 Inf 的特征")
    print()

    # 8. 总结
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"✅ 特征加载: 成功")
    print(f"✅ 新增特征数: {len(new_cols)}")
    print(f"✅ 交互特征数: {len(interaction_features)}")
    print(f"✅ 衍生特征数: {len(found_derived)}")
    if missing_derived:
        print(f"⚠️  缺失衍生特征: {len(missing_derived)} 个（可能因为依赖特征不存在）")
    print()

    return True


if __name__ == "__main__":
    success = test_feature_loading()
    sys.exit(0 if success else 1)
