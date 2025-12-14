"""
检查常数特征的依赖列值

分析为什么这些特征会是常数：
1. evt_x_trend_r2
2. vpin_signed_imbalance_x_trade_cluster_imbalance
3. vpin_x_trade_cluster_entropy
4. cvd_slope_5
5. vpin_x_wick_upper
6. vpin_x_wick_lower
7. atr_ratio
8. tbr_ma_5
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.time_series_model.strategy_config.loader import StrategyConfigLoader
from scripts.train_strategy_pipeline import (
    run_feature_pipeline,
    determine_feature_columns,
)
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.data_tools.data_loader import MarketDataLoader


def check_constant_feature_dependencies(
    strategy_config_path: str = "config/strategies/sr_reversal_long",
    symbol: str = "BTCUSDT",
    data_path: str = "data/parquet_data",
    sample_size: int = 2000,
):
    """检查常数特征的依赖列值"""
    print("=" * 80)
    print("检查常数特征的依赖列值")
    print("=" * 80)
    print()

    # 1. 加载配置和数据
    print("1. 加载配置和数据...")
    config_loader = StrategyConfigLoader(strategy_config_path)
    strategy_config = config_loader.load()

    data_loader = MarketDataLoader(data_path=data_path)
    df_raw = data_loader.load_data(symbol=symbol, timeframe="60T")
    print(f"   原始数据: {df_raw.shape}")
    print()

    # 2. 计算特征
    print("2. 计算特征...")
    feature_loader = StrategyFeatureLoader()

    # 配置 tick loader
    try:
        from scripts.train_strategy_pipeline import _ensure_ticks_configured

        start_ts = str(df_raw.index.min())
        end_ts = str(df_raw.index.max())
        requested_features = strategy_config.features.requested_features or []
        _ensure_ticks_configured(
            feature_loader, symbol, data_path, start_ts, end_ts, requested_features
        )
    except Exception as e:
        print(f"   ⚠️  Tick 配置失败: {e}")

    df_features = run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_config.features,
        fit=True,
    )

    # 使用样本
    df_sample = df_features.head(sample_size).copy()
    print(f"   样本数据: {df_sample.shape}")
    print()

    # 3. 检查每个常数特征的依赖列
    print("3. 检查常数特征的依赖列...")
    print()

    # 定义要检查的特征及其依赖列
    checks = [
        {
            "feature": "evt_x_trend_r2",
            "dependencies": ["evt_tail_shape", "trend_r2_20"],
            "formula": "evt_tail_shape * trend_r2_20",
        },
        {
            "feature": "vpin_signed_imbalance_x_trade_cluster_imbalance",
            "dependencies": ["vpin_signed_imbalance", "trade_cluster_imbalance_ratio"],
            "formula": "vpin_signed_imbalance * trade_cluster_imbalance_ratio",
        },
        {
            "feature": "vpin_x_trade_cluster_entropy",
            "dependencies": ["vpin", "trade_cluster_directional_entropy"],
            "formula": "vpin * trade_cluster_directional_entropy",
        },
        {
            "feature": "cvd_slope_5",
            "dependencies": ["cvd"],
            "formula": "rolling_slope(cvd, window=5)",
        },
        {
            "feature": "vpin_x_wick_upper",
            "dependencies": ["vpin", "wick_upper_ratio"],
            "formula": "vpin * wick_upper_ratio",
        },
        {
            "feature": "vpin_x_wick_lower",
            "dependencies": ["vpin", "wick_lower_ratio"],
            "formula": "vpin * wick_lower_ratio",
        },
        {
            "feature": "atr_ratio",
            "dependencies": ["atr", "close"],
            "formula": "atr / close",
        },
        {
            "feature": "tbr_ma_5",
            "dependencies": ["taker_buy_ratio"],
            "formula": "rolling_mean(taker_buy_ratio, window=5)",
        },
    ]

    for check in checks:
        feature_name = check["feature"]
        dependencies = check["dependencies"]
        formula = check["formula"]

        print(f"   {feature_name}:")
        print(f"     公式: {formula}")

        # 检查特征本身
        if feature_name in df_sample.columns:
            feat_values = df_sample[feature_name]
            unique_vals = feat_values[~feat_values.isna()].unique()
            var_val = feat_values.var()
            print(
                f"     特征值: 范围 [{feat_values.min():.6f}, {feat_values.max():.6f}], "
                f"唯一值数 = {len(unique_vals)}, 方差 = {var_val:.2e}"
            )
        else:
            print(f"     ⚠️  特征不存在于数据中")

        # 检查依赖列
        print(f"     依赖列:")
        for dep_col in dependencies:
            if dep_col in df_sample.columns:
                dep_values = df_sample[dep_col]
                unique_vals = dep_values[~dep_values.isna()].unique()
                var_val = dep_values.var()
                nan_count = dep_values.isna().sum()
                print(f"       - {dep_col}:")
                print(
                    f"         范围: [{dep_values.min():.6f}, {dep_values.max():.6f}]"
                )
                print(f"         唯一值数: {len(unique_vals)}")
                print(f"         方差: {var_val:.2e}")
                print(
                    f"         NaN 数量: {nan_count} ({nan_count/len(dep_values)*100:.1f}%)"
                )
                if len(unique_vals) <= 5:
                    print(f"         唯一值: {unique_vals.tolist()}")
            else:
                print(f"       - {dep_col}: ⚠️  列不存在")

        print()

    # 4. 检查 VPIN 相关特征为什么是 0
    print("4. 检查 VPIN 相关特征...")
    vpin_cols = [
        "vpin",
        "vpin_signed_imbalance",
        "trade_cluster_imbalance_ratio",
        "trade_cluster_directional_entropy",
    ]
    for col in vpin_cols:
        if col in df_sample.columns:
            values = df_sample[col]
            print(f"   {col}:")
            print(f"     范围: [{values.min():.6f}, {values.max():.6f}]")
            print(f"     均值: {values.mean():.6f}")
            print(f"     方差: {values.var():.2e}")
            print(
                f"     非零值数量: {(values != 0).sum()} ({(values != 0).sum()/len(values)*100:.1f}%)"
            )
            print(
                f"     NaN 数量: {values.isna().sum()} ({values.isna().sum()/len(values)*100:.1f}%)"
            )
        else:
            print(f"   {col}: ⚠️  列不存在")
    print()

    # 5. 检查 CVD 相关特征
    print("5. 检查 CVD 相关特征...")
    if "cvd" in df_sample.columns:
        cvd_values = df_sample["cvd"]
        print(f"   cvd:")
        print(f"     范围: [{cvd_values.min():.6f}, {cvd_values.max():.6f}]")
        print(f"     均值: {cvd_values.mean():.6f}")
        print(f"     方差: {cvd_values.var():.2e}")
        print(
            f"     变化率: {(cvd_values.diff() != 0).sum()} / {len(cvd_values)} ({(cvd_values.diff() != 0).sum()/len(cvd_values)*100:.1f}%)"
        )

        # 计算斜率
        def compute_slope(x):
            if len(x) > 1:
                return np.polyfit(range(len(x)), x, 1)[0]
            return 0.0

        cvd_slope = cvd_values.rolling(window=5, min_periods=1).apply(compute_slope)
        print(f"   cvd_slope_5 (手动计算):")
        print(f"     范围: [{cvd_slope.min():.6f}, {cvd_slope.max():.6f}]")
        print(f"     均值: {cvd_slope.mean():.6f}")
        print(f"     方差: {cvd_slope.var():.2e}")
        print(
            f"     非零值数量: {(cvd_slope != 0).sum()} ({(cvd_slope != 0).sum()/len(cvd_slope)*100:.1f}%)"
        )
    else:
        print(f"   cvd: ⚠️  列不存在")
    print()

    # 6. 检查 TBR 相关特征
    print("6. 检查 TBR 相关特征...")
    if "taker_buy_ratio" in df_sample.columns:
        tbr_values = df_sample["taker_buy_ratio"]
        print(f"   taker_buy_ratio:")
        print(f"     范围: [{tbr_values.min():.6f}, {tbr_values.max():.6f}]")
        print(f"     均值: {tbr_values.mean():.6f}")
        print(f"     方差: {tbr_values.var():.2e}")
        print(
            f"     接近 0.5 的数量: {((tbr_values >= 0.49) & (tbr_values <= 0.51)).sum()} ({((tbr_values >= 0.49) & (tbr_values <= 0.51)).sum()/len(tbr_values)*100:.1f}%)"
        )

        tbr_ma = tbr_values.rolling(window=5, min_periods=1).mean()
        print(f"   tbr_ma_5 (手动计算):")
        print(f"     范围: [{tbr_ma.min():.6f}, {tbr_ma.max():.6f}]")
        print(f"     均值: {tbr_ma.mean():.6f}")
        print(f"     方差: {tbr_ma.var():.2e}")
    else:
        print(f"   taker_buy_ratio: ⚠️  列不存在")
    print()

    print("=" * 80)
    print("检查完成")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy-config", default="config/strategies/sr_reversal_long"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--sample-size", type=int, default=2000)

    args = parser.parse_args()

    check_constant_feature_dependencies(
        strategy_config_path=args.strategy_config,
        symbol=args.symbol,
        data_path=args.data_path,
        sample_size=args.sample_size,
    )
