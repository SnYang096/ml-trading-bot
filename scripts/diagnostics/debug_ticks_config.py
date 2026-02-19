"""
调试 ticks_loader_json 配置问题

检查：
1. _ensure_ticks_configured 是否正确设置 ticks_loader_json
2. load_features_from_requested 是否能获取到 ticks_loader_json
3. compute_features_parallel 是否能正确传递 ticks_loader_json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.time_series_model.strategy_config.loader import StrategyConfigLoader
from scripts.train_strategy_pipeline import (
    run_feature_pipeline,
    _ensure_ticks_configured,
)
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.data_tools.data_handler import MarketDataLoader


def debug_ticks_config(
    strategy_config_path: str = "config/strategies/sr_reversal_long",
    symbol: str = "BTCUSDT",
    data_path: str = "data/parquet_data",
):
    """调试 ticks_loader_json 配置"""
    print("=" * 80)
    print("调试 ticks_loader_json 配置")
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

    # 2. 创建特征加载器
    print("2. 创建特征加载器...")
    feature_loader = StrategyFeatureLoader()

    # 检查初始状态
    features_cfg = feature_loader.feature_deps.get("features", {})
    if "vpin_features" in features_cfg:
        vpin_cfg = features_cfg["vpin_features"]
        compute_params = vpin_cfg.get("compute_params", {})
        print(f"   vpin_features 初始 compute_params: {list(compute_params.keys())}")
        print(f"   初始 ticks_loader_json: {'ticks_loader_json' in compute_params}")
    print()

    # 3. 配置 ticks_loader_json
    print("3. 配置 ticks_loader_json...")
    start_ts = str(df_raw.index.min())
    end_ts = str(df_raw.index.max())
    requested_features = strategy_config.features.requested_features or []

    try:
        _ensure_ticks_configured(
            feature_loader,
            symbol=symbol,
            data_path=data_path,
            start_ts=start_ts,
            end_ts=end_ts,
            requested_features=requested_features,
        )
    except Exception as e:
        print(f"   ❌ 配置失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 检查配置后状态
    features_cfg = feature_loader.feature_deps.get("features", {})
    if "vpin_features" in features_cfg:
        vpin_cfg = features_cfg["vpin_features"]
        compute_params = vpin_cfg.get("compute_params", {})
        print(f"   vpin_features 配置后 compute_params: {list(compute_params.keys())}")
        print(f"   配置后 ticks_loader_json: {'ticks_loader_json' in compute_params}")
        if "ticks_loader_json" in compute_params:
            ticks_json = compute_params["ticks_loader_json"]
            print(f"   ticks_loader_json 长度: {len(ticks_json) if ticks_json else 0}")
    print()

    # 4. 检查 load_features_from_requested 中的状态
    print("4. 检查 load_features_from_requested 中的状态...")
    features_cfg = feature_loader.feature_deps.get("features", {})
    if "vpin_features" in features_cfg:
        vpin_cfg = features_cfg["vpin_features"]
        compute_params = vpin_cfg.get("compute_params", {})
        print(
            f"   load_features_from_requested 中 vpin_features compute_params: {list(compute_params.keys())}"
        )
        print(
            f"   load_features_from_requested 中 ticks_loader_json: {'ticks_loader_json' in compute_params}"
        )
    print()

    # 5. 尝试计算特征（只计算少量数据）
    print("5. 尝试计算特征（只计算少量数据）...")
    df_sample = df_raw.head(100).copy()

    try:
        df_features = run_feature_pipeline(
            df_sample,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_config.features,
            fit=True,
        )
        print(f"   ✅ 特征计算成功: {df_features.shape}")

        # 检查 VPIN 相关列是否存在
        vpin_cols = [col for col in df_features.columns if "vpin" in col.lower()]
        print(f"   VPIN 相关列: {vpin_cols}")
    except Exception as e:
        print(f"   ❌ 特征计算失败: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 80)
    print("调试完成")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy-config", default="config/strategies/sr_reversal_long"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-path", default="data/parquet_data")

    args = parser.parse_args()

    debug_ticks_config(
        strategy_config_path=args.strategy_config,
        symbol=args.symbol,
        data_path=args.data_path,
    )
