"""
集成测试：诊断 strategy_feature_compare.py 中 VPIN 特征配置问题

问题：在 strategy_feature_compare.py 中运行 VPIN 特征时出错，
但单独训练时没问题。需要找出 ticks_loader_json 为什么没有正确传递。
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategies.evaluation.strategy_feature_compare import (
    _configure_vpin_ticks,
)
from src.time_series_model.strategy_config.loader import StrategyConfigLoader


def test_vpin_ticks_configuration():
    """测试 VPIN ticks 配置是否正确传递到特征计算"""
    print("=" * 80)
    print("测试 1: 检查 _configure_vpin_ticks 是否正确设置 ticks_loader_json")
    print("=" * 80)

    # 创建测试数据
    symbol = "BTCUSDT"
    data_path = "data/parquet_data"  # 假设这是数据路径

    # 创建模拟的 DataFrame
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="1H")
    df = pd.DataFrame(
        {
            "open": np.random.randn(len(dates)) * 100 + 50000,
            "high": np.random.randn(len(dates)) * 100 + 50100,
            "low": np.random.randn(len(dates)) * 100 + 49900,
            "close": np.random.randn(len(dates)) * 100 + 50000,
            "volume": np.random.randn(len(dates)) * 1000 + 10000,
        },
        index=dates,
    )

    start_ts = str(df.index.min())
    end_ts = str(df.index.max())

    print(f"Symbol: {symbol}")
    print(f"Data path: {data_path}")
    print(f"Start TS: {start_ts}")
    print(f"End TS: {end_ts}")
    print()

    # 创建 feature_loader
    feature_loader = StrategyFeatureLoader()

    # 检查配置前
    features_cfg_before = feature_loader.feature_deps.get("features", {})
    vpin_cfg_before = features_cfg_before.get("vpin_features", {})
    compute_params_before = vpin_cfg_before.get("compute_params", {})
    ticks_loader_json_before = compute_params_before.get("ticks_loader_json")

    print("配置前:")
    print(f"  vpin_features 存在: {'vpin_features' in features_cfg_before}")
    if "vpin_features" in features_cfg_before:
        print(f"  compute_params 存在: {'compute_params' in vpin_cfg_before}")
        print(f"  ticks_loader_json 存在: {ticks_loader_json_before is not None}")
    print()

    # 尝试配置
    try:
        _configure_vpin_ticks(feature_loader, symbol, data_path, start_ts, end_ts)
        print("✅ _configure_vpin_ticks 执行成功")
    except Exception as e:
        print(f"❌ _configure_vpin_ticks 执行失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 检查配置后
    features_cfg_after = feature_loader.feature_deps.get("features", {})
    vpin_cfg_after = features_cfg_after.get("vpin_features", {})
    compute_params_after = vpin_cfg_after.get("compute_params", {})
    ticks_loader_json_after = compute_params_after.get("ticks_loader_json")

    print("配置后:")
    print(f"  vpin_features 存在: {'vpin_features' in features_cfg_after}")
    if "vpin_features" in features_cfg_after:
        print(f"  compute_params 存在: {'compute_params' in vpin_cfg_after}")
        print(f"  ticks_loader_json 存在: {ticks_loader_json_after is not None}")
        if ticks_loader_json_after:
            print(f"  ticks_loader_json 长度: {len(ticks_loader_json_after)}")
    print()

    # 验证配置是否正确
    if ticks_loader_json_after:
        print("✅ ticks_loader_json 已正确配置")
    else:
        print("❌ ticks_loader_json 未配置！")
        return

    print("\n" + "=" * 80)
    print("测试 2: 检查特征计算时是否能获取到 ticks_loader_json")
    print("=" * 80)

    # 尝试计算 VPIN 特征
    try:
        from src.features.loader.parallel_computer import ParallelFeatureComputer

        computer = ParallelFeatureComputer(
            cache_dir="cache/features",
            use_disk_cache=False,  # 禁用磁盘缓存以便测试
            use_memory_cache=False,  # 禁用内存缓存以便测试
        )

        features = feature_loader.feature_deps.get("features", {})
        requested_features = ["vpin_features"]

        print(f"请求的特征: {requested_features}")
        print(f"特征配置中存在 vpin_features: {'vpin_features' in features}")

        if "vpin_features" in features:
            vpin_info = features["vpin_features"]
            print(f"vpin_features 配置:")
            print(f"  compute_func: {vpin_info.get('compute_func')}")
            compute_params = vpin_info.get("compute_params", {})
            print(f"  compute_params keys: {list(compute_params.keys())}")
            print(
                f"  ticks_loader_json in compute_params: {'ticks_loader_json' in compute_params}"
            )
            if "ticks_loader_json" in compute_params:
                print(
                    f"  ticks_loader_json 值: {compute_params['ticks_loader_json'][:100]}..."
                )

        print("\n尝试计算 VPIN 特征...")
        result_df = computer.compute_features_parallel(
            df,
            features,
            requested_features,
            fit=True,
        )

        print(f"✅ VPIN 特征计算成功")
        print(f"结果 DataFrame 列: {list(result_df.columns)}")
        print(f"结果 DataFrame 形状: {result_df.shape}")

    except Exception as e:
        print(f"❌ VPIN 特征计算失败: {e}")
        import traceback

        traceback.print_exc()

        # 详细诊断
        print("\n" + "=" * 80)
        print("详细诊断信息:")
        print("=" * 80)

        features = feature_loader.feature_deps.get("features", {})
        if "vpin_features" in features:
            vpin_info = features["vpin_features"]
            compute_params = vpin_info.get("compute_params", {})
            print(f"compute_params 内容: {compute_params}")
            print(
                f"ticks_loader_json 类型: {type(compute_params.get('ticks_loader_json'))}"
            )
            print(f"ticks_loader_json 值: {compute_params.get('ticks_loader_json')}")


def test_feature_loader_config_persistence():
    """测试 feature_loader 配置是否在多次调用间保持"""
    print("\n" + "=" * 80)
    print("测试 3: 检查 feature_loader 配置持久性")
    print("=" * 80)

    feature_loader = StrategyFeatureLoader()

    # 第一次配置
    symbol = "BTCUSDT"
    data_path = "data/parquet_data"
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="1H")
    df = pd.DataFrame(
        {
            "open": np.random.randn(len(dates)) * 100 + 50000,
            "high": np.random.randn(len(dates)) * 100 + 50100,
            "low": np.random.randn(len(dates)) * 100 + 49900,
            "close": np.random.randn(len(dates)) * 100 + 50000,
            "volume": np.random.randn(len(dates)) * 1000 + 10000,
        },
        index=dates,
    )

    start_ts = str(df.index.min())
    end_ts = str(df.index.max())

    try:
        _configure_vpin_ticks(feature_loader, symbol, data_path, start_ts, end_ts)
        print("✅ 第一次配置成功")
    except Exception as e:
        print(f"❌ 第一次配置失败: {e}")
        return

    # 检查配置
    features_cfg = feature_loader.feature_deps.get("features", {})
    vpin_cfg = features_cfg.get("vpin_features", {})
    compute_params = vpin_cfg.get("compute_params", {})
    ticks_loader_json_1 = compute_params.get("ticks_loader_json")

    print(f"第一次配置后 ticks_loader_json: {ticks_loader_json_1 is not None}")

    # 模拟多次调用（类似 strategy_feature_compare.py 中的场景）
    for i in range(3):
        features_cfg_check = feature_loader.feature_deps.get("features", {})
        vpin_cfg_check = features_cfg_check.get("vpin_features", {})
        compute_params_check = vpin_cfg_check.get("compute_params", {})
        ticks_loader_json_check = compute_params_check.get("ticks_loader_json")

        print(
            f"第 {i+1} 次检查: ticks_loader_json = {ticks_loader_json_check is not None}"
        )

        if ticks_loader_json_check != ticks_loader_json_1:
            print(f"❌ 配置不一致！第 {i+1} 次检查时配置丢失")
            return

    print("✅ 配置在所有检查中保持一致")


def test_strategy_feature_compare_flow():
    """模拟 strategy_feature_compare.py 的完整流程"""
    print("\n" + "=" * 80)
    print("测试 4: 模拟 strategy_feature_compare.py 完整流程")
    print("=" * 80)

    # 模拟 execute_single_run 的流程
    symbol = "BTCUSDT"
    data_path = "data/parquet_data"

    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="1H")
    df_train_raw = pd.DataFrame(
        {
            "open": np.random.randn(len(dates)) * 100 + 50000,
            "high": np.random.randn(len(dates)) * 100 + 50100,
            "low": np.random.randn(len(dates)) * 100 + 49900,
            "close": np.random.randn(len(dates)) * 100 + 50000,
            "volume": np.random.randn(len(dates)) * 1000 + 10000,
        },
        index=dates,
    )

    feature_loader = StrategyFeatureLoader()

    # 步骤 1: 配置 tick loader
    if symbol and data_path and not df_train_raw.empty:
        start_ts = str(df_train_raw.index.min())
        end_ts = str(df_train_raw.index.max())
        print(f"步骤 1: 配置 tick loader")
        print(f"  start_ts: {start_ts}")
        print(f"  end_ts: {end_ts}")

        try:
            _configure_vpin_ticks(feature_loader, symbol, data_path, start_ts, end_ts)
            print("  ✅ 配置成功")
        except Exception as e:
            print(f"  ❌ 配置失败: {e}")
            return

        # 检查配置
        features_cfg = feature_loader.feature_deps.get("features", {})
        vpin_cfg = features_cfg.get("vpin_features", {})
        compute_params = vpin_cfg.get("compute_params", {})
        ticks_loader_json = compute_params.get("ticks_loader_json")
        print(f"  ticks_loader_json 配置: {ticks_loader_json is not None}")

    # 步骤 2: 运行特征管道（模拟 run_feature_pipeline）
    print(f"\n步骤 2: 运行特征管道")
    print(f"  使用 feature_loader.load_features_from_requested")

    try:
        from src.features.loader.parallel_computer import ParallelFeatureComputer

        # 检查 feature_loader 的 computer 是否使用相同的配置
        print(
            f"  feature_loader.feature_deps 类型: {type(feature_loader.feature_deps)}"
        )
        print(
            f"  feature_loader.feature_deps 是 dict: {isinstance(feature_loader.feature_deps, dict)}"
        )

        # 再次检查配置
        features_cfg_check = feature_loader.feature_deps.get("features", {})
        vpin_cfg_check = features_cfg_check.get("vpin_features", {})
        compute_params_check = vpin_cfg_check.get("compute_params", {})
        ticks_loader_json_check = compute_params_check.get("ticks_loader_json")
        print(f"  在 load_features_from_requested 前检查:")
        print(f"    ticks_loader_json: {ticks_loader_json_check is not None}")

        # 尝试加载特征
        requested_features = ["vpin_features"]
        result_df = feature_loader.load_features_from_requested(
            df_train_raw,
            requested_features,
            fit=True,
        )

        print(f"  ✅ 特征加载成功")
        print(f"  结果列: {list(result_df.columns)}")

    except Exception as e:
        print(f"  ❌ 特征加载失败: {e}")
        import traceback

        traceback.print_exc()

        # 详细诊断
        print("\n  详细诊断:")
        features_cfg_diag = feature_loader.feature_deps.get("features", {})
        if "vpin_features" in features_cfg_diag:
            vpin_info = features_cfg_diag["vpin_features"]
            compute_params_diag = vpin_info.get("compute_params", {})
            print(f"    compute_params keys: {list(compute_params_diag.keys())}")
            print(
                f"    ticks_loader_json: {compute_params_diag.get('ticks_loader_json') is not None}"
            )


if __name__ == "__main__":
    print("VPIN Ticks 配置诊断测试")
    print("=" * 80)
    print()

    test_vpin_ticks_configuration()
    test_feature_loader_config_persistence()
    test_strategy_feature_compare_flow()

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
