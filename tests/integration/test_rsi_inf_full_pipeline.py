#!/usr/bin/env python3
"""
完整训练流程追踪 RSI inf 问题

模拟完整的训练流程，在每个关键步骤检查 RSI 状态
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_tools.data_loader import MarketDataLoader
from src.time_series_model.strategy_config import StrategyConfigLoader
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from scripts.train_strategy_pipeline import (
    run_feature_pipeline,
    determine_feature_columns,
    apply_filters,
    apply_post_label_filters,
    drop_inf_rows,
    import_callable,
)


def check_rsi_inf(df: pd.DataFrame, step_name: str) -> dict:
    """检查 RSI 的 inf 状态"""
    if "rsi" not in df.columns:
        return {"exists": False, "inf_count": 0}

    rsi = df["rsi"]
    inf_count = int(np.isinf(rsi).sum())
    nan_count = int(rsi.isna().sum())

    result = {
        "exists": True,
        "inf_count": inf_count,
        "nan_count": nan_count,
        "total": len(rsi),
    }

    if inf_count > 0:
        inf_indices = rsi[np.isinf(rsi)].index[:5]
        result["inf_indices"] = list(inf_indices)
        result["inf_values"] = [float(rsi.loc[idx]) for idx in inf_indices]

    return result


def trace_full_pipeline():
    """追踪完整训练流程"""
    print("=" * 80)
    print("完整训练流程 RSI Inf 追踪")
    print("=" * 80)

    # 加载数据
    data_path = project_root / "data" / "parquet_data"
    symbol = "BTCUSDT"
    timeframe = "240T"
    start_date = "2025-01-01"
    end_date = "2025-07-31"
    test_size = 0.15

    print(f"\n📂 加载数据...")
    loader = MarketDataLoader(data_path=str(data_path))
    df_raw = loader.load_data(
        symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
    )
    print(f"   ✅ 加载了 {len(df_raw)} 条数据")

    # 加载策略配置
    config_dir = project_root / "config" / "strategies" / "sr_reversal_long"
    print(f"\n📂 加载策略配置: {config_dir}")
    strategy_config_loader = StrategyConfigLoader(config_dir)
    strategy_config = strategy_config_loader.load()

    # 创建特征加载器
    feature_loader = StrategyFeatureLoader()

    # 步骤 1: 数据分割
    print(f"\n{'='*80}")
    print("步骤 1: 数据分割")
    print(f"{'='*80}")
    split_idx = int(len(df_raw) * (1 - test_size))
    df_train_raw = df_raw.iloc[:split_idx].copy()
    df_test_raw = df_raw.iloc[split_idx:].copy()
    print(f"   训练集: {len(df_train_raw)}, 测试集: {len(df_test_raw)}")

    # 步骤 2: 特征计算
    print(f"\n{'='*80}")
    print("步骤 2: 特征计算")
    print(f"{'='*80}")

    df_train_features = run_feature_pipeline(
        df_train_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_config.features,
        fit=True,
    )

    status = check_rsi_inf(df_train_features, "特征计算后")
    print(f"   RSI 存在: {status['exists']}")
    print(f"   Inf 数量: {status['inf_count']}")
    print(f"   NaN 数量: {status['nan_count']}")
    if status.get("inf_indices"):
        print(f"   Inf 位置: {status['inf_indices']}")

    # 步骤 3: 确定特征列
    print(f"\n{'='*80}")
    print("步骤 3: 确定特征列")
    print(f"{'='*80}")
    feature_cols = determine_feature_columns(
        df_train_features, strategy_config.features
    )
    print(f"   特征列数量: {len(feature_cols)}")
    print(f"   RSI 在特征列中: {'rsi' in feature_cols}")

    # 步骤 4: 标签生成
    print(f"\n{'='*80}")
    print("步骤 4: 标签生成")
    print(f"{'='*80}")

    label_func = import_callable(
        strategy_config.labels.generator.module,
        strategy_config.labels.generator.function,
    )

    # 注意：这里使用 copy()，但标签生成可能会修改 DataFrame
    df_train_with_label = df_train_features.copy()
    df_train_with_label[strategy_config.labels.target_column] = label_func(
        df_train_features.copy(), **strategy_config.labels.generator.params
    )

    status = check_rsi_inf(df_train_with_label, "标签生成后")
    print(f"   RSI 存在: {status['exists']}")
    print(f"   Inf 数量: {status['inf_count']}")
    print(f"   NaN 数量: {status['nan_count']}")
    if status.get("inf_indices"):
        print(f"   ⚠️  Inf 位置: {status['inf_indices']}")
        print(f"   Inf 值: {status['inf_values']}")

    # 步骤 5: 应用过滤器
    print(f"\n{'='*80}")
    print("步骤 5: 应用过滤器")
    print(f"{'='*80}")

    df_train_filtered = apply_filters(
        df_train_with_label, strategy_config.labels.filters
    )

    status = check_rsi_inf(df_train_filtered, "过滤后")
    print(f"   RSI 存在: {status['exists']}")
    print(f"   Inf 数量: {status['inf_count']}")
    print(f"   NaN 数量: {status['nan_count']}")
    if status.get("inf_indices"):
        print(f"   ⚠️  Inf 位置: {status['inf_indices']}")

    # 步骤 6: 应用后标签过滤器
    print(f"\n{'='*80}")
    print("步骤 6: 应用后标签过滤器")
    print(f"{'='*80}")

    df_train_post_filtered = apply_post_label_filters(
        df_train_filtered,
        strategy_config.labels.post_label_filters,
        feature_cols,
    )

    status = check_rsi_inf(df_train_post_filtered, "后标签过滤后")
    print(f"   RSI 存在: {status['exists']}")
    print(f"   Inf 数量: {status['inf_count']}")
    print(f"   NaN 数量: {status['nan_count']}")
    if status.get("inf_indices"):
        print(f"   ⚠️  Inf 位置: {status['inf_indices']}")

    # 步骤 7: 检查 drop_inf_rows 之前
    print(f"\n{'='*80}")
    print("步骤 7: drop_inf_rows 之前（这是训练日志显示的位置）")
    print(f"{'='*80}")

    # 模拟 _debug_inf 的检查逻辑
    if feature_cols and "rsi" in feature_cols:
        inf_mask = ~np.isfinite(df_train_post_filtered[["rsi"]])
        inf_count = inf_mask.sum().sum()
        print(f"   使用 _debug_inf 逻辑检查:")
        print(f"      RSI Inf 数量: {inf_count}")

        if inf_count > 0:
            rsi_col = df_train_post_filtered["rsi"]
            inf_indices = rsi_col[~np.isfinite(rsi_col)].index[:10]
            print(f"      ⚠️  发现 {inf_count} 个 inf 值！")
            print(f"      前 10 个位置:")
            for idx in inf_indices:
                val = rsi_col.loc[idx]
                print(f"         {idx}: {val} (type: {type(val).__name__})")

                # 检查是否是真正的 inf
                if np.isinf(val):
                    print(f"            ✅ 确认是 inf")
                elif pd.isna(val):
                    print(f"            ℹ️  是 NaN")
                else:
                    print(f"            ⚠️  不是 inf 也不是 NaN，但 isfinite 返回 False")

    # 步骤 8: drop_inf_rows
    print(f"\n{'='*80}")
    print("步骤 8: drop_inf_rows")
    print(f"{'='*80}")

    df_train_final = drop_inf_rows(df_train_post_filtered, feature_cols)

    status = check_rsi_inf(df_train_final, "drop_inf_rows 后")
    print(f"   RSI 存在: {status['exists']}")
    print(f"   Inf 数量: {status['inf_count']}")
    print(f"   NaN 数量: {status['nan_count']}")
    print(f"   剩余行数: {len(df_train_final)}")

    print(f"\n✅ 追踪完成")

    # 总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print(f"如果在步骤 7 发现了 inf，说明问题出现在:")
    print(f"  1. 特征计算（步骤 2）")
    print(f"  2. 标签生成（步骤 4）")
    print(f"  3. 过滤器（步骤 5-6）")
    print(f"  4. 或者是在特征列确定时的问题")


if __name__ == "__main__":
    trace_full_pipeline()
