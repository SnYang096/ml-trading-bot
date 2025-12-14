#!/usr/bin/env python3
"""
深入追踪 RSI inf 值问题

这个测试会：
1. 模拟完整的特征计算流程
2. 在每个步骤后检查 RSI 列的状态
3. 找出产生 inf 的具体步骤
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


def check_rsi_status(df: pd.DataFrame, step_name: str) -> dict:
    """检查 RSI 列的状态"""
    if "rsi" not in df.columns:
        return {"exists": False, "inf_count": 0, "nan_count": 0, "total": len(df)}

    rsi = df["rsi"]
    return {
        "exists": True,
        "inf_count": int(np.isinf(rsi).sum()),
        "nan_count": int(rsi.isna().sum()),
        "total": len(rsi),
        "valid_count": int(rsi.notna().sum()),
        "min": float(rsi.min()) if rsi.notna().any() else None,
        "max": float(rsi.max()) if rsi.notna().any() else None,
    }


def trace_rsi_through_pipeline():
    """追踪 RSI 在整个特征计算流程中的变化"""
    print("=" * 80)
    print("RSI Inf 值深度追踪测试")
    print("=" * 80)

    # 加载数据
    data_path = project_root / "data" / "parquet_data"
    symbol = "BTCUSDT"
    timeframe = "240T"
    start_date = "2025-01-01"
    end_date = "2025-07-31"

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

    # 步骤 1: 原始数据
    print(f"\n{'='*80}")
    print("步骤 1: 原始数据")
    print(f"{'='*80}")
    status = check_rsi_status(df_raw, "原始数据")
    print(f"   RSI 存在: {status['exists']}")
    if status["exists"]:
        print(f"   Inf 数量: {status['inf_count']}")
        print(f"   NaN 数量: {status['nan_count']}")

    # 步骤 2: 特征计算（使用完整的特征加载流程）
    print(f"\n{'='*80}")
    print("步骤 2: 特征计算（load_features_from_requested）")
    print(f"{'='*80}")

    # 模拟训练流程：先计算训练集特征
    split_idx = int(len(df_raw) * 0.85)
    df_train_raw = df_raw.iloc[:split_idx].copy()

    print(f"   训练集大小: {len(df_train_raw)}")
    print(f"   请求的特征: {strategy_config.features.requested_features[:10]}...")

    # 追踪特征计算过程
    try:
        # 在特征计算前检查
        print(f"\n   🔍 特征计算前检查...")
        status_before = check_rsi_status(df_train_raw, "计算前")
        print(f"      RSI 存在: {status_before['exists']}")

        # 执行特征计算
        print(f"\n   🔧 执行特征计算...")
        df_features = feature_loader.load_features_from_requested(
            df_train_raw,
            strategy_config.features.requested_features,
            fit=True,
        )

        # 在特征计算后检查
        print(f"\n   🔍 特征计算后检查...")
        status_after = check_rsi_status(df_features, "计算后")
        print(f"      RSI 存在: {status_after['exists']}")
        print(f"      Inf 数量: {status_after['inf_count']}")
        print(f"      NaN 数量: {status_after['nan_count']}")
        print(f"      有效数量: {status_after['valid_count']}")
        if status_after["min"] is not None:
            print(f"      范围: [{status_after['min']:.2f}, {status_after['max']:.2f}]")

        # 如果发现 inf，找出具体位置
        if status_after["inf_count"] > 0:
            print(f"\n   ⚠️  发现 {status_after['inf_count']} 个 inf 值！")
            rsi = df_features["rsi"]
            inf_mask = np.isinf(rsi)
            inf_indices = rsi[inf_mask].index[:20]

            print(f"\n   前 20 个 inf 值的位置:")
            for idx in inf_indices:
                rsi_val = rsi.loc[idx]
                print(f"      {idx}: RSI={rsi_val}")

                # 检查该位置的其他特征
                row = df_features.loc[idx]
                # 找出该行中也有 inf 的其他特征
                inf_cols = row[np.isinf(row)].index.tolist()
                if inf_cols:
                    print(f"         其他 inf 特征: {inf_cols[:5]}")

        # 步骤 3: 检查特征计算过程中的中间步骤
        print(f"\n{'='*80}")
        print("步骤 3: 检查特征计算中间过程")
        print(f"{'='*80}")

        # 尝试单独计算 RSI 特征
        print(f"\n   🔧 单独计算 RSI 特征...")
        from src.features.time_series.baseline_features import BaselineFeatureEngineer

        # 直接计算 RSI
        rsi_direct = BaselineFeatureEngineer.compute_rsi(
            df_train_raw["close"], period=14
        )
        status_direct = {
            "inf_count": int(np.isinf(rsi_direct).sum()),
            "nan_count": int(rsi_direct.isna().sum()),
        }
        print(
            f"      直接计算 RSI: Inf={status_direct['inf_count']}, NaN={status_direct['nan_count']}"
        )

        # 检查特征计算后的 RSI 与直接计算的差异
        if "rsi" in df_features.columns:
            rsi_computed = df_features["rsi"]

            # 对齐索引
            common_idx = rsi_direct.index.intersection(rsi_computed.index)
            if len(common_idx) > 0:
                rsi_direct_aligned = rsi_direct.loc[common_idx]
                rsi_computed_aligned = rsi_computed.loc[common_idx]

                # 检查差异
                inf_direct = np.isinf(rsi_direct_aligned)
                inf_computed = np.isinf(rsi_computed_aligned)

                diff_mask = inf_direct != inf_computed
                if diff_mask.any():
                    print(f"\n   ⚠️  发现差异！")
                    print(
                        f"      直接计算有 inf 但特征计算后没有: {(inf_direct & ~inf_computed).sum()}"
                    )
                    print(
                        f"      直接计算没有 inf 但特征计算后有: {(~inf_direct & inf_computed).sum()}"
                    )

                    # 找出差异位置
                    diff_indices = common_idx[diff_mask][:10]
                    for idx in diff_indices:
                        print(f"      {idx}:")
                        print(f"         直接计算: {rsi_direct_aligned.loc[idx]}")
                        print(f"         特征计算: {rsi_computed_aligned.loc[idx]}")

        # 步骤 4: 检查特征交互和组合
        print(f"\n{'='*80}")
        print("步骤 4: 检查特征交互和组合")
        print(f"{'='*80}")

        # 检查是否有其他特征列名包含 "rsi"
        rsi_related_cols = [col for col in df_features.columns if "rsi" in col.lower()]
        print(f"\n   包含 'rsi' 的特征列: {len(rsi_related_cols)}")
        for col in rsi_related_cols[:10]:
            col_data = df_features[col]
            inf_count = np.isinf(col_data).sum()
            if inf_count > 0:
                print(f"      ⚠️  {col}: {inf_count} 个 inf")
            else:
                print(f"      ✅ {col}: 无 inf")

        # 步骤 5: 检查数据过滤步骤
        print(f"\n{'='*80}")
        print("步骤 5: 检查数据过滤")
        print(f"{'='*80}")

        # 模拟 drop_inf_rows 操作
        feature_cols = [
            col
            for col in df_features.columns
            if col
            not in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "signal",
                "label",
                "atr",
                "_symbol",
                "symbol",
                "timestamp",
                "datetime",
                "date",
            ]
        ]

        if "rsi" in feature_cols:
            print(f"\n   检查 RSI 在特征列中...")
            status_before_filter = check_rsi_status(df_features, "过滤前")
            print(
                f"      过滤前: Inf={status_before_filter['inf_count']}, NaN={status_before_filter['nan_count']}"
            )

            # 模拟过滤（但不实际删除，只检查）
            rsi_before_filter = df_features["rsi"].copy()
            inf_mask = np.isinf(df_features[feature_cols])
            has_inf = inf_mask.any(axis=1)

            print(f"      有 inf 的行数: {has_inf.sum()}")
            print(
                f"      其中 RSI 有 inf 的行数: {(has_inf & np.isinf(rsi_before_filter)).sum()}"
            )
            print(
                f"      RSI 有 inf 但其他特征没有的行数: {(np.isinf(rsi_before_filter) & ~has_inf).sum()}"
            )

        print(f"\n✅ 追踪完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    trace_rsi_through_pipeline()
