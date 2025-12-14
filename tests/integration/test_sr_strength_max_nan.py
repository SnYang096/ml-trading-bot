#!/usr/bin/env python3
"""
集成测试：追踪 sr_strength_max 全部为 NaN 的问题

从训练日志看到：
- 训练集中 sr_strength_max 有 706 个 inf（实际可能是 NaN）
- 全部为 NaN，说明计算过程中某个步骤失败了

这个测试会：
1. 加载实际数据
2. 逐步追踪 sr_strength_max 的计算流程
3. 找出为什么全部为 NaN
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_tools.data_loader import MarketDataLoader
from src.features.loader.feature_wrappers import compute_sr_strength_max
from src.features.time_series.baseline_features import BaselineFeatureEngineer
from src.features.loader.feature_wrappers import (
    compute_sqs_hal_high,
    compute_sqs_hal_low,
)


def check_sr_strength_max_status(df: pd.DataFrame, step_name: str) -> dict:
    """检查 sr_strength_max 的状态"""
    if "sr_strength_max" not in df.columns:
        return {"exists": False, "inf_count": 0, "nan_count": 0, "total": len(df)}

    sr_strength = df["sr_strength_max"]
    return {
        "exists": True,
        "inf_count": int(np.isinf(sr_strength).sum()),
        "nan_count": int(sr_strength.isna().sum()),
        "total": len(sr_strength),
        "valid_count": int(sr_strength.notna().sum()),
        "zero_count": int((sr_strength == 0.0).sum()),
        "min": float(sr_strength.min()) if sr_strength.notna().any() else None,
        "max": float(sr_strength.max()) if sr_strength.notna().any() else None,
        "mean": float(sr_strength.mean()) if sr_strength.notna().any() else None,
    }


def trace_sr_strength_max():
    """追踪 sr_strength_max 的计算流程"""
    print("=" * 80)
    print("SR Strength Max NaN 问题诊断测试")
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

    if df_raw.empty:
        print("   ⚠️  数据为空，跳过测试")
        return

    print(f"   ✅ 加载了 {len(df_raw)} 条数据")

    # 模拟训练集分割
    split_idx = int(len(df_raw) * 0.85)
    df_train = df_raw.iloc[:split_idx].copy()
    print(f"   训练集大小: {len(df_train)}")

    # 步骤 1: 检查基础数据
    print(f"\n{'='*80}")
    print("步骤 1: 检查基础数据")
    print(f"{'='*80}")

    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df_train.columns]
    if missing_cols:
        print(f"   ❌ 缺少必需列: {missing_cols}")
        return

    print(f"   ✅ 基础列存在: {required_cols}")

    # 步骤 2: 检查依赖特征（sqs_hal_high 和 sqs_hal_low）
    print(f"\n{'='*80}")
    print("步骤 2: 检查依赖特征（sqs_hal_high, sqs_hal_low）")
    print(f"{'='*80}")

    df_with_sqs = df_train.copy()

    # 计算 sqs_hal_high
    if "sqs_hal_high" not in df_with_sqs.columns:
        print(f"   🔧 计算 sqs_hal_high...")
        try:
            df_with_sqs = compute_sqs_hal_high(df_with_sqs)
            sqs_high_status = check_sr_strength_max_status(df_with_sqs, "sqs_hal_high")
            print(f"      sqs_hal_high 存在: {'sqs_hal_high' in df_with_sqs.columns}")
            if "sqs_hal_high" in df_with_sqs.columns:
                sqs_high = df_with_sqs["sqs_hal_high"]
                print(f"      NaN 数量: {sqs_high.isna().sum()}")
                print(f"      有效数量: {sqs_high.notna().sum()}")
                if sqs_high.notna().any():
                    print(f"      范围: [{sqs_high.min():.4f}, {sqs_high.max():.4f}]")
        except Exception as e:
            print(f"      ❌ 计算 sqs_hal_high 失败: {e}")
            import traceback

            traceback.print_exc()

    # 计算 sqs_hal_low
    if "sqs_hal_low" not in df_with_sqs.columns:
        print(f"   🔧 计算 sqs_hal_low...")
        try:
            df_with_sqs = compute_sqs_hal_low(df_with_sqs)
            print(f"      sqs_hal_low 存在: {'sqs_hal_low' in df_with_sqs.columns}")
            if "sqs_hal_low" in df_with_sqs.columns:
                sqs_low = df_with_sqs["sqs_hal_low"]
                print(f"      NaN 数量: {sqs_low.isna().sum()}")
                print(f"      有效数量: {sqs_low.notna().sum()}")
                if sqs_low.notna().any():
                    print(f"      范围: [{sqs_low.min():.4f}, {sqs_low.max():.4f}]")
        except Exception as e:
            print(f"      ❌ 计算 sqs_hal_low 失败: {e}")
            import traceback

            traceback.print_exc()

    # 步骤 3: 检查 ATR（边界强度计算需要）
    print(f"\n{'='*80}")
    print("步骤 3: 检查 ATR（边界强度计算必需）")
    print(f"{'='*80}")

    if "atr" not in df_with_sqs.columns:
        print(f"   ⚠️  ATR 列不存在！这是导致边界强度计算失败的关键原因")
        print(f"   🔧 尝试计算 ATR...")
        try:
            from src.features.time_series.baseline_features import (
                BaselineFeatureEngineer,
            )

            df_with_sqs["atr"] = BaselineFeatureEngineer.compute_atr(
                df_with_sqs["high"], df_with_sqs["low"], df_with_sqs["close"], period=14
            )
            print(f"      ✅ ATR 计算成功")
            print(f"         NaN 数量: {df_with_sqs['atr'].isna().sum()}")
            print(f"         有效数量: {df_with_sqs['atr'].notna().sum()}")
        except Exception as e:
            print(f"      ❌ ATR 计算失败: {e}")
    else:
        print(f"   ✅ ATR 列存在")
        atr = df_with_sqs["atr"]
        print(f"      NaN 数量: {atr.isna().sum()}")
        print(f"      有效数量: {atr.notna().sum()}")
        if atr.notna().any():
            print(f"      范围: [{atr.min():.2f}, {atr.max():.2f}]")

    # 步骤 4: 检查边界定义
    print(f"\n{'='*80}")
    print("步骤 4: 检查边界定义（_get_sr_boundary_definitions）")
    print(f"{'='*80}")

    try:
        boundaries = BaselineFeatureEngineer._get_sr_boundary_definitions(df_with_sqs)
        print(f"   边界数量: {len(boundaries) if boundaries else 0}")

        if not boundaries:
            print(
                f"   ⚠️  没有找到边界定义！这是导致 sr_strength_max 全部为 NaN 的可能原因"
            )
            print(f"   检查依赖特征...")

            # 检查需要的列
            needed_cols = ["hal_high", "hal_low", "poc"]
            for col in needed_cols:
                if col in df_with_sqs.columns:
                    col_data = df_with_sqs[col]
                    print(
                        f"      {col}: 存在, NaN={col_data.isna().sum()}, 有效={col_data.notna().sum()}"
                    )
                else:
                    print(f"      {col}: ❌ 不存在")
        else:
            print(f"   ✅ 找到 {len(boundaries)} 个边界")
            for i, boundary in enumerate(boundaries[:5]):
                print(f"      边界 {i+1}: {boundary}")
                # 检查边界列是否存在且有效
                col_name = boundary["column"]
                if col_name in df_with_sqs.columns:
                    col_data = df_with_sqs[col_name]
                    print(
                        f"         列 '{col_name}': 存在, NaN={col_data.isna().sum()}, 有效={col_data.notna().sum()}"
                    )
                    if col_data.notna().any():
                        print(
                            f"         范围: [{col_data.min():.2f}, {col_data.max():.2f}]"
                        )
                else:
                    print(f"         列 '{col_name}': ❌ 不存在")
                    # 如果 poc 不存在，尝试计算
                    if col_name == "poc":
                        print(f"         🔧 尝试计算 poc...")
                        try:
                            result = BaselineFeatureEngineer.add_poc_hal_dimensionless_features(
                                df_with_sqs,
                                required_features={"poc"},
                                poc_window=160,
                            )
                            if "poc" in result.columns:
                                df_with_sqs["poc"] = result["poc"]
                                print(f"         ✅ poc 计算成功")
                            else:
                                print(f"         ❌ poc 计算失败")
                        except Exception as e:
                            print(f"         ❌ poc 计算失败: {e}")
    except Exception as e:
        print(f"   ❌ 获取边界定义失败: {e}")
        import traceback

        traceback.print_exc()

    # 步骤 5: 计算 sr_strength_max
    print(f"\n{'='*80}")
    print("步骤 5: 计算 sr_strength_max")
    print(f"{'='*80}")

    df_result = df_with_sqs.copy()

    try:
        df_result = compute_sr_strength_max(df_result)

        if df_result is None:
            print(f"   ❌ compute_sr_strength_max 返回 None")
            return

        status = check_sr_strength_max_status(df_result, "计算后")
        print(f"   sr_strength_max 存在: {status['exists']}")
        print(f"   Inf 数量: {status['inf_count']}")
        print(f"   NaN 数量: {status['nan_count']}")
        print(f"   有效数量: {status['valid_count']}")
        print(f"   零值数量: {status['zero_count']}")

        if status["min"] is not None:
            print(f"   范围: [{status['min']:.4f}, {status['max']:.4f}]")
            print(f"   均值: {status['mean']:.4f}")

        if status["nan_count"] == status["total"]:
            print(f"\n   ⚠️  全部为 NaN！")
            print(f"   可能的原因:")
            print(f"      1. 边界定义为空（_get_sr_boundary_definitions 返回空）")
            print(f"      2. 边界强度计算失败（_compute_boundary_strengths 返回空）")
            print(f"      3. 依赖特征（sqs_hal_high/sqs_hal_low）全部为 NaN")

            # 进一步诊断
            if (
                "sqs_hal_high" in df_result.columns
                and "sqs_hal_low" in df_result.columns
            ):
                sqs_high = df_result["sqs_hal_high"]
                sqs_low = df_result["sqs_hal_low"]
                print(f"\n   依赖特征状态:")
                print(
                    f"      sqs_hal_high: NaN={sqs_high.isna().sum()}, 有效={sqs_high.notna().sum()}"
                )
                print(
                    f"      sqs_hal_low: NaN={sqs_low.isna().sum()}, 有效={sqs_low.notna().sum()}"
                )

        elif status["valid_count"] > 0:
            print(f"   ✅ 有 {status['valid_count']} 个有效值")

            # 检查有效值的分布
            valid_values = df_result["sr_strength_max"][
                df_result["sr_strength_max"].notna()
            ]
            print(f"   有效值统计:")
            print(f"      最小值: {valid_values.min():.4f}")
            print(f"      最大值: {valid_values.max():.4f}")
            print(f"      均值: {valid_values.mean():.4f}")
            print(f"      中位数: {valid_values.median():.4f}")

    except Exception as e:
        print(f"   ❌ 计算 sr_strength_max 失败: {e}")
        import traceback

        traceback.print_exc()

    # 步骤 6: 手动检查边界强度计算
    print(f"\n{'='*80}")
    print("步骤 6: 手动检查边界强度计算")
    print(f"{'='*80}")

    try:
        boundaries = BaselineFeatureEngineer._get_sr_boundary_definitions(df_with_sqs)
        if boundaries:
            print(f"   尝试计算边界强度...")
            compression_series = df_with_sqs.get("compression_confidence")
            if compression_series is None:
                print(f"   ⚠️  compression_confidence 不存在，使用 None")

            boundary_strengths = BaselineFeatureEngineer._compute_boundary_strengths(
                data=df_with_sqs,
                boundaries=boundaries,
                window=60,
                tolerance_factor=0.5,
                compression_series=compression_series,
            )

            print(
                f"   边界强度数量: {len(boundary_strengths) if boundary_strengths else 0}"
            )

            if not boundary_strengths:
                print(f"   ⚠️  边界强度计算返回空！")
            else:
                print(f"   ✅ 计算了 {len(boundary_strengths)} 个边界强度序列")
                for i, (name, strength_series) in enumerate(
                    list(boundary_strengths.items())[:3]
                ):
                    print(f"      边界 {i+1} ({name}):")
                    print(f"         长度: {len(strength_series)}")
                    print(f"         有效值: {strength_series.notna().sum()}")
                    print(f"         NaN: {strength_series.isna().sum()}")
                    if strength_series.notna().any():
                        print(
                            f"         范围: [{strength_series.min():.4f}, {strength_series.max():.4f}]"
                        )
        else:
            print(f"   ⚠️  没有边界定义，跳过边界强度计算")
    except Exception as e:
        print(f"   ❌ 边界强度计算失败: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n✅ 测试完成")


if __name__ == "__main__":
    trace_sr_strength_max()
