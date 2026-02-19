#!/usr/bin/env python3
"""
实时特征 vs 批量特征一致性验证测试（快速版）

模拟实时流 → 对比批量重算，30分钟内完成验证

使用方法:
    python tests/test_live_vs_batch_features.py --symbol BTCUSDT --duration 30

原理:
    1. 从磁盘加载最近1天的 1min bars/ticks
    2. 模拟实时流：每15分钟触发一次特征计算
    3. 同时用批量方式重算同一窗口的特征
    4. 对比两者差异
"""

import sys

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from src.live_data_stream.feature_storage import StorageManager
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)


def load_historical_data(
    symbol: str,
    storage_manager: StorageManager,
    test_days: int = 1,
    warmup_days: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """加载历史数据用于模拟

    Args:
        symbol: 交易对
        storage_manager: 存储管理器
        test_days: 测试时间范围（天）- 在这个范围内触发特征计算
        warmup_days: warmup 数据量（天）- 特征计算需要的历史数据

    Returns:
        (bars_1min, ticks_1min, test_start, test_end)
    """
    end_date = datetime.now().strftime("%Y-%m-%d")

    # warmup 数据从更早开始（需要 90+ 天）
    warmup_start = datetime.now() - timedelta(days=warmup_days)
    warmup_start_date = warmup_start.strftime("%Y-%m-%d")

    # 测试时间范围（最后 test_days 天）
    test_start = datetime.now() - timedelta(days=test_days)
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    test_end_ts = pd.Timestamp(datetime.now(), tz="UTC")

    # 加载所有数据（从 warmup_start 到 now）
    bars = storage_manager.bar_1min.load_range(symbol, warmup_start_date, end_date)
    ticks = storage_manager.ticks.load_range(symbol, warmup_start_date, end_date)

    print(f"✅ 加载数据: {len(bars)} bars, {len(ticks)} ticks")
    print(f"   Warmup 范围: {warmup_start_date} ~ {end_date}")
    print(
        f"   测试范围: {test_start.strftime('%Y-%m-%d')} ~ {end_date} ({test_days} 天)"
    )

    return bars, ticks, test_start_ts, test_end_ts


def simulate_realtime_stream(
    bars_1min: pd.DataFrame,
    ticks_1min: pd.DataFrame,
    feature_computer: IncrementalFeatureComputer,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    interval_minutes: int = 15,
) -> List[Dict[str, float]]:
    """模拟实时流计算特征

    在 test_start ~ test_end 范围内，每 interval_minutes 分钟触发一次 compute_features_batch
    每次触发时，使用从历史开始到当前时刻的所有数据

    Returns:
        List of feature dicts (每个代表一个时间点的特征)
    """
    print("\n🔄 模拟实时流...")

    # 确保 timestamp 列
    if "timestamp" not in bars_1min.columns:
        bars_1min = bars_1min.reset_index()
    if "timestamp" not in ticks_1min.columns:
        ticks_1min = ticks_1min.reset_index()

    # 转换为 tz-aware
    bars_1min["timestamp"] = pd.to_datetime(bars_1min["timestamp"], utc=True)
    ticks_1min["timestamp"] = pd.to_datetime(ticks_1min["timestamp"], utc=True)

    # 按时间排序
    bars_1min = bars_1min.sort_values("timestamp")
    ticks_1min = ticks_1min.sort_values("timestamp")

    # 确定数据的实际范围
    data_start = bars_1min["timestamp"].min()
    data_end = bars_1min["timestamp"].max()

    # 调整测试范围（确保在数据范围内）
    actual_test_start = max(test_start, data_start)
    actual_test_end = min(test_end, data_end)

    # 生成触发时间点（在测试范围内）
    trigger_times = pd.date_range(
        start=actual_test_start + timedelta(minutes=interval_minutes),
        end=actual_test_end,
        freq=f"{interval_minutes}min",
    )

    print(f"   数据范围: {data_start} ~ {data_end}")
    print(f"   测试范围: {actual_test_start} ~ {actual_test_end}")
    print(f"   触发次数: {len(trigger_times)} 次（每 {interval_minutes} 分钟）")

    realtime_features = []

    for i, trigger_time in enumerate(trigger_times):
        # 截取到当前时间点的数据（模拟实时流）
        bars_up_to_now = bars_1min[bars_1min["timestamp"] <= trigger_time].copy()
        ticks_up_to_now = ticks_1min[ticks_1min["timestamp"] <= trigger_time].copy()

        # 计算特征
        features = feature_computer.compute_features_batch(
            bars_1min=bars_up_to_now,
            ticks_1min=ticks_up_to_now,
            primary_timeframe="240T",
        )

        # 记录时间戳
        features["timestamp"] = trigger_time
        features["_trigger_index"] = i

        realtime_features.append(features)

        if (i + 1) % 10 == 0:
            print(
                f"   进度: {i+1}/{len(trigger_times)} ({(i+1)/len(trigger_times)*100:.1f}%)"
            )

    print(f"✅ 实时流模拟完成: {len(realtime_features)} 个特征快照")
    return realtime_features


def batch_compute_features(
    bars_1min: pd.DataFrame,
    ticks_1min: pd.DataFrame,
    feature_computer: IncrementalFeatureComputer,
    trigger_times: List[pd.Timestamp],
) -> List[Dict[str, float]]:
    """批量计算特征（对照组）

    对每个 trigger_time，用批量方式重新计算特征

    Returns:
        List of feature dicts
    """
    print("\n🔄 批量重算特征...")

    batch_features = []

    for i, trigger_time in enumerate(trigger_times):
        # 截取到当前时间点的数据
        bars_up_to_now = bars_1min[bars_1min["timestamp"] <= trigger_time].copy()
        ticks_up_to_now = ticks_1min[ticks_1min["timestamp"] <= trigger_time].copy()

        # 批量计算
        features = feature_computer.compute_features_batch(
            bars_1min=bars_up_to_now,
            ticks_1min=ticks_up_to_now,
            primary_timeframe="240T",
        )

        features["timestamp"] = trigger_time
        features["_trigger_index"] = i

        batch_features.append(features)

        if (i + 1) % 10 == 0:
            print(
                f"   进度: {i+1}/{len(trigger_times)} ({(i+1)/len(trigger_times)*100:.1f}%)"
            )

    print(f"✅ 批量计算完成: {len(batch_features)} 个特征快照")
    return batch_features


def compare_features(
    realtime_features: List[Dict[str, float]],
    batch_features: List[Dict[str, float]],
    tolerance: float = 1e-6,
) -> Dict[str, any]:
    """对比实时 vs 批量特征

    Returns:
        结果字典包含:
        - total_comparisons: 总对比次数
        - identical_count: 完全一致的次数
        - max_diff: 最大差异（按特征）
        - mismatches: 不一致的记录
    """
    print("\n📊 对比实时 vs 批量特征...")

    # 转换为 DataFrame
    df_realtime = pd.DataFrame(realtime_features)
    df_batch = pd.DataFrame(batch_features)

    # 对齐 index
    common_cols = set(df_realtime.columns) & set(df_batch.columns)
    common_cols.discard("timestamp")
    common_cols.discard("_trigger_index")

    feature_cols = sorted([c for c in common_cols if c != "_trigger_index"])

    print(f"   对比特征: {len(feature_cols)} 列")
    print(f"   对比时间点: {len(df_realtime)} 个")

    # 计算差异
    max_diff = {}
    mismatches = []
    identical_count = 0

    for col in feature_cols:
        if col not in df_realtime.columns or col not in df_batch.columns:
            continue

        diff = (df_realtime[col] - df_batch[col]).abs()
        max_diff[col] = diff.max()

        # 检查是否超过容差
        if max_diff[col] > tolerance:
            mismatch_indices = diff[diff > tolerance].index.tolist()
            for idx in mismatch_indices:
                mismatches.append(
                    {
                        "trigger_index": df_realtime.loc[idx, "_trigger_index"],
                        "timestamp": df_realtime.loc[idx, "timestamp"],
                        "feature": col,
                        "realtime_value": df_realtime.loc[idx, col],
                        "batch_value": df_batch.loc[idx, col],
                        "diff": diff.loc[idx],
                    }
                )

    # 统计完全一致的时间点
    for idx in df_realtime.index:
        row_identical = True
        for col in feature_cols:
            if col not in df_realtime.columns or col not in df_batch.columns:
                continue
            diff = abs(df_realtime.loc[idx, col] - df_batch.loc[idx, col])
            if diff > tolerance:
                row_identical = False
                break
        if row_identical:
            identical_count += 1

    result = {
        "total_comparisons": len(df_realtime),
        "identical_count": identical_count,
        "max_diff": max_diff,
        "mismatches": mismatches,
        "feature_cols": feature_cols,
    }

    return result


def print_comparison_report(result: Dict[str, any]):
    """打印对比报告"""
    print("\n" + "=" * 80)
    print("📋 实时 vs 批量特征一致性验证报告")
    print("=" * 80)

    total = result["total_comparisons"]
    identical = result["identical_count"]
    mismatch_count = len(result["mismatches"])

    print(f"\n✅ 总对比次数: {total}")
    print(f"✅ 完全一致: {identical} / {total} ({identical/total*100:.1f}%)")
    print(f"{'❌' if mismatch_count > 0 else '✅'} 不一致: {mismatch_count} 处")

    if mismatch_count > 0:
        print(f"\n⚠️ 最大差异（前10个特征）:")
        max_diff_sorted = sorted(
            result["max_diff"].items(), key=lambda x: x[1], reverse=True
        )[:10]

        for feature, diff in max_diff_sorted:
            print(f"   {feature}: {diff:.6e}")

        print(f"\n⚠️ 不一致详情（前5条）:")
        for mismatch in result["mismatches"][:5]:
            print(f"   时间点 {mismatch['trigger_index']}: {mismatch['feature']}")
            print(
                f"      实时={mismatch['realtime_value']:.6f}, 批量={mismatch['batch_value']:.6f}, 差异={mismatch['diff']:.6e}"
            )
    else:
        print("\n🎉 所有特征完全一致！实时特征流计算正确。")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="实时 vs 批量特征一致性验证（快速版）")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易对符号")
    parser.add_argument("--days", type=int, default=1, help="测试时间范围（天）")
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=100,
        help="Warmup 数据量（天），特征计算需要 90+ 天",
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="触发间隔（分钟），默认 60 分钟"
    )
    parser.add_argument("--live-root", default="live/highcap", help="实盘数据根目录")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="容差")

    args = parser.parse_args()

    print("=" * 80)
    print("🧪 实时 vs 批量特征一致性验证测试（快速版）")
    print("=" * 80)
    print(f"交易对: {args.symbol}")
    print(f"测试范围: 最近 {args.days} 天")
    print(f"Warmup 数据: {args.warmup_days} 天")
    print(f"触发间隔: {args.interval} 分钟")
    print(f"容差: {args.tolerance}")

    # 1. 初始化
    storage_manager = StorageManager(f"{args.live_root}/data")

    feature_computer = IncrementalFeatureComputer(
        primary_timeframe="240T",
    )

    # 2. 加载历史数据（包含 warmup 数据）
    bars_1min, ticks_1min, test_start, test_end = load_historical_data(
        args.symbol,
        storage_manager,
        test_days=args.days,
        warmup_days=args.warmup_days,
    )

    if len(bars_1min) == 0 or len(ticks_1min) == 0:
        print(
            f"❌ 没有找到数据，请先运行: bash live/scripts/prepare_warmup_ticks.sh highcap 6"
        )
        return 1

    # 3. 模拟实时流
    realtime_features = simulate_realtime_stream(
        bars_1min,
        ticks_1min,
        feature_computer,
        test_start=test_start,
        test_end=test_end,
        interval_minutes=args.interval,
    )

    if len(realtime_features) == 0:
        print("❌ 实时流模拟失败")
        return 1

    # 4. 批量重算
    trigger_times = [f["timestamp"] for f in realtime_features]
    batch_features = batch_compute_features(
        bars_1min,
        ticks_1min,
        feature_computer,
        trigger_times,
    )

    # 5. 对比
    result = compare_features(
        realtime_features,
        batch_features,
        tolerance=args.tolerance,
    )

    # 6. 打印报告
    print_comparison_report(result)

    # 7. 返回状态
    if len(result["mismatches"]) == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
