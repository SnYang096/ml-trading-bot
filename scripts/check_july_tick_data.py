#!/usr/bin/env python3
"""
检查 7 月 tick 数据是否存在

用于验证测试集 Trade Clustering 特征全为 NaN 的原因
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.tick_loader import load_tick_data, list_tick_files


def check_july_tick_data(symbol: str = "BTCUSDT", data_path: str = "data/parquet_data"):
    """检查 7 月 tick 数据"""
    print("=" * 70)
    print(f"检查 {symbol} 7 月 tick 数据")
    print("=" * 70)

    ticks_dir = Path(data_path)
    if not ticks_dir.exists():
        print(f"❌ Tick 数据目录不存在: {ticks_dir}")
        return

    # 列出所有 tick 文件（通过扫描目录）
    print(f"\n📁 Tick 数据目录: {ticks_dir}")
    tick_files = []
    if ticks_dir.exists():
        # 扫描目录中的 parquet 和 zip 文件
        for pattern in [f"{symbol}_*.parquet", f"{symbol}-aggTrades-*.zip"]:
            tick_files.extend(list(ticks_dir.glob(pattern)))
    print(f"   找到 {len(tick_files)} 个 tick 文件")

    # 检查 7 月文件
    july_files = [
        f
        for f in tick_files
        if "2025-07" in str(f) or "2025_07" in str(f) or "-2025-07" in str(f)
    ]
    print(f"\n📅 7 月文件数量: {len(july_files)}")

    if july_files:
        print("   7 月文件列表:")
        for f in sorted(july_files):
            print(f"     - {f.name}")
    else:
        print("   ⚠️  未找到 7 月文件")

    # 尝试加载 7 月数据
    print(f"\n🔍 尝试加载 7 月数据...")
    start_ts = "2025-07-01 00:00:00"
    end_ts = "2025-07-31 23:59:59"

    try:
        ticks = load_tick_data(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            ticks_dir=ticks_dir,
            lookback_minutes=0,
        )

        if ticks is not None and len(ticks) > 0:
            print(f"   ✅ 成功加载 7 月数据")
            print(f"   📊 数据量: {len(ticks)} 条")
            print(f"   📅 时间范围: {ticks.index.min()} 到 {ticks.index.max()}")
            print(f"   📈 列: {list(ticks.columns)}")

            # 检查数据质量
            print(f"\n📊 数据质量检查:")
            print(f"   - 缺失值: {ticks.isna().sum().to_dict()}")
            print(
                f"   - 唯一 side 值: {ticks['side'].unique() if 'side' in ticks.columns else 'N/A'}"
            )
            print(
                f"   - side 分布: {ticks['side'].value_counts().to_dict() if 'side' in ticks.columns else 'N/A'}"
            )
        else:
            print(f"   ❌ 7 月数据为空或不存在")
            print(f"   💡 可能原因:")
            print(f"      1. 7 月 tick 数据文件不存在")
            print(f"      2. 数据文件存在但时间范围不匹配")
            print(f"      3. 数据文件损坏")

            # 检查相邻月份的数据
            print(f"\n🔍 检查相邻月份数据...")
            for month in ["2025-06", "2025-08"]:
                month_start = f"{month}-01 00:00:00"
                month_end = f"{month}-{31 if month == '2025-08' else '30'} 23:59:59"
                try:
                    month_ticks = load_tick_data(
                        symbol=symbol,
                        start_ts=month_start,
                        end_ts=month_end,
                        ticks_dir=ticks_dir,
                        lookback_minutes=0,
                    )
                    if month_ticks is not None and len(month_ticks) > 0:
                        print(f"   ✅ {month}: {len(month_ticks)} 条数据")
                    else:
                        print(f"   ❌ {month}: 无数据")
                except Exception as e:
                    print(f"   ⚠️  {month}: 加载失败 - {e}")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        import traceback

        traceback.print_exc()

    # 列出所有可用月份
    print(f"\n📅 所有可用月份:")
    months = set()
    for f in tick_files:
        # 尝试从文件名提取月份
        name = f.name
        if "2025" in name:
            # 格式可能是 2025-07 或 2025_07
            parts = name.replace("-", "_").split("_")
            for i, part in enumerate(parts):
                if part == "2025" and i + 1 < len(parts):
                    month_str = parts[i + 1]
                    if month_str.isdigit() and 1 <= int(month_str) <= 12:
                        months.add(f"2025-{month_str.zfill(2)}")

    for month in sorted(months):
        print(f"   - {month}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="检查 7 月 tick 数据")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对符号")
    parser.add_argument(
        "--data-path", type=str, default="data/parquet_data", help="数据路径"
    )

    args = parser.parse_args()
    check_july_tick_data(symbol=args.symbol, data_path=args.data_path)
