#!/usr/bin/env python3
"""
检查tick数据可用性

检查tick数据路径和格式，验证时间范围覆盖，确认包含必需的列。
"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_tools.tick_loader import list_tick_files, load_tick_data


def check_tick_data_availability(
    symbols: List[str],
    start_date: str,
    end_date: str,
    ticks_dir: str = "data/parquet_data",
) -> dict:
    """
    检查tick数据可用性

    Args:
        symbols: 交易对列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        ticks_dir: tick数据目录

    Returns:
        检查结果字典
    """
    results = {
        "ticks_dir": ticks_dir,
        "start_date": start_date,
        "end_date": end_date,
        "symbols": {},
        "overall_status": "unknown",
    }

    ticks_path = Path(ticks_dir)
    if not ticks_path.exists():
        print(f"❌ Tick数据目录不存在: {ticks_dir}")
        results["overall_status"] = "directory_missing"
        return results

    print(f"📁 Tick数据目录: {ticks_dir}")
    print(f"📅 时间范围: {start_date} 到 {end_date}")
    print("=" * 80)

    all_ok = True

    for symbol in symbols:
        print(f"\n检查 {symbol}:")
        symbol_result = {
            "status": "unknown",
            "files_found": [],
            "files_missing": [],
            "sample_data": None,
            "columns": [],
            "row_count": 0,
        }

        try:
            # 列出tick文件
            tick_files = list_tick_files(
                symbol=symbol,
                start_ts=f"{start_date} 00:00:00",
                end_ts=f"{end_date} 23:59:59",
                ticks_dir=ticks_dir,
            )

            symbol_result["files_found"] = tick_files
            print(f"  ✅ 找到 {len(tick_files)} 个tick文件")

            # 尝试加载样本数据
            try:
                sample_ticks = load_tick_data(
                    symbol=symbol,
                    start_ts=f"{start_date} 00:00:00",
                    end_ts=f"{start_date} 23:59:59",  # 只加载第一天作为样本
                    ticks_dir=ticks_dir,
                )

                if len(sample_ticks) > 0:
                    symbol_result["sample_data"] = "available"
                    symbol_result["row_count"] = len(sample_ticks)
                    symbol_result["columns"] = list(sample_ticks.columns)

                    # 检查必需的列（timestamp可能是索引）
                    required_cols = ["price", "volume", "side"]
                    missing_cols = [
                        col for col in required_cols if col not in sample_ticks.columns
                    ]

                    # 检查timestamp（可能是索引）
                    has_timestamp = (
                        sample_ticks.index.name == "timestamp"
                        or "timestamp" in sample_ticks.columns
                        or isinstance(sample_ticks.index, pd.DatetimeIndex)
                    )

                    if missing_cols:
                        print(f"  ❌ 缺少必需的列: {missing_cols}")
                        symbol_result["status"] = "missing_columns"
                        all_ok = False
                    elif not has_timestamp:
                        print(f"  ❌ 缺少timestamp（索引或列）")
                        symbol_result["status"] = "missing_timestamp"
                        all_ok = False
                    else:
                        print(f"  ✅ 必需的列都存在")
                        print(f"  📊 样本数据: {len(sample_ticks)} 行")
                        print(f"  📋 列: {', '.join(sample_ticks.columns)}")
                        if sample_ticks.index.name:
                            print(f"  📋 索引: {sample_ticks.index.name}")

                        # 检查数据类型
                        if isinstance(sample_ticks.index, pd.DatetimeIndex):
                            print(f"  ✅ timestamp索引类型正确")
                        if sample_ticks["side"].dtype not in [
                            "int64",
                            "int32",
                            "float64",
                            "float32",
                        ]:
                            print(f"  ⚠️  side列类型: {sample_ticks['side'].dtype}")

                        symbol_result["status"] = "ok"
                else:
                    print(f"  ⚠️  样本数据为空")
                    symbol_result["status"] = "empty_data"
                    all_ok = False

            except Exception as e:
                print(f"  ❌ 加载样本数据失败: {e}")
                symbol_result["status"] = "load_error"
                symbol_result["error"] = str(e)
                all_ok = False

        except FileNotFoundError as e:
            print(f"  ❌ 未找到tick文件: {e}")
            symbol_result["status"] = "files_not_found"
            symbol_result["error"] = str(e)
            all_ok = False
        except Exception as e:
            print(f"  ❌ 检查失败: {e}")
            symbol_result["status"] = "error"
            symbol_result["error"] = str(e)
            all_ok = False

        results["symbols"][symbol] = symbol_result

    results["overall_status"] = "ok" if all_ok else "issues_found"

    print("\n" + "=" * 80)
    if all_ok:
        print("✅ 所有symbol的tick数据检查通过")
    else:
        print("⚠️  部分symbol的tick数据存在问题")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="检查tick数据可用性")
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,ADAUSDT,BNBUSDT,SOLUSDT",
        help="交易对列表（逗号分隔）",
    )
    parser.add_argument(
        "--start-date", default="2025-05-01", help="开始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2025-10-31", help="结束日期 (YYYY-MM-DD)"
    )
    parser.add_argument("--ticks-dir", default="data/parquet_data", help="Tick数据目录")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    results = check_tick_data_availability(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        ticks_dir=args.ticks_dir,
    )

    # 返回退出码
    if results["overall_status"] == "ok":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
