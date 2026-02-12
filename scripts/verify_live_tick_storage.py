#!/usr/bin/env python3
"""
实盘tick数据存储验证脚本

验证内容：
1. tick数据正确保存（格式、频率、完整性）
2. VPIN特征计算使用新tick数据
3. 存储空间和性能监控
"""
import sys
import time
import psutil
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from live_data_stream.feature_storage import StorageManager, TickStorage
from features.time_series.utils_order_flow_features import compute_vpin_from_ticks


class LiveTickStorageVerifier:
    """实盘tick存储验证器"""

    def __init__(self, base_path: str = "live/highcap/data"):
        self.base_path = Path(base_path)
        self.storage_manager = StorageManager(base_path=self.base_path)
        self.results = {
            "timestamp": datetime.now(),
            "tests": {},
            "storage_stats": {},
            "performance_stats": {},
        }

    def verify_tick_format(self, symbol: str, date: str) -> dict:
        """验证1: tick数据格式正确"""
        print("\n" + "=" * 80)
        print("验证1: tick数据格式正确性")
        print("=" * 80)

        result = {
            "passed": False,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        try:
            # 加载tick数据
            ticks = self.storage_manager.ticks.load(symbol, date)

            if len(ticks) == 0:
                result["errors"].append(f"❌ 未找到{date}的tick数据")
                print(f"❌ 未找到{symbol} {date}的tick数据")
                return result

            print(f"✅ 成功加载 {len(ticks)} 条tick记录")

            # 1. 验证列格式
            required_cols = ["timestamp", "price", "volume", "side"]
            missing_cols = [col for col in required_cols if col not in ticks.columns]
            if missing_cols:
                result["errors"].append(f"缺少必要列: {missing_cols}")
                print(f"❌ 缺少必要列: {missing_cols}")
                return result
            print(f"✅ 列格式正确: {list(ticks.columns)}")

            # 2. 验证side值
            unique_sides = sorted(ticks["side"].unique())
            if not set(unique_sides).issubset({-1, 1}):
                result["errors"].append(f"side值异常: {unique_sides}")
                print(f"❌ side值异常: {unique_sides}")
                return result
            print(f"✅ side值正确: {unique_sides}")

            # 3. 验证时间戳顺序
            if not ticks["timestamp"].is_monotonic_increasing:
                result["warnings"].append("时间戳未严格递增")
                print(f"⚠️  时间戳未严格递增")
            else:
                print(f"✅ 时间戳严格递增")

            # 4. 统计每分钟的记录数
            ticks["minute"] = pd.to_datetime(ticks["timestamp"]).dt.floor("1min")
            records_per_minute = ticks.groupby("minute").size()

            print(f"\n📊 每分钟记录数统计:")
            print(f"   平均: {records_per_minute.mean():.2f} 条/分钟")
            print(f"   最小: {records_per_minute.min()} 条/分钟")
            print(f"   最大: {records_per_minute.max()} 条/分钟")
            print(f"   中位数: {records_per_minute.median():.0f} 条/分钟")

            # 期望每分钟2条（买卖分离）
            if records_per_minute.median() != 2:
                result["warnings"].append(
                    f"每分钟记录数中位数为{records_per_minute.median()}，期望为2"
                )
                print(f"⚠️  每分钟记录数中位数为{records_per_minute.median()}，期望为2")

            # 5. 检查买卖分布
            side_counts = ticks["side"].value_counts()
            print(f"\n📊 买卖分布:")
            print(
                f"   买方 (side=1): {side_counts.get(1, 0)} 条 ({side_counts.get(1, 0)/len(ticks)*100:.1f}%)"
            )
            print(
                f"   卖方 (side=-1): {side_counts.get(-1, 0)} 条 ({side_counts.get(-1, 0)/len(ticks)*100:.1f}%)"
            )

            # 6. 检查数据完整性（是否有大的时间间隔）
            ticks_sorted = ticks.sort_values("timestamp")
            time_diffs = ticks_sorted["timestamp"].diff()
            max_gap = time_diffs.max()

            print(f"\n📊 数据完整性:")
            print(
                f"   时间范围: {ticks['timestamp'].min()} ~ {ticks['timestamp'].max()}"
            )
            print(f"   最大间隔: {max_gap}")

            if max_gap > pd.Timedelta(minutes=5):
                result["warnings"].append(f"检测到大的时间间隔: {max_gap}")
                print(f"⚠️  检测到大的时间间隔: {max_gap}")

            result["passed"] = len(result["errors"]) == 0
            result["stats"] = {
                "total_records": len(ticks),
                "unique_minutes": len(records_per_minute),
                "avg_records_per_minute": float(records_per_minute.mean()),
                "buy_count": int(side_counts.get(1, 0)),
                "sell_count": int(side_counts.get(-1, 0)),
                "time_range": {
                    "start": str(ticks["timestamp"].min()),
                    "end": str(ticks["timestamp"].max()),
                },
                "max_gap_minutes": (
                    float(max_gap.total_seconds() / 60) if pd.notna(max_gap) else None
                ),
            }

            print(f"\n✅ tick数据格式验证通过！")

        except Exception as e:
            result["errors"].append(f"验证过程异常: {str(e)}")
            print(f"❌ 验证过程异常: {e}")
            import traceback

            traceback.print_exc()

        return result

    def verify_vpin_computation(self, symbol: str, date: str) -> dict:
        """验证2: VPIN特征计算使用新tick数据"""
        print("\n" + "=" * 80)
        print("验证2: VPIN特征计算使用新tick数据")
        print("=" * 80)

        result = {
            "passed": False,
            "errors": [],
            "warnings": [],
            "vpin_stats": {},
        }

        try:
            # 1. 加载tick数据
            ticks = self.storage_manager.ticks.load(symbol, date)

            if len(ticks) == 0:
                result["errors"].append(f"未找到{date}的tick数据")
                print(f"❌ 未找到tick数据")
                return result

            print(f"✅ 加载 {len(ticks)} 条tick记录")

            # 2. 验证tick数据格式
            required_cols = ["timestamp", "price", "volume", "side"]
            missing_cols = [col for col in required_cols if col not in ticks.columns]
            if missing_cols:
                result["errors"].append(f"tick数据缺少必要列: {missing_cols}")
                print(f"❌ tick数据缺少必要列: {missing_cols}")
                return result

            # 3. 设置timestamp为索引
            if "timestamp" in ticks.columns:
                ticks_indexed = ticks.set_index("timestamp").sort_index()
            else:
                ticks_indexed = ticks

            print(f"✅ tick数据格式正确")

            # 4. 计算VPIN
            print(f"\n📊 计算VPIN特征...")
            start_time = time.time()

            try:
                vpin_df = compute_vpin_from_ticks(
                    ticks=ticks_indexed,
                    bucket_volume=None,  # 使用自适应桶大小
                    n_buckets=50,
                )

                compute_time = time.time() - start_time

                print(f"✅ VPIN计算成功！耗时: {compute_time:.2f}秒")
                print(f"   生成特征: {list(vpin_df.columns)}")
                print(f"   特征条数: {len(vpin_df)}")

                # 5. 分析VPIN结果
                if "vpin" in vpin_df.columns:
                    vpin_values = vpin_df["vpin"].dropna()

                    print(f"\n📊 VPIN统计:")
                    print(f"   均值: {vpin_values.mean():.4f}")
                    print(f"   标准差: {vpin_values.std():.4f}")
                    print(f"   最小值: {vpin_values.min():.4f}")
                    print(f"   最大值: {vpin_values.max():.4f}")
                    print(f"   中位数: {vpin_values.median():.4f}")

                    # 检查VPIN值是否合理（应该在0-1之间）
                    if vpin_values.max() > 1.0 or vpin_values.min() < 0.0:
                        result["warnings"].append(
                            f"VPIN值超出[0,1]范围: [{vpin_values.min():.4f}, {vpin_values.max():.4f}]"
                        )
                        print(f"⚠️  VPIN值超出[0,1]范围")

                    # 检查VPIN是否全是相同值（可能计算失败）
                    if vpin_values.std() < 0.001:
                        result["warnings"].append(
                            f"VPIN标准差过小({vpin_values.std():.6f})，可能计算异常"
                        )
                        print(f"⚠️  VPIN标准差过小，可能计算异常")

                    result["vpin_stats"] = {
                        "mean": float(vpin_values.mean()),
                        "std": float(vpin_values.std()),
                        "min": float(vpin_values.min()),
                        "max": float(vpin_values.max()),
                        "median": float(vpin_values.median()),
                        "count": int(len(vpin_values)),
                        "compute_time_seconds": compute_time,
                    }

                    result["passed"] = True
                    print(f"\n✅ VPIN特征计算验证通过！")
                else:
                    result["errors"].append("VPIN计算结果中未包含'vpin'列")
                    print(f"❌ VPIN计算结果中未包含'vpin'列")

            except Exception as e:
                result["errors"].append(f"VPIN计算异常: {str(e)}")
                print(f"❌ VPIN计算异常: {e}")
                import traceback

                traceback.print_exc()

        except Exception as e:
            result["errors"].append(f"验证过程异常: {str(e)}")
            print(f"❌ 验证过程异常: {e}")
            import traceback

            traceback.print_exc()

        return result

    def verify_storage_performance(self, symbol: str, date: str) -> dict:
        """验证3: 存储空间和性能监控"""
        print("\n" + "=" * 80)
        print("验证3: 存储空间和性能监控")
        print("=" * 80)

        result = {
            "passed": False,
            "errors": [],
            "storage_stats": {},
            "performance_stats": {},
        }

        try:
            # 1. 存储空间统计
            print(f"\n📊 存储空间统计:")

            ticks_dir = self.base_path / "ticks" / symbol
            bars_dir = self.base_path / "bars" / symbol

            def get_dir_size(path: Path) -> int:
                """获取目录大小（字节）"""
                if not path.exists():
                    return 0
                total = 0
                for file in path.rglob("*.parquet"):
                    total += file.stat().st_size
                return total

            ticks_size = get_dir_size(ticks_dir)
            bars_size = get_dir_size(bars_dir)

            print(f"   ticks目录: {ticks_size / 1024 / 1024:.2f} MB")
            print(f"   bars目录: {bars_size / 1024 / 1024:.2f} MB")
            print(f"   总计: {(ticks_size + bars_size) / 1024 / 1024:.2f} MB")

            # 2. 单日数据大小
            tick_file = ticks_dir / f"{date}.parquet"
            bar_file = bars_dir / f"{date}.parquet"

            tick_file_size = tick_file.stat().st_size if tick_file.exists() else 0
            bar_file_size = bar_file.stat().st_size if bar_file.exists() else 0

            print(f"\n📊 {date} 单日数据:")
            print(f"   tick文件: {tick_file_size / 1024:.2f} KB")
            print(f"   bar文件: {bar_file_size / 1024:.2f} KB")
            print(f"   单日总计: {(tick_file_size + bar_file_size) / 1024:.2f} KB")

            # 3. 读取性能测试
            print(f"\n📊 读取性能测试:")

            # 读取tick数据
            start_time = time.time()
            ticks = self.storage_manager.ticks.load(symbol, date)
            tick_load_time = time.time() - start_time

            print(f"   tick加载: {len(ticks)} 条, 耗时 {tick_load_time*1000:.2f}ms")

            # 读取bar数据
            start_time = time.time()
            bars = self.storage_manager.bar_1min.load(symbol, date)
            bar_load_time = time.time() - start_time

            print(f"   bar加载: {len(bars)} 条, 耗时 {bar_load_time*1000:.2f}ms")

            # 4. 系统资源监控
            print(f"\n📊 系统资源:")
            disk_usage = psutil.disk_usage(str(self.base_path.parent))
            print(f"   磁盘总容量: {disk_usage.total / 1024 / 1024 / 1024:.2f} GB")
            print(
                f"   已使用: {disk_usage.used / 1024 / 1024 / 1024:.2f} GB ({disk_usage.percent}%)"
            )
            print(f"   剩余: {disk_usage.free / 1024 / 1024 / 1024:.2f} GB")

            memory = psutil.virtual_memory()
            print(f"\n   内存总量: {memory.total / 1024 / 1024 / 1024:.2f} GB")
            print(
                f"   已使用: {memory.used / 1024 / 1024 / 1024:.2f} GB ({memory.percent}%)"
            )
            print(f"   可用: {memory.available / 1024 / 1024 / 1024:.2f} GB")

            result["storage_stats"] = {
                "ticks_dir_mb": ticks_size / 1024 / 1024,
                "bars_dir_mb": bars_size / 1024 / 1024,
                "total_mb": (ticks_size + bars_size) / 1024 / 1024,
                "daily_tick_kb": tick_file_size / 1024,
                "daily_bar_kb": bar_file_size / 1024,
                "daily_total_kb": (tick_file_size + bar_file_size) / 1024,
            }

            result["performance_stats"] = {
                "tick_load_ms": tick_load_time * 1000,
                "bar_load_ms": bar_load_time * 1000,
                "tick_records": len(ticks),
                "bar_records": len(bars),
            }

            result["system_stats"] = {
                "disk_total_gb": disk_usage.total / 1024 / 1024 / 1024,
                "disk_used_gb": disk_usage.used / 1024 / 1024 / 1024,
                "disk_usage_percent": disk_usage.percent,
                "memory_total_gb": memory.total / 1024 / 1024 / 1024,
                "memory_used_gb": memory.used / 1024 / 1024 / 1024,
                "memory_usage_percent": memory.percent,
            }

            result["passed"] = True
            print(f"\n✅ 存储空间和性能监控完成！")

        except Exception as e:
            result["errors"].append(f"监控过程异常: {str(e)}")
            print(f"❌ 监控过程异常: {e}")
            import traceback

            traceback.print_exc()

        return result

    def run_all_verifications(
        self,
        symbol: str = "BTCUSDT",
        date: str = None,
    ) -> dict:
        """运行所有验证"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        print(f"\n{'='*80}")
        print(f"🚀 开始验证实盘tick数据存储")
        print(f"{'='*80}")
        print(f"币种: {symbol}")
        print(f"日期: {date}")
        print(f"基础路径: {self.base_path}")

        # 验证1: tick格式
        self.results["tests"]["tick_format"] = self.verify_tick_format(symbol, date)

        # 验证2: VPIN计算
        self.results["tests"]["vpin_computation"] = self.verify_vpin_computation(
            symbol, date
        )

        # 验证3: 存储性能
        self.results["tests"]["storage_performance"] = self.verify_storage_performance(
            symbol, date
        )

        # 汇总结果
        print(f"\n{'='*80}")
        print(f"📋 验证结果汇总")
        print(f"{'='*80}")

        all_passed = all(test["passed"] for test in self.results["tests"].values())

        for test_name, test_result in self.results["tests"].items():
            status = "✅ 通过" if test_result["passed"] else "❌ 失败"
            print(f"{test_name}: {status}")
            if test_result["errors"]:
                for error in test_result["errors"]:
                    print(f"  ❌ {error}")
            if test_result["warnings"]:
                for warning in test_result["warnings"]:
                    print(f"  ⚠️  {warning}")

        print(f"\n{'='*80}")
        if all_passed:
            print(f"🎉 所有验证通过！")
        else:
            print(f"⚠️  部分验证失败，请检查错误信息")
        print(f"{'='*80}")

        return self.results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="实盘tick数据存储验证")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易对符号")
    parser.add_argument("--date", default=None, help="验证日期 (YYYY-MM-DD)")
    parser.add_argument("--base-path", default="live/highcap/data", help="数据基础路径")

    args = parser.parse_args()

    # 创建验证器
    verifier = LiveTickStorageVerifier(base_path=args.base_path)

    # 运行验证
    results = verifier.run_all_verifications(
        symbol=args.symbol,
        date=args.date,
    )

    # 保存结果到JSON
    import json

    result_file = Path(__file__).parent / "verify_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 验证结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
