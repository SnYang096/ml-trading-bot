#!/usr/bin/env python3
"""
实盘tick数据存储实时监控脚本

监控内容：
1. 实时监控tick数据写入频率
2. 监控存储空间使用情况
3. 监控系统资源占用
4. 异常告警
"""
import sys
import time
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from live_data_stream.feature_storage import StorageManager


class LiveTickStorageMonitor:
    """实盘tick存储实时监控器"""

    def __init__(
        self,
        base_path: str = "live/highcap/data",
        check_interval: int = 60,  # 检查间隔（秒）
        history_size: int = 60,  # 历史数据保留（个数）
    ):
        self.base_path = Path(base_path)
        self.storage_manager = StorageManager(base_path=self.base_path)
        self.check_interval = check_interval
        self.history_size = history_size

        # 历史数据（用于趋势分析）
        self.tick_counts_history = deque(maxlen=history_size)
        self.storage_size_history = deque(maxlen=history_size)
        self.memory_usage_history = deque(maxlen=history_size)

        # 上一次检查的状态
        self.last_check_time = None
        self.last_tick_count = {}

        print(f"🚀 实盘tick存储监控器已启动")
        print(f"   基础路径: {self.base_path}")
        print(f"   检查间隔: {self.check_interval}秒")
        print(f"   历史保留: {self.history_size}个数据点")

    def get_current_tick_count(self, symbol: str) -> int:
        """获取当前tick数据总数"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            ticks = self.storage_manager.ticks.load(symbol, today)
            return len(ticks)
        except Exception as e:
            print(f"⚠️  获取tick计数失败: {e}")
            return 0

    def get_storage_size(self) -> dict:
        """获取存储空间使用情况"""

        def get_dir_size(path: Path) -> int:
            """获取目录大小（字节）"""
            if not path.exists():
                return 0
            total = 0
            for file in path.rglob("*.parquet"):
                total += file.stat().st_size
            return total

        ticks_dir = self.base_path / "ticks"
        bars_dir = self.base_path / "bars"

        return {
            "ticks_mb": get_dir_size(ticks_dir) / 1024 / 1024,
            "bars_mb": get_dir_size(bars_dir) / 1024 / 1024,
            "total_mb": (get_dir_size(ticks_dir) + get_dir_size(bars_dir))
            / 1024
            / 1024,
        }

    def get_system_resources(self) -> dict:
        """获取系统资源使用情况"""
        disk = psutil.disk_usage(str(self.base_path.parent))
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)

        return {
            "disk_used_gb": disk.used / 1024 / 1024 / 1024,
            "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            "disk_percent": disk.percent,
            "memory_used_gb": memory.used / 1024 / 1024 / 1024,
            "memory_available_gb": memory.available / 1024 / 1024 / 1024,
            "memory_percent": memory.percent,
            "cpu_percent": cpu,
        }

    def check_and_report(self, symbol: str = "BTCUSDT") -> dict:
        """执行一次检查并报告"""
        now = datetime.now()

        # 1. 获取tick计数
        tick_count = self.get_current_tick_count(symbol)

        # 2. 计算写入速率（如果有历史数据）
        write_rate = None
        if self.last_check_time and symbol in self.last_tick_count:
            time_diff = (now - self.last_check_time).total_seconds()
            count_diff = tick_count - self.last_tick_count[symbol]
            if time_diff > 0:
                write_rate = count_diff / time_diff  # 条/秒

        # 3. 获取存储空间
        storage_size = self.get_storage_size()

        # 4. 获取系统资源
        system_resources = self.get_system_resources()

        # 5. 更新历史数据
        self.tick_counts_history.append(
            {
                "timestamp": now,
                "symbol": symbol,
                "count": tick_count,
                "write_rate": write_rate,
            }
        )
        self.storage_size_history.append(
            {
                "timestamp": now,
                **storage_size,
            }
        )
        self.memory_usage_history.append(
            {
                "timestamp": now,
                **system_resources,
            }
        )

        # 6. 更新状态
        self.last_check_time = now
        self.last_tick_count[symbol] = tick_count

        # 7. 生成报告
        report = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "tick_count": tick_count,
            "write_rate_per_sec": write_rate,
            "storage": storage_size,
            "system": system_resources,
        }

        return report

    def print_report(self, report: dict):
        """打印报告"""
        print(f"\n{'='*80}")
        print(f"📊 实盘tick存储监控报告 - {report['timestamp']}")
        print(f"{'='*80}")

        # Tick数据
        print(f"\n📦 Tick数据 ({report['symbol']}):")
        print(f"   当前总数: {report['tick_count']:,} 条")
        if report["write_rate_per_sec"] is not None:
            print(f"   写入速率: {report['write_rate_per_sec']:.2f} 条/秒")
            print(f"             {report['write_rate_per_sec']*60:.1f} 条/分钟")

        # 存储空间
        print(f"\n💾 存储空间:")
        print(f"   ticks: {report['storage']['ticks_mb']:.2f} MB")
        print(f"   bars: {report['storage']['bars_mb']:.2f} MB")
        print(f"   总计: {report['storage']['total_mb']:.2f} MB")

        # 系统资源
        print(f"\n🖥️  系统资源:")
        print(
            f"   磁盘: {report['system']['disk_used_gb']:.2f} GB / {report['system']['disk_free_gb']:.2f} GB 可用 ({report['system']['disk_percent']:.1f}%)"
        )
        print(
            f"   内存: {report['system']['memory_used_gb']:.2f} GB / {report['system']['memory_available_gb']:.2f} GB 可用 ({report['system']['memory_percent']:.1f}%)"
        )
        print(f"   CPU: {report['system']['cpu_percent']:.1f}%")

        # 告警检查
        warnings = []

        # 检查写入速率（期望每分钟~2条）
        if report["write_rate_per_sec"] is not None:
            expected_rate = 2.0 / 60  # 2条/分钟 = 0.033条/秒
            if report["write_rate_per_sec"] < expected_rate * 0.5:
                warnings.append(
                    f"⚠️  写入速率过低: {report['write_rate_per_sec']*60:.1f} 条/分钟（期望~2条/分钟）"
                )

        # 检查磁盘空间
        if report["system"]["disk_percent"] > 90:
            warnings.append(f"⚠️  磁盘空间不足: {report['system']['disk_percent']:.1f}%")

        # 检查内存使用
        if report["system"]["memory_percent"] > 85:
            warnings.append(
                f"⚠️  内存使用过高: {report['system']['memory_percent']:.1f}%"
            )

        if warnings:
            print(f"\n⚠️  告警:")
            for warning in warnings:
                print(f"   {warning}")
        else:
            print(f"\n✅ 系统运行正常")

    def generate_trend_report(self):
        """生成趋势报告"""
        if len(self.tick_counts_history) < 2:
            print("\n⚠️  历史数据不足，无法生成趋势报告")
            return

        print(f"\n{'='*80}")
        print(f"📈 趋势分析（最近{len(self.tick_counts_history)}次检查）")
        print(f"{'='*80}")

        # 1. tick写入趋势
        write_rates = [
            h["write_rate"]
            for h in self.tick_counts_history
            if h["write_rate"] is not None
        ]
        if write_rates:
            avg_rate = sum(write_rates) / len(write_rates)
            print(f"\n📦 Tick写入速率:")
            print(f"   平均: {avg_rate*60:.2f} 条/分钟")
            print(f"   最小: {min(write_rates)*60:.2f} 条/分钟")
            print(f"   最大: {max(write_rates)*60:.2f} 条/分钟")

        # 2. 存储空间增长趋势
        if len(self.storage_size_history) >= 2:
            first = self.storage_size_history[0]
            last = self.storage_size_history[-1]
            time_diff_hours = (
                last["timestamp"] - first["timestamp"]
            ).total_seconds() / 3600

            if time_diff_hours > 0:
                ticks_growth = last["ticks_mb"] - first["ticks_mb"]
                bars_growth = last["bars_mb"] - first["bars_mb"]

                print(f"\n💾 存储空间增长（{time_diff_hours:.1f}小时）:")
                print(
                    f"   ticks: +{ticks_growth:.2f} MB ({ticks_growth/time_diff_hours:.2f} MB/小时)"
                )
                print(
                    f"   bars: +{bars_growth:.2f} MB ({bars_growth/time_diff_hours:.2f} MB/小时)"
                )

                # 预测一个月的存储需求
                ticks_monthly = ticks_growth / time_diff_hours * 24 * 30
                bars_monthly = bars_growth / time_diff_hours * 24 * 30
                print(f"\n📊 预测月度存储需求:")
                print(f"   ticks: {ticks_monthly:.2f} MB/月")
                print(f"   bars: {bars_monthly:.2f} MB/月")
                print(f"   总计: {ticks_monthly + bars_monthly:.2f} MB/月")

        # 3. 内存使用趋势
        if len(self.memory_usage_history) >= 2:
            memory_percents = [h["memory_percent"] for h in self.memory_usage_history]
            print(f"\n🖥️  内存使用:")
            print(f"   平均: {sum(memory_percents)/len(memory_percents):.1f}%")
            print(f"   最小: {min(memory_percents):.1f}%")
            print(f"   最大: {max(memory_percents):.1f}%")

    def run(self, symbol: str = "BTCUSDT", duration_minutes: int = 60):
        """运行监控（指定时长）"""
        print(f"\n🚀 开始监控（运行{duration_minutes}分钟）...")

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        check_count = 0

        try:
            while datetime.now() < end_time:
                # 执行检查
                report = self.check_and_report(symbol)
                self.print_report(report)

                check_count += 1

                # 每10次检查生成一次趋势报告
                if check_count % 10 == 0:
                    self.generate_trend_report()

                # 等待下一次检查
                print(f"\n⏳ 等待{self.check_interval}秒后进行下一次检查...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  监控已手动停止")

        # 生成最终报告
        print(f"\n{'='*80}")
        print(f"📋 监控完成")
        print(f"{'='*80}")
        print(
            f"   运行时长: {(datetime.now() - start_time).total_seconds() / 60:.1f} 分钟"
        )
        print(f"   检查次数: {check_count}")

        if check_count > 0:
            self.generate_trend_report()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="实盘tick数据存储实时监控")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易对符号")
    parser.add_argument("--duration", type=int, default=60, help="监控时长（分钟）")
    parser.add_argument("--interval", type=int, default=60, help="检查间隔（秒）")
    parser.add_argument("--base-path", default="live/highcap/data", help="数据基础路径")

    args = parser.parse_args()

    # 创建监控器
    monitor = LiveTickStorageMonitor(
        base_path=args.base_path,
        check_interval=args.interval,
    )

    # 运行监控
    monitor.run(
        symbol=args.symbol,
        duration_minutes=args.duration,
    )


if __name__ == "__main__":
    main()
