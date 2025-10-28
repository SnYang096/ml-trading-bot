#!/usr/bin/env python3
"""
批量下载Binance历史交易数据
====================================

功能：
1. 下载BTC、ETH、SOL的aggTrades数据
2. 时间范围：2021年1月 - 2025年9月
3. 自动跳过已存在的文件
4. 支持断点续传
5. 显示下载进度和统计
"""

import os
import sys
import requests
import zipfile
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import argparse


class BinanceMultiSymbolDownloader:
    """Binance多币种历史数据下载器"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Binance数据基础URL
        self.base_url = "https://data.binance.vision/data/futures/um/monthly/aggTrades"

        # 支持的交易对
        self.symbols = {
            "BTCUSDT": "Bitcoin",
            "ETHUSDT": "Ethereum",
            "SOLUSDT": "Solana",
        }

        # 下载统计
        self.stats = {
            "total": 0,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "total_size_mb": 0,
        }

    def get_month_list(
        self,
        start_year: int = 2021,
        start_month: int = 1,
        end_year: int = 2025,
        end_month: int = 9,
    ) -> List[Tuple[int, int]]:
        """生成月份列表"""
        months = []
        current_year = start_year
        current_month = start_month

        while (current_year < end_year) or (
            current_year == end_year and current_month <= end_month
        ):
            months.append((current_year, current_month))
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        return months

    def check_local_file(self, symbol: str, year: int, month: int) -> bool:
        """检查本地文件是否存在且完整"""
        filename = f"{symbol}-aggTrades-{year}-{month:02d}.zip"
        file_path = self.data_dir / filename

        if not file_path.exists():
            return False

        # 检查文件大小（至少应该有1MB，避免下载不完整的文件）
        file_size = file_path.stat().st_size
        if file_size < 1 * 1024 * 1024:  # 1MB
            print(f"   ⚠️  {filename} 文件太小 ({file_size} bytes)，将重新下载")
            file_path.unlink()  # 删除不完整的文件
            return False

        return True

    def download_file(
        self,
        symbol: str,
        year: int,
        month: int,
        retry_times: int = 3,
        timeout: int = 600,
    ) -> bool:
        """下载单个文件"""
        filename = f"{symbol}-aggTrades-{year}-{month:02d}.zip"
        file_path = self.data_dir / filename
        url = f"{self.base_url}/{symbol}/{filename}"

        for attempt in range(retry_times):
            try:
                print(
                    f"   📥 下载 {filename} (尝试 {attempt + 1}/{retry_times})...",
                    end=" ",
                )

                # 发送请求
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()

                # 获取文件大小
                total_size = int(response.headers.get("content-length", 0))

                # 下载文件
                downloaded_size = 0
                chunk_size = 8192

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                # 验证下载
                final_size = file_path.stat().st_size
                size_mb = final_size / 1024 / 1024

                if total_size > 0 and final_size != total_size:
                    raise Exception(f"文件大小不匹配: {final_size} != {total_size}")

                print(f"✅ ({size_mb:.1f}MB)")
                self.stats["downloaded"] += 1
                self.stats["total_size_mb"] += size_mb
                return True

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"❌ 文件不存在 (404)")
                    return False  # 不重试404错误
                else:
                    print(f"❌ HTTP错误 {e.response.status_code}")

            except Exception as e:
                print(f"❌ {str(e)}")

            # 删除不完整的文件
            if file_path.exists():
                file_path.unlink()

            # 等待后重试
            if attempt < retry_times - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间
                print(f"   ⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

        self.stats["failed"] += 1
        return False

    def download_symbol_data(self, symbol: str, months: List[Tuple[int, int]]) -> Dict:
        """下载单个币种的所有数据"""
        symbol_name = self.symbols.get(symbol, symbol)
        print(f"\n{'='*60}")
        print(f"📊 {symbol_name} ({symbol})")
        print(f"{'='*60}")

        symbol_stats = {
            "total": len(months),
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
        }

        for i, (year, month) in enumerate(months, 1):
            month_str = f"{year}-{month:02d}"
            print(f"[{i}/{len(months)}] {month_str}:", end=" ")

            self.stats["total"] += 1

            # 检查本地是否已有文件
            if self.check_local_file(symbol, year, month):
                filename = f"{symbol}-aggTrades-{year}-{month:02d}.zip"
                file_path = self.data_dir / filename
                size_mb = file_path.stat().st_size / 1024 / 1024
                print(f"✅ 已存在 ({size_mb:.1f}MB)")
                symbol_stats["skipped"] += 1
                self.stats["skipped"] += 1
                continue

            # 下载文件
            if self.download_file(symbol, year, month):
                symbol_stats["downloaded"] += 1
            else:
                symbol_stats["failed"] += 1

            # 避免请求过快
            time.sleep(0.5)

        return symbol_stats

    def download_all(
        self,
        start_year: int = 2021,
        start_month: int = 1,
        end_year: int = 2025,
        end_month: int = 9,
        symbols: List[str] = None,
    ) -> None:
        """下载所有币种的数据"""
        print("\n" + "=" * 60)
        print("🚀 Binance 历史交易数据批量下载器")
        print("=" * 60)
        print(
            f"📅 时间范围: {start_year}-{start_month:02d} 至 {end_year}-{end_month:02d}"
        )
        print(f"📂 保存目录: {self.data_dir.absolute()}")

        # 确定要下载的币种
        if symbols is None:
            symbols = list(self.symbols.keys())
        else:
            # 验证币种
            symbols = [s for s in symbols if s in self.symbols]

        print(f"💰 币种: {', '.join(symbols)}")

        # 生成月份列表
        months = self.get_month_list(start_year, start_month, end_year, end_month)
        total_files = len(months) * len(symbols)

        print(f"📦 预计文件数: {total_files} ({len(months)} 月 × {len(symbols)} 币种)")
        print()

        # 确认开始
        response = input("⚠️  这可能需要几小时时间和大量磁盘空间。是否继续? (y/N): ")
        if response.lower() != "y":
            print("❌ 已取消")
            return

        start_time = time.time()

        # 逐个币种下载
        for symbol in symbols:
            symbol_stats = self.download_symbol_data(symbol, months)
            print(f"\n{symbol} 统计:")
            print(f"  ✅ 下载: {symbol_stats['downloaded']}")
            print(f"  ⏭️  跳过: {symbol_stats['skipped']}")
            print(f"  ❌ 失败: {symbol_stats['failed']}")

        # 总体统计
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60

        print("\n" + "=" * 60)
        print("📊 下载统计")
        print("=" * 60)
        print(f"总文件数: {self.stats['total']}")
        print(f"✅ 新下载: {self.stats['downloaded']}")
        print(f"⏭️  已跳过: {self.stats['skipped']}")
        print(f"❌ 失败: {self.stats['failed']}")
        print(f"📦 总大小: {self.stats['total_size_mb']:.1f} MB")
        print(f"⏱️  总耗时: {elapsed_minutes:.1f} 分钟")

        if self.stats["downloaded"] > 0:
            avg_speed = (
                self.stats["total_size_mb"] / elapsed_minutes
                if elapsed_minutes > 0
                else 0
            )
            print(f"⚡ 平均速度: {avg_speed:.1f} MB/分钟")

        print("\n✅ 下载完成！")
        print(f"📂 数据保存在: {self.data_dir.absolute()}")

    def list_downloaded_files(self) -> Dict[str, List[str]]:
        """列出已下载的文件"""
        downloaded = {symbol: [] for symbol in self.symbols.keys()}

        for file_path in self.data_dir.glob("*-aggTrades-*.zip"):
            filename = file_path.name
            for symbol in self.symbols.keys():
                if filename.startswith(symbol):
                    # 提取日期
                    date_part = filename.replace(f"{symbol}-aggTrades-", "").replace(
                        ".zip", ""
                    )
                    downloaded[symbol].append(date_part)

        return {k: sorted(v) for k, v in downloaded.items()}

    def print_summary(self) -> None:
        """打印下载摘要"""
        print("\n" + "=" * 60)
        print("📁 本地数据摘要")
        print("=" * 60)

        downloaded = self.list_downloaded_files()

        for symbol, dates in downloaded.items():
            symbol_name = self.symbols[symbol]
            print(f"\n{symbol_name} ({symbol}): {len(dates)} 个月")

            if dates:
                print(f"  最早: {dates[0]}")
                print(f"  最新: {dates[-1]}")

                # 计算总大小
                total_size = 0
                for date in dates:
                    filename = f"{symbol}-aggTrades-{date}.zip"
                    file_path = self.data_dir / filename
                    if file_path.exists():
                        total_size += file_path.stat().st_size

                print(f"  大小: {total_size / 1024 / 1024:.1f} MB")

                # 检查缺失的月份
                if len(dates) > 1:
                    # 简单检查：应该有连续的月份
                    expected_count = self._count_months_between(dates[0], dates[-1])
                    if expected_count > len(dates):
                        missing = expected_count - len(dates)
                        print(f"  ⚠️  可能缺失 {missing} 个月的数据")

    def _count_months_between(self, start_date: str, end_date: str) -> int:
        """计算两个日期之间的月份数"""
        start_year, start_month = map(int, start_date.split("-"))
        end_year, end_month = map(int, end_date.split("-"))
        return (end_year - start_year) * 12 + (end_month - start_month) + 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载Binance历史交易数据 (2021-2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载所有币种的全部数据
  python download_training_data.py
  
  # 只下载BTC数据
  python download_training_data.py --symbols BTCUSDT
  
  # 下载BTC和ETH
  python download_training_data.py --symbols BTCUSDT ETHUSDT
  
  # 指定时间范围
  python download_training_data.py --start-year 2024 --start-month 1
  
  # 查看已下载的文件
  python download_training_data.py --summary
        """,
    )

    parser.add_argument(
        "--data-dir", default="data/raw", help="数据保存目录 (默认: data/raw)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        choices=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="指定要下载的币种",
    )
    parser.add_argument(
        "--start-year", type=int, default=2021, help="开始年份 (默认: 2021)"
    )
    parser.add_argument("--start-month", type=int, default=1, help="开始月份 (默认: 1)")
    parser.add_argument(
        "--end-year", type=int, default=2025, help="结束年份 (默认: 2025)"
    )
    parser.add_argument("--end-month", type=int, default=9, help="结束月份 (默认: 9)")
    parser.add_argument("--summary", action="store_true", help="只显示已下载文件的摘要")

    args = parser.parse_args()

    # 创建下载器
    downloader = BinanceMultiSymbolDownloader(args.data_dir)

    if args.summary:
        # 只显示摘要
        downloader.print_summary()
    else:
        # 执行下载
        downloader.download_all(
            start_year=args.start_year,
            start_month=args.start_month,
            end_year=args.end_year,
            end_month=args.end_month,
            symbols=args.symbols,
        )

        # 显示最终摘要
        downloader.print_summary()


if __name__ == "__main__":
    main()
