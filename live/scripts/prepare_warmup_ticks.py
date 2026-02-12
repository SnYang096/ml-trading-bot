#!/usr/bin/env python3
"""
准备实盘 warmup ticks 数据
=========================

功能：
1. 下载最近 N 个月的 monthly aggTrades（Binance UM 期货）
2. 下载最近 1 个月的 daily aggTrades（补齐到昨天）
3. 转换为 1min 聚合 ticks（格式: [timestamp, price, volume, side]）
4. 按日期拆分写入 live/{universe}/data/ticks/{SYMBOL}/{YYYY-MM-DD}.parquet
5. 同时生成 1min OHLCV bars 写入 live/{universe}/data/bars/{SYMBOL}/{YYYY-MM-DD}.parquet

用法：
    python live/scripts/prepare_warmup_ticks.py --universe highcap --months 6
    python live/scripts/prepare_warmup_ticks.py --universe highcap --months 6 --symbols BTCUSDT,ETHUSDT
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.download_training_data import BinanceMultiSymbolDownloader
from src.data_tools.zip_to_parquet import DataConverter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_universe_symbols(universe: str) -> List[str]:
    """从 live/{universe}/universe.yaml 读取 symbols 列表"""
    import yaml

    universe_path = PROJECT_ROOT / "live" / universe / "universe.yaml"
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")
    with open(universe_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return sorted((cfg.get("symbols") or {}).keys())


def ticks_to_bars(ticks_df: pd.DataFrame) -> pd.DataFrame:
    """将 1min 聚合 ticks（买卖分离）转为 1min OHLCV bars

    输入格式: [timestamp, price, volume, side]  (每分钟2条: buy + sell)
    输出格式: [timestamp, open, high, low, close, volume, buy_volume, sell_volume]
    """
    if ticks_df.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume",
                      "buy_volume", "sell_volume"]
        )

    df = ticks_df.copy()
    df["buy_volume"] = np.where(df["side"] == 1, df["volume"], 0.0)
    df["sell_volume"] = np.where(df["side"] == -1, df["volume"], 0.0)

    # 按 timestamp 聚合（同一分钟的 buy/sell 合并为一条 bar）
    agg = df.groupby("timestamp").agg(
        price_first=("price", "first"),
        price_max=("price", "max"),
        price_min=("price", "min"),
        price_last=("price", "last"),
        volume=("volume", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
    ).reset_index()

    agg = agg.rename(columns={
        "price_first": "open",
        "price_max": "high",
        "price_min": "low",
        "price_last": "close",
    })
    agg = agg.sort_values("timestamp").reset_index(drop=True)
    return agg[["timestamp", "open", "high", "low", "close", "volume",
                 "buy_volume", "sell_volume"]]


def split_by_date(df: pd.DataFrame, ts_col: str = "timestamp") -> Dict[str, pd.DataFrame]:
    """将 DataFrame 按日期拆分为 {YYYY-MM-DD: sub_df}"""
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    df["_date"] = ts.dt.strftime("%Y-%m-%d")
    result = {}
    for date_str, group in df.groupby("_date"):
        sub = group.drop(columns=["_date"]).reset_index(drop=True)
        result[date_str] = sub
    return result


def write_daily_parquets(
    daily_data: Dict[str, pd.DataFrame],
    output_dir: Path,
    symbol: str,
) -> int:
    """将按日期拆分的数据写入 output_dir/{symbol}/{YYYY-MM-DD}.parquet"""
    symbol_dir = output_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for date_str, df in sorted(daily_data.items()):
        if df.empty:
            continue
        out_path = symbol_dir / f"{date_str}.parquet"
        df.to_parquet(out_path, index=False)
        written += 1

    return written


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------

def compute_date_ranges(months: int):
    """计算 monthly 和 daily 的日期范围

    Returns:
        monthly_start_year, monthly_start_month, monthly_end_year, monthly_end_month,
        daily_start_date, daily_end_date
    """
    today = datetime.utcnow()

    # Monthly: 最近 N 个月（到上上个月，因为上个月的 monthly 可能还没发布）
    # Binance monthly 数据通常次月 10 号左右发布
    # 安全起见：monthly 到 (当前月 - 2)
    monthly_end = today.replace(day=1) - timedelta(days=1)  # 上个月最后一天
    # 如果当前日期 < 15，上个月的 monthly 可能还没发布，回退一个月
    if today.day < 15:
        monthly_end = (monthly_end.replace(day=1) - timedelta(days=1))

    monthly_end_year = monthly_end.year
    monthly_end_month = monthly_end.month

    # Monthly start: 往前推 N 个月
    monthly_start = monthly_end.replace(day=1)
    for _ in range(months - 1):
        monthly_start = (monthly_start - timedelta(days=1)).replace(day=1)
    monthly_start_year = monthly_start.year
    monthly_start_month = monthly_start.month

    # Daily: 从 monthly_end 的下一个月 1 号 到 昨天
    daily_start = (monthly_end + timedelta(days=1)).strftime("%Y-%m-%d")
    daily_end = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    return (
        monthly_start_year, monthly_start_month,
        monthly_end_year, monthly_end_month,
        daily_start, daily_end,
    )


def download_monthly(
    symbols: List[str],
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    data_dir: Path,
) -> None:
    """下载 monthly aggTrades ZIPs"""
    print(f"\n{'='*60}")
    print(f"📥 下载 monthly aggTrades: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d}")
    print(f"{'='*60}")

    downloader = BinanceMultiSymbolDownloader(
        data_dir=str(data_dir),
        granularity="monthly",
    )
    downloader.download_all(
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        symbols=symbols,
    )


def download_daily(
    symbols: List[str],
    start_date: str,
    end_date: str,
    data_dir: Path,
) -> None:
    """下载 daily aggTrades ZIPs"""
    print(f"\n{'='*60}")
    print(f"📥 下载 daily aggTrades: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    downloader = BinanceMultiSymbolDownloader(
        data_dir=str(data_dir),
        granularity="daily",
    )
    dates = downloader.get_date_list(start_date, end_date)
    print(f"   共 {len(dates)} 天, symbols: {symbols}")

    for symbol in symbols:
        print(f"\n   📦 {symbol}:")
        for date_str in dates:
            parts = date_str.split("-")
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])

            existing = downloader.check_local_file(symbol, year, month, day)
            if existing:
                continue

            success = downloader.download_file(symbol, year, month, day)
            if not success:
                print(f"      ⚠️ 下载失败: {symbol} {date_str}")


def convert_and_split(
    zip_dir: Path,
    symbols: List[str],
    ticks_output_dir: Path,
    bars_output_dir: Path,
) -> Dict[str, int]:
    """转换 ZIP → 1min 聚合 → 按日期拆分 → 写入 live 目录

    Returns:
        每个 symbol 的写入天数统计
    """
    print(f"\n{'='*60}")
    print("🔄 转换 ZIP → 1min 聚合 ticks/bars")
    print(f"{'='*60}")

    # 先用 DataConverter 将所有 ZIP 转为 1min 聚合 parquet
    tmp_parquet_dir = zip_dir / "_parquet_tmp"
    tmp_parquet_dir.mkdir(exist_ok=True)

    converter = DataConverter(
        input_dir=str(zip_dir),
        output_dir=str(tmp_parquet_dir),
        aggregate_freq="1min",
    )
    result = converter.convert_all_files(symbols=symbols)
    print(f"   转换完成: {len(result.get('converted_files', []))} 个文件")

    # 读取所有 parquet，按 symbol 合并，按日期拆分，写入 live 目录
    stats = {}
    for symbol in symbols:
        print(f"\n   📊 处理 {symbol}...")
        pattern = f"{symbol}_*.parquet"
        parquet_files = sorted(tmp_parquet_dir.glob(pattern))

        if not parquet_files:
            print(f"      ⚠️ 未找到 {symbol} 的 parquet 文件")
            stats[symbol] = 0
            continue

        # 合并所有月/日的 parquet
        dfs = []
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"      ⚠️ 读取失败 {pf.name}: {e}")

        if not dfs:
            stats[symbol] = 0
            continue

        all_ticks = pd.concat(dfs, ignore_index=True)
        all_ticks["timestamp"] = pd.to_datetime(all_ticks["timestamp"])
        all_ticks = all_ticks.sort_values("timestamp").reset_index(drop=True)

        # 去重
        all_ticks = all_ticks.drop_duplicates(
            subset=["timestamp", "side"], keep="last"
        )

        print(f"      总记录: {len(all_ticks):,} 条")
        print(f"      时间范围: {all_ticks['timestamp'].min()} ~ {all_ticks['timestamp'].max()}")

        # 写 ticks（按日期拆分）
        daily_ticks = split_by_date(all_ticks)
        n_tick_days = write_daily_parquets(daily_ticks, ticks_output_dir, symbol)

        # 生成 bars 并写入
        bars = ticks_to_bars(all_ticks)
        daily_bars = split_by_date(bars)
        n_bar_days = write_daily_parquets(daily_bars, bars_output_dir, symbol)

        stats[symbol] = n_tick_days
        print(f"      ✅ 写入 {n_tick_days} 天 ticks, {n_bar_days} 天 bars")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="准备实盘 warmup ticks 数据（下载 + 转换 + 拆分到 live 目录）"
    )
    parser.add_argument(
        "--universe", default="highcap",
        help="Universe 名称（默认: highcap）",
    )
    parser.add_argument(
        "--months", type=int, default=6,
        help="下载最近 N 个月的 monthly aggTrades（默认: 6）",
    )
    parser.add_argument(
        "--symbols", default=None,
        help="逗号分隔的 symbols（默认从 universe.yaml 读取）",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="跳过下载步骤（仅转换已有的 ZIP 文件）",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="ZIP 文件目录（默认: data/warmup_raw/{universe}）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 解析 symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_universe_symbols(args.universe)

    # 输出目录
    live_root = PROJECT_ROOT / "live" / args.universe
    ticks_dir = live_root / "data" / "ticks"
    bars_dir = live_root / "data" / "bars"
    ticks_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)

    # ZIP 下载目录（独立于 data/agg_data，避免混淆研究数据）
    if args.data_dir:
        zip_dir = Path(args.data_dir)
    else:
        zip_dir = PROJECT_ROOT / "data" / "warmup_raw" / args.universe
    zip_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 准备实盘 warmup ticks 数据")
    print("=" * 60)
    print(f"   Universe:   {args.universe}")
    print(f"   Symbols:    {', '.join(symbols)}")
    print(f"   Months:     {args.months}")
    print(f"   ZIP dir:    {zip_dir}")
    print(f"   Ticks dir:  {ticks_dir}")
    print(f"   Bars dir:   {bars_dir}")

    # 计算日期范围
    (
        m_start_y, m_start_m,
        m_end_y, m_end_m,
        d_start, d_end,
    ) = compute_date_ranges(args.months)

    print(f"\n   Monthly 范围: {m_start_y}-{m_start_m:02d} ~ {m_end_y}-{m_end_m:02d}")
    print(f"   Daily 范围:   {d_start} ~ {d_end}")

    # Step 1: 下载
    if not args.skip_download:
        download_monthly(symbols, m_start_y, m_start_m, m_end_y, m_end_m, zip_dir)
        download_daily(symbols, d_start, d_end, zip_dir)
    else:
        print("\n⏭️  跳过下载步骤")

    # Step 2: 转换 + 拆分 + 写入
    stats = convert_and_split(zip_dir, symbols, ticks_dir, bars_dir)

    # 汇总
    print(f"\n{'='*60}")
    print("✅ 完成！")
    print(f"{'='*60}")
    total_days = sum(stats.values())
    for sym, n_days in sorted(stats.items()):
        print(f"   {sym}: {n_days} 天")
    print(f"   共 {total_days} 天 × {len(symbols)} 币种")
    print(f"\n   Ticks 输出: {ticks_dir}")
    print(f"   Bars 输出:  {bars_dir}")
    print(f"\n   下一步: bash live/scripts/start_live.sh {args.universe}")


if __name__ == "__main__":
    main()
