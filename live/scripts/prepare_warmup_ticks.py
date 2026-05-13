#!/usr/bin/env python3
"""
准备实盘 warmup ticks 数据
=========================

功能（默认）：
1. 在所需 warmup 日历窗口内下载 Binance UM **daily** aggTrades（Vision:
   ``data/futures/um/daily/aggTrades/``）。按日 ZIP 解压压力远小于单月整包。
2. 可选 ``--monthly-zip``：沿用旧路径（按月 ZIP + 其后一段 daily），HTTP 更少但大月
   ZIP 峰值内存更高。
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


def _iter_symbol_dates(ticks_dir: Path, symbol: str) -> List[str]:
    sym_dir = ticks_dir / symbol
    if not sym_dir.exists():
        return []
    return sorted(f.stem for f in sym_dir.glob("*.parquet") if f.stat().st_size > 0)


def required_warmup_bounds(months: int) -> tuple[str, str]:
    """Return the contiguous calendar bounds for warmup (first day … Vision daily_end)."""
    (
        monthly_start_year,
        monthly_start_month,
        _monthly_end_year,
        _monthly_end_month,
        _daily_start_date,
        daily_end_date,
    ) = compute_date_ranges(months)
    return f"{monthly_start_year}-{monthly_start_month:02d}-01", daily_end_date


def has_required_warmup_coverage(
    ticks_dir: Path,
    symbols: List[str],
    months: int,
) -> bool:
    """Check that each symbol has the full date span expected by warmup prepare."""
    required_start, required_end = required_warmup_bounds(months)
    start_dt = datetime.strptime(required_start, "%Y-%m-%d")
    end_dt = datetime.strptime(required_end, "%Y-%m-%d")
    expected_dates = set()
    current = start_dt
    while current <= end_dt:
        expected_dates.add(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    for symbol in symbols:
        dates = set(_iter_symbol_dates(ticks_dir, symbol))
        if not dates:
            return False
        if not expected_dates.issubset(dates):
            return False
    return True


def ticks_to_bars(ticks_df: pd.DataFrame) -> pd.DataFrame:
    """将 1min 聚合 ticks（买卖分离）转为 1min OHLCV bars

    输入格式: [timestamp, price, volume, side]  (每分钟2条: buy + sell)
    输出格式: [timestamp, open, high, low, close, volume, buy_volume, sell_volume]
    """
    if ticks_df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "buy_volume",
                "sell_volume",
            ]
        )

    df = ticks_df.copy()
    df["buy_volume"] = np.where(df["side"] == 1, df["volume"], 0.0)
    df["sell_volume"] = np.where(df["side"] == -1, df["volume"], 0.0)

    # 按 timestamp 聚合（同一分钟的 buy/sell 合并为一条 bar）
    agg = (
        df.groupby("timestamp")
        .agg(
            price_first=("price", "first"),
            price_max=("price", "max"),
            price_min=("price", "min"),
            price_last=("price", "last"),
            volume=("volume", "sum"),
            buy_volume=("buy_volume", "sum"),
            sell_volume=("sell_volume", "sum"),
        )
        .reset_index()
    )

    agg = agg.rename(
        columns={
            "price_first": "open",
            "price_max": "high",
            "price_min": "low",
            "price_last": "close",
        }
    )
    agg = agg.sort_values("timestamp").reset_index(drop=True)
    return agg[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "buy_volume",
            "sell_volume",
        ]
    ]


def split_by_date(
    df: pd.DataFrame, ts_col: str = "timestamp"
) -> Dict[str, pd.DataFrame]:
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
# Local parquet_data source (--from-local)
# ---------------------------------------------------------------------------


def load_from_local_parquet(
    symbols: List[str],
    months: int,
    ticks_output_dir: Path,
    bars_output_dir: Path,
    parquet_data_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """从 data/parquet_data/ 读取已有 1min 聚合数据，按日期拆分写入 live 目录

    data/parquet_data/ 格式: {SYMBOL}_{YYYY-MM}.parquet
    列: [timestamp, price, volume, side, symbol]
    """
    if parquet_data_dir is None:
        parquet_data_dir = PROJECT_ROOT / "data" / "parquet_data"

    if not parquet_data_dir.exists():
        raise FileNotFoundError(f"parquet_data 目录不存在: {parquet_data_dir}")

    # 计算需要的月份列表
    today = datetime.utcnow()
    month_keys: List[str] = []
    cursor = today.replace(day=1)  # 当前月
    for _ in range(months + 1):  # +1 包含当前月（可能有部分数据）
        month_keys.append(cursor.strftime("%Y-%m"))
        cursor = (cursor - timedelta(days=1)).replace(day=1)
    month_keys.reverse()

    print(f"\n{'='*60}")
    print("📂 从本地 parquet_data 加载 1min 聚合数据")
    print(f"{'='*60}")
    print(f"   源目录: {parquet_data_dir}")
    print(f"   月份范围: {month_keys[0]} ~ {month_keys[-1]}")

    stats = {}
    for symbol in symbols:
        print(f"\n   📊 处理 {symbol}...")
        dfs = []
        for mk in month_keys:
            pf = parquet_data_dir / f"{symbol}_{mk}.parquet"
            if pf.exists():
                try:
                    df = pd.read_parquet(pf)
                    if len(df) > 0:
                        dfs.append(df)
                        print(f"      ✅ {pf.name}: {len(df):,} 条")
                except Exception as e:
                    print(f"      ⚠️ 读取失败 {pf.name}: {e}")
            else:
                print(f"      ⏭️  {pf.name} 不存在，跳过")

        if not dfs:
            stats[symbol] = 0
            print(f"      ❌ 无可用数据")
            continue

        all_ticks = pd.concat(dfs, ignore_index=True)

        # 标准化：去掉 symbol 列，加 UTC 时区
        if "symbol" in all_ticks.columns:
            all_ticks = all_ticks.drop(columns=["symbol"])
        all_ticks["timestamp"] = pd.to_datetime(all_ticks["timestamp"], utc=True)
        all_ticks = all_ticks.sort_values("timestamp").reset_index(drop=True)

        # 去重
        all_ticks = all_ticks.drop_duplicates(subset=["timestamp", "side"], keep="last")

        print(f"      总记录: {len(all_ticks):,} 条")
        print(
            f"      时间范围: {all_ticks['timestamp'].min()} ~ {all_ticks['timestamp'].max()}"
        )

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
# Core Pipeline (download mode)
# ---------------------------------------------------------------------------


def compute_date_ranges(months: int):
    """Warmup 用的月份边界与 trailing daily 段的日期字符串。

    用于 ``required_warmup_bounds``（默认整段改成 daily ZIP）以及
    ``--monthly-zip``（按月下载 + trailing daily）。

    Daily 上限为 UTC「前天」，以减少 Binance Vision 日线 ZIP 尚未发布的 404。

    Returns:
        monthly_start_year, monthly_start_month, monthly_end_year, monthly_end_month,
        daily_start_date, daily_end_date
    """
    today = datetime.utcnow()

    # Monthly: 最近 N 个月（到上个月）
    # Binance monthly 数据通常次月初即发布
    # 安全起见：monthly 到上个月（当月还没结束，不会有 monthly）
    monthly_end = today.replace(day=1) - timedelta(days=1)  # 上个月最后一天

    monthly_end_year = monthly_end.year
    monthly_end_month = monthly_end.month

    # Monthly start: 往前推 N 个月
    monthly_start = monthly_end.replace(day=1)
    for _ in range(months - 1):
        monthly_start = (monthly_start - timedelta(days=1)).replace(day=1)
    monthly_start_year = monthly_start.year
    monthly_start_month = monthly_start.month

    # Daily: 从 monthly_end 的下一个月 1 号起；截止到「前天」——Vision 日线常有 ~1 日公布延迟，
    # 截止「昨天」会对当日 ZIP 反复 404。新月开头几天若前天早于月初，则钳制到月初（单日或空列表）。
    daily_start_dt = monthly_end + timedelta(days=1)
    daily_end_dt = today - timedelta(days=2)
    effective_end_dt = (
        daily_end_dt if daily_end_dt >= daily_start_dt else daily_start_dt
    )
    daily_start = daily_start_dt.strftime("%Y-%m-%d")
    daily_end = effective_end_dt.strftime("%Y-%m-%d")

    return (
        monthly_start_year,
        monthly_start_month,
        monthly_end_year,
        monthly_end_month,
        daily_start,
        daily_end,
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
    print(
        f"📥 下载 monthly aggTrades: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d}"
    )
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
        auto_confirm=True,
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
        all_ticks = all_ticks.drop_duplicates(subset=["timestamp", "side"], keep="last")

        print(f"      总记录: {len(all_ticks):,} 条")
        print(
            f"      时间范围: {all_ticks['timestamp'].min()} ~ {all_ticks['timestamp'].max()}"
        )

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
# Fill-gap mode (--fill-gap)
# ---------------------------------------------------------------------------


def detect_last_date(ticks_dir: Path, symbols: List[str]) -> Optional[str]:
    """检测 live/data/ticks/ 中已有数据的最后日期

    Returns:
        最后日期字符串 'YYYY-MM-DD'，无数据则返回 None
    """
    latest = None
    for symbol in symbols:
        sym_dir = ticks_dir / symbol
        if not sym_dir.exists():
            continue
        for f in sym_dir.glob("*.parquet"):
            date_str = f.stem  # e.g. '2026-01-31'
            if latest is None or date_str > latest:
                latest = date_str
    return latest


def detect_gaps(
    ticks_dir: Path, symbols: List[str], max_lookback_days: int = 100
) -> List[str]:
    """检测数据中间缺口

    Args:
        ticks_dir: ticks数据目录
        symbols: 币种列表
        max_lookback_days: 最多向前检查多少天

    Returns:
        缺失的日期列表 ['YYYY-MM-DD', ...]
    """
    from datetime import timezone

    today = datetime.now(timezone.utc)
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # 收集所有币种的已有日期
    all_dates = set()
    for symbol in symbols:
        sym_dir = ticks_dir / symbol
        if not sym_dir.exists():
            continue
        for f in sym_dir.glob("*.parquet"):
            all_dates.add(f.stem)

    if not all_dates:
        return []

    # 找到最早和最晚日期
    min_date = min(all_dates)
    max_date = max(all_dates)

    # 检查范围：从最早日期到昨天
    start_dt = datetime.strptime(min_date, "%Y-%m-%d")
    end_dt = datetime.strptime(yesterday, "%Y-%m-%d")

    # 生成应有的日期序列
    missing = []
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        if date_str not in all_dates:
            missing.append(date_str)
        current += timedelta(days=1)

    return missing


def fill_gap(
    symbols: List[str],
    ticks_dir: Path,
    bars_dir: Path,
    zip_dir: Path,
) -> Dict[str, int]:
    """检测并补全数据缺口（包括中间缺口和尾部缺口）

    检测已有数据的日期范围，找出所有缺失的日期，下载并补全。
    """
    from datetime import timezone

    today = datetime.now(timezone.utc)
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    last_date = detect_last_date(ticks_dir, symbols)
    if last_date is None:
        print("\n⚠️  live/data/ticks/ 无已有数据，请先运行 --from-local 或默认模式")
        return {}

    # 检测中间缺口
    missing_dates = detect_gaps(ticks_dir, symbols)

    # 检测尾部缺口（最后日期到昨天）
    gap_start_dt = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
    gap_start = gap_start_dt.strftime("%Y-%m-%d")
    if gap_start <= yesterday:
        # 添加尾部缺失的日期
        current = gap_start_dt
        end_dt = datetime.strptime(yesterday, "%Y-%m-%d")
        while current <= end_dt:
            date_str = current.strftime("%Y-%m-%d")
            if date_str not in missing_dates:
                missing_dates.append(date_str)
            current += timedelta(days=1)

    # 去重并排序
    missing_dates = sorted(set(missing_dates))

    if not missing_dates:
        print(
            f"\n✅ 数据已是最新（最后日期: {last_date}，昨天: {yesterday}），无需补全"
        )
        return {}

    print(f"\n{'='*60}")
    print("🔍 补全缺失数据 (fill-gap)")
    print(f"{'='*60}")
    print(f"   检测到 {len(missing_dates)} 天缺口")
    if len(missing_dates) <= 10:
        print(f"   缺失日期: {', '.join(missing_dates)}")
    else:
        print(
            f"   缺失日期: {', '.join(missing_dates[:5])} ... {', '.join(missing_dates[-3:])}"
        )

    # 按日期范围分组下载（连续的日期一起下载）
    if missing_dates:
        start_date = missing_dates[0]
        end_date = missing_dates[-1]
        print(f"   下载范围: {start_date} ~ {end_date}")

        # 下载 daily
        download_daily(symbols, start_date, end_date, zip_dir)

        # 转换并写入
        stats = convert_and_split(zip_dir, symbols, ticks_dir, bars_dir)
        return stats

    return {}


def prepare_warmup_dataset(
    *,
    symbols: List[str],
    months: int,
    ticks_dir: Path,
    bars_dir: Path,
    zip_dir: Path,
    force_full: bool = False,
    skip_download: bool = False,
    use_monthly_zip: bool = False,
) -> Dict[str, int]:
    """Prepare live warmup ticks/bars from Binance Vision aggTrades.

    Default downloads **daily** ZIPs for the full contiguous warmup window
    (memory-friendly versus large monthly ZIPs).

    If every symbol already covers the expected warmup span, only gaps up to
    yesterday are filled. Otherwise the full download/convert path runs so cold
    starts do not begin with only same-day WebSocket data.

    Args:
        use_monthly_zip: If True, use legacy monthly ZIPs plus a trailing daily
            segment (fewer HTTP requests, higher peak RAM per convert).
    """
    ticks_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)

    if not force_full and has_required_warmup_coverage(ticks_dir, symbols, months):
        print("\n✅ warmup 覆盖范围充足，仅检查/补全 daily gap")
        return fill_gap(
            symbols=symbols,
            ticks_dir=ticks_dir,
            bars_dir=bars_dir,
            zip_dir=zip_dir,
        )

    required_start, required_end = required_warmup_bounds(months)
    print("\n" + "=" * 60)
    if use_monthly_zip:
        print("📦 warmup 覆盖不足（monthly ZIP + trailing daily）")
    else:
        print("📦 warmup 覆盖不足（daily aggTrades ZIP 全覆盖，默认）")
    print("=" * 60)
    print(f"   Required ticks: {required_start} ~ {required_end}")
    print(f"   Ticks dir:      {ticks_dir}")
    print(f"   Bars dir:       {bars_dir}")
    print(f"   ZIP dir:        {zip_dir}")

    (
        m_start_y,
        m_start_m,
        m_end_y,
        m_end_m,
        d_start,
        d_end,
    ) = compute_date_ranges(months)

    print(f"\n   回溯月数锚点: {m_start_y}-{m_start_m:02d} ~ {m_end_y}-{m_end_m:02d}（日历窗口）")

    if not skip_download:
        if use_monthly_zip:
            print(f"   Trailing daily（monthly 之后）: {d_start} ~ {d_end}")
            download_monthly(
                symbols, m_start_y, m_start_m, m_end_y, m_end_m, zip_dir
            )
            download_daily(symbols, d_start, d_end, zip_dir)
        else:
            print(f"   Daily Vision 下载: {required_start} ~ {required_end}")
            download_daily(symbols, required_start, required_end, zip_dir)
    else:
        print("\n⏭️  跳过下载步骤")

    return convert_and_split(zip_dir, symbols, ticks_dir, bars_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="准备实盘 warmup ticks 数据（下载 + 转换 + 拆分到 live 目录）"
    )
    parser.add_argument(
        "--universe",
        default="highcap",
        help="Universe 名称（默认: highcap）",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help=(
            "warmup 日历窗口按月回溯 N 个月（边界见脚本内 compute_date_ranges）"
            "；默认对该窗口内每一天下载 UM daily aggTrades ZIP。"
        ),
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="逗号分隔的 symbols（默认从 universe.yaml 读取）",
    )
    parser.add_argument(
        "--from-local",
        action="store_true",
        help="从 data/parquet_data/ 读取已有 1min 聚合数据（跳过下载和转换）",
    )
    parser.add_argument(
        "--fill-gap",
        action="store_true",
        help="只补全缺失的 daily 数据（检测已有数据最后日期，仅下载此后到昨天）",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载步骤（仅转换已有的 ZIP 文件）",
    )
    parser.add_argument(
        "--monthly-zip",
        action="store_true",
        help=(
            "使用 Binance Vision 按月 aggTrades ZIP + trailing daily（HTTP 更少，"
            "但单月 CSV 解压内存峰值更高；默认不推荐在内存紧张主机上使用）"
        ),
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "若 ticks 已覆盖所需月份窗口则只做 gap fill（与 feature publisher 默认一致）；"
            "默认仍为强制完整下载/转换。"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
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
    if args.from_local:
        mode_note = "from-local (parquet_data)"
    elif args.fill_gap:
        mode_note = "fill-gap (daily only)"
    elif args.monthly_zip:
        mode_note = "download + convert (monthly + trailing daily ZIP)"
    else:
        mode_note = "download + convert (daily Vision ZIP, default)"
    print(f"   Mode:       {mode_note}")
    print(f"   Ticks dir:  {ticks_dir}")
    print(f"   Bars dir:   {bars_dir}")

    if args.from_local:
        # --from-local: 直接从 data/parquet_data/ 读取
        stats = load_from_local_parquet(
            symbols=symbols,
            months=args.months,
            ticks_output_dir=ticks_dir,
            bars_output_dir=bars_dir,
        )
    elif args.fill_gap:
        # --fill-gap: 只补全缺失的 daily 数据
        stats = fill_gap(
            symbols=symbols,
            ticks_dir=ticks_dir,
            bars_dir=bars_dir,
            zip_dir=zip_dir,
        )
    else:
        stats = prepare_warmup_dataset(
            symbols=symbols,
            months=args.months,
            ticks_dir=ticks_dir,
            bars_dir=bars_dir,
            zip_dir=zip_dir,
            force_full=not args.incremental,
            skip_download=args.skip_download,
            use_monthly_zip=args.monthly_zip,
        )

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
