#!/usr/bin/env python3
"""refresh_funding_oi_data.py — 增量刷新 funding_rate + OI parquet

数据来源:
  - Funding Rate: Binance API /fapi/v1/fundingRate (每8h一条, 最多1000条≈333天)
  - Open Interest: Binance API /futures/data/openInterestHist (5m粒度, 每次500条)

输出格式 (与研究 pipeline 完全兼容):
  - data/funding_rate/parquet/<SYMBOL>_YYYY-MM_funding_rate.parquet
  - data/open_interest/parquet/<SYMBOL>_YYYY-MM_oi_5m.parquet

用法:
  # 刷新指定 symbols (默认刷新最近2个月)
  python scripts/refresh_funding_oi_data.py --symbols BTCUSDT,ETHUSDT

  # 被 start_live.sh 调用 (自动读取 MLBOT_LIVE_SYMBOLS)
  python scripts/refresh_funding_oi_data.py

不需要 API key, 使用 Binance 公开端点。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com"


# ── Funding Rate ────────────────────────────────────────────


def _fetch_funding_rate(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> List[dict]:
    """Fetch funding rate history via /fapi/v1/fundingRate."""
    all_rows: list[dict] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        try:
            resp = session.get(
                f"{BASE_URL}/fapi/v1/fundingRate", params=params, timeout=15
            )
            if not resp.ok:
                logger.warning("fundingRate API %d for %s", resp.status_code, symbol)
                break
            data = resp.json()
            if not data:
                break
            all_rows.extend(data)
            last_ts = max(int(r["fundingTime"]) for r in data)
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            time.sleep(0.2)
        except Exception as e:
            logger.warning("fundingRate fetch error for %s: %s", symbol, e)
            break

    return all_rows


def refresh_funding_rate(
    symbols: List[str],
    parquet_dir: Path,
    lookback_days: int = 60,
) -> int:
    """Download recent funding rate and save to monthly parquet files.

    Returns number of files written.
    """
    session = requests.Session()
    now = datetime.now(tz=timezone.utc)
    start_ms = int((now - timedelta(days=lookback_days)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    files_written = 0

    for sym in symbols:
        rows = _fetch_funding_rate(session, sym, start_ms, end_ms)
        if not rows:
            logger.info("  %s: funding_rate 无新数据", sym)
            continue

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(
            df["fundingTime"].astype(int), unit="ms", utc=True
        )
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # 按月拆分写入
        for (year, month), group in df.groupby([df.index.year, df.index.month]):
            fname = f"{sym}_{year}-{month:02d}_funding_rate.parquet"
            out_path = parquet_dir / fname

            # 如果已有旧文件，合并去重
            if out_path.exists():
                try:
                    old = pd.read_parquet(out_path)
                    if (
                        not isinstance(old.index, pd.DatetimeIndex)
                        and "datetime" in old.columns
                    ):
                        old.index = pd.to_datetime(old["datetime"], utc=True)
                    combined = pd.concat([old, group[["funding_rate"]]])
                    combined = combined[
                        ~combined.index.duplicated(keep="last")
                    ].sort_index()
                except Exception:
                    combined = group[["funding_rate"]]
            else:
                combined = group[["funding_rate"]]

            parquet_dir.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(out_path)
            files_written += 1

        logger.info(
            "  %s: funding_rate %d 条, 覆盖 %s ~ %s",
            sym,
            len(rows),
            df.index.min().strftime("%Y-%m-%d"),
            df.index.max().strftime("%Y-%m-%d"),
        )

    return files_written


# ── Open Interest ───────────────────────────────────────────


def _fetch_oi_page(
    session: requests.Session,
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int = 500,
) -> List[dict]:
    """Fetch one page of OI history."""
    params = {
        "symbol": symbol,
        "period": period,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    try:
        resp = session.get(
            f"{BASE_URL}/futures/data/openInterestHist",
            params=params,
            timeout=15,
        )
        if not resp.ok:
            logger.warning(
                "OI API %d for %s (range %s~%s): %s",
                resp.status_code,
                symbol,
                start_ms,
                end_ms,
                resp.text[:200],
            )
            return []
        return resp.json()
    except Exception as e:
        logger.warning("OI fetch error for %s: %s", symbol, e)
        return []


# Binance /futures/data/openInterestHist 对 5m period 限制查询范围 ~30 天
_OI_MAX_RANGE_DAYS = 29


def refresh_open_interest(
    symbols: List[str],
    parquet_dir: Path,
    period: str = "5m",
    lookback_days: int = 60,
) -> int:
    """Download recent OI and save to monthly parquet files.

    Returns number of files written.
    """
    session = requests.Session()
    now = datetime.now(tz=timezone.utc)
    global_start = now - timedelta(days=lookback_days)
    end_ms = int(now.timestamp() * 1000)
    files_written = 0

    for sym in symbols:
        all_rows: list[dict] = []
        # 分段查询: 每段最多 _OI_MAX_RANGE_DAYS 天
        seg_start = global_start
        while seg_start < now:
            seg_end = min(seg_start + timedelta(days=_OI_MAX_RANGE_DAYS), now)
            seg_start_ms = int(seg_start.timestamp() * 1000)
            seg_end_ms = int(seg_end.timestamp() * 1000)
            cursor = seg_start_ms

            while cursor < seg_end_ms:
                page = _fetch_oi_page(session, sym, period, cursor, seg_end_ms)
                if not page:
                    break
                all_rows.extend(page)
                last_ts = max(int(r["timestamp"]) for r in page)
                if last_ts <= cursor:
                    break
                cursor = last_ts + 1
                time.sleep(0.3)

            seg_start = seg_end

        if not all_rows:
            logger.info("  %s: OI 无新数据", sym)
            continue

        df = pd.DataFrame(all_rows)
        df["datetime"] = pd.to_datetime(
            df["timestamp"].astype(int), unit="ms", utc=True
        )
        df["oi_contracts"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
        df["oi_usd"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
        df["_symbol"] = sym
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        for (year, month), group in df.groupby([df.index.year, df.index.month]):
            fname = f"{sym}_{year}-{month:02d}_oi_{period}.parquet"
            out_path = parquet_dir / fname

            out_cols = ["_symbol", "oi_contracts", "oi_usd"]
            save_group = group[[c for c in out_cols if c in group.columns]]

            if out_path.exists():
                try:
                    old = pd.read_parquet(out_path)
                    if (
                        not isinstance(old.index, pd.DatetimeIndex)
                        and "datetime" in old.columns
                    ):
                        old.index = pd.to_datetime(old["datetime"], utc=True)
                    combined = pd.concat([old, save_group])
                    combined = combined[
                        ~combined.index.duplicated(keep="last")
                    ].sort_index()
                except Exception:
                    combined = save_group
            else:
                combined = save_group

            parquet_dir.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(out_path)
            files_written += 1

        logger.info(
            "  %s: OI %d 条, 覆盖 %s ~ %s",
            sym,
            len(all_rows),
            df.index.min().strftime("%Y-%m-%d"),
            df.index.max().strftime("%Y-%m-%d"),
        )

    return files_written


# ── Public API ──────────────────────────────────────────────


def refresh_all(
    symbols: List[str],
    data_root: str = "data",
    lookback_days: int = 60,
) -> dict:
    """Refresh both funding_rate and OI for given symbols.

    Returns: {"funding_rate_files": N, "oi_files": M}
    """
    root = Path(data_root)
    fr_dir = root / "funding_rate" / "parquet"
    oi_dir = root / "open_interest" / "parquet"

    logger.info("📊 刷新 Funding Rate (最近 %d 天)...", lookback_days)
    fr_count = refresh_funding_rate(symbols, fr_dir, lookback_days)

    logger.info("📊 刷新 Open Interest (最近 %d 天)...", lookback_days)
    oi_count = refresh_open_interest(symbols, oi_dir, lookback_days=lookback_days)

    return {"funding_rate_files": fr_count, "oi_files": oi_count}


# ── CLI ─────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbols_str = ""
    lookback = 60

    # 支持 --symbols / --lookback-days 参数
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--symbols" and i + 1 < len(args):
            symbols_str = args[i + 1]
            i += 2
        elif args[i] == "--lookback-days" and i + 1 < len(args):
            lookback = int(args[i + 1])
            i += 2
        else:
            i += 1

    # 优先命令行参数，其次环境变量
    if not symbols_str:
        symbols_str = os.getenv("MLBOT_LIVE_SYMBOLS", "BTCUSDT")

    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
    data_root = os.getenv("MLBOT_DATA_ROOT", "data")

    logger.info(
        "🔄 Refresh funding/OI: symbols=%s, lookback=%d days, data_root=%s",
        symbols,
        lookback,
        data_root,
    )

    result = refresh_all(symbols, data_root=data_root, lookback_days=lookback)
    logger.info(
        "✅ 完成: funding_rate=%d files, OI=%d files",
        result["funding_rate_files"],
        result["oi_files"],
    )


if __name__ == "__main__":
    main()
