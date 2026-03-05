"""数据缺失补全器

检测实时流中的数据缺失，并从交易所 API / Binance Vision 下载补全。

补数据策略（按 gap 大小自动选择）:
  - gap < 24h:  aggTrades API（实时，有 buy/sell 拆分）
  - gap >= 24h: Binance Vision 每日 CSV 包（批量更快）→ aggTrades 回退
"""

from __future__ import annotations

import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️ ccxt not installed. Install with: pip install ccxt")


class DataGapFiller:
    """
    数据缺失补全器

    功能：
    1. 检测数据缺失（通过时间戳连续性）
    2. 从交易所 API 下载缺失数据
    3. 验证和清洗下载的数据
    4. 返回补全的数据
    """

    def __init__(self, exchange: Any):
        """
        Args:
            exchange: ccxt Exchange 实例
        """
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt is required for DataGapFiller")

        self.exchange = exchange

    def detect_missing_bars(
        self,
        df: pd.DataFrame,
        timeframe: str,
        tolerance: Optional[pd.Timedelta] = None,
    ) -> List[pd.Timestamp]:
        """
        检测缺失的 K线数据

        Args:
            df: 已有的 K线数据（按时间排序）
            timeframe: 时间框架（如 "15T"）
            tolerance: 允许的时间误差（默认 10%）

        Returns:
            缺失的时间戳列表
        """
        if len(df) < 2:
            return []

        if "timestamp" not in df.columns:
            return []

        # 计算期望的间隔
        expected_interval = pd.Timedelta(timeframe)
        if tolerance is None:
            tolerance = expected_interval * 0.1

        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        missing_timestamps = []

        for i in range(len(df_sorted) - 1):
            current_time = pd.Timestamp(df_sorted.iloc[i]["timestamp"])
            next_time = pd.Timestamp(df_sorted.iloc[i + 1]["timestamp"])

            # 计算实际间隔
            gap = next_time - current_time

            # 如果间隔大于期望间隔 + 容差，认为有缺失
            if gap > expected_interval + tolerance:
                # 计算缺失的时间戳数量
                missing_count = int((gap - tolerance) / expected_interval)

                for j in range(1, missing_count + 1):
                    missing_time = current_time + expected_interval * j
                    # 确保缺失时间在容差范围内
                    if missing_time < next_time - tolerance:
                        missing_timestamps.append(missing_time)

        return missing_timestamps

    def download_missing_bars(
        self,
        symbol: str,
        missing_timestamps: List[pd.Timestamp],
        timeframe: str,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        从交易所下载缺失的 K线数据

        Args:
            symbol: 交易对符号（ccxt 格式，如 "BTC/USDT:USDT"）
            missing_timestamps: 缺失的时间戳列表
            timeframe: 时间框架（ccxt 格式，如 "15m"）
            max_retries: 最大重试次数

        Returns:
            下载的 K线数据 DataFrame
        """
        if not missing_timestamps:
            return pd.DataFrame()

        # 转换时间框架格式
        ccxt_timeframe = self._convert_timeframe(timeframe)

        # 找到时间范围
        start_time = min(missing_timestamps)
        end_time = max(missing_timestamps)

        # 计算需要下载的数据量
        expected_count = len(missing_timestamps)

        # 多下载一些，避免边界问题
        limit = expected_count + 100

        for attempt in range(max_retries):
            try:
                # 转换为毫秒时间戳
                since = int(start_time.timestamp() * 1000)

                # 下载数据
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=ccxt_timeframe,
                    since=since,
                    limit=limit,
                )

                if not ohlcv:
                    print(f"⚠️ 下载返回空数据")
                    return pd.DataFrame()

                # 转换为 DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                # 过滤出缺失的时间戳（允许小的时间误差）
                tolerance = pd.Timedelta(timeframe) * 0.1
                matched_bars = []

                for missing_ts in missing_timestamps:
                    # 找到最接近的时间戳
                    time_diffs = abs(df["timestamp"] - missing_ts)
                    min_diff_idx = time_diffs.idxmin()

                    if time_diffs.iloc[min_diff_idx] <= tolerance:
                        matched_bars.append(min_diff_idx)

                if matched_bars:
                    df_matched = df.iloc[matched_bars].copy()
                    df_matched = df_matched.drop_duplicates(subset=["timestamp"])
                    print(
                        f"✅ 下载了 {len(df_matched)} 条缺失数据（期望 {expected_count} 条）"
                    )
                    return df_matched
                else:
                    print(f"⚠️ 下载的数据中没有匹配的时间戳")
                    return pd.DataFrame()

            except Exception as e:
                print(f"⚠️ 下载缺失数据失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # 指数退避
                else:
                    print(f"❌ 下载失败，已重试 {max_retries} 次")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fill_missing_trades(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        从交易所 API 下载缺失的 trades 数据

        使用 Binance GET /fapi/v1/aggTrades 接口

        Args:
            symbol: 交易对符号（ccxt 格式，如 "BTC/USDT:USDT"）
            start_time: 开始时间
            end_time: 结束时间
            max_retries: 最大重试次数

        Returns:
            trades 数据 DataFrame，列: [timestamp, price, volume, side]
        """
        all_trades = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        for attempt in range(max_retries):
            try:
                while current_start < end_ms:
                    # 使用 ccxt 的 fetch_trades（内部调用 aggTrades）
                    trades = self.exchange.fetch_trades(
                        symbol,
                        since=current_start,
                        limit=1000,  # Binance 限制每次最多 1000 条
                    )

                    if not trades:
                        break

                    for trade in trades:
                        trade_ts = trade.get("timestamp", 0)
                        if trade_ts > end_ms:
                            break

                        all_trades.append(
                            {
                                "timestamp": pd.Timestamp(
                                    trade_ts, unit="ms", tz="UTC"
                                ),
                                "price": float(trade.get("price", 0)),
                                "volume": float(trade.get("amount", 0)),
                                # Binance: side='sell' 表示 taker 是卖方（即 buyer is maker）
                                "side": 1 if trade.get("side") == "buy" else -1,
                            }
                        )

                    # 更新起始时间为最后一条 trade 的时间 + 1ms
                    if trades:
                        last_ts = trades[-1].get("timestamp", current_start)
                        if last_ts <= current_start:
                            break  # 避免无限循环
                        current_start = last_ts + 1
                    else:
                        break

                    # 速率限制
                    time.sleep(0.1)

                if all_trades:
                    df = pd.DataFrame(all_trades)
                    df = df.drop_duplicates(subset=["timestamp", "price", "volume"])
                    df = df.sort_values("timestamp").reset_index(drop=True)
                    print(
                        f"✅ 下载了 {len(df)} 条 trades（{start_time} 到 {end_time}）"
                    )
                    return df
                else:
                    return pd.DataFrame()

            except Exception as e:
                print(f"⚠️ 下载 trades 失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    print(f"❌ 下载失败，已重试 {max_retries} 次")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fill_gap_with_aggtrades(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        用 aggTrades 补充 gap 并聚合为 1min bars（含 buy/sell 拆分）

        相比 klines，优势是有 buy_volume/sell_volume/delta 字段，
        VPIN/订单流特征在 gap 段也能正确计算。

        Args:
            symbol: ccxt 格式符号（如 "BTC/USDT:USDT"）
            start_time: 补充起始时间
            end_time: 补充结束时间
            max_retries: 最大重试次数

        Returns:
            1min bars DataFrame，列: [timestamp, open, high, low, close, volume,
                                      buy_volume, sell_volume, delta, trade_count]
        """
        gap_hours = (end_time - start_time).total_seconds() / 3600
        print(
            f"📥 aggTrades 补数据: {symbol} {start_time} → {end_time} ({gap_hours:.1f}h)"
        )

        # 分块下载（aggTrades API 限制 24h 窗口）
        all_trades = []
        chunk_start = start_time
        chunk_size = pd.Timedelta(hours=23)  # 留一小时余量

        while chunk_start < end_time:
            chunk_end = min(chunk_start + chunk_size, end_time)

            trades_df = self.fill_missing_trades(
                symbol, chunk_start, chunk_end, max_retries=max_retries
            )

            if len(trades_df) > 0:
                all_trades.append(trades_df)
                print(
                    f"   ✔ {chunk_start.strftime('%m-%d %H:%M')} ~ {chunk_end.strftime('%m-%d %H:%M')}: {len(trades_df)} trades"
                )
            else:
                print(
                    f"   ⚠ {chunk_start.strftime('%m-%d %H:%M')} ~ {chunk_end.strftime('%m-%d %H:%M')}: 无数据"
                )

            chunk_start = chunk_end

        if not all_trades:
            print("⚠️ aggTrades 补数据返回空")
            return pd.DataFrame()

        # 合并所有 trades
        trades_all = pd.concat(all_trades, ignore_index=True)
        trades_all = trades_all.drop_duplicates(subset=["timestamp", "price", "volume"])
        trades_all = trades_all.sort_values("timestamp").reset_index(drop=True)

        # 聚合为 1min bars
        bars_1min = self._aggregate_trades_to_1min(trades_all)
        print(
            f"✅ aggTrades 补数据完成: {len(bars_1min)} 条 1min bars（含 buy/sell 拆分）"
        )
        return bars_1min

    def _aggregate_trades_to_1min(self, trades: pd.DataFrame) -> pd.DataFrame:
        """将原始 trades 聚合为 1min bars（含订单流字段）

        Args:
            trades: 原始 trades DataFrame [timestamp, price, volume, side]

        Returns:
            1min bars DataFrame
        """
        if len(trades) == 0:
            return pd.DataFrame()

        # 计算每笔 trade 属于哪个 1min bar
        trades = trades.copy()
        trades["bar_ts"] = trades["timestamp"].dt.floor("1min")
        trades["buy_vol"] = trades["volume"].where(trades["side"] == 1, 0.0)
        trades["sell_vol"] = trades["volume"].where(trades["side"] == -1, 0.0)

        # 按 1min 分组聚合
        grouped = trades.groupby("bar_ts")

        bars = pd.DataFrame(
            {
                "timestamp": grouped["bar_ts"].first(),
                "open": grouped["price"].first(),
                "high": grouped["price"].max(),
                "low": grouped["price"].min(),
                "close": grouped["price"].last(),
                "volume": grouped["volume"].sum(),
                "buy_volume": grouped["buy_vol"].sum(),
                "sell_volume": grouped["sell_vol"].sum(),
                "trade_count": grouped["price"].count(),
            }
        )

        bars["delta"] = bars["buy_volume"] - bars["sell_volume"]
        bars["buy_ratio"] = (bars["buy_volume"] / bars["volume"]).fillna(0.5)
        bars["sell_ratio"] = (bars["sell_volume"] / bars["volume"]).fillna(0.5)

        bars = bars.sort_values("timestamp").reset_index(drop=True)
        return bars

    def fill_gap_with_binance_vision(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """从 Binance Vision 下载每日 aggTrades CSV 并聚合为 1min bars

        适用场景: gap >= 1 天，使用官方每日 CSV 数据包。
        URL 格式: https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip

        Args:
            symbol: ccxt 格式符号（如 "BTC/USDT:USDT"），内部转换为 BTCUSDT
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            (bars, raw_ticks): 1min bars + 原始 tick 数据 [timestamp, price, volume, side]
        """
        import io
        import zipfile

        _empty = (pd.DataFrame(), pd.DataFrame())
        try:
            import requests
        except ImportError:
            print("⚠️ requests 未安装，无法使用 Binance Vision")
            return _empty

        # 转换符号: BTC/USDT:USDT → BTCUSDT
        raw_symbol = symbol.replace("/", "").replace(":USDT", "")

        # 计算需要下载的日期列表（不含今天，Binance Vision 当天数据不可用）
        start_date = start_time.normalize()  # 取日期部分
        end_date = end_time.normalize()
        today = pd.Timestamp.now(tz="UTC").normalize()

        # 最多下到昨天
        if end_date >= today:
            end_date = today - pd.Timedelta(days=1)

        if start_date > end_date:
            print("⚠️ Binance Vision: 时间范围无效（start > end 或全在今天内）")
            return _empty

        dates = pd.date_range(start_date, end_date, freq="D")
        print(
            f"📦 Binance Vision 下载: {raw_symbol}, {len(dates)} 天 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})"
        )

        all_trades = []
        base_url = "https://data.binance.vision/data/futures/um/daily/aggTrades"

        session = requests.Session()
        # 代理兼容：与 WebSocket/_build_gap_filler 一致，检测 HTTP_PROXY 环境变量
        # TUN 模式下无需设置（透明代理）
        import os as _os

        for _key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            _proxy = _os.environ.get(_key)
            if _proxy:
                session.proxies = {"http": _proxy, "https": _proxy}
                break
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry = Retry(
                total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503)
            )
            session.mount("https://", HTTPAdapter(max_retries=retry))
        except Exception:
            pass

        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            filename = f"{raw_symbol}-aggTrades-{date_str}.zip"
            url = f"{base_url}/{raw_symbol}/{filename}"

            try:
                resp = session.get(url, timeout=60)
                if resp.status_code == 404:
                    print(f"   ⚠ {date_str}: 数据不存在(404)")
                    continue
                resp.raise_for_status()

                # 解压 CSV
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    if not csv_names:
                        print(f"   ⚠ {date_str}: ZIP 无 CSV")
                        continue
                    with zf.open(csv_names[0]) as f:
                        df = pd.read_csv(f, header=0)

                if df.empty:
                    continue

                # 列: agg_trade_id, price, quantity, first_trade_id, last_trade_id,
                #      transact_time, is_buyer_maker
                trades_day = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            df.iloc[:, 5], unit="ms", utc=True
                        ),  # transact_time
                        "price": df.iloc[:, 1].astype(float),  # price
                        "volume": df.iloc[:, 2].astype(float),  # quantity
                        "side": np.where(
                            df.iloc[:, 6] == True, -1, 1
                        ),  # is_buyer_maker: True = seller taker
                    }
                )

                all_trades.append(trades_day)
                print(f"   ✔ {date_str}: {len(trades_day)} trades")

            except Exception as e:
                print(f"   ⚠ {date_str}: {e}")
                continue

        if not all_trades:
            print("⚠️ Binance Vision 下载返回空")
            return _empty

        # 合并所有 trades 并聚合为 1min bars
        trades_all = pd.concat(all_trades, ignore_index=True)
        trades_all = trades_all.sort_values("timestamp").reset_index(drop=True)

        # 只保留请求的时间范围
        trades_all = trades_all[
            (trades_all["timestamp"] >= start_time)
            & (trades_all["timestamp"] <= end_time)
        ].reset_index(drop=True)

        bars = self._aggregate_trades_to_1min(trades_all)
        print(
            f"✅ Binance Vision 完成: {len(bars)} 条 1min bars, {len(trades_all)} 条 raw ticks"
        )
        return bars, trades_all

    def _convert_timeframe(self, timeframe: str) -> str:
        """
        转换时间框架格式

        Args:
            timeframe: 时间框架（如 "15T" 或 "15m"）

        Returns:
            ccxt 格式的时间框架（如 "15m"）
        """
        if "T" in timeframe:
            return timeframe.replace("T", "m")
        return timeframe

    def validate_downloaded_data(
        self,
        df: pd.DataFrame,
        expected_timestamps: List[pd.Timestamp],
        timeframe: str,
    ) -> pd.DataFrame:
        """
        验证下载的数据质量

        Args:
            df: 下载的数据
            expected_timestamps: 期望的时间戳列表
            timeframe: 时间框架

        Returns:
            验证通过的数据
        """
        if len(df) == 0:
            return df

        tolerance = pd.Timedelta(timeframe) * 0.1
        validated = []

        for expected_ts in expected_timestamps:
            # 找到最接近的数据
            time_diffs = abs(df["timestamp"] - expected_ts)
            min_diff_idx = time_diffs.idxmin()

            if time_diffs.iloc[min_diff_idx] <= tolerance:
                row = df.iloc[min_diff_idx].copy()

                # 验证数据合理性
                if self._validate_bar(row):
                    validated.append(row)

        if validated:
            return pd.DataFrame(validated).reset_index(drop=True)
        else:
            return pd.DataFrame()

    def _validate_bar(self, bar: pd.Series) -> bool:
        """
        验证单条 K线数据的合理性

        Args:
            bar: K线数据 Series

        Returns:
            是否通过验证
        """
        try:
            # 检查基本字段
            required_fields = ["open", "high", "low", "close", "volume"]
            for field in required_fields:
                if field not in bar or pd.isna(bar[field]):
                    return False

            # 检查价格合理性
            if not (bar["low"] <= bar["open"] <= bar["high"]):
                return False
            if not (bar["low"] <= bar["close"] <= bar["high"]):
                return False
            if not (bar["low"] <= bar["high"]):
                return False

            # 检查数值合理性
            if bar["volume"] < 0:
                return False
            if bar["open"] <= 0 or bar["close"] <= 0:
                return False

            return True

        except Exception:
            return False


# 使用示例
if __name__ == "__main__":
    if not CCXT_AVAILABLE:
        print("❌ 请先安装 ccxt: pip install ccxt")
        exit(1)

    # 创建交易所实例
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )

    # 创建数据补全器
    gap_filler = DataGapFiller(exchange)

    # 模拟已有数据（有缺失）
    df_existing = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01 10:00:00"),
                pd.Timestamp("2024-01-01 10:15:00"),
                # 缺失 10:30, 10:45
                pd.Timestamp("2024-01-01 11:00:00"),
            ],
            "open": [50000, 50100, 50200],
            "high": [50100, 50200, 50300],
            "low": [49900, 50000, 50100],
            "close": [50050, 50150, 50250],
            "volume": [100, 110, 120],
        }
    )

    # 检测缺失
    missing = gap_filler.detect_missing_bars(df_existing, timeframe="15T")
    print(f"检测到缺失数据: {missing}")

    # 下载缺失数据
    if missing:
        df_filled = gap_filler.download_missing_bars(
            symbol="BTC/USDT:USDT",
            missing_timestamps=missing,
            timeframe="15T",
        )
        print(f"下载的数据: {len(df_filled)} 条")
