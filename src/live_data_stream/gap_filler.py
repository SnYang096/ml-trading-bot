"""
数据补全器（集成Feature Store）

实现补数据逻辑：
1. Warmup时优先从Feature Store加载特征
2. 备选从Parquet warmup（ticks数据）
3. 一天以上从币安aggTrades API获取
"""

from __future__ import annotations

import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec

    FEATURE_STORE_AVAILABLE = True
except ImportError:
    FEATURE_STORE_AVAILABLE = False
    FeatureStore = None
    FeatureStoreSpec = None

from .feature_storage import StorageManager
from .data_gap_filler import DataGapFiller


class GapFiller:
    """
    数据补全器（集成Feature Store）

    功能：
    1. Warmup时优先从Feature Store加载特征
    2. 备选从Parquet warmup（ticks数据）
    3. 一天以上从币安aggTrades API获取
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        exchange: Optional[Any] = None,
        feature_store_dir: Optional[str] = None,
        feature_store_layer: Optional[str] = None,
    ):
        """
        Args:
            storage_manager: 存储管理器
            exchange: ccxt Exchange 实例（用于从币安API获取数据）
            feature_store_dir: Feature Store根目录
            feature_store_layer: Feature Store层名称
        """
        self.storage_manager = storage_manager

        # 数据补全器（用于从币安API获取数据）
        if exchange and CCXT_AVAILABLE:
            self.data_gap_filler = DataGapFiller(exchange)
        else:
            self.data_gap_filler = None

        # 后台补数据：Vision 404 的 gap 稍后重试（不阻塞启动）
        self._pending_vision_gaps: List[Dict[str, Any]] = []

        # Feature Store配置
        self.feature_store_dir = feature_store_dir
        self.feature_store_layer = feature_store_layer
        self.feature_store: Optional[FeatureStore] = None

        if feature_store_dir and FEATURE_STORE_AVAILABLE:
            try:
                self.feature_store = FeatureStore(feature_store_dir)
            except Exception as e:
                print(f"⚠️ 初始化Feature Store失败: {e}")
                self.feature_store = None

    def warmup_from_feature_store(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "15min",
    ) -> Optional[pd.DataFrame]:
        """
        从Feature Store加载特征（优先方案）

        Args:
            symbol: 交易对符号
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            timeframe: 时间框架（如 "15min"）

        Returns:
            特征DataFrame或None（如果Feature Store不可用）
        """
        if not self.feature_store or not self.feature_store_layer:
            return None

        try:
            # 创建 Feature Store 规格
            spec = FeatureStoreSpec(
                layer=self.feature_store_layer,
                symbol=symbol,
                timeframe=timeframe,
            )

            # 计算需要加载的月份
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            months = pd.period_range(start=start, end=end, freq="M")

            # 加载每个月的特征
            dfs = []
            for period in months:
                month = f"{period.year:04d}-{period.month:02d}"

                if not self.feature_store.has_month(spec, month):
                    continue

                try:
                    df_month = self.feature_store.read_month(spec, month)
                    if df_month is not None and len(df_month) > 0:
                        # 过滤日期范围
                        df_filtered = df_month[
                            (df_month.index >= pd.Timestamp(start_date))
                            & (df_month.index <= pd.Timestamp(end_date))
                        ]
                        if len(df_filtered) > 0:
                            dfs.append(df_filtered)
                except Exception as e:
                    print(f"⚠️ 加载 Feature Store 月份 {month} 失败: {e}")
                    continue

            if not dfs:
                return None

            # 合并数据
            combined = pd.concat(dfs)
            combined = combined.sort_index()
            combined = combined.drop_duplicates()

            # 确保 timestamp 作为列存在（Feature Store 使用 timestamp 作为索引）
            if (
                combined.index.name == "timestamp"
                and "timestamp" not in combined.columns
            ):
                combined = combined.reset_index()

            print(
                f"✅ 从 Feature Store 加载了 {len(combined)} 条特征（{start_date} 到 {end_date}）"
            )
            return combined

        except Exception as e:
            print(f"⚠️ 从Feature Store加载特征失败: {e}")
            return None

    def warmup_from_parquet(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_ticks: bool = True,
        use_bars: bool = False,
        use_features_15min: bool = False,
        use_features_4h: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        从parquet文件加载数据（备选方案）

        Args:
            symbol: 交易对符号
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            use_ticks: 是否加载tick数据（按买卖分离，用于VPIN等）
            use_bars: 是否加载1分钟bar数据（OHLCV）
            use_features_15min: 是否加载15分钟特征
            use_features_4h: 是否加载4小时特征

        Returns:
            包含不同类型数据的字典
        """
        result = {}

        if use_ticks:
            # 新增：加载1分钟聚合tick（按买卖分离）
            result["ticks_1min"] = self.storage_manager.ticks.load_range(
                symbol, start_date, end_date
            )

        if use_bars:
            # 加载1分钟OHLCV bar
            result["bars_1min"] = self.storage_manager.bar_1min.load_range(
                symbol, start_date, end_date
            )

        if use_features_15min:
            result["features_15min"] = self.storage_manager.feature_15min.load_range(
                symbol, start_date, end_date
            )

        if use_features_4h:
            result["features_4h"] = self.storage_manager.feature_4h.load_range(
                symbol, start_date, end_date
            )

        return result

    def fill_from_binance_api(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """
        从币安API获取数据（用于一天以上的数据缺失）

        Args:
            symbol: 交易对符号（如 "BTCUSDT"）
            start_time: 开始时间
            end_time: 结束时间
            timeframe: 时间框架（如 "1m"）

        Returns:
            下载的数据DataFrame
        """
        if not self.data_gap_filler:
            print("⚠️ 数据补全器未初始化（需要提供exchange）")
            return pd.DataFrame()

        # 转换符号格式（BTCUSDT -> BTC/USDT:USDT）
        ccxt_symbol = self._convert_symbol(symbol)

        # Pandas treats "m" as month-end; ccxt uses "1m" for one minute.
        pandas_freq = (
            timeframe.replace("m", "min") if timeframe.endswith("m") else timeframe
        )
        time_range = pd.date_range(start_time, end_time, freq=pandas_freq)

        # 下载数据
        df = self.data_gap_filler.download_missing_bars(
            symbol=ccxt_symbol,
            missing_timestamps=time_range.tolist(),
            timeframe=timeframe,
        )

        if len(df) > 0:
            print(f"✅ 从币安API下载了 {len(df)} 条数据（{start_time} 到 {end_time}）")

        return df

    def warmup(
        self,
        symbol: str,
        days: int = 30,
        end_date: Optional[str] = None,
        prefer_feature_store: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        执行Warmup（优先从Feature Store，备选从Parquet，缺失超过一天从币安API）

        Args:
            symbol: 交易对符号
            days: 加载最近N天的数据
            end_date: 结束日期 (YYYY-MM-DD)，默认为今天
            prefer_feature_store: 是否优先使用Feature Store

        Returns:
            包含不同类型数据的字典
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        end = datetime.strptime(end_date, "%Y-%m-%d")
        start = end - timedelta(days=days)
        start_date = start.strftime("%Y-%m-%d")

        result = {}

        # 1. 优先从 Feature Store 加载特征
        if prefer_feature_store:
            # 注意：Feature Store 使用 Pandas offset 格式（240T, 15T）
            # 而不是人类可读格式（4h, 15min）
            features_4h = self.warmup_from_feature_store(
                symbol, start_date, end_date, timeframe="240T"  # 4h = 240T
            )
            if features_4h is not None and len(features_4h) > 0:
                result["features_4h"] = features_4h

            features_15min = self.warmup_from_feature_store(
                symbol, start_date, end_date, timeframe="15T"  # 15min = 15T
            )
            if features_15min is not None and len(features_15min) > 0:
                result["features_15min"] = features_15min

        # 2. 从parquet加载数据（如果Feature Store没有或需要补充）
        parquet_data = self.warmup_from_parquet(
            symbol,
            start_date,
            end_date,
            use_ticks=True,  # 加载tick数据（用于VPIN等计算）
            use_bars=True,  # 加载bar数据（用于技术指标）
            use_features_15min=("features_15min" not in result),
            use_features_4h=("features_4h" not in result),
        )

        # 合并结果
        for key, df in parquet_data.items():
            if key in result:
                # 合并并去重
                combined = pd.concat([result[key], df])
                combined = combined.drop_duplicates()
                if isinstance(combined.index, pd.DatetimeIndex):
                    combined = combined.sort_index()
                elif "timestamp" in combined.columns:
                    combined = combined.sort_values("timestamp")
                result[key] = combined
            else:
                result[key] = df

        # 3. 补齐 1m bar gap，并把修复结果落盘。
        #    策略:
        #    - 内部 gap ≤1h: klines API（OHLCV，秒下）
        #    - 内部 gap >1h / 昨天及以前: Binance Vision（后台重试）
        #    - 今天的部分: klines API
        #    注: ticks（buy/sell 拆分）只通过 Vision 下载或本地上传获取
        if (
            "bars_1min" in result
            and len(result["bars_1min"]) > 0
            and self.data_gap_filler
        ):
            bars_1min = result["bars_1min"].copy()

            # 确保时区一致
            bars_1min["timestamp"] = pd.to_datetime(
                bars_1min["timestamp"], utc=True
            )
            bars_1min = bars_1min.sort_values("timestamp").reset_index(drop=True)

            now = pd.Timestamp.now(tz="UTC")
            ccxt_symbol = self._convert_symbol(symbol)
            total_filled = 0

            # 3a. 检测并补充内部 gap (>5min)
            from src.live_data_stream.system_mode import SystemModeManager

            temp_mgr = SystemModeManager()
            internal_gaps = temp_mgr._detect_gaps(bars_1min)
            large_internal = [g for g in internal_gaps if g["minutes"] > 5]

            if large_internal:
                print(
                    f"🔍 检测到 {len(large_internal)} 个内部 gap (>5min)，开始补充..."
                )
                for gap_info in large_internal:
                    gap_start = pd.Timestamp(gap_info["start_time"])
                    gap_end = pd.Timestamp(gap_info["end_time"])
                    if gap_start.tzinfo is None:
                        gap_start = gap_start.tz_localize("UTC")
                    if gap_end.tzinfo is None:
                        gap_end = gap_end.tz_localize("UTC")

                    gap_hours = gap_info["minutes"] / 60
                    # ≤1h: klines API（OHLCV，秒下）
                    # >1h: 推后台 Vision 重试
                    if gap_hours <= 1:
                        fill_data = self.fill_from_binance_api(
                            symbol, gap_start, gap_end, timeframe="1m"
                        )
                        if len(fill_data) > 0:
                            bars_1min = pd.concat([bars_1min, fill_data])
                            self._save_filled_data(symbol, fill_data)
                            total_filled += len(fill_data)
                    else:
                        print(f"  📦 内部 gap {gap_hours:.1f}h > 1h → 后台 Vision 重试")
                        self._queue_pending_vision_gap(
                            ccxt_symbol=ccxt_symbol,
                            raw_symbol=symbol,
                            start=gap_start,
                            end=gap_end,
                        )

                if total_filled > 0:
                    bars_1min = bars_1min.drop_duplicates(
                        subset=["timestamp"], keep="last"
                    )
                    bars_1min = bars_1min.sort_values("timestamp").reset_index(
                        drop=True
                    )
                    print(f"  ✅ 内部 gap 补充: {total_filled} 条 bars")

            # 3b. 补充末尾 gap（最后数据到当前时间）
            #    策略: 昨天及以前 → Vision, 今天 → klines API
            last_timestamp = bars_1min["timestamp"].max()
            tail_gap_seconds = (now - last_timestamp).total_seconds()

            if tail_gap_seconds > 300:
                tail_gap_hours = tail_gap_seconds / 3600
                print(f"📥 末尾 gap {tail_gap_hours:.1f}h，补齐中...")

                gap_start = last_timestamp + timedelta(minutes=1)
                today_start = now.normalize()  # 今天 00:00 UTC

                # 段一: 昨天及以前 — Binance Vision（只下可用天，404 加入后台重试）
                if gap_start < today_start:
                    vision_end = today_start - timedelta(minutes=1)
                    print(
                        f"  📦 昨天及以前: Binance Vision ({gap_start.strftime('%m-%d %H:%M')} ~ {vision_end.strftime('%m-%d %H:%M')})"
                    )
                    fill_vision, raw_ticks = (
                        self.data_gap_filler.fill_gap_with_binance_vision(
                            ccxt_symbol, gap_start, vision_end
                        )
                    )
                    if len(fill_vision) > 0:
                        bars_1min = pd.concat([bars_1min, fill_vision])
                        self._save_filled_data(symbol, fill_vision, raw_ticks)
                        total_filled += len(fill_vision)
                        # Vision 可能只覆盖了部分天（某些天 404）
                        vision_last = fill_vision["timestamp"].max()
                        remaining_hours = (
                            today_start - vision_last
                        ).total_seconds() / 3600
                        if remaining_hours > 1:
                            remaining_start = vision_last + timedelta(minutes=1)
                            print(
                                f"  📦 Vision 覆盖到 {vision_last.strftime('%m-%d %H:%M')}，"
                                f"剩余 {remaining_hours:.1f}h → 后台 Vision 重试"
                            )
                            self._queue_pending_vision_gap(
                                ccxt_symbol=ccxt_symbol,
                                raw_symbol=symbol,
                                start=remaining_start,
                                end=vision_end,
                            )
                        elif remaining_hours > 0.08:  # >5min 用 klines 快速补
                            remaining_start = vision_last + timedelta(minutes=1)
                            fill_small = self.fill_from_binance_api(
                                symbol, remaining_start, vision_end, timeframe="1m"
                            )
                            if len(fill_small) > 0:
                                bars_1min = pd.concat([bars_1min, fill_small])
                                self._save_filled_data(symbol, fill_small)
                                total_filled += len(fill_small)
                    else:
                        # Vision 完全失败
                        gap_hours = (vision_end - gap_start).total_seconds() / 3600
                        if gap_hours <= 1:
                            print("  ⚠️ Vision 失败，klines 补齐 (≤1h)...")
                            fill_kl = self.fill_from_binance_api(
                                symbol, gap_start, vision_end, timeframe="1m"
                            )
                            if len(fill_kl) > 0:
                                bars_1min = pd.concat([bars_1min, fill_kl])
                                self._save_filled_data(symbol, fill_kl)
                                total_filled += len(fill_kl)
                        else:
                            print(
                                f"  📦 Vision 失败，{gap_hours:.1f}h → 后台 Vision 重试"
                            )
                            self._queue_pending_vision_gap(
                                ccxt_symbol=ccxt_symbol,
                                raw_symbol=symbol,
                                start=gap_start,
                                end=vision_end,
                            )
                    gap_start = today_start

                # 段二: 今天 — 用 klines API 快速补最近部分（aggTrades 对 BTC 太慢）
                if gap_start < now:
                    today_gap_hours = (now - gap_start).total_seconds() / 3600
                    fill_start = (
                        gap_start if today_gap_hours <= 1 else now - timedelta(hours=1)
                    )
                    print(
                        f"  📥 今天: klines ({fill_start.strftime('%H:%M')} ~ {now.strftime('%H:%M')})"
                    )
                    fill_today = self.fill_from_binance_api(
                        symbol, fill_start, now, timeframe="1m"
                    )
                    if len(fill_today) > 0:
                        bars_1min = pd.concat([bars_1min, fill_today])
                        self._save_filled_data(symbol, fill_today)
                        total_filled += len(fill_today)

                bars_1min = bars_1min.drop_duplicates(
                    subset=["timestamp"], keep="last"
                )
                bars_1min = bars_1min.sort_values("timestamp").reset_index(drop=True)

            if total_filled > 0:
                result["bars_1min"] = bars_1min
                print(
                    f"✅ 补数据总计: {total_filled} 条 bars，数据连续到 {bars_1min['timestamp'].max()}"
                )
            elif tail_gap_seconds > 300:
                print(f"⚠️ 补数据全部失败，将依赖自动升级机制")

        return result

    def retry_pending_gaps(self) -> bool:
        """后台重试: 下载 Vision 404 的 gap

        Returns:
            True: 所有 gap 已填充
        """
        if not self._pending_vision_gaps or not self.data_gap_filler:
            return len(self._pending_vision_gaps) == 0

        remaining = []
        for gap in self._pending_vision_gaps:
            print(
                f"📦 后台重试 Vision: {gap['raw_symbol']} "
                f"{gap['start'].strftime('%m-%d %H:%M')} ~ {gap['end'].strftime('%m-%d %H:%M')}"
            )
            fill, raw_ticks = self.data_gap_filler.fill_gap_with_binance_vision(
                gap["symbol"], gap["start"], gap["end"]
            )
            if len(fill) > 0:
                # 保存到磁盘（下次启动不用重新下载）
                self._save_filled_data(gap["raw_symbol"], fill, raw_ticks)
                print(f"  ✅ 后台补数据成功: {len(fill)} 条 bars")
            else:
                remaining.append(gap)
                print(f"  ⚠️ 仍不可用，稍后重试")

        self._pending_vision_gaps = remaining
        return len(remaining) == 0

    def _queue_pending_vision_gap(
        self,
        *,
        ccxt_symbol: str,
        raw_symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        """Queue a Vision repair gap once; feature-bus retries these in background."""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")

        key = (raw_symbol, int(start.timestamp()), int(end.timestamp()))
        for gap in self._pending_vision_gaps:
            existing = (
                gap["raw_symbol"],
                int(pd.Timestamp(gap["start"]).timestamp()),
                int(pd.Timestamp(gap["end"]).timestamp()),
            )
            if existing == key:
                return

        self._pending_vision_gaps.append(
            {
                "symbol": ccxt_symbol,
                "raw_symbol": raw_symbol,
                "start": start,
                "end": end,
            }
        )

    def _save_filled_data(
        self, symbol: str, bars: pd.DataFrame, raw_ticks: pd.DataFrame | None = None
    ) -> None:
        """将补充的 bars 和原始 ticks 按天保存到 storage"""
        if self.storage_manager is None or len(bars) == 0:
            return
        if "timestamp" not in bars.columns:
            return

        bars_copy = bars.copy()
        bars_copy["timestamp"] = pd.to_datetime(bars_copy["timestamp"], utc=True)
        bars_copy["_date"] = bars_copy["timestamp"].dt.strftime("%Y-%m-%d")
        for date_str, day_bars in bars_copy.groupby("_date"):
            day_data = day_bars.drop(columns=["_date"])
            try:
                self.storage_manager.bar_1min.append(symbol, date_str, day_data)
            except Exception as e:
                print(f"  ⚠️ 保存 bars {symbol}/{date_str} 失败: {e}")

        # 原始 ticks 单独存
        if raw_ticks is not None and len(raw_ticks) > 0:
            self._save_raw_ticks(symbol, raw_ticks)

    def _save_raw_ticks(self, symbol: str, raw_ticks: pd.DataFrame) -> None:
        """将原始 tick 数据 [timestamp, price, volume, side] 按天保存"""
        if self.storage_manager is None or len(raw_ticks) == 0:
            return
        ticks_copy = raw_ticks.copy()
        ticks_copy["timestamp"] = pd.to_datetime(ticks_copy["timestamp"], utc=True)
        ticks_copy["_date"] = ticks_copy["timestamp"].dt.strftime("%Y-%m-%d")
        for date_str, day_ticks in ticks_copy.groupby("_date"):
            day_data = day_ticks.drop(columns=["_date"])
            try:
                self.storage_manager.ticks.append(symbol, date_str, day_data)
            except Exception as e:
                print(f"  ⚠️ 保存 ticks {symbol}/{date_str} 失败: {e}")

    def fill_gap(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        source: str = "auto",
    ) -> pd.DataFrame:
        """
        补数据（自动选择数据源）

        Args:
            symbol: 交易对符号
            start_time: 开始时间
            end_time: 结束时间
            source: 数据源（"auto", "feature_store", "parquet", "binance"）

        Returns:
            补全的数据DataFrame
        """
        gap_duration = end_time - start_time

        # 自动选择数据源
        if source == "auto":
            # 如果缺失超过1天，优先从币安API获取
            if gap_duration.total_seconds() > 86400:
                source = "binance"
            # 否则优先从Parquet获取
            else:
                source = "parquet"

        if source == "feature_store":
            start_date = start_time.strftime("%Y-%m-%d")
            end_date = end_time.strftime("%Y-%m-%d")
            df = self.warmup_from_feature_store(symbol, start_date, end_date)
            if df is not None:
                return df

        if source == "parquet":
            start_date = start_time.strftime("%Y-%m-%d")
            end_date = end_time.strftime("%Y-%m-%d")
            data = self.warmup_from_parquet(
                symbol, start_date, end_date, use_ticks=True
            )
            if "ticks_1min" in data:
                return data["ticks_1min"]

        if source == "binance":
            return self.fill_from_binance_api(symbol, start_time, end_time)

        return pd.DataFrame()

    def _convert_symbol(self, symbol: str) -> str:
        """
        转换符号格式（BTCUSDT -> BTC/USDT:USDT）

        Args:
            symbol: 交易对符号

        Returns:
            ccxt格式的符号
        """
        if "USDT" in symbol:
            base = symbol.replace("USDT", "")
            return f"{base}/USDT:USDT"
        return symbol

    def fill_missing_ticks(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        existing_ticks: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        补充缺失的 ticks 数据（短时断线后调用）

        使用 Binance GET /fapi/v1/aggTrades 接口

        Args:
            symbol: 交易对符号（如 "BTCUSDT"）
            start_time: 开始时间（断线开始）
            end_time: 结束时间（重连成功）
            existing_ticks: 已有的 ticks 数据（用于去重，避免覆盖）

        Returns:
            补充的 trades 数据 DataFrame，列: [timestamp, price, volume, side]
            只返回缺失的部分，不包含 existing_ticks 中已有的数据
        """
        if not self.data_gap_filler:
            print("⚠️ 数据补全器未初始化（需要提供 exchange）")
            return pd.DataFrame()

        # 转换符号格式
        ccxt_symbol = self._convert_symbol(symbol)

        # 下载缺失的 trades
        df = self.data_gap_filler.fill_missing_trades(
            symbol=ccxt_symbol,
            start_time=start_time,
            end_time=end_time,
        )

        if len(df) == 0:
            return pd.DataFrame()

        # 如果有已有数据，去除重复的 ticks
        if existing_ticks is not None and len(existing_ticks) > 0:
            original_count = len(df)

            # 确保 timestamp 列存在且格式一致
            if "timestamp" in existing_ticks.columns:
                existing_ts = pd.to_datetime(existing_ticks["timestamp"], utc=True)
                df_ts = pd.to_datetime(df["timestamp"], utc=True)

                # 按时间戳去重（只保留 existing 中没有的）
                # 使用时间戳字符串比较，精度到毫秒
                existing_ts_set = set(existing_ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f"))
                mask = ~df_ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f").isin(existing_ts_set)
                df = df[mask].reset_index(drop=True)

                removed_count = original_count - len(df)
                if removed_count > 0:
                    print(f"   跳过 {removed_count} 条已存在的 ticks")

        if len(df) > 0:
            gap_duration = (end_time - start_time).total_seconds()
            print(f"✅ 补充了 {len(df)} 条 ticks（断线 {gap_duration:.1f} 秒）")

        return df

    def merge_ticks(
        self,
        existing_ticks: pd.DataFrame,
        new_ticks: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        合并 ticks 数据（去重 + 排序）

        Args:
            existing_ticks: 已有的 ticks 数据
            new_ticks: 新下载的 ticks 数据

        Returns:
            合并后的 DataFrame（去重、按时间排序）
        """
        if existing_ticks is None or len(existing_ticks) == 0:
            return new_ticks
        if new_ticks is None or len(new_ticks) == 0:
            return existing_ticks

        # 合并
        combined = pd.concat([existing_ticks, new_ticks], ignore_index=True)

        # 确保 timestamp 格式一致
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)

        # 去重（基于 timestamp + price + volume）
        combined = combined.drop_duplicates(
            subset=["timestamp", "price", "volume"],
            keep="first",  # 保留已有的（existing_ticks 在前面）
        )

        # 按时间排序
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        return combined
