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
            # 创建Feature Store规格
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
                    df_month = self.feature_store.load_month(spec, month)
                    if len(df_month) > 0:
                        # 过滤日期范围
                        df_month = df_month[
                            (df_month.index >= pd.Timestamp(start_date))
                            & (df_month.index <= pd.Timestamp(end_date))
                        ]
                        if len(df_month) > 0:
                            dfs.append(df_month)
                except Exception as e:
                    print(f"⚠️ 加载Feature Store月份 {month} 失败: {e}")
                    continue
            
            if not dfs:
                return None
            
            # 合并数据
            combined = pd.concat(dfs)
            combined = combined.sort_index()
            combined = combined.drop_duplicates()
            
            print(f"✅ 从Feature Store加载了 {len(combined)} 条特征（{start_date} 到 {end_date}）")
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
        use_features_15min: bool = False,
        use_features_4h: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        从Parquet文件加载数据（备选方案）
        
        Args:
            symbol: 交易对符号
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            use_ticks: 是否加载ticks数据
            use_features_15min: 是否加载15分钟特征
            use_features_4h: 是否加载4小时特征
        
        Returns:
            包含不同类型数据的字典
        """
        result = {}
        
        if use_ticks:
            result["ticks_1min"] = self.storage_manager.tick_1min.load_range(
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
        
        # 生成需要下载的时间戳列表
        time_range = pd.date_range(start_time, end_time, freq=timeframe)
        
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
        
        # 1. 优先从Feature Store加载特征
        if prefer_feature_store:
            features_4h = self.warmup_from_feature_store(
                symbol, start_date, end_date, timeframe="4h"
            )
            if features_4h is not None and len(features_4h) > 0:
                result["features_4h"] = features_4h
            
            features_15min = self.warmup_from_feature_store(
                symbol, start_date, end_date, timeframe="15min"
            )
            if features_15min is not None and len(features_15min) > 0:
                result["features_15min"] = features_15min
        
        # 2. 从Parquet加载数据（如果Feature Store没有或需要补充）
        parquet_data = self.warmup_from_parquet(
            symbol,
            start_date,
            end_date,
            use_ticks=True,
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
        
        # 3. 检查数据缺失，如果缺失超过一天，从币安API获取
        if "ticks_1min" in result and len(result["ticks_1min"]) > 0:
            # 检查最后一条数据的时间
            last_timestamp = result["ticks_1min"]["timestamp"].max()
            now = pd.Timestamp.now(tz="UTC")
            
            # 如果缺失超过1天，从币安API获取
            if (now - last_timestamp).total_seconds() > 86400:
                print(f"⚠️ 检测到数据缺失超过1天，从币安API补数据...")
                fill_data = self.fill_from_binance_api(
                    symbol,
                    last_timestamp + timedelta(minutes=1),
                    now,
                    timeframe="1m",
                )
                
                if len(fill_data) > 0:
                    # 合并数据
                    combined = pd.concat([result["ticks_1min"], fill_data])
                    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                    combined = combined.sort_values("timestamp").reset_index(drop=True)
                    result["ticks_1min"] = combined
        
        return result
    
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
            data = self.warmup_from_parquet(symbol, start_date, end_date, use_ticks=True)
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
