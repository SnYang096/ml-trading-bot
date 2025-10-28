"""Enhanced feature engineering module using TA-Lib indicators.

基于TA-Lib库的增强特征工程模块，提供：
1. 使用TA-Lib替换自实现的技术指标
2. 添加更多传统技术指标（趋势、动量、波动率、成交量等）
3. 特征归一化（StandardScaler/MinMaxScaler/RobustScaler）
4. Scaler保存和加载功能

TA-Lib提供158+种技术指标，比自实现更准确和高效。
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import warnings

# 忽略TA-Lib的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)


class TalibFeatureEngineer:
    """Enhanced feature engineer using TA-Lib indicators."""

    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize the TA-Lib feature engineer.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scalers = {}  # Store scalers for each timeframe
        self.feature_stats = {}  # Store feature statistics

        # Choose scaler
        if scaler_type == "standard":
            self.scaler_class = StandardScaler
        elif scaler_type == "minmax":
            self.scaler_class = MinMaxScaler
        elif scaler_type == "robust":
            self.scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加趋势类指标."""
        df = data.copy()

        # 简单移动平均线 (SMA)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{period}"] = talib.SMA(df["close"].values, timeperiod=period)

        # 指数移动平均线 (EMA)
        for period in [5, 10, 20, 50, 100]:
            df[f"ema_{period}"] = talib.EMA(df["close"].values, timeperiod=period)

        # 加权移动平均线 (WMA)
        for period in [10, 20, 50]:
            df[f"wma_{period}"] = talib.WMA(df["close"].values, timeperiod=period)

        # 三角移动平均线 (TEMA)
        for period in [10, 20, 30]:
            df[f"tema_{period}"] = talib.TEMA(df["close"].values, timeperiod=period)

        # 考夫曼自适应移动平均线 (KAMA)
        for period in [10, 20, 30]:
            df[f"kama_{period}"] = talib.KAMA(df["close"].values, timeperiod=period)

        # 抛物线SAR
        df["sar"] = talib.SAR(df["high"].values, df["low"].values)
        df["sar_ext"] = talib.SAREXT(df["high"].values, df["low"].values)

        # 平均方向指数 (ADX)
        df["adx"] = talib.ADX(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=14
        )
        df["adxr"] = talib.ADXR(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=14
        )

        # 正负方向指标
        df["plus_di"] = talib.PLUS_DI(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=14
        )
        df["minus_di"] = talib.MINUS_DI(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=14
        )

        # 阿隆指标
        df["aroon_up"], df["aroon_down"] = talib.AROON(
            df["high"].values, df["low"].values, timeperiod=14
        )
        df["aroon_osc"] = talib.AROONOSC(
            df["high"].values, df["low"].values, timeperiod=14
        )

        return df

    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加动量类指标."""
        df = data.copy()

        # RSI (多个周期)
        for period in [7, 14, 21]:
            df[f"rsi_{period}"] = talib.RSI(df["close"].values, timeperiod=period)

        # 随机指标 (Stochastic)
        df["stoch_k"], df["stoch_d"] = talib.STOCH(
            df["high"].values, df["low"].values, df["close"].values
        )
        df["stochf_k"], df["stochf_d"] = talib.STOCHF(
            df["high"].values, df["low"].values, df["close"].values
        )
        df["stochrsi_k"], df["stochrsi_d"] = talib.STOCHRSI(
            df["close"].values, timeperiod=14
        )

        # 威廉指标 (Williams %R)
        df["willr"] = talib.WILLR(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=14
        )

        # 动量指标
        for period in [5, 10, 14]:
            df[f"mom_{period}"] = talib.MOM(df["close"].values, timeperiod=period)

        # 变化率 (ROC)
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = talib.ROC(df["close"].values, timeperiod=period)

        # 商品通道指数 (CCI)
        df["cci"] = talib.CCI(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=14
        )

        # 终极指标 (Ultimate Oscillator)
        df["ultosc"] = talib.ULTOSC(
            df["high"].values, df["low"].values, df["close"].values
        )

        # 真实强度指数 (TSI) - 使用自定义实现，因为TA-Lib没有TSI
        def compute_tsi(series, period1=14, period2=25):
            """计算真实强度指数 (TSI)."""
            momentum = series.diff()
            smoothed_momentum = (
                momentum.ewm(span=period1).mean().ewm(span=period2).mean()
            )
            smoothed_abs_momentum = (
                momentum.abs().ewm(span=period1).mean().ewm(span=period2).mean()
            )
            tsi = 100 * smoothed_momentum / smoothed_abs_momentum
            return tsi

        df["tsi"] = compute_tsi(df["close"])

        return df

    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加波动率类指标."""
        df = data.copy()

        # 布林带
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
            df["close"].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        # 布林带宽度和位置
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # 平均真实波幅 (ATR)
        for period in [7, 14, 21]:
            df[f"atr_{period}"] = talib.ATR(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                timeperiod=period,
            )

        # 真实波幅 (TRANGE)
        df["trange"] = talib.TRANGE(
            df["high"].values, df["low"].values, df["close"].values
        )

        # 平均方向指数 (ADX) - 也用于波动率
        df["natr"] = talib.NATR(
            df["high"].values, df["low"].values, df["close"].values, timeperiod=14
        )

        # 历史波动率
        returns = df["close"].pct_change()
        for period in [10, 20, 30]:
            df[f"volatility_{period}"] = returns.rolling(window=period).std()

        return df

    def add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加成交量类指标."""
        df = data.copy()

        # 成交量移动平均
        for period in [5, 10, 20]:
            df[f"volume_sma_{period}"] = talib.SMA(
                df["volume"].values, timeperiod=period
            )

        # 成交量比率
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # 平衡成交量 (OBV)
        df["obv"] = talib.OBV(df["close"].values, df["volume"].values)

        # 成交量加权平均价格 (VWAP) - 简化版
        df["vwap"] = (df["close"] * df["volume"]).rolling(window=20).sum() / df[
            "volume"
        ].rolling(window=20).sum()

        # 累积/派发线 (A/D Line)
        df["ad"] = talib.AD(
            df["high"].values, df["low"].values, df["close"].values, df["volume"].values
        )

        # 柴金资金流量 (CMF)
        df["cmf"] = talib.ADOSC(
            df["high"].values, df["low"].values, df["close"].values, df["volume"].values
        )

        # 成交量价格趋势 (VPT) - 使用自定义实现，因为TA-Lib没有VPT
        def compute_vpt(close, volume):
            """计算成交量价格趋势 (VPT)."""
            price_change = close.pct_change()
            vpt = (price_change * volume).cumsum()
            return vpt

        df["vpt"] = compute_vpt(df["close"], df["volume"])

        return df

    def add_cycle_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加周期类指标."""
        df = data.copy()

        # 希尔伯特变换 - 主导周期
        df["ht_dcperiod"] = talib.HT_DCPERIOD(df["close"].values)
        df["ht_dcphase"] = talib.HT_DCPHASE(df["close"].values)

        # 希尔伯特变换 - 相位
        df["ht_phasor_inphase"], df["ht_phasor_quadrature"] = talib.HT_PHASOR(
            df["close"].values
        )

        # 希尔伯特变换 - 正弦波
        df["ht_sine"], df["ht_leadsine"] = talib.HT_SINE(df["close"].values)

        # 希尔伯特变换 - 趋势模式
        df["ht_trendmode"] = talib.HT_TRENDMODE(df["close"].values)

        return df

    def add_pattern_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加形态识别指标."""
        df = data.copy()

        # 蜡烛图形态识别
        df["cdl_doji"] = talib.CDLDOJI(
            df["open"].values, df["high"].values, df["low"].values, df["close"].values
        )
        df["cdl_hammer"] = talib.CDLHAMMER(
            df["open"].values, df["high"].values, df["low"].values, df["close"].values
        )
        df["cdl_hanging_man"] = talib.CDLHANGINGMAN(
            df["open"].values, df["high"].values, df["low"].values, df["close"].values
        )
        df["cdl_engulfing"] = talib.CDLENGULFING(
            df["open"].values, df["high"].values, df["low"].values, df["close"].values
        )
        df["cdl_harami"] = talib.CDLHARAMI(
            df["open"].values, df["high"].values, df["low"].values, df["close"].values
        )
        df["cdl_doji_star"] = talib.CDLDOJISTAR(
            df["open"].values, df["high"].values, df["low"].values, df["close"].values
        )
        df["cdl_hammer"] = talib.CDLHAMMER(
            df["open"].values, df["high"].values, df["low"].values, df["close"].values
        )
        df["cdl_shooting_star"] = talib.CDLSHOOTINGSTAR(
            df["open"].values, df["high"].values, df["low"].values, df["close"].values
        )

        return df

    def add_math_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加数学运算指标."""
        df = data.copy()

        # 数学运算
        df["add"] = talib.ADD(df["high"].values, df["low"].values)
        df["div"] = talib.DIV(df["high"].values, df["low"].values)
        df["max"] = talib.MAX(df["close"].values, timeperiod=14)
        df["min"] = talib.MIN(df["close"].values, timeperiod=14)
        df["maxindex"] = talib.MAXINDEX(df["close"].values, timeperiod=14)
        df["minindex"] = talib.MININDEX(df["close"].values, timeperiod=14)

        # 统计指标
        df["stddev"] = talib.STDDEV(df["close"].values, timeperiod=14)
        df["var"] = talib.VAR(df["close"].values, timeperiod=14)

        return df

    def add_macd_variants(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加MACD变体指标."""
        df = data.copy()

        # 标准MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(df["close"].values)

        # MACD扩展
        df["macd_ext"], df["macd_ext_signal"], df["macd_ext_hist"] = talib.MACDEXT(
            df["close"].values, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # MACD固定
        df["macd_fix"], df["macd_fix_signal"], df["macd_fix_hist"] = talib.MACDFIX(
            df["close"].values, signalperiod=9
        )

        return df

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加所有TA-Lib技术指标."""
        if data.empty:
            return data

        df = data.copy()

        # 确保数据是数值类型并转换为double
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

        # 移除包含NaN的行
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        if df.empty:
            return df

        try:
            # 添加各类指标，每个类别独立处理以避免单个失败影响整体
            try:
                df = self.add_trend_indicators(df)
            except Exception as e:
                print(f"Warning: Error in trend indicators: {e}")

            try:
                df = self.add_momentum_indicators(df)
            except Exception as e:
                print(f"Warning: Error in momentum indicators: {e}")

            try:
                df = self.add_volatility_indicators(df)
            except Exception as e:
                print(f"Warning: Error in volatility indicators: {e}")

            try:
                df = self.add_volume_indicators(df)
            except Exception as e:
                print(f"Warning: Error in volume indicators: {e}")

            try:
                df = self.add_cycle_indicators(df)
            except Exception as e:
                print(f"Warning: Error in cycle indicators: {e}")

            try:
                df = self.add_pattern_indicators(df)
            except Exception as e:
                print(f"Warning: Error in pattern indicators: {e}")

            try:
                df = self.add_math_indicators(df)
            except Exception as e:
                print(f"Warning: Error in math indicators: {e}")

            try:
                df = self.add_macd_variants(df)
            except Exception as e:
                print(f"Warning: Error in MACD indicators: {e}")

            # 添加价格衍生特征
            df["price_change"] = df["close"].pct_change()
            df["high_low_ratio"] = df["high"] / df["low"]
            df["close_open_ratio"] = df["close"] / df["open"]

            # 添加移动平均比率
            df["sma_5_20_ratio"] = df["sma_5"] / df["sma_20"]
            df["sma_10_50_ratio"] = df["sma_10"] / df["sma_50"]
            df["ema_5_20_ratio"] = df["ema_5"] / df["ema_20"]

            # 添加RSI衍生特征
            df["rsi_14_normalized"] = (df["rsi_14"] - 50) / 50
            df["rsi_7_14_diff"] = df["rsi_7"] - df["rsi_14"]

            # 添加MACD衍生特征
            df["macd_normalized"] = df["macd"] / df["close"]
            df["macd_signal_ratio"] = df["macd_signal"] / df["macd"]

        except Exception as e:
            print(f"Warning: Error computing TA-Lib indicators: {e}")
            # 如果出错，返回原始数据
            return data

        # 填充NaN值
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        for col in feature_cols:
            df[col] = df[col].fillna(0)

        return df

    def normalize_features(
        self, data: pd.DataFrame, timeframe: str, fit: bool = True
    ) -> pd.DataFrame:
        """
        Improved feature normalization with feature grouping and outlier handling.

        Args:
            data: DataFrame with features
            timeframe: Timeframe identifier
            fit: Whether to fit the scaler (True for training, False for prediction)

        Returns:
            DataFrame with normalized features
        """
        df = data.copy()

        # Get feature columns (exclude OHLCV)
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        if not feature_cols:
            return df

        # 1. 预处理: 移除常数特征和异常值
        valid_features = []
        for col in feature_cols:
            values = df[col].dropna()
            if len(values) > 0:
                # 检查是否为常数特征
                if values.std() < 1e-10:
                    print(f"Warning: Removing constant feature: {col}")
                    continue

                # 检查异常值比例
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = values[(values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)]
                    outlier_ratio = len(outliers) / len(values)
                    if outlier_ratio > 0.5:  # 超过50%的异常值，跳过该特征
                        print(
                            f"Warning: Skipping feature {col} due to too many outliers ({outlier_ratio:.1%})"
                        )
                        continue

                valid_features.append(col)

        if not valid_features:
            print("Warning: No valid features found after preprocessing")
            return df

        # 2. 特征分组 - 根据特征类型选择不同的归一化策略
        price_features = [
            col
            for col in valid_features
            if any(x in col for x in ["sma", "ema", "wma", "tema", "kama", "sar"])
        ]
        ratio_features = [
            col
            for col in valid_features
            if "ratio" in col or "position" in col or "normalized" in col
        ]
        volatility_features = [
            col
            for col in valid_features
            if any(x in col for x in ["atr", "natr", "trange", "bb_", "volatility"])
        ]
        volume_features = [
            col
            for col in valid_features
            if any(x in col for x in ["volume", "obv", "ad", "vpt", "cmf"])
        ]
        momentum_features = [
            col
            for col in valid_features
            if any(
                x in col
                for x in ["rsi", "stoch", "willr", "mom", "roc", "cci", "ultosc", "tsi"]
            )
        ]
        index_features = [
            col
            for col in valid_features
            if any(x in col for x in ["maxindex", "minindex", "max", "min"])
        ]
        other_features = [
            col
            for col in valid_features
            if col
            not in price_features
            + ratio_features
            + volatility_features
            + volume_features
            + momentum_features
            + index_features
        ]

        # 3. 分组归一化
        feature_groups = {
            "price": (price_features, self.scaler_class()),
            "ratio": (ratio_features, MinMaxScaler()),
            "volatility": (volatility_features, RobustScaler()),
            "volume": (volume_features, self.scaler_class()),
            "momentum": (momentum_features, MinMaxScaler()),
            "index": (index_features, MinMaxScaler()),
            "other": (other_features, self.scaler_class()),
        }

        # 初始化group_scalers
        if not hasattr(self, "group_scalers"):
            self.group_scalers = {}

        for group_name, (group_features, scaler) in feature_groups.items():
            if not group_features:
                continue

            # 准备数据
            X = df[group_features].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # 对于成交量特征，先进行对数变换
            if group_name == "volume":
                X = np.log1p(np.abs(X)) * np.sign(X)  # log1p处理负值

            if fit:
                X_scaled = scaler.fit_transform(X)
                # 保存scaler
                self.group_scalers[f"{timeframe}_{group_name}"] = scaler
            else:
                # 使用已保存的scaler
                if f"{timeframe}_{group_name}" not in self.group_scalers:
                    raise ValueError(f"No scaler found for {timeframe}_{group_name}")
                scaler = self.group_scalers[f"{timeframe}_{group_name}"]
                X_scaled = scaler.transform(X)

            # 更新DataFrame
            for i, col in enumerate(group_features):
                df[col] = X_scaled[:, i]

        # 4. 存储特征统计信息
        if fit:
            all_features = []
            for group_features, _ in feature_groups.values():
                all_features.extend(group_features)

            if all_features:
                X_all = df[all_features].values
                self.feature_stats[timeframe] = {
                    "mean": np.mean(X_all, axis=0),
                    "std": np.std(X_all, axis=0),
                    "min": np.min(X_all, axis=0),
                    "max": np.max(X_all, axis=0),
                    "feature_groups": {
                        name: features
                        for name, (features, _) in feature_groups.items()
                        if features
                    },
                }

        return df

    def engineer_features(
        self, multi_tf_data: Dict[str, pd.DataFrame], fit: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for multi-timeframe data with normalization.

        Args:
            multi_tf_data: Dictionary mapping timeframe to DataFrame
            fit: Whether to fit scalers (True for training, False for prediction)

        Returns:
            Dictionary with engineered and normalized features for each timeframe
        """
        engineered_data = {}

        for timeframe, data in multi_tf_data.items():
            print(f"Engineering TA-Lib features for {timeframe}: {data.shape}")

            # Add technical indicators
            df_with_indicators = self.add_technical_indicators(data)
            print(
                f"Added TA-Lib indicators for {timeframe}: {df_with_indicators.shape}"
            )

            # Normalize features
            df_normalized = self.normalize_features(
                df_with_indicators, timeframe, fit=fit
            )
            print(f"Normalized features for {timeframe}: {df_normalized.shape}")

            engineered_data[timeframe] = df_normalized

        return engineered_data

    def save_scalers(self, filepath: str):
        """Save fitted scalers to file."""
        scaler_data = {
            "scalers": getattr(self, "scalers", {}),
            "group_scalers": getattr(self, "group_scalers", {}),
            "feature_stats": self.feature_stats,
            "scaler_type": self.scaler_type,
            "removed_features": getattr(self, "removed_features", {}),
        }

        with open(filepath, "wb") as f:
            pickle.dump(scaler_data, f)

        print(f"TA-Lib scalers saved to {filepath}")

    def load_scalers(self, filepath: str):
        """Load fitted scalers from file."""
        with open(filepath, "rb") as f:
            scaler_data = pickle.load(f)

        self.scalers = scaler_data.get("scalers", {})
        self.group_scalers = scaler_data.get("group_scalers", {})
        self.feature_stats = scaler_data.get("feature_stats", {})
        self.scaler_type = scaler_data.get("scaler_type", "standard")
        self.removed_features = scaler_data.get("removed_features", {})

        print(f"TA-Lib scalers loaded from {filepath}")

    def get_feature_importance_info(self, timeframe: str) -> Dict:
        """Get feature statistics for analysis."""
        if timeframe not in self.feature_stats:
            return {}

        stats = self.feature_stats[timeframe]
        return {
            "mean": stats["mean"].tolist(),
            "std": stats["std"].tolist(),
            "min": stats["min"].tolist(),
            "max": stats["max"].tolist(),
            "scaler_type": self.scaler_type,
        }

    def get_available_indicators(self) -> List[str]:
        """Get list of all available TA-Lib indicators."""
        return talib.get_functions()

    def get_feature_count(self, data: pd.DataFrame) -> int:
        """Get the number of features after adding indicators."""
        df_with_indicators = self.add_technical_indicators(data)
        feature_cols = [
            col
            for col in df_with_indicators.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        return len(feature_cols)
