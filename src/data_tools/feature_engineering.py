"""Feature engineering module for technical indicators.

这是基础版特征工程模块，提供常用的基础+TA-Lib指标集合。
所有基础指标计算函数已移至 base_indicators.py 以避免重复。
"""

import pandas as pd
from typing import Dict, Optional

from .base_indicators import add_common_derived_features
from .feature_engineering_talib import TalibFeatureEngineer


class FeatureEngineer:
    """基础特征工程类 - 产出统一的基础/TA-Lib/衍生特征集合."""

    def __init__(self) -> None:
        self._talib_engineer = TalibFeatureEngineer()

    def add_technical_indicators(self, data: pd.DataFrame, required_features: Optional[set] = None) -> pd.DataFrame:
        """组合基础指标、TA-Lib 指标以及常用衍生特征，如果指定了required_features，只计算需要的特征."""
        if data.empty:
            return data

        # 1) TA-Lib 指标（内部会处理数值化/缺失值）
        talib_df = self._talib_engineer.add_technical_indicators(data, required_features)

        # 2) 衍生指标（会确保基础指标完备，并填补常用派生列）
        # 如果指定了required_features，只添加需要的衍生特征
        enriched = add_common_derived_features(talib_df, required_features)

        return enriched

    def engineer_features(
            self,
            multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """为多时间框架数据工程统一的基础特征集."""
        engineered_data = {}
        for timeframe, data in multi_tf_data.items():
            print(f"Engineering basic features for {timeframe}: {data.shape}")
            engineered = self.add_technical_indicators(data)
            print(
                f"Engineered basic features for {timeframe}: {engineered.shape}"
            )
            engineered_data[timeframe] = engineered
        return engineered_data
