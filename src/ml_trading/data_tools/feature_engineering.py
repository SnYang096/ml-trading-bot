"""Feature engineering module for technical indicators.

这是基础版特征工程模块，提供13个基础技术指标。
所有基础指标计算函数已移至 base_indicators.py 以避免重复。
"""

import pandas as pd
from typing import Dict
from .base_indicators import add_basic_indicators


class FeatureEngineer:
    """基础特征工程类 - 处理市场数据的特征工程."""

    def __init__(self):
        """初始化特征工程器."""
        pass

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标到数据中.

        Args:
            data: OHLCV数据

        Returns:
            添加了技术指标的数据
        """
        return add_basic_indicators(data)

    def engineer_features(
        self, multi_tf_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        为多时间框架数据工程特征.

        Args:
            multi_tf_data: 时间框架到DataFrame的字典映射

        Returns:
            每个时间框架的工程特征字典
        """
        engineered_data = {}
        for timeframe, data in multi_tf_data.items():
            print(f"Engineering features for {timeframe}: {data.shape}")
            engineered_data[timeframe] = self.add_technical_indicators(data)
            print(
                f"Engineered features for {timeframe}: {engineered_data[timeframe].shape}"
            )
        return engineered_data
