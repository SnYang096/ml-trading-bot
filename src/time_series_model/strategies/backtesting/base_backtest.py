"""
统一回测接口基类

所有策略特定的回测类都应该继承这个基类，实现统一的接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class BaseBacktest(ABC):
    """统一回测接口基类"""

    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        task_type: str = "binary",
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行回测

        Args:
            df: 包含 OHLCV 数据和特征的 DataFrame
            predictions: 模型预测值（概率或回归值）
            task_type: 任务类型 ("binary", "multiclass", "regression")
            **kwargs: 其他策略特定参数

        Returns:
            统一格式的回测结果字典，包含：
            - total_return_pct: 总收益率（百分比）
            - sharpe: Sharpe 比率
            - max_drawdown_pct: 最大回撤（百分比）
            - win_rate: 胜率（百分比）
            - total_trades: 总交易数
            - winning_trades: 盈利交易数
            - losing_trades: 亏损交易数
            - avg_rr: 平均 R/R（可选）
            - profit_factor: 盈亏比（可选）
            - trades: 交易记录列表（可选，debug 模式）
        """
        pass

    def _normalize_result(
        self, result: Dict[str, Any], default_keys: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        标准化返回结果，确保包含所有必需字段

        Args:
            result: 原始回测结果
            default_keys: 默认字段列表

        Returns:
            标准化后的结果字典
        """
        if default_keys is None:
            default_keys = [
                "total_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "win_rate",
                "total_trades",
                "winning_trades",
                "losing_trades",
            ]

        normalized = {}
        for key in default_keys:
            normalized[key] = result.get(key, 0.0)

        # 可选字段
        optional_keys = ["avg_rr", "profit_factor", "trades", "equity_curve"]
        for key in optional_keys:
            if key in result:
                normalized[key] = result[key]

        return normalized
