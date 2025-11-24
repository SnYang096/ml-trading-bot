"""
策略专属回测模块

为每个策略提供定制化的回测方法，确保回测逻辑与标签设计完全一致。
"""

from .sr_reversal_backtest import backtest_sr_reversal
from .sr_breakout_backtest import backtest_sr_breakout
from .compression_breakout_backtest import backtest_compression_breakout
from .trend_following_backtest import backtest_trend_following

__all__ = [
    "backtest_sr_reversal",
    "backtest_sr_breakout",
    "backtest_compression_breakout",
    "backtest_trend_following",
]
