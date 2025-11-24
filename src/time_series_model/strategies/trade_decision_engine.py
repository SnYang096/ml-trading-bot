"""
交易决策引擎

将模型输出转换为具体交易指令（方向、止损/止盈、仓位、加仓/减仓）

核心原则：每种策略的决策逻辑完全不同，必须独立实现
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal, Tuple
from dataclasses import dataclass


@dataclass
class TradeDecision:
    """交易决策结果"""

    direction: int  # 1=Long, -1=Short, 0=Hold
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float  # 仓位大小（0~1）
    add_position: bool  # 是否加仓
    reduce_position: bool  # 是否减仓
    confidence: float  # 信号置信度（0~1）
    strategy_type: str  # 策略类型


class TradeDecisionEngine:
    """交易决策引擎基类"""

    def __init__(
        self,
        base_position_size: float = 0.1,
        min_confidence: float = 0.3,
    ):
        """
        Args:
            base_position_size: 基础仓位大小
            min_confidence: 最小置信度阈值
        """
        self.base_position_size = base_position_size
        self.min_confidence = min_confidence

    def generate_decision(
        self,
        model_output: float,
        features: pd.Series,
        current_price: float,
        atr: float,
        **kwargs,
    ) -> Optional[TradeDecision]:
        """
        生成交易决策

        Args:
            model_output: 模型输出（根据策略类型不同而不同）
            features: 特征 Series
            current_price: 当前价格
            atr: ATR 值
            **kwargs: 策略特定参数

        Returns:
            TradeDecision 或 None（不交易）
        """
        raise NotImplementedError


class SRReversalDecisionEngine(TradeDecisionEngine):
    """
    SR 反转策略决策引擎

    模型输出：P(success) = P(label=1)
    决策逻辑：
    - 方向：由 SR 类型决定（支撑区 → 做多，阻力区 → 做空）
    - 止损：入场价 ± 1×ATR（反向）
    - 止盈：入场价 ± 2×ATR（同向）
    - 仓位：position_size ∝ P(success)
    - 加仓：❌ 通常不加仓
    - 减仓：若价格未快速脱离，且 CVD 相位转负，可提前部分止盈
    """

    def generate_decision(
        self,
        model_output: float,  # P(success)
        features: pd.Series,
        current_price: float,
        atr: float,
        sr_type: Literal["support", "resistance"],
        **kwargs,
    ) -> Optional[TradeDecision]:
        """
        生成 SR 反转交易决策

        Args:
            model_output: P(success) = P(label=1)
            features: 特征 Series
            current_price: 当前价格
            atr: ATR 值
            sr_type: SR 类型（"support" 或 "resistance"）

        Returns:
            TradeDecision 或 None
        """
        # 检查置信度
        if model_output < self.min_confidence:
            return None

        # 确定方向
        if sr_type == "support":
            direction = 1  # 做多
        elif sr_type == "resistance":
            direction = -1  # 做空
        else:
            return None

        # 计算止损/止盈
        if direction == 1:  # Long
            stop_loss = current_price - 1.0 * atr
            take_profit = current_price + 2.0 * atr
        else:  # Short
            stop_loss = current_price + 1.0 * atr
            take_profit = current_price - 2.0 * atr

        # 计算仓位（与 P(success) 正相关）
        position_size = self.base_position_size * model_output

        # 检查是否需要减仓（CVD 相位转负）
        cvd_phase_negative = features.get("hilbert_cvd_leads", 0) < 0
        reduce_position = cvd_phase_negative and model_output < 0.7

        return TradeDecision(
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            add_position=False,  # 反转策略不加仓
            reduce_position=reduce_position,
            confidence=model_output,
            strategy_type="sr_reversal",
        )


class SRBreakoutDecisionEngine(TradeDecisionEngine):
    """
    SR 突破策略决策引擎

    模型输出：Predicted R/R（预期风险回报比）
    决策逻辑：
    - 方向：由突破方向决定（向上突破 → 做多，向下突破 → 做空）
    - 止损：突破点反向 1×ATR 或区间边界
    - 止盈：动态止盈（初始 TP = 入场 + 2×ATR，若 R/R > 2.0 启用 trailing stop）
    - 仓位：position_size ∝ predicted_R_R
    - 加仓：✅ 若价格回踩不破且 CVD 持续流入，可在 0.5×ATR 处加仓
    - 减仓：当实际 R/R 达到预测值的 80%，减半仓锁定利润
    """

    def generate_decision(
        self,
        model_output: float,  # Predicted R/R
        features: pd.Series,
        current_price: float,
        atr: float,
        breakout_direction: Literal["up", "down"],
        **kwargs,
    ) -> Optional[TradeDecision]:
        """
        生成 SR 突破交易决策

        Args:
            model_output: Predicted R/R
            features: 特征 Series
            current_price: 当前价格
            atr: ATR 值
            breakout_direction: 突破方向（"up" 或 "down"）

        Returns:
            TradeDecision 或 None
        """
        # 检查 R/R 是否足够
        if model_output < 1.0:  # R/R < 1.0 不值得交易
            return None

        # 确定方向
        if breakout_direction == "up":
            direction = 1  # 做多
        elif breakout_direction == "down":
            direction = -1  # 做空
        else:
            return None

        # 计算止损（突破点反向 1×ATR）
        if direction == 1:  # Long
            stop_loss = current_price - 1.0 * atr
            # 动态止盈：初始 TP = 入场 + 2×ATR，若 R/R > 2.0 启用 trailing stop
            if model_output > 2.0:
                take_profit = current_price + model_output * atr  # 使用预测 R/R
            else:
                take_profit = current_price + 2.0 * atr
        else:  # Short
            stop_loss = current_price + 1.0 * atr
            if model_output > 2.0:
                take_profit = current_price - model_output * atr
            else:
                take_profit = current_price - 2.0 * atr

        # 计算仓位（与 R/R 正相关）
        # R/R=2.5 比 R/R=1.2 重仓
        position_size = self.base_position_size * min(model_output / 2.0, 1.0)

        # 检查是否需要加仓（价格回踩不破且 CVD 持续流入）
        cvd_positive = features.get("hilbert_cvd_leads", 0) > 0
        price_above_entry = (
            direction == 1 and current_price > kwargs.get("entry_price", current_price)
        ) or (
            direction == -1 and current_price < kwargs.get("entry_price", current_price)
        )
        add_position = cvd_positive and price_above_entry and model_output > 1.5

        # 检查是否需要减仓（当实际 R/R 达到预测值的 80%）
        # 这需要在回测中动态计算，这里只返回标志
        reduce_position = False  # 由回测系统动态判断

        return TradeDecision(
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            add_position=add_position,
            reduce_position=reduce_position,
            confidence=min(model_output / 3.0, 1.0),  # 归一化到 [0, 1]
            strategy_type="sr_breakout",
        )


class CompressionBreakoutDecisionEngine(TradeDecisionEngine):
    """
    压缩区突破策略决策引擎

    模型输出：{-1, 0, +1}（向下有效突破、假突破/回补、向上有效突破）
    决策逻辑：
    - 方向：直接由标签符号决定（+1 → 多，-1 → 空）
    - 止损：区间另一侧边界 或 突破点 ±1×ATR
    - 止盈：初始 TP = 区间高度（Measured Move）
    - 仓位：position_size = base_size if label ≠ 0 else 0
    - 加仓：❌ 通常不加仓
    - 减仓：价格达到区间高度目标，减半仓
    """

    def generate_decision(
        self,
        model_output: int,  # -1, 0, +1
        features: pd.Series,
        current_price: float,
        atr: float,
        compression_range: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Optional[TradeDecision]:
        """
        生成压缩区突破交易决策

        Args:
            model_output: -1, 0, +1
            features: 特征 Series
            current_price: 当前价格
            atr: ATR 值
            compression_range: 压缩区间 (low, high)

        Returns:
            TradeDecision 或 None
        """
        # 如果 label = 0，不交易
        if model_output == 0:
            return None

        # 确定方向
        direction = model_output  # +1 → 多，-1 → 空

        # 计算止损（区间另一侧边界 或 突破点 ±1×ATR）
        if compression_range:
            range_low, range_high = compression_range
            if direction == 1:  # Long
                stop_loss = max(range_low, current_price - 1.0 * atr)
            else:  # Short
                stop_loss = min(range_high, current_price + 1.0 * atr)
        else:
            if direction == 1:  # Long
                stop_loss = current_price - 1.0 * atr
            else:  # Short
                stop_loss = current_price + 1.0 * atr

        # 计算止盈（区间高度 Measured Move）
        if compression_range:
            range_low, range_high = compression_range
            range_height = range_high - range_low
            if direction == 1:  # Long
                take_profit = current_price + range_height
            else:  # Short
                take_profit = current_price - range_height
        else:
            # 如果没有区间信息，使用 2×ATR
            if direction == 1:  # Long
                take_profit = current_price + 2.0 * atr
            else:  # Short
                take_profit = current_price - 2.0 * atr

        # 计算仓位（label ≠ 0 时使用基础仓位）
        position_size = self.base_position_size

        return TradeDecision(
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            add_position=False,  # 压缩区突破不加仓
            reduce_position=False,  # 由回测系统动态判断
            confidence=1.0 if model_output != 0 else 0.0,
            strategy_type="compression_breakout",
        )


class TrendFollowingDecisionEngine(TradeDecisionEngine):
    """
    趋势跟踪策略决策引擎

    模型输出：Rank（0~1 之间，收益率百分位）
    决策逻辑：
    - 方向：由动量方向决定（ROC(20) > 0 → 做多，ROC(20) < 0 → 做空）
    - 止损：动态 ATR 止损（如 2×ATR）或 trendline break
    - 止盈：无固定止盈，靠 Rank 下降退出（当 Rank < 0.4，平仓）
    - 仓位：position_size ∝ (label - 0.5)
    - 加仓：✅ 每当 Rank 创新高（>0.95），加仓
    - 减仓：Rank 从高位回落至 0.7 以下，减仓 50%
    """

    def generate_decision(
        self,
        model_output: float,  # Rank (0~1)
        features: pd.Series,
        current_price: float,
        atr: float,
        momentum_direction: Optional[int] = None,  # 1=Up, -1=Down
        previous_rank: Optional[float] = None,
        **kwargs,
    ) -> Optional[TradeDecision]:
        """
        生成趋势跟踪交易决策

        Args:
            model_output: Rank (0~1)
            features: 特征 Series
            current_price: 当前价格
            atr: ATR 值
            momentum_direction: 动量方向（1=Up, -1=Down），如果为 None 则从特征中提取
            previous_rank: 前一个 Rank 值（用于判断是否创新高）

        Returns:
            TradeDecision 或 None
        """
        # 如果 Rank < 0.4，不交易（退出信号）
        if model_output < 0.4:
            return None

        # 确定方向（从动量方向或特征中提取）
        if momentum_direction is None:
            # 从特征中提取 ROC(20) 方向
            roc_20 = features.get("roc_20", 0)
            if roc_20 > 0:
                direction = 1  # 做多
            elif roc_20 < 0:
                direction = -1  # 做空
            else:
                return None
        else:
            direction = momentum_direction

        # 计算止损（动态 ATR 止损，如 2×ATR）
        if direction == 1:  # Long
            stop_loss = current_price - 2.0 * atr
        else:  # Short
            stop_loss = current_price + 2.0 * atr

        # 无固定止盈（靠 Rank 下降退出）
        take_profit = None  # 由回测系统根据 Rank 动态退出

        # 计算仓位（position_size ∝ (Rank - 0.5)）
        # Rank=0.9 → 重仓，Rank=0.6 → 轻仓
        position_size = self.base_position_size * max(0, (model_output - 0.5) * 2.0)

        # 检查是否需要加仓（Rank 创新高 >0.95）
        add_position = (
            previous_rank is not None
            and model_output > 0.95
            and model_output > previous_rank
        )

        # 检查是否需要减仓（Rank 从高位回落至 0.7 以下）
        reduce_position = (
            previous_rank is not None and previous_rank > 0.7 and model_output < 0.7
        )

        return TradeDecision(
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,  # None 表示无固定止盈
            position_size=position_size,
            add_position=add_position,
            reduce_position=reduce_position,
            confidence=model_output,
            strategy_type="trend_following",
        )


def create_decision_engine(
    strategy_type: Literal[
        "sr_reversal", "sr_breakout", "compression_breakout", "trend_following"
    ],
    base_position_size: float = 0.1,
    min_confidence: float = 0.3,
) -> TradeDecisionEngine:
    """
    创建策略专属的交易决策引擎

    Args:
        strategy_type: 策略类型
        base_position_size: 基础仓位大小
        min_confidence: 最小置信度阈值

    Returns:
        TradeDecisionEngine 实例
    """
    if strategy_type == "sr_reversal":
        return SRReversalDecisionEngine(base_position_size, min_confidence)
    elif strategy_type == "sr_breakout":
        return SRBreakoutDecisionEngine(base_position_size, min_confidence)
    elif strategy_type == "compression_breakout":
        return CompressionBreakoutDecisionEngine(base_position_size, min_confidence)
    elif strategy_type == "trend_following":
        return TrendFollowingDecisionEngine(base_position_size, min_confidence)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
