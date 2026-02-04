"""
BPC策略V2 - 符合路径2.5数学特征分层使用规范

分层架构：
- Gate层：❌ 禁止使用任何数学特征（Hurst/WPT/Spectrum/Hilbert）
           ✅ 仅使用结构、订单流、regime特征
- Evidence层：❌ 禁止使用execution_noise_penalty或原始数学特征
             ✅ 仅基于structure/orderflow/regime评估alpha质量
- Execution层：✅ 同时消费evidence_score和noise_penalty两个独立输入
             ✅ 在Tier内部依据noise_penalty动态调整sl_r/tp_r/size_multiplier等参数
- noise_penalty作用：仅影响'怎么做'（Execution决策），绝不影响'做不做'（Gate/Evidence决策）
"""

from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np

from ..evidence.bpc_evidence_calculator import (
    BPCEvidenceCalculator,
    calculate_bpc_evidence_score,
)
from ..execution.execution_controller import create_bpc_execution_controller
from ..execution.tier import ExecutionParams


class BPCStrategyV2:
    """BPC策略V2 - 路径2.5架构"""

    def __init__(self):
        self.evidence_calculator = BPCEvidenceCalculator()
        self.execution_controller = create_bpc_execution_controller()

    def evaluate_trade_opportunity(
        self, df: pd.DataFrame
    ) -> Tuple[bool, Optional[Dict]]:
        """
        评估交易机会（符合路径2.5分层规范）

        Args:
            df: 包含所有特征的数据框

        Returns:
            (是否执行交易, 执行参数字典)
        """
        # ========== Gate层决策 ==========
        # 根据路径2.5规范：Gate层禁止使用任何数学特征
        # 仅使用结构、订单流、regime特征进行决策
        gate_approved = self._evaluate_gate_criteria(df)

        if not gate_approved:
            return False, None

        # ========== Evidence层评估 ==========
        # 根据路径2.5规范：Evidence层禁止使用execution_noise_penalty或原始数学特征
        # 仅基于structure/orderflow/regime评估alpha质量
        evidence_score = self.evidence_calculator.calculate_evidence_score(df)
        current_evidence_score = evidence_score.iloc[-1]

        # ========== Execution层参数调整 ==========
        # 根据路径2.5规范：Execution层同时消费evidence_score和noise_penalty
        # noise_penalty仅影响'怎么做'，不影响'做不做'
        execution_params, debug_info = self.execution_controller.get_execution_params(
            df=df, evidence_score=current_evidence_score
        )

        if execution_params is None:
            # 被拒绝（证据不足）
            return False, None

        # 返回执行参数
        result = {
            "evidence_score": current_evidence_score,
            "noise_penalty": debug_info["noise_penalty"],
            "execution_params": execution_params,
            "debug_info": debug_info,
        }

        return True, result

    def _evaluate_gate_criteria(self, df: pd.DataFrame) -> bool:
        """
        Gate层决策（仅基于结构、订单流、regime特征）
        根据路径2.5规范：禁止使用任何数学特征（Hurst/WPT/Spectrum/Hilbert）
        """
        latest = df.iloc[-1]

        # 基本结构条件
        bpc_breakout_score = latest.get("bpc_score_breakout", 0.0)
        bpc_pullback_score = latest.get("bpc_score_pullback", 0.0)
        bpc_continuation_score = latest.get("bpc_score_continuation", 0.0)

        # 阶段分数至少有一个达到阈值
        stage_condition = (
            max(bpc_breakout_score, bpc_pullback_score, bpc_continuation_score) >= 0.3
        )

        # 订单流一致性
        cvd_div_score = latest.get("cvd_divergence_score", 0.0)
        momentum_div_score = latest.get("price_momentum_divergence", 0.0)
        orderflow_condition = (
            abs(cvd_div_score) >= 0.2 or abs(momentum_div_score) >= 0.2
        )

        # 趋势强度
        trend_r2 = latest.get("trend_r2_20", 0.0)
        trend_condition = trend_r2 >= 0.3

        # 波动率状态（非极端状态）
        vol_percentile = latest.get("vol_regime_vol_percentile", 0.5)
        vol_condition = 0.1 <= vol_percentile <= 0.9  # 避免过高/过低波动率

        # 综合Gate条件
        gate_approved = (
            stage_condition
            and orderflow_condition
            and trend_condition
            and vol_condition
        )

        return gate_approved


def run_bpc_strategy_v2(df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
    """
    运行BPC策略V2的便捷函数
    """
    strategy = BPCStrategyV2()
    return strategy.evaluate_trade_opportunity(df)


# 使用示例
def example_usage():
    """
    示例：如何使用BPC策略V2
    """
    # 假设df是已计算好特征的数据框
    # df = load_and_calculate_features(...)  # 不包含数学特征用于Gate/Evidence

    # 策略评估
    # approved, params = run_bpc_strategy_v2(df)

    # if approved:
    #     print(f"交易获批，证据分数: {params['evidence_score']:.3f}")
    #     print(f"噪声惩罚: {params['noise_penalty']:.3f}")
    #     print(f"执行参数: SL={params['execution_params'].sl_r}R, "
    #           f"TP={params['execution_params'].tp_r}R, "
    #           f"Size={params['execution_params'].size_multiplier}x")
    # else:
    #     print("交易被拒绝")

    pass
