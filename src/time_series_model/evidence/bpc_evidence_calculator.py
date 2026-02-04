"""
BPC证据计算器

根据路径2.5规范：
- Evidence层：❌ 禁止使用execution_noise_penalty或原始数学特征
- Evidence层：✅ 仅基于structure/orderflow/regime评估alpha质量
- Evidence层：✅ 保持纯粹的alpha质量评估
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np


class BPCEvidenceCalculator:
    """BPC证据计算器 - 仅基于结构和订单流特征"""

    def __init__(self):
        # BPC核心证据特征权重
        self.feature_weights = {
            # BPC阶段分数权重最高
            "bpc_score_breakout": 0.25,
            "bpc_score_pullback": 0.25,
            "bpc_score_continuation": 0.25,
            # 结构特征
            "bpc_pullback_depth_pct": 0.05,
            "bpc_impulse_return_atr": 0.05,
            "bpc_dir_consistency_short": 0.03,
            "bpc_dir_consistency_mid": 0.03,
            "bpc_dir_consistency_long": 0.03,
            # 订单流特征
            "cvd_divergence_score": 0.03,
            "price_momentum_divergence": 0.03,
            "bpc_pullback_delta_absorption": 0.03,
            "cvd_change_5_pct": 0.02,
            # 趋势强度
            "trend_r2_20": 0.02,
            "path_efficiency_pct": 0.02,
            "price_dir_consistency_pct": 0.02,
            # 动量指标
            "macd_atr": 0.02,
            "rsi_normalized": 0.02,
            # 波动率相关
            "atr_percentile": 0.02,
            "bb_width_normalized_pct": 0.02,
            # 订单流指标
            "vpin_score": 0.02,
            "volume_ratio_pct": 0.02,
            "ofci_pct": 0.02,
            "shd_pct": 0.02,
            # 市场状态
            "vol_regime_score": 0.01,
            "vol_trend_score": 0.01,
            "sr_strength_max": 0.01,
        }

    def calculate_evidence_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算BPC证据分数（仅基于结构/订单流/规制特征）

        Args:
            df: 包含BPC相关特征的DataFrame

        Returns:
            证据分数序列（0-1之间）
        """
        # 验证必需的特征列
        required_features = []
        for feature in self.feature_weights.keys():
            if feature in df.columns:
                required_features.append(feature)

        if not required_features:
            raise ValueError("未找到任何必需的BPC特征")

        # 标准化各个特征（归一化到0-1）
        normalized_features = {}
        for feature in required_features:
            series = df[feature].fillna(0.0)
            # 使用min-max归一化到[0,1]区间
            min_val = series.min()
            max_val = series.max()
            if max_val != min_val:
                normalized_series = (series - min_val) / (max_val - min_val)
            else:
                normalized_series = pd.Series([0.5] * len(series), index=series.index)

            normalized_features[feature] = normalized_series

        # 加权求和
        weighted_sum = pd.Series([0.0] * len(df), index=df.index)
        total_weight = 0.0

        for feature, weight in self.feature_weights.items():
            if feature in normalized_features:
                weighted_sum += normalized_features[feature] * weight
                total_weight += weight

        # 归一化到[0,1]区间
        evidence_score = (
            weighted_sum / total_weight
            if total_weight > 0
            else pd.Series([0.0] * len(df), index=df.index)
        )

        # 钳制到[0,1]区间
        evidence_score = evidence_score.clip(lower=0.0, upper=1.0)

        return evidence_score

    def get_feature_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取各特征对证据分数的贡献度

        Args:
            df: 包含BPC相关特征的DataFrame

        Returns:
            各特征贡献度DataFrame
        """
        required_features = []
        for feature in self.feature_weights.keys():
            if feature in df.columns:
                required_features.append(feature)

        contributions = pd.DataFrame(index=df.index)

        for feature in required_features:
            series = df[feature].fillna(0.0)
            # 标准化到[0,1]
            min_val = series.min()
            max_val = series.max()
            if max_val != min_val:
                normalized_series = (series - min_val) / (max_val - min_val)
            else:
                normalized_series = pd.Series([0.5] * len(series), index=series.index)

            weight = self.feature_weights.get(feature, 0.0)
            contributions[f"{feature}_contribution"] = normalized_series * weight

        return contributions


def calculate_bpc_evidence_score(df: pd.DataFrame) -> pd.Series:
    """
    便捷函数：计算BPC证据分数
    """
    calculator = BPCEvidenceCalculator()
    return calculator.calculate_evidence_score(df)
