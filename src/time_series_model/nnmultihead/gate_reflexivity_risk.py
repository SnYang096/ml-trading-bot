"""
Gate规则中的反身性风险控制（分级响应）

实现文档要求的分级响应规则：
1. SHD: hard veto (shd_p > 0.9 → DENY)
2. LFI: soft veto (lfi_p > 0.9 → 限制仓位 max_position *= 0.3)
3. OFCI: 最soft (|ofci_p| > 0.9 → 降低aggressiveness max_position *= 0.6)

注意：由于Gate规则系统目前只支持hard veto，这个函数主要用于：
1. 在PCM层或Execution层应用position multiplier
2. 在Gate规则评估后，提供额外的反身性风险控制
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional


def gate_reflexivity_risk(
    features: Dict[str, Any],
) -> Tuple[bool, float, str]:
    """
    反身性风险 Gate 规则（分级响应）

    根据文档要求实现三档响应：
    1. SHD: hard veto (最高优先级)
    2. LFI: soft veto (限制仓位)
    3. OFCI: 最soft (降低aggressiveness)

    Args:
        features: 包含 ofci_p, lfi_p, shd_p 的特征字典（已归一化为分位数）
                 - ofci_p: [0, 1] 的percentile值（基于abs(ofci)计算）
                 - lfi_p: [0, 1] 的percentile值（LFI的percentile rank）
                 - shd_p: [0, 1] 的percentile值（SHD的percentile rank）

    Returns:
        (allow: bool, max_position_multiplier: float, reason: str)
        - allow: 是否允许交易（False表示hard veto）
        - max_position_multiplier: 仓位倍数 [0, 1]，用于调整最大仓位
        - reason: 原因说明
    """
    max_position_multiplier = 1.0

    # 规则1：SHD - 最高优先级，可以 hard veto
    shd_p = features.get("shd_p", 0.0)
    if shd_p > 0.9:
        return False, 0.0, "strategy_homogeneity: 策略同质化严重，反身性风险极高"

    # 规则2：LFI - 拒绝大仓 / 拒绝加仓（soft veto）
    # 注意：LFI需要订单簿数据，当前未实现，但保留接口
    lfi_p = features.get("lfi_p", 0.0)
    if lfi_p > 0.9:
        max_position_multiplier *= 0.3
        return True, max_position_multiplier, "fragile_liquidity: 流动性脆弱，限制仓位"

    # 规则3：OFCI - 只影响 aggressiveness（最 soft）
    # 注意：ofci_p已经是基于abs(ofci)计算的percentile，所以直接比较即可
    ofci_p = features.get("ofci_p", 0.0)
    if ofci_p > 0.9:
        max_position_multiplier *= 0.6
        return (
            True,
            max_position_multiplier,
            "high_consensus: 市场方向高度一致，降低 aggressiveness",
        )

    return True, max_position_multiplier, "reflexivity_risk_acceptable"


def apply_reflexivity_position_scaling(
    base_position: float,
    features: Dict[str, Any],
) -> Tuple[float, str]:
    """
    应用反身性风险控制的仓位调整

    这是一个辅助函数，用于在PCM层或Execution层应用反身性风险控制。

    Args:
        base_position: 基础仓位大小
        features: 包含反身性特征的特征字典

    Returns:
        (adjusted_position: float, reason: str)
    """
    allow, multiplier, reason = gate_reflexivity_risk(features)

    if not allow:
        # Hard veto: 返回0仓位
        return 0.0, reason

    # Soft veto: 应用仓位倍数
    adjusted_position = base_position * multiplier
    return adjusted_position, reason


def check_reflexivity_hard_veto(
    features: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    检查是否应该hard veto（仅检查SHD）

    这是一个快速检查函数，用于在Gate规则评估前快速判断是否需要hard veto。

    Args:
        features: 包含 shd_p 的特征字典

    Returns:
        (should_veto: bool, reason: Optional[str])
    """
    shd_p = features.get("shd_p", 0.0)
    if shd_p > 0.9:
        return True, "strategy_homogeneity: 策略同质化严重，反身性风险极高"
    return False, None
