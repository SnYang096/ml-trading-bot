"""系统运行模式管理

支持三种运行模式：
- NORMAL: 正常交易模式
- DEGRADED: 降级模式（只观察不交易）
- OFFLINE: 离线模式（拒绝启动）
"""

from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """系统运行模式"""
    NORMAL = "NORMAL"       # 正常交易
    DEGRADED = "DEGRADED"   # 降级模式（只观察）
    OFFLINE = "OFFLINE"     # 离线模式（拒绝启动）


class ModeDecision:
    """模式决策结果"""
    
    def __init__(
        self,
        mode: SystemMode,
        reason: str,
        bar_count: int,
        data_coverage_hours: float,
        missing_periods: Optional[list] = None,
    ):
        self.mode = mode
        self.reason = reason
        self.bar_count = bar_count
        self.data_coverage_hours = data_coverage_hours
        self.missing_periods = missing_periods or []
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "bar_count": self.bar_count,
            "data_coverage_hours": self.data_coverage_hours,
            "missing_periods": self.missing_periods,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def __repr__(self) -> str:
        return (
            f"ModeDecision(mode={self.mode.value}, "
            f"bar_count={self.bar_count}, "
            f"coverage={self.data_coverage_hours:.2f}h, "
            f"reason='{self.reason}')"
        )


class SystemModeManager:
    """系统模式管理器
    
    职责：
    1. 根据warmup数据质量决定启动模式
    2. 管理模式切换逻辑
    3. 记录模式切换历史
    """
    
    # 数据完整性阈值
    MIN_BARS_FOR_NORMAL = 240    # 4小时 = 240条1min bar
    MIN_BARS_FOR_DEGRADED = 120  # 2小时 = 120条1min bar
    MAX_GAP_MINUTES = 5          # 最大允许缺口（分钟）
    
    def __init__(self):
        self.current_mode = SystemMode.OFFLINE
        self.mode_history = []
    
    def decide_mode(
        self,
        warmup_data: Dict[str, Any],
    ) -> ModeDecision:
        """根据warmup数据决定运行模式
        
        Args:
            warmup_data: warmup返回的数据字典，包含：
                - ticks_1min: 1分钟tick数据DataFrame
                - features_15min: 15分钟特征DataFrame（可选）
                - features_4h: 4小时特征DataFrame（可选）
        
        Returns:
            ModeDecision对象，包含模式和原因
        """
        import pandas as pd
        
        ticks_1min = warmup_data.get("ticks_1min", pd.DataFrame())
        
        # 策略B：只用 ticks_1min 判定，不再从 features_4h 推算
        # 没有实时 ticks 累积时，直接返回 OFFLINE，等待累积
        if len(ticks_1min) == 0:
            return ModeDecision(
                mode=SystemMode.OFFLINE,
                reason="No ticks_1min data available (Strategy B: waiting for real-time accumulation)",
                bar_count=0,
                data_coverage_hours=0.0,
            )
        
        # 使用 ticks_1min 计算
        bar_count = len(ticks_1min)
        
        # 计算数据覆盖时长
        if "timestamp" in ticks_1min.columns:
            first_ts = pd.to_datetime(ticks_1min["timestamp"].iloc[0])
            last_ts = pd.to_datetime(ticks_1min["timestamp"].iloc[-1])
            coverage_hours = (last_ts - first_ts).total_seconds() / 3600
        else:
            # 如果没有timestamp列，使用bar数量估算（1min = 1bar）
            coverage_hours = bar_count / 60
        
        # 检查数据缺口
        missing_periods = self._detect_gaps(ticks_1min)
        has_large_gap = any(gap["minutes"] > self.MAX_GAP_MINUTES for gap in missing_periods)
        
        # 决策逻辑
        if bar_count < self.MIN_BARS_FOR_DEGRADED:
            # 数据 < 2小时 → OFFLINE
            return ModeDecision(
                mode=SystemMode.OFFLINE,
                reason=f"Insufficient data: {bar_count} bars < {self.MIN_BARS_FOR_DEGRADED} (2h minimum)",
                bar_count=bar_count,
                data_coverage_hours=coverage_hours,
                missing_periods=missing_periods,
            )
        
        elif bar_count < self.MIN_BARS_FOR_NORMAL or has_large_gap:
            # 数据 2-4小时 或有大缺口 → DEGRADED
            reason_parts = []
            if bar_count < self.MIN_BARS_FOR_NORMAL:
                reason_parts.append(f"{bar_count} bars < {self.MIN_BARS_FOR_NORMAL} (4h target)")
            if has_large_gap:
                largest_gap = max(missing_periods, key=lambda x: x["minutes"])
                reason_parts.append(f"Large gap detected: {largest_gap['minutes']:.1f}min")
            
            return ModeDecision(
                mode=SystemMode.DEGRADED,
                reason="Data incomplete: " + ", ".join(reason_parts),
                bar_count=bar_count,
                data_coverage_hours=coverage_hours,
                missing_periods=missing_periods,
            )
        
        else:
            # 数据 ≥ 4小时且无大缺口 → NORMAL
            return ModeDecision(
                mode=SystemMode.NORMAL,
                reason=f"Data complete: {bar_count} bars, {coverage_hours:.2f}h coverage",
                bar_count=bar_count,
                data_coverage_hours=coverage_hours,
                missing_periods=missing_periods,
            )
    
    def _detect_gaps(self, ticks_1min) -> list:
        """检测数据缺口
        
        Returns:
            缺口列表，每个缺口包含：start_time, end_time, minutes
        """
        import pandas as pd
        
        if len(ticks_1min) < 2 or "timestamp" not in ticks_1min.columns:
            return []
        
        gaps = []
        timestamps = pd.to_datetime(ticks_1min["timestamp"]).sort_values()
        
        for i in range(1, len(timestamps)):
            time_diff = (timestamps.iloc[i] - timestamps.iloc[i-1]).total_seconds() / 60
            
            # 如果时间差 > 1.5分钟，认为是缺口（考虑时间戳误差）
            if time_diff > 1.5:
                gaps.append({
                    "start_time": timestamps.iloc[i-1].isoformat(),
                    "end_time": timestamps.iloc[i].isoformat(),
                    "minutes": time_diff,
                })
        
        return gaps
    
    def set_mode(self, decision: ModeDecision) -> None:
        """设置系统模式
        
        Args:
            decision: 模式决策结果
        """
        old_mode = self.current_mode
        self.current_mode = decision.mode
        
        # 记录模式切换历史
        self.mode_history.append({
            "old_mode": old_mode.value,
            "new_mode": decision.mode.value,
            "timestamp": decision.timestamp.isoformat(),
            "reason": decision.reason,
        })
        
        if old_mode != decision.mode:
            logger.info(
                f"🔄 System mode changed: {old_mode.value} → {decision.mode.value}\n"
                f"   Reason: {decision.reason}\n"
                f"   Data: {decision.bar_count} bars, {decision.data_coverage_hours:.2f}h"
            )
        else:
            logger.info(f"✅ System mode confirmed: {decision.mode.value}")
    
    def get_current_mode(self) -> SystemMode:
        """获取当前模式"""
        return self.current_mode
    
    def is_trading_allowed(self) -> bool:
        """是否允许交易
        
        Returns:
            True: NORMAL模式，允许交易
            False: DEGRADED/OFFLINE模式，禁止交易
        """
        return self.current_mode == SystemMode.NORMAL
    
    def get_mode_history(self) -> list:
        """获取模式切换历史"""
        return self.mode_history.copy()
