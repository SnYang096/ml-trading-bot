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
    4. 实时 bar 累积后自动升级到 NORMAL（策略B）
    """
    
    # 数据完整性阈值
    MIN_BARS_FOR_NORMAL = 240    # 4小时 = 240杨1min bar
    MIN_BARS_FOR_DEGRADED = 120  # 2小时 = 120杨1min bar
    MAX_GAP_MINUTES = 5          # 最大允许缺口（分钟）
    ABUNDANT_DATA_THRESHOLD = 2400  # 充足数据阈值 (40h)，超过则容忍尾部 gap
    
    def __init__(self):
        self.current_mode = SystemMode.OFFLINE
        self.mode_history = []
        
        # 自动升级：实时 bar 累积计数
        # 策略B：即使 warmup 有 gap，累积足够实时 1min bar 后自动升级
        self._realtime_bar_count = 0
        self._auto_upgrade_logged = False
    
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
        
        elif bar_count >= self.ABUNDANT_DATA_THRESHOLD:
            # 充足历史数据 (≥40h) — 尾部 gap 不影响特征计算，直接 NORMAL
            # 理由: 200+ 天历史数据已充分初始化特征滚动窗口，
            #   1-2 天尾部 gap 不影响交易决策，实时数据累积后特征完全刷新
            gap_summary = ""
            if has_large_gap:
                largest_gap = max(missing_periods, key=lambda x: x["minutes"])
                gap_summary = f", largest_gap={largest_gap['minutes']:.0f}min (tolerated)"
            return ModeDecision(
                mode=SystemMode.NORMAL,
                reason=f"Abundant data: {bar_count} bars (>= {self.ABUNDANT_DATA_THRESHOLD}), "
                       f"{coverage_hours:.1f}h coverage{gap_summary}",
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
    
    def on_realtime_bar(self) -> bool:
        """收到一条实时 1min bar 时调用
        
        策略B 自动升级逻辑：
        - warmup 已把 6 个月历史 ticks 喂入 IncrementalFeatureComputer
        - 即使 warmup 有几小时 gap，滚动窗口状态已经建好
        - 累积到 MIN_BARS_FOR_NORMAL 条实时 bar 后，特征完全刷新，可安全交易
        - BPC/FER (4H): 需要 240 条 1min = 1 个完整 4H bar
        - ME (1H): 只需 60 条，但为安全统一等 240 条
        
        Returns:
            True: 发生了模式升级
        """
        self._realtime_bar_count += 1
        
        # 已经是 NORMAL，无需升级
        if self.current_mode == SystemMode.NORMAL:
            return False
        
        # 每 60 条 bar (1h) 打一次进度日志
        if self._realtime_bar_count % 60 == 0:
            remaining = max(0, self.MIN_BARS_FOR_NORMAL - self._realtime_bar_count)
            logger.info(
                f"📊 实时 bar 累积: {self._realtime_bar_count}/{self.MIN_BARS_FOR_NORMAL}"
                f" (剩余 {remaining} 条 ≈ {remaining}min 后可升级 NORMAL)"
            )
        
        # OFFLINE → DEGRADED (120 bars = 2h)
        if self.current_mode == SystemMode.OFFLINE and self._realtime_bar_count >= self.MIN_BARS_FOR_DEGRADED:
            decision = ModeDecision(
                mode=SystemMode.DEGRADED,
                reason=f"Auto-upgrade: {self._realtime_bar_count} realtime bars accumulated (>= {self.MIN_BARS_FOR_DEGRADED})",
                bar_count=self._realtime_bar_count,
                data_coverage_hours=self._realtime_bar_count / 60,
            )
            self.set_mode(decision)
            return True
        
        # DEGRADED → NORMAL (240 bars = 4h)
        if self.current_mode == SystemMode.DEGRADED and self._realtime_bar_count >= self.MIN_BARS_FOR_NORMAL:
            decision = ModeDecision(
                mode=SystemMode.NORMAL,
                reason=f"Auto-upgrade: {self._realtime_bar_count} realtime bars accumulated (>= {self.MIN_BARS_FOR_NORMAL}), trading enabled",
                bar_count=self._realtime_bar_count,
                data_coverage_hours=self._realtime_bar_count / 60,
            )
            self.set_mode(decision)
            logger.info("🟢 System auto-upgraded to NORMAL — trading is now ENABLED")
            return True
        
        return False
    
    def get_realtime_bar_count(self) -> int:
        """获取实时 bar 累积数"""
        return self._realtime_bar_count
