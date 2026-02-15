"""LivePCM — Live Portfolio Control Manager

多 archetype 信号仲裁层。

实现 decide(*, features, symbol, bars=None) 接口，
对 OrderFlowListener 完全透明（drop-in replacement）。

仲裁算法:
  同 archetype 跨 symbol → 比 Evidence Score（高的优先）
  同 symbol 不同 archetype → 固定优先级（条件越严格越优先）
    默认: Reversal > ME > BPC
    决策依据: 按语义要求的条件严格性划分——
      Reversal 需要耗竭确认+反转信号（最严格）
      ME 需要动量爆发确认（中等）
      BPC 是最常见的压缩→突破→回踩（最宽松）
  跨 symbol slot 控制（可选）：超过 max_slots 时拒绝新入场

单策略时行为等价于直接挂 BPCLiveStrategy（零额外开销）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.time_series_model.core.trade_intent import TradeIntent

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────
# Strategy 接口协议（duck typing）
# ────────────────────────────────────────────────────


@runtime_checkable
class DecisionHandler(Protocol):
    """任何实现 decide(features, symbol) → List[TradeIntent] 的对象"""

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TradeIntent]:  # noqa: E704
        ...


# ────────────────────────────────────────────────────
# LivePCM
# ────────────────────────────────────────────────────


# 默认优先级：按条件严格性排序（越严格越优先）
DEFAULT_ARCHETYPE_PRIORITY = ["Reversal", "ME", "BPC"]


class LivePCM:
    """
    Live Portfolio Control Manager

    职责:
      1. 注册多个策略（BPC, ME, …），每个策略实现 decide() 接口
      2. 同 symbol 不同 archetype 同时触发 → 固定优先级选最高
      3. 同 archetype 跨 symbol → Evidence Score 高的优先
      4. 跨 symbol slot 控制（通过 open_slot_count 回调）

    用法:
        pcm = LivePCM(archetype_priority=['Reversal', 'ME', 'BPC'], max_slots=2)
        pcm.register('bpc', bpc_strategy)
        pcm.register('me', me_strategy)       # 未来

        # 替代: listener.decision_handler = bpc_strategy
        listener.decision_handler = pcm
    """

    def __init__(
        self,
        archetype_priority: Optional[List[str]] = None,
        max_slots: int = 2,
        get_open_slot_count: Optional[callable] = None,
    ):
        """
        Args:
            archetype_priority: archetype 固定优先级列表（索引越小优先级越高）
                默认: ['Reversal', 'ME', 'BPC']
                决策依据: 按语义要求的条件严格性划分
            max_slots: 最大同时持仓 slot 数
            get_open_slot_count: 可选回调，返回当前已占用 slot 数
                签名: () -> int
                用于跨 symbol slot 控制。不提供时不做跨 symbol 限制。
        """
        self._strategies: Dict[str, DecisionHandler] = {}
        self._archetype_priority = archetype_priority or list(
            DEFAULT_ARCHETYPE_PRIORITY
        )
        self._max_slots = max_slots
        self._get_open_slot_count = get_open_slot_count

    # ── 注册 / 管理 ──

    def register(self, archetype: str, strategy: DecisionHandler) -> None:
        """注册一个 archetype 策略实例"""
        self._strategies[archetype] = strategy
        logger.info(
            "PCM: 注册策略 archetype=%s (%s)", archetype, type(strategy).__name__
        )

    def unregister(self, archetype: str) -> None:
        """移除一个 archetype 策略"""
        self._strategies.pop(archetype, None)

    @property
    def registered_archetypes(self) -> List[str]:
        return list(self._strategies.keys())

    @property
    def archetype_priority(self) -> List[str]:
        return list(self._archetype_priority)

    # ── 核心决策接口 ──

    def _get_priority_rank(self, archetype: str) -> int:
        """获取 archetype 的优先级排名（越小越优先）。
        不在列表中的 archetype 排到最后。"""
        arch_lower = archetype.lower()
        for i, a in enumerate(self._archetype_priority):
            if a.lower() == arch_lower:
                return i
        return len(self._archetype_priority)  # 未知 archetype 排最后

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TradeIntent]:
        """
        多策略仲裁 → 返回最优 TradeIntent

        与 BPCLiveStrategy.decide() 签名完全一致，
        可直接赋给 OrderFlowListener.decision_handler。

        算法:
          1. 遍历所有注册策略，收集候选 TradeIntent
          2. 如果只有一个候选 → 直接返回（快速路径）
          3. 多个候选（同 symbol 不同 archetype）
             → 按固定优先级选最高（Reversal > ME > BPC）
          4. 跨 symbol slot 检查（如果配置了回调）

        Returns:
            List[TradeIntent]（0 或 1 个元素）
        """
        if not self._strategies:
            return []

        # ── 1. 收集所有策略的候选信号 ──
        all_intents: List[TradeIntent] = []
        for arch_name, strategy in self._strategies.items():
            try:
                intents = strategy.decide(features=features, symbol=symbol, bars=bars)
                all_intents.extend(intents)
            except Exception:
                logger.exception(
                    "PCM: 策略 %s 对 %s 调用 decide() 异常", arch_name, symbol
                )

        if not all_intents:
            return []

        # ── 2. 快速路径：单候选直接返回 ──
        if len(all_intents) == 1:
            intent = all_intents[0]
            if not self._slot_available(symbol, intent.archetype):
                logger.info(
                    "PCM: %s slot 已满 (%d/%d)，拒绝 %s",
                    symbol,
                    self._current_slot_count(),
                    self._max_slots,
                    intent.archetype,
                )
                return []
            return [intent]

        # ── 3. 多候选：固定优先级 + Evidence 排序 ──
        # 排序键: (priority_rank, -evidence)  → priority_rank 越小越好，evidence 越大越好
        def _sort_key(intent: TradeIntent):
            rank = self._get_priority_rank(intent.archetype)
            evidence = intent.confidence if intent.confidence is not None else 0.5
            return (rank, -evidence)

        best_intent = min(all_intents, key=_sort_key)

        evidence = best_intent.confidence if best_intent.confidence is not None else 0.5
        rank = self._get_priority_rank(best_intent.archetype)

        for intent in all_intents:
            ev = intent.confidence if intent.confidence is not None else 0.5
            r = self._get_priority_rank(intent.archetype)
            logger.debug(
                "PCM: %s/%s priority=%d evidence=%.2f",
                symbol,
                intent.archetype,
                r,
                ev,
            )

        # ── 4. Slot 检查 ──
        if not self._slot_available(symbol, best_intent.archetype):
            logger.info(
                "PCM: %s slot 已满 (%d/%d)，拒绝优先级最高 %s (evidence=%.2f)",
                symbol,
                self._current_slot_count(),
                self._max_slots,
                best_intent.archetype,
                evidence,
            )
            return []

        logger.info(
            "PCM: %s 选中 %s (priority=%d, evidence=%.2f)",
            symbol,
            best_intent.archetype,
            rank,
            evidence,
        )
        return [best_intent]

    # ── 内部方法 ──

    def _current_slot_count(self) -> int:
        """当前已占用 slot 数"""
        if self._get_open_slot_count is not None:
            return self._get_open_slot_count()
        return 0

    def _slot_available(self, symbol: str, archetype: str) -> bool:
        """检查是否有可用 slot"""
        if self._get_open_slot_count is None:
            return True  # 未配置回调，不做跨 symbol 限制
        return self._current_slot_count() < self._max_slots

    # ── Quantiles 透传 ──

    def set_quantiles(self, features_df) -> None:
        """
        将 quantiles 设置给所有注册的策略（如果策略支持）。
        用于 warmup 后的 evidence quantiles 计算。
        """
        for arch_name, strategy in self._strategies.items():
            if hasattr(strategy, "set_quantiles"):
                strategy.set_quantiles(features_df)
                logger.info("PCM: 已设置 quantiles 给 %s", arch_name)

    def set_quantiles_from_df(self, features_df) -> None:
        """
        透传给内部策略的 set_quantiles_from_df()。
        兼容 BPCLiveStrategy 接口。
        """
        for arch_name, strategy in self._strategies.items():
            if hasattr(strategy, "set_quantiles_from_df"):
                strategy.set_quantiles_from_df(features_df)
                logger.info("PCM: 已设置 quantiles_from_df 给 %s", arch_name)
            elif hasattr(strategy, "set_quantiles"):
                strategy.set_quantiles(features_df)
                logger.info("PCM: 已设置 quantiles 给 %s", arch_name)

    # ── 配置加载 ──

    def load_all_configs(self) -> None:
        """调用所有注册策略的 load_configs()"""
        for arch_name, strategy in self._strategies.items():
            if hasattr(strategy, "load_configs"):
                strategy.load_configs()
                logger.info("PCM: 已加载配置给 %s", arch_name)


def create_live_pcm(
    archetype_priority: Optional[List[str]] = None,
    max_slots: int = 2,
    get_open_slot_count: Optional[callable] = None,
) -> LivePCM:
    """
    创建 LivePCM 实例

    Args:
        archetype_priority: 优先级列表，默认 ['Reversal', 'ME', 'BPC']
        max_slots: 最大 slot 数
        get_open_slot_count: 可选回调

    Returns:
        初始化好的 LivePCM（尚未注册策略）
    """
    return LivePCM(
        archetype_priority=archetype_priority,
        max_slots=max_slots,
        get_open_slot_count=get_open_slot_count,
    )
