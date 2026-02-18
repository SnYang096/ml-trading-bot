"""LivePCM — Live Portfolio Control Manager (Regime-Aware + Override)

多 archetype 信号仲裁层，支持三层控制框架。

三层控制框架（来自 slot 分配文档）:
  Layer 1: 静态资金结构 — 固定 budget，不随 regime 变化
  Layer 2: 优先级动态 — Regime 检测驱动优先级切换
     - NORMAL:        BPC > ME > FER > LV  (常态)
     - HIGH_VOL:      ME > BPC > FER > LV  (高波动扩张)
     - HIGH_LEVERAGE:  LV > FER > ME > BPC  (高杠杆脆弱)
  Layer 3: Override（极端信号覆盖）— 特定条件下允许跨层覆盖
     - LV 覆盖所有: 非线性事件最高抢占权
     - ME 覆盖 BPC: 强动能机会
     - FER 覆盖 ME: 失败/反转信号

优先级哲学:
  系统默认：BPC > ME | 强动能机会：ME > BPC
  极端事件：LV > ALL | 失败信号：FER 覆盖 ME

单策略时行为等价于直接挂 GenericLiveStrategy（零额外开销）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import yaml

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
# Regime Detector
# ────────────────────────────────────────────────────

# 3 个 Regime
REGIME_NORMAL = "NORMAL"
REGIME_HIGH_VOL = "HIGH_VOL"
REGIME_HIGH_LEVERAGE = "HIGH_LEVERAGE"

# 每个 Regime 的默认优先级（可被 YAML 覆盖）
DEFAULT_REGIME_PRIORITIES = {
    REGIME_NORMAL: ["BPC", "ME", "FER", "LV"],
    REGIME_HIGH_VOL: ["ME", "BPC", "FER", "LV"],
    REGIME_HIGH_LEVERAGE: ["LV", "FER", "ME", "BPC"],
}

# 默认检测阈值
DEFAULT_DETECTION = {
    REGIME_HIGH_LEVERAGE: {
        "conditions": [
            {"feature": "oi_zscore", "operator": ">", "threshold": 1.5},
            {"feature": "funding_rate_abs_zscore", "operator": ">", "threshold": 2.0},
        ],
        "logic": "AND",
    },
    REGIME_HIGH_VOL: {
        "conditions": [
            {"feature": "atr_percentile", "operator": ">", "threshold": 0.7},
        ],
        "logic": "AND",
    },
}


class RegimeDetector:
    """极简 Regime 状态机 (< 40 行核心逻辑)

    检测顺序: HIGH_LEVERAGE → HIGH_VOL → NORMAL (严格条件优先)
    防抖: min_bars_in_regime 根 bar 内不允许切换
    """

    def __init__(
        self,
        regime_priorities: Optional[Dict[str, List[str]]] = None,
        detection: Optional[Dict[str, Dict]] = None,
        min_bars_in_regime: int = 3,
    ):
        self._regime_priorities = regime_priorities or dict(DEFAULT_REGIME_PRIORITIES)
        self._detection = detection or dict(DEFAULT_DETECTION)
        self._min_bars = min_bars_in_regime

        # 状态
        self._current_regime: str = REGIME_NORMAL
        self._bars_in_current: int = 0

        # 统计
        self._regime_history: List[Tuple[str, str]] = []  # (regime, symbol)
        self._switch_count: int = 0

    @property
    def current_regime(self) -> str:
        return self._current_regime

    @property
    def switch_count(self) -> int:
        return self._switch_count

    @property
    def current_priority(self) -> List[str]:
        return list(
            self._regime_priorities.get(
                self._current_regime,
                DEFAULT_REGIME_PRIORITIES[REGIME_NORMAL],
            )
        )

    def detect(self, features: Dict[str, Any]) -> str:
        """根据特征检测当前 regime，带防抖。

        Returns:
            当前 regime 名称
        """
        raw_regime = self._raw_detect(features)

        self._bars_in_current += 1

        if raw_regime != self._current_regime:
            if self._bars_in_current >= self._min_bars:
                old = self._current_regime
                self._current_regime = raw_regime
                self._bars_in_current = 0
                self._switch_count += 1
                logger.info(
                    "PCM Regime: %s → %s (switch #%d)",
                    old,
                    raw_regime,
                    self._switch_count,
                )
            # else: 防抖中，保持原 regime

        self._regime_history.append((self._current_regime, ""))
        return self._current_regime

    def _raw_detect(self, features: Dict[str, Any]) -> str:
        """无防抖的纯检测逻辑。检测顺序: HIGH_LEVERAGE → HIGH_VOL → NORMAL"""
        # 优先检测 HIGH_LEVERAGE（最严格条件）
        for regime_name in [REGIME_HIGH_LEVERAGE, REGIME_HIGH_VOL]:
            det = self._detection.get(regime_name)
            if det and self._check_conditions(features, det):
                return regime_name
        return REGIME_NORMAL

    @staticmethod
    def _check_conditions(features: Dict[str, Any], det: Dict) -> bool:
        """检查一组条件是否满足"""
        conditions = det.get("conditions", [])
        logic = det.get("logic", "AND").upper()

        results = []
        for cond in conditions:
            feat_name = cond["feature"]
            op = cond["operator"]
            thr = float(cond["threshold"])
            val = features.get(feat_name)

            if val is None:
                results.append(False)
                continue

            try:
                val = float(val)
            except (ValueError, TypeError):
                results.append(False)
                continue

            if op == ">":
                results.append(val > thr)
            elif op == ">=":
                results.append(val >= thr)
            elif op == "<":
                results.append(val < thr)
            elif op == "<=":
                results.append(val <= thr)
            else:
                results.append(False)

        if not results:
            return False
        if logic == "AND":
            return all(results)
        return any(results)  # OR

    def reset(self) -> None:
        """重置状态（用于回测分段）"""
        self._current_regime = REGIME_NORMAL
        self._bars_in_current = 0
        self._switch_count = 0
        self._regime_history.clear()


def load_regime_config(
    config_path: str = "config/pcm_regime.yaml",
) -> Dict[str, Any]:
    """加载 PCM regime 配置文件"""
    p = Path(config_path)
    if not p.exists():
        logger.warning("PCM regime config not found: %s, using defaults", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_regime_detector(cfg: Dict[str, Any]) -> RegimeDetector:
    """从配置字典创建 RegimeDetector（内部辅助）"""
    if not cfg:
        return RegimeDetector()

    priorities = {}
    for regime_name, regime_cfg in cfg.get("regimes", {}).items():
        if "priority" in regime_cfg:
            priorities[regime_name] = regime_cfg["priority"]

    detection = cfg.get("detection", {})
    min_bars = cfg.get("min_bars_in_regime", 3)

    return RegimeDetector(
        regime_priorities=priorities or None,
        detection=detection or None,
        min_bars_in_regime=min_bars,
    )


def create_regime_detector_from_config(
    config_path: str = "config/pcm_regime.yaml",
) -> RegimeDetector:
    """从 YAML 配置创建 RegimeDetector"""
    cfg = load_regime_config(config_path)
    return _build_regime_detector(cfg)


# ────────────────────────────────────────────────────
# LivePCM
# ────────────────────────────────────────────────────

# 向后兼容：旧的固定优先级常量
DEFAULT_ARCHETYPE_PRIORITY = ["BPC", "ME", "FER", "LV"]


class LivePCM:
    """
    Live Portfolio Control Manager (Regime-Aware)

    职责:
      1. 注册多个策略（BPC, ME, FER, LV），每个策略实现 decide() 接口
      2. Regime 检测 → 动态切换优先级
      3. 同 symbol 不同 archetype 同时触发 → 当前 regime 优先级选最高
      4. 同优先级比 Evidence Score（高的优先）
      5. 跨 symbol slot 控制（通过 open_slot_count 回调）

    用法:
        pcm = LivePCM(max_slots=2)  # 使用默认 regime 配置
        pcm.register('bpc', bpc_strategy)
        pcm.register('me', me_strategy)
        pcm.register('fer', fer_strategy)
        pcm.register('lv', lv_strategy)

        # 替代: listener.decision_handler = bpc_strategy
        listener.decision_handler = pcm
    """

    def __init__(
        self,
        archetype_priority: Optional[List[str]] = None,
        max_slots: int = 2,
        get_open_slot_count: Optional[callable] = None,
        regime_detector: Optional[RegimeDetector] = None,
        regime_config_path: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            archetype_priority: archetype 静态优先级列表（索引越小优先级越高）。
                如果同时提供 regime_detector，此参数被忽略。
                默认: ['BPC', 'ME', 'FER', 'LV']
            max_slots: 最大同时持仓 slot 数
            get_open_slot_count: 可选回调，返回当前已占用 slot 数
            regime_detector: 可选 RegimeDetector 实例。提供后启用动态优先级。
            regime_config_path: 可选 YAML 配置路径。提供后自动创建 RegimeDetector + 加载 Override。
            override_config: Layer 3 Override 配置。显式传入时优先于 YAML。
        """
        self._strategies: Dict[str, DecisionHandler] = {}
        self._max_slots = max_slots
        self._get_open_slot_count = get_open_slot_count

        # Layer 3: Override 配置
        self._override_config: Dict[str, Any] = override_config or {}

        # Regime detector + override from config
        if regime_detector is not None:
            self._regime_detector = regime_detector
        elif regime_config_path is not None:
            cfg = load_regime_config(regime_config_path)
            self._regime_detector = _build_regime_detector(cfg)
            # 同时加载 override 配置（除非显式提供）
            if not self._override_config:
                self._override_config = cfg.get("override", {})
        else:
            self._regime_detector = None

        # 静态优先级（当无 regime detector 时使用）
        self._archetype_priority = archetype_priority or list(
            DEFAULT_ARCHETYPE_PRIORITY
        )

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
        """\u5f53\u524d\u6709\u6548\u4f18\u5148\u7ea7\u5217\u8868\uff08\u5982\u6709 regime detector \u5219\u8fd4\u56de\u52a8\u6001\u4f18\u5148\u7ea7\uff09"""
        if self._regime_detector is not None:
            return self._regime_detector.current_priority
        return list(self._archetype_priority)

    @property
    def regime_detector(self) -> Optional[RegimeDetector]:
        return self._regime_detector

    @property
    def current_regime(self) -> str:
        if self._regime_detector is not None:
            return self._regime_detector.current_regime
        return REGIME_NORMAL

    # ── 核心决策接口 ──

    def _get_priority_rank(self, archetype: str) -> int:
        """获取 archetype 的优先级排名（越小越优先）。
        如有 regime detector，使用动态优先级。
        不在列表中的 archetype 排到最后。"""
        priority = self.archetype_priority
        arch_lower = archetype.lower()
        for i, a in enumerate(priority):
            if a.lower() == arch_lower:
                return i
        return len(priority)  # 未知 archetype 排最后

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TradeIntent]:
        """
        多策略仲裁 → 返回最优 TradeIntent (三层控制)

        算法:
          1. Regime 检测 (Layer 2) → 动态切换优先级
          2. 遍历所有注册策略，收集候选 TradeIntent
          3. 单候选 → 快速路径
          4. Override 检查 (Layer 3) → 极端信号覆盖
          5. 多候选 → 按当前 regime 优先级 + Evidence 排序
          6. 跨 symbol slot 检查

        Returns:
            List[TradeIntent]（0 或 1 个元素）
        """
        if not self._strategies:
            return []

        # ── 1. Regime 检测 (Layer 2) ──
        if self._regime_detector is not None:
            self._regime_detector.detect(features)

        # ── 2. 收集所有策略的候选信号 ──
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

        # ── 3. 快速路径：单候选直接返回 ──
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

        regime_str = f" [regime={self.current_regime}]" if self._regime_detector else ""

        # ── 4. Layer 3: Override 检查（极端信号覆盖）──
        override_winner = self._check_override(all_intents, features)
        if override_winner is not None:
            ev = (
                override_winner.confidence
                if override_winner.confidence is not None
                else 0.5
            )
            if not self._slot_available(symbol, override_winner.archetype):
                logger.info(
                    "PCM: %s slot 已满 (%d/%d)，拒绝 Override 优胜 %s (evidence=%.2f)%s",
                    symbol,
                    self._current_slot_count(),
                    self._max_slots,
                    override_winner.archetype,
                    ev,
                    regime_str,
                )
                return []
            logger.info(
                "PCM: %s Override 选中 %s (evidence=%.2f)%s",
                symbol,
                override_winner.archetype,
                ev,
                regime_str,
            )
            return [override_winner]

        # ── 5. 多候选：动态优先级 + Evidence 排序 (Layer 2) ──
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
                "PCM: %s/%s priority=%d evidence=%.2f%s",
                symbol,
                intent.archetype,
                r,
                ev,
                regime_str,
            )

        # ── 6. Slot 检查 ──
        if not self._slot_available(symbol, best_intent.archetype):
            logger.info(
                "PCM: %s slot 已满 (%d/%d)，拒绝优先级最高 %s (evidence=%.2f)%s",
                symbol,
                self._current_slot_count(),
                self._max_slots,
                best_intent.archetype,
                evidence,
                regime_str,
            )
            return []

        logger.info(
            "PCM: %s 选中 %s (priority=%d, evidence=%.2f)%s",
            symbol,
            best_intent.archetype,
            rank,
            evidence,
            regime_str,
        )
        return [best_intent]

    # ── Layer 3: Override 极端信号覆盖 ──

    def _check_override(
        self,
        intents: List[TradeIntent],
        features: Dict[str, Any],
    ) -> Optional[TradeIntent]:
        """Layer 3: 极端信号覆盖

        文档精神:
          - LV 属于非线性事件，应有最高抢占权 → LV 覆盖所有
          - 强动能机会 → ME 覆盖 BPC
          - 失败/反转信号 → FER 覆盖 ME

        检查顺序按抢占权高低: LV → FER → ME

        Returns:
            覆盖后的优胜 TradeIntent，或 None（无覆盖触发）
        """
        if not self._override_config:
            return None

        # 构建 archetype → intent 映射（大写键，保留最高 evidence）
        intent_map: Dict[str, TradeIntent] = {}
        for intent in intents:
            key = intent.archetype.upper()
            existing = intent_map.get(key)
            if existing is None or (intent.confidence or 0) > (
                existing.confidence or 0
            ):
                intent_map[key] = intent

        # 按抢占权顺序检查: LV > FER > ME
        for arch_name in ["LV", "FER", "ME"]:
            rule = self._override_config.get(arch_name)
            if rule is None:
                continue

            if arch_name not in intent_map:
                continue  # 该 archetype 未触发信号

            candidate = intent_map[arch_name]
            evidence = candidate.confidence if candidate.confidence is not None else 0.5

            # 检查最低 evidence 阈值
            min_ev = float(rule.get("min_evidence", 0.0))
            if evidence < min_ev:
                continue

            # 检查额外特征条件
            conditions = rule.get("conditions", [])
            if conditions:
                det = {"conditions": conditions, "logic": rule.get("logic", "AND")}
                if not RegimeDetector._check_conditions(features, det):
                    continue

            # 检查覆盖目标
            overrides_target = rule.get("overrides", [])
            if isinstance(overrides_target, str) and overrides_target.upper() == "ALL":
                # 覆盖所有：只要该 archetype 触发就赢
                logger.info(
                    "PCM Override: %s 覆盖所有 (evidence=%.2f)",
                    arch_name,
                    evidence,
                )
                return candidate
            else:
                # 覆盖特定 archetype
                if isinstance(overrides_target, str):
                    targets = [overrides_target.upper()]
                else:
                    targets = [t.upper() for t in overrides_target]

                # 只有当被覆盖目标也在候选中时才触发
                if any(t in intent_map for t in targets):
                    logger.info(
                        "PCM Override: %s 覆盖 %s (evidence=%.2f)",
                        arch_name,
                        ", ".join(targets),
                        evidence,
                    )
                    return candidate

        return None

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
        兼容 GenericLiveStrategy 接口。
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

    # ── PCM 统计 ──

    def get_stats(self) -> Dict[str, Any]:
        """获取 PCM 运行统计信息（用于 KPI 评估）"""
        stats: Dict[str, Any] = {
            "registered_archetypes": self.registered_archetypes,
            "current_priority": self.archetype_priority,
            "max_slots": self._max_slots,
            "override_enabled": bool(self._override_config),
            "override_rules": (
                list(self._override_config.keys()) if self._override_config else []
            ),
        }
        if self._regime_detector is not None:
            stats["current_regime"] = self._regime_detector.current_regime
            stats["regime_switch_count"] = self._regime_detector.switch_count
        return stats


def create_live_pcm(
    archetype_priority: Optional[List[str]] = None,
    max_slots: int = 2,
    get_open_slot_count: Optional[callable] = None,
    regime_config_path: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None,
) -> LivePCM:
    """
    创建 LivePCM 实例

    Args:
        archetype_priority: 静态优先级列表，默认 ['BPC', 'ME', 'FER', 'LV']。
            如果同时提供 regime_config_path，静态优先级被忽略。
        max_slots: 最大 slot 数
        get_open_slot_count: 可选回调
        regime_config_path: 可选 YAML 配置路径，提供后启用动态 regime + override
        override_config: Layer 3 Override 配置（显式传入，优先于 YAML）

    Returns:
        初始化好的 LivePCM（尚未注册策略）
    """
    return LivePCM(
        archetype_priority=archetype_priority,
        max_slots=max_slots,
        get_open_slot_count=get_open_slot_count,
        regime_config_path=regime_config_path,
        override_config=override_config,
    )
