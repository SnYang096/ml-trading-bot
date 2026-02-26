"""LivePCM — Live Portfolio Control Manager (Regime-Aware, v3)

多 archetype 信号仲裁层，硬约束从 constitution.yaml 读取。

控制框架 (v3 职责分工):
  constitution.yaml:  硬约束上限 (slot_count, risk_per_slot, per_strategy_limits)
  pcm_regime.yaml:    仲裁策略 (优先级, Regime 检测, 仓位缩放)

  Layer 1: 硬约束 — slot/risk 从 constitution 读取，不可突破
  Layer 2: Regime 感知 — 动态优先级 + 仓位缩放
     - NORMAL:        LV > FER > ME > BPC  (全仓)
     - HIGH_VOL:      LV > ME > FER > BPC  (缩仓 50%)
     - HIGH_LEVERAGE:  LV > FER > ME > BPC  (缩仓 70%)

优先级依据: 信号条件严格性（越严格越优先）
  LV (liquidation cluster) > FER (均衡偏离反转) > ME (动能扩张) > BPC (趋势延续)

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
# 决策依据: 信号条件严格性（越严格越优先）
DEFAULT_REGIME_PRIORITIES = {
    REGIME_NORMAL: ["LV", "FER", "ME", "BPC"],
    REGIME_HIGH_VOL: ["LV", "ME", "FER", "BPC"],
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
    仓位缩放: 每个 regime 带 position_scale + per_archetype_scale
    """

    def __init__(
        self,
        regime_priorities: Optional[Dict[str, List[str]]] = None,
        detection: Optional[Dict[str, Dict]] = None,
        min_bars_in_regime: int = 3,
        regime_scales: Optional[Dict[str, float]] = None,
        regime_archetype_scales: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self._regime_priorities = regime_priorities or dict(DEFAULT_REGIME_PRIORITIES)
        self._detection = detection or dict(DEFAULT_DETECTION)
        self._min_bars = min_bars_in_regime
        # Regime 仓位缩放: {regime_name: scale}
        self._regime_scales: Dict[str, float] = regime_scales or {
            REGIME_NORMAL: 1.0,
            REGIME_HIGH_VOL: 0.5,
            REGIME_HIGH_LEVERAGE: 0.3,
        }
        # Per-archetype 覆盖: {regime_name: {archetype: scale}}
        self._regime_archetype_scales: Dict[str, Dict[str, float]] = (
            regime_archetype_scales or {}
        )

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

    @property
    def current_position_scale(self) -> float:
        """当前 Regime 的全局仓位缩放因子"""
        return self._regime_scales.get(self._current_regime, 1.0)

    def get_archetype_scale(self, archetype: str) -> float:
        """获取当前 Regime 下特定 archetype 的仓位缩放因子。

        优先级: per_archetype_scale > regime 全局 scale > 1.0
        """
        per_arch = self._regime_archetype_scales.get(self._current_regime, {})
        if archetype.upper() in per_arch:
            return per_arch[archetype.upper()]
        return self.current_position_scale

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
    regime_scales: Dict[str, float] = {}
    regime_archetype_scales: Dict[str, Dict[str, float]] = {}
    for regime_name, regime_cfg in cfg.get("regimes", {}).items():
        if "priority" in regime_cfg:
            priorities[regime_name] = regime_cfg["priority"]
        if "position_scale" in regime_cfg:
            regime_scales[regime_name] = float(regime_cfg["position_scale"])
        per_arch = regime_cfg.get("per_archetype_scale")
        if per_arch and isinstance(per_arch, dict):
            regime_archetype_scales[regime_name] = {
                k.upper(): float(v) for k, v in per_arch.items()
            }

    detection = cfg.get("detection", {})
    min_bars = cfg.get("min_bars_in_regime", 3)

    return RegimeDetector(
        regime_priorities=priorities or None,
        detection=detection or None,
        min_bars_in_regime=min_bars,
        regime_scales=regime_scales or None,
        regime_archetype_scales=regime_archetype_scales or None,
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

# 默认优先级（按信号条件严格性排序）
DEFAULT_ARCHETYPE_PRIORITY = ["LV", "FER", "ME", "BPC"]


def _load_constitution_constraints(
    constitution_yaml: Optional[str],
) -> Dict[str, Any]:
    """Load hard constraints from constitution.yaml.

    Returns dict with keys: slot_count, risk_per_slot, per_strategy_limits,
    add_position_rules.
    Falls back to safe defaults if file not found.
    """
    defaults = {
        "slot_count": 2,
        "risk_per_slot": 0.01,
        "per_strategy_limits": {},
        "add_position_rules": {},
    }
    if not constitution_yaml:
        return defaults
    p = Path(constitution_yaml)
    if not p.exists():
        logger.warning("Constitution YAML not found: %s, using defaults", p)
        return defaults
    try:
        obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("Failed to load constitution YAML: %s, using defaults", e)
        return defaults

    slots = obj.get("slots") or {}
    ra = obj.get("resource_allocation") or {}
    add_rules = (
        ra.get("add_position_rules")
        or ra.get("add_position")
        or obj.get("add_position")
        or {}
    )
    return {
        "slot_count": int(slots.get("slot_count", 2)),
        "risk_per_slot": float(slots.get("risk_per_slot", 0.01)),
        "per_strategy_limits": dict(ra.get("per_strategy_limits") or {}),
        "add_position_rules": dict(add_rules),
    }


class LivePCM:
    """
    Live Portfolio Control Manager (Regime-Aware, v3)

    职责:
      1. 注册多个策略（BPC, ME, FER, LV），每个策略实现 decide() 接口
      2. Regime 检测 → 动态优先级 + 仓位缩放
      3. 同 symbol 不同 archetype 同时触发 → 当前 regime 优先级选最高
      4. 同优先级比 Evidence Score（高的优先）
      5. 跨 symbol slot 控制（从 constitution 读 max_slots）
      6. Regime 仓位缩放 → size_multiplier 调整

    配置来源:
        constitution.yaml:  slot_count, risk_per_slot, per_strategy_limits
        pcm_regime.yaml:    regimes (priority + position_scale), detection
    """

    def __init__(
        self,
        archetype_priority: Optional[List[str]] = None,
        max_slots: Optional[int] = None,
        get_open_slot_count: Optional[callable] = None,
        regime_detector: Optional[RegimeDetector] = None,
        regime_config_path: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
        constitution_yaml: Optional[str] = None,
    ):
        """
        Args:
            archetype_priority: archetype 静态优先级列表。如有 regime_detector 则被忽略。
            max_slots: 显式指定 max_slots（覆盖 constitution）。
                未提供时从 constitution_yaml 读取，均未提供时默认 2。
            get_open_slot_count: 可选回调，返回当前已占用 slot 数
            regime_detector: 可选 RegimeDetector 实例。
            regime_config_path: 可选 pcm_regime.yaml 路径。
            override_config: Layer 3 Override 配置。显式传入优先于 YAML。
            constitution_yaml: 可选 constitution.yaml 路径。
                提供后从中读取 slot_count、risk_per_slot、per_strategy_limits。
                未提供时尝试从 pcm_regime.yaml 的 constitution_ref 自动发现。
        """
        self._strategies: Dict[str, DecisionHandler] = {}
        self._strategy_timeframes: Dict[str, str] = {}  # archetype → timeframe
        self._get_open_slot_count = get_open_slot_count

        # Layer 3: Override 配置
        self._override_config: Dict[str, Any] = override_config or {}

        # 加载 regime 配置
        self._regime_cfg: Dict[str, Any] = {}
        if regime_detector is not None:
            self._regime_detector = regime_detector
        elif regime_config_path is not None:
            self._regime_cfg = load_regime_config(regime_config_path)
            self._regime_detector = _build_regime_detector(self._regime_cfg)
            if not self._override_config:
                self._override_config = self._regime_cfg.get("override", {})
        else:
            self._regime_detector = None

        # 从 constitution 加载硬约束
        _const_yaml = constitution_yaml
        if not _const_yaml and self._regime_cfg:
            _const_yaml = self._regime_cfg.get("constitution_ref")
        self._constitution = _load_constitution_constraints(_const_yaml)

        # max_slots: 显式参数 > constitution > 默认 2
        if max_slots is not None:
            self._max_slots = max_slots
        else:
            self._max_slots = self._constitution["slot_count"]
        logger.info(
            "PCM: max_slots=%d (source=%s)",
            self._max_slots,
            "explicit" if max_slots is not None else "constitution",
        )

        # 静态优先级（当无 regime detector 时使用）
        self._archetype_priority = archetype_priority or list(
            DEFAULT_ARCHETYPE_PRIORITY
        )

        # 可选: 监控统计收集器
        self.stats_collector = None  # 通过外部注入 StatsCollector 实例

    # ── 注册 / 管理 ──

    def register(
        self,
        archetype: str,
        strategy: DecisionHandler,
        *,
        timeframe: Optional[str] = None,
    ) -> None:
        """注册一个 archetype 策略实例

        Args:
            archetype: 策略名称 (如 "bpc", "me", "fer")
            strategy: 实现 decide() 接口的策略实例
            timeframe: 该策略使用的主时间框架 (如 "240T", "60T")
                用于多时间框架模式下路由特征。
        """
        self._strategies[archetype] = strategy
        if timeframe is not None:
            self._strategy_timeframes[archetype] = timeframe
        logger.info(
            "PCM: 注册策略 archetype=%s (%s) timeframe=%s",
            archetype,
            type(strategy).__name__,
            timeframe or "default",
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

    @property
    def constitution(self) -> Dict[str, Any]:
        """Constitution 硬约束 (只读)"""
        return dict(self._constitution)

    def resolve_risk_for_strategy(self, archetype: str) -> float:
        """Return effective risk fraction for a strategy.

        Logic: min(risk_per_slot, strategy.max_risk_per_trade)
        If strategy has no max_risk_per_trade, returns risk_per_slot.
        """
        risk_per_slot = float(self._constitution.get("risk_per_slot", 0.01))
        limits = self._constitution.get("per_strategy_limits") or {}
        strat = limits.get(archetype.lower()) or {}
        strat_risk = strat.get("max_risk_per_trade")
        if strat_risk is not None:
            return min(risk_per_slot, float(strat_risk))
        return risk_per_slot

    @property
    def current_position_scale(self) -> float:
        """当前 Regime 的全局仓位缩放因子"""
        if self._regime_detector is not None:
            return self._regime_detector.current_position_scale
        return 1.0

    def get_archetype_scale(self, archetype: str) -> float:
        """获取当前 Regime 下特定 archetype 的仓位缩放因子"""
        if self._regime_detector is not None:
            return self._regime_detector.get_archetype_scale(archetype)
        return 1.0

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
        features_by_timeframe: Optional[Dict[str, Dict[str, Any]]] = None,
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

        Args:
            features: 主时间框架特征 (默认 4H)
            symbol: 交易对
            bars: 近期 bars (用于执行规则)
            features_by_timeframe: 多时间框架特征 {timeframe: features_dict}
                用于多策略多 timeframe 路由。各策略绑定的 timeframe
                通过 register(timeframe=...) 注册。

        Returns:
            List[TradeIntent]（0 或 1 个元素）
        """
        if not self._strategies:
            return []

        # ── 1. Regime 检测 (Layer 2) ──
        # Regime 使用主时间框架 (4H) 特征检测
        if self._regime_detector is not None:
            self._regime_detector.detect(features)

        # ── 2. 收集所有策略的候选信号 ──
        all_intents: List[TradeIntent] = []
        for arch_name, strategy in self._strategies.items():
            try:
                # 多时间框架路由: 使用策略绑定的 timeframe 对应的特征
                strat_features = features
                if features_by_timeframe and arch_name in self._strategy_timeframes:
                    tf = self._strategy_timeframes[arch_name]
                    if tf in features_by_timeframe:
                        strat_features = features_by_timeframe[tf]
                intents = strategy.decide(
                    features=strat_features, symbol=symbol, bars=bars
                )
                all_intents.extend(intents)

                # 收集漏斗统计
                if self.stats_collector is not None:
                    funnel = getattr(strategy, "_last_funnel", {})
                    self.stats_collector.record_strategy_eval(
                        symbol=symbol,
                        strategy=arch_name,
                        funnel=funnel,
                    )
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
            if self.stats_collector is not None:
                self.stats_collector.record_pcm_selected(symbol, intent.archetype)
            return [self._apply_regime_scale(intent)]

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
            if self.stats_collector is not None:
                self.stats_collector.record_pcm_selected(
                    symbol, override_winner.archetype
                )
            return [self._apply_regime_scale(override_winner)]

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
            "PCM: %s 选中 %s (priority=%d, evidence=%.2f, scale=%.2f)%s",
            symbol,
            best_intent.archetype,
            rank,
            evidence,
            self.get_archetype_scale(best_intent.archetype),
            regime_str,
        )
        if self.stats_collector is not None:
            self.stats_collector.record_pcm_selected(symbol, best_intent.archetype)
        return [self._apply_regime_scale(best_intent)]

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

    # ── Regime 仓位缩放 ──

    def _apply_regime_scale(self, intent: TradeIntent) -> TradeIntent:
        """Apply regime-aware position scaling to TradeIntent.

        缩放因子乘在 size_multiplier 上，不修改其他字段。
        """
        scale = self.get_archetype_scale(intent.archetype)
        if scale >= 1.0:
            return intent

        existing_mult = (
            intent.size_multiplier if intent.size_multiplier is not None else 1.0
        )
        new_mult = existing_mult * scale

        logger.debug(
            "PCM regime scale: %s %s scale=%.2f (%.2f → %.2f)",
            intent.symbol,
            intent.archetype,
            scale,
            existing_mult,
            new_mult,
        )
        # TradeIntent 是 frozen dataclass，需要重建
        return TradeIntent(
            action=intent.action,
            symbol=intent.symbol,
            archetype=intent.archetype,
            execution_strategy=intent.execution_strategy,
            confidence=intent.confidence,
            quantity=intent.quantity,
            size_multiplier=new_mult,
            position_id=intent.position_id,
            add_position=intent.add_position,
            parent_position_id=intent.parent_position_id,
            current_r=intent.current_r,
            locked_profit=intent.locked_profit,
            execution_tags=intent.execution_tags,
            execution_evidence=intent.execution_evidence,
            execution_profile=intent.execution_profile,
            pcm_budget=intent.pcm_budget,
        )

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
            "max_slots_source": (
                "constitution"
                if not hasattr(self, "_explicit_max_slots")
                else "explicit"
            ),
            "override_enabled": bool(self._override_config),
            "override_rules": (
                list(self._override_config.keys()) if self._override_config else []
            ),
            "constitution": {
                "slot_count": self._constitution.get("slot_count"),
                "risk_per_slot": self._constitution.get("risk_per_slot"),
                "per_strategy_limits": self._constitution.get("per_strategy_limits"),
            },
        }
        if self._regime_detector is not None:
            stats["current_regime"] = self._regime_detector.current_regime
            stats["regime_switch_count"] = self._regime_detector.switch_count
            stats["current_position_scale"] = (
                self._regime_detector.current_position_scale
            )
        return stats


def create_live_pcm(
    archetype_priority: Optional[List[str]] = None,
    max_slots: Optional[int] = None,
    get_open_slot_count: Optional[callable] = None,
    regime_config_path: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None,
    constitution_yaml: Optional[str] = None,
) -> LivePCM:
    """
    创建 LivePCM 实例 (v3)

    Args:
        archetype_priority: 静态优先级列表。
        max_slots: 显式指定 max_slots。未提供时从 constitution 读取。
        get_open_slot_count: 可选回调
        regime_config_path: 可选 pcm_regime.yaml 路径
        override_config: Layer 3 Override 配置
        constitution_yaml: 可选 constitution.yaml 路径

    Returns:
        初始化好的 LivePCM（尚未注册策略）
    """
    return LivePCM(
        archetype_priority=archetype_priority,
        max_slots=max_slots,
        get_open_slot_count=get_open_slot_count,
        regime_config_path=regime_config_path,
        override_config=override_config,
        constitution_yaml=constitution_yaml,
    )
