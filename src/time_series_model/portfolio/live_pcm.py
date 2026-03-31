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
from dataclasses import replace
from datetime import datetime
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
        "slots_enabled": True,
        "risk_per_slot": 0.01,
        "per_strategy_limits": {},
        "add_position_rules": {},
        "dynamic_slot_policy": {},
        "intent_selection_policy": {},
        "direction_policy": {},
        "risk_budget_policy": {},
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
        "slots_enabled": bool(slots.get("enabled", True)),
        "risk_per_slot": float(slots.get("risk_per_slot", 0.01)),
        "per_strategy_limits": dict(ra.get("per_strategy_limits") or {}),
        "add_position_rules": dict(add_rules),
        "dynamic_slot_policy": dict(ra.get("dynamic_slot_policy") or {}),
        "intent_selection_policy": dict(ra.get("intent_selection_policy") or {}),
        "direction_policy": dict(ra.get("direction_policy") or {}),
        "risk_budget_policy": dict(ra.get("risk_budget_policy") or {}),
        "evidence_min_score": float(ra.get("evidence_min_score", 0.0)),
        "evidence_position_scale": bool(ra.get("evidence_position_scale", True)),
    }


def _capacity_limit_from_cfg(
    cfg: Optional[Dict[str, Any]],
    *,
    default: int,
) -> int:
    """Read strategy/policy parallel capacity from a mapping."""
    if not isinstance(cfg, dict):
        return int(default)
    if "capacity_limit" in cfg:
        return int(cfg["capacity_limit"])
    return int(default)


class LivePCM:
    """
    Live Portfolio Control Manager (Regime-Aware, v3)

    职责:
      1. 注册多个策略（BPC, ME, FER, LV），每个策略实现 decide() 接口
      2. Regime 检测 → 动态优先级 + 仓位缩放
      3. 同 symbol 不同 archetype 同时触发 → 当前 regime 优先级选最高
      4. 同优先级比 Evidence Score（高的优先）
      5. 跨 symbol slot 控制（从 constitution 读 capacity_limit）
      6. Regime 仓位缩放 + Evidence 仓位缩放 → size_multiplier 调整
      7. Evidence 入场门槛（score < min_score → 拒绝开仓）

    配置来源:
        constitution.yaml:  slot_count, risk_per_slot, per_strategy_limits
        pcm_regime.yaml:    regimes (priority + position_scale), detection
    """

    def __init__(
        self,
        archetype_priority: Optional[List[str]] = None,
        capacity_limit: Optional[int] = None,
        get_open_slot_count: Optional[callable] = None,
        regime_detector: Optional[RegimeDetector] = None,
        regime_config_path: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
        constitution_yaml: Optional[str] = None,
    ):
        """
        Args:
            archetype_priority: archetype 静态优先级列表。如有 regime_detector 则被忽略。
            capacity_limit: 显式指定容量上限（覆盖 constitution）。
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

        # Slot 追踪: key = "{symbol}:{archetype}", value = True
        self._slot_evidence: Dict[str, float] = {}
        # Per-slot loss metric in R (higher means worse loss, e.g. max(0, -current_r)).
        self._slot_loss_r: Dict[str, float] = {}
        # Per-slot risk usage (fraction of equity), used by percent budget policy.
        self._slot_risk_frac: Dict[str, float] = {}
        self._slot_stop_pct: Dict[str, float] = {}
        self._risk_reject_counts: Dict[str, int] = {
            "symbol_cap": 0,
            "family_cap": 0,
            "total_cap": 0,
            "stress_cap": 0,
            "deleverage_freeze": 0,
        }
        self._risk_last_snapshot: Dict[str, Any] = {}
        self._last_evictions: List[Tuple[str, str]] = []
        self._latest_features: Dict[str, Any] = {}
        self._dir_state: Dict[str, Dict[str, Any]] = {}

        # Layer 3: Override 配置
        self._override_config: Dict[str, Any] = override_config or {}

        # 加载 regime 配置
        self._regime_cfg: Dict[str, Any] = {}
        if regime_detector is not None:
            self._regime_detector = regime_detector
        elif regime_config_path is not None:
            self._regime_cfg = load_regime_config(regime_config_path)
            # 若未显式配置 regimes/detection，则视为关闭 regime 缩放与切换
            # （避免误用默认阈值和默认缩放）
            _has_regime_logic = bool(
                dict(self._regime_cfg.get("regimes") or {})
                or dict(self._regime_cfg.get("detection") or {})
            )
            if _has_regime_logic:
                self._regime_detector = _build_regime_detector(self._regime_cfg)
            else:
                self._regime_detector = None
                logger.info(
                    "PCM: regime detector disabled (no regimes/detection in %s)",
                    regime_config_path,
                )
            if not self._override_config:
                self._override_config = self._regime_cfg.get("override", {})
        else:
            self._regime_detector = None

        # 从 constitution 加载硬约束
        _const_yaml = constitution_yaml
        if not _const_yaml and self._regime_cfg:
            _const_yaml = self._regime_cfg.get("constitution_ref")
        self._constitution = _load_constitution_constraints(_const_yaml)
        _risk_pol = dict(self._constitution.get("risk_budget_policy") or {})
        # Percent-only runtime: slot gate is permanently disabled.
        self._slot_gate_enabled = False
        self._capacity_limit = int(capacity_limit) if capacity_limit is not None else 0
        logger.info(
            "PCM: percent-only mode — slot gate disabled (capacity_limit_compat=%d)",
            self._capacity_limit,
        )

        # 静态优先级（当无 regime detector 时使用）
        self._archetype_priority = archetype_priority or list(
            DEFAULT_ARCHETYPE_PRIORITY
        )

        # Direction filter source is constitution.direction_policy.
        # Keep these legacy attrs for compatibility, but disable regime-file EMA gate.
        self._ema_filter_enabled: bool = False
        self._ema_close_feature: str = "close"
        self._ema_feature: str = "ema_200"
        self._ema_bull_allowed: set = set()
        self._ema_bear_allowed: set = set()
        self._ema_fallback: str = "allow_all"

        # 可选: 监控统计收集器
        self.stats_collector = None  # 通过外部注入 StatsCollector 实例

    def _parse_family_and_side(
        self,
        archetype: str,
        action: Optional[str] = None,
    ) -> tuple[str, str]:
        a = str(archetype or "").lower().strip()
        parts = [p for p in a.split("-") if p]
        # strip trailing timeframe token: 60t / 240t
        if parts and parts[-1].endswith("t") and parts[-1][:-1].isdigit():
            parts = parts[:-1]
        # handle "<family>-<long|short>(-...)" naming
        if len(parts) >= 2 and parts[1] in {"long", "short"}:
            return parts[0], parts[1]
        act = str(action or "").upper().strip()
        if act == "LONG":
            return a, "long"
        if act == "SHORT":
            return a, "short"
        return a, "unknown"

    def _limit_cfg_for_archetype(self, archetype: str) -> Dict[str, Any]:
        limits = dict(self._constitution.get("per_strategy_limits") or {})
        key = str(archetype or "").lower().strip()
        parts = [p for p in key.split("-") if p]
        cands: list[str] = [key]

        # strip trailing timeframe token
        if parts and parts[-1].endswith("t") and parts[-1][:-1].isdigit():
            cands.append("-".join(parts[:-1]))
            parts = parts[:-1]

        # family-direction key
        if len(parts) >= 2 and parts[1] in {"long", "short"}:
            cands.append("-".join(parts[:2]))

        # family only
        if parts:
            cands.append(parts[0])

        seen: set[str] = set()
        for k in cands:
            kk = str(k).lower().strip()
            if not kk or kk in seen:
                continue
            seen.add(kk)
            cand = limits.get(kk) or {}
            if isinstance(cand, dict) and cand:
                return cand
        return {}

    def _estimate_intent_risk_frac(
        self,
        intent: TradeIntent,
        meta: Dict[str, Any],
    ) -> float:
        """Estimate incremental risk fraction to equity for intent.

        Percent budget mode uses risk fraction directly:
          delta_risk ~= resolve_risk_for_strategy * size_multiplier
        """
        risk_frac = float(self.resolve_risk_for_strategy(str(intent.archetype)))
        size_mult = float(intent.size_multiplier or 1.0)
        return max(0.0, risk_frac * max(0.0, size_mult))

    def rollback_intent_reservation(self, intent: TradeIntent) -> None:
        """Rollback previously reserved risk for an unfilled intent.

        PCM books risk when an intent is accepted. If downstream execution fails
        (e.g. add trigger not met/order rejected), caller should rollback to
        avoid ghost risk occupancy.
        """
        try:
            symbol = str(getattr(intent, "symbol", "") or "").strip()
            archetype = str(getattr(intent, "archetype", "") or "").strip()
            if not symbol or not archetype:
                return
            key = f"{symbol}:{archetype}"
            cur = float(self._slot_risk_frac.get(key, 0.0) or 0.0)
            if cur <= 0.0:
                return
            delta = float(self._estimate_intent_risk_frac(intent, {}) or 0.0)
            if delta <= 0.0:
                return
            new_v = max(0.0, cur - delta)
            if new_v <= 1e-12:
                # New entry reservation that never became a real position.
                self._slot_risk_frac.pop(key, None)
                self._slot_stop_pct.pop(key, None)
                self._slot_loss_r.pop(key, None)
                self._slot_evidence.pop(key, None)
            else:
                self._slot_risk_frac[key] = new_v
        except Exception:
            logger.exception("PCM: rollback_intent_reservation failed")

    def _current_total_risk_frac(self) -> float:
        return float(
            sum(max(0.0, float(v or 0.0)) for v in self._slot_risk_frac.values())
        )

    def _current_symbol_risk_frac(self, symbol: str) -> float:
        sym = str(symbol)
        total = 0.0
        for k, v in self._slot_risk_frac.items():
            if k.startswith(f"{sym}:"):
                total += max(0.0, float(v or 0.0))
        return float(total)

    def _current_family_risk_frac(self, family: str) -> float:
        fam = str(family or "").lower().strip()
        total = 0.0
        for k, v in self._slot_risk_frac.items():
            try:
                _, arch = k.split(":", 1)
            except Exception:
                continue
            f, _ = self._parse_family_and_side(arch)
            if f == fam:
                total += max(0.0, float(v or 0.0))
        return float(total)

    def _slot_risk_breakdown(
        self,
        *,
        family: str = "",
        symbol: str = "",
        limit: int = 12,
    ) -> str:
        """Format current slot risk ledger for debugging rejects."""
        fam_filter = str(family or "").lower().strip()
        sym_filter = str(symbol or "").strip()
        rows = []
        for key, raw in self._slot_risk_frac.items():
            risk = max(0.0, float(raw or 0.0))
            if risk <= 0.0:
                continue
            if sym_filter and not key.startswith(f"{sym_filter}:"):
                continue
            if fam_filter:
                try:
                    _, arch = key.split(":", 1)
                except Exception:
                    continue
                fam, _ = self._parse_family_and_side(arch)
                if fam != fam_filter:
                    continue
            rows.append((key, risk))
        if not rows:
            return "<empty>"
        rows.sort(key=lambda x: x[1], reverse=True)
        show = rows[: max(1, int(limit or 12))]
        s = ", ".join(f"{k}={v:.4f}" for k, v in show)
        if len(rows) > len(show):
            s += f", ... +{len(rows) - len(show)} more"
        return s

    def _released_slot_count(self, family: str, residual: float) -> int:
        fam = str(family or "").lower().strip()
        n = 0
        thr = max(0.0, float(residual)) + 1e-9
        for key, risk in self._slot_risk_frac.items():
            try:
                _, arch = key.split(":", 1)
            except Exception:
                continue
            f, _ = self._parse_family_and_side(arch)
            if f == fam and float(risk or 0.0) <= thr and float(risk or 0.0) > 0.0:
                n += 1
        return n

    def _drawdown_cap_multiplier(self, policy: Dict[str, Any]) -> float:
        shrink = dict(policy.get("shrink") or {})
        if not bool(shrink.get("enabled", False)):
            return 1.0
        dd = max(0.0, float(self._latest_features.get("drawdown", 0.0) or 0.0))
        tiers = list(shrink.get("by_drawdown") or [])
        mult = 1.0
        for t in tiers:
            if not isinstance(t, dict):
                continue
            try:
                dd_thr = float(t.get("drawdown_gte", 0.0) or 0.0)
                cap_mult = float(t.get("cap_multiplier", 1.0) or 1.0)
            except Exception:
                continue
            if dd >= dd_thr:
                mult = min(mult, max(0.05, cap_mult))
        return float(mult)

    def _dynamic_caps(
        self,
        *,
        family: str,
        policy: Dict[str, Any],
    ) -> Dict[str, float]:
        total_cap = max(0.0, float(policy.get("max_total_risk_pct", 0.05) or 0.0))
        family_cap = max(0.0, float(policy.get("max_family_risk_pct", 0.03) or 0.0))
        symbol_cap = max(0.0, float(policy.get("max_symbol_risk_pct", 0.015) or 0.0))

        # Profit-up expansion
        exp = dict(policy.get("expansion") or {})
        total_mult = 1.0
        family_mult = 1.0
        symbol_mult = 1.0
        if bool(exp.get("enabled", False)):
            targets = {
                str(x).lower().strip() for x in (exp.get("target_families") or ["bpc"])
            }
            if str(family or "").lower().strip() in targets:
                residual = float(
                    policy.get("breakeven_residual_risk_pct", 0.001) or 0.0
                )
                released = self._released_slot_count(
                    str(family or "").lower().strip(), residual
                )
                trigger = int(exp.get("trigger_released_slots", 0) or 0)
                step_slots = max(1, int(exp.get("step_released_slots", 1) or 1))
                step_mult = float(exp.get("step_multiplier", 0.25) or 0.0)
                if released >= trigger:
                    steps = ((released - trigger) // step_slots) + 1
                    bump = max(0.0, float(steps) * max(0.0, step_mult))
                    family_mult = 1.0 + bump
                    total_mult = 1.0 + bump
                    symbol_mult = 1.0 + bump * 0.5
                family_mult = min(
                    family_mult,
                    max(1.0, float(exp.get("max_family_multiplier", 3.0) or 3.0)),
                )
                total_mult = min(
                    total_mult,
                    max(1.0, float(exp.get("max_total_multiplier", 2.0) or 2.0)),
                )
                symbol_mult = min(
                    symbol_mult,
                    max(1.0, float(exp.get("max_symbol_multiplier", 1.5) or 1.5)),
                )

        shrink_mult = self._drawdown_cap_multiplier(policy)
        total_cap_eff = total_cap * total_mult * shrink_mult
        family_cap_eff = family_cap * family_mult * shrink_mult
        symbol_cap_eff = symbol_cap * symbol_mult * shrink_mult
        return {
            "total_cap": float(total_cap_eff),
            "family_cap": float(family_cap_eff),
            "symbol_cap": float(symbol_cap_eff),
            "total_mult": float(total_mult),
            "family_mult": float(family_mult),
            "symbol_mult": float(symbol_mult),
            "shrink_mult": float(shrink_mult),
        }

    def _slot_stress_loss(self, slot_key: str, shock_pct: float) -> float:
        risk = max(0.0, float(self._slot_risk_frac.get(slot_key, 0.0) or 0.0))
        stop_pct = max(1e-6, float(self._slot_stop_pct.get(slot_key, 0.0) or 0.0))
        if risk <= 0.0 or stop_pct <= 0.0:
            return 0.0
        notional = risk / stop_pct
        return max(0.0, notional * max(0.0, float(shock_pct)))

    def _estimate_stress_usage(
        self,
        *,
        policy: Dict[str, Any],
        slot_key: str,
        delta_risk: float,
        delta_stop_pct: float,
    ) -> Dict[str, float]:
        stress_cfg = dict(policy.get("stress") or {})
        if not bool(stress_cfg.get("enabled", False)):
            return {"before": 0.0, "after": 0.0, "cap": 0.0}
        shock_pct = max(0.0, float(stress_cfg.get("shock_pct", 0.0) or 0.0))
        cap = max(0.0, float(stress_cfg.get("max_stress_loss_pct", 0.0) or 0.0))
        before = 0.0
        for key in self._slot_risk_frac.keys():
            before += self._slot_stress_loss(key, shock_pct)
        delta = 0.0
        if delta_risk > 0.0 and delta_stop_pct > 0.0:
            delta = (delta_risk / max(1e-6, delta_stop_pct)) * shock_pct
        after = before + max(0.0, delta)
        return {"before": float(before), "after": float(after), "cap": float(cap)}

    def _plan_tiered_deleveraging(
        self,
        *,
        symbol: str,
        policy: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, str]], bool]:
        delev = dict(policy.get("deleveraging") or {})
        if not bool(delev.get("enabled", False)):
            return [], False

        total_cap = max(1e-9, float(policy.get("max_total_risk_pct", 0.05) or 0.05))
        total_now = self._current_total_risk_frac()
        usage = total_now / total_cap if total_cap > 0 else 0.0

        freeze_ratio = float(delev.get("freeze_new_entries_ratio", 1.0) or 1.0)
        freeze_new = usage >= freeze_ratio

        tiers = []
        for t in list(delev.get("tiers") or []):
            if not isinstance(t, dict):
                continue
            try:
                trig = float(t.get("trigger_ratio", 0.0) or 0.0)
                reduce_to = float(t.get("reduce_to_ratio", 1.0) or 1.0)
            except Exception:
                continue
            tiers.append((trig, reduce_to))
        tiers = sorted(tiers, key=lambda x: x[0], reverse=True)

        target_ratio = None
        for trig, reduce_to in tiers:
            if usage >= trig:
                target_ratio = reduce_to
                break
        if target_ratio is None:
            return [], freeze_new

        target_total = max(0.0, target_ratio * total_cap)
        to_reduce = max(0.0, total_now - target_total)
        if to_reduce <= 1e-12:
            return [], freeze_new

        cands: List[Tuple[str, str, float, float]] = []
        for key, risk in self._slot_risk_frac.items():
            if not key.startswith(f"{symbol}:"):
                continue
            try:
                sym, arch = key.split(":", 1)
            except Exception:
                continue
            loss_r = float(self._slot_loss_r.get(key, 0.0) or 0.0)
            cands.append((sym, arch, max(0.0, float(risk or 0.0)), max(0.0, loss_r)))
        # Worst-loss first, then higher risk concentration first.
        cands.sort(key=lambda x: (-x[3], -x[2]))

        evictions: List[Tuple[str, str]] = []
        reduced = 0.0
        for sym, arch, risk, _ev in cands:
            if reduced >= to_reduce - 1e-12:
                break
            key = f"{sym}:{arch}"
            reduced += risk
            evictions.append((sym, arch))
            self._slot_evidence.pop(key, None)
            self._slot_risk_frac.pop(key, None)
            self._slot_stop_pct.pop(key, None)
            self._slot_loss_r.pop(key, None)

        if evictions:
            logger.warning(
                "PCM: 触发分级去杠杆 usage=%.3f target=%.3f reduce=%.4f evictions=%s",
                usage,
                float(target_ratio),
                float(to_reduce),
                evictions,
            )
        return evictions, freeze_new

    def _winner_priority_ok(
        self,
        intent: TradeIntent,
        *,
        family: str,
        policy: Dict[str, Any],
    ) -> bool:
        wp = dict(policy.get("winner_priority") or {})
        if not bool(wp.get("enabled", False)):
            return False
        allow_families = [
            str(x).lower().strip() for x in (wp.get("allow_families") or ["bpc"])
        ]
        if family not in set(allow_families):
            return False
        if bool(wp.get("require_breakeven_locked", True)) and not bool(
            intent.locked_profit
        ):
            return False
        min_r = float(wp.get("require_min_current_r", 1.0))
        cur_r = float(intent.current_r or 0.0)
        if cur_r < min_r:
            return False
        return True

    def _observed_market_side(
        self,
        *,
        features: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> str:
        close_key = str(policy.get("close_feature", "close"))
        ema_key = str(policy.get("ema_feature", "ema_200"))
        close_v = features.get(close_key)
        ema_v = features.get(ema_key)
        if close_v is None or ema_v is None:
            return "unknown"
        try:
            close_f = float(close_v)
            ema_f = float(ema_v)
        except Exception:
            return "unknown"
        return "long" if close_f > ema_f else "short"

    def _effective_market_side(
        self,
        *,
        symbol: str,
        features: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> str:
        """返回防抖后的有效主方向: long/short/unknown"""
        observed = self._observed_market_side(features=features, policy=policy)
        debounce_bars = int(policy.get("debounce_bars", 1) or 1)
        if debounce_bars <= 1:
            return observed
        st = self._dir_state.get(symbol) or {
            "confirmed_side": "unknown",
            "pending_side": None,
            "pending_count": 0,
        }
        confirmed = str(st.get("confirmed_side", "unknown"))
        pending = st.get("pending_side")
        pending_count = int(st.get("pending_count", 0))
        if observed in {"long", "short"}:
            if confirmed == "unknown":
                if pending == observed:
                    pending_count += 1
                else:
                    pending = observed
                    pending_count = 1
                if pending_count >= debounce_bars:
                    confirmed = observed
                    pending = None
                    pending_count = 0
            elif observed != confirmed:
                if pending == observed:
                    pending_count += 1
                else:
                    pending = observed
                    pending_count = 1
                if pending_count >= debounce_bars:
                    confirmed = observed
                    pending = None
                    pending_count = 0
            else:
                pending = None
                pending_count = 0
        self._dir_state[symbol] = {
            "confirmed_side": confirmed,
            "pending_side": pending,
            "pending_count": pending_count,
        }
        return confirmed

    def _is_direction_allowed(
        self,
        intent: TradeIntent,
        *,
        market_side: str,
        policy: Dict[str, Any],
    ) -> bool:
        if not bool(policy.get("enabled", False)):
            return True
        mode = str(policy.get("mode", "ema200_single_direction")).lower().strip()
        if mode != "ema200_single_direction":
            return True

        if market_side == "unknown":
            default_side = str(policy.get("default_side", "both")).lower().strip()
            if default_side == "both":
                return True
            _, s = self._parse_family_and_side(intent.archetype, intent.action)
            return s == default_side

        family, intent_side = self._parse_family_and_side(
            intent.archetype, intent.action
        )
        if intent_side == "unknown":
            return True
        if intent_side == market_side:
            return True

        reverse_allowed = bool(policy.get("fer_reverse_allowed", True))
        reverse_families = [
            str(x).lower().strip()
            for x in (policy.get("reverse_exempt_families") or ["fer"])
        ]
        if reverse_allowed and family in set(reverse_families):
            return True
        return False

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
        strat = self._limit_cfg_for_archetype(archetype)
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

    def _parse_timeframe_minutes(self, arch_name: str) -> int:
        tf = str(self._strategy_timeframes.get(arch_name, "") or "").upper().strip()
        if tf.endswith("T"):
            try:
                return int(tf[:-1])
            except Exception:
                return 0
        return 0

    def _to_ts_ns(self, val: Any) -> int:
        if val is None:
            return 2**63 - 1
        if isinstance(val, (int, float)):
            # 保持数量级，不做单位猜测；仅用于稳定排序
            return int(val)
        if isinstance(val, datetime):
            return int(val.timestamp() * 1e9)
        try:
            if hasattr(val, "value"):  # pandas.Timestamp
                return int(val.value)
        except Exception:
            pass
        try:
            if hasattr(val, "timestamp"):
                return int(float(val.timestamp()) * 1e9)
        except Exception:
            pass
        return 2**63 - 1

    def _effective_stop_pct_from_intent(
        self,
        intent: TradeIntent,
        feat: Dict[str, Any],
    ) -> float:
        ep = dict(intent.execution_profile or {})
        rr = dict(ep.get("rr_constraints") or {})
        try:
            sl_r = float(rr.get("stop_loss_r", 0.0) or 0.0)
        except Exception:
            sl_r = 0.0
        close = feat.get("close")
        atr = feat.get("atr")
        try:
            close_f = float(close or 0.0)
            atr_f = float(atr or 0.0)
        except Exception:
            return float("inf")
        if sl_r <= 0 or close_f <= 0 or atr_f <= 0:
            return float("inf")
        atr_stop = max(0.0, sl_r * atr_f / close_f)
        eff = atr_stop
        if rr.get("min_stop_pct") is not None:
            try:
                eff = max(eff, float(rr.get("min_stop_pct")))
            except Exception:
                pass
        if rr.get("max_stop_pct") is not None:
            try:
                eff = min(eff, float(rr.get("max_stop_pct")))
            except Exception:
                pass
        return max(0.0, float(eff))

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[List[Dict[str, Any]]] = None,
        features_by_timeframe: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[TradeIntent]:
        """
        多策略独立 slot 仲裁 → 返回所有通过 per-strategy slot 检查的 TradeIntent

        算法:
          1. Regime 检测 (Layer 2) → 动态缩放
          2. 遍历所有注册策略，收集候选 TradeIntent
          3. 每策略独立检查 per-strategy slot (无跨策略竞争)
          4. slot 满时同 archetype 内 evidence 竞争

        Args:
            features: 主时间框架特征 (默认 4H)
            symbol: 交易对
            bars: 近期 bars (用于执行规则)
            features_by_timeframe: 多时间框架特征 {timeframe: features_dict}
                用于多策略多 timeframe 路由。各策略绑定的 timeframe
                通过 register(timeframe=...) 注册。

        Returns:
            List[TradeIntent]（每策略最多 1 个，可返回多个）
        """
        if not self._strategies:
            return []

        # 每次 decide() 重置驱逐列表 — 供调用方 (事件回测/实盘) 关闭被替换仓位
        self._last_evictions: List[Tuple[str, str]] = (
            []
        )  # [(evicted_symbol, archetype), ...]
        self._latest_features = dict(features or {})

        # ── 1. Regime 检测 (Layer 2) ──
        # Regime 使用主时间框架 (4H) 特征检测
        if self._regime_detector is not None:
            self._regime_detector.detect(features)

        # ── 2. 收集所有策略的候选信号 ──
        all_intents: List[TradeIntent] = []
        _intent_meta: Dict[int, Dict[str, Any]] = {}
        _collect_seq = 0
        for arch_name, strategy in self._strategies.items():
            try:
                # 多时间框架路由: 使用策略绑定的 timeframe 对应的特征
                strat_features = features
                if features_by_timeframe and arch_name in self._strategy_timeframes:
                    tf = self._strategy_timeframes[arch_name]
                    if tf in features_by_timeframe:
                        strat_features = features_by_timeframe[tf]
                    else:
                        # 该策略的 timeframe 当前不可用 → 跳过
                        # (例: 60T bar 上不评估 240T 策略，与实盘行为一致)
                        logger.debug(
                            "PCM: 跳过 %s — timeframe %s 不在当前可用 %s",
                            arch_name,
                            tf,
                            list(features_by_timeframe.keys()),
                        )
                        # 清空 _last_funnel 避免诊断代码读到上一次的结果
                        strategy._last_funnel = {}
                        continue

                intents = strategy.decide(
                    features=strat_features, symbol=symbol, bars=bars
                )
                all_intents.extend(intents)
                for it in intents:
                    _intent_meta[id(it)] = {
                        "strategy_name": arch_name,
                        "timeframe_minutes": self._parse_timeframe_minutes(arch_name),
                        "archetype": str(getattr(it, "archetype", "")).lower().strip(),
                        "effective_stop_pct": self._effective_stop_pct_from_intent(
                            it, strat_features
                        ),
                        "signal_ts_ns": self._to_ts_ns(strat_features.get("timestamp")),
                        "collect_seq": _collect_seq,
                    }
                    _collect_seq += 1

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

        # deterministic v1:
        # timeframe(大优先) -> archetype(BPC>ME>FER) -> effective_stop_pct(小优先) -> FIFO
        _sel = dict(self._constitution.get("intent_selection_policy") or {})
        _arch_pri = [
            str(x).lower().strip()
            for x in (_sel.get("archetype_priority") or ["bpc", "me", "fer"])
        ]
        _arch_rank = {a: i for i, a in enumerate(_arch_pri)}

        def _intent_sort_key(intent: TradeIntent) -> tuple:
            md = _intent_meta.get(id(intent), {})
            return (
                -int(md.get("timeframe_minutes", 0) or 0),
                int(_arch_rank.get(str(md.get("archetype", "")), 99)),
                float(md.get("effective_stop_pct", float("inf"))),
                int(md.get("signal_ts_ns", 2**63 - 1)),
                int(md.get("collect_seq", 10**9)),
            )

        all_intents = sorted(all_intents, key=_intent_sort_key)

        # 同 symbol + 同 archetype 的“新开仓”候选，只保留 deterministic 排序后的第一条。
        _dedup: Dict[tuple[str, str], TradeIntent] = {}
        _dedup_out: List[TradeIntent] = []
        for intent in all_intents:
            if bool(intent.add_position):
                _dedup_out.append(intent)
                continue
            _k = (str(intent.symbol), str(intent.archetype).lower())
            if _k not in _dedup:
                _dedup[_k] = intent
        all_intents = list(_dedup.values()) + _dedup_out

        # ── 3. 每策略独立 slot 检查 (无跨策略竞争) ──
        _dir_pol = dict(self._constitution.get("direction_policy") or {})
        _risk_pol = dict(self._constitution.get("risk_budget_policy") or {})
        _mode = str(_risk_pol.get("risk_budget_mode", "percent")).strip().lower()
        if bool(_risk_pol.get("enabled", True)) and _mode != "percent":
            logger.error(
                "PCM: unsupported risk_budget_mode=%s (expected percent)", _mode
            )
            return []

        _evictions, _freeze_new_entries = self._plan_tiered_deleveraging(
            symbol=str(symbol),
            policy=_risk_pol,
        )
        if _evictions:
            self._last_evictions.extend(_evictions)
        _market_side = self._effective_market_side(
            symbol=str(symbol),
            features=features,
            policy=_dir_pol,
        )
        _ts_ctx = str(
            features.get("_pcm_ts")
            or features.get("timestamp")
            or features.get("time")
            or ""
        )
        accepted: List[TradeIntent] = []
        for intent in all_intents:
            if not self._is_direction_allowed(
                intent,
                market_side=_market_side,
                policy=_dir_pol,
            ):
                logger.info(
                    "PCM: 方向过滤拒绝 ts=%s %s %s action=%s (market_side=%s)",
                    _ts_ctx,
                    intent.symbol,
                    intent.archetype,
                    intent.action,
                    _market_side,
                )
                continue
            # 同 symbol + 同策略家族（BPC/ME/FER/...）只允许单方向持仓（避免 one-way 模式净仓歧义）
            fam, side = self._parse_family_and_side(intent.archetype, intent.action)
            if side in {"long", "short"}:
                _conflict = False
                for k in self._slot_evidence:
                    if not k.startswith(f"{intent.symbol}:"):
                        continue
                    try:
                        _, arch2 = k.split(":", 1)
                    except Exception:
                        continue
                    fam2, side2 = self._parse_family_and_side(arch2)
                    if fam2 == fam and side2 in {"long", "short"} and side2 != side:
                        _conflict = True
                        break
                if _conflict:
                    logger.info(
                        "PCM: 家族方向冲突拒绝 %s %s (family=%s side=%s)",
                        intent.symbol,
                        intent.archetype,
                        fam,
                        side,
                    )
                    continue

            # ── 3.1 Notional 预算约束 (soft/hard + winner priority) ──
            fam, side = self._parse_family_and_side(intent.archetype, intent.action)
            delta_risk = self._estimate_intent_risk_frac(
                intent, _intent_meta.get(id(intent), {})
            )
            _slot_key = f"{intent.symbol}:{intent.archetype}"
            # Break-even positions release risk budget (keeps a tiny residual buffer).
            if (
                _slot_key in self._slot_risk_frac
                and bool(_risk_pol.get("breakeven_release_enabled", True))
                and bool(intent.locked_profit)
            ):
                residual = float(
                    _risk_pol.get("breakeven_residual_risk_pct", 0.001) or 0.0
                )
                self._slot_risk_frac[_slot_key] = min(
                    float(self._slot_risk_frac.get(_slot_key, 0.0) or 0.0),
                    max(0.0, residual),
                )

            total_before = self._current_total_risk_frac()
            sym_before = self._current_symbol_risk_frac(intent.symbol)
            fam_before = self._current_family_risk_frac(fam)
            total_after = total_before + delta_risk
            sym_after = sym_before + delta_risk
            fam_after = fam_before + delta_risk

            if bool(_risk_pol.get("enabled", True)):
                caps = self._dynamic_caps(family=fam, policy=_risk_pol)
                total_cap = float(caps.get("total_cap", 0.0) or 0.0)
                fam_cap = float(caps.get("family_cap", 0.0) or 0.0)
                sym_cap = float(caps.get("symbol_cap", 0.0) or 0.0)
                logger.debug(
                    "PCM Caps: symbol=%s family=%s caps(total=%.4f,family=%.4f,symbol=%.4f) mult(total=%.2f,family=%.2f,symbol=%.2f) shrink=%.2f",
                    intent.symbol,
                    fam,
                    total_cap,
                    fam_cap,
                    sym_cap,
                    float(caps.get("total_mult", 1.0)),
                    float(caps.get("family_mult", 1.0)),
                    float(caps.get("symbol_mult", 1.0)),
                    float(caps.get("shrink_mult", 1.0)),
                )

                if _freeze_new_entries and not bool(intent.add_position):
                    self._risk_reject_counts["deleverage_freeze"] = (
                        int(self._risk_reject_counts.get("deleverage_freeze", 0)) + 1
                    )
                    logger.info(
                        "PCM: 去杠杆冻结新仓拒绝 %s %s (total_usage=%.3f)",
                        intent.symbol,
                        intent.archetype,
                        self._current_total_risk_frac() / max(1e-9, total_cap),
                    )
                    continue

                if sym_cap > 0 and sym_after > sym_cap + 1e-12:
                    self._risk_reject_counts["symbol_cap"] = (
                        int(self._risk_reject_counts.get("symbol_cap", 0)) + 1
                    )
                    logger.info(
                        "PCM: symbol 风险预算超限拒绝 ts=%s %s %s "
                        "(before=%.4f + delta=%.4f => after=%.4f > cap=%.4f, remain=%.4f) "
                        "| slot_risk_frac[symbol]=[%s]",
                        _ts_ctx,
                        intent.symbol,
                        intent.archetype,
                        sym_before,
                        delta_risk,
                        sym_after,
                        sym_cap,
                        max(0.0, sym_cap - sym_before),
                        self._slot_risk_breakdown(symbol=intent.symbol),
                    )
                    continue
                if fam_cap > 0 and fam_after > fam_cap + 1e-12:
                    self._risk_reject_counts["family_cap"] = (
                        int(self._risk_reject_counts.get("family_cap", 0)) + 1
                    )
                    logger.info(
                        "PCM: family 风险预算超限拒绝 ts=%s %s %s "
                        "(before=%.4f + delta=%.4f => after=%.4f > cap=%.4f, remain=%.4f) "
                        "| slot_risk_frac[family]=[%s]",
                        _ts_ctx,
                        intent.symbol,
                        intent.archetype,
                        fam_before,
                        delta_risk,
                        fam_after,
                        fam_cap,
                        max(0.0, fam_cap - fam_before),
                        self._slot_risk_breakdown(family=fam),
                    )
                    continue
                if total_cap > 0 and total_after > total_cap + 1e-12:
                    self._risk_reject_counts["total_cap"] = (
                        int(self._risk_reject_counts.get("total_cap", 0)) + 1
                    )
                    logger.info(
                        "PCM: total 风险预算超限拒绝 ts=%s %s %s "
                        "(before=%.4f + delta=%.4f => after=%.4f > cap=%.4f, remain=%.4f) "
                        "| slot_risk_frac[total]=[%s]",
                        _ts_ctx,
                        intent.symbol,
                        intent.archetype,
                        total_before,
                        delta_risk,
                        total_after,
                        total_cap,
                        max(0.0, total_cap - total_before),
                        self._slot_risk_breakdown(),
                    )
                    continue
                _meta = _intent_meta.get(id(intent), {})
                _stop = float(_meta.get("effective_stop_pct", 0.0) or 0.0)
                stress = self._estimate_stress_usage(
                    policy=_risk_pol,
                    slot_key=_slot_key,
                    delta_risk=delta_risk,
                    delta_stop_pct=_stop,
                )
                if stress["cap"] > 0 and stress["after"] > stress["cap"] + 1e-12:
                    self._risk_reject_counts["stress_cap"] = (
                        int(self._risk_reject_counts.get("stress_cap", 0)) + 1
                    )
                    logger.info(
                        "PCM: stress 预算超限拒绝 %s %s "
                        "(before=%.4f + delta=%.4f => after=%.4f > cap=%.4f)",
                        intent.symbol,
                        intent.archetype,
                        stress["before"],
                        max(0.0, stress["after"] - stress["before"]),
                        stress["after"],
                        stress["cap"],
                    )
                    continue
            if _slot_key in self._slot_evidence and not bool(intent.add_position):
                # 已有同 symbol+archetype 仓位：将意图标记为加仓，让下游走 try_add_position
                # (此前直接 continue 会导致加仓链路完全不触发)
                intent = replace(intent, add_position=True)
                logger.info(
                    "PCM: %s %s 已有持仓，转为 add_position 意图",
                    intent.symbol,
                    intent.archetype,
                )
            if not bool(intent.add_position):
                self._record_slot(symbol, intent.archetype, 0.0)
            self._slot_risk_frac[_slot_key] = max(
                0.0, float(self._slot_risk_frac.get(_slot_key, 0.0) or 0.0)
            ) + max(0.0, delta_risk)
            _meta = _intent_meta.get(id(intent), {})
            try:
                _eff_stop = float(_meta.get("effective_stop_pct", 0.0) or 0.0)
            except Exception:
                _eff_stop = 0.0
            if _eff_stop > 0:
                self._slot_stop_pct[_slot_key] = _eff_stop
            self._risk_last_snapshot = {
                "total_risk_frac": self._current_total_risk_frac(),
                "symbol_risk_frac": self._current_symbol_risk_frac(intent.symbol),
                "family_risk_frac": self._current_family_risk_frac(fam),
                "symbol": str(intent.symbol),
                "family": str(fam),
                "archetype": str(intent.archetype),
                "dynamic_caps": (
                    dict(caps) if bool(_risk_pol.get("enabled", True)) else {}
                ),
                "stress_usage": self._estimate_stress_usage(
                    policy=_risk_pol,
                    slot_key=_slot_key,
                    delta_risk=0.0,
                    delta_stop_pct=0.0,
                ),
                "deleveraging_evictions": list(self._last_evictions),
            }
            if self.stats_collector is not None:
                self.stats_collector.record_pcm_selected(symbol, intent.archetype)
            accepted.append(self._apply_regime_scale(intent))
            logger.info(
                "PCM: %s 选中 %s (scale=%.2f, total_risk=%.4f)",
                symbol,
                intent.archetype,
                self.get_archetype_scale(intent.archetype),
                float(self._risk_last_snapshot.get("total_risk_frac", 0.0)),
            )
        return accepted

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
        最终 size_multiplier = original × regime_scale
        """
        existing_mult = (
            intent.size_multiplier if intent.size_multiplier is not None else 1.0
        )

        # Regime 缩放
        regime_scale = self.get_archetype_scale(intent.archetype)
        new_mult = existing_mult * regime_scale

        # Evidence 缩放已包含在 intent.size_multiplier 中 (GenericLiveStrategy.decide)

        if new_mult >= 1.0 and regime_scale >= 1.0:
            return intent

        logger.debug(
            "PCM scale: %s %s regime=%.2f total=%.2f → %.2f",
            intent.symbol,
            intent.archetype,
            regime_scale,
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

    def _capacity_limit_for_strategy(self, archetype: str) -> int:
        """获取策略容量上限（从 per_strategy_limits 读取，缺省回退全局）"""
        strat = self._limit_cfg_for_archetype(archetype)
        hard_cap = _capacity_limit_from_cfg(strat, default=self._capacity_limit)
        dynamic_cap = self._resolve_dynamic_slots(archetype, hard_cap)
        return max(0, min(hard_cap, dynamic_cap))

    def _resolve_dynamic_slots(self, archetype: str, hard_cap: int) -> int:
        policy = dict(self._constitution.get("dynamic_slot_policy") or {})
        bpc_cfg = dict(policy.get("bpc") or {})
        if not bpc_cfg or not bool(bpc_cfg.get("enabled", False)):
            return hard_cap
        fam, _ = self._parse_family_and_side(archetype)
        if fam != "bpc":
            return hard_cap

        base_slots = int(bpc_cfg.get("base_slots", 1))
        family_capacity_limit = _capacity_limit_from_cfg(bpc_cfg, default=hard_cap)
        slots = max(1, min(hard_cap, family_capacity_limit, base_slots))
        dd = float(self._latest_features.get("drawdown", 0.0) or 0.0)
        daily_loss = float(self._latest_features.get("daily_loss", 0.0) or 0.0)
        step2 = dict(bpc_cfg.get("step2") or {})
        step3 = dict(bpc_cfg.get("step3") or {})
        total_risk_cap = float(policy.get("total_risk_cap", 0.10))
        risk_per_slot = float(self._constitution.get("risk_per_slot", 0.01))
        strat_risk = self._risk_for_strategy(archetype)
        used_risk = float(self._current_slot_count()) * risk_per_slot
        remaining_risk = max(0.0, total_risk_cap - used_risk)
        bpc_in_use = self._count_family_slots("bpc")
        if (
            family_capacity_limit >= 2
            and dd <= float(step2.get("max_drawdown", 0.08))
            and daily_loss <= float(step2.get("max_daily_loss", 0.03))
            and bpc_in_use >= int(step2.get("min_active_bpc_slots", 1))
            and remaining_risk >= strat_risk
        ):
            slots = max(slots, 2)
        if (
            family_capacity_limit >= 3
            and dd <= float(step3.get("max_drawdown", 0.05))
            and daily_loss <= float(step3.get("max_daily_loss", 0.02))
            and bpc_in_use >= int(step3.get("min_active_bpc_slots", 2))
            and remaining_risk >= strat_risk
        ):
            slots = max(slots, 3)
        return max(1, min(hard_cap, family_capacity_limit, slots))

    def _risk_for_strategy(self, archetype: str) -> float:
        strat = self._limit_cfg_for_archetype(archetype)
        risk_per_slot = float(self._constitution.get("risk_per_slot", 0.01))
        if "max_risk_per_trade" not in strat:
            return risk_per_slot
        try:
            return min(risk_per_slot, float(strat.get("max_risk_per_trade")))
        except Exception:
            return risk_per_slot

    def _count_family_slots(self, family: str) -> int:
        fam = str(family or "").lower().strip()
        if not fam:
            return 0
        n = 0
        for k in self._slot_evidence:
            try:
                _, arch = k.split(":", 1)
            except Exception:
                continue
            f, _ = self._parse_family_and_side(arch)
            if f == fam:
                n += 1
        return n

    def _count_archetype_slots(self, archetype: str) -> int:
        """统计某 archetype 当前占用的 slot 数"""
        suffix = f":{archetype}"
        return sum(1 for k in self._slot_evidence if k.endswith(suffix))

    def _slot_available(self, symbol: str, archetype: str) -> bool:
        """检查策略是否有可用 slot (per-strategy 独立 + 全局上限)"""
        if self._get_open_slot_count is None:
            return True  # 未配置回调，不做限制
        # Per-strategy slot 上限
        strategy_capacity_limit = self._capacity_limit_for_strategy(archetype)
        if self._count_archetype_slots(archetype) >= strategy_capacity_limit:
            return False
        # 全局 slot 上限
        return self._current_slot_count() < self._capacity_limit

    def _record_slot(self, symbol: str, archetype: str, evidence: float) -> None:
        """记录已入场 slot"""
        key = f"{symbol}:{archetype}"
        self._slot_evidence[key] = evidence

    def notify_position_closed(self, symbol: str, archetype: str = "") -> int:
        """外部通知仓位已平仓，清理 slot 追踪

        由 PositionManager/OrderFlowListener 在仓位关闭时调用。
        """
        removed = 0
        if archetype:
            key = f"{symbol}:{archetype}"
            hit = key in self._slot_evidence or key in self._slot_risk_frac
            self._slot_evidence.pop(key, None)
            self._slot_risk_frac.pop(key, None)
            self._slot_stop_pct.pop(key, None)
            self._slot_loss_r.pop(key, None)
            if hit:
                removed += 1
            else:
                # Fallback: normalize archetype key in case of case/format mismatch.
                a_norm = str(archetype or "").strip().lower()
                candidates = [
                    kk
                    for kk in (set(self._slot_evidence) | set(self._slot_risk_frac))
                    if kk.startswith(f"{symbol}:")
                ]
                for k in candidates:
                    try:
                        _, arch = k.split(":", 1)
                    except Exception:
                        continue
                    if str(arch).strip().lower() != a_norm:
                        continue
                    self._slot_evidence.pop(k, None)
                    self._slot_risk_frac.pop(k, None)
                    self._slot_stop_pct.pop(k, None)
                    self._slot_loss_r.pop(k, None)
                    removed += 1
        else:
            # archetype 未知，清理该 symbol 的所有 slot
            to_remove = [k for k in self._slot_evidence if k.startswith(f"{symbol}:")]
            for k in to_remove:
                del self._slot_evidence[k]
                removed += 1
            to_remove_r = [
                k for k in self._slot_risk_frac if k.startswith(f"{symbol}:")
            ]
            for k in to_remove_r:
                del self._slot_risk_frac[k]
            to_remove_s = [k for k in self._slot_stop_pct if k.startswith(f"{symbol}:")]
            for k in to_remove_s:
                del self._slot_stop_pct[k]
            to_remove_l = [k for k in self._slot_loss_r if k.startswith(f"{symbol}:")]
            for k in to_remove_l:
                del self._slot_loss_r[k]
        if removed > 0:
            logger.info(
                "PCM: notify_position_closed released symbol=%s archetype=%s removed=%d remain_total=%.4f",
                symbol,
                archetype or "*",
                removed,
                self._current_total_risk_frac(),
            )
        return removed

    def update_slot_loss_r(self, symbol: str, archetype: str, current_r: float) -> None:
        """Update per-slot loss metric for deleveraging ranking.

        current_r: unrealized R multiple of an open position. Negative means loss.
        """
        key = f"{symbol}:{archetype}"
        try:
            cur = float(current_r)
        except Exception:
            return
        self._slot_loss_r[key] = max(0.0, -cur)

    # ── Quantiles 透传 ──

    def set_quantiles(self, features_df) -> None:
        """
        将 quantiles 设置给所有注册的策略（如果策略支持）。
        用于 warmup 后的 Gate quantile 规则计算。
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
            "percent_only_mode": True,
            "override_enabled": bool(self._override_config),
            "override_rules": (
                list(self._override_config.keys()) if self._override_config else []
            ),
            "constitution": {
                "risk_per_slot": self._constitution.get("risk_per_slot"),
                "per_strategy_limits": self._constitution.get("per_strategy_limits"),
                "risk_budget_policy": self._constitution.get("risk_budget_policy"),
            },
            "risk_budget_runtime": {
                "total_risk_frac": self._current_total_risk_frac(),
                "symbol_risk_frac": {
                    str(k.split(":", 1)[0]): self._current_symbol_risk_frac(
                        str(k.split(":", 1)[0])
                    )
                    for k in self._slot_risk_frac.keys()
                },
                "family_risk_frac": {
                    fam: self._current_family_risk_frac(fam)
                    for fam in {"bpc", "me", "fer", "lv"}
                },
                "slot_risk_frac": dict(self._slot_risk_frac),
                "slot_stop_pct": dict(self._slot_stop_pct),
                "slot_loss_r": dict(self._slot_loss_r),
                "reject_counts": dict(self._risk_reject_counts),
                "last_snapshot": dict(self._risk_last_snapshot),
                "last_evictions": list(self._last_evictions),
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
    capacity_limit: Optional[int] = None,
    get_open_slot_count: Optional[callable] = None,
    regime_config_path: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None,
    constitution_yaml: Optional[str] = None,
) -> LivePCM:
    """
    创建 LivePCM 实例 (v3)

    Args:
        archetype_priority: 静态优先级列表。
        capacity_limit: 显式指定容量上限。未提供时从 constitution 读取。
        get_open_slot_count: 可选回调
        regime_config_path: 可选 pcm_regime.yaml 路径
        override_config: Layer 3 Override 配置
        constitution_yaml: 可选 constitution.yaml 路径

    Returns:
        初始化好的 LivePCM（尚未注册策略）
    """
    return LivePCM(
        archetype_priority=archetype_priority,
        capacity_limit=capacity_limit,
        get_open_slot_count=get_open_slot_count,
        regime_config_path=regime_config_path,
        override_config=override_config,
        constitution_yaml=constitution_yaml,
    )
