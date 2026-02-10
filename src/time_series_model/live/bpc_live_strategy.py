"""
BPC Live Strategy — 纯逻辑决策引擎

BPC = Breakout → Pullback → Continuation

决策管线:
  1. bpc_breakout_direction → 方向 (+1/-1/0)
  2. Gate (hard deny + soft weight) → 结构性否决
  3. Entry Filter (bb OR liq_silence) → 入场时机
  4. Evidence Score → Tier 选择
  5. Tier → 执行参数 (SL/TP/size/timeout)

不使用 ML 模型。所有信号来自 BPC 结构特征。

输出 TradeIntent，可直接
由 OrderFlowListener._execute_intent() 执行。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.archetype.loader import (
    StrategyArchetype,
    load_strategy_archetype,
)
from src.time_series_model.execution.entry_filter import (
    DerivedEntryFeatureState,
    check_entry_filters_or_single,
    load_entry_filters_config,
)
from src.time_series_model.live.execution_profile_apply import pick_atr

logger = logging.getLogger(__name__)


# ================================================================
# Tier selection helper
# ================================================================


def select_tier(
    evidence_score: float,
    tiers_cfg: Dict[str, Any],
    exec_config: Dict[str, Any],
) -> Dict[str, Any]:
    """根据 evidence_score 选择 tier，返回执行参数。

    Returns dict with keys:
        tier_name, initial_r, activation_r, trail_r,
        time_stop_bars, size_multiplier
    """
    levels = tiers_cfg.get("levels", [])
    # 按 evidence_min 从高到低排序
    levels = sorted(levels, key=lambda x: x.get("evidence_min", 0), reverse=True)

    for level in levels:
        if evidence_score >= level.get("evidence_min", 0):
            sl_cfg = level.get("stop_loss", {})
            trail_cfg = sl_cfg.get("trailing", {})
            return {
                "tier_name": level.get("name", "unknown"),
                "initial_r": sl_cfg.get("initial_r", 2.0),
                "activation_r": trail_cfg.get("activation_r", 1.0),
                "trail_r": trail_cfg.get("trail_r", 1.5),
                "time_stop_bars": level.get("time_stop_bars", 50),
                "size_multiplier": level.get("size_multiplier", 1.0),
            }

    # 未匹配任何 tier → 使用全局默认参数
    global_sl = exec_config.get("stop_loss", {})
    global_trail = global_sl.get("trailing", {})
    return {
        "tier_name": "default",
        "initial_r": global_sl.get("initial_r", 2.0),
        "activation_r": global_trail.get("activation_r", 1.0),
        "trail_r": global_trail.get("trail_r", 1.5),
        "time_stop_bars": exec_config.get("holding", {}).get("time_stop_bars", 50),
        "size_multiplier": 1.0,
    }


# ================================================================
# BPCLiveStrategy — pure-logic decision engine
# ================================================================


class BPCLiveStrategy:
    """
    BPC 实盘决策引擎 — 纯逻辑，不依赖 Nautilus

    接口与 OrderFlowListener 对齐:
        decide(features, symbol) → List[TradeIntent]

    可直接赋给 OrderFlowListener.decision_handler 使用。

    与旧版 BPCLiveStrategy(EventDrivenStrategy) 的区别:
    - 无 Nautilus 依赖
    - 无 EventDrivenStrategy 基类
    - 不管理 tick/bar/定时器/下单
    - 只负责 "给定特征 → 输出交易意图"
    - 持仓管理由 OrderFlowListener._enforce_open_positions() 负责
    """

    def __init__(
        self,
        strategies_root: str = "config/strategies",
        holding_yaml_path: Optional[str] = None,
        trade_size: float = 1.0,
        primary_timeframe: str = "240T",
        bar_minutes: int = 240,
    ):
        self.strategies_root = strategies_root
        self.trade_size = trade_size
        self._primary_timeframe = primary_timeframe
        self._bar_minutes = bar_minutes
        self._holding_yaml_path = holding_yaml_path

        # 配置 — 在 load_configs() 中初始化
        self._archetype: Optional[StrategyArchetype] = None
        self._entry_cfg: Dict[str, Any] = {}
        self._exec_config: Dict[str, Any] = {}
        self._tiers_cfg: Dict[str, Any] = {}
        self._holding_cfg: Dict[str, Any] = {}
        self._ef_state: Optional[DerivedEntryFeatureState] = None

        # 当前决策缓存 (上一次 decide() 的 tier params，供日志用)
        self._last_tier_params: Optional[Dict[str, Any]] = None

    # ────────────────────────────────────────────────────
    # 配置加载
    # ────────────────────────────────────────────────────

    def load_configs(self) -> None:
        """加载所有 BPC 配置（archetype / entry_filter / execution / holding）"""
        try:
            # 1. StrategyArchetype（Gate + Evidence + Execution 三层）
            self._archetype = load_strategy_archetype(
                "bpc", strategies_root=self.strategies_root
            )
            logger.info(
                "✅ BPC Archetype loaded: "
                f"{len(self._archetype.gate.all_rules)} gate rules, "
                f"{len(self._archetype.evidence.features)} evidence features"
            )

            # 2. Entry Filters
            self._entry_cfg = load_entry_filters_config(
                "bpc", strategies_root=self.strategies_root
            )
            enabled = [
                f["id"]
                for f in self._entry_cfg.get("filters", [])
                if f.get("enabled", False)
            ]
            logger.info(
                f"✅ Entry Filters loaded: {len(enabled)} enabled ({', '.join(enabled)})"
            )

            # 3. Execution 配置（tiers + 全局参数）
            exec_path = (
                Path(self.strategies_root) / "bpc" / "archetypes" / "execution.yaml"
            )
            if exec_path.exists():
                with open(exec_path, "r", encoding="utf-8") as f:
                    self._exec_config = yaml.safe_load(f) or {}
            self._tiers_cfg = self._exec_config.get("tiers", {})
            tiers_enabled = self._tiers_cfg.get("enabled", False)
            n_tiers = len(self._tiers_cfg.get("levels", []))
            logger.info(
                f"✅ Execution config: tiers={'ON' if tiers_enabled else 'OFF'} ({n_tiers} levels)"
            )

            # 4. Holding 配置（breakeven_lock 等）
            holding_path = self._holding_yaml_path
            if holding_path is None:
                candidates = [
                    "z实验_001_bpc/holding.yaml",
                    "config/strategies/bpc/archetypes/holding.yaml",
                ]
                for c in candidates:
                    if Path(c).exists():
                        holding_path = c
                        break
            if holding_path and Path(holding_path).exists():
                with open(holding_path, "r", encoding="utf-8") as f:
                    self._holding_cfg = yaml.safe_load(f) or {}
                be = self._holding_cfg.get("breakeven_lock", {})
                logger.info(
                    f"✅ Holding config: breakeven_lock="
                    f"{'ON' if be.get('enabled') else 'OFF'} "
                    f"(trigger_r={be.get('trigger_r', 'N/A')})"
                )

            # 5. DerivedEntryFeatureState
            self._ef_state = DerivedEntryFeatureState()
            logger.info("✅ DerivedEntryFeatureState initialized")
            logger.info("🚀 BPC Live Strategy fully initialized")

        except Exception as e:
            logger.error(f"❌ BPC initialization error: {e}")
            import traceback

            logger.error(traceback.format_exc())

    # ────────────────────────────────────────────────────
    # 核心决策接口 — decide(features, symbol) → List[TradeIntent]
    # ────────────────────────────────────────────────────

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TradeIntent]:
        """
        BPC 决策管线 → 产出 TradeIntent list (0 or 1 items)

        与 decide() 标准签名，可直接替换。
        """
        should_enter, signal_info = self._evaluate_entry_signal(features)
        if not should_enter:
            return []

        tier = signal_info.get("tier", {})
        direction = signal_info.get("direction", 0)
        action = "LONG" if direction == 1 else "SHORT"
        side_str = "BUY" if direction == 1 else "SELL"

        # 构建 execution_profile，供 OrderFlowListener._execute_intent() 使用
        initial_r = tier.get("initial_r", 2.0)
        activation_r = tier.get("activation_r", 1.0)
        trail_r = tier.get("trail_r", 1.5)
        time_stop_bars = tier.get("time_stop_bars", 50)

        # Breakeven lock 配置
        be_cfg = self._holding_cfg.get("breakeven_lock", {})
        be_enabled = be_cfg.get("enabled", False)
        be_trigger_r = be_cfg.get("trigger_r", 1.0)

        exec_profile = {
            "rr_constraints": {
                "stop_loss_r": initial_r,
                "take_profit_r": initial_r * 2.5,
                "allow_trailing": True,
                "trailing_atr": trail_r,
                "max_holding_bars": time_stop_bars,
            },
            # BPC 扩展字段 — OrderFlowListener 识别这些字段
            "bpc_position_config": {
                "activation_r": activation_r,
                "trail_r": trail_r,
                "breakeven_enabled": be_enabled,
                "breakeven_trigger_r": be_trigger_r,
                "bar_minutes": self._bar_minutes,
            },
        }

        intent = TradeIntent(
            action=action,
            symbol=symbol,
            archetype="bpc",
            execution_strategy="bpc",
            confidence=signal_info.get("evidence_score", 0.5),
            size_multiplier=tier.get("size_multiplier", 1.0),
            execution_tags=["bpc", side_str],
            execution_profile=exec_profile,
        )
        return [intent]

    # ────────────────────────────────────────────────────
    # 内部决策管线
    # ────────────────────────────────────────────────────

    def _evaluate_entry_signal(
        self,
        features: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        BPC 入场信号评估

        管线:
          1. bpc_breakout_direction → 方向
          2. Gate → 结构性否决
          3. Entry Filter → 入场时机
          4. Evidence → Tier 选择
          5. 返回 (should_enter, signal_info)
        """
        if not features:
            return False, {}

        # ── 1. 方向: bpc_breakout_direction ──
        direction = features.get("bpc_breakout_direction", 0)
        try:
            direction = int(float(direction))
        except (TypeError, ValueError):
            direction = 0

        if direction == 0:
            return False, {"reject_reason": "no_direction"}

        side_str = "BUY" if direction == 1 else "SELL"

        # ── 2. Gate 检查 ──
        if self._archetype is not None:
            gate_passed, gate_reasons, gate_weight = self._archetype.apply_gate(
                features
            )
            if not gate_passed:
                return False, {
                    "reject_reason": "gate_deny",
                    "gate_reasons": gate_reasons,
                }
        else:
            gate_weight = 1.0

        # ── 3. Entry Filter 检查 ──
        if self._ef_state is not None:
            ef_features = self._ef_state.update(features)
            merged = {**features, **ef_features}
        else:
            merged = features

        if self._entry_cfg:
            ef_passed = check_entry_filters_or_single(merged, self._entry_cfg)
            if not ef_passed:
                return False, {"reject_reason": "entry_filter_deny"}

        # ── 4. Evidence Score + Tier 选择 ──
        evidence_score = 0.5
        evidence_breakdown = {}
        if self._archetype is not None:
            evidence_score, evidence_breakdown = self._archetype.compute_evidence_score(
                features
            )

        # 应用 gate soft filter weight
        adjusted_score = evidence_score * gate_weight

        # Tier 选择
        tier_params = select_tier(adjusted_score, self._tiers_cfg, self._exec_config)

        # 缓存 tier 参数
        self._last_tier_params = tier_params

        # ── 5. 构建 signal_info ──
        atr = features.get("atr", 0.0)

        signal_info = {
            "side": side_str,
            "direction": direction,
            "reason": (
                f"BPC_{side_str} "
                f"(evidence={adjusted_score:.2f}, "
                f"tier={tier_params['tier_name']}, "
                f"gate_w={gate_weight:.2f})"
            ),
            # Execution params
            "evidence_score": adjusted_score,
            "evidence_breakdown": evidence_breakdown,
            "gate_weight": gate_weight,
            "tier": tier_params,
            "atr": atr,
            # BPC-specific features for logging
            "bpc_breakout_direction": direction,
            "bpc_score_breakout": features.get("bpc_score_breakout", 0),
            "bpc_was_in_pullback": features.get("bpc_was_in_pullback", 0),
        }

        return True, signal_info

    # ────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────

    def reset(self) -> None:
        """重置有状态组件"""
        if self._ef_state:
            self._ef_state.reset()
        self._last_tier_params = None
