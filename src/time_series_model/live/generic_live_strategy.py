"""
GenericLiveStrategy - 配置驱动的通用策略解析引擎

将策略逻辑完全从代码中解耦，通过 YAML 配置文件驱动决策流程。
支持任意策略的统一实现，只需提供对应的 archetype 配置。

核心组件：
1. DirectionEvaluator: 解析 direction.yaml 规则
2. GateEvaluator: 评估 gate.yaml 条件
3. EntryFilterChecker: 检查 entry_filters.yaml
4. EvidenceScorer: 计算 evidence.yaml 评分
5. ExecutionParamGenerator: 生成 execution.yaml 参数
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
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

logger = logging.getLogger(__name__)


# =============================================================================
# 1. 方向规则解析器
# =============================================================================


class DirectionEvaluator:
    """解析 direction.yaml 规则，确定交易方向"""

    def __init__(self, direction_config: Dict[str, Any]):
        self.config = direction_config
        self.rules = direction_config.get("direction_rules", [])
        self.causal_source = direction_config.get("causal_source", "unknown")

    def evaluate(self, features: Dict[str, Any]) -> Tuple[int, Optional[str]]:
        """
        评估方向规则

        Returns:
            (direction: int, matched_rule_id: Optional[str])
            direction: +1(多) / -1(空) / 0(无方向)
        """
        if not self.rules:
            return 0, None

        for rule in self.rules:
            rule_id = rule.get("id", "unknown")
            feature_name = rule.get("feature", "")
            transform = rule.get("transform", "raw")
            description = rule.get("description", "")

            # 获取特征值
            value = features.get(feature_name)
            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            # 应用变换
            direction = self._apply_transform(value, transform)

            if direction != 0:
                logger.debug(
                    f"方向匹配: rule={rule_id}, feature={feature_name}, "
                    f"value={value:.4f}, transform={transform} → direction={direction}"
                )
                return direction, rule_id

        return 0, None

    def _apply_transform(self, value: float, transform: str) -> int:
        """应用变换函数"""
        if transform == "raw":
            return int(value)
        elif transform == "sign":
            return int(np.sign(value))
        elif transform == "negate_sign":
            return int(-np.sign(value))
        elif transform == "center_sign":
            return int(np.sign(value - 0.5))
        elif transform == "threshold":
            # 需要额外参数
            threshold = 0.0
            return 1 if value > threshold else -1
        else:
            return int(value)  # 默认 raw


# =============================================================================
# 2. Gate 条件评估引擎
# =============================================================================


class GateEvaluator:
    """评估 gate.yaml 条件，进行结构性过滤"""

    def __init__(self, archetype: StrategyArchetype):
        self.archetype = archetype

    def evaluate(
        self, features: Dict[str, Any], quantiles: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str], float]:
        """
        评估 Gate 条件

        Returns:
            (passed: bool, reasons: List[str], weight: float)
        """
        if self.archetype is None:
            return True, [], 1.0

        return self.archetype.apply_gate(features, quantiles)


# =============================================================================
# 3. Entry Filter 检查器
# =============================================================================


class EntryFilterChecker:
    """检查 entry_filters.yaml 条件"""

    def __init__(self, entry_config: Dict[str, Any]):
        self.config = entry_config
        self.ef_state = DerivedEntryFeatureState()

    def check(self, features: Dict[str, Any]) -> bool:
        """检查是否满足入场条件"""
        if not self.config:
            return True

        # 更新派生特征
        ef_features = self.ef_state.update(features)
        merged = {**features, **ef_features}

        return check_entry_filters_or_single(merged, self.config)


# =============================================================================
# 4. [REMOVED] Evidence 评分计算 — 已删除 (evidence 无效)
# =============================================================================


# =============================================================================
# 5. Execution 参数生成器
# =============================================================================


class ExecutionParamGenerator:
    """根据 evidence score 生成执行参数"""

    def __init__(self, execution_config: Dict[str, Any]):
        self.config = execution_config

    def generate_params(self, evidence_score: float) -> Dict[str, Any]:
        """生成执行参数 — 统一使用全局参数（grid search 优化的）

        方案 A: 移除 tier 分档，evidence 只在 gate 层做开/关决策，
        不影响执行参数。确保向量回测 = 事件回测 = 实盘。
        """
        sl_cfg = self.config.get("stop_loss", {})
        trail_cfg = sl_cfg.get("trailing", {})

        # take_profit: 必须检查 enabled 标志，读 target_r (与向量回测一致)
        tp_cfg = self.config.get("take_profit", {})
        tp_enabled = tp_cfg.get("enabled", False)
        take_profit_r = float(tp_cfg.get("target_r", 0.0)) if tp_enabled else 0.0

        # time_stop_bars: 0 表示禁用时间止损 (fat tail 模式)
        # 注意: 不能用 `or 50`，因为 Python 中 0 or 50 = 50
        _raw_tsb = self.config.get("holding", {}).get("time_stop_bars")
        _tsb = int(_raw_tsb) if _raw_tsb is not None and int(_raw_tsb) > 0 else 0

        return {
            "tier_name": "global",
            "initial_r": float(sl_cfg.get("initial_r", 2.0)),
            "activation_r": float(trail_cfg.get("activation_r", 1.0)),
            "trail_r": float(trail_cfg.get("trail_r", 1.5)),
            "take_profit_r": take_profit_r,
            "time_stop_bars": _tsb,
            "max_holding_bars": _tsb,
            "size_multiplier": 1.0,
            "structural_exit": sl_cfg.get("structural_exit"),  # "ema200" / None
        }


# =============================================================================
# 6. 通用 LiveStrategy 主类
# =============================================================================


class GenericLiveStrategy:
    """
    配置驱动的通用 LiveStrategy 解析引擎

    通过加载策略的 archetype 配置文件，自动构建决策管线。
    支持任意策略，只需提供标准的配置文件结构。
    """

    def __init__(
        self,
        strategy_name: str,
        strategies_root: str = "config/strategies",
        holding_yaml_path: Optional[str] = None,
        trade_size: float = 1.0,
        primary_timeframe: str = "240T",
        bar_minutes: int = 240,
    ):
        self.strategy_name = strategy_name
        self.strategies_root = strategies_root
        self.trade_size = trade_size
        self.primary_timeframe = primary_timeframe
        self.bar_minutes = bar_minutes
        self.holding_yaml_path = holding_yaml_path

        # 配置组件
        self.archetype: Optional[StrategyArchetype] = None
        self.direction_evaluator: Optional[DirectionEvaluator] = None
        self.gate_evaluator: Optional[GateEvaluator] = None
        self.entry_filter_checker: Optional[EntryFilterChecker] = None
        self.execution_generator: Optional[ExecutionParamGenerator] = None

        # 状态
        self._quantiles: Dict[str, Dict[str, float]] = {}
        self._last_tier_params: Optional[Dict[str, Any]] = None
        self._last_funnel: Dict[str, Any] = (
            {}
        )  # 上次 decide() 的漏斗结果 (含丰富元数据)

        # 加载配置
        self.load_configs()

    def load_configs(self) -> None:
        """加载所有配置文件"""
        try:
            # 1. 加载 Archetype (Gate + Evidence + Execution)
            self.archetype = load_strategy_archetype(
                self.strategy_name, self.strategies_root
            )
            logger.info(
                f"✅ Archetype loaded: {len(self.archetype.gate.all_rules)} gate rules"
            )

            # 2. 加载 Direction 配置
            dir_path = (
                Path(self.strategies_root)
                / self.strategy_name
                / "archetypes"
                / "direction.yaml"
            )
            if dir_path.exists():
                with open(dir_path, "r", encoding="utf-8") as f:
                    direction_cfg = yaml.safe_load(f) or {}
                self.direction_evaluator = DirectionEvaluator(direction_cfg)
                logger.info(
                    f"✅ Direction config loaded: "
                    f"{len(direction_cfg.get('direction_rules', []))} rules, "
                    f"source={direction_cfg.get('causal_source', 'unknown')}"
                )
            else:
                logger.warning(f"⚠️  Direction config not found: {dir_path}")

            # 3. 加载 Entry Filter 配置
            self.entry_filter_checker = EntryFilterChecker(
                load_entry_filters_config(self.strategy_name, self.strategies_root)
            )
            logger.info("✅ Entry filter config loaded")

            # 4. 初始化其他评估器
            self.gate_evaluator = GateEvaluator(self.archetype)
            self.execution_generator = ExecutionParamGenerator(
                self.archetype.execution.raw or {}
            )

        except Exception as e:
            logger.error(f"❌ Failed to load configs for {self.strategy_name}: {e}")
            raise

    def set_quantiles(self, features_df) -> None:
        """设置分位数阈值（用于 Gate quantile 规则）"""
        if self.archetype is None:
            return

        quantiles = {}
        n_computed = 0

        # Gate quantile 规则 — 扫描 gate 中引用 quantile_* 的特征
        n_gate_quantiles = 0
        for rule in self.archetype.gate.all_rules:
            for feat_name, cond in rule.when.items():
                if not isinstance(cond, dict):
                    continue
                has_q = any(
                    k in cond
                    for k in (
                        "quantile_lt",
                        "quantile_lte",
                        "quantile_gt",
                        "quantile_gte",
                    )
                )
                if not has_q or feat_name in quantiles:
                    continue
                if feat_name not in features_df.columns:
                    continue
                values = pd.to_numeric(features_df[feat_name], errors="coerce").dropna()
                if len(values) < 10:
                    continue
                feat_q = {}
                for b in [
                    0.05,
                    0.1,
                    0.15,
                    0.2,
                    0.25,
                    0.3,
                    0.5,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ]:
                    q_val = float(values.quantile(b))
                    q_key = f"{b:.2f}".rstrip("0").rstrip(".")
                    feat_q[q_key] = q_val
                quantiles[feat_name] = feat_q
                n_gate_quantiles += 1

        self._quantiles = quantiles
        logger.info(
            "Quantiles 已计算: gate=%d, 基于 %d 行数据",
            n_gate_quantiles,
            len(features_df),
        )

    # set_quantiles_from_df: 兼容别名
    set_quantiles_from_df = set_quantiles

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TradeIntent]:
        """
        核心决策接口 - 通用策略解析引擎

        决策管线:
          1. Direction: 从 direction.yaml 确定方向
          2. Gate: 从 gate.yaml 进行结构性过滤
          3. Entry Filter: 从 entry_filters.yaml 检查入场时机
          4. Evidence: 从 evidence.yaml 计算评分
          5. Execution: 从 execution.yaml 生成执行参数
        """
        if not features:
            self._last_funnel = {}
            return []

        # 漏斗跟踪 (bool 标记 + 丰富元数据)
        funnel: Dict[str, Any] = {}

        # ── 1. 方向判定 ──
        if self.direction_evaluator is None:
            logger.error("❌ Direction evaluator not initialized")
            self._last_funnel = funnel
            return []

        direction, rule_id = self.direction_evaluator.evaluate(features)
        funnel["direction"] = direction != 0
        funnel["direction_value"] = direction  # 1=long, -1=short, 0=none
        if direction == 0:
            logger.debug("❌ No valid direction found")
            self._last_funnel = funnel
            return []

        side_str = "BUY" if direction == 1 else "SELL"
        logger.debug(f"🎯 Direction: {side_str} (rule: {rule_id})")

        # ── 2. Gate 过滤 ──
        gate_weight = 0.0
        if self.gate_evaluator is not None:
            gate_passed, gate_reasons, gate_weight = self.gate_evaluator.evaluate(
                features, self._quantiles
            )
            if not gate_passed:
                logger.debug(f"❌ Gate denied: {gate_reasons}")
                funnel["gate"] = False
                funnel["gate_reasons"] = gate_reasons  # 拦截原因列表
                self._last_funnel = funnel
                return []
            funnel["gate"] = True
            funnel["gate_weight"] = round(gate_weight, 4)
            logger.debug(f"✅ Gate passed (weight: {gate_weight:.3f})")

        # ── 3. Entry Filter 检查 ──
        if self.entry_filter_checker is not None:
            ef_passed = self.entry_filter_checker.check(features)
            if not ef_passed:
                logger.debug("❌ Entry filter denied")
                funnel["entry_filter"] = False
                self._last_funnel = funnel
                return []
            funnel["entry_filter"] = True
            logger.debug("✅ Entry filter passed")

        # ── 4. Evidence 评分 (已删除, 固定 0.5) ──
        evidence_score = 0.5

        funnel["evidence"] = True  # 始终通过

        # ── 5. 执行参数生成 ──
        exec_params = {}
        if self.execution_generator is not None:
            exec_params = self.execution_generator.generate_params(evidence_score)
            self._last_tier_params = exec_params
            logger.debug(f"⚙️  Execution params: {exec_params}")

        # ── 6. 构建 TradeIntent ──
        action = "LONG" if direction == 1 else "SHORT"
        intent = TradeIntent(
            action=action,
            symbol=symbol,
            archetype=self.strategy_name,
            execution_strategy=self.strategy_name,
            confidence=evidence_score,
            size_multiplier=1.0,
            execution_tags=[self.strategy_name, side_str],
            execution_profile={
                "rr_constraints": {
                    "stop_loss_r": exec_params.get("initial_r", 2.0),
                    "take_profit_r": exec_params.get("take_profit_r", 2.5),
                    "allow_trailing": True,
                    "activation_r": exec_params.get("activation_r", 1.0),
                    "trailing_atr": exec_params.get("trail_r", 1.5),
                    "max_holding_bars": exec_params.get("time_stop_bars", 50),
                    "structural_exit": exec_params.get("structural_exit"),
                },
                "strategy_specific": {
                    "direction_rule": rule_id,
                    "evidence_score": evidence_score,
                    "gate_weight": gate_weight,
                    "tier_name": exec_params.get("tier_name", "default"),
                },
            },
        )

        logger.info(
            f"✅ Signal generated: {action} {symbol} "
            f"(evidence={evidence_score:.3f}, tier={exec_params.get('tier_name')})"
        )
        self._last_funnel = funnel
        return [intent]

    # ── 兼容属性: 脚本通过 strat._archetype 访问 archetype ──

    @property
    def _archetype(self):
        """兼容属性: 脚本通过 strat._archetype 访问"""
        return self.archetype

    # ── 诊断接口: _evaluate_entry_signal ──

    def _evaluate_entry_signal(
        self,
        features: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        入场信号评估（诊断用）

        管线:
          1. direction.yaml 规则 → 方向
          2. Gate → 结构性否决
          3. Entry Filter → 入场时机
          4. Evidence → Tier 选择
          5. 返回 (should_enter, signal_info)
        """
        if not features:
            return False, {}

        # ── 1. 方向判定 ──
        if self.direction_evaluator is None:
            return False, {"reject_reason": "no_direction_config"}

        direction, rule_id = self.direction_evaluator.evaluate(features)
        if direction == 0:
            return False, {"reject_reason": "no_direction"}

        side_str = "BUY" if direction == 1 else "SELL"

        # ── 2. Gate 检查 ──
        gate_weight = 1.0
        if self.gate_evaluator is not None:
            gate_passed, gate_reasons, gate_weight = self.gate_evaluator.evaluate(
                features, self._quantiles
            )
            if not gate_passed:
                return False, {
                    "reject_reason": "gate_deny",
                    "gate_reasons": gate_reasons,
                }

        # ── 3. Entry Filter 检查 ──
        if self.entry_filter_checker is not None:
            ef_passed = self.entry_filter_checker.check(features)
            if not ef_passed:
                return False, {"reject_reason": "entry_filter_deny"}

        # ── 4. Evidence Score (已删除, 固定 0.5) ──
        evidence_score = 0.5
        adjusted_score = 0.5

        # Tier 选择
        exec_params = {}
        if self.execution_generator is not None:
            exec_params = self.execution_generator.generate_params(adjusted_score)
            self._last_tier_params = exec_params

        # ── 5. 构建 signal_info ──
        signal_info = {
            "side": side_str,
            "direction": direction,
            "reason": (
                f"{self.strategy_name.upper()}_{side_str} "
                f"(gate_w={gate_weight:.2f})"
            ),
            "evidence_score": adjusted_score,
            "evidence_breakdown": {},
            "gate_weight": gate_weight,
            "tier": exec_params,
            "atr": features.get("atr", 0.0),
        }

        return True, signal_info

    def reset(self) -> None:
        """重置状态"""
        if self.entry_filter_checker and self.entry_filter_checker.ef_state:
            self.entry_filter_checker.ef_state.reset()
        self._last_tier_params = None
