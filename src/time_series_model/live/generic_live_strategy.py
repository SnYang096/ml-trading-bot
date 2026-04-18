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

from collections import deque
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
from src.time_series_model.live.fer_diagnostics import record_fer_entry_eval
from src.time_series_model.live.srb_regime import (
    pick_srb_true_sr_level,
    should_reject_srb_wide_entry,
)
from src.time_series_model.live.direction_rule_ops import (
    dual_position_agree_deadband_scalar,
    is_direction_rule_enabled,
    parse_dual_rule,
    parse_signal_match_position_band_rule,
    parse_single_position_band_rule,
    single_position_band_scalar,
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
        # fixed_direction: long/short → 忽略 direction_rules，强制固定方向
        _fd = direction_config.get("fixed_direction", None)
        if _fd == "long":
            self._fixed = 1
        elif _fd == "short":
            self._fixed = -1
        else:
            self._fixed = None
        # direction_filter: long/short → 方向模型正常运行，但只接受指定方向
        # 方向模型说 SHORT 时返回 0（跳过），不强制反向
        _df = direction_config.get("direction_filter", None)
        if _df == "long":
            self._filter = 1
        elif _df == "short":
            self._filter = -1
        else:
            self._filter = None

    def evaluate(self, features: Dict[str, Any]) -> Tuple[int, Optional[str]]:
        """
        评估方向规则

        Returns:
            (direction: int, matched_rule_id: Optional[str])
            direction: +1(多) / -1(空) / 0(无方向)
        """
        # fixed_direction 优先 — 跳过所有规则直接返回固定方向
        if self._fixed is not None:
            return self._fixed, "fixed_direction"

        if not self.rules:
            return 0, None

        for rule in self.rules:
            if not is_direction_rule_enabled(rule):
                continue
            rule_id = rule.get("id", "unknown")
            compound = parse_signal_match_position_band_rule(rule)
            if compound is not None:
                consensus = compound.get("consensus_mode", "first")
                candidate = 0
                if consensus == "all":
                    votes = []
                    for sr in compound["signal_rules"]:
                        if not isinstance(sr, dict):
                            continue
                        if not is_direction_rule_enabled(sr):
                            continue
                        d_atom = self._evaluate_atomic_direction_rule(sr, features)
                        if d_atom != 0:
                            votes.append(d_atom)
                    if votes and all(v == votes[0] for v in votes):
                        candidate = votes[0]
                else:
                    for sr in compound["signal_rules"]:
                        if not isinstance(sr, dict):
                            continue
                        if not is_direction_rule_enabled(sr):
                            continue
                        d_atom = self._evaluate_atomic_direction_rule(sr, features)
                        if d_atom != 0:
                            candidate = d_atom
                            break
                if candidate == 0:
                    continue
                band_dir = single_position_band_scalar(
                    features.get(compound["band_feature"]),
                    float(compound["inner_abs"]),
                    float(compound["outer_abs"]),
                )
                if band_dir != candidate:
                    continue
                logger.debug(
                    "方向匹配: rule=%s signal_match_position_band → direction=%s",
                    rule_id,
                    candidate,
                )
                if self._filter is not None and candidate != self._filter:
                    return 0, None
                return candidate, str(rule_id)

            dual = parse_dual_rule(rule)
            if dual is not None:
                col_a, col_b, eps = dual
                direction = dual_position_agree_deadband_scalar(
                    features.get(col_a), features.get(col_b), eps
                )
                if direction != 0:
                    logger.debug(
                        "方向匹配: rule=%s dual_deadband %s/%s eps=%s → direction=%s",
                        rule_id,
                        col_a,
                        col_b,
                        eps,
                        direction,
                    )
                    if self._filter is not None and direction != self._filter:
                        return 0, None
                    return direction, str(rule_id)
                continue

            band = parse_single_position_band_rule(rule)
            if band is not None:
                col, inner_abs, outer_abs = band
                direction = single_position_band_scalar(
                    features.get(col), inner_abs, outer_abs
                )
                if direction != 0:
                    if self._filter is not None and direction != self._filter:
                        return 0, None
                    return direction, str(rule_id)
                continue

            feature_name = rule.get("feature", "")
            transform = rule.get("transform", "raw")

            value = features.get(feature_name)
            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            direction = self._apply_transform(value, transform)

            if direction != 0:
                logger.debug(
                    f"方向匹配: rule={rule_id}, feature={feature_name}, "
                    f"value={value:.4f}, transform={transform} → direction={direction}"
                )
                if self._filter is not None and direction != self._filter:
                    return 0, None
                return direction, str(rule_id)

        return 0, None

    def _evaluate_atomic_direction_rule(
        self, rule: Dict[str, Any], features: Dict[str, Any]
    ) -> int:
        """单条子规则（不含 signal_match_position_band 嵌套）→ ±1 或 0。"""
        if not isinstance(rule, dict):
            return 0
        if str(rule.get("method", "")).strip().lower() == "signal_match_position_band":
            return 0
        dual = parse_dual_rule(rule)
        if dual is not None:
            col_a, col_b, eps = dual
            return dual_position_agree_deadband_scalar(
                features.get(col_a), features.get(col_b), eps
            )
        band = parse_single_position_band_rule(rule)
        if band is not None:
            col, inner_abs, outer_abs = band
            return single_position_band_scalar(features.get(col), inner_abs, outer_abs)
        feature_name = rule.get("feature", "")
        transform = rule.get("transform", "raw")
        value = features.get(feature_name)
        if value is None:
            return 0
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0
        return self._apply_transform(value, transform)

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
# 4. Evidence 评分 — 由 archetype.evidence_config.compute_composite_score() 计算
#    在 GenericLiveStrategy.decide() 和 check_signal() 中内联调用
# =============================================================================


# =============================================================================
# 5. Execution 参数生成器
# =============================================================================


class ExecutionParamGenerator:
    """根据 evidence score 生成执行参数"""

    def __init__(self, execution_config: Dict[str, Any]):
        self.config = execution_config

    def generate_params(
        self,
        evidence_score: float,
        features: Optional[Dict[str, Any]] = None,
        direction: Optional[int] = None,
    ) -> Dict[str, Any]:
        """生成执行参数 — 统一使用全局参数（grid search 优化的）

        可选 ``features`` / ``direction``：SRB 实验用（regime_execution / sr_structural_exit），
        由事件回测注入 ``srb_regime_*`` / ``srb_sr_*`` 后传入；缺省行为与旧版完全一致。
        """
        features = features or {}
        sl_cfg = self.config.get("stop_loss", {})
        trail_cfg = sl_cfg.get("trailing", {}) or {}
        guardrails = sl_cfg.get("guardrails", {}) or {}
        breakeven_cfg = sl_cfg.get("breakeven", {}) or {}
        exec_constraints = self.config.get("execution_constraints", {}) or {}
        trailing_enabled = bool(trail_cfg.get("enabled", True))

        # take_profit: 必须检查 enabled 标志，读 target_r (与向量回测一致)
        tp_cfg = self.config.get("take_profit", {})
        tp_enabled = tp_cfg.get("enabled", False)
        take_profit_r = float(tp_cfg.get("target_r", 0.0)) if tp_enabled else 0.0

        # time_stop_bars: 0 表示禁用时间止损 (fat tail 模式)
        # 注意: 不能用 `or 50`，因为 Python 中 0 or 50 = 50
        holding = self.config.get("holding", {}) or {}
        _raw_tsb = holding.get("time_stop_bars")
        _raw_mhb = holding.get("max_holding_bars")
        if _raw_tsb is not None and int(_raw_tsb) == 0:
            _tsb = 0
        elif _raw_tsb is not None and int(_raw_tsb) > 0:
            _tsb = int(_raw_tsb)
        elif _raw_mhb is not None and int(_raw_mhb) > 0:
            _tsb = int(_raw_mhb)
        else:
            _tsb = 0
        # 加仓前需要利润锁定；默认对 allow_add_on 策略启用 breakeven lock。
        breakeven_enabled = bool(
            breakeven_cfg.get(
                "enabled",
                bool(exec_constraints.get("allow_add_on", False)),
            )
        )
        breakeven_trigger_r = float(breakeven_cfg.get("trigger_r", 1.0))
        breakeven_lock_profit_atr = float(
            breakeven_cfg.get("lock_profit_atr", 0.0) or 0.0
        )

        activation_r = (
            float(trail_cfg.get("activation_r", 1.0)) if trailing_enabled else None
        )
        trail_r = float(trail_cfg.get("trail_r", 1.5)) if trailing_enabled else None
        trail_expand_primary_atr = bool(trail_cfg.get("expand_with_primary_atr", False))

        result: Dict[str, Any] = {
            "tier_name": "global",
            "initial_r": float(sl_cfg.get("initial_r", 2.0)),
            "activation_r": activation_r,
            "trail_r": trail_r,
            "take_profit_r": take_profit_r,
            "time_stop_bars": _tsb,
            "max_holding_bars": _tsb,
            "size_multiplier": 1.0,
            "structural_exit": sl_cfg.get("structural_exit"),
            "min_stop_pct": guardrails.get("min_stop_pct"),
            "max_stop_pct": guardrails.get("max_stop_pct"),
            "breakeven_enabled": breakeven_enabled,
            "breakeven_trigger_r": breakeven_trigger_r,
            "breakeven_lock_profit_atr": breakeven_lock_profit_atr,
            "allow_trailing": trailing_enabled and activation_r is not None,
            "trail_expand_primary_atr": trail_expand_primary_atr,
        }

        re_cfg = self.config.get("regime_execution") or {}
        if re_cfg.get("enabled"):
            bucket = str(features.get("srb_regime_bucket", "unknown"))
            buckets = re_cfg.get("buckets") or {}
            patch = buckets.get(bucket) or buckets.get("default") or {}
            for k in ("initial_r", "activation_r", "trail_r", "take_profit_r"):
                if k in patch and patch[k] is not None:
                    try:
                        result[k] = float(patch[k])
                    except (TypeError, ValueError):
                        pass
            if "size_multiplier" in patch and patch["size_multiplier"] is not None:
                try:
                    result["size_multiplier"] = float(patch["size_multiplier"])
                except (TypeError, ValueError):
                    pass
            if "allow_trailing" in patch:
                at = bool(patch["allow_trailing"])
                result["allow_trailing"] = at and result.get("activation_r") is not None

        se_cfg = self.config.get("sr_structural_exit") or {}
        if se_cfg.get("enabled") and direction is not None:
            buf = float(se_cfg.get("buffer_atr", 0.25) or 0.25)
            sp: Optional[float] = None
            if direction == 1:
                sp = features.get("srb_sr_support")
            elif direction == -1:
                sp = features.get("srb_sr_resistance")
            if sp is not None:
                try:
                    fp = float(sp)
                    if np.isfinite(fp):
                        result["structural_exit"] = "sr_break_level"
                        result["sr_exit_price"] = fp
                        result["sr_exit_buffer_atr"] = buf
                except (TypeError, ValueError):
                    pass

        return result


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
        self._decision_alignment: Dict[str, Any] = {
            "enabled": False,
            "mode": "prefilter_recent_window",
            "window_bars": 0,
            "layers_allow_recent_window": [],
            "layers_required_same_bar": [],
        }
        self._prefilter_recent_state: Dict[str, deque] = {}

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
                f"\u2705 Archetype loaded: "
                f"{len(self.archetype.prefilter.rules)} prefilter rules, "
                f"{len(self.archetype.gate.all_rules)} gate rules"
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
                _fd = direction_cfg.get("fixed_direction")
                _n_rules = len(direction_cfg.get("direction_rules", []))
                logger.info(
                    f"✅ Direction config loaded: "
                    f"fixed_direction={_fd or 'none'}, {_n_rules} rules"
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
            self._decision_alignment = self._load_decision_alignment_config()
            logger.info(
                "✅ Decision alignment: enabled=%s mode=%s window_bars=%d",
                self._decision_alignment.get("enabled", False),
                self._decision_alignment.get("mode", "prefilter_recent_window"),
                self._decision_alignment.get("window_bars", 0),
            )

        except Exception as e:
            logger.error(f"❌ Failed to load configs for {self.strategy_name}: {e}")
            raise

    def _load_decision_alignment_config(self) -> Dict[str, Any]:
        """从 strategy meta.yaml 读取跨 bar 对齐配置。"""
        cfg: Dict[str, Any] = {
            "enabled": False,
            "mode": "prefilter_recent_window",
            "window_bars": 0,
            "layers_allow_recent_window": [],
            "layers_required_same_bar": [],
        }
        meta_path = Path(self.strategies_root) / self.strategy_name / "meta.yaml"
        if not meta_path.exists():
            return cfg

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                raw_meta = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("⚠️  Failed to read meta.yaml for alignment config: %s", exc)
            return cfg

        meta_strategy = raw_meta.get("strategy", raw_meta)
        da = (
            meta_strategy.get("decision_alignment")
            or raw_meta.get("decision_alignment")
            or {}
        )
        if not isinstance(da, dict):
            return cfg

        window_bars = int(
            da.get("window_bars", da.get("alignment_window_bars", 0)) or 0
        )
        mode = str(da.get("mode", "prefilter_recent_window")).strip().lower()
        enabled = bool(da.get("enabled", False)) and window_bars > 0
        if mode not in {"prefilter_recent_window"}:
            mode = "prefilter_recent_window"

        cfg.update(
            {
                "enabled": enabled,
                "mode": mode,
                "window_bars": max(0, window_bars),
                "layers_allow_recent_window": list(
                    da.get("layers_allow_recent_window") or []
                ),
                "layers_required_same_bar": list(
                    da.get("layers_required_same_bar") or []
                ),
            }
        )
        return cfg

    def _update_prefilter_recent_state(self, symbol: str, passed: bool) -> None:
        """更新按 symbol 隔离的 prefilter 近期通过窗口。"""
        if not self._decision_alignment.get("enabled", False):
            return
        if self._decision_alignment.get("mode") != "prefilter_recent_window":
            return
        if "prefilter" not in self._decision_alignment.get(
            "layers_allow_recent_window", []
        ):
            return

        window = int(self._decision_alignment.get("window_bars", 0))
        if window <= 0:
            return
        key = str(symbol or "")
        if key not in self._prefilter_recent_state:
            self._prefilter_recent_state[key] = deque(maxlen=window)
        self._prefilter_recent_state[key].append(bool(passed))

    def _has_recent_prefilter_pass(self, symbol: str) -> bool:
        """最近 window_bars 内是否有 prefilter 通过。"""
        key = str(symbol or "")
        q = self._prefilter_recent_state.get(key)
        if not q:
            return False
        return any(q)

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
          0. Prefilter: 从 prefilter.yaml 检查前置环境条件
          1. Direction: 从 direction.yaml 确定方向
          2. Gate: 从 gate.yaml 进行结构性过滤
          3. Entry Filter: 从 entry_filters.yaml 检查入场时机
          4. Evidence: 从 evidence.yaml 计算评分
          5. Execution: 从 execution.yaml 生成执行参数
        """
        if not features:
            self._last_funnel = {}
            record_fer_entry_eval(
                strategy=self.strategy_name,
                symbol=symbol,
                signal_ts=None,
                outcome="empty_features",
                funnel={},
                features={},
            )
            return []

        # 漏斗跟踪 (bool 标记 + 丰富元数据)
        funnel: Dict[str, Any] = {}
        _sig_ts = features.get("timestamp")

        # ── 0. Prefilter 前置条件检查 ──
        if self.archetype and self.archetype.prefilter.rules:
            pf_passed, pf_reason = self.archetype.prefilter.evaluate(features)
            funnel["prefilter"] = pf_passed
            self._update_prefilter_recent_state(symbol, pf_passed)
            if not pf_passed:
                alignment_used = self._has_recent_prefilter_pass(symbol)
                funnel["prefilter_recent_pass"] = alignment_used
                funnel["alignment_used"] = alignment_used
                logger.debug(f"❌ Prefilter denied: {pf_reason}")
                if alignment_used:
                    logger.debug(
                        "↪️ Alignment override: recent prefilter pass within last %d bars",
                        int(self._decision_alignment.get("window_bars", 0)),
                    )
                    funnel["prefilter_alignment_override"] = True
                else:
                    funnel["prefilter_alignment_override"] = False
                funnel["prefilter_reason"] = pf_reason
                if not alignment_used:
                    self._last_funnel = funnel
                    record_fer_entry_eval(
                        strategy=self.strategy_name,
                        symbol=symbol,
                        signal_ts=_sig_ts,
                        outcome="prefilter_deny",
                        funnel=funnel,
                        features=features,
                    )
                    return []
            logger.debug("✅ Prefilter passed")

        # ── 1. 方向判定 ──
        if self.direction_evaluator is None:
            logger.error("❌ Direction evaluator not initialized")
            self._last_funnel = funnel
            record_fer_entry_eval(
                strategy=self.strategy_name,
                symbol=symbol,
                signal_ts=_sig_ts,
                outcome="no_direction_config",
                funnel=funnel,
                features=features,
            )
            return []

        direction, rule_id = self.direction_evaluator.evaluate(features)
        funnel["direction"] = direction != 0
        funnel["direction_value"] = direction  # 1=long, -1=short, 0=none
        funnel["direction_rule"] = rule_id
        if direction == 0:
            logger.debug("❌ No valid direction found")
            self._last_funnel = funnel
            record_fer_entry_eval(
                strategy=self.strategy_name,
                symbol=symbol,
                signal_ts=_sig_ts,
                outcome="no_direction",
                funnel=funnel,
                features=features,
            )
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
                record_fer_entry_eval(
                    strategy=self.strategy_name,
                    symbol=symbol,
                    signal_ts=_sig_ts,
                    outcome="gate_deny",
                    funnel=funnel,
                    features=features,
                )
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
                record_fer_entry_eval(
                    strategy=self.strategy_name,
                    symbol=symbol,
                    signal_ts=_sig_ts,
                    outcome="entry_filter_deny",
                    funnel=funnel,
                    features=features,
                )
                return []
            funnel["entry_filter"] = True
            logger.debug("✅ Entry filter passed")

        # ── 4. Evidence 评分 ──
        evidence_score = 0.5
        evidence_active = False
        evidence_breakdown = {}
        if (
            self.archetype
            and self.archetype.evidence
            and self.archetype.evidence.features
        ):
            feature_values = {
                feat.feature: features.get(feat.feature)
                for feat in self.archetype.evidence.features
                if features.get(feat.feature) is not None
            }
            if feature_values:
                evidence_active = True
                evidence_score, evidence_breakdown = (
                    self.archetype.evidence.compute_composite_score(
                        feature_values, self._quantiles
                    )
                )
                logger.debug(
                    f"📊 Evidence: score={evidence_score:.3f}, "
                    f"breakdown={evidence_breakdown}"
                )

        funnel["evidence"] = evidence_active
        funnel["evidence_score"] = round(evidence_score, 4)

        # ── 5. 执行参数生成 ──
        exec_params = {}
        if self.execution_generator is not None:
            exec_params = self.execution_generator.generate_params(
                evidence_score, features=features, direction=direction
            )
            self._last_tier_params = exec_params
            logger.debug(f"⚙️  Execution params: {exec_params}")

        action = "LONG" if direction == 1 else "SHORT"
        _srb_true_sr: Optional[float] = None

        if str(self.strategy_name).lower() == "srb":
            funnel["srb_regime_bucket"] = features.get("srb_regime_bucket")
            funnel["srb_regime_adx14"] = features.get("srb_regime_adx14")
            funnel["srb_regime_er20"] = features.get("srb_regime_er20")
            funnel["srb_sr_support"] = features.get("srb_sr_support")
            funnel["srb_sr_resistance"] = features.get("srb_sr_resistance")
            funnel["srb_sr_support_wide"] = features.get("srb_sr_support_wide")
            funnel["srb_sr_resistance_wide"] = features.get("srb_sr_resistance_wide")

            _raw_ex = (self.archetype.execution.raw or {}) if self.archetype else {}
            _wg = _raw_ex.get("sr_wide_entry_guard") or {}
            if _wg.get("enabled"):
                _mn = float(_wg.get("min_distance_atr", 0) or 0)
                _cl = float(features.get("close") or 0)
                _at = float(features.get("atr") or 0)
                if should_reject_srb_wide_entry(
                    action,
                    _cl,
                    _at,
                    features.get("srb_sr_support_wide"),
                    features.get("srb_sr_resistance_wide"),
                    _mn,
                ):
                    funnel["reject_srb_wide_sr_too_close"] = True
                    self._last_funnel = funnel
                    record_fer_entry_eval(
                        strategy=self.strategy_name,
                        symbol=symbol,
                        signal_ts=_sig_ts,
                        outcome="srb_wide_sr_guard",
                        funnel=funnel,
                        features=features,
                    )
                    return []

            _fbr = _raw_ex.get("fake_break_reverse") or {}
            _fb_atr = float(_fbr.get("true_sr_wide_fallback_atr", 0) or 0)
            _srb_true_sr = pick_srb_true_sr_level(
                action,
                float(features.get("close") or 0),
                float(features.get("atr") or 0),
                narrow_support=features.get("srb_sr_support"),
                narrow_resistance=features.get("srb_sr_resistance"),
                wide_support=features.get("srb_sr_support_wide"),
                wide_resistance=features.get("srb_sr_resistance_wide"),
                fallback_atr=_fb_atr,
            )

        # ── 6. 构建 TradeIntent ──
        # evidence 缩放: score 0→0.5x, 0.5→0.75x, 1→1.0x；可选 regime size_multiplier
        _reg_sm = float(exec_params.get("size_multiplier", 1.0) or 1.0)
        ev_size_multiplier = (0.5 + evidence_score) * _reg_sm
        intent = TradeIntent(
            action=action,
            symbol=symbol,
            archetype=self.strategy_name,
            execution_strategy=self.strategy_name,
            confidence=evidence_score,
            size_multiplier=ev_size_multiplier,
            execution_tags=[self.strategy_name, side_str],
            execution_profile={
                "rr_constraints": {
                    "stop_loss_r": exec_params.get("initial_r", 2.0),
                    "take_profit_r": exec_params.get("take_profit_r", 2.5),
                    "allow_trailing": bool(exec_params.get("allow_trailing", True)),
                    "activation_r": exec_params.get("activation_r"),
                    "trailing_atr": exec_params.get("trail_r"),
                    "max_holding_bars": exec_params.get("time_stop_bars", 50),
                    "structural_exit": exec_params.get("structural_exit"),
                    "sr_exit_price": exec_params.get("sr_exit_price"),
                    "sr_exit_buffer_atr": exec_params.get("sr_exit_buffer_atr"),
                    "min_stop_pct": exec_params.get("min_stop_pct"),
                    "max_stop_pct": exec_params.get("max_stop_pct"),
                    "trail_expand_primary_atr": bool(
                        exec_params.get("trail_expand_primary_atr", False)
                    ),
                },
                "bpc_position_config": {
                    **(
                        {
                            "activation_r": exec_params.get("activation_r"),
                            "trail_r": exec_params.get("trail_r"),
                        }
                        if exec_params.get("activation_r") is not None
                        else {}
                    ),
                    "breakeven_enabled": exec_params.get("breakeven_enabled", False),
                    "breakeven_trigger_r": exec_params.get("breakeven_trigger_r", 1.0),
                    "breakeven_lock_profit_atr": exec_params.get(
                        "breakeven_lock_profit_atr", 0.0
                    ),
                },
                "strategy_specific": {
                    "direction_rule": rule_id,
                    "evidence_score": evidence_score,
                    "gate_weight": gate_weight,
                    "tier_name": exec_params.get("tier_name", "default"),
                    **(
                        {"srb_true_sr_level": float(_srb_true_sr)}
                        if _srb_true_sr is not None
                        else {}
                    ),
                },
                "add_position": (
                    (self.archetype.execution.raw or {}).get("add_position") or {}
                ),
            },
        )

        tier_name = exec_params.get("tier_name")
        if evidence_active:
            logger.info(
                f"✅ Signal generated: {action} {symbol} "
                f"(tier={tier_name}, evidence={evidence_score:.3f})"
            )
        else:
            logger.info(
                f"✅ Signal generated: {action} {symbol} "
                f"(tier={tier_name}, evidence=off)"
            )
        self._last_funnel = funnel
        record_fer_entry_eval(
            strategy=self.strategy_name,
            symbol=symbol,
            signal_ts=_sig_ts,
            outcome="signal",
            funnel=funnel,
            features=features,
        )
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

        # ── 4. Evidence Score ──
        evidence_score = 0.5
        adjusted_score = 0.5
        evidence_breakdown = {}
        if (
            self.archetype
            and self.archetype.evidence
            and self.archetype.evidence.features
        ):
            feature_values = {
                feat.feature: features.get(feat.feature)
                for feat in self.archetype.evidence.features
                if features.get(feat.feature) is not None
            }
            if feature_values:
                adjusted_score, evidence_breakdown = (
                    self.archetype.evidence.compute_composite_score(
                        feature_values, self._quantiles
                    )
                )
                evidence_score = adjusted_score

        # Tier 选择
        exec_params = {}
        if self.execution_generator is not None:
            exec_params = self.execution_generator.generate_params(
                adjusted_score, features=features, direction=direction
            )
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
            "evidence_breakdown": evidence_breakdown,
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
        self._prefilter_recent_state.clear()
