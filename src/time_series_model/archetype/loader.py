"""
Archetype Loader - 三层配置加载器

从 config/strategies/{strategy}/archetypes/ 加载：
- gate.yaml: Gate 规则 (硬 veto)
- evidence.yaml: Evidence 规则 (软调整)
- execution.yaml: Execution 约束 (RR/持仓)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# =============================================================================
# Gate Config
# =============================================================================


@dataclass
class GateRule:
    """单条 Gate 规则"""

    id: str
    tag: str
    phase: str  # system_safety / hard_gate / guardrail
    priority: int
    reason: str
    when: Dict[str, Any]
    then: Dict[str, Any]
    frozen: bool = False  # 禁止优化阈值

    @property
    def is_hard(self) -> bool:
        return self.phase in ("system_safety", "hard_gate", "guardrail")


# Prefilter operator → gate deny-condition mapping.
# Prefilter says "pass if feature >= 0.922"; gate denies the complement.
_PREFILTER_OP_TO_DENY: Dict[str, str] = {
    ">=": "value_lt",  # pass >= X  → deny < X
    ">": "value_lte",  # pass > X   → deny <= X
    "<=": "value_gt",  # pass <= X  → deny > X
    "<": "value_gte",  # pass < X   → deny >= X
}


def _load_prefilter_as_guardrails(prefilter_path: Path) -> List[GateRule]:
    """Load ``prefilter.yaml`` and convert each rule to a guardrail :class:`GateRule`.

    The conversion inverts the prefilter's *pass* condition so the gate
    **denies** rows that would NOT have passed prefilter during training.
    This keeps ``prefilter.yaml`` as the single source of truth for the
    archetype's semantic boundary — no need to duplicate rules in gate.yaml.
    """
    if not prefilter_path.exists():
        return []
    raw = yaml.safe_load(prefilter_path.read_text(encoding="utf-8")) or {}
    rules_list = raw.get("rules") or []
    guardrails: List[GateRule] = []
    for idx, r in enumerate(rules_list):
        feature = str(r.get("feature", ""))
        operator = str(r.get("operator", "")).strip()
        value = r.get("value")
        rationale = str(r.get("rationale", ""))
        deny_op = _PREFILTER_OP_TO_DENY.get(operator)
        if not deny_op or not feature:
            continue
        guardrails.append(
            GateRule(
                id=f"prefilter_{feature}",
                tag=f"PREFILTER_{feature.upper()}",
                phase="guardrail",
                priority=100 + idx,  # after hard_gates
                reason=f"Prefilter guardrail: {feature} {operator} {value} ({rationale})",
                when={feature: {deny_op: value}},
                then={"action": "deny"},
                frozen=True,  # prefilter 阈值由 prefilter.yaml 管理
            )
        )
    return guardrails


@dataclass
class GateConfig:
    """Gate 配置 - 从 gate.yaml + prefilter.yaml 加载"""

    hard_gates: List[GateRule] = field(default_factory=list)
    system_safety: List[GateRule] = field(default_factory=list)
    guardrails: List[GateRule] = field(default_factory=list)
    governance: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_rules(self) -> List[GateRule]:
        """按 phase -> priority 排序的所有规则（含 guardrails）"""
        phase_order = {"system_safety": 0, "hard_gate": 1, "guardrail": 2}
        all_rules = self.system_safety + self.hard_gates + self.guardrails
        return sorted(
            all_rules, key=lambda r: (phase_order.get(r.phase, 99), r.priority)
        )

    @property
    def hard_rules(self) -> List[GateRule]:
        """所有硬规则 (system_safety + hard_gate + guardrail)"""
        return self.system_safety + self.hard_gates + self.guardrails

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        prefilter_path: Optional[Path] = None,
    ) -> "GateConfig":
        """从 gate.yaml 加载，可选自动注入 prefilter.yaml 作为 guardrails。"""
        if not path.exists():
            return cls()

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        def _parse_rules(rules_list: List[Dict], default_phase: str) -> List[GateRule]:
            result = []
            for r in rules_list or []:
                result.append(
                    GateRule(
                        id=str(r.get("id", "")),
                        tag=str(r.get("tag", r.get("id", ""))),
                        phase=str(r.get("phase", default_phase)),
                        priority=int(r.get("priority", 99)),
                        reason=str(r.get("reason", "")),
                        when=dict(r.get("when") or {}),
                        then=dict(r.get("then") or {}),
                        frozen=bool(r.get("frozen", False)),
                    )
                )
            return result

        # gate.yaml 中的 guardrails（向后兼容）+ prefilter.yaml 自动注入
        yaml_guardrails = _parse_rules(raw.get("guardrails"), "guardrail")
        prefilter_guardrails = (
            _load_prefilter_as_guardrails(prefilter_path)
            if prefilter_path is not None
            else []
        )

        return cls(
            hard_gates=_parse_rules(raw.get("hard_gates"), "hard_gate"),
            system_safety=_parse_rules(raw.get("system_safety"), "system_safety"),
            guardrails=yaml_guardrails + prefilter_guardrails,
            governance=dict(
                raw.get("governance") or raw.get("schema", {}).get("governance") or {}
            ),
        )


# =============================================================================
# Evidence Config
# =============================================================================

_DIRECTION_MAP = {
    "positive": "higher_is_better",
    "negative": "lower_is_better",
    "higher_is_better": "higher_is_better",
    "lower_is_better": "lower_is_better",
}


def _map_direction(raw: str) -> str:
    """Map YAML direction values to internal direction constants."""
    return _DIRECTION_MAP.get(str(raw).lower().strip(), "higher_is_better")


@dataclass
class EvidenceFeature:
    """单个 Evidence 特征"""

    id: str
    feature: str
    rank: int
    split_count: int
    usage_hint: str
    affects: List[str]
    quantile_bins: List[float]
    quantile_labels: List[str]
    threshold_examples: List[float]
    distribution_hint: str
    # ❗ Bug 1 修复: 特征方向
    # "higher_is_better": 值越大越好 (如 strength, momentum)
    # "lower_is_better": 值越小越好 (如 volatility, risk, drawdown)
    direction: str = "higher_is_better"

    def compute_label(self, value: float, quantiles: Dict[str, float]) -> str:
        """
        根据 quantile_mapping 计算语义标签

        Args:
            value: 特征原始值
            quantiles: {feature: {0.2: v1, 0.4: v2, ...}} 分位数查找表

        Returns:
            语义标签: suppress/downweight/neutral/favor/amplify
        """
        # 处理 quantiles 为 None 的情况 - 用 value 直接作为分位数
        if quantiles is None:
            # 假设 value 已经是 [0, 1] 范围的分位数
            percentile = value
            for i, bin_val in enumerate(self.quantile_bins):
                if percentile <= bin_val:
                    return (
                        self.quantile_labels[i]
                        if i < len(self.quantile_labels)
                        else "neutral"
                    )
            return self.quantile_labels[-1] if self.quantile_labels else "neutral"

        feat_q = quantiles.get(self.feature, {})
        if not feat_q:
            return "neutral"

        # 获取分位数阈值
        thresholds = []
        for q in self.quantile_bins:
            q_key = f"{q:.2f}".rstrip("0").rstrip(".")
            if q_key in feat_q:
                thresholds.append(float(feat_q[q_key]))
            elif str(q) in feat_q:
                thresholds.append(float(feat_q[str(q)]))
            else:
                return "neutral"  # 缺少分位数数据

        # 根据阈值确定标签
        for i, thresh in enumerate(thresholds):
            if value <= thresh:
                return (
                    self.quantile_labels[i]
                    if i < len(self.quantile_labels)
                    else "neutral"
                )

        # 超过所有阈值，返回最后一个标签
        return self.quantile_labels[-1] if self.quantile_labels else "neutral"

    def compute_score(self, value: float, quantiles: Dict[str, float]) -> float:
        """
        计算 Evidence 评分 (0-1 范围)

        标签映射:
        - suppress: 0.0
        - downweight: 0.25
        - neutral: 0.5
        - favor: 0.75
        - amplify: 1.0
        """
        label = self.compute_label(value, quantiles)
        score_map = {
            "suppress": 0.0,
            "downweight": 0.25,
            "neutral": 0.5,
            "favor": 0.75,
            "amplify": 1.0,
        }
        return score_map.get(label, 0.5)


@dataclass
class EvidenceConfig:
    """Evidence 配置 - 从 evidence.yaml 加载"""

    features: List[EvidenceFeature] = field(default_factory=list)
    label_semantics: Dict[str, str] = field(default_factory=dict)

    def compute_composite_score(
        self,
        feature_values: Dict[str, float],
        quantiles: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算 Evidence 综合评分

        Args:
            feature_values: {feature_name: value} 特征值字典
            quantiles: 分位数查找表

        Returns:
            (composite_score, {feature_id: score}) 综合分和各特征得分
        """
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for feat in self.features:
            if feat.feature not in feature_values:
                continue

            value = feature_values[feat.feature]
            score = feat.compute_score(value, quantiles)
            scores[feat.id] = score

            # 用 rank 作为权重 (rank 越低越重要)
            weight = 1.0 / max(1, feat.rank)
            weighted_sum += score * weight
            total_weight += weight

        composite = weighted_sum / total_weight if total_weight > 0 else 0.5
        return composite, scores

    @classmethod
    def from_yaml(cls, path: Path) -> "EvidenceConfig":
        """从 YAML 文件加载"""
        if not path.exists():
            return cls()

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        features = []
        for e in raw.get("evidence") or []:
            qm = e.get("quantile_mapping") or {}
            features.append(
                EvidenceFeature(
                    id=str(e.get("id", "")),
                    feature=str(e.get("feature", "")),
                    rank=int(e.get("rank", 99)),
                    split_count=int(e.get("split_count", 0)),
                    usage_hint=str(e.get("usage_hint", "")),
                    affects=list(e.get("affects") or []),
                    quantile_bins=list(qm.get("bins") or [0.2, 0.4, 0.6, 0.8]),
                    quantile_labels=list(
                        qm.get("labels")
                        or ["suppress", "downweight", "neutral", "favor", "amplify"]
                    ),
                    threshold_examples=list(e.get("threshold_examples") or []),
                    distribution_hint=str(e.get("distribution_hint", "")),
                    direction=_map_direction(e.get("direction", "positive")),
                )
            )

        schema = raw.get("schema") or {}
        return cls(
            features=features,
            label_semantics=dict(schema.get("label_semantics") or {}),
        )


# =============================================================================
# Execution Config
# =============================================================================


@dataclass
class ExecutionConfig:
    """Execution 配置 - 从 execution.yaml 加载"""

    allow_add_on: bool = False
    min_order_interval_minutes: int = 60
    stop_loss_r: float = 1.0
    take_profit_r: float = 2.5
    max_holding_bars: Optional[int] = None
    min_holding_bars: Optional[int] = None
    direction_source: str = "structure"
    direction_method: str = "trend_sign"
    direction_lookback_bars: int = 5
    direction_min_consistency: float = 0.6
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExecutionConfig":
        """从 YAML 文件加载"""
        if not path.exists():
            return cls()

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        ec = raw.get("execution_constraints") or {}
        fixed_rr = ec.get("fixed_rr") or {}
        dp = raw.get("direction_policy") or {}
        sd = dp.get("structure_direction") or {}

        return cls(
            allow_add_on=bool(ec.get("allow_add_on", False)),
            min_order_interval_minutes=int(ec.get("min_order_interval_minutes", 60)),
            stop_loss_r=float(fixed_rr.get("stop_loss_r", 1.0)),
            take_profit_r=float(fixed_rr.get("take_profit_r", 2.5)),
            max_holding_bars=fixed_rr.get("max_holding_bars"),
            min_holding_bars=fixed_rr.get("min_holding_bars"),
            direction_source=str(dp.get("direction_source", "structure")),
            direction_method=str(sd.get("method", "trend_sign")),
            direction_lookback_bars=int(sd.get("lookback_bars", 5)),
            direction_min_consistency=float(sd.get("min_consistency", 0.6)),
            raw=raw,
        )


# =============================================================================
# Strategy Archetype
# =============================================================================


@dataclass
class StrategyArchetype:
    """策略 Archetype - 组合 Gate / Evidence / Execution 三层配置"""

    name: str
    gate: GateConfig
    evidence: EvidenceConfig
    execution: ExecutionConfig

    # ==========================================================================
    # 向后兼容属性 (兼容旧的 ExecutionArchetype 接口)
    # ==========================================================================

    @property
    def gate_rules(self) -> Dict[str, Any]:
        """兼容旧接口：返回 when_then_rules 格式"""
        rules = []
        for r in self.gate.all_rules:
            rules.append(
                {
                    "id": r.id,
                    "phase": r.phase,
                    "priority": r.priority,
                    "reason": r.reason,
                    "when": r.when,
                    "then": r.then,
                }
            )
        return {
            "when_then_rules": rules,
            "default_action": "allow",
        }

    @property
    def direction_policy(self) -> Dict[str, Any]:
        """兼容旧接口：返回 direction_policy"""
        return self.execution.raw.get(
            "direction_policy",
            {
                "direction_source": self.execution.direction_source,
                "structure_direction": {
                    "method": self.execution.direction_method,
                    "lookback_bars": self.execution.direction_lookback_bars,
                    "min_consistency": self.execution.direction_min_consistency,
                },
            },
        )

    @property
    def execution_constraints(self) -> Dict[str, Any]:
        """兼容旧接口：返回 execution_constraints"""
        return self.execution.raw.get(
            "execution_constraints",
            {
                "allow_add_on": self.execution.allow_add_on,
                "min_order_interval_minutes": self.execution.min_order_interval_minutes,
                "fixed_rr": {
                    "stop_loss_r": self.execution.stop_loss_r,
                    "take_profit_r": self.execution.take_profit_r,
                    "max_holding_bars": self.execution.max_holding_bars,
                },
            },
        )

    @property
    def when_then_rules(self) -> List[Dict[str, Any]]:
        """兼容旧接口：返回 when_then_rules 列表"""
        return self.gate_rules.get("when_then_rules", [])

    @property
    def default_action(self) -> str:
        """兼容旧接口：返回默认动作"""
        return self.gate_rules.get("default_action", "allow")

    @property
    def evidence_rules(self) -> List[Dict[str, Any]]:
        """兼容旧接口：返回 evidence_rules 格式"""
        # 返回空列表，因为新架构使用 compute_evidence_score
        return []

    @property
    def regime(self) -> str:
        """兼容旧接口：返回 regime"""
        return "ANY"  # 新架构不再使用 regime 分流

    def apply_gate(
        self,
        features: Dict[str, Any],
        quantiles: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str], float]:
        """
        应用 Gate 规则

        Args:
            features: 特征值字典
            quantiles: 分位数查找表 (quantile_* 规则需要)

        Returns:
            (passed, deny_reasons, cumulative_weight)
            - passed: 是否通过 Gate
            - deny_reasons: 触发的 deny 规则 tag 列表
            - cumulative_weight: 始终返回 1.0（保持接口兼容）
        """
        deny_reasons = []

        for rule in self.gate.all_rules:
            matched = _evaluate_when_clause(rule.when, features, quantiles)

            if matched:
                action = rule.then.get("action", "deny")
                if action == "deny":
                    deny_reasons.append(rule.tag)
                    return False, deny_reasons, 0.0

        return True, deny_reasons, 1.0

    def compute_evidence_score(
        self,
        features: Dict[str, Any],
        quantiles: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算 Evidence 综合评分

        Args:
            features: 特征值字典
            quantiles: 分位数查找表

        Returns:
            (composite_score, {feature_id: score})
        """
        return self.evidence.compute_composite_score(features, quantiles or {})


# =============================================================================
# 条件评估
# =============================================================================


def _evaluate_when_clause(
    when: Dict[str, Any],
    features: Dict[str, Any],
    quantiles: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    评估 when 子句

    支持的格式：
    - {feature: {value_lt: 0.5}}
    - {feature: {quantile_gt: 0.7}}
    - {all_of: [...]}
    - {any_of: [...]}
    """
    if not when:
        return False

    # all_of
    if "all_of" in when:
        conditions = when["all_of"]
        min_matches = when.get("min_matches", len(conditions))
        matches = sum(
            1 for c in conditions if _evaluate_when_clause(c, features, quantiles)
        )
        return matches >= min_matches

    # any_of
    if "any_of" in when:
        conditions = when["any_of"]
        min_matches = when.get("min_matches", 1)
        matches = sum(
            1 for c in conditions if _evaluate_when_clause(c, features, quantiles)
        )
        return matches >= min_matches

    # 单个条件: {feature: {op: value}}
    for key, cond in when.items():
        if key in ("all_of", "any_of", "min_matches"):
            continue

        if not isinstance(cond, dict):
            continue

        value = features.get(key)
        if value is None:
            # on_missing 处理
            on_missing = cond.get("on_missing", "false")
            if on_missing == "true":
                return True
            elif on_missing == "error":
                raise ValueError(f"Feature {key} is missing")
            return False

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        # 直接阈值比较
        if "value_lt" in cond:
            if not (value < float(cond["value_lt"])):
                return False
        if "value_lte" in cond:
            if not (value <= float(cond["value_lte"])):
                return False
        if "value_gt" in cond:
            if not (value > float(cond["value_gt"])):
                return False
        if "value_gte" in cond:
            if not (value >= float(cond["value_gte"])):
                return False

        # 分位数比较
        has_quantile_cond = any(
            k in cond
            for k in ("quantile_lt", "quantile_lte", "quantile_gt", "quantile_gte")
        )
        if has_quantile_cond:
            if not quantiles:
                raise ValueError(
                    f"Gate rule requires quantiles for '{key}' but quantiles=None. "
                    f"Ensure set_quantiles_from_df() is called before apply_gate()."
                )
            feat_q = quantiles.get(key, {})
            if not feat_q:
                raise ValueError(
                    f"Gate rule requires quantile for '{key}' but it is missing from quantiles dict. "
                    f"Available keys: {list(quantiles.keys())}"
                )

            if "quantile_lt" in cond:
                q = float(cond["quantile_lt"])
                thresh = _get_quantile_threshold(feat_q, q)
                if thresh is None:
                    return False  # 无法获取阈值 → 不匹配
                if not (value < thresh):
                    return False

            if "quantile_lte" in cond:
                q = float(cond["quantile_lte"])
                thresh = _get_quantile_threshold(feat_q, q)
                if thresh is None:
                    return False
                if not (value <= thresh):
                    return False

            if "quantile_gt" in cond:
                q = float(cond["quantile_gt"])
                thresh = _get_quantile_threshold(feat_q, q)
                if thresh is None:
                    return False
                if not (value > thresh):
                    return False

            if "quantile_gte" in cond:
                q = float(cond["quantile_gte"])
                thresh = _get_quantile_threshold(feat_q, q)
                if thresh is None:
                    return False
                if not (value >= thresh):
                    return False

    return True


def _get_quantile_threshold(feat_q: Dict[str, Any], q: float) -> Optional[float]:
    """获取分位数阈值"""
    if not feat_q:
        return None

    # 尝试多种 key 格式
    q_keys = [
        f"{q:.2f}".rstrip("0").rstrip("."),
        str(q),
        f"q{int(q * 100)}",
    ]

    for k in q_keys:
        if k in feat_q:
            try:
                return float(feat_q[k])
            except (TypeError, ValueError):
                pass

    return None


# =============================================================================
# 加载函数
# =============================================================================


def load_strategy_archetype(
    strategy: str,
    strategies_root: str | Path = "config/strategies",
) -> StrategyArchetype:
    """
    加载单个策略的 Archetype 配置

    自动检测 ``archetypes/prefilter.yaml``，将其规则作为 guardrails
    注入 GateConfig，保持 prefilter.yaml 为 single source of truth。

    Args:
        strategy: 策略名 (如 "bpc")
        strategies_root: 策略配置根目录

    Returns:
        StrategyArchetype 实例
    """
    root = Path(strategies_root)
    arch_dir = root / strategy / "archetypes"

    if not arch_dir.exists():
        raise FileNotFoundError(f"Archetype directory not found: {arch_dir}")

    prefilter_path = arch_dir / "prefilter.yaml"

    return StrategyArchetype(
        name=strategy,
        gate=GateConfig.from_yaml(
            arch_dir / "gate.yaml",
            prefilter_path=prefilter_path if prefilter_path.exists() else None,
        ),
        evidence=EvidenceConfig.from_yaml(arch_dir / "evidence.yaml"),
        execution=ExecutionConfig.from_yaml(arch_dir / "execution.yaml"),
    )


def load_all_strategy_archetypes(
    strategies_root: str | Path = "config/strategies",
) -> Dict[str, StrategyArchetype]:
    """
    加载所有策略的 Archetype 配置

    Args:
        strategies_root: 策略配置根目录

    Returns:
        {strategy_name: StrategyArchetype}
    """
    root = Path(strategies_root)
    archetypes = {}

    for strategy_dir in root.iterdir():
        if not strategy_dir.is_dir():
            continue

        arch_dir = strategy_dir / "archetypes"
        if not arch_dir.exists():
            continue

        strategy_name = strategy_dir.name
        try:
            archetypes[strategy_name] = load_strategy_archetype(
                strategy_name,
                strategies_root,
            )
        except Exception as e:
            # 加载失败时跳过，打印警告
            import warnings

            warnings.warn(f"Failed to load archetype for {strategy_name}: {e}")

    return archetypes
