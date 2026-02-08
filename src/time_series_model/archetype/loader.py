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
    phase: str  # system_safety / hard_gate / soft_filter
    priority: int
    reason: str
    when: Dict[str, Any]
    then: Dict[str, Any]

    @property
    def is_hard(self) -> bool:
        return self.phase in ("system_safety", "hard_gate")

    @property
    def weight(self) -> float:
        """soft_filter 的降权权重，hard 返回 0"""
        if self.is_hard:
            return 0.0
        return float(self.then.get("weight", 1.0))


@dataclass
class GateConfig:
    """Gate 配置 - 从 gate.yaml 加载"""

    hard_gates: List[GateRule] = field(default_factory=list)
    soft_filters: List[GateRule] = field(default_factory=list)
    system_safety: List[GateRule] = field(default_factory=list)
    governance: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_rules(self) -> List[GateRule]:
        """按 phase -> priority 排序的所有规则"""
        phase_order = {"system_safety": 0, "hard_gate": 1, "soft_filter": 2}
        all_rules = self.system_safety + self.hard_gates + self.soft_filters
        return sorted(
            all_rules, key=lambda r: (phase_order.get(r.phase, 99), r.priority)
        )

    @property
    def hard_rules(self) -> List[GateRule]:
        """所有硬规则 (system_safety + hard_gate)"""
        return self.system_safety + self.hard_gates

    @property
    def soft_filter_floor(self) -> float:
        """soft_filter 累积权重下限"""
        floor_cfg = self.governance.get("soft_filter_floor", {})
        if floor_cfg.get("enabled", True):
            return float(floor_cfg.get("min_cumulative_weight", 0.25))
        return 0.0

    @classmethod
    def from_yaml(cls, path: Path) -> "GateConfig":
        """从 YAML 文件加载"""
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
                    )
                )
            return result

        return cls(
            hard_gates=_parse_rules(raw.get("hard_gates"), "hard_gate"),
            soft_filters=_parse_rules(raw.get("soft_filters"), "soft_filter"),
            system_safety=_parse_rules(raw.get("system_safety"), "system_safety"),
            governance=dict(
                raw.get("governance") or raw.get("schema", {}).get("governance") or {}
            ),
        )


# =============================================================================
# Evidence Config
# =============================================================================


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
            - cumulative_weight: soft_filter 累积权重 (1.0 = 无降权)
        """
        deny_reasons = []
        cumulative_weight = 1.0

        for rule in self.gate.all_rules:
            matched = _evaluate_when_clause(rule.when, features, quantiles)

            if matched:
                if rule.is_hard:
                    # 硬规则：直接 deny
                    action = rule.then.get("action", "deny")
                    if action == "deny":
                        deny_reasons.append(rule.tag)
                        return False, deny_reasons, 0.0
                else:
                    # 软规则：累积降权
                    action = rule.then.get("action", "downweight")
                    if action == "downweight":
                        weight = rule.weight
                        cumulative_weight *= weight

        # 应用 floor 保护
        floor = self.gate.soft_filter_floor
        if cumulative_weight < floor:
            cumulative_weight = floor

        return True, deny_reasons, cumulative_weight

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
        if quantiles:
            feat_q = quantiles.get(key, {})

            if "quantile_lt" in cond:
                q = float(cond["quantile_lt"])
                thresh = _get_quantile_threshold(feat_q, q)
                if thresh is not None and not (value < thresh):
                    return False

            if "quantile_lte" in cond:
                q = float(cond["quantile_lte"])
                thresh = _get_quantile_threshold(feat_q, q)
                if thresh is not None and not (value <= thresh):
                    return False

            if "quantile_gt" in cond:
                q = float(cond["quantile_gt"])
                thresh = _get_quantile_threshold(feat_q, q)
                if thresh is not None and not (value > thresh):
                    return False

            if "quantile_gte" in cond:
                q = float(cond["quantile_gte"])
                thresh = _get_quantile_threshold(feat_q, q)
                if thresh is not None and not (value >= thresh):
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

    return StrategyArchetype(
        name=strategy,
        gate=GateConfig.from_yaml(arch_dir / "gate.yaml"),
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
