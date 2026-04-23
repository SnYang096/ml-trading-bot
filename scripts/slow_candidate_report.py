#!/usr/bin/env python3
"""Slow pipeline candidate-discovery reports (T1-T4).

慢管线定位为 "候选发现工具" (wave3/02, wave3/04). 此脚本聚合四类报告供人审:

  manifest    T1 每月 Prefilter / Gate / Entry Filter 选出的特征与规则清单.
  drift       T2 相对"锁定基线"(turbo run 的 strategies_calibrated/) 的 per-month
              feature / rule 增减 diff.
  digest      T3 将 manifest + drift + compare_monthly_pnl 的月度 R delta 组装
              成一张"值不值"总览 — 一眼看特征变化换到多少 R.
  consensus   T4 多方法共识矩阵 (methods × features) — 需要 pipeline 先 dump
              每个方法的候选 (见 auto_research_pipeline 的 _candidates/method=* 逻辑).

Slow run 目录布局 (典型):
  results/<strategy>/slow-rolling-sim/_rolling_sim/<timestamp>/
    slow_snapshot_<YYYY-MM>/
      strategies/<strategy>/
        archetypes/{prefilter,gate,entry_filters}.yaml
        gate_draft.yaml
        features_prefilter.yaml
        features_entry_filter.yaml
        _candidates/method=<m>/         # T4 only, 需要 pipeline dump
          prefilter.yaml
          entry_filters.yaml

Turbo baseline 目录布局:
  results/<strategy>/turbo-rolling-sim/_rolling_sim/<timestamp>/
    fast_month_<YYYY-MM>/strategies_calibrated/<strategy>/
      (同 slow_snapshot 下的 strategies/<strategy>/ 结构)

示例:
  # 1) 仅看慢管线每月挑了什么
  python scripts/slow_candidate_report.py manifest \
      --run-dir results/bpc/slow-rolling-sim/_rolling_sim/20260421_174335 \
      --strategy bpc \
      --output results/bpc/slow_candidate_reports/manifest.md

  # 2) 对比锁定基线 (turbo run) 每月特征/规则的增减
  python scripts/slow_candidate_report.py drift \
      --slow-run-dir results/bpc/slow-rolling-sim/_rolling_sim/20260421_174335 \
      --baseline-run-dir results/bpc/turbo-rolling-sim/_rolling_sim/20260409_171133 \
      --strategy bpc \
      --output results/bpc/slow_candidate_reports/drift.md

  # 3) 组合 manifest + drift + monthly R delta
  python scripts/slow_candidate_report.py digest \
      --slow-run-dir results/bpc/slow-rolling-sim/_rolling_sim/20260421_174335 \
      --baseline-run-dir results/bpc/turbo-rolling-sim/_rolling_sim/20260409_171133 \
      --strategy bpc \
      --output results/bpc/slow_candidate_reports/digest.md

  # 4) Multi-method consensus matrix (需要 slow run 已 dump _candidates/)
  python scripts/slow_candidate_report.py consensus \
      --run-dir results/bpc/slow-rolling-sim/_rolling_sim/<ts> \
      --strategy bpc \
      --output results/bpc/slow_candidate_reports/consensus.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RuleSig:
    """Canonical signature of a single prefilter rule for diffing."""

    feature: str
    operator: str  # ">=", "<=", ">", "<", "range"
    value: Any  # float for single-threshold, (lo, hi) tuple for range
    locked: bool = False

    @property
    def key(self) -> str:
        """Key for diff identity. Same (feature, operator) → same rule."""
        return f"{self.feature}|{self.operator}"

    def display(self) -> str:
        if self.operator == "range":
            lo, hi = self.value
            return f"{self.feature} ∈ [{lo:.4g}, {hi:.4g}]"
        return f"{self.feature} {self.operator} {self.value!s}"


@dataclass
class GateRuleSig:
    """Canonical signature of a single gate hard_gate / guardrail / system_safety rule."""

    rule_id: str
    phase: str  # hard_gate / system_safety / guardrail
    features: Tuple[str, ...]  # all features referenced in when.all_of
    conditions: Tuple[str, ...]  # rendered "feat op val" strings
    reason: str = ""

    @property
    def key(self) -> str:
        return self.rule_id

    def display(self) -> str:
        return f"[{self.phase}] {self.rule_id}: " + " & ".join(self.conditions)


@dataclass
class EntryFilterSig:
    filter_id: str
    enabled: bool
    locked: bool
    conditions: Tuple[str, ...]

    @property
    def key(self) -> str:
        return self.filter_id

    def display(self) -> str:
        flags = []
        if self.locked:
            flags.append("locked")
        if not self.enabled:
            flags.append("disabled")
        flag_str = f" ({','.join(flags)})" if flags else ""
        return f"{self.filter_id}{flag_str}: " + " & ".join(self.conditions)


@dataclass
class SnapshotDigest:
    """All the feature-selection artifacts a single monthly snapshot carries."""

    month: str
    strategy: str
    source_dir: Path
    prefilter_rules: List[RuleSig] = field(default_factory=list)
    gate_rules: List[GateRuleSig] = field(default_factory=list)
    entry_filters: List[EntryFilterSig] = field(default_factory=list)
    pf_requested: List[str] = field(default_factory=list)
    ef_requested: List[str] = field(default_factory=list)

    # Derived counts for the manifest summary table
    @property
    def n_pf_rules(self) -> int:
        return len(self.prefilter_rules)

    @property
    def n_gate_hard(self) -> int:
        return sum(1 for g in self.gate_rules if g.phase == "hard_gate")

    @property
    def n_gate_safety(self) -> int:
        return sum(1 for g in self.gate_rules if g.phase == "system_safety")

    @property
    def n_gate_guardrail(self) -> int:
        return sum(1 for g in self.gate_rules if g.phase == "guardrail")

    @property
    def n_ef(self) -> int:
        return len(self.entry_filters)


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_yaml_load(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  failed to parse {path}: {exc}", file=sys.stderr)
        return None


def _parse_prefilter_yaml(path: Path) -> List[RuleSig]:
    """Parse archetypes/prefilter.yaml → list of RuleSig.

    Supports single-threshold rules (operator in {>=, <=, >, <}) and also range
    rules expressed as two rules on the same feature (lo + hi). Range detection
    is done at callsite via grouping; here we return each atomic rule.
    """
    payload = _safe_yaml_load(path) or {}
    raw_rules = payload.get("rules") or []
    out: List[RuleSig] = []
    for r in raw_rules:
        if not isinstance(r, dict):
            continue
        feat = str(r.get("feature", "")).strip()
        op = str(r.get("operator", "")).strip()
        val = r.get("value")
        locked = bool(r.get("locked", False))
        if not feat or not op:
            continue
        try:
            val_norm: Any = float(val)
        except (TypeError, ValueError):
            val_norm = val
        out.append(RuleSig(feature=feat, operator=op, value=val_norm, locked=locked))
    return out


def _render_condition(feat: str, op_dict: Dict[str, Any]) -> Optional[str]:
    """Render one gate when-condition into 'feat op val' text."""
    for op_key in ("value_gt", "value_ge", "value_lt", "value_le", "value_eq"):
        if op_key in op_dict:
            op_map = {
                "value_gt": ">",
                "value_ge": ">=",
                "value_lt": "<",
                "value_le": "<=",
                "value_eq": "==",
            }
            try:
                v = float(op_dict[op_key])
                return f"{feat} {op_map[op_key]} {v:.4g}"
            except (TypeError, ValueError):
                return f"{feat} {op_map[op_key]} {op_dict[op_key]}"
    return None


def _parse_gate_yaml(path: Path) -> List[GateRuleSig]:
    """Parse gate_draft.yaml (archetype format) → list of GateRuleSig.

    Covers hard_gates / system_safety / guardrails sections. Each entry's
    when.all_of is expanded into a list of condition strings.
    """
    payload = _safe_yaml_load(path) or {}
    out: List[GateRuleSig] = []

    def _ingest(section: str, entries: Any, default_phase: str) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rule_id = str(entry.get("id") or entry.get("tag") or "").strip()
            if not rule_id:
                continue
            phase = str(entry.get("phase") or default_phase)
            reason = str(entry.get("reason") or entry.get("comment") or "")
            when = entry.get("when") or {}
            feats: List[str] = []
            conditions: List[str] = []
            if isinstance(when, dict):
                all_of = when.get("all_of")
                if isinstance(all_of, list):
                    # List-of-dicts form: when.all_of = [{feat: {value_gt: ...}}, ...]
                    for cond in all_of:
                        if not isinstance(cond, dict) or len(cond) != 1:
                            continue
                        feat, spec = next(iter(cond.items()))
                        if isinstance(spec, dict):
                            cond_str = _render_condition(str(feat), spec)
                            if cond_str:
                                conditions.append(cond_str)
                                feats.append(str(feat))
                else:
                    # Direct form: when = {feat: {value_lt: ...}, feat2: {...}}
                    for feat, spec in when.items():
                        if not isinstance(spec, dict):
                            continue
                        cond_str = _render_condition(str(feat), spec)
                        if cond_str:
                            conditions.append(cond_str)
                            feats.append(str(feat))
            out.append(
                GateRuleSig(
                    rule_id=rule_id,
                    phase=phase,
                    features=tuple(sorted(set(feats))),
                    conditions=tuple(conditions),
                    reason=reason,
                )
            )

    _ingest("hard_gates", payload.get("hard_gates"), "hard_gate")
    _ingest("system_safety", payload.get("system_safety"), "system_safety")
    _ingest("guardrails", payload.get("guardrails"), "guardrail")
    return out


def _parse_entry_filters_yaml(path: Path) -> List[EntryFilterSig]:
    """Parse archetypes/entry_filters.yaml → list of EntryFilterSig."""
    payload = _safe_yaml_load(path) or {}
    raw = payload.get("filters") or []
    out: List[EntryFilterSig] = []
    for f in raw:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id", "")).strip()
        if not fid:
            continue
        enabled = bool(f.get("enabled", True))
        locked = bool(f.get("locked", False))
        conds = f.get("conditions") or []
        cond_strs: List[str] = []
        if isinstance(conds, list):
            for c in conds:
                if not isinstance(c, dict):
                    continue
                feat = str(c.get("feature", "")).strip()
                op = str(c.get("operator", "")).strip()
                val = c.get("value")
                if not feat or not op:
                    continue
                try:
                    val_fmt = f"{float(val):.4g}"
                except (TypeError, ValueError):
                    val_fmt = str(val)
                cond_strs.append(f"{feat} {op} {val_fmt}")
        out.append(
            EntryFilterSig(
                filter_id=fid,
                enabled=enabled,
                locked=locked,
                conditions=tuple(cond_strs),
            )
        )
    return out


def _parse_requested_features(path: Path) -> List[str]:
    payload = _safe_yaml_load(path) or {}
    fp = payload.get("feature_pipeline") or {}
    req = fp.get("requested_features") or []
    return [str(x) for x in req if str(x).strip()]


def _parse_strategy_dir(strat_dir: Path, month: str, strategy: str) -> SnapshotDigest:
    """Parse a strategies/<strategy>/ dir into a SnapshotDigest."""
    digest = SnapshotDigest(month=month, strategy=strategy, source_dir=strat_dir)
    digest.prefilter_rules = _parse_prefilter_yaml(
        strat_dir / "archetypes" / "prefilter.yaml"
    )
    digest.gate_rules = _parse_gate_yaml(strat_dir / "gate_draft.yaml")
    digest.entry_filters = _parse_entry_filters_yaml(
        strat_dir / "archetypes" / "entry_filters.yaml"
    )
    digest.pf_requested = _parse_requested_features(
        strat_dir / "features_prefilter.yaml"
    )
    digest.ef_requested = _parse_requested_features(
        strat_dir / "features_entry_filter.yaml"
    )
    return digest


def parse_slow_run(run_dir: Path, strategy: str) -> Dict[str, SnapshotDigest]:
    """Parse all slow_snapshot_<YYYY-MM>/strategies/<strategy>/ under a slow run."""
    out: Dict[str, SnapshotDigest] = {}
    for snap_dir in sorted(run_dir.glob("slow_snapshot_*")):
        if not snap_dir.is_dir():
            continue
        month = snap_dir.name.replace("slow_snapshot_", "", 1)
        strat_dir = snap_dir / "strategies" / strategy
        if not strat_dir.is_dir():
            continue
        out[month] = _parse_strategy_dir(strat_dir, month, strategy)
    return out


def parse_turbo_baseline(run_dir: Path, strategy: str) -> Dict[str, SnapshotDigest]:
    """Parse all fast_month_<YYYY-MM>/strategies_calibrated/<strategy>/ under a turbo run.

    Turbo locks features, so each month's snapshot typically carries the same
    prefilter/gate/entry_filter but calibrated thresholds. We still parse per-month
    so that drift vs slow is month-aligned.
    """
    out: Dict[str, SnapshotDigest] = {}
    for month_dir in sorted(run_dir.glob("fast_month_*")):
        if not month_dir.is_dir():
            continue
        month = month_dir.name.replace("fast_month_", "", 1)
        strat_dir = month_dir / "strategies_calibrated" / strategy
        if not strat_dir.is_dir():
            continue
        out[month] = _parse_strategy_dir(strat_dir, month, strategy)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diff helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ListDiff:
    """Generic added/dropped/changed diff (by .key) between two digest sections."""

    added: List[Any] = field(default_factory=list)
    dropped: List[Any] = field(default_factory=list)
    changed: List[Tuple[Any, Any]] = field(default_factory=list)  # (base, new)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.dropped or self.changed)


def _diff_by_key(base_list: Sequence[Any], new_list: Sequence[Any], eq_fn) -> ListDiff:
    base_map = {x.key: x for x in base_list}
    new_map = {x.key: x for x in new_list}
    diff = ListDiff()
    for k, nv in new_map.items():
        bv = base_map.get(k)
        if bv is None:
            diff.added.append(nv)
        elif not eq_fn(bv, nv):
            diff.changed.append((bv, nv))
    for k, bv in base_map.items():
        if k not in new_map:
            diff.dropped.append(bv)
    return diff


def _pf_eq(a: RuleSig, b: RuleSig) -> bool:
    return a.operator == b.operator and a.value == b.value and a.locked == b.locked


def _gate_eq(a: GateRuleSig, b: GateRuleSig) -> bool:
    return a.phase == b.phase and a.conditions == b.conditions


def _ef_eq(a: EntryFilterSig, b: EntryFilterSig) -> bool:
    return (
        a.enabled == b.enabled and a.locked == b.locked and a.conditions == b.conditions
    )


def _list_diff(
    base_list: Sequence[str], new_list: Sequence[str]
) -> Tuple[List[str], List[str]]:
    base_set: Set[str] = set(base_list)
    new_set: Set[str] = set(new_list)
    return sorted(new_set - base_set), sorted(base_set - new_set)


@dataclass
class MonthDiff:
    month: str
    pf_rules_diff: ListDiff
    gate_rules_diff: ListDiff
    ef_filters_diff: ListDiff
    pf_req_added: List[str]
    pf_req_dropped: List[str]
    ef_req_added: List[str]
    ef_req_dropped: List[str]

    @property
    def any_change(self) -> bool:
        return (
            not self.pf_rules_diff.is_empty
            or not self.gate_rules_diff.is_empty
            or not self.ef_filters_diff.is_empty
            or bool(self.pf_req_added or self.pf_req_dropped)
            or bool(self.ef_req_added or self.ef_req_dropped)
        )


def diff_digest(base: SnapshotDigest, new: SnapshotDigest) -> MonthDiff:
    pf_diff = _diff_by_key(base.prefilter_rules, new.prefilter_rules, _pf_eq)
    gate_diff = _diff_by_key(base.gate_rules, new.gate_rules, _gate_eq)
    ef_diff = _diff_by_key(base.entry_filters, new.entry_filters, _ef_eq)
    pf_add, pf_drop = _list_diff(base.pf_requested, new.pf_requested)
    ef_add, ef_drop = _list_diff(base.ef_requested, new.ef_requested)
    return MonthDiff(
        month=new.month,
        pf_rules_diff=pf_diff,
        gate_rules_diff=gate_diff,
        ef_filters_diff=ef_diff,
        pf_req_added=pf_add,
        pf_req_dropped=pf_drop,
        ef_req_added=ef_add,
        ef_req_dropped=ef_drop,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────────────────────


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_manifest(
    snapshots: Dict[str, SnapshotDigest], strategy: str, slow_run_dir: Path
) -> str:
    months = sorted(snapshots.keys())
    lines: List[str] = []
    lines.append(f"# Slow snapshot feature manifest — {strategy}")
    lines.append("")
    lines.append(f"- **Run**: `{slow_run_dir}`")
    lines.append(f"- **Strategy**: `{strategy}`")
    lines.append(f"- **Snapshots**: {len(months)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Month | PF rules | Gate hard | Gate safety | Gate guardrail | EF filters | PF req feats | EF req feats |"
    )
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for m in months:
        d = snapshots[m]
        lines.append(
            f"| {m} | {d.n_pf_rules} | {d.n_gate_hard} | {d.n_gate_safety} | "
            f"{d.n_gate_guardrail} | {d.n_ef} | {len(d.pf_requested)} | "
            f"{len(d.ef_requested)} |"
        )
    lines.append("")
    lines.append("## Per-month details")
    lines.append("")
    for m in months:
        d = snapshots[m]
        lines.append(f"### {m}")
        lines.append("")
        lines.append(
            f"Source: `{d.source_dir.relative_to(slow_run_dir.parent.parent.parent) if slow_run_dir.parent.parent.parent in d.source_dir.parents else d.source_dir}`"
        )
        lines.append("")
        lines.append("**Prefilter rules**")
        if d.prefilter_rules:
            for r in d.prefilter_rules:
                mark = " 🔒" if r.locked else ""
                lines.append(f"- {_md_escape(r.display())}{mark}")
        else:
            lines.append("- *(empty)*")
        lines.append("")
        lines.append("**Gate rules**")
        if d.gate_rules:
            for g in d.gate_rules:
                lines.append(f"- {_md_escape(g.display())}")
        else:
            lines.append("- *(empty)*")
        lines.append("")
        lines.append("**Entry filters**")
        if d.entry_filters:
            for f in d.entry_filters:
                lines.append(f"- {_md_escape(f.display())}")
        else:
            lines.append("- *(empty)*")
        lines.append("")
        lines.append(
            f"**Prefilter requested_features ({len(d.pf_requested)})**: "
            + (
                ", ".join(f"`{x}`" for x in d.pf_requested)
                if d.pf_requested
                else "*(empty)*"
            )
        )
        lines.append("")
        lines.append(
            f"**Entry filter requested_features ({len(d.ef_requested)})**: "
            + (
                ", ".join(f"`{x}`" for x in d.ef_requested)
                if d.ef_requested
                else "*(empty)*"
            )
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_diff_section(title: str, diff: ListDiff) -> List[str]:
    lines = [f"**{title}**"]
    if diff.is_empty:
        lines.append("- *(no change)*")
        return lines
    for item in diff.added:
        lines.append(f"- ➕ added: {_md_escape(item.display())}")
    for item in diff.dropped:
        lines.append(f"- ➖ dropped: {_md_escape(item.display())}")
    for base, new in diff.changed:
        lines.append(
            f"- ✏️  changed: `{_md_escape(base.display())}` → `{_md_escape(new.display())}`"
        )
    return lines


def _render_reqfeat_section(
    title: str, added: List[str], dropped: List[str]
) -> List[str]:
    lines = [f"**{title}**"]
    if not added and not dropped:
        lines.append("- *(no change)*")
        return lines
    if added:
        lines.append("- ➕ added: " + ", ".join(f"`{x}`" for x in added))
    if dropped:
        lines.append("- ➖ dropped: " + ", ".join(f"`{x}`" for x in dropped))
    return lines


def render_drift(
    base: Dict[str, SnapshotDigest],
    new: Dict[str, SnapshotDigest],
    strategy: str,
    slow_run_dir: Path,
    baseline_run_dir: Path,
) -> str:
    months = sorted(set(new) | set(base))
    lines: List[str] = []
    lines.append(f"# Slow vs locked-baseline feature drift — {strategy}")
    lines.append("")
    lines.append(f"- **Slow run**: `{slow_run_dir}`")
    lines.append(f"- **Baseline (turbo)**: `{baseline_run_dir}`")
    lines.append(f"- **Strategy**: `{strategy}`")
    lines.append("")
    lines.append("## Summary (#changes per month)")
    lines.append("")
    lines.append(
        "| Month | PF rules Δ | Gate rules Δ | EF filters Δ | PF req Δ | EF req Δ |"
    )
    lines.append("|:---|---:|---:|---:|---:|---:|")
    month_diffs: Dict[str, MonthDiff] = {}
    for m in months:
        if m not in new or m not in base:
            continue
        md = diff_digest(base[m], new[m])
        month_diffs[m] = md
        pf_n = (
            len(md.pf_rules_diff.added)
            + len(md.pf_rules_diff.dropped)
            + len(md.pf_rules_diff.changed)
        )
        gate_n = (
            len(md.gate_rules_diff.added)
            + len(md.gate_rules_diff.dropped)
            + len(md.gate_rules_diff.changed)
        )
        ef_n = (
            len(md.ef_filters_diff.added)
            + len(md.ef_filters_diff.dropped)
            + len(md.ef_filters_diff.changed)
        )
        pfreq_n = len(md.pf_req_added) + len(md.pf_req_dropped)
        efreq_n = len(md.ef_req_added) + len(md.ef_req_dropped)
        lines.append(f"| {m} | {pf_n} | {gate_n} | {ef_n} | {pfreq_n} | {efreq_n} |")

    missing_in_base = sorted(set(new) - set(base))
    missing_in_new = sorted(set(base) - set(new))
    if missing_in_base or missing_in_new:
        lines.append("")
        lines.append("### Month coverage gaps")
        if missing_in_base:
            lines.append(f"- slow has, baseline missing: {', '.join(missing_in_base)}")
        if missing_in_new:
            lines.append(f"- baseline has, slow missing: {', '.join(missing_in_new)}")

    lines.append("")
    lines.append("## Per-month details")
    lines.append("")
    for m in sorted(month_diffs.keys()):
        md = month_diffs[m]
        lines.append(f"### {m}")
        lines.append("")
        if not md.any_change:
            lines.append("*(no change vs baseline)*")
            lines.append("")
            continue
        lines.extend(_render_diff_section("Prefilter rules", md.pf_rules_diff))
        lines.append("")
        lines.extend(_render_diff_section("Gate rules", md.gate_rules_diff))
        lines.append("")
        lines.extend(_render_diff_section("Entry filters", md.ef_filters_diff))
        lines.append("")
        lines.extend(
            _render_reqfeat_section(
                "Prefilter requested_features", md.pf_req_added, md.pf_req_dropped
            )
        )
        lines.append("")
        lines.extend(
            _render_reqfeat_section(
                "Entry filter requested_features", md.ef_req_added, md.ef_req_dropped
            )
        )
        lines.append("")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# T3: digest (manifest + drift + monthly PnL delta)
# ─────────────────────────────────────────────────────────────────────────────


def _load_monthly_r(
    run_dir: Path, strategy: str, attribution: str = "linear_days"
) -> Dict[str, Tuple[int, float]]:
    """Return {month: (n_trades, total_r)} for a rolling_sim run.

    Reuses compare_monthly_pnl helpers (same dedup + attribution logic) but
    accepts an arbitrary run_dir path (works for both slow-rolling-sim and
    turbo-rolling-sim subtrees).

    run_dir layout: results/<strat>/<pipeline-dir>/_rolling_sim/<ts>/
    """
    import csv  # noqa: WPS433
    import glob  # noqa: WPS433
    import os  # noqa: WPS433

    from scripts.compare_monthly_pnl import (  # noqa: WPS433
        _parse_ts,
        aggregate_monthly,
    )

    trades: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str, str]] = set()
    csv_paths: List[Tuple[str, str]] = []
    for p in sorted(
        glob.glob(
            str(run_dir / "fast_month_*" / strategy / f"event_trades_{strategy}.csv")
        ),
        key=os.path.getmtime,
    ):
        tag = p.split("fast_month_")[1].split("/")[0]
        csv_paths.append((tag, p))
    by_tag: Dict[str, str] = {}
    for tag, p in csv_paths:
        by_tag[tag] = p  # later-modified wins on duplicate month
    for tag, p in by_tag.items():
        try:
            with open(p, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    key = (
                        row.get("symbol", ""),
                        row.get("side", ""),
                        row.get("entry_time", ""),
                        row.get("exit_time", ""),
                        row.get("is_add_position", ""),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        row["_pnl_r"] = float(row.get("pnl_r", "") or 0.0)
                    except (TypeError, ValueError):
                        continue
                    row["_entry_dt"] = _parse_ts(row.get("entry_time", ""))
                    row["_exit_dt"] = _parse_ts(row.get("exit_time", ""))
                    row["_src_target"] = tag
                    row["_src_path"] = p
                    trades.append(row)
        except FileNotFoundError:
            continue
    by_r, by_n = aggregate_monthly(trades, attribution=attribution)
    months = sorted(set(by_r) | set(by_n))
    return {m: (by_n.get(m, 0), float(by_r.get(m, 0.0))) for m in months}


def render_digest(
    snapshots: Dict[str, SnapshotDigest],
    base: Dict[str, SnapshotDigest],
    slow_run_dir: Path,
    baseline_run_dir: Path,
    strategy: str,
    attribution: str,
) -> str:
    try:
        slow_r = _load_monthly_r(slow_run_dir, strategy, attribution)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  slow monthly R load failed: {exc}", file=sys.stderr)
        slow_r = {}
    try:
        base_r = _load_monthly_r(baseline_run_dir, strategy, attribution)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  baseline monthly R load failed: {exc}", file=sys.stderr)
        base_r = {}

    months = sorted(set(snapshots) | set(base))
    lines: List[str] = []
    lines.append(f"# Slow candidate digest — {strategy}")
    lines.append("")
    lines.append(f"- **Slow run**: `{slow_run_dir}`")
    lines.append(f"- **Baseline (turbo)**: `{baseline_run_dir}`")
    lines.append(f"- **Attribution**: `{attribution}`")
    lines.append("")
    lines.append("## Feature changes vs monthly R delta")
    lines.append("")
    lines.append(
        "| Month | Slow n/R | Base n/R | ΔR | PF rules Δ | Gate Δ | EF Δ | PF req Δ | EF req Δ |"
    )
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")

    verdict_rows: List[Tuple[str, float, int]] = []
    for m in months:
        sl = slow_r.get(m)
        bs = base_r.get(m)
        sl_str = f"{sl[0]}/{sl[1]:+.1f}" if sl else "—"
        bs_str = f"{bs[0]}/{bs[1]:+.1f}" if bs else "—"
        if sl and bs:
            delta = sl[1] - bs[1]
            delta_str = f"{delta:+.1f}"
        else:
            delta = float("nan")
            delta_str = "—"
        if m in snapshots and m in base:
            md = diff_digest(base[m], snapshots[m])
            pf_n = (
                len(md.pf_rules_diff.added)
                + len(md.pf_rules_diff.dropped)
                + len(md.pf_rules_diff.changed)
            )
            gate_n = (
                len(md.gate_rules_diff.added)
                + len(md.gate_rules_diff.dropped)
                + len(md.gate_rules_diff.changed)
            )
            ef_n = (
                len(md.ef_filters_diff.added)
                + len(md.ef_filters_diff.dropped)
                + len(md.ef_filters_diff.changed)
            )
            pfreq_n = len(md.pf_req_added) + len(md.pf_req_dropped)
            efreq_n = len(md.ef_req_added) + len(md.ef_req_dropped)
        else:
            pf_n = gate_n = ef_n = pfreq_n = efreq_n = 0

        lines.append(
            f"| {m} | {sl_str} | {bs_str} | {delta_str} | {pf_n} | {gate_n} | "
            f"{ef_n} | {pfreq_n} | {efreq_n} |"
        )
        if sl and bs:
            verdict_rows.append((m, delta, pf_n + gate_n + ef_n + pfreq_n + efreq_n))

    lines.append("")
    # Simple verdict: months where slow changed something → aggregate R delta
    if verdict_rows:
        total_delta = sum(d for _, d, _ in verdict_rows)
        changed_months = [(m, d) for m, d, n in verdict_rows if n > 0]
        unchanged_months = [(m, d) for m, d, n in verdict_rows if n == 0]
        lines.append("## Verdict")
        lines.append("")
        lines.append(f"- Total ΔR across covered months: **{total_delta:+.1f}R**")
        if changed_months:
            changed_delta = sum(d for _, d in changed_months)
            lines.append(
                f"- Months where slow touched features/rules: {len(changed_months)}, "
                f"cumulative ΔR = **{changed_delta:+.1f}R**"
            )
        if unchanged_months:
            unchanged_delta = sum(d for _, d in unchanged_months)
            lines.append(
                f"- Months where slow matched baseline: {len(unchanged_months)}, "
                f"cumulative ΔR = {unchanged_delta:+.1f}R *(pure threshold / calibration drift)*"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# T4: multi-method consensus matrix
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MethodCandidate:
    method: str
    prefilter_rules: List[RuleSig]
    entry_filters: List[EntryFilterSig]


def parse_method_candidates(
    snapshot_dir: Path, strategy: str
) -> Dict[str, MethodCandidate]:
    """Parse `_candidates/method=<name>/` subdirs (T4 dump format).

    Missing `_candidates/` dir simply returns {}; caller should warn.
    """
    out: Dict[str, MethodCandidate] = {}
    cand_root = snapshot_dir / "strategies" / strategy / "_candidates"
    if not cand_root.is_dir():
        return out
    for method_dir in sorted(cand_root.iterdir()):
        if not method_dir.is_dir() or not method_dir.name.startswith("method="):
            continue
        method = method_dir.name.replace("method=", "", 1)
        pf = _parse_prefilter_yaml(method_dir / "prefilter.yaml")
        ef = _parse_entry_filters_yaml(method_dir / "entry_filters.yaml")
        out[method] = MethodCandidate(
            method=method, prefilter_rules=pf, entry_filters=ef
        )
    return out


def render_consensus(run_dir: Path, strategy: str) -> str:
    lines: List[str] = []
    lines.append(f"# Multi-method consensus matrix — {strategy}")
    lines.append("")
    lines.append(f"- **Run**: `{run_dir}`")
    lines.append("")

    snap_dirs = sorted(run_dir.glob("slow_snapshot_*"))
    any_data = False
    for snap_dir in snap_dirs:
        if not snap_dir.is_dir():
            continue
        month = snap_dir.name.replace("slow_snapshot_", "", 1)
        candidates = parse_method_candidates(snap_dir, strategy)
        lines.append(f"## {month}")
        lines.append("")
        if not candidates:
            lines.append(
                "*(no `_candidates/method=*/` dump — pipeline must write "
                "per-method candidates for this snapshot to render matrix)*"
            )
            lines.append("")
            continue
        any_data = True
        methods = sorted(candidates.keys())

        # Prefilter matrix
        lines.append("### Prefilter candidates")
        lines.append("")
        lines.extend(
            _render_matrix(
                methods,
                {m: {r.key: r for r in candidates[m].prefilter_rules} for m in methods},
                render_value=lambda r: r.display(),
            )
        )
        lines.append("")

        # Entry filter matrix
        lines.append("### Entry filter candidates")
        lines.append("")
        lines.extend(
            _render_matrix(
                methods,
                {m: {f.key: f for f in candidates[m].entry_filters} for m in methods},
                render_value=lambda f: f.display(),
            )
        )
        lines.append("")

    if not any_data:
        lines.append(
            "> ⚠️ No snapshot has `_candidates/method=*/` dump. To enable the "
            "consensus matrix, re-run the slow pipeline after the `auto_research_pipeline` "
            "per-method dump path is active (see wave3/04)."
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_matrix(
    methods: List[str],
    per_method: Dict[str, Dict[str, Any]],
    render_value,
) -> List[str]:
    """Render a methods × rule-key matrix.

    Collect all rule keys seen across methods. One row per key, columns = methods.
    Each cell: ✓ if method emitted this rule (+ short display on the first non-empty cell),
              blank otherwise. A 'hits' column counts how many methods agreed.
    """
    all_keys: List[str] = sorted({k for d in per_method.values() for k in d})
    if not all_keys:
        return ["*(no candidates across methods)*"]
    header = "| Rule | " + " | ".join(methods) + " | hits |"
    sep = "|:---|" + "|".join([":---:"] * len(methods)) + "|---:|"
    out = [header, sep]
    for key in all_keys:
        # Pick a display string from the first method that has the rule
        display = ""
        for m in methods:
            if key in per_method[m]:
                display = per_method[m][key].display()
                break
        cells = ["✓" if key in per_method[m] else "" for m in methods]
        hits = sum(1 for c in cells if c == "✓")
        out.append(f"| `{_md_escape(display)}` | " + " | ".join(cells) + f" | {hits} |")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _write_or_print(content: str, output: Optional[Path]) -> None:
    if output is None:
        sys.stdout.write(content)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"✅ wrote {output} ({len(content.splitlines())} lines)")


def _cmd_manifest(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"❌ run dir not found: {run_dir}", file=sys.stderr)
        return 2
    snaps = parse_slow_run(run_dir, args.strategy)
    if not snaps:
        print(f"⚠️  no slow_snapshot_* under {run_dir}", file=sys.stderr)
    content = render_manifest(snaps, args.strategy, run_dir)
    _write_or_print(content, Path(args.output).resolve() if args.output else None)
    return 0


def _cmd_drift(args: argparse.Namespace) -> int:
    slow_dir = Path(args.slow_run_dir).resolve()
    base_dir = Path(args.baseline_run_dir).resolve()
    if not slow_dir.is_dir():
        print(f"❌ slow run dir not found: {slow_dir}", file=sys.stderr)
        return 2
    if not base_dir.is_dir():
        print(f"❌ baseline run dir not found: {base_dir}", file=sys.stderr)
        return 2
    slow = parse_slow_run(slow_dir, args.strategy)
    base = parse_turbo_baseline(base_dir, args.strategy)
    content = render_drift(base, slow, args.strategy, slow_dir, base_dir)
    _write_or_print(content, Path(args.output).resolve() if args.output else None)
    return 0


def _cmd_digest(args: argparse.Namespace) -> int:
    slow_dir = Path(args.slow_run_dir).resolve()
    base_dir = Path(args.baseline_run_dir).resolve()
    if not slow_dir.is_dir():
        print(f"❌ slow run dir not found: {slow_dir}", file=sys.stderr)
        return 2
    if not base_dir.is_dir():
        print(f"❌ baseline run dir not found: {base_dir}", file=sys.stderr)
        return 2
    slow = parse_slow_run(slow_dir, args.strategy)
    base = parse_turbo_baseline(base_dir, args.strategy)
    content = render_digest(
        slow, base, slow_dir, base_dir, args.strategy, args.attribution
    )
    _write_or_print(content, Path(args.output).resolve() if args.output else None)
    return 0


def _cmd_consensus(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"❌ run dir not found: {run_dir}", file=sys.stderr)
        return 2
    content = render_consensus(run_dir, args.strategy)
    _write_or_print(content, Path(args.output).resolve() if args.output else None)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Slow pipeline candidate-discovery reports (T1-T4)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("manifest", help="T1 Per-month feature/rule manifest")
    p1.add_argument("--run-dir", required=True, help="slow rolling_sim run dir")
    p1.add_argument("--strategy", required=True)
    p1.add_argument("--output", help="markdown output (default stdout)")
    p1.set_defaults(func=_cmd_manifest)

    p2 = sub.add_parser("drift", help="T2 Slow vs locked-baseline feature drift")
    p2.add_argument("--slow-run-dir", required=True)
    p2.add_argument(
        "--baseline-run-dir", required=True, help="turbo rolling_sim run dir"
    )
    p2.add_argument("--strategy", required=True)
    p2.add_argument("--output")
    p2.set_defaults(func=_cmd_drift)

    p3 = sub.add_parser("digest", help="T3 Feature changes + monthly R delta")
    p3.add_argument("--slow-run-dir", required=True)
    p3.add_argument("--baseline-run-dir", required=True)
    p3.add_argument("--strategy", required=True)
    p3.add_argument(
        "--attribution",
        default="linear_days",
        choices=["entry_month", "exit_month", "linear_days"],
    )
    p3.add_argument("--output")
    p3.set_defaults(func=_cmd_digest)

    p4 = sub.add_parser("consensus", help="T4 Multi-method consensus matrix")
    p4.add_argument("--run-dir", required=True)
    p4.add_argument("--strategy", required=True)
    p4.add_argument("--output")
    p4.set_defaults(func=_cmd_consensus)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
