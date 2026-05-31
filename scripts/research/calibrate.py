"""mlbot research calibrate — write structured draft yaml from plateau / lift json."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from scripts.research._common import PROJECT_ROOT
from src.research.gate_when import (
    apply_gate_threshold_to_when,
    gate_threshold_skip_reason,
    resolve_gate_deny_operator,
)

_OPTIMIZER_APPLY_STATUSES = frozenset(
    {"stable_plateau_found", "no_stable_plateau", "robust_but_unproven"}
)
_OPTIMIZER_SKIP_STATUSES = frozenset({"skipped_locked", "frozen", "skip"})


@dataclass
class CalibrateSkip:
    rule_id: str
    reason: str
    detail: Optional[str] = None


@dataclass
class CalibrateResult:
    draft_text: str
    skips: List[CalibrateSkip] = field(default_factory=list)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar_draft(data: Dict[str, Any], src: Path) -> CalibrateResult:
    rec = (
        data.get("recommended")
        or data.get("mid")
        or data.get("recommended_threshold")
        or data.get("plateau_mid")
    )
    text = (
        f"# DRAFT — human review required before promote\n"
        f"# source: {src}\n"
        f"recommended_threshold: {rec}\n"
    )
    return CalibrateResult(draft_text=text)


def _gate_rule_threshold_update(
    rule: Dict[str, Any],
    opt: Dict[str, Any],
    *,
    rule_id: str,
) -> tuple[Dict[str, Any], Optional[CalibrateSkip]]:
    """Apply lift optimization result; preserve ``all_of`` siblings for other features."""
    out = dict(rule)
    feature = opt.get("feature")
    deny_op = opt.get("operator") or opt.get("deny_operator")
    if not feature or not deny_op:
        return out, CalibrateSkip(
            rule_id,
            "missing_optimization_fields",
            detail=f"feature={feature!r} operator={deny_op!r}",
        )

    interval = opt.get("threshold_interval") or {}
    interval_pair: Optional[tuple[float, float]] = None
    if interval.get("start") is not None and interval.get("end") is not None:
        interval_pair = (float(interval["start"]), float(interval["end"]))

    rec = opt.get("recommended_threshold") or opt.get("recommended")
    if rec is None and interval_pair is None:
        return out, CalibrateSkip(
            rule_id,
            "missing_recommended_threshold",
            detail="no recommended_threshold and no threshold_interval",
        )

    when = rule.get("when") or {}
    if not isinstance(when, dict):
        return out, CalibrateSkip(rule_id, "invalid_when", detail=type(when).__name__)

    skip_reason = gate_threshold_skip_reason(when, str(feature), interval=interval_pair)
    if skip_reason:
        return out, CalibrateSkip(
            rule_id,
            skip_reason,
            detail=f"feature={feature} interval={interval_pair}",
        )

    new_when = apply_gate_threshold_to_when(
        when,
        str(feature),
        str(deny_op),
        float(rec if rec is not None else interval_pair[0]),  # type: ignore[index]
        interval=interval_pair,
    )
    if new_when == when:
        return out, CalibrateSkip(
            rule_id,
            "when_unchanged",
            detail=f"feature={feature} operator={deny_op}",
        )

    out["when"] = new_when
    return out, None


def _format_skip_header(skips: List[CalibrateSkip]) -> str:
    if not skips:
        return ""
    parts = [f"{s.rule_id} ({s.reason})" for s in skips]
    return f"# calibrate skips ({len(skips)}): " + "; ".join(parts) + "\n"


def _gate_batch_draft(
    data: Dict[str, Any],
    src: Path,
    *,
    strategy: Optional[str],
    strategies_root: Path,
) -> CalibrateResult:
    strategy = strategy or data.get("strategy")
    if not strategy:
        raise ValueError("gate batch json requires --strategy or strategy field")

    gate_path = strategies_root / strategy / "archetypes" / "gate.yaml"
    if not gate_path.is_file():
        raise ValueError(f"gate.yaml not found: {gate_path}")

    base = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}
    rules_map: Dict[str, Dict[str, Any]] = data.get("rules") or {}
    skips: List[CalibrateSkip] = []

    for section in ("hard_gates", "system_safety"):
        items = base.get(section) or []
        if not isinstance(items, list):
            continue
        updated: List[Dict[str, Any]] = []
        for rule in items:
            if not isinstance(rule, dict):
                updated.append(rule)
                continue
            rid = str(rule.get("id", ""))
            opt = rules_map.get(rid)
            if not opt:
                updated.append(rule)
                continue
            status = opt.get("status")
            if status in _OPTIMIZER_SKIP_STATUSES:
                skips.append(
                    CalibrateSkip(
                        rid,
                        f"optimizer_{status}",
                        detail=opt.get("reason"),
                    )
                )
                updated.append(rule)
                continue
            if status not in _OPTIMIZER_APPLY_STATUSES:
                skips.append(
                    CalibrateSkip(
                        rid,
                        "optimizer_status_not_applicable",
                        detail=str(status),
                    )
                )
                updated.append(rule)
                continue
            new_rule, skip = _gate_rule_threshold_update(rule, opt, rule_id=rid)
            if skip:
                skips.append(skip)
            updated.append(new_rule)
        base[section] = updated

    header = (
        f"# DRAFT — human review required before promote\n"
        f"# source: {src}\n"
        f"# strategy: {strategy}\n" + _format_skip_header(skips)
    )
    return CalibrateResult(
        draft_text=header + yaml.safe_dump(base, sort_keys=False, allow_unicode=True),
        skips=skips,
    )


def _single_lift_gate_draft(data: Dict[str, Any], src: Path) -> CalibrateResult:
    feature = data.get("feature")
    deny_op = resolve_gate_deny_operator(
        str(data.get("deny_operator") or data.get("operator") or "gt")
    )
    rec = data.get("recommended") or data.get("recommended_threshold")
    interval = None
    if (
        data.get("start_threshold") is not None
        and data.get("end_threshold") is not None
    ):
        interval = (float(data["start_threshold"]), float(data["end_threshold"]))
    when = apply_gate_threshold_to_when(
        {},
        str(feature),
        deny_op,
        float(rec if rec is not None else interval[0]),  # type: ignore[index]
        interval=interval,
    )
    block = {
        "hard_gates": [
            {
                "id": f"draft_gate_{feature}",
                "phase": "hard_gate",
                "when": when,
                "then": {"action": "deny"},
                "comment": f"draft from {src.name}",
            }
        ]
    }
    header = f"# DRAFT — human review required before promote\n" f"# source: {src}\n"
    return CalibrateResult(
        draft_text=header + yaml.safe_dump(block, sort_keys=False, allow_unicode=True)
    )


def calibrate_draft_result(
    data: Dict[str, Any],
    src: Path,
    *,
    strategy: Optional[str] = None,
    strategies_root: Optional[Path] = None,
) -> CalibrateResult:
    if data.get("kpi") == "lift" and isinstance(data.get("rules"), dict):
        root = strategies_root or (PROJECT_ROOT / "config/strategies")
        return _gate_batch_draft(data, src, strategy=strategy, strategies_root=root)
    if data.get("kpi") == "lift" and data.get("feature"):
        return _single_lift_gate_draft(data, src)
    return _scalar_draft(data, src)


def calibrate_draft_text(
    data: Dict[str, Any],
    src: Path,
    *,
    strategy: Optional[str] = None,
    strategies_root: Optional[Path] = None,
) -> str:
    return calibrate_draft_result(
        data, src, strategy=strategy, strategies_root=strategies_root
    ).draft_text


def write_calibrate_skips(skips: List[CalibrateSkip], path: Path) -> None:
    payload = {
        "skip_count": len(skips),
        "skips": [asdict(s) for s in skips],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research calibrate (structured draft yaml from plateau json)"
    )
    p.add_argument("--from-plateau", required=True)
    p.add_argument("--output", required=True, help="Draft yaml path")
    p.add_argument("--strategy", default=None)
    p.add_argument("--strategies-root", default="config/strategies")
    p.add_argument(
        "--skips-output",
        default=None,
        help="Skip manifest json (default: <output>.skips.json when skips exist)",
    )
    args = p.parse_args(argv)

    src = Path(args.from_plateau)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    data = _load_json(src)
    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    try:
        result = calibrate_draft_result(
            data,
            src,
            strategy=args.strategy,
            strategies_root=PROJECT_ROOT / args.strategies_root,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.draft_text, encoding="utf-8")
    print(f"wrote draft {out}")

    if result.skips:
        skips_path = (
            Path(args.skips_output)
            if args.skips_output
            else out.with_suffix(out.suffix + ".skips.json")
        )
        if not skips_path.is_absolute():
            skips_path = PROJECT_ROOT / skips_path
        write_calibrate_skips(result.skips, skips_path)
        print(f"wrote skip manifest ({len(result.skips)} rules) {skips_path}")
        for skip in result.skips:
            detail = f" — {skip.detail}" if skip.detail else ""
            print(f"  skip: {skip.rule_id} ({skip.reason}){detail}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
