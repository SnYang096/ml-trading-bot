#!/usr/bin/env python3
"""Tier-0 regime threshold calibration driver.

季度跑一次（或 regime_drift_monitor 触发时）。

Tier-0 contract:
    1. 在共享标注 parquet 上对 ``tpc_semantic_chop`` 扫 plateau；
    2. 用 ``plateau_stability.decide_plateau_update`` 检查与上一次 plateau 是否重叠：
       - 重叠 → ADOPT 新 mid
       - 不重叠 → ALERT，保留旧 value，要求人工复核
    3. 默认 ``--dry-run``：只产出 ``results/regime_threshold/<ts>/proposal.json``；
       带 ``--apply`` 才会原子写回 N 个 ``archetypes/regime.yaml``（先全部生成 tmp，
       全部成功后再 rename）；
    4. 决策日志写入 ``docs/decisions/regime_thresholds/<ts>.md`` 供季度复盘。

用法示例：
    python scripts/regime_threshold_calibrate.py \\
        --strategies bpc,tpc,me,srb \\
        --labeled-parquet results/<...>/features_labeled.parquet \\
        --policy keep_if_no_overlap \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.regime.threshold_calibrator import (
    StrategyCalibration,
    calibrate_strategies,
)


def parse_strategies(csv: str) -> List[str]:
    return [s.strip() for s in csv.split(",") if s.strip()]


def build_items(
    strategies: Iterable[str], parquet_path: Path, strategies_root: Path
) -> List[StrategyCalibration]:
    items: List[StrategyCalibration] = []
    for s in strategies:
        items.append(
            StrategyCalibration(
                strategy=s,
                regime_yaml_path=strategies_root / s / "archetypes" / "regime.yaml",
                parquet_path=parquet_path,
            )
        )
    return items


def write_proposal_json(out_path: Path, items: List[StrategyCalibration]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [
            {
                "strategy": it.strategy,
                "regime_yaml_path": str(it.regime_yaml_path),
                "parquet_path": str(it.parquet_path),
                "current_value": it.current_value,
                "last_plateau": (
                    {
                        "start": it.last_plateau.start,
                        "end": it.last_plateau.end,
                        "mid": it.last_plateau.mid,
                    }
                    if it.last_plateau is not None
                    else None
                ),
                "new_plateau": (
                    {
                        "start": it.new_plateau.start,
                        "end": it.new_plateau.end,
                        "mid": it.new_plateau.mid,
                    }
                    if it.new_plateau is not None
                    else None
                ),
                "decision": it.decision,
                "skipped_reason": it.skipped_reason,
            }
            for it in items
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_decision_log(
    out_path: Path, items: List[StrategyCalibration], feature: str, operator_str: str
) -> None:
    lines = [
        f"# Tier-0 regime threshold calibration",
        f"",
        f"- Feature: `{feature}`",
        f"- Operator: `{operator_str}`",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"",
        f"## Proposal",
        f"",
        f"| Strategy | Current | Last plateau | New plateau | Action | Chosen | Reason |",
        f"|----------|---------|--------------|-------------|--------|--------|--------|",
    ]
    for it in items:
        if it.skipped_reason:
            lines.append(
                f"| {it.strategy} | - | - | - | SKIPPED | - | {it.skipped_reason} |"
            )
            continue
        dec = it.decision or {}
        old_str = (
            f"[{it.last_plateau.start:.4g},{it.last_plateau.end:.4g}]"
            if it.last_plateau is not None
            else "n/a"
        )
        new_str = (
            f"[{it.new_plateau.start:.4g},{it.new_plateau.end:.4g}]"
            if it.new_plateau is not None
            else "n/a"
        )
        lines.append(
            f"| {it.strategy} | {it.current_value} | {old_str} | {new_str} "
            f"| {dec.get('action')} | {dec.get('chosen_value')} | {dec.get('reason', '')} |"
        )
    lines.append("")
    lines.append("## Approval checklist (manual)")
    lines.append("- [ ] reviewed each ALERT decision and signed off")
    lines.append(
        "- [ ] confirmed ADOPT proposals do not contradict ongoing live trades"
    )
    lines.append("- [ ] ran `regime_drift_monitor` after apply to verify distribution")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def apply_updates_atomic(items: List[StrategyCalibration]) -> None:
    """先把所有 updated_regime 写到 tmp，全部成功后再 rename。"""
    tmp_paths: List[tuple[Path, Path]] = []
    try:
        for it in items:
            if it.updated_regime is None or it.skipped_reason is not None:
                continue
            tmp = it.regime_yaml_path.with_suffix(".yaml.tmp_calibrate")
            tmp.write_text(
                yaml.safe_dump(it.updated_regime, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            tmp_paths.append((tmp, it.regime_yaml_path))
        for tmp, dst in tmp_paths:
            os.replace(tmp, dst)
    except Exception:
        for tmp, _ in tmp_paths:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        raise


def main() -> int:
    p = argparse.ArgumentParser(description="Tier-0 regime threshold calibration")
    p.add_argument(
        "--strategies",
        default="bpc,tpc,me,srb",
        help="comma-separated strategy slugs (default: bpc,tpc,me,srb)",
    )
    p.add_argument(
        "--labeled-parquet",
        required=True,
        help="features_labeled.parquet (共享数据源，所有策略用同一份)",
    )
    p.add_argument(
        "--strategies-root",
        default="config/strategies",
        help="archetype YAML root",
    )
    p.add_argument("--feature", default="tpc_semantic_chop")
    p.add_argument("--operator", default="<=")
    p.add_argument("--label-col", default="success_no_rr_extreme")
    p.add_argument("--scan-points", type=int, default=25)
    p.add_argument(
        "--policy",
        default="keep_if_no_overlap",
        choices=["keep_if_no_overlap", "adopt_anyway"],
    )
    p.add_argument(
        "--output-dir",
        default="results/regime_threshold",
        help="proposal.json + decision log root",
    )
    p.add_argument(
        "--decisions-root",
        default="docs/decisions/regime_thresholds",
        help="markdown decision log destination",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="实际写回 archetypes/regime.yaml（默认 dry-run，仅生成 proposal.json + log）",
    )
    args = p.parse_args()

    parquet_path = Path(args.labeled_parquet)
    if not parquet_path.is_absolute():
        parquet_path = (PROJECT_ROOT / parquet_path).resolve()

    strategies_root = Path(args.strategies_root)
    if not strategies_root.is_absolute():
        strategies_root = (PROJECT_ROOT / strategies_root).resolve()

    strategies = parse_strategies(args.strategies)
    items = build_items(strategies, parquet_path, strategies_root)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timestamp_iso = datetime.now(timezone.utc).isoformat()

    calibrate_strategies(
        items,
        feature=args.feature,
        operator_str=args.operator,
        label_col=args.label_col,
        scan_points=int(args.scan_points),
        policy=args.policy,
        timestamp_iso=timestamp_iso,
    )

    out_dir = (PROJECT_ROOT / args.output_dir / ts).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_proposal_json(out_dir / "proposal.json", items)
    write_decision_log(
        Path(args.decisions_root) / f"{ts}.md",
        items,
        feature=args.feature,
        operator_str=args.operator,
    )

    n_adopt = sum(1 for it in items if it.decision and it.decision["action"] == "ADOPT")
    n_alert = sum(1 for it in items if it.decision and it.decision["action"] == "ALERT")
    n_skip = sum(1 for it in items if it.skipped_reason)
    print(
        f"Calibrated {len(items)} strategies: ADOPT={n_adopt} ALERT={n_alert} SKIPPED={n_skip}"
    )
    print(f"  proposal: {out_dir / 'proposal.json'}")
    print(f"  decision log: {Path(args.decisions_root) / (ts + '.md')}")

    if not args.apply:
        print("[dry-run] no YAML changed. Re-run with --apply to commit.")
        return 0

    apply_updates_atomic(items)
    print(f"Applied {len(items) - n_skip} regime.yaml updates atomically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
