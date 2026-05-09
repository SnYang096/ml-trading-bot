#!/usr/bin/env python3
"""Validate multi-leg pipeline + constitution alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.pipeline.config import load_pipeline_config
from src.live_data_stream.constitution_config import (
    load_constitution_dict,
    resolve_multi_leg_risk_limits_from_constitution,
    validate_pipeline_constitution_alignment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _strategy_type(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("strategy_type", "") or "").strip().lower()


def _resolve_constitution_path(cfg: Dict[str, Any], override: str) -> Path:
    if str(override).strip():
        p = Path(override)
    else:
        p = Path(str(((cfg.get("constitution") or {}).get("path", "") or "").strip()))
        if not str(p).strip():
            p = Path("config/constitution/constitution.yaml")
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _ensure_strategy_files(cfg: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for name, scfg in (cfg.get("strategies") or {}).items():
        st = _strategy_type(scfg)
        if st not in {"grid", "dual_add_trend"}:
            errs.append(
                f"strategies.{name}: unsupported strategy_type={st!r} for multileg"
            )
            continue
        cfg_dir = str((scfg or {}).get("config", "") or "").strip()
        if not cfg_dir:
            errs.append(f"strategies.{name}: missing config directory")
            continue
        root = Path(cfg_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        req_file = "grid.yaml" if st == "grid" else "dual_add.yaml"
        for rel in (
            req_file,
            "features.yaml",
            "research/turbo.yaml",
            "archetypes/prefilter.yaml",
            "archetypes/execution.yaml",
        ):
            if not (root / rel).exists():
                errs.append(f"strategies.{name}: missing {(root / rel)}")
    return errs


def _ensure_risk_limits(limits: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    required = (
        "account_equity_usdt",
        "max_drawdown_pct",
        "max_gross_notional",
        "max_net_notional",
        "max_symbol_gross_notional",
        "max_symbol_net_notional",
        "max_resting_orders",
    )
    for key in required:
        v = limits.get(key)
        if v is None:
            errs.append(f"constitution.multi_leg missing {key}")
    return errs


def main() -> int:
    p = argparse.ArgumentParser(description="Validate multi-leg pipeline config.")
    p.add_argument(
        "--config",
        default="config/pipelines/multileg_orchestrate_2h.yaml",
        help="multi-leg pipeline YAML",
    )
    p.add_argument(
        "--constitution-yaml",
        default="",
        help="optional constitution YAML override",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_pipeline_config(cfg_path)

    constitution_path = _resolve_constitution_path(cfg, args.constitution_yaml)
    constitution = load_constitution_dict(str(constitution_path))
    if not constitution:
        raise ValueError(f"failed to load constitution: {constitution_path}")

    alignment = validate_pipeline_constitution_alignment(
        pipeline_cfg=cfg,
        constitution_cfg=constitution,
        context_label="multileg_validate_config",
    )
    limits = resolve_multi_leg_risk_limits_from_constitution(constitution)

    errors = []
    errors.extend(_ensure_strategy_files(cfg))
    errors.extend(_ensure_risk_limits(limits))

    if errors:
        raise ValueError(
            "multi-leg config validation failed:\n- " + "\n- ".join(errors)
        )

    report = {
        "ok": True,
        "config": str(cfg_path),
        "constitution": str(constitution_path),
        "strategies": sorted((cfg.get("strategies") or {}).keys()),
        "alignment": alignment,
        "risk_limits": limits,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
