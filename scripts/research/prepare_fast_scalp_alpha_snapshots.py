#!/usr/bin/env python3
"""Materialize fast_scalp alpha-rebuild variant trees under config_experiments/ (TPC-style).

Each snapshot is a frozen copy of deploy ``tree_strategies/fast_scalp`` only.
Symbol cohort (alts_4 / majors_2 / pooled_6) is chosen at event/score-export time,
not by duplicating strategy packages.

Usage:
  PYTHONPATH=src:scripts python scripts/research/prepare_fast_scalp_alpha_snapshots.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.strategy_layout import copy_strategy_package  # noqa: E402

DEPLOY_ROOT = PROJECT_ROOT / "config/strategies/tree_strategies"
EXP_ROOT = PROJECT_ROOT / "config_experiments"
EXP_OVERRIDES = (
    PROJECT_ROOT / "config/experiments/20260602_fast_scalp_tree_validate/overrides"
)
STRATEGY_PKG = "fast_scalp"
PACKAGES = (STRATEGY_PKG,)


def _direction_path(root: Path) -> Path:
    return root / STRATEGY_PKG / "archetypes" / "direction.yaml"


def _archetype_path(root: Path, name: str) -> Path:
    return root / STRATEGY_PKG / "archetypes" / f"{name}.yaml"


def _read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _copy_tree(dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for pkg in PACKAGES:
        copy_strategy_package(DEPLOY_ROOT / pkg, dst_root / pkg, dirs_exist_ok=True)
        exp_dir = dst_root / pkg / "archetypes" / "_experiment"
        if exp_dir.exists():
            shutil.rmtree(exp_dir)


def _patch_direction_short(root: Path) -> None:
    path = _direction_path(root)
    data = _read_yaml(path)
    data["direction_filter"] = "short"
    _write_yaml(path, data)


def _patch_direction_both_sides(root: Path, *, tau: Dict[str, Any]) -> None:
    """Bidirectional H=3: signed pred + holdout τ (level mode)."""
    path = _direction_path(root)
    data = _read_yaml(path)
    data.pop("direction_filter", None)
    data["description"] = str(
        tau.get("description")
        or "H=3 signed both sides — holdout τ @ level (20260602 both_sides scan)"
    )
    data["thresholds"] = {
        "long_entry": float(tau["long_entry"]),
        "long_exit": float(tau.get("long_exit", tau["long_entry"] - 0.05)),
        "short_entry": float(tau["short_entry"]),
        "short_exit": float(tau.get("short_exit", tau["short_entry"] + 0.05)),
        "dead_zone": bool(tau.get("dead_zone", True)),
        "entry_mode": str(tau.get("entry_mode", "level")),
    }
    data["per_symbol_thresholds"] = {}
    _write_yaml(path, data)


def _patch_regime_empty(root: Path) -> None:
    path = _archetype_path(root, "regime")
    data = _read_yaml(path)
    data["rules"] = []
    data["description"] = "Regime OFF ablation — no EMA1200 dead-zone rules"
    _write_yaml(path, data)


def _patch_regime_ema_slope_side(root: Path) -> None:
    """EMA dead zone + position/slope side mask (G20)."""
    src = EXP_OVERRIDES / "regime_ema_slope_side_mask.yaml"
    if not src.is_file():
        raise FileNotFoundError(f"regime side mask override not found: {src}")
    raw = _read_yaml(src)
    path = _archetype_path(root, "regime")
    _write_yaml(path, raw)


def _patch_features_ema_slope(root: Path) -> None:
    """Ensure ema_1200_slope_10 is computed at event time (G20)."""
    path = root / STRATEGY_PKG / "features.yaml"
    data = _read_yaml(path)
    pipeline = data.setdefault("feature_pipeline", {})
    requested = list(pipeline.get("requested_features") or [])
    if "ema_1200_slope_f" not in requested:
        requested.append("ema_1200_slope_f")
    pipeline["requested_features"] = requested
    _write_yaml(path, data)


def _patch_execution(root: Path, patch: Dict[str, Any]) -> None:
    path = _archetype_path(root, "execution")
    base = _read_yaml(path)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **val}
        else:
            base[key] = val
    _write_yaml(path, base)


EXEC_TIMEOUT = {
    "stop_loss": {"initial_r": 50.0, "trailing": {"enabled": False}},
    "take_profit": {"enabled": False},
    "holding": {"max_holding_bars": 6, "time_stop_bars": 6},
}

EXEC_TIGHT = {
    "stop_loss": {"initial_r": 1.5, "trailing": {"enabled": False}},
    "take_profit": {"enabled": True, "target_r": 1.0},
    "holding": {"max_holding_bars": 6, "time_stop_bars": 6},
}

EXEC_TRAIL = {
    "stop_loss": {
        "initial_r": 2.5,
        "trailing": {"enabled": True, "activation_r": 1.0, "trail_r": 0.5},
    },
    "take_profit": {"enabled": True, "target_r": 1.5},
    "holding": {"max_holding_bars": 12, "time_stop_bars": 12},
}

# trend_scalp-inspired: wide catastrophic-like SL + tight TP (in R: 0.12R × 8R stop ≈ 1 ATR)
EXEC_WIDE_TIGHT = {
    "stop_loss": {"initial_r": 8.0, "trailing": {"enabled": False}},
    "take_profit": {"enabled": True, "target_r": 0.12},
    "holding": {"max_holding_bars": 24, "time_stop_bars": 24},
}

# pooled-6 g5-label holdout τ-scan recommended @ q=0.30
# (20260602 track_b/tau_scan_g5; non-plateau, all-negative vector Sharpe — event is judge)
G5LABEL_TAU = {
    "long_entry": -0.21338247736253388,
    "long_exit": -0.27,
    "short_entry": -0.3090097524537344,
    "short_exit": -0.25,
    "dead_zone": True,
    "entry_mode": "cross",
}


def _patch_g5label_tau(root: Path) -> None:
    """Apply g5-label holdout τ (not H=3 deploy τ) to direction.yaml."""
    path = _direction_path(root)
    data = _read_yaml(path)
    data["description"] = (
        "g5-label artifact holdout τ @ q=0.30 "
        "(20260602 track_b/tau_scan_g5; not H=3 deploy τ)"
    )
    data["thresholds"] = dict(G5LABEL_TAU)
    data["per_symbol_thresholds"] = {}
    _write_yaml(path, data)


def _patch_wide_tight_short(root: Path, *, regime_off: bool) -> None:
    _patch_direction_short(root)
    if regime_off:
        _patch_regime_empty(root)
    _patch_execution(root, EXEC_WIDE_TIGHT)


def _patch_wide_tight_short_gate(root: Path) -> None:
    _patch_direction_short(root)
    _patch_execution(root, EXEC_WIDE_TIGHT)
    _patch_gate_only(root)


def _patch_gate_only(root: Path) -> None:
    overlay_path = next((p for p in GATE_OVERLAY_CANDIDATES if p.is_file()), None)
    if overlay_path is None:
        raise FileNotFoundError(f"gate overlay not found in {GATE_OVERLAY_CANDIDATES}")
    _patch_gate_overlay(root, overlay_path)


def _build_snapshot(name: str, mutator: Callable[[Path], None] | None = None) -> Path:
    dst = EXP_ROOT / name
    _copy_tree(dst)
    if mutator:
        mutator(dst)
    return dst


def _patch_dual_head(root: Path) -> None:
    src = EXP_OVERRIDES / "direction_dual_head.yaml"
    if not src.is_file():
        raise FileNotFoundError(f"dual_head override not found: {src}")
    dual_block = (_read_yaml(src) or {}).get("dual_head")
    if not dual_block:
        raise ValueError(f"{src} missing dual_head block")
    path = _direction_path(root)
    data = _read_yaml(path)
    data["dual_head"] = dual_block
    _write_yaml(path, data)


def _patch_gate_overlay(root: Path, overlay_path: Path) -> None:
    if not overlay_path.is_file():
        raise FileNotFoundError(f"gate overlay not found: {overlay_path}")
    overlay = _read_yaml(overlay_path)
    for pkg in PACKAGES:
        path = root / pkg / "archetypes" / "gate.yaml"
        if not path.exists():
            continue
        data = _read_yaml(path)
        data["enabled"] = bool(overlay.get("enabled", True))
        if overlay.get("gate_model"):
            data["gate_model"] = overlay["gate_model"]
        if overlay.get("gate_feature_names"):
            data["gate_feature_names"] = overlay["gate_feature_names"]
        if overlay.get("reject_if_prob_bad_gt") is not None:
            data["reject_if_prob_bad_gt"] = overlay["reject_if_prob_bad_gt"]
        _write_yaml(path, data)


GATE_OVERLAY_CANDIDATES = (
    PROJECT_ROOT
    / "results/rd_loop/fast_scalp_tree_validate/track_a/gate/g3_ic_prune_v2/gate_overlay.yaml",
    PROJECT_ROOT
    / "results/rd_loop/fast_scalp_tree_validate/track_b/gate/ic_prune_v2/gate_overlay.yaml",
    PROJECT_ROOT
    / "results/rd_loop/fast_scalp_ic_plateau/track_exec_aligned/gate/ic_prune_v2/gate_overlay.yaml",
    PROJECT_ROOT
    / "results/rd_loop/fast_scalp_ic_plateau/alpha_rebuild/gate/adverse_gate_oos/gate_overlay.yaml",
)


def _load_h3_both_sides_tau() -> Dict[str, Any]:
    summary = (
        PROJECT_ROOT
        / "results/rd_loop/fast_scalp_tree_validate/track_a/tau_scan_h3_both/tau_scan_holdout_rr.json"
    )
    if not summary.is_file():
        raise FileNotFoundError(
            f"Run run_h3_tau_scan_both_sides.sh first; missing {summary}"
        )
    import json

    data = json.loads(summary.read_text(encoding="utf-8"))
    rec = data.get("recommended") or {}
    q = rec.get("top_quantile")
    if q is None:
        plateau = data.get("quantile_plateau") or {}
        best = plateau.get("recommended") or plateau.get("best") or {}
        q = best.get("top_quantile")
    rows = data.get("quantile_scan") or []
    row = next((r for r in rows if r.get("top_quantile") == q), rows[0] if rows else {})
    return {
        "description": f"H=3 both sides holdout τ q={q} (pred quantile scan)",
        "long_entry": float(
            row.get("pred_threshold_long", rec.get("long_entry_threshold", 0.28))
        ),
        "short_entry": float(
            row.get("pred_threshold_short", rec.get("short_entry_threshold", 0.18))
        ),
        "entry_mode": "level",
        "dead_zone": True,
        "top_quantile": q,
    }


def _patch_g19_both_sides(root: Path) -> None:
    _patch_direction_both_sides(root, tau=_load_h3_both_sides_tau())
    _patch_regime_empty(root)


def _patch_g20_both_sides_regime(root: Path) -> None:
    _patch_direction_both_sides(root, tau=_load_h3_both_sides_tau())
    _patch_features_ema_slope(root)
    _patch_regime_ema_slope_side(root)


def _patch_g3_h3_gate(root: Path) -> None:
    """G3 short + regime off + adverse gate trained on H=3 entry scores."""
    _patch_direction_short(root)
    _patch_regime_empty(root)
    overlay_path = (
        PROJECT_ROOT
        / "results/rd_loop/fast_scalp_tree_validate/track_a/gate/g3_ic_prune_v2/gate_overlay.yaml"
    )
    if not overlay_path.is_file():
        raise FileNotFoundError(f"g3 gate overlay not found: {overlay_path}")
    _patch_gate_overlay(root, overlay_path)


def _patch_dual_head_reg(root: Path) -> None:
    """Dual head block only (same as G7); scores come from reg-trained artifact inject."""
    _patch_dual_head(root)


def _patch_g3_gate(root: Path) -> None:
    overlay_path = next((p for p in GATE_OVERLAY_CANDIDATES if p.is_file()), None)
    if overlay_path is None:
        raise FileNotFoundError(f"gate overlay not found in {GATE_OVERLAY_CANDIDATES}")
    _patch_direction_short(root)
    _patch_regime_empty(root)
    _patch_gate_overlay(root, overlay_path)


def _patch_gate_exec(root: Path, exec_patch: Dict[str, Any]) -> None:
    _patch_gate_only(root)
    _patch_execution(root, exec_patch)


def _patch_g8_exec(root: Path, exec_patch: Dict[str, Any]) -> None:
    _patch_g3_gate(root)
    _patch_execution(root, exec_patch)


SNAPSHOTS: Dict[str, Callable[[Path], None] | None] = {
    "fast_scalp_alpha_G0_baseline_strategies": None,
    "fast_scalp_alpha_G1_short_only_strategies": _patch_direction_short,
    "fast_scalp_alpha_G2_regime_off_strategies": _patch_regime_empty,
    "fast_scalp_alpha_G3_short_regime_off_strategies": lambda r: (
        _patch_direction_short(r),
        _patch_regime_empty(r),
    ),
    "fast_scalp_alpha_G4_exec_timeout_strategies": lambda r: _patch_execution(
        r, EXEC_TIMEOUT
    ),
    "fast_scalp_alpha_G5_short_regimeoff_tight_exec_strategies": lambda r: (
        _patch_direction_short(r),
        _patch_regime_empty(r),
        _patch_execution(r, EXEC_TIGHT),
    ),
    "fast_scalp_alpha_G6_short_regimeoff_trail_exec_strategies": lambda r: (
        _patch_direction_short(r),
        _patch_regime_empty(r),
        _patch_execution(r, EXEC_TRAIL),
    ),
    "fast_scalp_alpha_G7_dual_head_strategies": _patch_dual_head,
    "fast_scalp_alpha_G17_dual_head_reg_strategies": _patch_dual_head_reg,
    "fast_scalp_alpha_G17_dual_head_reg_regimeoff_strategies": lambda r: (
        _patch_dual_head(r),
        _patch_regime_empty(r),
    ),
    "fast_scalp_alpha_G18_g3_h3_gate_strategies": _patch_g3_h3_gate,
    "fast_scalp_alpha_G19_h3_both_sides_strategies": _patch_g19_both_sides,
    "fast_scalp_alpha_G20_h3_both_sides_ema_regime_strategies": _patch_g20_both_sides_regime,
    "fast_scalp_alpha_G8_short_regimeoff_gate_strategies": _patch_g3_gate,
    "fast_scalp_alpha_G9_short_wide_tight_regimeon_strategies": lambda r: _patch_wide_tight_short(
        r, regime_off=False
    ),
    "fast_scalp_alpha_G10_short_wide_tight_regimeoff_strategies": lambda r: _patch_wide_tight_short(
        r, regime_off=True
    ),
    "fast_scalp_alpha_G11_short_wide_tight_regimeon_gate_strategies": _patch_wide_tight_short_gate,
    "fast_scalp_alpha_G12_short_regimeoff_gate_tight_exec_strategies": lambda r: _patch_g8_exec(
        r, EXEC_TIGHT
    ),
    "fast_scalp_alpha_G13_short_regimeoff_gate_wide_tight_exec_strategies": lambda r: _patch_g8_exec(
        r, EXEC_WIDE_TIGHT
    ),
    "fast_scalp_alpha_G14_g5label_g5exec_strategies": lambda r: (
        _patch_execution(r, EXEC_TIGHT),
        _patch_g5label_tau(r),
    ),
    "fast_scalp_alpha_G15_g10label_g10exec_strategies": lambda r: _patch_execution(
        r, EXEC_WIDE_TIGHT
    ),
    "fast_scalp_alpha_G16_g5label_g5exec_gate_strategies": lambda r: (
        _patch_gate_exec(r, EXEC_TIGHT),
        _patch_g5label_tau(r),
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Build only named snapshots (basename without path)",
    )
    args = ap.parse_args()
    only = set(args.only) if args.only else None

    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    for name, mutator in SNAPSHOTS.items():
        if only and name not in only:
            continue
        try:
            path = _build_snapshot(name, mutator)
            print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
        except FileNotFoundError as exc:
            print(f"Skip {name}: {exc}")
    built = len(SNAPSHOTS) if not only else len(only)
    print(f"Done: snapshots under config_experiments/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
