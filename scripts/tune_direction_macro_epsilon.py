#!/usr/bin/env python3
"""
Grid-search direction thresholds:

- dual_position_agree_deadband: epsilon grid (macro VWAP + EMA position).
- single_position_band: inner_abs × outer_abs grid (e.g. BPC VWAP band).

Uses the same cascade as compute_direction_series / live: clone direction_rules from
archetypes/direction.yaml, override only the tuned rule, then validate_direction_quality.

Pipeline: enable under fast_loop.direction_tuning.macro_epsilon_grid in the train
pipeline YAML; run_strategy_pipeline patches exp strategies before Direction Validate.
CLI remains available for ad-hoc sweeps.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.live.direction_rule_ops import (
    parse_dual_rule,
    parse_single_position_band_rule,
)

from scripts.direction_strict_validation import (
    compute_direction_series_from_rules,
    validate_direction_quality,
)


def _parse_epsilon_grid(args: argparse.Namespace) -> List[float]:
    if args.epsilon_grid:
        out: List[float] = []
        for part in args.epsilon_grid.split(","):
            part = part.strip()
            if part:
                out.append(float(part))
        return out
    lo, hi, steps = args.epsilon_min, args.epsilon_max, args.epsilon_steps
    if steps < 2:
        return [float(lo)]
    step = (hi - lo) / (steps - 1)
    return [round(lo + i * step, 10) for i in range(steps)]


def _patch_dual_epsilon(rules: list, epsilon: float) -> bool:
    for r in rules:
        if parse_dual_rule(r) is not None:
            r["epsilon"] = float(epsilon)
            return True
    return False


def _comma_separated_floats(raw: Any) -> List[float]:
    if raw is None or not str(raw).strip():
        return []
    out: List[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def inner_abs_grid_values_from_config(meps: Dict[str, Any]) -> List[float]:
    """Optional inner_abs grid for single_position_band (overrides epsilon_grid for band path)."""
    if not isinstance(meps, dict):
        return []
    return _comma_separated_floats(meps.get("inner_abs_grid"))


def outer_abs_grid_values_from_config(meps: Dict[str, Any]) -> List[float]:
    """Optional outer_abs grid for single_position_band."""
    if not isinstance(meps, dict):
        return []
    return _comma_separated_floats(meps.get("outer_abs_grid"))


def load_archetype_direction_rules(
    strategies_root: Path, strategy: str
) -> tuple[List[dict], bool]:
    """Return (direction_rules list, file_exists)."""
    direction_path = strategies_root / strategy / "archetypes" / "direction.yaml"
    if not direction_path.exists():
        return [], False
    with open(direction_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    rules = cfg.get("direction_rules")
    if not isinstance(rules, list):
        return [], True
    return rules, True


def archetypes_has_dual_band_rules(
    strategies_root: Path, strategy: str
) -> tuple[bool, bool]:
    """Whether archetypes/direction.yaml contains a dual deadband and/or single_position_band rule."""
    rules, exists = load_archetype_direction_rules(strategies_root, strategy)
    if not exists or not rules:
        return False, False
    has_dual = any(parse_dual_rule(r) is not None for r in rules)
    has_band = any(parse_single_position_band_rule(r) is not None for r in rules)
    return has_dual, has_band


def epsilon_grid_values_from_config(meps: Dict[str, Any]) -> List[float]:
    """Build ε list from fast_loop.direction_tuning.macro_epsilon_grid."""
    if not isinstance(meps, dict):
        return []
    raw_grid = meps.get("epsilon_grid")
    if raw_grid is not None and str(raw_grid).strip():
        out: List[float] = []
        for part in str(raw_grid).split(","):
            part = part.strip()
            if part:
                out.append(float(part))
        return out
    lo = float(meps.get("epsilon_min", 0.001))
    hi = float(meps.get("epsilon_max", 0.02))
    steps = int(meps.get("epsilon_steps", 10))
    if steps < 2:
        return [lo]
    step = (hi - lo) / (steps - 1)
    return [round(lo + i * step, 10) for i in range(steps)]


def run_macro_epsilon_sweep(
    df: pd.DataFrame,
    strategy: str,
    strategies_root: Path,
    epsilons: List[float],
) -> tuple[List[Dict[str, Any]], bool]:
    """
    For each ε, compute direction via same rules as live and score with validate_direction_quality.

    Returns (rows, has_dual_rule). If archetypes/direction.yaml has no dual deadband rule,
    returns ([], False).
    """
    direction_path = strategies_root / strategy / "archetypes" / "direction.yaml"
    if not direction_path.exists():
        return [], False
    with open(direction_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_rules = copy.deepcopy(cfg.get("direction_rules", []))
    if not base_rules or not any(parse_dual_rule(r) is not None for r in base_rules):
        return [], False

    rows: List[Dict[str, Any]] = []
    for eps in epsilons:
        rules = copy.deepcopy(base_rules)
        if not _patch_dual_epsilon(rules, eps):
            continue
        direction = compute_direction_series_from_rules(df, rules)
        q = validate_direction_quality(strategy, df, direction)
        rows.append(_row(eps, q))
    return rows, True


def _patch_band_abs(rules: list, inner: float, outer: float) -> bool:
    for r in rules:
        if parse_single_position_band_rule(r) is not None:
            r["inner_abs"] = float(inner)
            r["outer_abs"] = float(outer)
            return True
    return False


def _baseline_band_abs(rules: list) -> Optional[tuple[float, float]]:
    for r in rules:
        p = parse_single_position_band_rule(r)
        if p is not None:
            _f, inn, out = p
            return float(inn), float(out)
    return None


def run_single_position_band_sweep(
    df: pd.DataFrame,
    strategy: str,
    strategies_root: Path,
    inner_list: Optional[List[float]],
    outer_list: Optional[List[float]],
) -> tuple[List[Dict[str, Any]], bool]:
    """
    Grid inner_abs × outer_abs on the first single_position_band rule.

    If inner_list / outer_list is None or empty, that axis uses the baseline from YAML.
    Skips pairs with inner >= outer or inner < 0.

    Returns (rows, has_band_rule).
    """
    direction_path = strategies_root / strategy / "archetypes" / "direction.yaml"
    if not direction_path.exists():
        return [], False
    with open(direction_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_rules = copy.deepcopy(cfg.get("direction_rules", []))
    if not base_rules or not any(
        parse_single_position_band_rule(r) is not None for r in base_rules
    ):
        return [], False

    base = _baseline_band_abs(base_rules)
    if base is None:
        return [], False
    b_inn, b_out = base
    inners = list(inner_list) if inner_list else [b_inn]
    outers = list(outer_list) if outer_list else [b_out]

    rows: List[Dict[str, Any]] = []
    for inn in inners:
        for out in outers:
            if inn < 0 or out <= inn:
                continue
            rules = copy.deepcopy(base_rules)
            if not _patch_band_abs(rules, inn, out):
                continue
            direction = compute_direction_series_from_rules(df, rules)
            q = validate_direction_quality(strategy, df, direction)
            rows.append(_row_band(inn, out, q))
    return rows, True


def pick_best_median_epsilon(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Pick ε with highest median_in_direction among status==OK rows."""
    ok = [
        x
        for x in rows
        if x.get("status") == "OK" and x.get("median_in_direction") is not None
    ]
    if not ok:
        return None
    return float(max(ok, key=lambda x: float(x["median_in_direction"]))["epsilon"])


def pick_best_median_band(
    rows: List[Dict[str, Any]],
) -> Optional[tuple[float, float]]:
    """Pick (inner_abs, outer_abs) with highest median_in_direction among status==OK rows."""
    ok = [
        x
        for x in rows
        if x.get("status") == "OK" and x.get("median_in_direction") is not None
    ]
    if not ok:
        return None
    best = max(ok, key=lambda x: float(x["median_in_direction"]))
    return float(best["inner_abs"]), float(best["outer_abs"])


def apply_dual_rule_epsilon_to_strategy_configs(
    strategy_dir: Path,
    epsilon: float,
    *,
    patch_workspace: bool = True,
) -> bool:
    """
    Set epsilon on the first dual_position_agree_deadband rule in:
    archetypes/direction.yaml and optionally features_direction.yaml.
    """
    any_patched = False
    targets = [strategy_dir / "archetypes" / "direction.yaml"]
    if patch_workspace:
        targets.append(strategy_dir / "features_direction.yaml")
    for path in targets:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rules = data.get("direction_rules")
        if not isinstance(rules, list):
            continue
        file_patched = False
        for r in rules:
            if parse_dual_rule(r) is not None:
                r["epsilon"] = float(epsilon)
                file_patched = True
                any_patched = True
                break
        if file_patched:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return any_patched


def apply_band_thresholds_to_strategy_configs(
    strategy_dir: Path,
    inner_abs: float,
    outer_abs: float,
    *,
    patch_workspace: bool = False,
) -> bool:
    """
    Set inner_abs / outer_abs on the first single_position_band rule.
    By default only archetypes/direction.yaml (BPC skips features_direction workspace).
    """
    any_patched = False
    targets = [strategy_dir / "archetypes" / "direction.yaml"]
    if patch_workspace:
        targets.append(strategy_dir / "features_direction.yaml")
    for path in targets:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rules = data.get("direction_rules")
        if not isinstance(rules, list):
            continue
        file_patched = False
        for r in rules:
            if parse_single_position_band_rule(r) is not None:
                r["inner_abs"] = float(inner_abs)
                r["outer_abs"] = float(outer_abs)
                file_patched = True
                any_patched = True
                break
        if file_patched:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return any_patched


def _row(eps: float, q: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if q is None:
        return {"epsilon": eps, "status": "no_rr_col"}
    st = q.get("status", "")
    if st == "INSUFFICIENT_DATA":
        return {
            "epsilon": eps,
            "status": st,
            "n_valid": q.get("n_valid"),
        }
    return {
        "epsilon": eps,
        "status": st,
        "median_in_direction": q.get("median_in_direction"),
        "bad_rate_in_direction": q.get("bad_rate_in_direction"),
        "p_random": q.get("p_random"),
        "n_valid": q.get("n_valid"),
    }


def _row_band(
    inner: float, outer: float, q: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    base = _row(0.0, q)
    base.pop("epsilon", None)
    base["inner_abs"] = float(inner)
    base["outer_abs"] = float(outer)
    return base


def main() -> int:
    p = argparse.ArgumentParser(
        description="Grid dual deadband epsilon and/or single_position_band inner/outer"
    )
    p.add_argument(
        "--logs",
        type=Path,
        required=True,
        help="features_labeled.parquet or predictions",
    )
    p.add_argument("--strategy", type=str, default="bpc")
    p.add_argument(
        "--strategies-root",
        type=Path,
        default=PROJECT_ROOT / "config" / "strategies",
        help="Root containing <strategy>/archetypes/direction.yaml",
    )
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--epsilon-grid",
        type=str,
        help="Comma-separated epsilons (dual rule), or inner_abs candidates for band if no dual rule",
    )
    g.add_argument(
        "--epsilon-range", action="store_true", help="Use --epsilon-min/max/steps"
    )
    p.add_argument("--epsilon-min", type=float, default=0.001)
    p.add_argument("--epsilon-max", type=float, default=0.02)
    p.add_argument("--epsilon-steps", type=int, default=10)
    p.add_argument(
        "--inner-abs-grid",
        default=None,
        help="Comma-separated inner_abs values for single_position_band",
    )
    p.add_argument(
        "--outer-abs-grid",
        default=None,
        help="Comma-separated outer_abs values for single_position_band",
    )
    p.add_argument(
        "--print-best",
        action="store_true",
        help="Print best threshold(s) by max median_in_direction among OK rows",
    )
    args = p.parse_args()

    if args.epsilon_range:
        epsilons = _parse_epsilon_grid(args)
    elif args.epsilon_grid:
        epsilons = _parse_epsilon_grid(
            argparse.Namespace(
                epsilon_grid=args.epsilon_grid,
                epsilon_min=args.epsilon_min,
                epsilon_max=args.epsilon_max,
                epsilon_steps=args.epsilon_steps,
            )
        )
    else:
        epsilons = []

    inners = _comma_separated_floats(args.inner_abs_grid)
    outers = _comma_separated_floats(args.outer_abs_grid)

    if not epsilons and not inners and not outers:
        p.error(
            "Provide --epsilon-grid/--epsilon-range and/or --inner-abs-grid/--outer-abs-grid"
        )

    df = pd.read_parquet(args.logs)
    sr = Path(args.strategies_root)
    has_dual, has_band = archetypes_has_dual_band_rules(sr, args.strategy)

    rows: List[Dict[str, Any]] = []
    mode = ""

    if has_dual and epsilons:
        rows, ok = run_macro_epsilon_sweep(df, args.strategy, sr, epsilons)
        if ok and rows:
            mode = "dual"

    if not rows and has_band:
        band_inners = inners if inners else epsilons
        band_outers = outers
        if band_inners or band_outers:
            rows, ok = run_single_position_band_sweep(
                df,
                args.strategy,
                sr,
                band_inners if band_inners else None,
                band_outers if band_outers else None,
            )
            if ok and rows:
                mode = "band"

    if not rows:
        print(
            "No sweep rows: check archetypes/direction.yaml (dual or single_position_band) "
            "and grid arguments",
            file=sys.stderr,
        )
        return 1

    if mode == "dual":
        print(
            f"{'epsilon':>12} {'status':>18} {'med_rr×dir':>12} "
            f"{'bad_rate':>10} {'p_perm':>8} {'n_valid':>8}"
        )
        for r in rows:
            print(
                f"{r['epsilon']:>12.6g} {str(r['status']):>18} "
                f"{r.get('median_in_direction', ''):>12} "
                f"{r.get('bad_rate_in_direction', ''):>10} "
                f"{r.get('p_random', ''):>8} "
                f"{r.get('n_valid', ''):>8}"
            )
        if args.print_best:
            best_eps = pick_best_median_epsilon(rows)
            if best_eps is not None:
                print(f"\nBest epsilon (max median_in_direction): {best_eps}")
            else:
                print("\nNo OK row to pick best from.")
    else:
        print(
            f"{'inner_abs':>12} {'outer_abs':>12} {'status':>18} {'med_rr×dir':>12} "
            f"{'bad_rate':>10} {'p_perm':>8} {'n_valid':>8}"
        )
        for r in rows:
            print(
                f"{r['inner_abs']:>12.6g} {r['outer_abs']:>12.6g} {str(r['status']):>18} "
                f"{r.get('median_in_direction', ''):>12} "
                f"{r.get('bad_rate_in_direction', ''):>10} "
                f"{r.get('p_random', ''):>8} "
                f"{r.get('n_valid', ''):>8}"
            )
        if args.print_best:
            best_pair = pick_best_median_band(rows)
            if best_pair is not None:
                inn, out = best_pair
                print(
                    f"\nBest inner_abs, outer_abs (max median_in_direction): {inn}, {out}"
                )
            else:
                print("\nNo OK row to pick best from.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
