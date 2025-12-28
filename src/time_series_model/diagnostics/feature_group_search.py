"""
Feature group search (greedy forward selection) for strategy configs.

This is designed to be reproducible and compatible with existing training/backtest pipeline:
we generate temporary strategy config dirs with different `features.yaml` contents, then call
`scripts/train_strategy_pipeline.py` and aggregate `strategy_pipeline_metrics.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import yaml


@dataclass(frozen=True)
class SearchConfig:
    base_strategy_dir: Path
    timeframe: str
    symbol: str
    start_date: str
    end_date: str
    test_size: float
    seeds: List[int]
    output_dir: Path
    deterministic: bool
    no_docker: bool


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, obj: dict) -> None:
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _backup_if_exists(path: Path) -> Optional[Path]:
    """Create a timestamped backup if file exists; returns backup path if created."""
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, bak)
    return bak


def _writeback_features_yaml(
    *,
    base_strategy_dir: Path,
    out_path: Path,
    requested_features: List[str],
    meta: dict,
    invert_candidates: Optional[List[str]] = None,
) -> dict:
    """
    Write a features.yaml compatible file with requested_features set to the provided list.
    Keeps existing keys from base strategy's features.yaml where possible (e.g., ensure_signal_column).
    Adds `feature_group_search` metadata for provenance.
    """
    base_yaml_path = base_strategy_dir / "features.yaml"
    base_obj = _load_yaml(base_yaml_path) if base_yaml_path.exists() else {}

    # Normalize structure
    base_obj.setdefault("feature_pipeline", {})
    fp = base_obj["feature_pipeline"] or {}
    fp["requested_features"] = list(requested_features)

    # Write back only the final invert_features, not an unbounded candidate list.
    # If `invert_candidates` is provided, we treat it as "candidate invert list" and prune by final features.
    inv_cand = invert_candidates
    if inv_cand is None:
        inv_cand = fp.get("invert_features") or []
    inv_cand = [str(x).strip() for x in (inv_cand or []) if str(x).strip()]
    req_set = set(requested_features)
    fp["invert_features"] = [f for f in inv_cand if f in req_set]

    base_obj["feature_pipeline"] = fp

    # Ensure name is present and indicate it's suggested
    base_name = base_obj.get("name") or base_strategy_dir.name
    base_obj["name"] = f"{base_name}__suggested"

    # Attach provenance metadata
    base_obj["feature_group_search"] = meta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bak = _backup_if_exists(out_path)
    _write_yaml(out_path, base_obj)
    return {"written": str(out_path), "backup": str(bak) if bak else None}


def _load_invert_candidates(path: Path) -> List[str]:
    """
    Load invert candidates from a YAML file.
    Supports:
      - YAML list: [feat1, feat2, ...]
      - full features config: {feature_pipeline: {invert_features: [...]}}
      - {invert_candidates: [...]} (preferred naming for pool artifacts)
    """
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return []
    if isinstance(obj, list):
        return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj, dict):
        if "invert_candidates" in obj and isinstance(obj["invert_candidates"], list):
            return [str(x).strip() for x in obj["invert_candidates"] if str(x).strip()]
        fp = obj.get("feature_pipeline")
        if isinstance(fp, dict) and isinstance(fp.get("invert_features"), list):
            return [str(x).strip() for x in fp["invert_features"] if str(x).strip()]
    raise ValueError(
        "invert-candidates-yaml must be a YAML list or contain feature_pipeline.invert_features"
    )


def _strategy_name(strategy_dir: Path) -> str:
    feats = _load_yaml(strategy_dir / "features.yaml")
    name = feats.get("name") or strategy_dir.name
    return str(name)


def _make_temp_strategy(
    *,
    base_dir: Path,
    tmp_root: Path,
    name_suffix: str,
    requested_features: List[str],
) -> Path:
    """
    Copy strategy dir to tmp_root/<base_name>__<suffix> and overwrite `features.yaml`.
    """
    base_name = base_dir.name
    out_dir = tmp_root / f"{base_name}__{name_suffix}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir)

    feats = _load_yaml(out_dir / "features.yaml")
    feats["name"] = f"{feats.get('name', base_name)}__{name_suffix}"
    feats.setdefault("feature_pipeline", {})
    feats["feature_pipeline"]["requested_features"] = list(requested_features)
    _write_yaml(out_dir / "features.yaml", feats)
    return out_dir


def _run_one_seed(
    *,
    strategy_dir: Path,
    cfg: SearchConfig,
    seed: int,
    out_root: Path,
) -> Path:
    """
    Run training pipeline; returns path to results.json.
    """
    _ensure_dir(out_root)
    args = [
        "--config",
        str(strategy_dir),
        "--symbol",
        cfg.symbol,
        "--data-path",
        "data/parquet_data",
        "--timeframe",
        cfg.timeframe,
        "--test-size",
        str(cfg.test_size),
        "--output-root",
        str(out_root),
        "--seed",
        str(seed),
    ]
    if cfg.deterministic:
        args.append("--deterministic")

    cmd = ["python3", "scripts/train_strategy_pipeline.py"] + args
    env = os.environ.copy()
    # Train pipeline supports optional cropping via env vars.
    env["TRAIN_START_DATE"] = cfg.start_date
    env["TRAIN_END_DATE"] = cfg.end_date
    subprocess.run(cmd, check=True, cwd=str(Path.cwd()), env=env)

    results = list(out_root.rglob("results.json"))
    if len(results) != 1:
        raise RuntimeError(
            f"Expected 1 results.json under {out_root}, found {len(results)}"
        )
    return results[0]


def run_seed_sweep_for_strategy(
    *,
    strategy_dir: Path,
    cfg: SearchConfig,
    run_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (all_rows, summary_by_strategy).
    """
    rows = []
    for seed in cfg.seeds:
        seed_out = cfg.output_dir / "runs" / run_id / f"seed_{seed}"
        results_json = _run_one_seed(
            strategy_dir=strategy_dir, cfg=cfg, seed=seed, out_root=seed_out
        )
        payload = json.loads(results_json.read_text(encoding="utf-8"))
        row = {
            "strategy": payload.get("strategy"),
            "task": payload.get("task_type"),
            "train": payload.get("n_train_samples"),
            "CV": payload.get("avg_cv_metric"),
            "corr": (payload.get("evaluation") or {}).get("test_correlation"),
            "return%": (payload.get("backtest") or {}).get("total_return_pct"),
            "Sharpe": (payload.get("backtest") or {}).get("sharpe"),
            "DD%": (payload.get("backtest") or {}).get("max_drawdown_pct"),
            "trades": (payload.get("backtest") or {}).get("total_trades"),
            "seed": int(seed),
            "run_id": run_id,
        }
        rows.append(pd.DataFrame([row]))

    all_df = pd.concat(rows, ignore_index=True)
    metric_cols = [
        c
        for c in ["train", "CV", "corr", "return%", "Sharpe", "DD%", "trades"]
        if c in all_df.columns
    ]
    gb = all_df.groupby(["strategy", "task"], dropna=False)
    summary = gb[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return all_df, summary


def greedy_forward_search(
    *,
    cfg: SearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    min_trades: int,
) -> dict:
    """
    Greedy forward selection over feature groups.

    objective: one of ["Sharpe_mean", "return%_mean", "corr_mean", "CV_mean"] (must exist in summary).
    """
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)

    selected: List[str] = []
    remaining = list(groups.keys())
    history: List[dict] = []
    # For transparency/debuggability: store all evaluated candidates per step (not just the winner).
    candidates_history: List[dict] = []

    current_features = list(base_features)
    best_score = None
    baseline_score = None
    eps_improve = 1e-9
    stop_reason = "unknown"
    rejected_groups: List[str] = []
    baseline_summary = None

    # Always evaluate baseline (no added groups) so the search can legitimately return "select nothing".
    tmp_root = cfg.output_dir / "tmp_strategies"
    _ensure_dir(tmp_root)
    baseline_suffix = "baseline"
    baseline_dir = _make_temp_strategy(
        base_dir=cfg.base_strategy_dir,
        tmp_root=tmp_root,
        name_suffix=baseline_suffix,
        requested_features=current_features,
    )
    _, baseline_summ = run_seed_sweep_for_strategy(
        strategy_dir=baseline_dir,
        cfg=cfg,
        run_id=baseline_suffix,
    )
    if baseline_summ.empty:
        raise RuntimeError("Baseline seed sweep returned empty summary")
    baseline_row = baseline_summ.iloc[0].to_dict()
    if objective not in baseline_row:
        raise ValueError(
            f"Objective '{objective}' not found in baseline summary columns: {list(baseline_row.keys())}"
        )
    baseline_score = float(baseline_row[objective])
    best_score = float(baseline_score)
    baseline_summary = baseline_row

    for step in range(int(max_steps)):
        best_candidate = None
        best_candidate_score = None
        best_candidate_summary = None
        step_candidates: List[dict] = []

        for g in remaining:
            feats = current_features + groups[g]
            suffix = f"step{step+1}_add_{g}"
            strat_dir = _make_temp_strategy(
                base_dir=cfg.base_strategy_dir,
                tmp_root=tmp_root,
                name_suffix=suffix,
                requested_features=feats,
            )

            run_id = suffix
            _, summary = run_seed_sweep_for_strategy(
                strategy_dir=strat_dir, cfg=cfg, run_id=run_id
            )
            if summary.empty:
                step_candidates.append(
                    {
                        "step": step + 1,
                        "candidate_group": g,
                        "score": None,
                        "valid": False,
                        "reject_reason": "empty_summary",
                        "summary": {},
                    }
                )
                continue
            row = summary.iloc[0].to_dict()

            # trade sanity
            trades_col = "trades_mean"
            trades_ok = True
            if trades_col in row:
                trades_ok = float(row[trades_col]) >= float(min_trades)
            if not trades_ok:
                step_candidates.append(
                    {
                        "step": step + 1,
                        "candidate_group": g,
                        "score": float(row[objective]) if objective in row else None,
                        "valid": False,
                        "reject_reason": "min_trades",
                        "summary": row,
                    }
                )
                continue

            if objective not in row:
                raise ValueError(
                    f"Objective '{objective}' not found in summary columns: {list(row.keys())}"
                )

            score = float(row[objective])
            step_candidates.append(
                {
                    "step": step + 1,
                    "candidate_group": g,
                    "score": score,
                    "valid": True,
                    "reject_reason": None,
                    "summary": row,
                }
            )
            if best_candidate_score is None or score > best_candidate_score:
                best_candidate_score = score
                best_candidate = g
                best_candidate_summary = row

        if best_candidate is None:
            stop_reason = "no_valid_candidates"
            break

        # Record all candidate scores for this step (useful when search space is big / noisy).
        candidates_history.append(
            {
                "step": step + 1,
                "current_selected": list(selected),
                "candidates": step_candidates,
            }
        )

        # Default-safe behavior: stop if we can't improve the objective.
        # Note: this may miss "synergy" that requires temporary deterioration; if needed, upgrade to beam/SFFS.
        if best_score is not None and best_candidate_score is not None:
            if float(best_candidate_score) <= float(best_score) + eps_improve:
                stop_reason = "no_improvement"
                # At this point, none of the remaining groups can improve the objective.
                rejected_groups = list(remaining)
                break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_features = current_features + groups[best_candidate]
        best_score = best_candidate_score
        history.append(
            {
                "step": step + 1,
                "added_group": best_candidate,
                "objective": objective,
                "score": best_score,
                "summary": best_candidate_summary,
            }
        )

    if stop_reason == "unknown":
        stop_reason = (
            "max_steps_reached" if len(selected) >= int(max_steps) else "completed"
        )
        if stop_reason == "max_steps_reached":
            rejected_groups = list(remaining)

    return {
        "base_strategy": cfg.base_strategy_dir.name,
        "base_features": base_features,
        "baseline": {"score": baseline_score, "summary": baseline_summary},
        "selected_groups": selected,
        "final_features": current_features,
        "history": history,
        "candidates_history": candidates_history,
        "stop_reason": stop_reason,
        "rejected_groups": rejected_groups,
        "objective": objective,
        "min_trades": min_trades,
        "seeds": cfg.seeds,
    }


def _default_groups() -> Dict[str, List[str]]:
    """
    Conservative group list (keeps "feature explosion" manageable).
    All keys are group names; values are requested_features nodes.
    """
    return {
        # Cheap, non-tick
        "kline_core": [
            "macd_f",
            "rsi_f",
            "sma_200_f",
            "atr_f",
            "trend_r2_20_f",
            "bb_width_f",
            "wick_ratios_f",
        ],
        "sr_structure_min": [
            "poc_hal_features_close_f",
            "sqs_hal_high_f",
            "sqs_hal_low_f",
            "sr_strength_max_close_f",
        ],
        "volume_profile": ["volume_profile_volatility_features_f"],
        "volume_profile_scene": ["volume_profile_scene_semantic_scores_f"],
        "wpt_energy": ["wpt_volume_energy_f"],
        "wpt_scene": ["wpt_scene_semantic_scores_f"],
        "liquidity_void": ["liquidity_void_f"],
        "liquidity_void_scene": ["liquidity_void_scene_semantic_scores_f"],
        "compression": ["compression_score_f", "compression_energy_f"],
        "wick_scene": ["wick_scene_semantic_scores_f"],
        # Tick-heavy
        "footprint_basic": ["footprint_basic_f"],
        "fp_scene": ["fp_imbalance_scene_semantic_scores_f"],
        "vpin_block": ["vpin_features_f"],
        "vpin_scene": ["vpin_scene_semantic_scores_f"],
        "trade_cluster_semantic": ["trade_cluster_semantic_scores_f"],
        "trade_cluster_scene": ["trade_cluster_scene_semantic_scores_f"],
    }


def _normalize_groups(obj: object) -> Dict[str, List[str]]:
    """
    Normalize a groups mapping loaded from YAML/JSON.
    Supports either:
      - {group_name: [node1, node2, ...]}
      - {"groups": {group_name: [node1, node2, ...]}}
    """
    if obj is None:
        return {}
    if isinstance(obj, dict) and "groups" in obj and isinstance(obj["groups"], dict):
        obj = obj["groups"]
    if not isinstance(obj, dict):
        raise ValueError(
            "groups must be a mapping: {group_name: [requested_feature_nodes...]}"
        )
    out: Dict[str, List[str]] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if v is None:
            out[k] = []
            continue
        if not isinstance(v, list):
            raise ValueError(f"group '{k}' must be a list, got: {type(v)}")
        cleaned = [str(x).strip() for x in v if str(x).strip()]
        out[str(k).strip()] = cleaned
    return out


def _apply_feature_blacklist(
    *,
    base_features: List[str],
    groups: Dict[str, List[str]],
    blacklist: List[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Remove blacklisted requested_feature nodes from base_features and groups."""
    bl = {b.strip() for b in blacklist if isinstance(b, str) and b.strip()}
    if not bl:
        return base_features, groups
    base_out = [f for f in base_features if f not in bl]
    groups_out: Dict[str, List[str]] = {}
    for g, feats in groups.items():
        groups_out[g] = [f for f in (feats or []) if f not in bl]
    return base_out, groups_out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-strategy-config",
        required=True,
        help="Base strategy config dir (single strategy)",
    )
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seeds", default="1,2,3", help="Comma-separated seeds")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-docker", action="store_true", default=True)
    p.add_argument("--objective", default="Sharpe_mean")
    p.add_argument("--min-trades", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument(
        "--groups-json", default=None, help="Optional groups override (JSON file path)"
    )
    p.add_argument(
        "--groups-yaml", default=None, help="Optional groups override (YAML file path)"
    )
    p.add_argument(
        "--pool-b-yaml",
        default=None,
        help=(
            "Optional Pool B YAML exported by factor-eval (features_pool_b.yaml). "
            "If provided, the tool will auto-generate extra singleton groups for any "
            "feature_pipeline.requested_features not already present in groups."
        ),
    )
    p.add_argument(
        "--base-features-yaml",
        default=None,
        help="Optional base features list YAML path",
    )
    p.add_argument(
        "--writeback-yaml",
        default=None,
        help="Optional output path to write a features_suggested.yaml (requested_features=final_features) with provenance metadata.",
    )
    p.add_argument(
        "--invert-candidates-yaml",
        default=None,
        help=(
            "Optional YAML path providing invert candidates. Accepts either a YAML list, or a full "
            "features config containing feature_pipeline.invert_features. On writeback, the tool "
            "will set invert_features = invert_candidates ∩ final_requested_features."
        ),
    )
    p.add_argument(
        "--feature-blacklist",
        default="",
        help="Comma-separated requested_feature nodes to exclude from BOTH base and candidate groups",
    )
    return p.parse_args()


def _load_groups_with_source(
    *,
    strategy_dir_name: str,
    groups_json: str | None,
    groups_yaml: str | None,
) -> tuple[Dict[str, List[str]], str, bool]:
    """
    Resolve candidate groups + provenance.

    Priority:
      1) --groups-json
      2) --groups-yaml
      3) config/feature_groups_<strategy_dir_name>_semantic.yaml (auto, if exists)
      4) config/feature_groups.yaml (auto, if exists)
      5) _default_groups() (code fallback)
    """
    if groups_json:
        p = Path(groups_json)
        return (
            _normalize_groups(json.loads(p.read_text(encoding="utf-8"))),
            f"groups_json:{groups_json}",
            False,
        )
    if groups_yaml:
        p = Path(groups_yaml)
        return (
            _normalize_groups(yaml.safe_load(p.read_text(encoding="utf-8"))),
            f"groups_yaml:{groups_yaml}",
            False,
        )

    # Auto strategy-specific convention first: config/feature_groups_<strategy>_semantic.yaml
    strategy_yaml = Path("config") / f"feature_groups_{strategy_dir_name}_semantic.yaml"
    if strategy_yaml.exists():
        return (
            _normalize_groups(
                yaml.safe_load(strategy_yaml.read_text(encoding="utf-8"))
            ),
            f"groups_yaml:auto:{strategy_yaml.as_posix()}",
            True,
        )

    # Auto global convention: config/feature_groups.yaml
    auto_yaml = Path("config") / "feature_groups.yaml"
    if auto_yaml.exists():
        return (
            _normalize_groups(yaml.safe_load(auto_yaml.read_text(encoding="utf-8"))),
            f"groups_yaml:auto:{auto_yaml.as_posix()}",
            True,
        )

    return _default_groups(), "default_groups", False


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_strategy_config)
    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    cfg = SearchConfig(
        base_strategy_dir=base_dir,
        timeframe=str(args.timeframe),
        symbol=str(args.symbol),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        test_size=float(args.test_size),
        seeds=seeds,
        output_dir=out_dir,
        deterministic=bool(args.deterministic),
        no_docker=bool(args.no_docker),
    )

    groups, resolved_groups_source, groups_yaml_auto = _load_groups_with_source(
        strategy_dir_name=base_dir.name,
        groups_json=args.groups_json,
        groups_yaml=args.groups_yaml,
    )

    # Convention-by-default:
    # If user didn't pass pool-b-yaml / invert-candidates-yaml, try the conventional location
    # derived from the base strategy directory name:
    #   results/pools/<strategy_dir_name>/pool_b/features_pool_b.yaml
    # (We intentionally do NOT hard fail if absent; the tool can still run without Pool B.)
    conventional_pool_b = (
        Path("results") / "pools" / base_dir.name / "pool_b" / "features_pool_b.yaml"
    )
    if not args.pool_b_yaml and conventional_pool_b.exists():
        args.pool_b_yaml = str(conventional_pool_b)
    if not args.invert_candidates_yaml and conventional_pool_b.exists():
        args.invert_candidates_yaml = str(conventional_pool_b)

    # Optional: merge Pool B (factor-eval) candidates as singleton groups.
    # This is how we make a semantic groups file compatible with a dynamic factor pool.
    if args.pool_b_yaml:
        pool_obj = (
            yaml.safe_load(Path(args.pool_b_yaml).read_text(encoding="utf-8")) or {}
        )
        pool_fp = (
            pool_obj.get("feature_pipeline") if isinstance(pool_obj, dict) else None
        )
        pool_req = (
            pool_fp.get("requested_features") if isinstance(pool_fp, dict) else None
        )
        pool_req = pool_req if isinstance(pool_req, list) else []

        used_nodes = set()
        for feats in (groups or {}).values():
            for f in feats or []:
                used_nodes.add(str(f))

        # Add singleton groups for any pool feature not already present in groups.
        for f in pool_req:
            f = str(f).strip()
            if not f:
                continue
            if f in used_nodes:
                continue
            key = f"poolb__{f}"
            # Ensure unique group name even if collisions occur
            if key in groups:
                i = 2
                while f"{key}__{i}" in groups:
                    i += 1
                key = f"{key}__{i}"
            groups[key] = [f]

    if args.base_features_yaml:
        base_features = (
            yaml.safe_load(Path(args.base_features_yaml).read_text(encoding="utf-8"))
            or []
        )
        if not isinstance(base_features, list):
            raise ValueError("base-features-yaml must be a YAML list")
    else:
        # Default base = base strategy's current requested_features.
        # This matches the mental model: "start from what I'm currently using, then add candidate groups".
        base_cfg = _load_yaml(base_dir / "features.yaml")
        fp = base_cfg.get("feature_pipeline") if isinstance(base_cfg, dict) else None
        req = fp.get("requested_features") if isinstance(fp, dict) else None
        base_features = req if isinstance(req, list) else []

    blacklist = [s.strip() for s in str(args.feature_blacklist).split(",") if s.strip()]
    base_features, groups = _apply_feature_blacklist(
        base_features=list(base_features),
        groups=dict(groups),
        blacklist=blacklist,
    )

    invert_candidates = None
    if args.invert_candidates_yaml:
        invert_candidates = _load_invert_candidates(Path(args.invert_candidates_yaml))

    t0 = time.time()
    result = greedy_forward_search(
        cfg=cfg,
        base_features=base_features,
        groups=groups,
        max_steps=int(args.max_steps),
        objective=str(args.objective),
        min_trades=int(args.min_trades),
    )
    result["elapsed_sec"] = round(time.time() - t0, 2)

    (out_dir / "feature_group_search_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Write a compact CSV of history (one row per step)
    hist_rows = []
    for h in result.get("history", []):
        row = {"step": h["step"], "added_group": h["added_group"], "score": h["score"]}
        summ = h.get("summary") or {}
        for k in [
            "Sharpe_mean",
            "return%_mean",
            "DD%_mean",
            "trades_mean",
            "corr_mean",
            "CV_mean",
        ]:
            if k in summ:
                row[k] = summ[k]
        hist_rows.append(row)
    pd.DataFrame(hist_rows).to_csv(
        out_dir / "feature_group_search_history.csv", index=False
    )

    # Write candidate table (one row per evaluated candidate group per step)
    cand_rows = []
    for step_obj in result.get("candidates_history", []) or []:
        step_n = step_obj.get("step")
        for c in step_obj.get("candidates", []) or []:
            summ = c.get("summary") or {}
            row = {
                "step": step_n,
                "candidate_group": c.get("candidate_group"),
                "score": c.get("score"),
            }
            for k in [
                "Sharpe_mean",
                "return%_mean",
                "DD%_mean",
                "trades_mean",
                "corr_mean",
                "CV_mean",
            ]:
                if k in summ:
                    row[k] = summ[k]
            cand_rows.append(row)
    if cand_rows:
        pd.DataFrame(cand_rows).to_csv(
            out_dir / "feature_group_search_candidates.csv", index=False
        )

    # Optional YAML writeback (features_suggested.yaml)
    if args.writeback_yaml:
        meta = {
            "base_strategy_dir": str(cfg.base_strategy_dir),
            "objective": result.get("objective"),
            "min_trades": result.get("min_trades"),
            "seeds": result.get("seeds"),
            "selected_groups": result.get("selected_groups", []),
            "stop_reason": result.get("stop_reason"),
            "base_features": result.get("base_features", []),
            "final_features": result.get("final_features", []),
            "groups_source": resolved_groups_source,
            "groups_yaml_auto": groups_yaml_auto,
            "pool_b_yaml": str(args.pool_b_yaml) if args.pool_b_yaml else None,
            "pool_b_merged_singletons": bool(args.pool_b_yaml),
            "feature_blacklist": blacklist,
        }
        writeback_info = _writeback_features_yaml(
            base_strategy_dir=cfg.base_strategy_dir,
            out_path=Path(args.writeback_yaml),
            requested_features=list(result.get("final_features") or []),
            meta=meta,
            invert_candidates=invert_candidates,
        )
        result["writeback"] = writeback_info
        (out_dir / "feature_group_search_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Minimal HTML
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Feature Group Search</title>")
    html.append(
        "<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px} th{background:#f5f5f5}</style>"
    )
    html.append("</head><body>")
    html.append("<h2>Feature Group Search (Greedy Forward Selection)</h2>")
    html.append(
        f"<p><b>Base</b>: {cfg.base_strategy_dir} &nbsp; <b>Symbol</b>: {cfg.symbol} &nbsp; <b>Timeframe</b>: {cfg.timeframe}<br/>"
    )
    html.append(
        f"<b>Window</b>: {cfg.start_date} → {cfg.end_date} &nbsp; <b>Test-size</b>: {cfg.test_size} &nbsp; <b>Seeds</b>: {', '.join(map(str,cfg.seeds))}<br/>"
    )
    html.append(
        f"<b>Objective</b>: {result.get('objective')} &nbsp; <b>Min trades</b>: {result.get('min_trades')}</p>"
    )

    # Artifact quick links (same directory)
    html.append("<h3>Artifacts</h3>")
    html.append(
        "<ul>"
        "<li><b>result</b>: <a href='feature_group_search_result.json'><code>feature_group_search_result.json</code></a></li>"
        "<li><b>history</b>: <a href='feature_group_search_history.csv'><code>feature_group_search_history.csv</code></a></li>"
        "<li><b>candidates</b>: <a href='feature_group_search_candidates.csv'><code>feature_group_search_candidates.csv</code></a></li>"
        "<li><b>report</b>: <a href='feature_group_search_report.html'><code>feature_group_search_report.html</code></a></li>"
        "<li><b>why (drilldown)</b>: <a href='feature_group_search_why.html'><code>feature_group_search_why.html</code></a></li>"
        "</ul>"
    )

    # Baseline + stop reason (make it obvious)
    html.append("<h3>Baseline &amp; stop</h3>")
    baseline = result.get("baseline") or {}
    baseline_score = baseline.get("score")
    stop_reason = result.get("stop_reason")
    rejected = result.get("rejected_groups") or []
    html.append(
        "<p>"
        f"<b>baseline_score</b>: {baseline_score} &nbsp; "
        f"<b>stop_reason</b>: {stop_reason} &nbsp; "
        f"<b>rejected_groups</b>: {len(rejected)}"
        "</p>"
    )
    baseline_summary = baseline.get("summary")
    if isinstance(baseline_summary, dict) and baseline_summary:
        keep_keys = [
            "Sharpe_mean",
            "return%_mean",
            "DD%_mean",
            "trades_mean",
            "CV_mean",
            "corr_mean",
        ]
        row = {k: baseline_summary.get(k) for k in keep_keys if k in baseline_summary}
        row = {"baseline_" + k: v for k, v in row.items()}
        if row:
            html.append(pd.DataFrame([row]).to_html(index=False, escape=False))
    if rejected:
        html.append(
            "<details><summary><b>Rejected groups</b> (click to expand)</summary>"
        )
        html.append("<pre>" + "\n".join([str(x) for x in rejected]) + "</pre>")
        html.append("</details>")
    html.append("<h3>Selected groups</h3>")
    html.append("<pre>" + "\n".join(result.get("selected_groups", [])) + "</pre>")
    html.append("<h3>History</h3>")
    if hist_rows:
        html.append(pd.DataFrame(hist_rows).to_html(index=False, escape=False))
    if cand_rows:
        html.append("<h3>All candidate scores (per step)</h3>")
        html.append(
            "<p>Tip: use the <b>why (drilldown)</b> HTML for step-by-step explanations.</p>"
        )
        html.append(pd.DataFrame(cand_rows).to_html(index=False, escape=False))
    html.append("<h3>Final requested_features</h3>")
    html.append("<pre>" + "\n".join(result.get("final_features", [])) + "</pre>")
    if result.get("writeback"):
        html.append("<h3>Writeback</h3>")
        html.append(
            "<pre>"
            + json.dumps(result.get("writeback"), ensure_ascii=False, indent=2)
            + "</pre>"
        )
    html.append("</body></html>")
    (out_dir / "feature_group_search_report.html").write_text(
        "\n".join(html), encoding="utf-8"
    )

    # Candidate drilldown HTML ("why selected / why rejected")
    why = []
    why.append(
        "<html><head><meta charset='utf-8'><title>Feature Group Search - Why</title>"
    )
    why.append(
        "<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px} th{background:#f5f5f5} code{background:#f5f5f5;padding:1px 3px;border-radius:3px}</style>"
    )
    why.append("</head><body>")
    why.append("<h2>Why selected vs rejected (per step)</h2>")
    why.append(
        "<p><a href='feature_group_search_report.html'>← back to main report</a></p>"
    )

    baseline = result.get("baseline") or {}
    why.append("<h3>Baseline</h3>")
    why.append(f"<p><b>baseline_score</b>: {baseline.get('score')}</p>")
    bsum = baseline.get("summary") or {}
    if isinstance(bsum, dict) and bsum:
        keep = [
            "Sharpe_mean",
            "return%_mean",
            "DD%_mean",
            "trades_mean",
            "CV_mean",
            "corr_mean",
        ]
        why.append(
            pd.DataFrame([{k: bsum.get(k) for k in keep if k in bsum}]).to_html(
                index=False, escape=False
            )
        )

    hist_by_step = {h.get("step"): h for h in (result.get("history") or [])}
    cand_hist = result.get("candidates_history") or []
    for step_obj in cand_hist:
        step_n = step_obj.get("step")
        chosen = (hist_by_step.get(step_n) or {}).get("added_group")
        why.append(f"<h3 id='step_{step_n}'>Step {step_n}</h3>")
        why.append(
            f"<p><b>chosen</b>: {chosen or '(none)'} &nbsp; <b>stop_reason</b>: {result.get('stop_reason') if chosen is None else ''}</p>"
        )
        cands = step_obj.get("candidates") or []
        rows = []
        for c in cands:
            summ = c.get("summary") or {}
            rows.append(
                {
                    "step": step_n,
                    "candidate_group": c.get("candidate_group"),
                    "valid": c.get("valid"),
                    "reject_reason": c.get("reject_reason"),
                    "score": c.get("score"),
                    "trades_mean": summ.get("trades_mean"),
                    "Sharpe_mean": summ.get("Sharpe_mean"),
                    "return%_mean": summ.get("return%_mean"),
                    "DD%_mean": summ.get("DD%_mean"),
                    "corr_mean": summ.get("corr_mean"),
                    "CV_mean": summ.get("CV_mean"),
                    "picked": c.get("candidate_group") == chosen,
                }
            )
        if rows:
            df = pd.DataFrame(rows)
            # Sort: picked first, then valid desc by score, then invalid last
            if "score" in df.columns:
                df["_score_sort"] = pd.to_numeric(df["score"], errors="coerce")
                df = df.sort_values(
                    by=["picked", "valid", "_score_sort"],
                    ascending=[False, False, False],
                ).drop(columns=["_score_sort"])
            why.append(df.to_html(index=False, escape=False))

    why.append("</body></html>")
    (out_dir / "feature_group_search_why.html").write_text(
        "\n".join(why), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
