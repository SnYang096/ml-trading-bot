"""
nnmultihead feature-group-search (primitives objective)

This mirrors tree-side `feature_group_search.py` upgrades (greedy/halving/beam/sffs/pipeline),
but uses nnmultihead training metrics as the objective instead of Sharpe/backtest.

Key idea:
- Each candidate group corresponds to adding some feature compute functions (xxx_f) to the nn config.
- For each evaluation, we create a temp nn config dir and run a short nnmultihead train,
  then read `metrics.json` and score the requested objective (e.g. dir_auc, roll_icir__dir).

Tests:
- Unit tests stub out the evaluator so we don't train models in CI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class NNFeatureSearchConfig:
    base_config_dir: Path
    symbols: str
    timeframe: str
    start_date: str
    end_date: str
    features_store_root: str
    features_store_layer: str
    output_dir: Path
    no_docker: bool = True

    # Training budget knobs (used by algorithms as "budget dimension")
    epochs: int = 10
    batch_size: int = 512
    lr: float = 2e-4
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.1
    device: Optional[str] = None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _flatten_requested_features(req: Any) -> List[str]:
    if isinstance(req, list):
        return [str(x) for x in req if str(x).strip()]
    if isinstance(req, dict):
        out: List[str] = []
        out.extend([str(x) for x in (req.get("required") or []) if str(x).strip()])
        blocks = req.get("optional_blocks") or {}
        if isinstance(blocks, dict):
            for _, v in blocks.items():
                if isinstance(v, list):
                    out.extend([str(x) for x in v if str(x).strip()])
        return out
    return []


def _load_features_list_yaml(path: Path) -> List[str]:
    """
    Load a YAML list of feature functions.
    Accepts either:
    - a plain YAML list: ['atr_f', 'rsi_f', ...]
    - a dict with feature_pipeline.requested_features (list or structured)
    """
    obj = _load_yaml(path)
    if isinstance(obj, list):
        return [str(x) for x in obj if str(x).strip()]
    if isinstance(obj, dict):
        fp = obj.get("feature_pipeline") or {}
        req = fp.get("requested_features")
        feats = _flatten_requested_features(req)
        if feats:
            return feats
        # also accept {base_features: [...]}
        bf = obj.get("base_features")
        if isinstance(bf, list):
            return [str(x) for x in bf if str(x).strip()]
    return []


def _read_base_features_from_config(base_config_dir: Path) -> List[str]:
    feats = _load_yaml(base_config_dir / "features.yaml")
    fp = feats.get("feature_pipeline") or {}
    req = fp.get("requested_features")
    return _flatten_requested_features(req)


def _make_temp_nnmultihead_config(
    *,
    base_config_dir: Path,
    tmp_root: Path,
    name_suffix: str,
    requested_features: List[str],
) -> Path:
    """
    Create a temp nn config directory:
    - copies labels.yaml + model.yaml from base
    - writes features.yaml with requested_features and selector limiting feature_cols
    """
    out_dir = tmp_root / f"{base_config_dir.name}__{name_suffix}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(base_config_dir / "labels.yaml", out_dir / "labels.yaml")
    shutil.copy2(base_config_dir / "model.yaml", out_dir / "model.yaml")

    # Try to preserve missingness policy / optional blocks structure from base, but we enforce selector.
    base_feats = _load_yaml(base_config_dir / "features.yaml")
    fp = base_feats.get("feature_pipeline") or {}

    # Use structured format for readability in artifacts (all required in this temp config).
    req_struct = {"required": list(requested_features), "optional_blocks": {}}

    fp_out = dict(fp)
    fp_out["requested_features"] = req_struct
    fp_out["selector"] = {
        "module": "src.time_series_model.models.nn.feature_selector",
        "function": "select_columns_from_requested_features",
        "params": {
            "requested_features": list(requested_features),
            "feature_deps_path": "config/feature_dependencies.yaml",
            "drop_constant": True,
        },
    }
    # Keep existing missingness_policy if present
    if "missingness_policy" not in fp_out:
        fp_out["missingness_policy"] = {
            "append_block_mask": True,
            "block_dropout_p": 0.05,
        }

    out_features = {
        "description": f"temp nnmultihead config for feature search: {name_suffix}",
        "feature_pipeline": fp_out,
    }
    _dump_yaml(out_dir / "features.yaml", out_features)
    return out_dir


def _read_metrics_json(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"metrics.json not found in run_dir: {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _eval_train_score_default(
    *,
    cfg: NNFeatureSearchConfig,
    temp_config_dir: Path,
    objective: str,
    run_id: str,
    epochs: int,
) -> Tuple[float | None, bool, str | None, Dict[str, Any]]:
    """
    Default evaluator: run a short nnmultihead train and score objective from metrics.json.
    """
    out_root = cfg.output_dir / "tmp_runs"
    _ensure_dir(out_root)

    # Unique run output dir per evaluation
    run_out = out_root / run_id
    if run_out.exists():
        shutil.rmtree(run_out)
    run_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mlbot",
        "nnmultihead",
        "train",
        "--no-docker" if cfg.no_docker else "--docker",
        "--config",
        str(temp_config_dir),
        "--symbols",
        cfg.symbols,
        "--timeframe",
        cfg.timeframe,
        "--start-date",
        cfg.start_date,
        "--end-date",
        cfg.end_date,
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(cfg.batch_size)),
        "--lr",
        str(float(cfg.lr)),
        "--hidden",
        str(int(cfg.hidden)),
        "--depth",
        str(int(cfg.depth)),
        "--dropout",
        str(float(cfg.dropout)),
        "--feature-store-root",
        str(cfg.features_store_root),
        "--feature-store-layer",
        str(cfg.features_store_layer),
        "--output-dir",
        str(run_out),
    ]
    if cfg.device:
        cmd.extend(["--device", str(cfg.device)])

    # Run train
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return (
            None,
            False,
            "train_failed",
            {"stderr": e.stderr[-4000:], "stdout": e.stdout[-4000:]},
        )

    # Trainer writes run under output-dir/<run_name>
    # Find the single subdir containing metrics.json.
    run_dirs = [
        p for p in run_out.iterdir() if p.is_dir() and (p / "metrics.json").exists()
    ]
    if not run_dirs:
        return None, False, "missing_metrics", {}
    run_dir = run_dirs[0]
    metrics = _read_metrics_json(run_dir)
    if objective not in metrics:
        return (
            None,
            False,
            "objective_missing",
            {"metrics_keys": sorted(metrics.keys())},
        )
    try:
        score = float(metrics[objective])
    except Exception:
        return (
            None,
            False,
            "objective_not_float",
            {"objective": objective, "value": metrics.get(objective)},
        )
    return score, True, None, metrics


def greedy_forward_search(
    *,
    cfg: NNFeatureSearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    eps_improve: float = 1e-9,
    evaluator: Callable[
        ..., Tuple[float | None, bool, str | None, Dict[str, Any]]
    ] = _eval_train_score_default,
    budget_epochs: int | None = None,
) -> Dict[str, Any]:
    tmp_root = cfg.output_dir / "tmp_configs"
    _ensure_dir(tmp_root)

    # Baseline
    base_cfg_dir = _make_temp_nnmultihead_config(
        base_config_dir=cfg.base_config_dir,
        tmp_root=tmp_root,
        name_suffix="baseline",
        requested_features=list(base_features),
    )
    base_score, base_valid, base_reject, base_meta = evaluator(
        cfg=cfg,
        temp_config_dir=base_cfg_dir,
        objective=objective,
        run_id="baseline",
        epochs=int(budget_epochs or cfg.epochs),
    )
    if not base_valid or base_score is None:
        base_score = -999.0

    selected: List[str] = []
    remaining = list(groups.keys())
    current_features = list(base_features)
    history: List[dict] = []
    stop_reason = "completed"

    for step in range(int(max_steps)):
        best_g = None
        best_score = None
        best_meta: Dict[str, Any] = {}
        best_reject = None
        for g in remaining:
            feats = list(current_features) + (groups.get(g) or [])
            sig = "__".join(selected + [g])
            run_id = f"greedy_step{step+1}_add_{g}__sel_{sig}"
            tdir = _make_temp_nnmultihead_config(
                base_config_dir=cfg.base_config_dir,
                tmp_root=tmp_root,
                name_suffix=run_id,
                requested_features=feats,
            )
            score, valid, reject_reason, meta = evaluator(
                cfg=cfg,
                temp_config_dir=tdir,
                objective=objective,
                run_id=run_id,
                epochs=int(budget_epochs or cfg.epochs),
            )
            if valid and score is not None:
                if (best_score is None) or (float(score) > float(best_score)):
                    best_score = float(score)
                    best_g = g
                    best_meta = meta
                    best_reject = reject_reason
        if best_g is None or best_score is None:
            stop_reason = "no_valid_candidates"
            break
        if best_score <= float(base_score) + float(eps_improve):
            stop_reason = "no_strict_improve"
            break
        # accept
        selected.append(best_g)
        current_features = list(current_features) + (groups.get(best_g) or [])
        remaining = [g for g in remaining if g != best_g]
        base_score = best_score
        history.append(
            {
                "step": step + 1,
                "added_group": best_g,
                "score": best_score,
                "summary": best_meta,
            }
        )

        if not remaining:
            stop_reason = "exhausted_candidates"
            break

    return {
        "search_algo": "greedy",
        "objective": objective,
        "baseline": {"score": base_score, "summary": base_meta, "reject": base_reject},
        "selected_groups": selected,
        "final_features": current_features,
        "history": history,
        "stop_reason": stop_reason,
    }


def successive_halving_search(
    *,
    cfg: NNFeatureSearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    stages: List[int],
    top_fraction: float,
    min_survivors: int,
    evaluator: Callable[
        ..., Tuple[float | None, bool, str | None, Dict[str, Any]]
    ] = _eval_train_score_default,
) -> Dict[str, Any]:
    """
    Successive Halving per greedy step.
    Budget dimension: epochs stages.
    """
    tmp_root = cfg.output_dir / "tmp_configs"
    _ensure_dir(tmp_root)

    current = list(base_features)
    selected: List[str] = []
    remaining = list(groups.keys())
    history: List[dict] = []

    stages = [int(x) for x in stages if int(x) > 0]
    if not stages:
        stages = [max(1, int(cfg.epochs))]
    stages = sorted(set(stages))
    if stages[-1] != int(cfg.epochs):
        stages.append(int(cfg.epochs))

    baseline_score = -999.0
    base_cfg_dir = _make_temp_nnmultihead_config(
        base_config_dir=cfg.base_config_dir,
        tmp_root=tmp_root,
        name_suffix="baseline",
        requested_features=current,
    )
    s0, v0, _, m0 = evaluator(
        cfg=cfg,
        temp_config_dir=base_cfg_dir,
        objective=objective,
        run_id="baseline",
        epochs=stages[-1],
    )
    if v0 and s0 is not None:
        baseline_score = float(s0)

    for step in range(int(max_steps)):
        if not remaining:
            break

        candidates = list(remaining)
        stage_tables: List[dict] = []
        last_scores: Dict[str, float] = {}

        for si, ep in enumerate(stages):
            scored: List[tuple[str, float]] = []
            rows: List[dict] = []
            for g in candidates:
                feats = list(current) + (groups.get(g) or [])
                run_id = f"halving_step{step+1}_add_{g}__e{ep}"
                tdir = _make_temp_nnmultihead_config(
                    base_config_dir=cfg.base_config_dir,
                    tmp_root=tmp_root,
                    name_suffix=run_id,
                    requested_features=feats,
                )
                score, valid, reject_reason, meta = evaluator(
                    cfg=cfg,
                    temp_config_dir=tdir,
                    objective=objective,
                    run_id=run_id,
                    epochs=int(ep),
                )
                rows.append(
                    {
                        "stage": si + 1,
                        "epochs": int(ep),
                        "candidate_group": g,
                        "score": score,
                        "valid": bool(valid),
                        "reject_reason": reject_reason,
                        "summary": meta,
                    }
                )
                if valid and score is not None:
                    scored.append((g, float(score)))
            stage_tables.append(
                {"stage": si + 1, "epochs": int(ep), "candidates": rows}
            )
            if not scored:
                candidates = []
                break
            scored.sort(key=lambda x: x[1], reverse=True)
            keep_n = max(int(min_survivors), int(len(scored) * float(top_fraction)))
            keep_n = max(1, min(keep_n, len(scored)))
            candidates = [g for g, _ in scored[:keep_n]]
            if ep == stages[-1]:
                last_scores = {g: s for g, s in scored}

        if not candidates:
            break

        # Pick the best from final stage among candidates
        best = max(candidates, key=lambda g: float(last_scores.get(g, -999.0)))
        best_score = float(last_scores.get(best, -999.0))
        if best_score <= float(baseline_score) + 1e-9:
            break

        selected.append(best)
        current = list(current) + (groups.get(best) or [])
        remaining = [g for g in remaining if g != best]
        baseline_score = best_score
        history.append(
            {
                "step": step + 1,
                "added_group": best,
                "score": best_score,
                "halving": stage_tables,
            }
        )

    return {
        "search_algo": "successive_halving",
        "objective": objective,
        "selected_groups": selected,
        "final_features": current,
        "history": history,
    }


def beam_search(
    *,
    cfg: NNFeatureSearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    beam_width: int,
    evaluator: Callable[
        ..., Tuple[float | None, bool, str | None, Dict[str, Any]]
    ] = _eval_train_score_default,
) -> Dict[str, Any]:
    tmp_root = cfg.output_dir / "tmp_configs"
    _ensure_dir(tmp_root)
    beam_width = max(1, int(beam_width))

    # baseline score
    base_cfg_dir = _make_temp_nnmultihead_config(
        base_config_dir=cfg.base_config_dir,
        tmp_root=tmp_root,
        name_suffix="baseline",
        requested_features=list(base_features),
    )
    base_score, base_valid, _, base_meta = evaluator(
        cfg=cfg,
        temp_config_dir=base_cfg_dir,
        objective=objective,
        run_id="baseline",
        epochs=int(cfg.epochs),
    )
    if not base_valid or base_score is None:
        base_score = -999.0

    # Beam items: (selected_groups, features, score, meta)
    beam: List[tuple[List[str], List[str], float, dict]] = [
        ([], list(base_features), float(base_score), dict(base_meta or {}))
    ]
    best_item = beam[0]
    history: List[dict] = []

    for step in range(int(max_steps)):
        expansions: List[tuple[List[str], List[str], float, dict]] = []
        step_candidates: List[dict] = []
        for sel, feats, _, _ in beam:
            for g in groups.keys():
                if g in sel:
                    continue
                sel2 = list(sel) + [g]
                feats2 = list(feats) + (groups.get(g) or [])
                sig = "__".join(sel2)
                run_id = f"beam_step{step+1}_sel_{sig}"
                tdir = _make_temp_nnmultihead_config(
                    base_config_dir=cfg.base_config_dir,
                    tmp_root=tmp_root,
                    name_suffix=run_id,
                    requested_features=feats2,
                )
                score, valid, reject_reason, meta = evaluator(
                    cfg=cfg,
                    temp_config_dir=tdir,
                    objective=objective,
                    run_id=run_id,
                    epochs=int(cfg.epochs),
                )
                step_candidates.append(
                    {
                        "step": step + 1,
                        "selected": sel2,
                        "score": score,
                        "valid": bool(valid),
                        "reject_reason": reject_reason,
                    }
                )
                if valid and score is not None:
                    expansions.append((sel2, feats2, float(score), meta))

        history.append({"step": step + 1, "candidates": step_candidates})
        if not expansions:
            break
        expansions.sort(key=lambda x: x[2], reverse=True)
        beam = expansions[:beam_width]
        if beam[0][2] > best_item[2] + 1e-9:
            best_item = beam[0]

    return {
        "search_algo": "beam",
        "objective": objective,
        "selected_groups": best_item[0],
        "final_features": best_item[1],
        "history": history,
    }


def sffs_search(
    *,
    cfg: NNFeatureSearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    max_backward_per_step: int,
    evaluator: Callable[
        ..., Tuple[float | None, bool, str | None, Dict[str, Any]]
    ] = _eval_train_score_default,
) -> Dict[str, Any]:
    tmp_root = cfg.output_dir / "tmp_configs"
    _ensure_dir(tmp_root)

    # baseline
    base_cfg_dir = _make_temp_nnmultihead_config(
        base_config_dir=cfg.base_config_dir,
        tmp_root=tmp_root,
        name_suffix="baseline",
        requested_features=list(base_features),
    )
    base_score, base_valid, _, base_meta = evaluator(
        cfg=cfg,
        temp_config_dir=base_cfg_dir,
        objective=objective,
        run_id="baseline",
        epochs=int(cfg.epochs),
    )
    if not base_valid or base_score is None:
        base_score = -999.0

    selected: List[str] = []
    current_feats = list(base_features)
    best_score = float(base_score)
    history: List[dict] = []

    for step in range(int(max_steps)):
        # Forward add best
        best_add = None
        best_add_score = None
        best_add_meta: Dict[str, Any] = {}
        for g in groups.keys():
            if g in selected:
                continue
            sel2 = list(selected) + [g]
            feats2 = list(current_feats) + (groups.get(g) or [])
            sig = "__".join(sel2)
            run_id = f"sffs_step{step+1}_fwd_sel_{sig}"
            tdir = _make_temp_nnmultihead_config(
                base_config_dir=cfg.base_config_dir,
                tmp_root=tmp_root,
                name_suffix=run_id,
                requested_features=feats2,
            )
            score, valid, _, meta = evaluator(
                cfg=cfg,
                temp_config_dir=tdir,
                objective=objective,
                run_id=run_id,
                epochs=int(cfg.epochs),
            )
            if valid and score is not None:
                if (best_add_score is None) or (float(score) > float(best_add_score)):
                    best_add_score = float(score)
                    best_add = g
                    best_add_meta = meta
        if best_add is None or best_add_score is None:
            break
        if best_add_score <= best_score + 1e-9:
            break
        selected.append(best_add)
        current_feats = list(current_feats) + (groups.get(best_add) or [])
        best_score = float(best_add_score)
        history.append(
            {
                "step": step + 1,
                "action": "add",
                "group": best_add,
                "score": best_score,
                "summary": best_add_meta,
            }
        )

        # Backward floating remove
        for _ in range(max(1, int(max_backward_per_step))):
            improved = False
            if len(selected) <= 1:
                break
            for rm in list(selected):
                sel2 = [g for g in selected if g != rm]
                feats2 = list(base_features)
                for gg in sel2:
                    feats2 = feats2 + (groups.get(gg) or [])
                sig = "__".join(sel2) if sel2 else "none"
                run_id = f"sffs_step{step+1}_bwd_sel_{sig}__rm_{rm}"
                tdir = _make_temp_nnmultihead_config(
                    base_config_dir=cfg.base_config_dir,
                    tmp_root=tmp_root,
                    name_suffix=run_id,
                    requested_features=feats2,
                )
                score, valid, _, meta = evaluator(
                    cfg=cfg,
                    temp_config_dir=tdir,
                    objective=objective,
                    run_id=run_id,
                    epochs=int(cfg.epochs),
                )
                if valid and score is not None and float(score) > best_score + 1e-9:
                    selected = sel2
                    current_feats = feats2
                    best_score = float(score)
                    history.append(
                        {
                            "step": step + 1,
                            "action": "remove",
                            "group": rm,
                            "score": best_score,
                            "summary": meta,
                        }
                    )
                    improved = True
                    break
            if not improved:
                break

    return {
        "search_algo": "sffs",
        "objective": objective,
        "selected_groups": selected,
        "final_features": current_feats,
        "history": history,
    }


def pipeline_sh_beam_sffs(
    *,
    cfg: NNFeatureSearchConfig,
    base_features: List[str],
    groups: Dict[str, List[str]],
    max_steps: int,
    objective: str,
    stages: List[int],
    top_fraction: float,
    min_survivors: int,
    target_survivors: int,
    beam_width: int,
    sffs_max_backward_steps: int,
    evaluator: Callable[
        ..., Tuple[float | None, bool, str | None, Dict[str, Any]]
    ] = _eval_train_score_default,
) -> Dict[str, Any]:
    # 1) halving prefilter: rank single-add candidates and keep top target_survivors
    tmp_root = cfg.output_dir / "tmp_configs"
    _ensure_dir(tmp_root)

    # Evaluate each group as a single add with successive halving and pick survivors.
    stages = sorted(set([int(x) for x in stages if int(x) > 0] + [int(cfg.epochs)]))
    survivors = list(groups.keys())
    final_scores: Dict[str, float] = {}
    for ep in stages:
        scored: List[tuple[str, float]] = []
        for g in survivors:
            feats = list(base_features) + (groups.get(g) or [])
            run_id = f"prefilter_add_{g}__e{ep}"
            tdir = _make_temp_nnmultihead_config(
                base_config_dir=cfg.base_config_dir,
                tmp_root=tmp_root,
                name_suffix=run_id,
                requested_features=feats,
            )
            score, valid, _, _ = evaluator(
                cfg=cfg,
                temp_config_dir=tdir,
                objective=objective,
                run_id=run_id,
                epochs=int(ep),
            )
            if valid and score is not None:
                scored.append((g, float(score)))
        if not scored:
            survivors = []
            break
        scored.sort(key=lambda x: x[1], reverse=True)
        keep_n = max(int(min_survivors), int(len(scored) * float(top_fraction)))
        keep_n = max(1, min(keep_n, len(scored)))
        survivors = [g for g, _ in scored[:keep_n]]
        if ep == stages[-1]:
            final_scores = {g: s for g, s in scored}

    survivors = survivors[: max(1, int(target_survivors))]
    groups2 = {g: groups[g] for g in survivors if g in groups}

    # 2) beam on survivors
    beam_res = beam_search(
        cfg=cfg,
        base_features=base_features,
        groups=groups2,
        max_steps=max_steps,
        objective=objective,
        beam_width=beam_width,
        evaluator=evaluator,
    )

    # 3) prune-only stage: starting from beam-selected set, try removing any single group
    # if removal improves the objective. This matches tree-side prune-only SFFS stage.
    def _feats_for(sel: List[str]) -> List[str]:
        feats = list(base_features)
        for g in sel:
            feats = feats + (groups2.get(g) or [])
        return feats

    sel = list(beam_res.get("selected_groups") or [])
    best_sel = list(sel)
    best_feats = _feats_for(best_sel)
    best_score: float = -999.0
    best_meta: Dict[str, Any] = {}

    if best_sel:
        run_id0 = f"prune_init__{'__'.join(best_sel)}"
        tdir0 = _make_temp_nnmultihead_config(
            base_config_dir=cfg.base_config_dir,
            tmp_root=tmp_root,
            name_suffix=run_id0,
            requested_features=best_feats,
        )
        s0, v0, _, m0 = evaluator(
            cfg=cfg,
            temp_config_dir=tdir0,
            objective=objective,
            run_id=run_id0,
            epochs=int(cfg.epochs),
        )
        if v0 and s0 is not None:
            best_score = float(s0)
            best_meta = dict(m0 or {})

    prune_history: List[dict] = []
    for _ in range(max(1, int(sffs_max_backward_steps))):
        if len(best_sel) <= 1:
            break
        improved = False
        for rm in list(best_sel):
            keep = [g for g in best_sel if g != rm]
            feats = _feats_for(keep)
            sig = "__".join(keep) if keep else "none"
            run_id = f"prune_try__keep_{sig}__rm_{rm}"
            tdir = _make_temp_nnmultihead_config(
                base_config_dir=cfg.base_config_dir,
                tmp_root=tmp_root,
                name_suffix=run_id,
                requested_features=feats,
            )
            s, v, _, m = evaluator(
                cfg=cfg,
                temp_config_dir=tdir,
                objective=objective,
                run_id=run_id,
                epochs=int(cfg.epochs),
            )
            if v and s is not None and float(s) > float(best_score) + 1e-9:
                best_score = float(s)
                best_sel = keep
                best_feats = feats
                best_meta = dict(m or {})
                prune_history.append(
                    {
                        "action": "remove",
                        "removed": rm,
                        "kept": keep,
                        "score": best_score,
                    }
                )
                improved = True
                break
        if not improved:
            break

    return {
        "search_algo": "pipeline_sh_beam_sffs",
        "objective": objective,
        "algo_params": {
            "stages": stages,
            "beam_width": int(beam_width),
            "sffs_max_backward_steps": int(sffs_max_backward_steps),
        },
        "prefilter": {"survivors": survivors, "final_scores": final_scores},
        "beam": {"selected_groups": beam_res.get("selected_groups")},
        "prune": {
            "selected_groups": best_sel,
            "score": best_score,
            "history": prune_history,
        },
        "selected_groups": best_sel or (beam_res.get("selected_groups") or []),
        "final_features": best_feats or (beam_res.get("final_features") or []),
        "history": (beam_res.get("history") or []) + prune_history,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True, help="Base nnmultihead config dir")
    p.add_argument(
        "--base-features-yaml",
        default=None,
        help="Optional base feature funcs YAML (Pool A). If omitted, will try <base-config>/features_base.yaml; otherwise starts from empty base.",
    )
    p.add_argument("--symbols", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--features-store-root", default="feature_store")
    p.add_argument("--features-store-layer", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--objective", default="dir_auc", help="Metric key in metrics.json")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument(
        "--search-algo",
        default="greedy",
        choices=["greedy", "halving", "beam", "sffs", "pipeline"],
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default=None)
    p.add_argument("--no-docker", action="store_true", default=True)

    # Candidates / groups
    p.add_argument(
        "--pool-b-yaml",
        required=True,
        help="PoolB YAML with feature_pipeline.requested_features",
    )

    # Halving params (epochs stages)
    p.add_argument("--halving-stages", default="3,6,10")
    p.add_argument("--halving-top-fraction", type=float, default=0.25)
    p.add_argument("--halving-min-survivors", type=int, default=5)

    # Beam params
    p.add_argument("--beam-width", type=int, default=3)

    # SFFS params
    p.add_argument("--sffs-max-backward-per-step", type=int, default=2)

    # Pipeline params
    p.add_argument("--pipeline-survivors", type=int, default=30)
    return p.parse_args()


def _parse_int_list(csv: str) -> List[int]:
    out: List[int] = []
    for part in str(csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    _ensure_dir(out_dir)

    base_config_dir = Path(args.base_config).resolve()
    if not base_config_dir.exists():
        raise FileNotFoundError(f"base-config not found: {base_config_dir}")

    cfg = NNFeatureSearchConfig(
        base_config_dir=base_config_dir,
        symbols=str(args.symbols),
        timeframe=str(args.timeframe),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        features_store_root=str(args.features_store_root),
        features_store_layer=str(args.features_store_layer),
        output_dir=out_dir,
        no_docker=bool(args.no_docker),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden=int(args.hidden),
        depth=int(args.depth),
        dropout=float(args.dropout),
        device=str(args.device) if args.device else None,
    )

    # Base features (Pool A): must be minimal and non-optimizable.
    # DO NOT default to reading base-config/features.yaml because that often contains the full feature set.
    base_features: List[str] = []
    if args.base_features_yaml:
        base_features = _load_features_list_yaml(
            Path(args.base_features_yaml).resolve()
        )
    else:
        auto = base_config_dir / "features_base.yaml"
        if auto.exists():
            base_features = _load_features_list_yaml(auto)
    # Fallback: if user really passed a minimal base-config, keep compatibility.
    if not base_features:
        base_features = []

    pool = _load_yaml(Path(args.pool_b_yaml))
    requested = _flatten_requested_features(
        (pool.get("feature_pipeline") or {}).get("requested_features")
    )
    if not requested:
        raise ValueError("pool-b-yaml contains no feature_pipeline.requested_features")
    # Remove any base features from candidates (they are always included).
    base_set = set(base_features)
    requested = [f for f in requested if f not in base_set]
    groups = {f"poolb__{f}": [f] for f in requested}

    algo = str(args.search_algo)
    objective = str(args.objective)

    if algo == "greedy":
        result = greedy_forward_search(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=objective,
        )
    elif algo == "halving":
        stages = _parse_int_list(str(args.halving_stages))
        result = successive_halving_search(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=objective,
            stages=stages,
            top_fraction=float(args.halving_top_fraction),
            min_survivors=int(args.halving_min_survivors),
        )
    elif algo == "beam":
        result = beam_search(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=objective,
            beam_width=int(args.beam_width),
        )
    elif algo == "sffs":
        result = sffs_search(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=objective,
            max_backward_per_step=int(args.sffs_max_backward_per_step),
        )
    elif algo == "pipeline":
        stages = _parse_int_list(str(args.halving_stages))
        result = pipeline_sh_beam_sffs(
            cfg=cfg,
            base_features=base_features,
            groups=groups,
            max_steps=int(args.max_steps),
            objective=objective,
            stages=stages,
            top_fraction=float(args.halving_top_fraction),
            min_survivors=int(args.halving_min_survivors),
            target_survivors=int(args.pipeline_survivors),
            beam_width=int(args.beam_width),
            sffs_max_backward_steps=int(args.sffs_max_backward_per_step),
        )
    else:
        raise ValueError(f"Unknown search algo: {algo}")

    (out_dir / "nn_feature_group_search_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"✅ Wrote: {out_dir / 'nn_feature_group_search_result.json'}")


if __name__ == "__main__":
    main()
