#!/usr/bin/env python3
"""
One-shot nnmultihead search workflow:
1) Build candidates list that matches FeatureStore layer columns (fast, avoids missing-column waste)
2) nnmultihead factor-eval -> PoolB primitives YAML
3) nnmultihead feature-group-search (PoolA from <config>/features_base.yaml + PoolB + semantic groups)
4) Optionally train a "best_config" model

This is intentionally minimal and file-based (token efficient).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec


def _dump_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _first_symbol(symbols_csv: str) -> str:
    parts = [s.strip() for s in str(symbols_csv).split(",") if s.strip()]
    if not parts:
        raise ValueError("No symbols provided")
    return parts[0]


def build_candidates_from_layer(
    *,
    feature_store_root: str,
    layer: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    feature_deps_path: str = "config/feature_dependencies.yaml",
) -> List[str]:
    store = FeatureStore(str(feature_store_root))
    spec = FeatureStoreSpec(
        layer=str(layer), symbol=str(symbol), timeframe=str(timeframe)
    )
    df = store.read_range(
        spec, start=pd.Timestamp(start_date), end=pd.Timestamp(end_date)
    )
    cols = set(df.columns)

    deps = yaml.safe_load(Path(feature_deps_path).read_text(encoding="utf-8")) or {}
    features = deps.get("features", {}) or {}

    cand: List[str] = []
    for feat, info in features.items():
        outs = (
            [str(c) for c in (info.get("output_columns") or [])]
            if isinstance(info, dict)
            else []
        )
        if outs and any((c in cols) for c in outs):
            cand.append(str(feat))
    return sorted(set(cand))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Base nn config dir")
    p.add_argument("--symbols", required=True, help="Comma-separated symbols")
    p.add_argument("--timeframe", default="240T")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--features-store-root", default="feature_store")
    p.add_argument("--features-store-layer", required=True)
    p.add_argument("--objective", default="dir_auc")
    p.add_argument("--search-algo", default="pipeline")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--exclude-columns", default="atr")
    p.add_argument("--tag", default=None)
    p.add_argument("--expand-semantic-singletons", action="store_true", default=False)
    p.add_argument("--run-train", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config).resolve()
    tag = (
        str(args.tag)
        if args.tag
        else f"{cfg_dir.name}_{args.start_date}_{args.end_date}"
    )

    out_dir = Path("results") / "search" / "nnmultihead" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) candidates
    sym0 = _first_symbol(args.symbols)
    candidates = build_candidates_from_layer(
        feature_store_root=str(args.features_store_root),
        layer=str(args.features_store_layer),
        symbol=sym0,
        timeframe=str(args.timeframe),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
    )
    cand_yaml = out_dir / "candidates_from_layer.yaml"
    _dump_yaml(
        cand_yaml,
        {
            "feature_pipeline": {
                "requested_features": candidates,
                "invert_features": [],
                "post_processors": [],
                "selector": None,
            },
            "_comment": f"Auto-generated from layer={args.features_store_layer}, symbol={sym0}, n={len(candidates)}",
        },
    )

    # 2) factor-eval -> PoolB
    poolb_dir = out_dir / "pool_b_primitives"
    poolb_yaml = poolb_dir / "features_pool_b_primitives.yaml"
    cmd_fe = [
        "mlbot",
        "nnmultihead",
        "factor-eval",
        "--no-docker",
        "--config-dir",
        str(cfg_dir),
        "--candidates-yaml",
        str(cand_yaml),
        "--symbols",
        str(args.symbols),
        "--timeframe",
        str(args.timeframe),
        "--features-store-root",
        str(args.features_store_root),
        "--features-store-layer",
        str(args.features_store_layer),
        "--start-date",
        str(args.start_date),
        "--end-date",
        str(args.end_date),
        "--output-dir",
        str(poolb_dir),
        "--export-yaml",
        str(poolb_yaml),
    ]
    subprocess.run(cmd_fe, check=True)

    # 3) feature-group-search (PoolA auto from <config>/features_base.yaml)
    search_dir = out_dir / "feature_group_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    cmd_search = [
        "mlbot",
        "nnmultihead",
        "feature-group-search",
        "--no-docker",
        "--base-config",
        str(cfg_dir),
        "--symbols",
        str(args.symbols),
        "--timeframe",
        str(args.timeframe),
        "--start-date",
        str(args.start_date),
        "--end-date",
        str(args.end_date),
        "--features-store-root",
        str(args.features_store_root),
        "--features-store-layer",
        str(args.features_store_layer),
        "--pool-b-yaml",
        str(poolb_yaml),
        "--objective",
        str(args.objective),
        "--search-algo",
        str(args.search_algo),
        "--epochs",
        str(int(args.epochs)),
        "--exclude-columns",
        str(args.exclude_columns),
        "--output-dir",
        str(search_dir),
    ]
    if args.expand_semantic_singletons:
        cmd_search.append("--expand-semantic-singletons")
    subprocess.run(cmd_search, check=True)

    result_path = search_dir / "nn_feature_group_search_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selected_groups = result.get("selected_groups") or []
    final_features = result.get("final_features") or []

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "config": str(cfg_dir),
                "symbols": str(args.symbols),
                "timeframe": str(args.timeframe),
                "start_date": str(args.start_date),
                "end_date": str(args.end_date),
                "layer": str(args.features_store_layer),
                "objective": str(args.objective),
                "selected_groups": selected_groups,
                "final_features": final_features,
                "poolb_yaml": str(poolb_yaml),
                "search_result": str(result_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 4) optional train best config: we reuse base config dir but write a best_config features.yaml
    if args.run_train:
        best_cfg = out_dir / "best_config"
        best_cfg.mkdir(parents=True, exist_ok=True)
        # copy labels/model
        (best_cfg / "labels.yaml").write_text(
            (cfg_dir / "labels.yaml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        (best_cfg / "model.yaml").write_text(
            (cfg_dir / "model.yaml").read_text(encoding="utf-8"), encoding="utf-8"
        )

        # Build features.yaml that requests exactly final_features (feature funcs + output cols)
        # and uses selector + exclude_columns.
        base_obj = _load_yaml(cfg_dir / "features.yaml")
        fp = (
            (base_obj.get("feature_pipeline") or {})
            if isinstance(base_obj, dict)
            else {}
        )
        fp_out = dict(fp)
        fp_out["requested_features"] = {
            "required": list(final_features),
            "optional_blocks": {},
        }
        fp_out["selector"] = {
            "module": "src.time_series_model.models.nn.feature_selector",
            "function": "select_columns_from_requested_features",
            "params": {
                "requested_features": list(final_features),
                "feature_deps_path": "config/feature_dependencies.yaml",
                "drop_constant": True,
                "exclude_columns": [
                    c.strip()
                    for c in str(args.exclude_columns or "").split(",")
                    if c.strip()
                ],
            },
        }
        best_obj = {
            "description": f"best_config from search tag={tag}",
            "feature_pipeline": fp_out,
        }
        _dump_yaml(best_cfg / "features.yaml", best_obj)

        train_out = out_dir / "train_best"
        cmd_train = [
            "mlbot",
            "nnmultihead",
            "train",
            "--no-docker",
            "--config",
            str(best_cfg),
            "--symbols",
            str(args.symbols),
            "--timeframe",
            str(args.timeframe),
            "--start-date",
            str(args.start_date),
            "--end-date",
            str(args.end_date),
            "--epochs",
            str(int(args.epochs)),
            "--feature-store-root",
            str(args.features_store_root),
            "--feature-store-layer",
            str(args.features_store_layer),
            "--output-dir",
            str(train_out),
        ]
        subprocess.run(cmd_train, check=True)


if __name__ == "__main__":
    main()
