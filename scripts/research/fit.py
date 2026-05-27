"""mlbot research fit — layer-agnostic LightGBM research training."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.research.layer_registry import feature_pool_path, resolve_features_parquet
from src.research.subjects.feature import FeaturePool
from src.research.tree_trainer import train_lightgbm_classifier

from scripts.research._common import PROJECT_ROOT, build_base_mask, load_research_frame


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Research fit (exploratory LightGBM)")
    p.add_argument("--strategy", required=True)
    p.add_argument("--layer", default="prefilter")
    p.add_argument("--features-parquet", default=None)
    p.add_argument("--feature-pool", default=None)
    p.add_argument("--target", default="success_no_rr_extreme")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    df = load_research_frame(args)
    pool_path = (
        Path(args.feature_pool)
        if args.feature_pool
        else feature_pool_path(args.strategy, args.layer)
    )
    pool = FeaturePool.from_yaml(pool_path)
    feature_cols = [c for c in pool.features if c in df.columns]
    if not feature_cols:
        print("ERROR: no feature columns from pool found in parquet", file=sys.stderr)
        return 3
    if args.target not in df.columns:
        from src.research.labels import derive_is_good_from_forward_rr

        derive_is_good_from_forward_rr(df, label_col=args.target)

    mask = build_base_mask(df, args)
    sub = df.loc[mask].copy()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = (
            PROJECT_ROOT / "results/research/fit" / args.strategy / args.layer / run_id
        )
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    result = train_lightgbm_classifier(sub, feature_cols, args.target, out_dir)
    print(f"✅ model: {result.model_path}")
    print(f"   metrics: {result.metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
