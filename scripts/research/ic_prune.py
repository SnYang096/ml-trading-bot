"""mlbot research ic-prune — holdout IC feature-node prune for tree strategies."""

from __future__ import annotations

import argparse
import sys

from scripts.research._common import (
    PROJECT_ROOT,
    add_common_research_args,
    add_filter_args,
    resolve_output_path,
)
from src.research.stat_kernels.ic_prune import DEFAULT_TARGET, run_ic_prune


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Holdout IC prune (tree feature selection)")
    add_common_research_args(p, include_target=False)
    add_filter_args(p)
    p.add_argument(
        "--parquet",
        default=None,
        help="Alias for --features-parquet (rd_loop compatibility)",
    )
    p.add_argument("--holdout-start", default="2025-10-01")
    p.add_argument("--holdout-end", default="2026-04-01")
    p.add_argument("--horizons", default="1,2,3,4,5")
    p.add_argument("--max-lag", type=int, default=5)
    p.add_argument("--min-ic", type=float, default=0.02)
    p.add_argument("--min-n", type=int, default=200)
    p.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Label column for IC (default {DEFAULT_TARGET})",
    )
    p.add_argument(
        "--write-features-yaml",
        default=None,
        help="Update strategy features.yaml with pruned requested_features",
    )
    p.add_argument(
        "--no-write-features-yaml",
        action="store_true",
        help="Do not write features.yaml",
    )
    p.add_argument("--top-n-nodes", type=int, default=None)
    p.add_argument("--intersect-features-yaml", default=None)
    p.add_argument(
        "--always-include",
        default="atr_f",
        help="Comma-separated node names always kept",
    )
    p.add_argument(
        "--invert-mode",
        choices=["none", "auto"],
        default="none",
        help="none: omit invert_features (default for trees); auto: write negative-IC columns",
    )
    p.add_argument(
        "--emit-monotone-constraints",
        default=None,
        help="Write review-only monotone constraint hints to this path",
    )
    args = p.parse_args(argv)

    parquet = args.features_parquet or args.parquet
    if not parquet:
        print(
            "ERROR: pass --features-parquet or --parquet",
            file=sys.stderr,
        )
        return 3

    out_dir = resolve_output_path(args, "ic_prune_holdout.json")
    if out_dir is None:
        print("ERROR: pass --out-dir or --output", file=sys.stderr)
        return 3
    output_dir = out_dir.parent

    always = [x.strip() for x in args.always_include.split(",") if x.strip()]
    write_yaml = None if args.no_write_features_yaml else args.write_features_yaml

    mono_out = args.emit_monotone_constraints

    try:
        run_ic_prune(
            parquet=parquet,
            output_dir=output_dir,
            holdout_start=args.holdout_start,
            holdout_end=args.holdout_end,
            horizons=args.horizons,
            max_lag=args.max_lag,
            min_ic=args.min_ic,
            min_n=args.min_n,
            target=args.target,
            write_features_yaml=write_yaml,
            top_n_nodes=args.top_n_nodes,
            intersect_features_yaml=args.intersect_features_yaml,
            always_include=always,
            invert_mode=args.invert_mode,
            emit_monotone_constraints=mono_out,
            project_root=PROJECT_ROOT,
        )
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
