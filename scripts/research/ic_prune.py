"""mlbot research ic-prune — holdout IC feature-node prune for tree strategies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.research._common import (
    PROJECT_ROOT,
    add_common_research_args,
    add_filter_args,
    resolve_output_path,
)
from src.research.stat_kernels.ic_prune import DEFAULT_TARGET, run_ic_prune
from src.research.stat_kernels.ic_screen_config import resolve_ic_prune_params


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Holdout IC prune (tree feature selection)")
    add_common_research_args(p, include_target=False)
    add_filter_args(p)
    p.add_argument(
        "--parquet",
        default=None,
        help="Alias for --features-parquet (rd_loop compatibility)",
    )
    p.add_argument(
        "--config-dir",
        default=None,
        help="Strategy config dir containing ic_screen.yaml (overrides --strategy path)",
    )
    p.add_argument("--holdout-start", default=None)
    p.add_argument("--holdout-end", default=None)
    p.add_argument("--horizons", default=None)
    p.add_argument("--max-lag", type=int, default=None)
    p.add_argument(
        "--allowed-best-lags",
        default=None,
        help="Comma-separated best_lag whitelist (default from ic_screen.yaml)",
    )
    p.add_argument(
        "--reject-peak-at",
        type=int,
        default=None,
        help="Drop columns whose |IC| peaks at this lag (default from ic_screen.yaml)",
    )
    p.add_argument("--min-ic", type=float, default=None)
    p.add_argument("--min-n", type=int, default=None)
    p.add_argument(
        "--target",
        default=None,
        help=f"Label column for IC (default from ic_screen.yaml or {DEFAULT_TARGET})",
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
    p.add_argument(
        "--writeback-mode",
        choices=["columns", "nodes"],
        default=None,
        help="columns: top-|IC| output columns; nodes: whole compute nodes",
    )
    p.add_argument(
        "--top-n-columns",
        type=int,
        default=None,
        help="Max model input columns when writeback_mode=columns",
    )
    p.add_argument("--top-n-nodes", type=int, default=None)
    p.add_argument("--intersect-features-yaml", default=None)
    p.add_argument(
        "--write-model-features-yaml",
        default=None,
        help="Archetype model_features.yaml path",
    )
    p.add_argument(
        "--no-write-model-features-yaml",
        action="store_true",
        help="Skip archetypes/model_features.yaml writeback",
    )
    p.add_argument(
        "--always-include",
        default=None,
        help="Comma-separated node names always kept",
    )
    p.add_argument(
        "--invert-mode",
        choices=["none", "auto"],
        default=None,
        help="none: omit invert_features; auto: write negative-IC columns",
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

    cli_overrides = {
        k: v
        for k, v in {
            "holdout_start": args.holdout_start,
            "holdout_end": args.holdout_end,
            "horizons": args.horizons,
            "max_lag": args.max_lag,
            "allowed_best_lags": args.allowed_best_lags,
            "reject_peak_at": args.reject_peak_at,
            "min_ic": args.min_ic,
            "min_n": args.min_n,
            "target": args.target,
            "writeback_mode": args.writeback_mode,
            "top_n_columns": args.top_n_columns,
            "top_n_nodes": args.top_n_nodes,
            "intersect_features_yaml": args.intersect_features_yaml,
            "always_include": args.always_include,
            "invert_mode": args.invert_mode,
            "write_features_yaml": (
                False if args.no_write_features_yaml else args.write_features_yaml
            ),
            "write_model_features_yaml": (
                False
                if args.no_write_model_features_yaml
                else args.write_model_features_yaml
            ),
        }.items()
        if v is not None
    }

    params = resolve_ic_prune_params(
        strategy=args.strategy,
        config_dir=args.config_dir,
        overrides=cli_overrides,
        project_root=PROJECT_ROOT,
    )
    summary = params.pop("_ic_screen_summary", None)
    params.pop("_strategy_config_dir", None)
    if summary:
        print(f"ic_screen: {summary}")

    out_dir = resolve_output_path(args, "ic_prune_holdout.json")
    if out_dir is None:
        print("ERROR: pass --out-dir or --output", file=sys.stderr)
        return 3
    output_dir = out_dir.parent

    always_raw = params.pop("always_include", "atr_f")
    always = [x.strip() for x in str(always_raw).split(",") if x.strip()]
    write_yaml = params.pop("write_features_yaml", None)
    model_features_yaml = params.pop("write_model_features_yaml", None)
    intersect_yaml = params.pop("intersect_features_yaml", None)
    holdout_start = params.pop("holdout_start", "2025-10-01")
    holdout_end = params.pop("holdout_end", "2026-04-01")

    for path_key in (
        "write_features_yaml",
        "write_model_features_yaml",
        "intersect_features_yaml",
    ):
        val = params.get(path_key)
        if val and val is not False:
            p = Path(str(val))
            if not p.is_absolute():
                params[path_key] = str((PROJECT_ROOT / p).resolve())

    mono_out = args.emit_monotone_constraints

    try:
        run_ic_prune(
            parquet=parquet,
            output_dir=output_dir,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            intersect_features_yaml=intersect_yaml,
            write_features_yaml=write_yaml,
            write_model_features_yaml=model_features_yaml,
            strategy=args.strategy,
            project_root=PROJECT_ROOT,
            always_include=always,
            emit_monotone_constraints=mono_out,
            **params,
        )
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
