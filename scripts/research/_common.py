"""Shared helpers for ``mlbot research`` subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.research.expr import build_calendar_mask, parse_clause
from src.research.layer_registry import (
    build_layer_mask,
    resolve_features_parquet,
    resolve_layer_context,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_common_research_args(
    p: argparse.ArgumentParser, *, include_target: bool = True
) -> None:
    p.add_argument(
        "--strategy", default=None, help="Strategy slug (bpc/tpc/me/srb/...)"
    )
    p.add_argument(
        "--layer",
        default=None,
        choices=["regime", "prefilter", "gate", "entry", "direction"],
        help="Resolve subset mask + writeback yaml (CLI only; kernels stay layer-agnostic)",
    )
    p.add_argument("--features-parquet", default=None)
    if include_target:
        p.add_argument("--target", default="success_no_rr_extreme")
    p.add_argument(
        "--subset", default=None, help="Extra filter DSL (AND with layer mask)"
    )
    p.add_argument("--calendar-window", default=None)
    p.add_argument("--output", "--out", dest="output", default=None)
    p.add_argument("--out-dir", default=None)


def load_research_frame(args: argparse.Namespace) -> pd.DataFrame:
    pq: Optional[Path] = None
    if args.features_parquet:
        pq = Path(args.features_parquet)
        if not pq.is_absolute():
            pq = (PROJECT_ROOT / pq).resolve()
    elif args.strategy:
        pq = resolve_features_parquet(args.strategy)
    if pq is None or not pq.exists():
        print(
            "ERROR: features parquet not found; pass --features-parquet",
            file=sys.stderr,
        )
        sys.exit(3)
    return pd.read_parquet(pq)


def add_filter_args(p: argparse.ArgumentParser) -> None:
    """Repeatable subset DSL clauses (ANDed); matches rd_loop yaml ``filter: [...]``."""
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Subset DSL clause (repeatable; ANDed with layer/subset mask).",
    )


def build_base_mask(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    strategy = getattr(args, "strategy", None)
    layer = getattr(args, "layer", None)
    if strategy and layer:
        mask = mask & build_layer_mask(df, strategy, layer)
    subset = getattr(args, "subset", None)
    if subset:
        mask = mask & parse_clause(subset)(df)
    for clause in getattr(args, "filter", None) or []:
        mask = mask & parse_clause(str(clause))(df)
    calendar_window = getattr(args, "calendar_window", None)
    if calendar_window:
        mask = mask & build_calendar_mask(df, calendar_window)
    return mask


def resolve_output_path(args: argparse.Namespace, default_name: str) -> Optional[Path]:
    """Resolve target output path for a single artifact.

    - ``--output`` (single file): only the *primary* artifact (whose default_name
      suffix matches ``--output``) is honored; secondary artifacts return None
      so they don't overwrite the primary file.
    - ``--out-dir``: each artifact lands at ``<out-dir>/<default_name>``.
    - neither set: returns None (caller falls back to stdout).
    """
    if args.output:
        p = Path(args.output)
        p = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
        if p.suffix and Path(default_name).suffix != p.suffix:
            return None
        return p
    if args.out_dir:
        d = Path(args.out_dir)
        if not d.is_absolute():
            d = (PROJECT_ROOT / d).resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d / default_name
    return None


def layer_writeback_hint(args: argparse.Namespace) -> None:
    if args.strategy and args.layer:
        _, wb = resolve_layer_context(args.strategy, args.layer)
        if wb:
            print(f"ℹ️  layer writeback target: {wb}")


def resolve_research_feature_column(
    df: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, str]:
    """Resolve --feature or --subject to a concrete parquet column."""
    from src.research.subjects.resolve import attach_subject_column, subject_from_args

    try:
        subject = subject_from_args(
            subject=getattr(args, "subject", None),
            feature=getattr(args, "feature", None),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(3)
    try:
        return attach_subject_column(df, subject)
    except (KeyError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(3)
