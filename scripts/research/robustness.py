"""mlbot research robustness — temporal fold or gate robustness score."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.research.stat_kernels.robustness import (
    UnifiedOptimizationConfig,
    compute_robustness_score,
)
from src.research.stat_kernels.z_test import two_proportion_z

from scripts.research._common import (
    add_common_research_args,
    build_base_mask,
    load_research_frame,
    resolve_output_path,
    resolve_research_feature_column,
)

_ENTRY_OP_TO_DENY = {
    "<=": "gt",
    "<": "gt",
    "le": "gt",
    ">=": "lt",
    ">": "lt",
    "ge": "lt",
}


def _temporal_fold_report(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    if "timestamp" not in df.columns and df.index.name != "timestamp":
        raise ValueError("need timestamp for temporal folds")
    if "timestamp" not in df.columns:
        df = df.reset_index()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.assign(_fold=pd.qcut(ts.rank(method="first"), args.folds, labels=False))
    if args.label not in df.columns:
        raise ValueError(f"label '{args.label}' missing")
    label = df[args.label].astype(bool)
    zs = []
    for fold in sorted(df["_fold"].dropna().unique()):
        m = build_base_mask(df, args) & (df["_fold"] == fold)
        sub = df.loc[m]
        if len(sub) < 20:
            continue
        feat = pd.to_numeric(sub[args.feature], errors="coerce")
        hit = (
            feat <= args.threshold
            if args.operator in ("<=", "<", "le")
            else feat >= args.threshold
        )
        y = label.loc[m]
        p_hit = float(y[hit].mean()) if hit.any() else 0.0
        p_oth = float(y[~hit].mean()) if (~hit).any() else 0.0
        zs.append(two_proportion_z(p_hit, int(hit.sum()), p_oth, int((~hit).sum())))
    return {
        "kernel": "temporal",
        "fold_z_scores": zs,
        "mean_z": float(np.mean(zs)) if zs else 0.0,
        "std_z": float(np.std(zs)) if zs else 0.0,
    }


def _gate_robustness_report(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    m = build_base_mask(df, args)
    sub = df.loc[m].copy()
    if args.label not in sub.columns:
        raise ValueError(f"label '{args.label}' missing")
    deny_op = _ENTRY_OP_TO_DENY.get(args.operator)
    if deny_op is None:
        raise ValueError(f"unsupported operator for gate kernel: {args.operator}")
    label_col = args.label
    if sub[label_col].dtype == bool:
        sub = sub.assign(_is_good=sub[label_col].astype(int))
        label_col = "_is_good"
    cfg = UnifiedOptimizationConfig(temporal_cv_folds=args.folds)
    score = compute_robustness_score(
        sub,
        args.feature,
        deny_op,
        args.threshold,
        label_col=label_col,
        config=cfg,
    )
    return {"kernel": "gate", "robustness": score.to_dict(), "deny_operator": deny_op}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research robustness (temporal folds or gate score)"
    )
    add_common_research_args(p)
    p.add_argument("--label", default="success_no_rr_extreme")
    p.add_argument("--feature", default=None)
    p.add_argument(
        "--subject",
        default=None,
        help="feature:COL or model.score:PATH|COL",
    )
    p.add_argument("--operator", default="<=")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--kernel",
        choices=("temporal", "gate"),
        default="temporal",
        help="temporal: fold z-tests; gate: compute_robustness_score",
    )
    args = p.parse_args(argv)

    df = load_research_frame(args)
    df, feature_col = resolve_research_feature_column(df, args)
    args.feature = feature_col
    try:
        if args.kernel == "gate":
            report = _gate_robustness_report(df, args)
        else:
            report = _temporal_fold_report(df, args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    text = json.dumps(report, indent=2)
    out = resolve_output_path(args, "robustness.json")
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
