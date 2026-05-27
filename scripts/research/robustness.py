"""mlbot research robustness — temporal fold stability on label scan."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.research.stat_kernels.z_test import two_proportion_z

from scripts import quick_layer_scan
from scripts.research._common import (
    add_common_research_args,
    build_base_mask,
    load_research_frame,
    resolve_output_path,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research robustness (temporal fold label scan)"
    )
    add_common_research_args(p)
    p.add_argument("--label", default="success_no_rr_extreme")
    p.add_argument("--feature", required=True)
    p.add_argument("--operator", default="<=")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args(argv)

    df = load_research_frame(args)
    if "timestamp" not in df.columns and df.index.name != "timestamp":
        print("ERROR: need timestamp for temporal folds", file=sys.stderr)
        return 3
    if "timestamp" not in df.columns:
        df = df.reset_index()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.assign(_fold=pd.qcut(ts.rank(method="first"), args.folds, labels=False))
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
    report = {
        "fold_z_scores": zs,
        "mean_z": float(np.mean(zs)) if zs else 0.0,
        "std_z": float(np.std(zs)) if zs else 0.0,
    }
    out = resolve_output_path(args, "robustness.json")
    import json

    text = json.dumps(report, indent=2)
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
