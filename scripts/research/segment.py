"""mlbot research segment — bucket-by segmented scans."""

from __future__ import annotations

import argparse
import sys

from scripts import quick_layer_scan
from scripts.research._common import (
    add_common_research_args,
    build_base_mask,
    load_research_frame,
    resolve_output_path,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Research segment (bucket-by)")
    add_common_research_args(p)
    p.add_argument("--label", default="success_no_rr_extreme")
    p.add_argument("--bucket-by", required=True)
    p.add_argument(
        "--mode", default="condition-set", choices=["condition-set", "feature-plateau"]
    )
    p.add_argument("--condition", action="append", default=[])
    p.add_argument("--feature", default=None)
    p.add_argument("--operator", default="<=")
    p.add_argument("--grid", default="")
    args = p.parse_args(argv)

    if args.mode == "condition-set" and not args.condition:
        print("ERROR: --condition required for condition-set mode", file=sys.stderr)
        return 3
    if args.mode == "feature-plateau" and not args.feature:
        print("ERROR: --feature required for feature-plateau mode", file=sys.stderr)
        return 3

    df = load_research_frame(args)
    label = (
        df[args.label].astype(bool)
        if args.label in df.columns
        else df.index.to_series().astype(bool)
    )
    base_mask = build_base_mask(df, args)
    ns = argparse.Namespace(
        mode=args.mode,
        feature=args.feature,
        operator=args.operator,
        grid=args.grid,
        condition=args.condition,
        bucket_by=args.bucket_by,
    )
    report = quick_layer_scan._bucketed_report(ns, df, label, base_mask, args.bucket_by)
    out = resolve_output_path(args, "segment.md")
    if out:
        out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
