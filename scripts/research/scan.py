"""mlbot research scan — label / condition effect screening."""

from __future__ import annotations

import argparse
import sys

from scripts import quick_layer_scan
from scripts.research._common import (
    add_common_research_args,
    build_base_mask,
    layer_writeback_hint,
    load_research_frame,
    resolve_output_path,
    resolve_research_feature_column,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research scan (condition-set / feature-plateau / pair-scan)"
    )
    sub = p.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    add_common_research_args(common)
    common.add_argument("--label", default="success_no_rr_extreme")
    common.add_argument("--filter", nargs="*", default=[])

    cs = sub.add_parser("condition-set", parents=[common])
    cs.add_argument("--condition", action="append", required=True)

    fp = sub.add_parser("feature-plateau", parents=[common])
    fp.add_argument("--feature", default=None)
    fp.add_argument(
        "--subject",
        default=None,
        help="feature:COL or model.score:PATH|COL",
    )
    fp.add_argument("--operator", default="<=")
    fp.add_argument("--grid", required=True)

    ps = sub.add_parser("pair-scan", parents=[common])
    ps.add_argument(
        "--pair-a",
        required=True,
        help="'feature:op:grid' e.g. 'vol_persistence:>:0.003,0.01,0.03'",
    )
    ps.add_argument("--pair-b", required=True)

    args = p.parse_args(argv)
    layer_writeback_hint(args)
    df = load_research_frame(args)
    if args.label not in df.columns:
        print(f"ERROR: label '{args.label}' missing", file=sys.stderr)
        return 3
    label = df[args.label].astype(bool)
    base_mask = build_base_mask(df, args)

    if args.mode == "feature-plateau":
        df, feature_col = resolve_research_feature_column(df, args)
        ns = argparse.Namespace(
            feature=feature_col,
            operator=getattr(args, "operator", "<="),
            grid=getattr(args, "grid", ""),
        )
        report = quick_layer_scan.mode_feature_plateau(ns, df, label, base_mask)
    else:
        ns = argparse.Namespace(
            feature=getattr(args, "feature", None),
            operator=getattr(args, "operator", "<="),
            grid=getattr(args, "grid", ""),
            condition=getattr(args, "condition", []),
            pair_a=getattr(args, "pair_a", None),
            pair_b=getattr(args, "pair_b", None),
        )
        if args.mode == "condition-set":
            report = quick_layer_scan.mode_condition_set(ns, df, label, base_mask)
        else:
            report = quick_layer_scan.mode_pair_scan(ns, df, label, base_mask)
    out = resolve_output_path(args, f"scan_{args.mode}.md")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
