"""mlbot research plateau — threshold plateau (label proxy scan)."""

from __future__ import annotations

import argparse
import json
import sys

from scripts import quick_layer_scan
from scripts.research._common import (
    add_common_research_args,
    build_base_mask,
    load_research_frame,
    resolve_output_path,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Research plateau (feature threshold scan)")
    add_common_research_args(p)
    p.add_argument("--label", default="success_no_rr_extreme")
    p.add_argument("--feature", required=True)
    p.add_argument("--operator", default="<=")
    p.add_argument("--grid", required=True)
    args = p.parse_args(argv)

    df = load_research_frame(args)
    label = df[args.label].astype(bool)
    base_mask = build_base_mask(df, args)
    ns = argparse.Namespace(
        feature=args.feature, operator=args.operator, grid=args.grid
    )
    report = quick_layer_scan.mode_feature_plateau(ns, df, label, base_mask)
    out_md = resolve_output_path(args, "plateau.md")
    if out_md:
        out_md.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out_md}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
