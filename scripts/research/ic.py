"""mlbot research ic — IC decay with horizon shift."""

from __future__ import annotations

import argparse
import json
import sys

from scripts.research._common import (
    add_common_research_args,
    add_filter_args,
    build_base_mask,
    load_research_frame,
    resolve_output_path,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Research IC decay")
    add_common_research_args(p, include_target=False)
    add_filter_args(p)
    p.add_argument(
        "--label",
        default=None,
        help="Ignored; rd_loop yaml compatibility only.",
    )
    p.add_argument("--features", required=True, help="Comma-separated feature columns")
    p.add_argument("--horizons", default="1,3,5,10,20")
    p.add_argument("--target", default="forward_rr", help="Column for rank IC target")
    p.add_argument("--baseline-json", default=None)
    args = p.parse_args(argv)

    df = load_research_frame(args)
    features = [x.strip() for x in args.features.split(",") if x.strip()]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    base_mask = build_base_mask(df, args)

    ns = argparse.Namespace(
        features=",".join(features),
        horizons=",".join(str(h) for h in horizons),
        target=args.target,
        baseline_json=args.baseline_json,
    )
    from scripts import quick_layer_scan

    report = quick_layer_scan.mode_ic_decay(ns, df, base_mask)
    out = resolve_output_path(args, "ic_decay.md")
    if out:
        out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(report)

    rows = __import__(
        "src.research.stat_kernels.ic", fromlist=["ic_decay_rows"]
    ).ic_decay_rows(df, features, horizons, args.target, mask=base_mask)
    json_out = resolve_output_path(args, "ic_decay.json")
    if json_out:
        json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
