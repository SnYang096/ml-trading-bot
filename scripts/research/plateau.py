"""mlbot research plateau — threshold plateau (label proxy or snotio KPI)."""

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
from src.research.stat_kernels.snotio_calc import snotio_plateau_payload


def _format_snotio_report(payload: dict) -> str:
    feature = payload["feature"]
    op = payload["operator"]
    md = [f"# snotio_plateau · {feature} {op} ?", ""]
    md.append("| threshold | trades | snotio |")
    md.append("|---:|---:|---:|")
    for row in payload.get("rows", []):
        sn = row.get("snotio", 0.0)
        flag = " (too_few)" if row.get("too_few") else ""
        md.append(f"| {row['threshold']:.3g} | {row['trades']} | {sn:.4f}{flag} |")
    md.append("")
    if payload.get("is_plateau"):
        md.append(
            f"**Plateau**: [{payload.get('start_threshold')}, {payload.get('end_threshold')}] "
            f"recommended={payload.get('recommended')} conf={payload.get('confidence')}"
        )
    else:
        md.append(f"**No plateau**: {payload.get('reason', 'n/a')}")
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Research plateau (feature threshold scan)")
    add_common_research_args(p)
    p.add_argument("--label", default="success_no_rr_extreme")
    p.add_argument("--feature", required=True)
    p.add_argument("--operator", default="<=")
    p.add_argument("--grid", required=True)
    p.add_argument(
        "--kpi",
        choices=("label", "snotio"),
        default="label",
        help="label: success-rate proxy (default); snotio: mean(forward_rr) scan",
    )
    p.add_argument("--r-col", default="forward_rr", help="R column when --kpi snotio")
    p.add_argument("--min-trades", type=int, default=20)
    args = p.parse_args(argv)

    df = load_research_frame(args)
    base_mask = build_base_mask(df, args)
    grid = [float(x) for x in args.grid.split(",") if x.strip()]

    if args.kpi == "snotio":
        payload = snotio_plateau_payload(
            df,
            args.feature,
            args.operator,
            grid,
            base_mask,
            r_col=args.r_col,
            min_trades=args.min_trades,
        )
        report = _format_snotio_report(payload)
    else:
        if args.label not in df.columns:
            print(f"ERROR: label '{args.label}' missing", file=sys.stderr)
            return 3
        label = df[args.label].astype(bool)
        ns = argparse.Namespace(
            feature=args.feature, operator=args.operator, grid=args.grid
        )
        payload = quick_layer_scan.feature_plateau_payload(ns, df, label, base_mask)
        report = quick_layer_scan.mode_feature_plateau(ns, df, label, base_mask)

    out_md = resolve_output_path(args, "plateau.md")
    if out_md:
        out_md.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out_md}")
    else:
        print(report)

    json_out = resolve_output_path(args, "plateau.json")
    if json_out:
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
