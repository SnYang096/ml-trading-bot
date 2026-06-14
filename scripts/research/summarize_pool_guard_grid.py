#!/usr/bin/env python3
"""Summarize TPC trend_pool_guard variant grid (capital + funnel rejects)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

FUNNEL_KEYS = (
    "reject_pcm_trend_pool_unprotected_cap",
    "reject_pcm_trend_pool_post_unlock_cap",
    "reject_pcm_trend_pool_corr",
    "reject_pcm_trend_pool_anchor_first",
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_from_variant(name: str) -> str:
    for seg in (
        "bear_2022",
        "bull_2023_2024",
        "recent_range_to_bear",
        "recent_6m_oos",
    ):
        if name.endswith(f"_{seg}"):
            return seg
    return ""


def _variant_family(name: str) -> str:
    for prefix in ("G0_prod_1_2", "G1_be1_3", "G2_be3_3", "G3_be3_6", "G4_guard_off"):
        if name.startswith(prefix):
            return prefix
    return name.split("_bear")[0].split("_bull")[0]


def summarize_root(root: Path) -> None:
    rows: list[dict] = []
    for cap_path in sorted(root.rglob("capital_report.json")):
        rel = cap_path.parent.relative_to(root)
        variant = (
            str(rel).replace("/", "_") if rel != Path(".") else cap_path.parent.name
        )
        if cap_path.parent.name == "capital_report.json":
            continue
        # segment matrix: canonical/G0_prod_1_2/bear_2022/capital_report.json
        parts = cap_path.parts
        if "canonical" in parts or "smoke" in parts:
            idx = (
                parts.index("canonical")
                if "canonical" in parts
                else parts.index("smoke")
            )
            variant = (
                f"{parts[idx + 1]}_{parts[idx + 2]}"
                if len(parts) > idx + 2
                else parts[idx + 1]
            )

        cap = _load_json(cap_path)
        eb_path = cap_path.parent / "event_backtest.json"
        eb = _load_json(eb_path)
        funnel = eb.get("funnel") or {}
        rows.append(
            {
                "variant": variant,
                "family": _variant_family(variant),
                "segment": _segment_from_variant(variant),
                "total_r": cap.get("total_r"),
                "max_dd_pct": cap.get("max_drawdown_pct"),
                "trades": cap.get("trades"),
                "cagr": cap.get("cagr"),
                **{k: int(funnel.get(k) or 0) for k in FUNNEL_KEYS},
            }
        )

    if not rows:
        print(f"No capital_report.json under {root}", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: (r["family"], r["segment"]))

    print(f"# pool guard summary — {root}\n")
    print(
        "| variant | segment | total_r | max_dd | trades | unprot_reject | post_unlock | corr |"
    )
    print(
        "|---------|---------|--------:|-------:|-------:|--------------:|------------:|-----:|"
    )
    for r in rows:
        print(
            f"| {r['variant']} | {r['segment'] or '-'} | "
            f"{r['total_r']:.2f} | {100 * (r['max_dd_pct'] or 0):.1f}% | {r['trades']} | "
            f"{r['reject_pcm_trend_pool_unprotected_cap']} | "
            f"{r['reject_pcm_trend_pool_post_unlock_cap']} | "
            f"{r['reject_pcm_trend_pool_corr']} |"
        )

    by_family: dict[str, list] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)

    print("\n## Sum R by family (segments present)\n")
    for fam, seg_rows in sorted(by_family.items()):
        sum_r = sum(float(x["total_r"] or 0) for x in seg_rows)
        worst_dd = min(float(x["max_dd_pct"] or 0) for x in seg_rows)
        print(
            f"- **{fam}**: sum_r={sum_r:.2f}, worst_segment_maxDD={100 * worst_dd:.1f}% "
            f"({len(seg_rows)} segments)"
        )

    canonical_fams = sorted(
        fam for fam in by_family if any(r["segment"] for r in by_family[fam])
    )
    if canonical_fams:
        seg_order = (
            "bear_2022",
            "bull_2023_2024",
            "recent_range_to_bear",
        )
        print("\n## Canonical matrix (promote-style)\n")
        print(
            f"{'family':<14} {'bear':>8} {'bull':>8} {'recent':>8} {'sum':>8} {'worstDD':>8}"
        )
        for fam in canonical_fams:
            by_seg = {r["segment"]: r for r in by_family[fam]}
            seg_r = [float(by_seg[s]["total_r"] or 0) for s in seg_order if s in by_seg]
            seg_dd = [
                float(by_seg[s]["max_dd_pct"] or 0) for s in seg_order if s in by_seg
            ]
            if not seg_r:
                continue
            cols = [by_seg[s]["total_r"] if s in by_seg else None for s in seg_order]
            print(
                f"{fam:<14} "
                + " ".join(f"{(c or 0):8.2f}" for c in cols)
                + f" {sum(seg_r):8.2f} {100 * min(seg_dd):7.1f}%"
            )


def main() -> None:
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else _REPO / "results/tpc/experiments/pool_guard_20260612"
    )
    summarize_root(root.resolve())


if __name__ == "__main__":
    main()
