#!/usr/bin/env python3
"""Summarize tpc_entry_semantic_validate grid capital_report.json files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/tpc/experiments/entry_semantic_validate_20260604"

VARIANTS = [
    "E0_prod",
    "S50_depth_gt50",
    "S51_depth_gt50_ema_near",
    "E1_depth_ge15",
    "E2_anti_chase",
    "E3_gate_pe",
    "E4_turbo_exec",
]
SEGMENTS = ["bear_2022", "bull_2023_2024", "recent_range_to_bear"]
FULL_VARIANTS = [
    ("full/E0_prod", "E0_prod_full"),
    ("full/S50_depth_gt50", "S50_full"),
    ("full/S51_depth_gt50_ema_near", "S51_full"),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows: list[dict] = []
    for var in VARIANTS:
        seg_r: dict[str, float] = {}
        seg_dd: dict[str, float] = {}
        seg_trades: dict[str, int] = {}
        for seg in SEGMENTS:
            cap = ROOT / var / seg / "capital_report.json"
            if not cap.is_file():
                continue
            d = _load(cap)
            seg_r[seg] = float(d.get("total_r") or 0.0)
            seg_dd[seg] = float(d.get("max_drawdown_pct") or 0.0)
            seg_trades[seg] = int(d.get("trades") or 0)
        if not seg_r:
            continue
        rows.append(
            {
                "variant": var,
                "bear_r": seg_r.get("bear_2022"),
                "bull_r": seg_r.get("bull_2023_2024"),
                "recent_r": seg_r.get("recent_range_to_bear"),
                "sum_r": sum(seg_r.values()),
                "worst_dd": min(seg_dd.values()) if seg_dd else None,
                "trades": sum(seg_trades.values()),
            }
        )

    print("segment_matrix (canonical 3 segments)")
    print(
        f"{'variant':<28} {'bear':>8} {'bull':>8} {'recent':>8} {'sum':>8} {'maxDD':>8} {'trades':>7}"
    )
    for r in rows:
        print(
            f"{r['variant']:<28} "
            f"{r.get('bear_r') or 0:8.2f} "
            f"{r.get('bull_r') or 0:8.2f} "
            f"{r.get('recent_r') or 0:8.2f} "
            f"{r['sum_r']:8.2f} "
            f"{(r.get('worst_dd') or 0)*100:7.1f}% "
            f"{r['trades']:7d}"
        )

    print("\nfull window highcap")
    for sub, label in FULL_VARIANTS:
        cap = ROOT / sub / "capital_report.json"
        if not cap.is_file():
            print(f"  {label}: missing")
            continue
        d = _load(cap)
        print(
            f"  {label}: R={d.get('total_r', 0):.2f} "
            f"maxDD={float(d.get('max_drawdown_pct', 0))*100:.1f}% "
            f"trades={d.get('trades', 0)}"
        )

    n = len(list(ROOT.rglob("capital_report.json")))
    print(f"\nreports found: {n}/24")
    return 0 if n >= 24 else 1


if __name__ == "__main__":
    raise SystemExit(main())
