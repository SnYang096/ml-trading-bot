#!/usr/bin/env python3
"""Summarize tree_holdout_tau_rr_scan outputs across strategies and segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/rd_loop/tree_legacy_bull_bear_20260604"
SEGMENTS = ["bear_2022", "bull_2023_2024", "recent_range_to_bear"]
STRATEGIES = [
    "compression_breakout",
    "sr_breakout",
    "trend_following",
    "sr_reversal_rr_reg_long",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pool_bt(data: dict) -> dict:
    by_sym = (
        data.get("holdout_rr_backtest") or data.get("final_backtest_by_symbol") or {}
    )
    if not by_sym:
        return {}
    sharpes = []
    rets = []
    trades = 0
    for bt in by_sym.values():
        sh = bt.get("sharpe")
        if sh is not None:
            sharpes.append(float(sh))
        tr = bt.get("total_return_pct")
        if tr is not None:
            rets.append(float(tr))
        trades += int(bt.get("total_trades") or 0)
    return {
        "sharpe_mean": sum(sharpes) / len(sharpes) if sharpes else None,
        "return_mean_pct": sum(rets) / len(rets) if rets else None,
        "trades": trades,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    root: Path = args.root
    if not root.is_absolute():
        root = (REPO / root).resolve()

    rows: list[dict] = []
    for strat in STRATEGIES:
        for seg in SEGMENTS:
            jpath = root / strat / seg / "tau_scan_holdout_rr.json"
            if not jpath.is_file():
                rows.append({"strategy": strat, "segment": seg, "missing": True})
                continue
            data = _load_json(jpath)
            pooled = _pool_bt(data)
            rows.append(
                {
                    "strategy": strat,
                    "segment": seg,
                    "missing": False,
                    "fixed_q": data.get("recommended_quantile"),
                    "sharpe": pooled.get("sharpe_mean"),
                    "return_pct": pooled.get("return_mean_pct"),
                    "trades": pooled.get("trades"),
                }
            )

    lines = [
        "# Tree legacy strategies — bull/bear segment matrix",
        "",
        f"Root: `{root}`",
        "",
        "Fixed entry quantile: **q=0.10** (vectorbt RR, BTC+ETH pooled mean).",
        "",
        "| strategy | bear_2022 Sharpe | bear ret% | bull Sharpe | bull ret% | recent Sharpe | recent ret% |",
        "|----------|------------------|-----------|-------------|-----------|---------------|-------------|",
    ]
    by_strat: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_strat.setdefault(r["strategy"], {})[r["segment"]] = r

    for strat in STRATEGIES:
        m = by_strat.get(strat, {})

        def _cell(seg: str, key: str) -> str:
            r = m.get(seg) or {}
            if r.get("missing"):
                return "—"
            v = r.get(key)
            if v is None:
                return "—"
            if key == "sharpe":
                return f"{v:.2f}"
            if key == "return_pct":
                return f"{v:.1f}"
            return str(v)

        lines.append(
            f"| {strat} "
            f"| {_cell('bear_2022', 'sharpe')} "
            f"| {_cell('bear_2022', 'return_pct')} "
            f"| {_cell('bull_2023_2024', 'sharpe')} "
            f"| {_cell('bull_2023_2024', 'return_pct')} "
            f"| {_cell('recent_range_to_bear', 'sharpe')} "
            f"| {_cell('recent_range_to_bear', 'return_pct')} |"
        )

    out_md = root / "segment_matrix_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
