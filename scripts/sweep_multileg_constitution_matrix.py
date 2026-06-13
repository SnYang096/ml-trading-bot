#!/usr/bin/env python3
"""Sweep multi-leg constitution limits from live YAML baseline (one knob per variant).

Loads ``live/highcap/config/constitution/constitution.yaml``, deep-copies, injects a
single override path, and replays joint chop_grid + trend_scalp trades with
``simulate_account_with_constitution``.

Usage
-----
    python scripts/sweep_multileg_constitution_matrix.py \\
        --chop-root results/multileg_joint/sizing_072_20260613/chop_grid \\
        --trend-root results/multileg_joint/sizing_072_20260613/trend_scalp \\
        --segments bear_2022 bull_2023_2024 recent_range_to_bear recent_6m_oos

Output: CSV + markdown table under ``--output-dir``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.multileg_sizing import resolve_multi_leg_unit_notionals
from src.live_data_stream.constitution_config import (
    load_constitution_dict,
    multi_leg_section,
    resolve_multileg_sim_limits,
)
from src.sim.multileg_account_sim import (
    apply_multileg_segment_gates,
    filter_trades_by_segment_blocks,
    load_chop_segments,
    load_chop_trades,
    load_trend_segments,
    load_trend_trades,
    simulate_account_with_constitution,
)

PathTuple = Tuple[str, ...]


def _set_nested(cfg: Dict[str, Any], path: PathTuple, value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _load_combined_trades(
    *,
    chop_root: Optional[Path],
    trend_root: Optional[Path],
    segments: Sequence[str],
    max_concurrent: int,
) -> pd.DataFrame:
    chop_tr_parts: List[pd.DataFrame] = []
    trend_tr_parts: List[pd.DataFrame] = []
    chop_seg_parts: List[pd.DataFrame] = []
    trend_seg_parts: List[pd.DataFrame] = []

    for seg in segments:
        if chop_root is not None:
            ct = load_chop_trades(chop_root, seg)
            if not ct.empty:
                chop_tr_parts.append(ct)
            cs = load_chop_segments(chop_root, seg)
            if not cs.empty:
                chop_seg_parts.append(cs)
        if trend_root is not None:
            tt = load_trend_trades(trend_root, seg)
            if not tt.empty:
                trend_tr_parts.append(tt)
            ts = load_trend_segments(trend_root, seg)
            if not ts.empty:
                trend_seg_parts.append(ts)

    chop_tr = (
        pd.concat(chop_tr_parts, ignore_index=True) if chop_tr_parts else pd.DataFrame()
    )
    trend_tr = (
        pd.concat(trend_tr_parts, ignore_index=True)
        if trend_tr_parts
        else pd.DataFrame()
    )
    chop_seg = (
        pd.concat(chop_seg_parts, ignore_index=True)
        if chop_seg_parts
        else pd.DataFrame()
    )
    trend_seg = (
        pd.concat(trend_seg_parts, ignore_index=True)
        if trend_seg_parts
        else pd.DataFrame()
    )

    gate_stats = apply_multileg_segment_gates(
        chop_seg,
        trend_seg,
        max_concurrent_multi_leg_symbols=int(max_concurrent or 0),
    )
    chop_allowed = filter_trades_by_segment_blocks(
        chop_tr, gate_stats.blocked_chop_segment_ids
    )
    trend_allowed = filter_trades_by_segment_blocks(
        trend_tr, gate_stats.blocked_trend_segment_ids
    )
    return pd.concat(
        [x for x in (chop_allowed, trend_allowed) if not x.empty],
        ignore_index=True,
    )


def _matrix_variants() -> List[Dict[str, Any]]:
    """One override (or fuse mode) per row — avoids incomplete hand-written YAML."""
    dd_paths = (
        ("kill_switch", "max_dd"),
        ("multi_leg", "account", "max_drawdown_pct"),
    )
    net_path = ("multi_leg", "risk_limits", "max_symbol_net_notional_pct")

    return [
        {
            "variant": "prod",
            "label": "prod baseline",
            "overrides": [],
            "fuse_mode": "hard",
        },
        {
            "variant": "max_dd_half",
            "label": "max_dd 20%→10% (both paths)",
            "overrides": [(dd_paths, 0.10)],
            "fuse_mode": "hard",
        },
        {
            "variant": "max_dd_15",
            "label": "max_dd 20%→15%",
            "overrides": [(dd_paths, 0.15)],
            "fuse_mode": "hard",
        },
        {
            "variant": "net_cap_100",
            "label": "max_symbol_net 1.80→1.00",
            "overrides": [(net_path, 1.00)],
            "fuse_mode": "hard",
        },
        {
            "variant": "net_cap_120",
            "label": "max_symbol_net 1.80→1.20",
            "overrides": [(net_path, 1.20)],
            "fuse_mode": "hard",
        },
        {
            "variant": "fuse_derate",
            "label": "prod + tier_derate (soft@50% max_dd, size×0.5)",
            "overrides": [],
            "fuse_mode": "tier_derate",
        },
        {
            "variant": "fuse_daily_scaled",
            "label": "prod + tier_daily_scaled (daily loss ∝ gross)",
            "overrides": [],
            "fuse_mode": "tier_daily_scaled",
        },
        {
            "variant": "max_dd_half_fuse_derate",
            "label": "max_dd 10% + tier_derate",
            "overrides": [(dd_paths, 0.10)],
            "fuse_mode": "tier_derate",
        },
        {
            "variant": "max_dd_half_fuse_daily",
            "label": "max_dd 10% + tier_daily_scaled",
            "overrides": [(dd_paths, 0.10)],
            "fuse_mode": "tier_daily_scaled",
        },
    ]


def _run_variant(
    *,
    base_cfg: Dict[str, Any],
    variant: Dict[str, Any],
    trades: pd.DataFrame,
    equity: float,
    units: Dict[str, float],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for paths, value in variant.get("overrides") or []:
        if isinstance(paths[0], tuple):
            for path in paths:
                _set_nested(cfg, path, value)
        else:
            _set_nested(cfg, paths, value)

    limits = resolve_multileg_sim_limits(cfg)
    fuse_mode = str(variant.get("fuse_mode") or "hard")
    sim_kwargs = {
        k: limits[k]
        for k in (
            "max_drawdown_pct",
            "daily_loss_limit_pct",
            "max_gross_notional_pct",
            "max_net_notional_pct",
            "max_symbol_gross_notional_pct",
            "max_symbol_net_notional_pct",
            "max_gross_leverage",
        )
        if limits.get(k) is not None
    }
    sim_kwargs["fuse_mode"] = fuse_mode

    u_chop = units.get("chop_grid", 556.0)
    u_trend = units.get("trend_scalp", 556.0)
    unit_by = {"chop_grid": u_chop, "trend_scalp": u_trend}

    m = simulate_account_with_constitution(
        trades,
        equity=equity,
        unit_notional=0.0,
        unit_by_strategy=unit_by,
        **sim_kwargs,
    )
    row = {
        "variant": variant["variant"],
        "label": variant.get("label", variant["variant"]),
        "fuse_mode": fuse_mode,
        "max_dd": limits.get("max_drawdown_pct"),
        "net_cap": limits.get("max_symbol_net_notional_pct"),
        "daily_loss": limits.get("daily_loss_limit_pct"),
        "ret_pct": round(float(m["ret_pct"]), 2),
        "max_dd_pct": round(float(m["max_dd_pct"]), 2),
        "max_dd_peak_pct": round(float(m.get("max_dd_peak_pct") or 0.0), 2),
        "peak_gross_pct": round(float(m["peak_gross_pct"]), 1),
        "halted": bool(m.get("halted")),
        "halted_day": m.get("halted_day") or "",
        "halted_reason": m.get("halted_reason") or "",
        "n_trades": int(m.get("n_trades") or 0),
        "n_rejected": int(m.get("n_rejected") or 0),
        "n_reject_daily": int(m.get("n_reject_daily") or 0),
        "n_reject_net": int(m.get("n_reject_net") or 0),
        "n_reject_max_dd": int(m.get("n_reject_max_dd") or 0),
        "n_derated": int(m.get("n_derated") or 0),
    }
    return row


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    headers = [
        "variant",
        "max_dd",
        "net_cap",
        "fuse",
        "ret%",
        "DD%(init)",
        "DD%(peak)",
        "halted",
        "halt_day",
        "trades",
        "rejected",
    ]
    lines = [
        "# Multi-leg constitution matrix",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["variant"]),
                    f"{float(r['max_dd'] or 0):.0%}" if r.get("max_dd") else "-",
                    f"{float(r['net_cap'] or 0):.2f}" if r.get("net_cap") else "-",
                    str(r["fuse_mode"]),
                    f"{r['ret_pct']:+.1f}",
                    f"{r['max_dd_pct']:.1f}",
                    f"{r['max_dd_peak_pct']:.1f}",
                    "yes" if r["halted"] else "no",
                    str(r["halted_day"] or "-"),
                    str(r["n_trades"]),
                    str(r["n_rejected"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--constitution-yaml",
        type=Path,
        default=PROJECT_ROOT / "live/highcap/config/constitution/constitution.yaml",
    )
    ap.add_argument("--chop-root", type=Path, required=True)
    ap.add_argument("--trend-root", type=Path, required=True)
    ap.add_argument(
        "--segments",
        nargs="+",
        default=[
            "bear_2022",
            "bull_2023_2024",
            "recent_range_to_bear",
            "recent_6m_oos",
        ],
    )
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument(
        "--strategies-root",
        default="live/highcap/config/strategies",
    )
    ap.add_argument("--max-concurrent-multi-leg-symbols", type=int, default=6)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "config/experiments/20260613_multileg_sizing_validate/quick_scan",
    )
    args = ap.parse_args()

    base_cfg = load_constitution_dict(str(args.constitution_yaml))
    if not base_cfg:
        raise SystemExit(f"empty constitution: {args.constitution_yaml}")

    ml = multi_leg_section(base_cfg)
    sr = Path(args.strategies_root)
    if not sr.is_absolute():
        sr = PROJECT_ROOT / sr
    units = resolve_multi_leg_unit_notionals(
        ml,
        equity_usdt=float(args.equity),
        chop_grid_execution_path=sr / "chop_grid" / "archetypes" / "execution.yaml",
        trend_scalp_execution_path=sr / "trend_scalp" / "archetypes" / "execution.yaml",
    )

    trades = _load_combined_trades(
        chop_root=args.chop_root,
        trend_root=args.trend_root,
        segments=args.segments,
        max_concurrent=int(args.max_concurrent_multi_leg_symbols),
    )
    if trades.empty:
        raise SystemExit("no trades after segment gates")

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for spec in _matrix_variants():
        row = _run_variant(
            base_cfg=base_cfg,
            variant=spec,
            trades=trades,
            equity=float(args.equity),
            units=units,
        )
        rows.append(row)
        halt = "HALT" if row["halted"] else "ok"
        print(
            f"{row['variant']:<28} ret={row['ret_pct']:+.1f}% "
            f"DD={row['max_dd_pct']:.1f}%/{row['max_dd_peak_pct']:.1f}% "
            f"{halt} trades={row['n_trades']} rej={row['n_rejected']} "
            f"(daily={row['n_reject_daily']} net={row['n_reject_net']})"
        )

    csv_path = out_dir / "constitution_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "constitution_matrix.md"
    _write_markdown(rows, md_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
