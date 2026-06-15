#!/usr/bin/env python3
"""Sweep multi-leg constitution limits via ``backtest_multileg_timeline`` (LiveEngine).

Loads ``live/highcap/config/constitution/constitution.yaml``, deep-copies, injects a
single override path, and runs joint chop_grid + trend_scalp timeline backtest.

Usage
-----
    python scripts/sweep_multileg_constitution_matrix.py \\
        --start 2025-12-01 --end 2026-05-31 \\
        --chop-config config/experiments/20260613_multileg_sizing_validate/variants/chop_prod/meta.yaml \\
        --trend-config config/experiments/20260613_multileg_sizing_validate/variants/trend_prod/meta.yaml

Output: CSV + markdown table under ``--output-dir``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_multileg_timeline import run_timeline_backtest

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


def _matrix_variants() -> List[Dict[str, Any]]:
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
        },
        {
            "variant": "max_dd_half",
            "label": "max_dd 20%→10% (both paths)",
            "overrides": [(dd_paths, 0.10)],
        },
        {
            "variant": "max_dd_15",
            "label": "max_dd 20%→15%",
            "overrides": [(dd_paths, 0.15)],
        },
        {
            "variant": "net_cap_100",
            "label": "max_symbol_net 1.80→1.00",
            "overrides": [(net_path, 1.00)],
        },
        {
            "variant": "net_cap_120",
            "label": "max_symbol_net 1.80→1.20",
            "overrides": [(net_path, 1.20)],
        },
    ]


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    headers = [
        "variant",
        "max_dd",
        "net_cap",
        "ret%",
        "DD%(peak)",
        "halted",
        "trades_ok",
        "trades_rej",
    ]
    lines = [
        "# Multi-leg constitution matrix (timeline)",
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
                    f"{r['return_pct']:+.1f}",
                    f"{r['max_drawdown_pct']:.1f}",
                    "yes" if r["halted"] else "no",
                    str(r["trades_ok"]),
                    str(r["trades_rej"]),
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
    ap.add_argument("--start", default="2025-12-01")
    ap.add_argument("--end", default="2026-05-31")
    ap.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
    )
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--chop-config", type=Path, required=True)
    ap.add_argument("--trend-config", type=Path, required=True)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "config/experiments/20260613_multileg_sizing_validate/quick_scan",
    )
    ap.add_argument("--load-preload", type=Path, default=None)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(args.constitution_yaml.read_text(encoding="utf-8")) or {}
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start).tz_localize("UTC")
    end = pd.Timestamp(args.end).tz_localize("UTC")

    chop_cfg = args.chop_config
    trend_cfg = args.trend_config
    if not chop_cfg.is_absolute():
        chop_cfg = PROJECT_ROOT / chop_cfg
    if not trend_cfg.is_absolute():
        trend_cfg = PROJECT_ROOT / trend_cfg

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    preload = args.load_preload
    for i, spec in enumerate(_matrix_variants()):
        cfg = copy.deepcopy(base_cfg)
        for paths, value in spec.get("overrides") or []:
            if isinstance(paths[0], tuple):
                for path in paths:
                    _set_nested(cfg, path, value)
            else:
                _set_nested(cfg, paths, value)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(cfg, tmp)
            const_path = Path(tmp.name)

        try:
            acct, _meta = run_timeline_backtest(
                start=start,
                end=end,
                symbols=symbols,
                equity=float(args.equity),
                chop_config=chop_cfg,
                trend_config=trend_cfg,
                constitution_yaml=const_path,
                load_preload=preload if preload and preload.exists() else None,
                save_preload=preload if i == 0 and preload else None,
                clean_state=True,
                progress=True,
            )
        finally:
            const_path.unlink(missing_ok=True)

        summary = acct.to_summary()
        ks = cfg.get("kill_switch", {})
        ml = cfg.get("multi_leg", {})
        rs = ml.get("risk_limits", {})
        row = {
            "variant": spec["variant"],
            "label": spec.get("label", spec["variant"]),
            "max_dd": ks.get("max_dd"),
            "net_cap": rs.get("max_symbol_net_notional_pct"),
            **summary,
        }
        rows.append(row)
        halt = "HALT" if row["halted"] else "ok"
        print(
            f"{row['variant']:<20} ret={row['return_pct']:+.1f}% "
            f"DD={row['max_drawdown_pct']:.1f}% {halt} "
            f"trades={row['trades_ok']}/{row['trades_rej']}"
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
