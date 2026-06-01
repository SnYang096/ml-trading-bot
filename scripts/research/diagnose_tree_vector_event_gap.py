#!/usr/bin/env python3
"""Diagnose fast_scalp tree vectorbt vs event_backtest trade / funnel gaps."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_vector_event_consistency import (  # noqa: E402
    _check_consistency,
    _compute_stats,
    _print_report,
)


def _hour_distribution(df: pd.DataFrame, col: str = "entry_time") -> dict[str, int]:
    if df.empty or col not in df.columns:
        return {}
    ts = pd.to_datetime(df[col], utc=True)
    counts = ts.dt.hour.value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}


def _distribution_table(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {}
    vc = series.value_counts(normalize=True)
    return {str(k): round(float(v), 4) for k, v in vc.items()}


def _load_funnel(path: Path | None) -> dict[str, int]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    funnel = data.get("funnel") or {}
    return {str(k): int(v) for k, v in funnel.items()}


def diagnose(
    *,
    vector_csv: Path,
    event_csv: Path,
    output_dir: Path,
    event_capital_json: Path | None = None,
    max_trade_diff_pct: float = 0.15,
) -> dict[str, Any]:
    v_df = pd.read_csv(vector_csv)
    e_df = pd.read_csv(event_csv)
    v_stats = _compute_stats(v_df)
    e_stats = _compute_stats(e_df)
    checks = _check_consistency(
        v_stats,
        e_stats,
        thresholds={"max_trade_diff_pct": max_trade_diff_pct},
    )

    report: dict[str, Any] = {
        "vector_stats": v_stats,
        "event_stats": e_stats,
        "checks": checks,
        "by_symbol_trades": {
            "vector": v_stats.get("per_symbol_trades", {}),
            "event": e_stats.get("per_symbol_trades", {}),
        },
        "exit_reasons": {
            "vector": v_stats.get("exit_reasons", {}),
            "event": e_stats.get("exit_reasons", {}),
        },
        "entry_hour_distribution": {
            "vector": _hour_distribution(v_df),
            "event": _hour_distribution(e_df),
        },
        "event_exit_reason_counts": dict(
            Counter(e_df.get("exit_reason", pd.Series(dtype=str)))
        ),
        "vector_exit_reason_counts": dict(
            Counter(v_df.get("exit_reason", pd.Series(dtype=str)))
        ),
    }

    funnel = _load_funnel(event_capital_json)
    if funnel:
        report["event_funnel"] = funnel
        total = max(int(funnel.get("total_signals_checked", 0)), 1)
        report["event_funnel_rates"] = {
            k: round(v / total, 4)
            for k, v in funnel.items()
            if k.startswith("reject_") or k.startswith("signals_")
        }
        vector_entries = len(v_df)
        report["entry_bars"] = {
            "vector_trades": vector_entries,
            "event_trades": len(e_df),
            "event_reject_regime": funnel.get("reject_regime", 0),
            "event_reject_no_direction": funnel.get("reject_no_direction", 0),
            "event_reject_open_duplicate_archetype": funnel.get(
                "reject_open_duplicate_archetype", 0
            ),
        }

    sign_match = np.sign(v_stats.get("mean_r", 0)) == np.sign(
        e_stats.get("mean_r", 0)
    ) or (v_stats.get("mean_r", 0) == 0 and e_stats.get("mean_r", 0) == 0)
    report["acceptance"] = {
        "return_sign_match": bool(sign_match),
        "trade_count_pass": bool(checks.get("trade_count", {}).get("pass", False)),
        "trade_diff_pct": checks.get("trade_count", {}).get("diff_pct"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vector_event_gap_report.json"
    md_path = output_dir / "vector_event_gap_report.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    md_lines = [
        "# vector ↔ event gap diagnosis",
        "",
        f"- vector trades: `{vector_csv}` ({v_stats['n_trades']})",
        f"- event trades: `{event_csv}` ({e_stats['n_trades']})",
        "",
        "## Core metrics",
        "",
        f"| metric | vector | event | pass |",
        f"|--------|-------:|------:|:----:|",
        f"| trades | {v_stats['n_trades']} | {e_stats['n_trades']} | {checks['trade_count']['pass']} |",
        f"| Sharpe | {v_stats['sharpe']:.3f} | {e_stats['sharpe']:.3f} | {checks['sharpe']['pass']} |",
        f"| mean_r | {v_stats['mean_r']:.4f} | {e_stats['mean_r']:.4f} | {checks['mean_r']['pass']} |",
        "",
        "## Acceptance (Phase 1)",
        "",
        f"- return sign match: **{report['acceptance']['return_sign_match']}**",
        f"- trade count ≤ {max_trade_diff_pct:.0%}: **{report['acceptance']['trade_count_pass']}** "
        f"(diff {report['acceptance']['trade_diff_pct']:.1%})",
        "",
    ]
    if funnel:
        md_lines.extend(
            [
                "## Event funnel (top rejects)",
                "",
            ]
        )
        for k in sorted(funnel.keys()):
            if k.startswith("reject_") and funnel[k]:
                md_lines.append(f"- {k}: {funnel[k]}")
        md_lines.append("")

    md_lines.extend(
        [
            "## Exit reason mix (event)",
            "",
            str(report["event_exit_reason_counts"]),
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    fail_count = _print_report(checks)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"Acceptance: sign_match={report['acceptance']['return_sign_match']} "
        f"trade_pass={report['acceptance']['trade_count_pass']}"
    )
    report["_fail_count"] = fail_count
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vector-csv", required=True)
    ap.add_argument("--event-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--event-capital-json", default=None)
    ap.add_argument("--max-trade-diff-pct", type=float, default=0.15)
    args = ap.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    cap = Path(args.event_capital_json) if args.event_capital_json else None
    if cap and not cap.is_absolute():
        cap = (PROJECT_ROOT / cap).resolve()

    vec = Path(args.vector_csv)
    if not vec.is_absolute():
        vec = (PROJECT_ROOT / vec).resolve()
    evt = Path(args.event_csv)
    if not evt.is_absolute():
        evt = (PROJECT_ROOT / evt).resolve()

    report = diagnose(
        vector_csv=vec,
        event_csv=evt,
        output_dir=out,
        event_capital_json=cap,
        max_trade_diff_pct=args.max_trade_diff_pct,
    )
    ok = (
        report["acceptance"]["return_sign_match"]
        and report["acceptance"]["trade_count_pass"]
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
