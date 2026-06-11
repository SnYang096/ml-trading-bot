#!/usr/bin/env python3
"""C-system regime watchdog — extensions.multileg entry pass_rate vs baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.regime_health import evaluate_multileg_entry_health
from src.time_series_model.regime.threshold_calibrator import load_regime_yaml


def run_watchdog_c(args: argparse.Namespace) -> int:
    pq = Path(args.window_parquet)
    if not pq.is_absolute():
        pq = (PROJECT_ROOT / pq).resolve()
    if not pq.exists():
        print(f"ERROR: window parquet not found: {pq}", file=sys.stderr)
        return 3
    window_df = pd.read_parquet(pq)

    baseline: Dict[str, Any] = {}
    if args.baseline_json:
        bp = Path(args.baseline_json)
        if not bp.is_absolute():
            bp = (PROJECT_ROOT / bp).resolve()
        if bp.is_file():
            baseline = json.loads(bp.read_text(encoding="utf-8"))

    strategies_root = Path(args.strategies_root)
    if not strategies_root.is_absolute():
        strategies_root = (PROJECT_ROOT / strategies_root).resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir / ts).resolve()
    else:
        out_dir = (out_dir / ts).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    any_alert = False
    for s in [x.strip() for x in args.strategies.split(",") if x.strip()]:
        regime_path = strategies_root / s / "archetypes" / "regime.yaml"
        regime_raw = load_regime_yaml(regime_path) if regime_path.is_file() else {}
        if not regime_raw:
            reports.append(
                {
                    "strategy": s,
                    "any_alert": False,
                    "status": "SKIPPED",
                    "skipped": f"regime.yaml not found at {regime_path}",
                    "items": [],
                }
            )
            continue
        base_for_s = (baseline or {}).get(s) or {}
        result = evaluate_multileg_entry_health(
            strategy=s,
            regime_yaml=regime_raw,
            window_df=window_df,
            baseline_entry=base_for_s if isinstance(base_for_s, dict) else None,
            pass_rate_tol=float(args.pass_rate_tol),
        )
        reports.append(result)
        any_alert = any_alert or bool(result.get("any_alert"))

    out_json = {
        "ts": ts,
        "window_parquet": str(pq),
        "any_alert": any_alert,
        "reports": reports,
    }
    (out_dir / "report.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines: List[str] = [
        f"regime_watchdog_c @ {ts}  window={pq.name}  alert={'YES' if any_alert else 'no'}"
    ]
    for r in reports:
        s = r.get("strategy", "?")
        if r.get("skipped"):
            lines.append(f"  [{s}] {r.get('status', 'SKIPPED')}: {r['skipped']}")
            continue
        item = (r.get("items") or [{}])[0]
        lines.append(
            f"  [{s}] pass_rate={item.get('current_pass_rate', 0):.1%}"
            f"  entry_min={item.get('entry_min')}"
            f"  feature={item.get('entry_feature')}"
        )
        for a in r.get("alerts") or []:
            lines.append(f"      ALERT: {a}")
    print("\n".join(lines))
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return 1 if any_alert else 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="C-system multileg entry pass_rate watchdog"
    )
    p.add_argument("--window-parquet", required=True)
    p.add_argument("--strategies", required=True, help="Comma-separated C slugs")
    p.add_argument(
        "--strategies-root",
        default="live/highcap/config/strategies",
    )
    p.add_argument(
        "--baseline-json",
        default="config/monitoring/regime_watchdog_baseline.json",
    )
    p.add_argument("--out-dir", default="results/regime_watchdog_c")
    p.add_argument("--pass-rate-tol", type=float, default=0.10)
    args = p.parse_args()
    return run_watchdog_c(args)


if __name__ == "__main__":
    raise SystemExit(main())
