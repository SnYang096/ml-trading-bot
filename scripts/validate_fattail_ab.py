#!/usr/bin/env python3
"""A/B 验证: fat-tail 新方案上线门槛检查.

用法:
  python scripts/validate_fattail_ab.py \
    --baseline reports/base.json \
    --candidate reports/new.json \
    --max-dd-increase 0.01 \
    --min-tail-improve 0.00 \
    --min-leverage-hit-improve 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_float(obj: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in obj and obj[k] is not None:
            try:
                return float(obj[k])
            except Exception:
                continue
    return float(default)


def _extract_metrics(
    payload: Dict[str, Any], leverage_target: float
) -> Dict[str, float]:
    kpis = payload.get("kpis") or {}
    metrics = payload.get("metrics") or {}
    summary = payload.get("summary") or {}

    tail = _pick_float(payload, "tail_contribution_rate", default=-1.0)
    if tail < 0:
        tail = _pick_float(kpis, "tail_contribution_rate", default=0.0)

    add_exp = _pick_float(payload, "add_expectancy", default=-1e9)
    if add_exp < -1e8:
        add_exp = _pick_float(kpis, "add_expectancy", "add_mean_r", default=0.0)

    mfe_capture = _pick_float(payload, "mfe_capture_ratio", default=-1.0)
    if mfe_capture < 0:
        mfe_capture = _pick_float(
            kpis, "mfe_capture_ratio", "MFE_capture", "mfe_capture", default=0.0
        )

    max_dd = _pick_float(
        payload,
        "max_drawdown",
        default=_pick_float(metrics, "max_dd", "max_drawdown", default=0.0),
    )

    lev_hit = _pick_float(payload, "leverage_hit_rate", default=-1.0)
    if lev_hit < 0:
        lev_hit = _pick_float(
            kpis, "leverage_hit_rate", "target_leverage_hit_rate", default=-1.0
        )

    if lev_hit < 0:
        trades = payload.get("trades") or []
        if isinstance(trades, list) and trades:
            hit = 0
            total = 0
            for t in trades:
                if not isinstance(t, dict):
                    continue
                lev = _pick_float(
                    t, "current_leverage", "realized_leverage", default=-1.0
                )
                if lev < 0:
                    continue
                total += 1
                if lev >= leverage_target:
                    hit += 1
            lev_hit = (hit / total) if total > 0 else 0.0
        else:
            lev_hit = _pick_float(summary, "leverage_hit_rate", default=0.0)

    return {
        "tail_contribution_rate": tail,
        "add_expectancy": add_exp,
        "mfe_capture": mfe_capture,
        "max_dd": max_dd,
        "leverage_hit_rate": lev_hit,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fat-tail A/B validation gate")
    ap.add_argument("--baseline", required=True, help="baseline result json")
    ap.add_argument("--candidate", required=True, help="candidate result json")
    ap.add_argument("--leverage-target", type=float, default=5.0)
    ap.add_argument("--max-dd-increase", type=float, default=0.01)
    ap.add_argument("--min-tail-improve", type=float, default=0.0)
    ap.add_argument("--min-add-expectancy-improve", type=float, default=0.0)
    ap.add_argument("--min-mfe-capture-improve", type=float, default=0.0)
    ap.add_argument("--min-leverage-hit-improve", type=float, default=0.03)
    args = ap.parse_args()

    base = _extract_metrics(_load_json(args.baseline), args.leverage_target)
    cand = _extract_metrics(_load_json(args.candidate), args.leverage_target)

    deltas = {k: cand[k] - base[k] for k in base}
    checks = {
        "tail_contribution_rate": deltas["tail_contribution_rate"]
        >= args.min_tail_improve,
        "add_expectancy": deltas["add_expectancy"] >= args.min_add_expectancy_improve,
        "mfe_capture": deltas["mfe_capture"] >= args.min_mfe_capture_improve,
        "max_dd": deltas["max_dd"] <= args.max_dd_increase,
        "leverage_hit_rate": deltas["leverage_hit_rate"]
        >= args.min_leverage_hit_improve,
    }

    print("=== Fat-tail A/B Validation ===")
    for k in [
        "tail_contribution_rate",
        "add_expectancy",
        "mfe_capture",
        "max_dd",
        "leverage_hit_rate",
    ]:
        print(
            f"{k:24s} base={base[k]:8.4f} cand={cand[k]:8.4f} delta={deltas[k]:+8.4f} "
            f"=> {'PASS' if checks[k] else 'FAIL'}"
        )

    all_pass = all(checks.values())
    print(f"\nGate result: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
