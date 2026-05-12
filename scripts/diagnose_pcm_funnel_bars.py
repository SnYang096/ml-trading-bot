#!/usr/bin/env python3
"""从 event_backtest_*.json 的 funnel_per_bar 筛 bar，并用 PCM 诊断字段归类。

新字段（需用当前仓库 event_backtest 重跑后才有）：
  pcm_n_candidates, pcm_n_accepted, pcm_drop_direction_policy, pcm_drop_family_conflict,
  pcm_drop_daily_limit, pcm_drop_slot

用法示例：
  python scripts/diagnose_pcm_funnel_bars.py \\
    results/me/research_roll.features_on/_rolling_sim/<run>/fast_month_2024-11/me/event_backtest_me.json \\
    --symbol ADAUSDT --strategy me --month 2024-11 --structural-only

  # 仅列出「结构全过」的 bar（prefilter∧gate∧entry∧direction，且 pcm_direction_filter 未显式 False）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _truthy(v: Any) -> bool:
    return v is True


def _structural_pass(row: Dict[str, Any]) -> bool:
    if row.get("pcm_direction_filter") is False:
        return False
    if not _truthy(row.get("prefilter")):
        return False
    if not _truthy(row.get("direction")):
        return False
    if row.get("gate") is False:
        return False
    if row.get("entry_filter") is False:
        return False
    return True


def _pcm_case(row: Dict[str, Any]) -> str:
    if "pcm_n_candidates" not in row:
        return "no_pcm_trace_rerun_event_backtest"
    n_c = int(row.get("pcm_n_candidates") or 0)
    n_a = int(row.get("pcm_n_accepted") or 0)
    if n_c <= 0:
        return "no_candidate_intents_strategy_or_pcm_prefilter"
    if n_a > 0:
        return f"pcm_accepted_{n_a}"
    parts: List[str] = []
    for k, label in (
        ("pcm_drop_direction_policy", "direction_policy"),
        ("pcm_drop_family_conflict", "family_conflict"),
        ("pcm_drop_daily_limit", "daily_throttle"),
        ("pcm_drop_slot", "slot_full"),
    ):
        v = int(row.get(k) or 0)
        if v:
            parts.append(f"{label}×{v}")
    return "pcm_rejected:" + (",".join(parts) if parts else "unknown_zero_drops")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path", type=Path, help="event_backtest_*.json")
    ap.add_argument("--symbol", default="", help="如 ADAUSDT")
    ap.add_argument("--strategy", default="", help="如 me（小写）")
    ap.add_argument("--month", default="", help="如 2024-11，匹配 timestamp 字符串前缀")
    ap.add_argument(
        "--structural-only",
        action="store_true",
        help="只保留结构全过的行（与漏斗讨论口径一致）",
    )
    ap.add_argument("--limit", type=int, default=200, help="最多打印行数")
    args = ap.parse_args()

    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = list(data.get("funnel_per_bar") or [])
    sym_u = args.symbol.strip().upper()
    strat_lc = args.strategy.strip().lower()
    month = args.month.strip()

    out: List[Dict[str, Any]] = []
    for r in rows:
        if sym_u and str(r.get("symbol", "")).upper() != sym_u:
            continue
        if strat_lc and str(r.get("strategy", "")).lower() != strat_lc:
            continue
        ts = str(r.get("timestamp", ""))
        if month and month not in ts[:7] and month not in ts:
            continue
        if args.structural_only and not _structural_pass(r):
            continue
        out.append(r)

    print(f"path={args.json_path}")
    print(f"matched_rows={len(out)}")
    if not out:
        return

    from collections import Counter

    ctr = Counter(_pcm_case(r) for r in out)
    print("case_counts:")
    for k, v in ctr.most_common():
        print(f"  {k}: {v}")

    print("\n--- sample (first N) ---")
    for r in out[: max(0, args.limit)]:
        case = _pcm_case(r)
        print(
            f"{r.get('timestamp')} {r.get('symbol')} {r.get('strategy')} "
            f"cand={r.get('pcm_n_candidates','?')} acc={r.get('pcm_n_accepted','?')} "
            f"case={case}"
        )


if __name__ == "__main__":
    main()
