#!/usr/bin/env python3
"""
从固定训练 backtest 的 results.json 中按月份汇总，生成 monthly_results.json。

用于对比长时间段 vs 短时间段在各月的表现（是否被 regime shift 影响）。
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def build_monthly(results_path: Path) -> list:
    """从 results.json 的 backtest.debug.trades 按月份汇总 return_pct、trades。"""
    if not results_path.exists():
        return []
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    backtest = data.get("backtest") or {}
    debug = backtest.get("debug") or {}
    trades = debug.get("trades") or []
    if not trades:
        return []

    by_month = defaultdict(lambda: {"return_sum": 0.0, "trades": 0})
    for t in trades:
        entry_ts = t.get("Entry Timestamp") or t.get("entry_timestamp") or ""
        ret = t.get("Return") or t.get("return") or 0.0
        if isinstance(ret, (list, tuple)):
            ret = float(ret[0]) if ret else 0.0
        else:
            ret = float(ret)
        if entry_ts:
            # "2025-05-12 00:00:00" -> "2025-05"
            month = entry_ts[:7]
            by_month[month]["return_sum"] += ret
            by_month[month]["trades"] += 1

    out = []
    for month in sorted(by_month.keys()):
        r = by_month[month]
        # return_pct: 按月累计收益率（小数转百分比）
        return_pct = r["return_sum"] * 100.0
        out.append(
            {
                "month": month,
                "return_pct": round(return_pct, 4),
                "trades": r["trades"],
            }
        )
    return out


def main():
    # 可指定目录，默认扫 fixed_long 和 fixed_short 下各策略
    if len(sys.argv) > 1:
        dirs = [Path(p) for p in sys.argv[1:]]
    else:
        dirs = []
        for base in ["results/fixed_long", "results/fixed_short"]:
            base_path = ROOT / base
            if not base_path.exists():
                continue
            for strategy in base_path.iterdir():
                if not strategy.is_dir():
                    continue
                # results/fixed_long/<strategy>/<strategy>/results.json
                sub = strategy / strategy.name
                if sub.is_dir():
                    dirs.append(sub)

    for artifact_dir in dirs:
        results_path = artifact_dir / "results.json"
        monthly = build_monthly(results_path)
        out_path = artifact_dir / "monthly_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(monthly, f, indent=2, ensure_ascii=False)
        print(f"  {out_path} ({len(monthly)} months)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
