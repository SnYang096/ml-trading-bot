#!/usr/bin/env python3
"""Regime drift monitor — 监测 regime 慢变量是否漂出上次 plateau 范围。

每周/每日运行一次（看监控节奏）。读取最近 N 个交易日的特征分布，与
``archetypes/regime.yaml`` 的 ``last_calibration.plateaus`` 比较，
任一 feature 的当前分位 (P25/P50/P75) 漂出 plateau 区间则 ALERT。

用途：
    - 触发 Tier-0 季度校准提前到来；
    - 给 PCM 一个明确的 ``regime_alert=true`` 信号，可暂停新进单。

输出：JSON 报告 + 简短文本摘要；非零退出码表示有 ALERT。

调用示例：
    python scripts/regime_drift_monitor.py \\
        --strategies bpc,tpc,me,srb \\
        --window-parquet results/<...>/recent_features.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plateau_stability import PlateauRange, plateau_range_from_dict
from src.time_series_model.regime.threshold_calibrator import load_regime_yaml


def _percentile(series: pd.Series, q: float) -> Optional[float]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(s) < 5:
        return None
    return float(s.quantile(q))


def evaluate_strategy_drift(
    *,
    strategy: str,
    regime_yaml: Dict[str, Any],
    window_df: pd.DataFrame,
    drift_quantile: float = 0.5,
    tail_band_q: tuple[float, float] = (0.25, 0.75),
) -> Dict[str, Any]:
    """检查 last_calibration.plateaus 下的 feature 当前分位是否漂出 plateau。"""
    plateaus = (regime_yaml.get("last_calibration") or {}).get("plateaus") or []
    items: List[Dict[str, Any]] = []
    any_alert = False
    for entry in plateaus:
        if not isinstance(entry, dict):
            continue
        feature = str(entry.get("feature") or "")
        operator = str(entry.get("operator") or "")
        plateau: Optional[PlateauRange] = plateau_range_from_dict(entry.get("plateau"))
        if not feature or plateau is None:
            continue
        if feature not in window_df.columns:
            items.append(
                {
                    "feature": feature,
                    "operator": operator,
                    "status": "MISSING_FEATURE",
                    "plateau": {"start": plateau.start, "end": plateau.end},
                }
            )
            any_alert = True
            continue
        col = window_df[feature]
        p_low = _percentile(col, tail_band_q[0])
        p_mid = _percentile(col, drift_quantile)
        p_high = _percentile(col, tail_band_q[1])
        if p_mid is None:
            items.append(
                {
                    "feature": feature,
                    "operator": operator,
                    "status": "INSUFFICIENT_DATA",
                    "plateau": {"start": plateau.start, "end": plateau.end},
                }
            )
            any_alert = True
            continue
        in_band = plateau.start <= p_mid <= plateau.end
        items.append(
            {
                "feature": feature,
                "operator": operator,
                "plateau": {"start": plateau.start, "end": plateau.end},
                "window_p25": p_low,
                "window_p50": p_mid,
                "window_p75": p_high,
                "status": "OK" if in_band else "DRIFT",
            }
        )
        if not in_band:
            any_alert = True
    return {
        "strategy": strategy,
        "any_alert": any_alert,
        "items": items,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Regime drift monitor")
    p.add_argument("--strategies", default="bpc,me,tpc,srb")
    p.add_argument(
        "--window-parquet",
        required=True,
        help="parquet of recent feature snapshots (last N bars).",
    )
    p.add_argument("--strategies-root", default="config/strategies")
    p.add_argument(
        "--out-dir",
        default="results/regime_drift_monitor",
    )
    p.add_argument("--drift-quantile", type=float, default=0.5)
    p.add_argument(
        "--emit-rd-loop-suggestions",
        action="store_true",
        help="On ALERT, write rd_loop yaml snippets under results/drift_suggestions/",
    )
    args = p.parse_args()

    pq = Path(args.window_parquet)
    if not pq.is_absolute():
        pq = (PROJECT_ROOT / pq).resolve()
    if not pq.exists():
        print(f"ERROR: window parquet not found: {pq}", file=sys.stderr)
        return 3
    window_df = pd.read_parquet(pq)

    strategies_root = Path(args.strategies_root)
    if not strategies_root.is_absolute():
        strategies_root = (PROJECT_ROOT / strategies_root).resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.out_dir / ts).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report: List[Dict[str, Any]] = []
    any_alert = False
    for s in [x.strip() for x in args.strategies.split(",") if x.strip()]:
        regime_yaml = load_regime_yaml(
            strategies_root / s / "archetypes" / "regime.yaml"
        )
        if not regime_yaml:
            report.append(
                {
                    "strategy": s,
                    "any_alert": True,
                    "items": [],
                    "skipped": "no regime.yaml",
                }
            )
            any_alert = True
            continue
        r = evaluate_strategy_drift(
            strategy=s,
            regime_yaml=regime_yaml,
            window_df=window_df,
            drift_quantile=float(args.drift_quantile),
        )
        report.append(r)
        if r["any_alert"]:
            any_alert = True

    out_json = out_dir / "drift_report.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "window_parquet": str(pq),
                "any_alert": any_alert,
                "report": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out_json}")
    for r in report:
        flag = "ALERT" if r["any_alert"] else "OK"
        print(f"  {r['strategy']:>5}: {flag} ({len(r['items'])} feature(s))")

    if args.emit_rd_loop_suggestions and any_alert:
        from scripts.research.drift_suggestions import write_drift_suggestions

        sug_dir = (PROJECT_ROOT / "results/drift_suggestions" / ts).resolve()
        written = write_drift_suggestions(
            report,
            features_parquet=str(pq),
            out_dir=sug_dir,
        )
        for pth in written:
            print(f"  suggestion: {pth}")

    return 1 if any_alert else 0


if __name__ == "__main__":
    raise SystemExit(main())
