#!/usr/bin/env python3
"""Monitor multi-leg replay/live health and emit drift verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.pipeline.config import load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _latest_rolling_root(history_dir: Path) -> Path:
    rs = history_dir / "_rolling_sim"
    if not rs.exists():
        raise FileNotFoundError(f"rolling root missing: {rs}")
    runs = [d for d in rs.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"no rolling runs under: {rs}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _load_monthly_strategy_rows(
    run_root: Path, strategies: List[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ledger_path = run_root / "monthly_ledger.jsonl"
    if not ledger_path.exists():
        return rows
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        month = str(row.get("month", "") or "")
        run_month_root = Path(str(row.get("run_root", "") or ""))
        if not month or not run_month_root.exists():
            continue
        for s in strategies:
            p = run_month_root / s / "multileg_summary.json"
            if not p.exists():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(
                {
                    "month": month,
                    "strategy": s,
                    "metrics": obj.get("metrics", {}) or {},
                    "run_root": str(run_month_root),
                }
            )
    rows.sort(key=lambda r: str(r.get("month", "")))
    return rows


def _feature_hash(run_root: Path, strategy: str) -> str:
    p = run_root / "strategies" / strategy / "features.yaml"
    if not p.exists():
        return ""
    raw = p.read_bytes()
    return hashlib.sha1(raw).hexdigest()


def _split_baseline_recent(vals: List[float]) -> Tuple[List[float], List[float]]:
    if len(vals) <= 1:
        return vals, vals
    mid = max(1, len(vals) // 2)
    return vals[:mid], vals[mid:]


def _mean(vs: List[float]) -> float:
    if not vs:
        return 0.0
    return float(sum(vs) / len(vs))


def _render_html(path: Path, report: Dict[str, Any]) -> None:
    rows = report.get("strategy_rows", []) or []
    lines = [
        "<!doctype html>",
        "<html><meta charset='utf-8'><title>multi_leg_monitor</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:20px;max-width:1100px}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px}"
        "th{background:#f4f4f4;text-align:left}.warn{color:#b26a00}.bad{color:#a80c0c}.ok{color:#0a7a0a}</style>",
        "<h1>Multi-leg Monitor Report</h1>",
        f"<p>decision: <b>{report.get('decision')}</b></p>",
        "<h2>Signals</h2><pre>",
        json.dumps(report.get("signals", {}), ensure_ascii=False, indent=2),
        "</pre>",
        "<h2>Strategy Rows</h2>",
        "<table><thead><tr><th>month</th><th>strategy</th><th>trades</th><th>total_r</th><th>forced</th><th>risk_stop</th><th>max_dd_r</th></tr></thead><tbody>",
    ]
    for r in rows:
        m = r.get("metrics", {})
        lines.append(
            "<tr>"
            f"<td>{r.get('month')}</td>"
            f"<td>{r.get('strategy')}</td>"
            f"<td>{m.get('n_trades', 0)}</td>"
            f"<td>{float(m.get('total_r', 0.0) or 0.0):.4f}</td>"
            f"<td>{float(m.get('forced_rate', 0.0) or 0.0):.2%}</td>"
            f"<td>{float(m.get('risk_stop_rate', m.get('near_stop_rate', 0.0)) or 0.0):.2%}</td>"
            f"<td>{float(m.get('max_drawdown_r', 0.0) or 0.0):.4f}</td>"
            "</tr>"
        )
    lines.append("</tbody></table></html>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Monitor multi-leg drift and health.")
    p.add_argument("--config", default="config/pipelines/multileg_orchestrate_2h.yaml")
    p.add_argument("--run-id", default="", help="optional rolling_sim run id")
    p.add_argument("--lookback-months", type=int, default=6)
    p.add_argument("--out-json", default="")
    p.add_argument("--out-html", default="")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_pipeline_config(cfg_path)

    history_dir = PROJECT_ROOT / str(
        (cfg.get("output") or {}).get("history_dir", "") or ""
    )
    run_root = (
        (history_dir / "_rolling_sim" / str(args.run_id).strip())
        if str(args.run_id).strip()
        else _latest_rolling_root(history_dir)
    )
    if not run_root.exists():
        raise FileNotFoundError(f"run root not found: {run_root}")

    strategies = []
    for name, scfg in (cfg.get("strategies") or {}).items():
        st = str((scfg or {}).get("strategy_type", "") or "").strip().lower()
        if st in {"grid", "dual_add_trend"}:
            strategies.append(name)

    rows = _load_monthly_strategy_rows(run_root, strategies)
    if int(args.lookback_months) > 0 and len(rows) > int(args.lookback_months) * max(
        len(strategies), 1
    ):
        rows = rows[-int(args.lookback_months) * max(len(strategies), 1) :]

    monitor_cfg = cfg.get("multi_leg_monitor", {}) or {}
    max_drawdown_r = float(monitor_cfg.get("max_drawdown_r", 0.10) or 0.10)
    forced_delta = float(monitor_cfg.get("drift_forced_rate_delta", 0.10) or 0.10)
    trade_ratio_floor = float(monitor_cfg.get("drift_trade_ratio_floor", 0.60) or 0.60)
    total_r_floor = float(monitor_cfg.get("threshold_total_r_floor", -0.10) or -0.10)

    forced_vals = [
        float((r.get("metrics") or {}).get("forced_rate", 0.0) or 0.0) for r in rows
    ]
    trade_vals = [
        float((r.get("metrics") or {}).get("n_trades", 0) or 0.0) for r in rows
    ]
    r_vals = [float((r.get("metrics") or {}).get("total_r", 0.0) or 0.0) for r in rows]
    dd_vals = [
        float((r.get("metrics") or {}).get("max_drawdown_r", 0.0) or 0.0) for r in rows
    ]

    b_forced, r_forced = _split_baseline_recent(forced_vals)
    b_trades, r_trades = _split_baseline_recent(trade_vals)
    b_r, r_r = _split_baseline_recent(r_vals)

    forced_shift = (_mean(r_forced) - _mean(b_forced)) > forced_delta
    trade_shift = _mean(r_trades) < (
        _mean(b_trades) * trade_ratio_floor if _mean(b_trades) > 0 else 0.0
    )
    threshold_shift = _mean(r_r) < total_r_floor
    risk_offline = (min(dd_vals) if dd_vals else 0.0) < -max_drawdown_r

    # Feature shift proxy: features.yaml hash changed between first/last month snapshots.
    feature_shift = False
    for s in strategies:
        month_roots = [
            Path(str(r.get("run_root", ""))) for r in rows if r.get("strategy") == s
        ]
        if len(month_roots) < 2:
            continue
        first_h = _feature_hash(month_roots[0], s)
        last_h = _feature_hash(month_roots[-1], s)
        if first_h and last_h and first_h != last_h:
            feature_shift = True
            break

    if risk_offline:
        decision = "OFFLINE"
    elif feature_shift:
        decision = "FEATURE_REVIEW"
    elif threshold_shift:
        decision = "RETUNE_THRESHOLDS"
    elif forced_shift or trade_shift:
        decision = "WATCH"
    else:
        decision = "OK"

    report = {
        "decision": decision,
        "run_root": str(run_root),
        "config": str(cfg_path),
        "signals": {
            "forced_shift": forced_shift,
            "trade_shift": trade_shift,
            "threshold_shift": threshold_shift,
            "feature_shift": feature_shift,
            "risk_offline": risk_offline,
        },
        "summary": {
            "baseline_forced_rate": _mean(b_forced),
            "recent_forced_rate": _mean(r_forced),
            "baseline_trades": _mean(b_trades),
            "recent_trades": _mean(r_trades),
            "baseline_total_r": _mean(b_r),
            "recent_total_r": _mean(r_r),
            "worst_drawdown_r": min(dd_vals) if dd_vals else 0.0,
        },
        "strategy_rows": rows,
    }

    out_json = (
        Path(args.out_json)
        if str(args.out_json).strip()
        else run_root / "multi_leg_health_report.json"
    )
    out_html = (
        Path(args.out_html)
        if str(args.out_html).strip()
        else run_root / "multi_leg_health_report.html"
    )
    if not out_json.is_absolute():
        out_json = PROJECT_ROOT / out_json
    if not out_html.is_absolute():
        out_html = PROJECT_ROOT / out_html
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _render_html(out_html, report)
    print(
        json.dumps(
            {"decision": decision, "json": str(out_json), "html": str(out_html)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
