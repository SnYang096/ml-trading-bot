#!/usr/bin/env python3
"""Evaluate multi-leg deployment gate for one run directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from scripts.pipeline.config import load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _strategy_type(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("strategy_type", "") or "").strip().lower()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _grid_metrics_from_standalone(path: Path) -> Dict[str, Any]:
    obj = _read_json(path)
    metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}
    trade = metrics.get("trade_summary", {}) or {}
    segment = metrics.get("segment_summary", {}) or {}
    n_trades = int(trade.get("trades", 0) or 0)
    pnl = float(trade.get("sum_pnl_per_capital", 0.0) or 0.0)
    return {
        "n_trades": n_trades,
        "sharpe_r": float(trade.get("trade_sharpe", 0.0) or 0.0),
        "mean_r": float(pnl / max(n_trades, 1)),
        "total_r": pnl,
        "win_rate": float(trade.get("win_rate", 0.0) or 0.0),
        "max_drawdown_r": float(trade.get("max_drawdown", 0.0) or 0.0),
        "near_stop_rate": float(trade.get("forced_rate", 0.0) or 0.0),
        "worst_segment": float(segment.get("worst_segment", 0.0) or 0.0),
        "segment_win_rate": float(segment.get("segment_win_rate", 0.0) or 0.0),
        "forced_rate": float(trade.get("forced_rate", 0.0) or 0.0),
    }


def _dual_add_metrics_from_standalone(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        rows = []
    if not rows:
        return {}
    row = rows[0]
    n_trades = int(float(row.get("trades", 0) or 0))
    pnl = float(row.get("sum_pnl_per_capital", 0.0) or 0.0)
    risk_stop = float(row.get("risk_stop_rate", 0.0) or 0.0)
    forced = float(row.get("forced_rate", 0.0) or 0.0)
    return {
        "n_trades": n_trades,
        "sharpe_r": 0.0,
        "mean_r": float(pnl / max(n_trades, 1)),
        "total_r": pnl,
        "win_rate": float(row.get("trade_win_rate", 0.0) or 0.0),
        "segment_win_rate": float(row.get("segment_win_rate", 0.0) or 0.0),
        "max_drawdown_r": float(row.get("median_drawdown", 0.0) or 0.0),
        "near_stop_rate": risk_stop,
        "worst_segment": float(row.get("worst_segment", 0.0) or 0.0),
        "risk_stop_rate": risk_stop,
        "forced_rate": forced,
    }


def _trade_sharpe_from_csv(path: Path) -> float:
    if not path.exists():
        return 0.0
    vals: List[float] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                vals.append(float(row.get("pnl_per_capital", 0.0) or 0.0))
    except Exception:
        return 0.0
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / max(1, n - 1)
    std = math.sqrt(var)
    if std <= 0:
        return 0.0
    return float(mean / std)


def _load_strategy_gate_metrics(
    *, run_dir: Path, strategy: str, strategy_type: str
) -> Dict[str, Any]:
    summary = _read_json(run_dir / strategy / "multileg_summary.json")
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    if isinstance(metrics, dict) and metrics:
        return metrics
    if strategy_type == "grid":
        return _grid_metrics_from_standalone(run_dir / strategy / "metrics.json")
    if strategy_type in ("dual_add_trend", "trend_scalp"):
        out = _dual_add_metrics_from_standalone(run_dir / strategy / "summary.csv")
        if out:
            out["sharpe_r"] = _trade_sharpe_from_csv(
                run_dir / strategy / "dual_add_trades.csv"
            )
        return out
    return {}


def _render_html(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "<!doctype html>",
        '<html lang="en"><meta charset="utf-8"><title>multi_leg_gate</title>',
        "<style>body{font-family:system-ui,sans-serif;margin:20px;max-width:980px}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px}"
        "th{background:#f4f4f4;text-align:left}.ok{color:#0a7a0a}.bad{color:#a80c0c}</style>",
        "<h1>Multi-leg Gate Report</h1>",
        f"<p>decision: <b class=\"{'ok' if report.get('decision') == 'READY_SHADOW' else 'bad'}\">{report.get('decision')}</b></p>",
        "<h2>Rules</h2>",
        "<table><thead><tr><th>rule</th><th>value</th><th>threshold</th><th>pass</th></tr></thead><tbody>",
    ]
    for r in report.get("rules", []):
        lines.append(
            "<tr>"
            f"<td>{r.get('rule')}</td>"
            f"<td>{r.get('value')}</td>"
            f"<td>{r.get('threshold')}</td>"
            f"<td>{'yes' if r.get('pass') else 'no'}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    lines.append("<h2>Per-strategy</h2>")
    lines.append(
        "<table><thead><tr><th>strategy</th><th>type</th><th>trades</th><th>total_r</th><th>forced</th><th>risk_stop</th><th>max_dd_r</th></tr></thead><tbody>"
    )
    for row in report.get("strategies", []):
        m = row.get("metrics", {})
        lines.append(
            "<tr>"
            f"<td>{row.get('strategy')}</td>"
            f"<td>{row.get('strategy_type')}</td>"
            f"<td>{m.get('n_trades', 0)}</td>"
            f"<td>{m.get('total_r', 0.0):.4f}</td>"
            f"<td>{m.get('forced_rate', 0.0):.2%}</td>"
            f"<td>{m.get('risk_stop_rate', m.get('near_stop_rate', 0.0)):.2%}</td>"
            f"<td>{m.get('max_drawdown_r', 0.0):.4f}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    lines.append("</html>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Compute multi-leg deploy gate decision.")
    p.add_argument(
        "--run-dir",
        required=True,
        help="run dir containing <strategy>/multileg_summary.json",
    )
    p.add_argument(
        "--config",
        default="config/pipelines/multileg_orchestrate_2h.yaml",
        help="multi-leg pipeline YAML",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="output json (default: <run-dir>/multi_leg_gate_report.json)",
    )
    p.add_argument(
        "--out-html",
        default="",
        help="output html (default: <run-dir>/multi_leg_gate_report.html)",
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (PROJECT_ROOT / run_dir).resolve()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_pipeline_config(cfg_path)
    gate = cfg.get("multi_leg_gate", {}) or {}

    rows: List[Dict[str, Any]] = []
    for strategy, scfg in (cfg.get("strategies") or {}).items():
        st = _strategy_type(scfg)
        if st not in {"grid", "dual_add_trend", "trend_scalp"}:
            continue
        metrics = _load_strategy_gate_metrics(
            run_dir=run_dir,
            strategy=strategy,
            strategy_type=st,
        )
        rows.append(
            {
                "strategy": strategy,
                "strategy_type": st,
                "summary_path": str(run_dir / strategy / "multileg_summary.json"),
                "metrics": metrics if isinstance(metrics, dict) else {},
            }
        )

    total_trades = sum(
        int((r.get("metrics") or {}).get("n_trades", 0) or 0) for r in rows
    )
    total_r = sum(
        float((r.get("metrics") or {}).get("total_r", 0.0) or 0.0) for r in rows
    )
    max_dd_r = (
        min(
            float((r.get("metrics") or {}).get("max_drawdown_r", 0.0) or 0.0)
            for r in rows
        )
        if rows
        else 0.0
    )
    max_forced = (
        max(
            float((r.get("metrics") or {}).get("forced_rate", 0.0) or 0.0) for r in rows
        )
        if rows
        else 0.0
    )
    max_risk_stop = (
        max(
            float(
                (r.get("metrics") or {}).get(
                    "risk_stop_rate",
                    (r.get("metrics") or {}).get("near_stop_rate", 0.0),
                )
                or 0.0
            )
            for r in rows
        )
        if rows
        else 0.0
    )
    seg_win = [
        float((r.get("metrics") or {}).get("segment_win_rate", 0.0) or 0.0)
        for r in rows
        if (r.get("metrics") or {}).get("segment_win_rate") is not None
    ]
    min_seg_win = min(seg_win) if seg_win else 0.0

    thresholds = {
        "min_total_trades": int(gate.get("min_total_trades", 120) or 120),
        "min_total_r": float(gate.get("min_total_r", 0.5) or 0.5),
        "max_drawdown_r": float(gate.get("max_drawdown_r", 0.08) or 0.08),
        "max_forced_rate": float(gate.get("max_forced_rate", 0.50) or 0.50),
        "max_risk_stop_rate": float(gate.get("max_risk_stop_rate", 0.10) or 0.10),
        "min_segment_win_rate": float(gate.get("min_segment_win_rate", 0.45) or 0.45),
    }
    rules = [
        {
            "rule": "min_total_trades",
            "value": total_trades,
            "threshold": f">= {thresholds['min_total_trades']}",
            "pass": total_trades >= thresholds["min_total_trades"],
        },
        {
            "rule": "min_total_r",
            "value": total_r,
            "threshold": f">= {thresholds['min_total_r']:.4f}",
            "pass": total_r >= thresholds["min_total_r"],
        },
        {
            "rule": "max_drawdown_r",
            "value": max_dd_r,
            "threshold": f">= -{thresholds['max_drawdown_r']:.4f}",
            "pass": max_dd_r >= -thresholds["max_drawdown_r"],
        },
        {
            "rule": "max_forced_rate",
            "value": max_forced,
            "threshold": f"<= {thresholds['max_forced_rate']:.2%}",
            "pass": max_forced <= thresholds["max_forced_rate"],
        },
        {
            "rule": "max_risk_stop_rate",
            "value": max_risk_stop,
            "threshold": f"<= {thresholds['max_risk_stop_rate']:.2%}",
            "pass": max_risk_stop <= thresholds["max_risk_stop_rate"],
        },
        {
            "rule": "min_segment_win_rate",
            "value": min_seg_win,
            "threshold": f">= {thresholds['min_segment_win_rate']:.2%}",
            "pass": min_seg_win >= thresholds["min_segment_win_rate"],
        },
    ]

    hard_failed = any(
        (r["rule"] in {"max_drawdown_r", "max_forced_rate", "max_risk_stop_rate"})
        and not bool(r["pass"])
        for r in rules
    )
    if hard_failed:
        decision = "OFFLINE"
    elif all(bool(r["pass"]) for r in rules):
        decision = "READY_SHADOW"
    elif total_r < thresholds["min_total_r"]:
        decision = "RETUNE_THRESHOLDS"
    else:
        decision = "RESEARCH_ONLY"

    report = {
        "run_dir": str(run_dir),
        "decision": decision,
        "strategies": rows,
        "summary": {
            "total_trades": total_trades,
            "total_r": total_r,
            "max_drawdown_r": max_dd_r,
            "max_forced_rate": max_forced,
            "max_risk_stop_rate": max_risk_stop,
            "min_segment_win_rate": min_seg_win,
        },
        "thresholds": thresholds,
        "rules": rules,
    }

    out_json = (
        Path(args.out_json)
        if str(args.out_json).strip()
        else run_dir / "multi_leg_gate_report.json"
    )
    out_html = (
        Path(args.out_html)
        if str(args.out_html).strip()
        else run_dir / "multi_leg_gate_report.html"
    )
    if not out_json.is_absolute():
        out_json = PROJECT_ROOT / out_json
    if not out_html.is_absolute():
        out_html = PROJECT_ROOT / out_html
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
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
