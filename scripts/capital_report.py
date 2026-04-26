"""Capital-denominated reports for research backtests.

The project has two historical PnL units:

* single-position event backtests: ``pnl_r`` is a raw R-multiple sum. Money needs
  an explicit ``risk_per_r`` assumption.
* multi-leg inventory backtests: ``pnl_per_capital`` is already normalized by the
  strategy capital bucket.

This module writes a small JSON/HTML report so research outputs can answer the
plain question: "if I start with 10,000 USD, what does this run imply?"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def write_capital_report_from_trades(
    *,
    trades_path: str | Path,
    out_dir: str | Path,
    unit: str,
    title: str,
    initial_capital: float = 10_000.0,
    risk_per_r: float = 0.01,
    start_date: str = "",
    end_date: str = "",
    total_r: Optional[float] = None,
) -> Dict[str, Any]:
    """Write ``capital_report.json/html`` from a trade CSV.

    Args:
        unit: ``capital_normalized`` for ``pnl_per_capital``; ``r_multiple`` for
            ``pnl_r``. ``r_multiple`` converts to money as
            ``initial_capital * risk_per_r * pnl_r``.
    """
    trades_path = Path(trades_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "capital_report.json"
    html_path = out_dir / "capital_report.html"

    if not trades_path.exists():
        report = _empty_report(
            title=title,
            unit=unit,
            initial_capital=initial_capital,
            risk_per_r=risk_per_r,
            reason=f"trades file missing: {trades_path}",
        )
        _write_report(report, report_path, html_path)
        return report

    trades = pd.read_csv(trades_path)
    if trades.empty:
        report = _empty_report(
            title=title,
            unit=unit,
            initial_capital=initial_capital,
            risk_per_r=risk_per_r,
            reason="no trades",
        )
        _write_report(report, report_path, html_path)
        return report

    time_col = "exit_time" if "exit_time" in trades.columns else None
    if time_col is None:
        report = _empty_report(
            title=title,
            unit=unit,
            initial_capital=initial_capital,
            risk_per_r=risk_per_r,
            reason="missing exit_time",
        )
        _write_report(report, report_path, html_path)
        return report

    if unit == "capital_normalized":
        pnl_col = "pnl_per_capital"
        if pnl_col not in trades.columns:
            report = _empty_report(
                title=title,
                unit=unit,
                initial_capital=initial_capital,
                risk_per_r=risk_per_r,
                reason="missing pnl_per_capital",
            )
            _write_report(report, report_path, html_path)
            return report
        returns = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0)
        unit_explanation = (
            "total_r equals sum(pnl_per_capital): capital-bucket-normalized net "
            "return. 1.0 means +100% on the strategy capital bucket."
        )
        effective_total_r = float(returns.sum())
        dollars = initial_capital * returns
    elif unit == "r_multiple":
        pnl_col = "pnl_r"
        if pnl_col not in trades.columns:
            report = _empty_report(
                title=title,
                unit=unit,
                initial_capital=initial_capital,
                risk_per_r=risk_per_r,
                reason="missing pnl_r",
            )
            _write_report(report, report_path, html_path)
            return report
        r_values = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0)
        unit_explanation = (
            "total_r equals raw sum(pnl_r): cumulative R-multiples. Money is "
            "estimated with initial_capital * risk_per_r * pnl_r."
        )
        effective_total_r = float(r_values.sum() if total_r is None else total_r)
        dollars = initial_capital * float(risk_per_r) * r_values
        returns = float(risk_per_r) * r_values
    else:
        raise ValueError(f"unsupported capital report unit: {unit}")

    df = pd.DataFrame(
        {
            "exit_time": pd.to_datetime(trades[time_col], utc=True, errors="coerce"),
            "pnl_usd_est": dollars,
            "return_increment": returns,
        }
    ).dropna(subset=["exit_time"])
    df = df.sort_values("exit_time")
    if df.empty:
        report = _empty_report(
            title=title,
            unit=unit,
            initial_capital=initial_capital,
            risk_per_r=risk_per_r,
            reason="no valid exit_time rows",
        )
        _write_report(report, report_path, html_path)
        return report

    df["equity"] = float(initial_capital) + df["pnl_usd_est"].cumsum()
    df["peak"] = df["equity"].cummax()
    df["drawdown_usd"] = df["equity"] - df["peak"]
    df["drawdown_pct"] = df["drawdown_usd"] / df["peak"].replace(0.0, pd.NA)

    start_ts = (
        pd.Timestamp(start_date, tz="UTC") if start_date else df["exit_time"].min()
    )
    end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else df["exit_time"].max()
    years = max((end_ts - start_ts).days / 365.25, 1.0 / 365.25)
    final_capital = float(df["equity"].iloc[-1])
    total_return = final_capital / float(initial_capital) - 1.0
    cagr = (
        (final_capital / float(initial_capital)) ** (1.0 / years) - 1.0
        if final_capital > 0
        else -1.0
    )

    report = {
        "title": title,
        "initial_capital": float(initial_capital),
        "final_capital": final_capital,
        "estimated_profit_usd": final_capital - float(initial_capital),
        "total_return": total_return,
        "cagr": cagr,
        "cagr_explanation": (
            "CAGR is compound annual growth rate: the constant yearly return that "
            "turns initial_capital into final_capital over the measured period."
        ),
        "max_drawdown_usd": float(df["drawdown_usd"].min()),
        "max_drawdown_pct": float(df["drawdown_pct"].min()),
        "trades": int(len(df)),
        "start": str(start_ts),
        "end": str(end_ts),
        "years": float(years),
        "unit": unit,
        "unit_explanation": unit_explanation,
        "total_r": float(effective_total_r),
        "risk_per_r": float(risk_per_r),
        "assumptions": {
            "event_r_multiple": "1R risks risk_per_r of initial_capital; default 1%.",
            "multi_leg_capital_normalized": "pnl_per_capital is applied to initial_capital.",
            "compounding": "Equity is additive per trade with fixed initial_capital sizing.",
            "excluded": "Funding/liquidation/margin usage beyond modeled trade PnL unless present in the backtest trades.",
        },
        "trades_path": str(trades_path),
        "equity_tail": [
            {
                "exit_time": str(row.exit_time),
                "equity": float(row.equity),
                "drawdown_pct": float(row.drawdown_pct),
            }
            for row in df.tail(20).itertuples(index=False)
        ],
    }
    _write_report(report, report_path, html_path)
    return report


def _empty_report(
    *,
    title: str,
    unit: str,
    initial_capital: float,
    risk_per_r: float,
    reason: str,
) -> Dict[str, Any]:
    return {
        "title": title,
        "initial_capital": float(initial_capital),
        "final_capital": float(initial_capital),
        "estimated_profit_usd": 0.0,
        "total_return": 0.0,
        "cagr": 0.0,
        "max_drawdown_usd": 0.0,
        "max_drawdown_pct": 0.0,
        "trades": 0,
        "unit": unit,
        "risk_per_r": float(risk_per_r),
        "reason": reason,
    }


def _write_report(report: Dict[str, Any], json_path: Path, html_path: Path) -> None:
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    html_path.write_text(_render_html(report), encoding="utf-8")


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "N/A"


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "N/A"


def _render_html(report: Dict[str, Any]) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{report.get('title', 'Capital Report')}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .note {{ color: #555; max-width: 980px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>{report.get('title', 'Capital Report')}</h1>
  <table>
    <tr><th>Initial capital</th><td>{_money(report.get('initial_capital'))}</td></tr>
    <tr><th>Final capital</th><td>{_money(report.get('final_capital'))}</td></tr>
    <tr><th>Estimated profit</th><td>{_money(report.get('estimated_profit_usd'))}</td></tr>
    <tr><th>Total return</th><td>{_pct(report.get('total_return'))}</td></tr>
    <tr><th>CAGR</th><td>{_pct(report.get('cagr'))}</td></tr>
    <tr><th>Max drawdown</th><td>{_money(report.get('max_drawdown_usd'))} ({_pct(report.get('max_drawdown_pct'))})</td></tr>
    <tr><th>Trades</th><td>{report.get('trades')}</td></tr>
    <tr><th>Total R</th><td>{float(report.get('total_r', 0.0) or 0.0):.6f}</td></tr>
  </table>
  <h2>Definitions</h2>
  <p class="note"><b>CAGR:</b> {report.get('cagr_explanation', '')}</p>
  <p class="note"><b>Total R unit:</b> {report.get('unit_explanation', '')}</p>
  <p class="note"><b>Assumptions:</b> {json.dumps(report.get('assumptions', {}), ensure_ascii=False)}</p>
</body>
</html>
"""
