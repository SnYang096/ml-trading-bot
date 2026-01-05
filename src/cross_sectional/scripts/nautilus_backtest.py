#!/usr/bin/env python3
"""
CS Nautilus-style backtest (bar-only, re-run).

Goal:
  - Re-run a CS portfolio backtest from a panel + (model or factor-combo signal),
    producing a rebalance audit log + per-symbol trade list + HTML report.

Why "Nautilus-style":
  - The execution model is explicitly bar-driven (rebalance-only) and is designed to
    mirror the live "rebalance loop" logic. This provides a clean path to later plug
    into Nautilus Trader Strategy/OMS if desired.

Inputs:
  - --panel: parquet/csv with at least [timestamp, symbol, close] and feature columns
  - --model-path: optional joblib model (if omitted and --signal=factor_combo, uses factors)
  - --feature-file / --feature-cols: feature columns for model/factor-combo

Outputs (output-dir):
  - rebalance_log.csv (one row per rebalance with long/short lists)
  - equity.csv
  - trades.csv (per symbol per rebalance)
  - metrics.json
  - report.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from joblib import load as joblib_load

from cross_sectional.model_portfolio_backtest import (
    PortfolioBacktestConfig,
    portfolio_backtest_with_rebalance_log,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CS nautilus-style backtest (bar-only, re-run)."
    )
    p.add_argument(
        "--panel",
        required=True,
        help="Panel parquet/csv (timestamp,symbol,close,features...)",
    )
    p.add_argument(
        "--output-dir",
        default="results/cross_sectional/nautilus_backtest",
        help="Output directory",
    )
    p.add_argument(
        "--signal",
        choices=["model", "factor_combo"],
        default="model",
        help="Signal source: model predictions or factor-combo zscore mean",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Path to trained CS model joblib (for --signal=model)",
    )
    p.add_argument(
        "--feature-file",
        default=None,
        help="Text file with feature columns (one per line)",
    )
    p.add_argument(
        "--feature-cols",
        default=None,
        help="Comma-separated feature columns (overrides feature-file)",
    )
    p.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name")
    p.add_argument("--symbol-col", default="symbol", help="Symbol column name")
    p.add_argument("--close-col", default="close", help="Close price column name")

    # execution config (match train.backtest_cfg)
    p.add_argument(
        "--mode", default="market_neutral", help="long_only | market_neutral"
    )
    p.add_argument("--holding", type=int, default=12, help="Holding period in bars")
    p.add_argument("--lag", type=int, default=1, help="Execution lag in bars")
    p.add_argument("--topk", type=int, default=10, help="Top-K longs")
    p.add_argument(
        "--bottomk", type=int, default=10, help="Bottom-K shorts (market_neutral)"
    )
    p.add_argument(
        "--gross-leverage", type=float, default=1.0, help="Gross leverage cap"
    )
    p.add_argument(
        "--max-weight", type=float, default=0.10, help="Max abs weight per asset"
    )
    p.add_argument(
        "--turnover-limit",
        type=float,
        default=None,
        help="Optional turnover limit per rebalance",
    )
    p.add_argument(
        "--cash-buffer", type=float, default=0.10, help="Cash buffer fraction [0..1]"
    )
    p.add_argument("--equity-mode", default="compound", help="simple|compound|log")
    p.add_argument(
        "--fee-bps", type=float, default=2.0, help="Fee bps applied on turnover"
    )
    p.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Slippage bps applied on turnover",
    )
    p.add_argument(
        "--funding-bps-per-bar",
        type=float,
        default=0.0,
        help="Funding bps per bar on short exposure",
    )
    p.add_argument(
        "--borrow-bps-per-bar",
        type=float,
        default=0.0,
        help="Borrow bps per bar on short exposure",
    )
    p.add_argument(
        "--min-assets", type=int, default=12, help="Min assets per timestamp"
    )
    p.add_argument(
        "--periods-per-year",
        type=float,
        default=None,
        help="Annualisation factor (bars/year)",
    )

    # report
    p.add_argument(
        "--html", default="report.html", help="HTML report filename under output-dir"
    )
    p.add_argument(
        "--max-trades", type=int, default=300, help="Max trades to show inline in HTML"
    )
    return p.parse_args()


def _read_panel(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _load_feature_list(args: argparse.Namespace, panel_cols: List[str]) -> List[str]:
    if args.feature_cols:
        cols = [c.strip() for c in str(args.feature_cols).split(",") if c.strip()]
        return cols
    if args.feature_file:
        lines = [
            x.strip()
            for x in Path(args.feature_file).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        return lines
    # fallback: use all numeric non-price cols (explicitly conservative)
    exclude = {
        args.timestamp_col,
        args.symbol_col,
        args.close_col,
        "open",
        "high",
        "low",
        "volume",
    }
    out = [c for c in panel_cols if c not in exclude]
    return out


def _zscore_cross_section(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd is None or sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def _compute_factor_combo(panel: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    # panel is MultiIndex (timestamp, symbol)
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise KeyError(f"Missing feature cols: {missing[:10]}")
    # zscore each factor within timestamp, then mean
    zcols: List[pd.Series] = []
    for c in feature_cols:
        # Use transform to preserve the original MultiIndex shape (timestamp, symbol)
        zc = panel[c].groupby(level=0).transform(_zscore_cross_section)
        zcols.append(zc.rename(c))
    zdf = pd.concat(zcols, axis=1)
    return zdf.mean(axis=1).rename("factor_combo")


def _render_html(
    *,
    out_dir: Path,
    metrics: Dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    max_trades: int,
    html_name: str,
) -> None:
    def _fmt(x: Any) -> str:
        try:
            if x is None:
                return "NA"
            if isinstance(x, float):
                if np.isnan(x):
                    return "NA"
                return f"{x:.6g}"
            return str(x)
        except Exception:
            return str(x)

    def _sharpe_grade(sh: float) -> str:
        if not np.isfinite(sh):
            return "NA"
        if sh >= 2.0:
            return "EXCELLENT / 很强"
        if sh >= 1.0:
            return "GOOD / 不错"
        if sh >= 0.0:
            return "MARGINAL / 一般"
        return "BAD / 差"

    sharpe = float(metrics.get("sharpe_net", float("nan")))
    verdict = _sharpe_grade(sharpe)

    eq_tail = equity.reset_index().tail(120) if not equity.empty else pd.DataFrame()
    tr_head = trades.head(int(max_trades)) if not trades.empty else pd.DataFrame()

    html = f"""\
<html>
  <head>
    <meta charset="utf-8"/>
    <title>CS Nautilus Backtest Report</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; padding: 20px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
      th {{ background: #f6f6f6; position: sticky; top: 0; }}
      code {{ background: #f6f8fa; padding: 2px 4px; }}
      summary {{ cursor: pointer; }}
    </style>
  </head>
  <body>
    <h1>CS Nautilus Backtest Report (bar-only)</h1>
    <h2>Conclusion</h2>
    <ul>
      <li><b>Sharpe(net)</b>={_fmt(sharpe)} → <b>{verdict}</b></li>
      <li><b>Total return(net)</b>={_fmt(metrics.get('total_return_net'))}, <b>Max drawdown</b>={_fmt(metrics.get('max_drawdown'))}</li>
      <li><b>n_periods</b>={_fmt(metrics.get('n_timestamps'))} (rebalance periods)</li>
    </ul>

    <h2>Artifacts</h2>
    <ul>
      <li><code>metrics.json</code></li>
      <li><code>equity.csv</code></li>
      <li><code>trades.csv</code></li>
      <li><code>rebalance_log.csv</code></li>
    </ul>

    <h2>Equity (tail)</h2>
    <details open><summary><b>equity.csv (tail)</b></summary>
      {(eq_tail.to_html(index=False, float_format=lambda x: f"{x:.6g}") if not eq_tail.empty else "<p>(empty)</p>")}
    </details>

    <h2>Trade list</h2>
    <p>Inline shows first {int(max_trades)} trades. Full CSV: <code>trades.csv</code></p>
    <details open><summary><b>trades.csv (head)</b></summary>
      {(tr_head.to_html(index=False, float_format=lambda x: f"{x:.6g}") if not tr_head.empty else "<p>(empty)</p>")}
    </details>

    <h2>Assumptions / 说明</h2>
    <ul>
      <li><b>执行模型</b>：bar-only；只在换仓点交易；按 holding/lag 生成换仓序列。</li>
      <li><b>成本</b>：turnover × (fee_bps + slippage_bps)，并支持 funding/borrow (short exposure per bar)。</li>
      <li><b>信号</b>：model = joblib 模型预测；factor_combo = 选中因子的截面 zscore 均值。</li>
    </ul>
  </body>
</html>
"""
    (out_dir / html_name).write_text(html, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_panel(args.panel)
    if args.timestamp_col not in raw.columns or args.symbol_col not in raw.columns:
        raise KeyError("panel must contain timestamp & symbol columns")
    raw[args.timestamp_col] = pd.to_datetime(
        raw[args.timestamp_col], utc=True, errors="coerce"
    )
    raw = raw.dropna(subset=[args.timestamp_col, args.symbol_col])
    raw[args.symbol_col] = raw[args.symbol_col].astype(str)
    # MultiIndex panel
    panel = raw.set_index([args.timestamp_col, args.symbol_col]).sort_index()

    feature_cols = _load_feature_list(args, list(panel.columns))
    if args.signal == "model":
        if not args.model_path:
            raise ValueError("--model-path is required for --signal=model")
        model = joblib_load(args.model_path)
        # only numeric cols
        X = panel[feature_cols].apply(pd.to_numeric, errors="coerce")
        preds = pd.Series(
            model.predict(X.values), index=panel.index, name="model_prediction"
        )
        panel2 = panel[[args.close_col]].copy()
        panel2["model_prediction"] = preds
        signal_col = "model_prediction"
    else:
        panel2 = panel[[args.close_col]].copy()
        # need access to factors for combo
        for c in feature_cols:
            panel2[c] = pd.to_numeric(panel.get(c), errors="coerce")
        panel2["factor_combo"] = _compute_factor_combo(panel2, feature_cols)
        signal_col = "factor_combo"

    cfg = PortfolioBacktestConfig(
        mode=str(args.mode),
        holding_period_bars=int(args.holding),
        execution_lag_bars=int(args.lag),
        top_k=int(args.topk),
        bottom_k=int(args.bottomk),
        gross_leverage=float(args.gross_leverage),
        max_weight=float(args.max_weight),
        turnover_limit=(
            float(args.turnover_limit) if args.turnover_limit is not None else None
        ),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        funding_bps_per_bar=float(args.funding_bps_per_bar),
        borrow_bps_per_bar=float(args.borrow_bps_per_bar),
        min_assets=int(args.min_assets),
        periods_per_year=(
            float(args.periods_per_year) if args.periods_per_year is not None else None
        ),
        cash_buffer=float(args.cash_buffer),
        equity_mode=str(args.equity_mode),
        initial_capital=1.0,
    )

    ts, metrics, rb = portfolio_backtest_with_rebalance_log(
        panel2, signal_col=signal_col, close_col=args.close_col, cfg=cfg
    )
    # Save
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    ts.reset_index().to_csv(out_dir / "equity.csv", index=False)
    rb.to_csv(out_dir / "rebalance_log.csv", index=False)

    # Build per-symbol trade list (entry at rebalance, exit at next rebalance)
    trades: List[Dict[str, Any]] = []
    if not rb.empty and args.close_col in panel.columns:
        px = pd.to_numeric(panel[args.close_col], errors="coerce").rename("px")
        for i in range(len(rb) - 1):
            t0 = pd.to_datetime(rb.loc[i, "rebalance_ts"], utc=True, errors="coerce")
            t1 = pd.to_datetime(
                rb.loc[i + 1, "rebalance_ts"], utc=True, errors="coerce"
            )
            if pd.isna(t0) or pd.isna(t1):
                continue
            # weights are not in rb; reconstruct from long/short lists
            longs = (
                json.loads(rb.loc[i, "long_symbols_json"])
                if "long_symbols_json" in rb.columns
                else []
            )
            shorts = (
                json.loads(rb.loc[i, "short_symbols_json"])
                if "short_symbols_json" in rb.columns
                else []
            )
            # approximate weights equal-weight within lists using cfg
            w = {}
            invest = float(1.0 - cfg.cash_buffer) * float(cfg.gross_leverage)
            if cfg.mode == "long_only":
                if longs:
                    ww = min(invest / float(len(longs)), cfg.max_weight)
                    for s in longs:
                        w[str(s)] = float(ww)
            else:
                if longs and shorts:
                    half = invest / 2.0
                    wl = min(half / float(len(longs)), cfg.max_weight)
                    ws = min(half / float(len(shorts)), cfg.max_weight)
                    for s in longs:
                        w[str(s)] = float(wl)
                    for s in shorts:
                        w[str(s)] = -float(ws)

            for sym, ww in w.items():
                try:
                    a = float(px.loc[(t0, sym)])
                    b = float(px.loc[(t1, sym)])
                except Exception:
                    continue
                if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
                    continue
                r = b / a - 1.0
                trades.append(
                    {
                        "rebalance_ts": t0,
                        "exit_ts": t1,
                        "symbol": sym,
                        "weight": float(ww),
                        "entry_price": a,
                        "exit_price": b,
                        "return": float(r),
                        "side": "LONG" if ww > 0 else "SHORT",
                    }
                )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["rebalance_ts", "symbol"])
    trades_df.to_csv(out_dir / "trades.csv", index=False)

    _render_html(
        out_dir=out_dir,
        metrics=metrics,
        equity=ts.reset_index(),
        trades=trades_df,
        max_trades=int(args.max_trades),
        html_name=str(args.html),
    )
    print(f"✅ CS nautilus-style backtest done. out_dir={out_dir}")


if __name__ == "__main__":
    main()
