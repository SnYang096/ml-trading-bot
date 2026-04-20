"""B+ backtest — from leverage-capacity parquet to trades + equity curve.

Signal rows: parquet from `analyze_leverage_capacity_v*.py`.
Execution: enter at sample `close`, exit at **close[H bars later]** on FS series.

PnL (simplified margin return per unit stake):

  ``pnl_on_stake ≈ eff_L * r_hold - fund_cost_window - fee_round_trip_frac``

See module docstring in argparse description for details.

Examples:
  python scripts/lottery_backtest_bplus.py \\
      --config config/strategies/bad-candidates/lottery100/backtest_bplus.yaml
  python scripts/lottery_backtest_bplus.py \\
      --samples reports/leverage_capacity_v4_bull_only/BTCUSDT_120T_H120_samples_bull_only.parquet \\
      --output-dir reports/lottery_bplus_run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_v2():
    import importlib.util

    path = ROOT / "scripts" / "analyze_leverage_capacity_v2.py"
    mod_name = "_lev_cap_v2_bt"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.__name__ = mod_name
    return mod


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_samples(path_glob: str) -> pd.DataFrame:
    path_glob = str(path_glob)
    if "*" in path_glob or "?" in path_glob:
        import glob

        files = sorted(glob.glob(path_glob))
        if not files:
            raise FileNotFoundError(path_glob)
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, axis=0)
    else:
        df = pd.read_parquet(Path(path_glob))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_close_series(fs_layer: str, symbol: str, timeframe: str) -> pd.Series:
    v2 = _load_v2()
    ohlc = v2.load_feature_store(fs_layer, symbol, timeframe)
    return ohlc["close"].astype(float).sort_index()


def run_backtest(
    samples: pd.DataFrame,
    close: pd.Series,
    horizon: int,
    symbol: str,
    side: str,
    min_lmax: float,
    leverage: float,
    cap_leverage: bool,
    fee_rt_bps: float,
    non_overlap: bool,
    liquidate_over: bool,
    compound: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = samples[
        (samples["symbol"] == symbol)
        & (samples["side"] == side)
        & (samples["horizon"] == horizon)
        & (samples["lmax_adj"] >= min_lmax)
        & samples["lmax_adj"].notna()
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    cdf = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(close.index),
            "fs_close": close.values.astype(float),
        }
    )
    cdf["exit_close"] = cdf["fs_close"].shift(-horizon)
    cdf["exit_ts"] = cdf["timestamp"].shift(-horizon)

    merged = df.merge(cdf, on="timestamp", how="inner")
    merged = merged.dropna(subset=["exit_close", "exit_ts"])
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    fee_frac = fee_rt_bps / 10000.0
    trades: List[Dict[str, Any]] = []
    last_exit_ts: Optional[pd.Timestamp] = None

    for _, row in merged.iterrows():
        t = pd.Timestamp(row["timestamp"])
        exit_ts = pd.Timestamp(row["exit_ts"])
        if non_overlap and last_exit_ts is not None and t <= last_exit_ts:
            continue

        entry = float(row["close"])
        exit_px = float(row["exit_close"])
        r_hold = exit_px / entry - 1.0
        lmax = float(row["lmax_adj"])
        fund = float(row.get("fund_cost_window", 0.0) or 0.0)

        eff_l = min(leverage, lmax) if cap_leverage else leverage
        if liquidate_over and leverage > lmax:
            pnl = -1.0
            reason = "liquidated_over_capacity"
        else:
            pnl = eff_l * r_hold - fund - fee_frac
            reason = "closed"

        trades.append(
            {
                "entry_ts": t,
                "exit_ts": exit_ts,
                "symbol": symbol,
                "side": side,
                "horizon": horizon,
                "entry": entry,
                "exit": exit_px,
                "r_hold": r_hold,
                "lmax_adj": lmax,
                "eff_leverage": eff_l,
                "fund_cost_window": fund,
                "fee_round_trip_frac": fee_frac,
                "pnl_on_stake": pnl,
                "reason": reason,
            }
        )
        if non_overlap:
            last_exit_ts = exit_ts

    tradf = pd.DataFrame(trades)
    if tradf.empty:
        return tradf, pd.DataFrame(), {"n_trades": 0}

    if compound:
        eq = np.cumprod(1 + tradf["pnl_on_stake"].values)
    else:
        eq = np.cumsum(tradf["pnl_on_stake"].values) + 1.0
    ecurve = pd.DataFrame(
        {
            "exit_ts": tradf["exit_ts"],
            "equity": eq,
            "pnl_on_stake": tradf["pnl_on_stake"].values,
        }
    )

    pnl = tradf["pnl_on_stake"].values
    summary = {
        "n_trades": int(len(tradf)),
        "mean_pnl_on_stake": float(np.mean(pnl)),
        "std_pnl_on_stake": float(np.std(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "total_pnl_additive": float(np.sum(pnl)),
        "compound_end_equity": float(eq[-1]) if len(eq) else 1.0,
        "max_drawdown_approx": float(_max_dd(eq)),
        "fee_round_trip_bps": fee_rt_bps,
        "leverage_cap": cap_leverage,
        "nominal_leverage": leverage,
    }
    return tradf, ecurve, summary


def _max_dd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


def parse_cli() -> Dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--samples", type=str, default=None)
    p.add_argument("--fs-layer", type=str, default="features_me_120T_e98fe79b58")
    p.add_argument("--timeframe", type=str, default="120T")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--side", type=str, default="long", choices=["long", "short"])
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--min-lmax", type=float, default=100.0)
    p.add_argument("--leverage", type=float, default=50.0)
    p.add_argument("--no-cap-lmax", action="store_true")
    p.add_argument("--fee-round-trip-bps", type=float, default=8.0)
    p.add_argument("--overlap-ok", action="store_true")
    p.add_argument("--liquidate-if-over-capacity", action="store_true")
    p.add_argument("--compound", action="store_true")
    p.add_argument("--output-dir", type=str, default="reports/lottery_bplus")
    args = p.parse_args()

    kw: Dict[str, Any] = {
        "samples_glob": args.samples
        or "results/lottery100_bundle/v4/BTCUSDT_120T_H120_samples_bull_only.parquet",
        "fs_layer": args.fs_layer,
        "timeframe": args.timeframe,
        "symbol": args.symbol,
        "side": args.side,
        "horizon": args.horizon,
        "min_lmax": args.min_lmax,
        "leverage": args.leverage,
        "cap_leverage": not args.no_cap_lmax,
        "fee_rt_bps": args.fee_round_trip_bps,
        "non_overlap": not args.overlap_ok,
        "liquidate_over": args.liquidate_if_over_capacity,
        "compound": args.compound,
        "output_dir": Path(args.output_dir),
    }

    if args.config:
        cfg = load_yaml(Path(args.config))
        data = cfg.get("data", {})
        sig = cfg.get("signal", {})
        ex = cfg.get("execution", {})
        out = cfg.get("output", {})
        if args.samples:
            kw["samples_glob"] = args.samples
        elif data.get("samples_glob"):
            kw["samples_glob"] = data["samples_glob"]
        kw["fs_layer"] = data.get("fs_layer", kw["fs_layer"])
        kw["timeframe"] = data.get("timeframe", kw["timeframe"])
        kw["symbol"] = sig.get("symbol", kw["symbol"])
        kw["side"] = sig.get("side", kw["side"])
        kw["horizon"] = int(sig.get("horizon", kw["horizon"]))
        kw["min_lmax"] = float(sig.get("min_lmax_adj", kw["min_lmax"]))
        kw["leverage"] = float(ex.get("leverage", kw["leverage"]))
        kw["cap_leverage"] = bool(ex.get("cap_leverage_to_lmax", kw["cap_leverage"]))
        kw["fee_rt_bps"] = float(ex.get("fee_round_trip_bps", kw["fee_rt_bps"]))
        kw["non_overlap"] = bool(ex.get("non_overlapping", kw["non_overlap"]))
        if out.get("directory"):
            kw["output_dir"] = Path(out["directory"])

    return kw


def main() -> None:
    kw = parse_cli()
    out = kw["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    samples = load_samples(str(kw["samples_glob"]))
    close = load_close_series(kw["fs_layer"], kw["symbol"], kw["timeframe"])

    tradf, ecurve, summary = run_backtest(
        samples,
        close,
        horizon=int(kw["horizon"]),
        symbol=str(kw["symbol"]),
        side=str(kw["side"]),
        min_lmax=float(kw["min_lmax"]),
        leverage=float(kw["leverage"]),
        cap_leverage=bool(kw["cap_leverage"]),
        fee_rt_bps=float(kw["fee_rt_bps"]),
        non_overlap=bool(kw["non_overlap"]),
        liquidate_over=bool(kw["liquidate_over"]),
        compound=bool(kw["compound"]),
    )

    out.mkdir(parents=True, exist_ok=True)
    tradf.to_csv(out / "trades.csv", index=False)
    ecurve.to_csv(out / "equity_curve.csv", index=False)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    smd = (
        f"# Lottery B+ backtest summary\n\n"
        f"- samples: `{kw['samples_glob']}`\n"
        f"- symbol: {kw['symbol']} | side: {kw['side']} | H: {kw['horizon']}\n"
        f"- min_lmax_adj: {kw['min_lmax']} | leverage: {kw['leverage']} "
        f"| cap_to_lmax: {kw['cap_leverage']}\n"
        f"- fee_round_trip_bps: {kw['fee_rt_bps']}\n\n"
        f"## Metrics\n\n```json\n{json.dumps(summary, indent=2, ensure_ascii=False)}\n```\n"
    )
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(smd)

    print(smd)
    print(f"\nWrote: {out / 'trades.csv'}, {out / 'equity_curve.csv'}, summary.json/md")


if __name__ == "__main__":
    main()
