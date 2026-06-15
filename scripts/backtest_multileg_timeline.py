#!/usr/bin/env python3
"""Multi-leg timeline backtest — shared account, constitution, 1min execution.

Reuses the same engine + execution adapter stack as run_multi_leg_live.py,
swapping real BinanceAPI for MockBinanceAPI.  All symbols share one equity pool;
orders fill instantly at bar-close prices (or at limit for pending orders).

Usage::

    python scripts/backtest_multileg_timeline.py \\
      --start 2025-12-01 --end 2026-05-31 \\
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \\
      --chop-config config/experiments/20260613_multileg_sizing_validate/variants/chop_prod/meta.yaml \\
      --trend-config config/experiments/20260613_multileg_sizing_validate/variants/trend_prod/meta.yaml \\
      --constitution-yaml live/highcap/config/constitution/constitution.yaml \\
      --equity 50000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_1m_bars(data_dir, symbols, start, end):
    from scripts.diagnose_crf_edge import _load_symbol_1m

    frames = []
    for sym in symbols:
        raw = _load_symbol_1m(Path(data_dir), sym, start - pd.Timedelta(days=60), end)
        if raw.empty:
            print(f"  WARNING: no 1m data for {sym}")
            continue
        raw = raw[raw.index >= start]
        if raw.empty:
            continue
        raw["symbol"] = sym
        frames.append(raw)
    if not frames:
        raise SystemExit("no bar data loaded")
    return pd.concat(frames).sort_index()


def _build_features(data_dir, symbols):
    from scripts.diagnose_chop_grid import GridConfig, build_features
    from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv
    from src.features.time_series.baseline_features import (
        compute_trend_confidence_from_series,
    )

    feats = {}
    for sym in symbols:
        raw = _load_symbol_1m(
            Path(data_dir),
            sym,
            pd.Timestamp("2020-01-01").tz_localize("UTC"),
            pd.Timestamp("2026-06-01").tz_localize("UTC"),
        )
        if raw.empty:
            continue
        bars = _resample_ohlcv(raw, "2h")
        if bars.empty:
            continue
        df = build_features(sym, bars, GridConfig())
        if "trend_confidence" not in df.columns:
            bundle = compute_trend_confidence_from_series(close=df["close"])
            for col in bundle.columns:
                df[col] = bundle[col]
        feats[sym] = df
    return feats


def _lookup(feats, sym, ts):
    df = feats.get(sym)
    if df is None or df.empty:
        return {}
    df = df[df.index <= ts]
    if df.empty:
        return {}
    r = df.iloc[-1]
    return {
        "semantic_chop": float(r.get("semantic_chop", r.get("bpc_semantic_chop", 0.5))),
        "bpc_semantic_chop": float(
            r.get("bpc_semantic_chop", r.get("semantic_chop", 0.5))
        ),
        "box_pos_60": float(r.get("box_pos_60", 0.5)),
        "box_prefilter": bool(r.get("box_prefilter", False)),
        "trend_confidence": float(r.get("trend_confidence", 0.0)),
        "trend_direction": str(r.get("trend_direction", "UP")),
    }


def _is_valid_entry_time(
    ts: pd.Timestamp, exclude_asian: bool = True, exclude_weekend: bool = True
) -> bool:
    """Check if timestamp is in a valid entry window for trend_scalp.

    Args:
        ts: UTC timestamp of the current bar.
        exclude_asian: If True, block entries during Asian session (UTC 00:00–08:59).
        exclude_weekend: If True, block entries on Saturday/Sunday UTC.

    Returns:
        True if entries are allowed at this time.
    """
    if exclude_weekend and ts.dayofweek >= 5:  # Saturday=5, Sunday=6
        return False
    if exclude_asian:
        hour = ts.hour
        if hour < 9:  # UTC 00:00–08:59 = Asian session
            return False
    return True


@dataclass
class Account:
    equity: float
    peak_equity: float
    cum_pnl: float = 0.0
    daily_pnl: float = 0.0
    current_day: str = ""
    halted: bool = False
    halt_reason: str = ""
    trades_ok: int = 0
    trades_rej: int = 0
    max_dd_peak: float = 0.0

    @property
    def current(self):
        return self.equity + self.cum_pnl

    def to_summary(self) -> Dict[str, Any]:
        ret_pct = self.cum_pnl / max(self.equity, 1.0) * 100.0
        return {
            "equity_start": self.equity,
            "equity_end": self.current,
            "return_pct": ret_pct,
            "peak_equity": self.peak_equity,
            "max_drawdown_pct": self.max_dd_peak * 100.0,
            "trades_ok": self.trades_ok,
            "trades_rej": self.trades_rej,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }


def clean_bt_state(
    symbols: Optional[List[str]] = None, *, run_id: Optional[str] = None
) -> None:
    """Remove per-engine JSON state under /tmp so runs do not cross-contaminate."""
    patterns = ["/tmp/bt_chop_*.json", "/tmp/bt_trend_*.json"]
    if run_id:
        patterns = [
            f"/tmp/bt_chop_{run_id}_*.json",
            f"/tmp/bt_trend_{run_id}_*.json",
        ]
    if symbols:
        for sym in symbols:
            if run_id:
                patterns.append(f"/tmp/bt_chop_{run_id}_{sym}.json")
                patterns.append(f"/tmp/bt_trend_{run_id}_{sym}.json")
            else:
                patterns.append(f"/tmp/bt_chop_{sym}.json")
                patterns.append(f"/tmp/bt_trend_{sym}.json")
    seen: set[str] = set()
    for pat in patterns:
        for f in glob.glob(pat):
            if f in seen:
                continue
            seen.add(f)
            try:
                os.remove(f)
            except OSError:
                pass


def run_timeline_backtest(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbols: List[str],
    equity: float,
    chop_config: Path,
    trend_config: Path,
    constitution_yaml: Path,
    data_dir: Path = Path("data/parquet_data"),
    no_chop: bool = False,
    no_trend: bool = False,
    dry_run: bool = False,
    trend_time_filter: bool = False,
    save_preload: Optional[Path] = None,
    load_preload: Optional[Path] = None,
    clean_state: bool = True,
    progress: bool = True,
    run_id: Optional[str] = None,
) -> Tuple[Account, Dict[str, Any]]:
    run_id = run_id or str(os.getpid())
    if clean_state:
        clean_bt_state(symbols, run_id=run_id)

    bars_1m = feats = None
    if load_preload and load_preload.exists():
        import pickle as _pickle

        if progress:
            print(f"Loading pre-computed features from {load_preload} ...")
        pre = _pickle.loads(load_preload.read_bytes())
        feats = pre.get("feats") or pre
        if progress:
            print(f"  {len(feats)} symbol feature frames ready\n")
    else:
        if progress:
            print("Computing 2h features...")
        feats = _build_features(data_dir, symbols)
        if progress:
            print(f"  {len(feats)} symbols ready\n")

        if save_preload:
            import pickle as _pickle

            save_preload.parent.mkdir(parents=True, exist_ok=True)
            save_preload.write_bytes(_pickle.dumps({"feats": feats}))
            if progress:
                print(f"Saved feature preload to {save_preload}\n")

    if progress:
        print(f"Loading 1m bars: {start} → {end}")
    bars_1m = _load_1m_bars(data_dir, symbols, start, end)
    bars_1m = bars_1m[(bars_1m.index >= start) & (bars_1m.index <= end)]
    if progress:
        print(f"  {len(bars_1m)} 1m bars, {bars_1m['symbol'].nunique()} symbols\n")

    from src.live_data_stream.constitution_config import (
        load_constitution_dict,
        multi_leg_section,
    )
    from src.config.multileg_sizing import resolve_multi_leg_unit_notionals

    const = load_constitution_dict(constitution_yaml)
    ml = multi_leg_section(const)
    rs = ml.get("risk_limits", {})
    ks = const.get("kill_switch", {})
    max_gross = float(rs.get("max_gross_notional_pct", 2.70)) * equity
    daily_loss = float(ks.get("daily_loss_limit", 0.06)) * equity
    max_dd = float(ks.get("max_dd", 0.20))
    max_sym = int(rs.get("max_concurrent_multi_leg_symbols", 6))
    cd = int(rs.get("strategy_switch_cooldown_bars", 3))
    units = resolve_multi_leg_unit_notionals(
        ml, equity_usdt=equity, strategies=["chop_grid", "trend_scalp"]
    )
    chop_u = units.get("chop_grid", 4000.0)
    trend_u = units.get("trend_scalp", 9000.0)
    if progress:
        print(
            f"Constitution: gross≤{max_gross:.0f} daily≤{daily_loss:.0f} "
            f"dd≤{max_dd*100:.0f}% sym≤{max_sym}"
        )
        print(f"Sizing: chop={chop_u:.0f}/lvl trend={trend_u:.0f}/leg\n")

    from src.order_management.chop_grid_concurrency import MultiLegConcurrencyGate
    from src.time_series_model.live.chop_grid_live_engine import ChopGridLiveEngine
    from src.time_series_model.live.dual_add_trend_live_engine import (
        DualAddTrendLiveEngine,
    )

    gate = MultiLegConcurrencyGate(max_sym, cooldown_bars=cd)
    engines: Dict[str, Dict] = {}
    for sym in symbols:
        engines[sym] = {}
        if not no_chop:
            ce = ChopGridLiveEngine(
                config_path=chop_config,
                state_path=Path(f"/tmp/bt_chop_{run_id}_{sym}.json"),
                level_notional=chop_u,
                metrics_strategy="bt",
            )
            ce.state.symbol = sym
            gate.register(sym, ce, strategy="chop_grid")
            engines[sym]["chop"] = ce
        if not no_trend:
            te = DualAddTrendLiveEngine(
                config_path=trend_config,
                state_path=Path(f"/tmp/bt_trend_{run_id}_{sym}.json"),
                unit_notional=trend_u,
                metrics_strategy="bt",
            )
            te.state.symbol = sym
            gate.register(sym, te, strategy="trend_scalp")
            engines[sym]["trend"] = te

    from src.order_management.mock_binance_api import MockBinanceAPI
    from src.order_management.grid_execution_adapter import MultiLegExecutionAdapter
    from src.order_management.multi_leg_risk_governor import (
        MultiLegPortfolioRiskGovernor,
        MultiLegRiskLimits,
    )

    mock = MockBinanceAPI()
    mock.hedge_mode = True
    adapter = MultiLegExecutionAdapter(
        mock,
        require_hedge_mode=False,
        shadow=False,
        client_id_prefix="bt",
        default_symbol=symbols[0],
    )
    MultiLegPortfolioRiskGovernor(
        MultiLegRiskLimits(
            max_gross_notional=max_gross,
            max_net_notional=max_gross * 0.75,
            max_symbol_gross_notional=max_gross,
            max_symbol_net_notional=max_gross * 0.66,
            max_resting_orders=60,
            account_equity_usdt=equity,
            max_drawdown_pct=max_dd,
        )
    )

    acct = Account(equity=equity, peak_equity=equity)

    if progress:
        print(f"=== Timeline Backtest ({len(bars_1m)} bars) ===\n")
    t0 = time.monotonic()
    last_2h: Dict[str, pd.Timestamp] = {}
    n_1m = 0
    n_2h = 0

    for idx, row in bars_1m.iterrows():
        n_1m += 1
        sym = str(row["symbol"])
        close = float(row["close"])
        mock.set_price(sym, close)

        bar_2h = idx.floor("2h")
        if sym in last_2h and bar_2h <= last_2h[sym]:
            continue
        last_2h[sym] = bar_2h
        n_2h += 1

        f = _lookup(feats, sym, idx)
        if not f:
            continue

        inv_snapshot = {}
        for ek, eng in engines.get(sym, {}).items():
            state = getattr(eng, "state", None)
            if state and hasattr(state, "inventory"):
                inv_snapshot[ek] = [
                    {
                        "side": str(getattr(p, "side", "")).upper(),
                        "quantity": float(getattr(p, "quantity", 0)),
                        "entry_price": float(getattr(p, "entry_price", 0)),
                    }
                    for p in state.inventory
                ]

        bar_ts = idx
        engs = engines.get(sym, {})
        chop_actions: list = []
        trend_actions: list = []
        for ek in ("chop", "trend"):
            e = engs.get(ek)
            if e:
                try:
                    acts = (
                        e.on_bar(
                            symbol=sym,
                            timestamp=str(bar_ts),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=close,
                            atr=float(row.get("atr14", close * 0.02)),
                            features=f,
                        )
                        or []
                    )
                except Exception as _exc:
                    import logging

                    logging.getLogger("bt").warning(
                        f"on_bar error {sym}/{ek}: {_exc}", exc_info=False
                    )
                    acts = []
                if ek == "chop":
                    chop_actions = acts
                else:
                    trend_actions = acts

        if trend_time_filter:
            valid_entry = _is_valid_entry_time(bar_ts)
            trend_actions_filtered = []
            for a in trend_actions:
                kind = str(a.get("action", "")).lower()
                if kind == "place" and not valid_entry:
                    acct.trades_rej += 1
                    continue
                trend_actions_filtered.append(a)
            trend_actions = trend_actions_filtered

        actions = chop_actions + trend_actions

        if dry_run:
            acct.trades_ok += sum(1 for a in actions if a.get("action") == "place")
            continue

        for a in actions:
            if a.get("action") != "market_exit":
                continue
            side = str(a.get("side", "")).upper()
            qty = float(a.get("quantity", 0))
            exit_px = float(a.get("exit_price", close))
            for _ek, positions in inv_snapshot.items():
                for pos in positions:
                    if pos["side"] == side and abs(pos["quantity"] - qty) < 1e-8:
                        if side == "LONG":
                            pnl = qty * (exit_px - pos["entry_price"])
                        else:
                            pnl = qty * (pos["entry_price"] - exit_px)
                        acct.cum_pnl += pnl
                        acct.daily_pnl += pnl
                        break

        day_key = str(idx.date())
        if acct.current_day and day_key != acct.current_day:
            acct.daily_pnl = 0.0
        acct.current_day = day_key

        ceq = acct.current
        if not acct.halted and ceq <= acct.peak_equity * (1.0 - max_dd):
            acct.halted = True
            acct.halt_reason = f"dd>{max_dd*100:.0f}% at {idx}"

        allowed, rejected = [], []
        for a in actions:
            kind = str(a.get("action", "")).lower()
            if kind == "market_exit":
                allowed.append(a)
                continue
            if kind not in ("place", "cancel", "place_protection"):
                allowed.append(a)
                continue
            if acct.halted:
                rejected.append(a)
                continue
            if daily_loss > 0 and acct.daily_pnl <= -daily_loss:
                rejected.append(a)
                continue
            allowed.append(a)
        acct.trades_rej += len(rejected)

        phase1 = [
            a for a in allowed if a["action"] in ("place", "cancel", "place_protection")
        ]
        results_p1 = adapter.execute_actions(phase1) if phase1 else []
        for eng in engs.values():
            if hasattr(eng, "on_execution_results"):
                eng.on_execution_results(results_p1)
        for _ in range(8):
            follow_ups = []
            for eng in engs.values():
                if hasattr(eng, "pop_pending_actions"):
                    follow_ups.extend(eng.pop_pending_actions() or [])
            if not follow_ups:
                break
            fu_results = adapter.execute_actions(follow_ups)
            for eng in engs.values():
                if hasattr(eng, "on_execution_results"):
                    eng.on_execution_results(fu_results)
            results_p1.extend(fu_results)

        phase2 = [a for a in allowed if a["action"] == "market_exit"]
        results_p2 = adapter.execute_actions(phase2) if phase2 else []
        for eng in engs.values():
            if hasattr(eng, "on_execution_results"):
                eng.on_execution_results(results_p2)
        for _ in range(8):
            follow_ups = []
            for eng in engs.values():
                if hasattr(eng, "pop_pending_actions"):
                    follow_ups.extend(eng.pop_pending_actions() or [])
            if not follow_ups:
                break
            fu_results = adapter.execute_actions(follow_ups)
            for eng in engs.values():
                if hasattr(eng, "on_execution_results"):
                    eng.on_execution_results(fu_results)
            results_p2.extend(fu_results)

        acct.trades_ok += len(phase1) + len(phase2)

        acct.peak_equity = max(acct.peak_equity, acct.current)
        dd = (acct.current - acct.peak_equity) / max(acct.peak_equity, 1.0)
        acct.max_dd_peak = min(acct.max_dd_peak, dd)
        if progress and n_2h % 200 == 0:
            print(f"  [{idx}] eq={acct.current:.0f} dd={dd*100:.1f}%")

    elapsed = time.monotonic() - t0
    meta = {
        "start": str(start),
        "end": str(end),
        "symbols": symbols,
        "elapsed_sec": elapsed,
        "bars_1m": n_1m,
        "bars_2h": n_2h,
        "chop_config": str(chop_config),
        "trend_config": str(trend_config),
        "constitution_yaml": str(constitution_yaml),
    }
    if progress:
        print(f"\n=== Results ({elapsed:.0f}s) ===")
        print(
            f"Equity:  {equity:,.0f} → {acct.current:,.0f}  "
            f"({acct.cum_pnl/equity*100:+.2f}%)"
        )
        print(f"Peak:    {acct.peak_equity:,.0f}")
        print(f"Max DD:  {acct.max_dd_peak*100:.2f}% (from peak)")
        print(f"Trades:  {acct.trades_ok} ok / {acct.trades_rej} rej")
        print(f"Halted:  {acct.halted} {acct.halt_reason}")
        print(f"1m bars: {n_1m}  2h events: {n_2h}")
    return acct, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--equity", type=float, default=50000.0)
    ap.add_argument("--chop-config", required=True)
    ap.add_argument("--trend-config", required=True)
    ap.add_argument(
        "--constitution-yaml",
        default="live/highcap/config/constitution/constitution.yaml",
    )
    ap.add_argument("--no-chop", action="store_true")
    ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--trend-time-filter",
        action="store_true",
        help="Restrict trend_scalp entries to non-Asian (UTC 09:00+) weekdays only",
    )
    ap.add_argument(
        "--save-preload", help="Save pre-computed bars+features to this pickle file"
    )
    ap.add_argument(
        "--load-preload", help="Load pre-computed bars+features from this pickle file"
    )
    ap.add_argument(
        "--summary-json",
        help="Write run summary JSON to this path",
    )
    ap.add_argument(
        "--no-clean-state",
        action="store_true",
        help="Do not delete /tmp/bt_*.json engine state before run",
    )
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start).tz_localize("UTC")
    end = pd.Timestamp(args.end).tz_localize("UTC")
    data_dir = Path(args.data_dir)

    acct, meta = run_timeline_backtest(
        start=start,
        end=end,
        symbols=symbols,
        equity=float(args.equity),
        chop_config=Path(args.chop_config),
        trend_config=Path(args.trend_config),
        constitution_yaml=Path(str(args.constitution_yaml)),
        data_dir=data_dir,
        no_chop=bool(args.no_chop),
        no_trend=bool(args.no_trend),
        dry_run=bool(args.dry_run),
        trend_time_filter=bool(args.trend_time_filter),
        save_preload=Path(args.save_preload) if args.save_preload else None,
        load_preload=Path(args.load_preload) if args.load_preload else None,
        clean_state=not args.no_clean_state,
        progress=True,
    )

    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {**meta, **acct.to_summary()}
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
