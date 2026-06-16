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
from dataclasses import dataclass
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


def _aggregate_2h_ohlc(
    bars_1m: pd.DataFrame,
) -> Dict[Tuple[str, pd.Timestamp], Dict[str, float]]:
    if bars_1m.empty:
        return {}
    tmp = bars_1m.copy()
    tmp["bar_2h"] = tmp.index.floor("2h")
    out: Dict[Tuple[str, pd.Timestamp], Dict[str, float]] = {}
    for (sym, bar_2h), grp in tmp.groupby(["symbol", "bar_2h"], sort=False):
        c = float(grp["close"].iloc[-1])
        out[(str(sym), pd.Timestamp(bar_2h))] = {
            "open": float(grp["open"].iloc[0]),
            "high": float(grp["high"].max()),
            "low": float(grp["low"].min()),
            "close": c,
            "atr14": (
                float(grp["atr14"].iloc[-1]) if "atr14" in grp.columns else c * 0.02
            ),
        }
    return out


@dataclass(frozen=True)
class TimelineRuntime:
    name: str
    symbol: str
    engine: Any
    orchestrator: Any
    fee_bps: float
    client_id_prefix: str = ""


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
    compound_sizing: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    from scripts.multileg_timeline_account import MultilegTimelineAccount
    from scripts.multileg_timeline_sizing import (
        build_risk_limits,
        sync_multileg_timeline_sizing,
    )
    from src.order_management.chop_grid_concurrency import MultiLegConcurrencyGate
    from src.order_management.grid_execution_adapter import (
        MultiLegExecutionAdapter,
        MultiLegExecutionResult,
    )
    from src.order_management.mock_binance_api import MockBinanceAPI
    from src.order_management.multi_leg_orchestrator import MultiLegLiveOrchestrator
    from src.order_management.multi_leg_reconciliation import (
        MultiLegReconciler,
        ReconciliationPolicy,
    )
    from src.order_management.multi_leg_risk_governor import (
        MultiLegPortfolioRiskGovernor,
    )
    from src.order_management.multileg_symbol_owner import (
        filter_places_for_owner,
        refresh_symbol_owner,
    )
    from src.live_data_stream.constitution_config import (
        load_constitution_dict,
        multi_leg_section,
    )
    from src.time_series_model.live.chop_grid_live_engine import ChopGridLiveEngine
    from src.time_series_model.live.dual_add_trend_live_engine import (
        DualAddTrendLiveEngine,
    )

    run_id = run_id or str(os.getpid())
    if clean_state:
        clean_bt_state(symbols, run_id=run_id)

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
    ohlc_2h = _aggregate_2h_ohlc(bars_1m)
    if progress:
        print(f"  {len(bars_1m)} 1m bars, {bars_1m['symbol'].nunique()} symbols\n")

    const = load_constitution_dict(constitution_yaml)
    ml = multi_leg_section(const)
    rs = ml.get("risk_limits", {})
    ks = const.get("kill_switch", {})
    max_dd = float(ks.get("max_dd", 0.20))
    max_sym = int(rs.get("max_concurrent_multi_leg_symbols", 6))
    cd = int(rs.get("strategy_switch_cooldown_bars", 3))
    daily_starts = int(rs.get("max_segment_starts_per_symbol_per_day", 0) or 0)
    daily_loss = float(ks.get("daily_loss_limit", 0.06)) * equity

    mock = MockBinanceAPI(initial_wallet_usdt=equity)
    mock.hedge_mode = True
    account = MultilegTimelineAccount(initial_equity=equity, mock=mock)
    governor = MultiLegPortfolioRiskGovernor(
        build_risk_limits(ml=ml, const=const, equity_usdt=equity)
    )

    gate = MultiLegConcurrencyGate(
        max_sym, cooldown_bars=cd, max_segment_starts_per_symbol_per_day=daily_starts
    )
    engines: Dict[str, Dict[str, Any]] = {}
    runtimes: List[TimelineRuntime] = []
    multi_engine_symbols = set()

    def _fee_bps(engine: Any) -> float:
        maker = getattr(engine, "_maker_fee_bps", None)
        if callable(maker):
            try:
                return float(maker())
            except Exception:
                pass
        cfg = getattr(engine, "cfg", None)
        if cfg is not None:
            return float(getattr(cfg, "fee_bps", 4.0) or 4.0)
        return 4.0

    def _add_runtime(
        *,
        name: str,
        sym: str,
        engine: Any,
        prefix: str,
    ) -> None:
        skip_pos = sym in multi_engine_symbols
        adapter = MultiLegExecutionAdapter(
            mock,
            require_hedge_mode=False,
            shadow=False,
            client_id_prefix=prefix,
            default_symbol=sym,
            strategy_name=name,
        )
        reconciler = MultiLegReconciler(
            ReconciliationPolicy(
                client_id_prefixes={f"{prefix}_"},
                cancel_orphan_exchange_orders=False,
                skip_position_reconciliation=skip_pos,
            )
        )
        orchestrator = MultiLegLiveOrchestrator(
            engine=engine,
            governor=governor,
            adapter=adapter,
            reconciler=reconciler,
            execute_reconciliation_actions=False,
            strategy_name=name,
            symbol=sym,
            drawdown_pct_provider=account.drawdown_pct,
        )
        runtimes.append(
            TimelineRuntime(
                name=name,
                symbol=sym,
                engine=engine,
                orchestrator=orchestrator,
                fee_bps=_fee_bps(engine),
                client_id_prefix=prefix,
            )
        )

    for sym in symbols:
        engines[sym] = {}
        if not no_chop:
            ce = ChopGridLiveEngine(
                config_path=chop_config,
                state_path=Path(f"/tmp/bt_chop_{run_id}_{sym}.json"),
                level_notional=4000.0,
                metrics_strategy="bt",
                bar_simulation=False,
            )
            ce.state.symbol = sym
            gate.register(sym, ce, strategy="chop_grid")
            engines[sym]["chop"] = ce
        if not no_trend:
            te = DualAddTrendLiveEngine(
                config_path=trend_config,
                state_path=Path(f"/tmp/bt_trend_{run_id}_{sym}.json"),
                unit_notional=9000.0,
                metrics_strategy="bt",
            )
            te.state.symbol = sym
            gate.register(sym, te, strategy="trend_scalp")
            engines[sym]["trend"] = te
        if (not no_chop) and (not no_trend):
            multi_engine_symbols.add(sym)

    for sym in symbols:
        if not no_chop:
            _add_runtime(
                name="chop_grid", sym=sym, engine=engines[sym]["chop"], prefix="cg"
            )
    for sym in symbols:
        if not no_trend:
            _add_runtime(
                name="trend_scalp", sym=sym, engine=engines[sym]["trend"], prefix="dat"
            )

    units = sync_multileg_timeline_sizing(
        engines=engines,
        ml=ml,
        const=const,
        equity_usdt=equity,
        initial_equity=equity,
        compound_sizing=compound_sizing,
        governors=[governor],
    )
    if progress:
        print(
            f"Constitution: gross≤{governor.limits.max_gross_notional:.0f} "
            f"daily≤{daily_loss:.0f} dd≤{max_dd*100:.0f}% sym≤{max_sym}"
        )
        print(
            f"Sizing: chop={units['chop_grid']:.0f}/lvl "
            f"trend={units['trend_scalp']:.0f}/leg "
            f"compound={compound_sizing}\n"
        )

    if progress:
        print(f"=== Timeline Backtest ({len(bars_1m)} bars) ===\n")

    def _pending_fill_to_result(fill: Dict[str, Any]) -> MultiLegExecutionResult:
        """Convert a match_pending_orders fill dict to MultiLegExecutionResult."""
        return MultiLegExecutionResult(
            action="market_exit" if fill.get("reduce_only") else "place",
            status="filled",
            symbol=str(fill.get("symbol", "")),
            order_id=str(fill.get("order_id", "")),
            client_order_id=str(fill.get("client_order_id", "")),
            raw={
                **fill,
                "local_order_id": fill.get("client_order_id", fill.get("order_id", "")),
            },
        )

    t0 = time.monotonic()
    last_2h: Dict[str, pd.Timestamp] = {}
    symbol_owner: Dict[str, str] = {}
    n_1m = 0
    n_2h = 0

    for idx, row in bars_1m.iterrows():
        n_1m += 1
        sym = str(row["symbol"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # --- Every 1m bar: update price & match pending orders ---
        mock.set_price(sym, close)
        fills = mock.match_pending_orders(sym, high, low)
        if fills:
            account.record_pending_fills(fills)
            # Feed fill events to the engine that placed the order
            for fill in fills:
                cid = str(fill.get("client_order_id", ""))
                for rt in runtimes:
                    if rt.symbol == sym and cid.startswith(f"{rt.client_id_prefix}_"):
                        try:
                            rt.engine.on_execution_results(
                                [_pending_fill_to_result(fill)]
                            )
                        except Exception:
                            pass
                        break

        # --- Only process engine on_bar at 2h boundaries ---
        bar_2h = idx.floor("2h")
        if sym in last_2h and bar_2h <= last_2h[sym]:
            continue
        last_2h[sym] = bar_2h
        n_2h += 1

        agg = ohlc_2h.get((sym, bar_2h))
        if agg is None:
            continue
        f = _lookup(feats, sym, bar_2h)
        if not f:
            continue

        if daily_starts > 0:
            gate.set_evaluation_utc_day(
                pd.Timestamp(bar_2h).tz_convert("UTC").strftime("%Y-%m-%d")
            )

        bar_ts = bar_2h
        matching = [rt for rt in runtimes if rt.symbol == sym]

        for rt in matching:
            if trend_time_filter and rt.name == "trend_scalp":
                if not _is_valid_entry_time(bar_ts):
                    pass
            try:
                actions = (
                    rt.engine.on_bar(
                        symbol=sym,
                        timestamp=str(bar_ts),
                        high=float(agg["high"]),
                        low=float(agg["low"]),
                        close=float(agg["close"]),
                        atr=float(agg.get("atr14", close * 0.02)),
                        features=f,
                    )
                    or []
                )
            except Exception as _exc:
                import logging

                logging.getLogger("bt").warning(
                    f"on_bar error {sym}/{rt.name}: {_exc}", exc_info=False
                )
                actions = []

            if trend_time_filter and rt.name == "trend_scalp":
                filtered = []
                for a in actions:
                    if str(
                        a.get("action", "")
                    ).lower() == "place" and not _is_valid_entry_time(bar_ts):
                        account.trades_rej += 1
                        continue
                    filtered.append(a)
                actions = filtered

            refresh_symbol_owner(runtimes, symbol_owner, sym)
            owner = symbol_owner.get(sym.upper(), "")
            actions, dropped = filter_places_for_owner(
                actions, owner=owner, runtime_name=rt.name
            )

            if account.halted or account.daily_loss_blocks_new_entries(daily_loss):
                kept = []
                for a in actions:
                    kind = str(a.get("action", "")).lower()
                    if kind in {"market_exit", "cancel", "cancel_protection"}:
                        kept.append(a)
                    elif kind == "place":
                        account.trades_rej += 1
                    else:
                        kept.append(a)
                actions = kept

            if account.halted:
                continue

            if dry_run:
                account.trades_ok += sum(
                    1 for a in actions if a.get("action") == "place"
                )
                continue

            report = rt.orchestrator.run_actions(actions, reconcile=False)
            account.record_orchestration(report, symbol_conflict_drops=dropped)
            account.record_execution_results(
                report.execution_results,
                strategy=rt.name,
                fee_bps=rt.fee_bps,
            )
            refresh_symbol_owner(runtimes, symbol_owner, sym)

        if dry_run:
            continue

        account.sync_engine_realized_bridge(engines)
        if compound_sizing:
            sync_multileg_timeline_sizing(
                engines=engines,
                ml=ml,
                const=const,
                equity_usdt=account.current,
                initial_equity=equity,
                compound_sizing=True,
                governors=[governor],
            )

        day_key = str(idx.date())
        was_halted_before = account.halted
        account.on_bar_close(
            day_key=day_key,
            max_dd=max_dd,
            daily_loss_limit=daily_loss,
            ts_label=str(bar_2h),
        )
        # On fresh halt: cancel all non-reduce_only pending orders (entry LIMITs)
        # so they don't fill after the account is already dead.
        if account.halted and not was_halted_before:
            mock.cancel_all_pending_entries()
        # Don't break on halt — continue so pending orders can still match
        # and market_exit/cancel actions can execute on future bars.
        # The governor already blocks risk-increasing actions when halted.
        if progress and n_2h % 200 == 0:
            dd = account.drawdown_pct() * 100.0
            print(f"  [{bar_2h}] eq={account.current:.0f} dd={dd:.1f}%")

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
        "compound_sizing": compound_sizing,
    }
    if progress:
        ret = (account.current - equity) / max(equity, 1.0) * 100.0
        print(f"\n=== Results ({elapsed:.0f}s) ===")
        print(f"Equity:  {equity:,.0f} → {account.current:,.0f}  ({ret:+.2f}%)")
        print(f"Peak:    {account.peak_equity:,.0f}")
        print(f"Max DD:  {account.max_dd_peak*100:.2f}% (from peak)")
        print(f"Trades:  {account.trades_ok} ok / {account.trades_rej} rej")
        print(f"Halted:  {account.halted} {account.halt_reason}")
        print(f"1m bars: {n_1m}  2h events: {n_2h}")
    return account, meta


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
        "--no-compound-sizing",
        action="store_true",
        help="Keep initial-equity sizing (no compound refresh per bar)",
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
        compound_sizing=not args.no_compound_sizing,
    )

    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {**meta, **acct.to_summary()}
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
