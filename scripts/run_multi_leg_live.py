#!/usr/bin/env python3
"""Shadow/testnet daemon for standalone multi-leg strategies.

This runner is separate from ``scripts/run_live.py`` because ``chop_grid`` and
``dual_add_trend`` own multi-leg inventory instead of single ``TradeIntent``
positions.

**Parallel with classic live (BPC / TPC / ME / SRB, etc.)**

- Classic production: ``scripts/run_live.py`` — market data over **Binance
  WebSocket** (ticks / bars) → ``MultiSymbolManager`` → ``GenericLiveStrategy`` →
  ``OrderManager`` (see that file’s docstring).
- Multi-leg: this script — ``--bar-source parquet`` for replay/shadow, or
  ``--bar-source websocket`` to reuse the classic market-data WebSocket stack
  for slow signal features. In   ``--mode testnet`` or ``--mode mainnet``, it also starts **Futures User
  Data Stream WebSocket** (``BinanceUserStream``) for fills/order updates into
  ``MultiLegLiveOrchestrator.on_execution_report``.

**Second Binance account (recommended)**

- Prefer ``MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY`` /
  ``MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET`` for this process so it never
  shares keys with ``run_live`` / ``OrderManager`` (which use
  ``BINANCE_FUTURES_TESTNET_*`` or ``BINANCE_API_KEY`` depending on config).
- If the ``MULTI_LEG_*`` vars are unset, this script refuses to start unless
  ``--allow-shared-account`` is explicitly set.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.strategy_symbols import (  # noqa: E402
    resolve_strategy_symbols,
)
from src.config.strategy_layout import resolve_strategy_config_input  # noqa: E402
from src.order_management.binance_api import BinanceAPI  # noqa: E402
from src.order_management.binance_user_stream import BinanceUserStream  # noqa: E402
from src.order_management.grid_execution_adapter import (
    MultiLegExecutionAdapter,
)  # noqa: E402
from src.order_management.mock_binance_api import MockBinanceAPI  # noqa: E402
from src.order_management.multi_leg_storage import MultiLegStorage  # noqa: E402
from src.order_management.multi_leg_feature_store_provider import (  # noqa: E402
    FeatureStoreBarProvider,
)
from src.order_management.multi_leg_ws_provider import (  # noqa: E402
    MultiLegWebSocketBarProvider,
)
from src.order_management.multi_leg_daemon import (  # noqa: E402
    MultiLegBarEvent,
    MultiLegLiveDaemon,
    StrategyRuntime,
)
from src.order_management.multi_leg_orchestrator import (
    MultiLegLiveOrchestrator,
)  # noqa: E402
from src.order_management.multi_leg_reconciliation import (  # noqa: E402
    MultiLegReconciler,
    ReconciliationPolicy,
)
from src.order_management.multi_leg_risk_governor import (  # noqa: E402
    MultiLegPortfolioRiskGovernor,
    MultiLegRiskLimits,
)
from src.live_data_stream.constitution_config import (  # noqa: E402
    apply_multi_leg_args_from_constitution,
)
from src.time_series_model.live.metrics_exporter import (  # noqa: E402
    start_metrics_server,
    METRICS,
)
from src.time_series_model.live.chop_grid_live_engine import (
    ChopGridLiveEngine,
)  # noqa: E402
from src.time_series_model.live.dual_add_trend_live_engine import (  # noqa: E402
    DualAddTrendLiveEngine,
)


class ParquetFeatureBarProvider:
    """Load the latest completed feature bar from local parquet data."""

    def __init__(
        self,
        *,
        data_dir: str,
        timeframe: str,
        lookback_days: int,
        now: pd.Timestamp | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.timeframe = timeframe
        self.lookback_days = int(lookback_days)
        self.now = now

    def latest_closed_bars(self, symbols: Iterable[str]) -> List[MultiLegBarEvent]:
        # Lazy imports: diagnose_* pulls heavy stacks; unused when --bar-source feature-store.
        from scripts.diagnose_chop_grid import GridConfig, build_features
        from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv
        from src.features.time_series.baseline_features import (
            compute_trend_confidence_from_series,
        )

        now = self.now or pd.Timestamp.utcnow()
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        cutoff = now - pd.Timedelta(self.timeframe)
        start = now - pd.Timedelta(days=self.lookback_days)
        out: List[MultiLegBarEvent] = []
        for symbol in symbols:
            raw = _load_symbol_1m(self.data_dir, symbol, start, now)
            if raw.empty:
                continue
            bars = _resample_ohlcv(raw, self.timeframe)
            if bars.empty:
                continue
            df = build_features(symbol, bars, GridConfig())
            if (
                "trend_confidence" not in df.columns
                or "trend_direction" not in df.columns
            ):
                bundle = compute_trend_confidence_from_series(close=df["close"])
                for col in bundle.columns:
                    df[col] = bundle[col]
            df = df[df.index <= cutoff]
            if df.empty:
                continue
            row = df.iloc[-1]
            ts = df.index[-1]
            out.append(
                MultiLegBarEvent(
                    symbol=symbol,
                    timestamp=str(ts),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    atr=float(row["atr14"]),
                    features=_features_from_row(row),
                )
            )
        return out


def _features_from_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "semantic_chop": float(row.get("semantic_chop", 0.0)),
        "bpc_semantic_chop": float(row.get("semantic_chop", 0.0)),
        "box_prefilter": bool(row.get("box_prefilter", False)),
        "trend_confidence": float(row.get("trend_confidence", 0.0)),
        "trend_direction": str(row.get("trend_direction", "UP")),
    }


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _make_api(mode: str, *, allow_shared_account: bool = False) -> Any:
    if mode == "shadow":
        api = MockBinanceAPI()
        api.hedge_mode = True
        return api
    if mode == "mainnet":
        api_key = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_KEY", "") or os.getenv(
            "MULTI_LEG_BINANCE_API_KEY", ""
        )
        api_secret = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_SECRET", "") or os.getenv(
            "MULTI_LEG_BINANCE_API_SECRET", ""
        )
        using_shared = False
        if (not api_key or not api_secret) and allow_shared_account:
            api_key = os.getenv("BINANCE_API_KEY", "") or os.getenv(
                "BINANCE_FUTURES_API_KEY", ""
            )
            api_secret = os.getenv("BINANCE_API_SECRET", "") or os.getenv(
                "BINANCE_FUTURES_API_SECRET", ""
            )
            using_shared = bool(api_key and api_secret)
        if not api_key or not api_secret:
            raise RuntimeError(
                "mainnet mode requires API keys: set "
                "MULTI_LEG_BINANCE_FUTURES_API_KEY/SECRET (dedicated futures account) "
                "or pass --allow-shared-account to reuse BINANCE_API_KEY/SECRET."
            )
        if using_shared:
            print(
                "WARNING: run_multi_leg_live.py is reusing BINANCE_* for mainnet; "
                "prefer MULTI_LEG_BINANCE_FUTURES_API_KEY/SECRET for isolation.",
                file=sys.stderr,
            )
        return BinanceAPI(
            api_key=api_key, api_secret=api_secret, testnet=False, use_proxy=None
        )
    # testnet
    api_key = os.getenv(
        "MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY",
        "",
    )
    api_secret = os.getenv(
        "MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET",
        "",
    )
    using_shared = False
    if not api_key and not api_secret and allow_shared_account:
        api_key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY", "")
        api_secret = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET", "")
        using_shared = bool(api_key or api_secret)
    if not api_key or not api_secret:
        raise RuntimeError(
            "testnet mode requires API keys: set "
            "MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY/SECRET for the dedicated "
            "multi-leg account. To intentionally reuse BINANCE_FUTURES_TESTNET_* "
            "from the classic live stack, pass --allow-shared-account."
        )
    if using_shared:
        print(
            "WARNING: run_multi_leg_live.py is reusing BINANCE_FUTURES_TESTNET_*; "
            "prefer MULTI_LEG_BINANCE_FUTURES_TESTNET_* for account isolation.",
            file=sys.stderr,
        )
    return BinanceAPI(
        api_key=api_key, api_secret=api_secret, testnet=True, use_proxy=None
    )


def build_daemon(
    args: argparse.Namespace,
) -> Tuple[MultiLegLiveDaemon, Any, MultiLegStorage | None, str | None]:
    apply_multi_leg_args_from_constitution(args)
    symbols = _parse_symbols(args.symbols)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategy_symbols: Dict[str, List[str]] = {}
    active_symbol_set = set()
    for strategy in strategies:
        selected = _resolve_live_strategy_symbols(
            strategy=strategy, args=args, base_symbols=symbols
        )
        if selected:
            strategy_symbols[strategy] = selected
            active_symbol_set.update(selected)
            if selected != symbols:
                print(
                    f"[multi-leg symbols] {strategy}: {selected} "
                    f"(base={len(symbols)} -> selected={len(selected)})"
                )
        else:
            print(f"[multi-leg symbols] {strategy}: no symbols after meta filter, skip")
    if not strategy_symbols:
        raise RuntimeError("No active strategy symbols after meta symbol filters")
    symbols = sorted(active_symbol_set)
    api = _make_api(args.mode, allow_shared_account=args.allow_shared_account)
    if (
        args.mode in ("mainnet", "testnet")
        and isinstance(api, BinanceAPI)
        and _env_truthy("MLBOT_MULTI_LEG_SET_HEDGE_ON_START")
    ):
        if api.hedge_mode_probe_error:
            logger.warning(
                "MLBOT_MULTI_LEG_SET_HEDGE_ON_START ignored: hedge probe failed (%s)",
                api.hedge_mode_probe_error,
            )
        elif not api.hedge_mode:
            try:
                api.set_dual_side_position(True)
                logger.info(
                    "MLBOT_MULTI_LEG_SET_HEDGE_ON_START: requested hedge "
                    "(POST /fapi/v1/positionSide/dual)"
                )
            except Exception as exc:
                logger.warning(
                    "MLBOT_MULTI_LEG_SET_HEDGE_ON_START failed (%s); close positions "
                    "and cancel orders before switching position mode.",
                    exc,
                )
            api.refresh_hedge_mode()
    storage: MultiLegStorage | None = None
    run_id: str | None = None
    db_path = str(getattr(args, "multi_leg_db_path", "") or "").strip()
    if db_path:
        storage = MultiLegStorage(db_path)
        run_id = storage.create_run(
            mode=args.mode,
            strategies=strategies,
            symbols=symbols,
            account_label=(
                "multi_leg_testnet"
                if args.mode == "testnet"
                else ("multi_leg_mainnet" if args.mode == "mainnet" else "shadow")
            ),
            config=vars(args),
        )
    risk_limits = MultiLegRiskLimits(
        max_gross_notional=args.max_gross_notional,
        max_net_notional=args.max_net_notional,
        max_symbol_gross_notional=args.max_symbol_gross_notional,
        max_symbol_net_notional=args.max_symbol_net_notional,
        max_resting_orders=args.max_resting_orders,
        account_equity_usdt=getattr(args, "account_equity_usdt", None),
        max_drawdown_pct=getattr(args, "max_drawdown_pct", None),
    )

    def _drawdown_pct_from_env() -> float | None:
        raw = os.getenv("MULTI_LEG_CURRENT_DRAWDOWN_PCT", "")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    if args.bar_source == "feature-store":
        provider = FeatureStoreBarProvider(
            feature_bus_root=args.feature_bus_root,
            timeframe=args.feature_store_timeframe,
            execution_timeframe=args.feature_store_execution_timeframe,
        )
    elif args.bar_source == "websocket":
        provider = MultiLegWebSocketBarProvider(
            symbols=symbols,
            storage_base_path=args.live_storage_base,
            feature_compute_interval_minutes=args.feature_compute_interval_minutes,
            memory_window_hours=args.memory_window_hours,
            orderflow_window_minutes=args.orderflow_window_minutes,
            feature_4h_interval_hours=args.feature_4h_interval_hours,
            warmup_days=args.warmup_days,
        )
    else:
        provider = ParquetFeatureBarProvider(
            data_dir=args.data_dir,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
        )
    runtimes: List[StrategyRuntime] = []
    for strategy in strategies:
        strat_symbols = strategy_symbols.get(strategy, [])
        for symbol in strat_symbols:
            prefix = "cg" if strategy == "chop_grid" else "dat"
            adapter = MultiLegExecutionAdapter(
                api,
                require_hedge_mode=args.mode != "shadow",
                shadow=args.mode == "shadow",
                client_id_prefix=prefix,
                default_symbol=symbol,
                storage=storage,
                run_id=run_id,
                strategy_name=strategy,
            )
            governor = MultiLegPortfolioRiskGovernor(risk_limits)
            reconciler = MultiLegReconciler(
                ReconciliationPolicy(client_id_prefixes={f"{prefix}_"})
            )
            engine = _make_engine(strategy, symbol=symbol, args=args)
            orchestrator = MultiLegLiveOrchestrator(
                engine=engine,
                governor=governor,
                adapter=adapter,
                reconciler=reconciler,
                execute_reconciliation_actions=True,
                storage=storage,
                run_id=run_id,
                strategy_name=strategy,
                symbol=symbol,
                drawdown_pct_provider=_drawdown_pct_from_env,
            )
            runtimes.append(
                StrategyRuntime(
                    name=strategy,
                    symbol=symbol,
                    engine=engine,
                    orchestrator=orchestrator,
                )
            )
    return (
        MultiLegLiveDaemon(
            bar_provider=provider,
            runtimes=runtimes,
            poll_seconds=args.poll_seconds,
        ),
        api,
        storage,
        run_id,
    )


def _resolve_live_strategy_symbols(
    *, strategy: str, args: argparse.Namespace, base_symbols: List[str]
) -> List[str]:
    cfg_path = ""
    if strategy == "chop_grid":
        cfg_path = str(getattr(args, "chop_grid_config", "") or "").strip()
    elif strategy == "dual_add_trend":
        cfg_path = str(getattr(args, "dual_add_config", "") or "").strip()
    if not cfg_path:
        return list(base_symbols)
    cfg_dir, _profile_path, _engine_path = resolve_strategy_config_input(Path(cfg_path))
    sel = resolve_strategy_symbols(
        strategy=strategy, base_symbols=base_symbols, strategy_config_dir=cfg_dir
    )
    return list(sel.resolved_symbols)


def _make_engine(strategy: str, *, symbol: str, args: argparse.Namespace) -> Any:
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    if strategy == "chop_grid":
        return ChopGridLiveEngine(
            config_path=args.chop_grid_config,
            state_path=state_dir / f"chop_grid_{symbol}.json",
            level_notional=args.unit_notional,
        )
    if strategy == "dual_add_trend":
        return DualAddTrendLiveEngine(
            config_path=args.dual_add_config,
            state_path=state_dir / f"dual_add_trend_{symbol}.json",
            unit_notional=args.unit_notional,
        )
    raise ValueError(f"unsupported strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Mainnet: MULTI_LEG_BINANCE_FUTURES_API_KEY/SECRET or "
            "--allow-shared-account + BINANCE_API_KEY/SECRET. "
            "Testnet: MULTI_LEG_BINANCE_FUTURES_TESTNET_* or shared "
            "BINANCE_FUTURES_TESTNET_*."
        ),
    )
    p.add_argument("--mode", choices=["shadow", "testnet", "mainnet"], default="shadow")
    p.add_argument("--strategies", default="chop_grid,dual_add_trend")
    p.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    p.add_argument("--data-dir", default="data/parquet_data")
    p.add_argument(
        "--bar-source",
        choices=["parquet", "websocket", "feature-store"],
        default="parquet",
    )
    p.add_argument("--live-storage-base", default="data/live_storage")
    p.add_argument("--feature-bus-root", default="live/shared_feature_bus")
    p.add_argument("--feature-store-timeframe", default="2h")
    p.add_argument("--feature-store-execution-timeframe", default="1min")
    p.add_argument("--timeframe", default="2h")
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--warmup-days", type=int, default=0)
    p.add_argument("--memory-window-hours", type=float, default=4.0)
    p.add_argument("--feature-compute-interval-minutes", type=int, default=15)
    p.add_argument("--orderflow-window-minutes", type=int, default=None)
    p.add_argument("--feature-4h-interval-hours", type=int, default=4)
    p.add_argument(
        "--websocket-prime-seconds",
        type=float,
        default=0.0,
        help="When --once and --bar-source=websocket, wait this many seconds before run_once.",
    )
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--max-iterations", type=int, default=0)
    p.add_argument("--state-dir", default="results/multi_leg_live/state")
    p.add_argument("--multi-leg-db-path", default="data/multi_leg_order_management.db")
    p.add_argument("--unit-notional", type=float, default=100.0)
    p.add_argument("--account-equity-usdt", type=float, default=10_000.0)
    p.add_argument("--max-drawdown-pct", type=float, default=0.12)
    p.add_argument("--max-gross-notional", type=float, default=2_000.0)
    p.add_argument("--max-net-notional", type=float, default=1_000.0)
    p.add_argument("--max-symbol-gross-notional", type=float, default=800.0)
    p.add_argument("--max-symbol-net-notional", type=float, default=400.0)
    p.add_argument("--max-resting-orders", type=int, default=60)
    p.add_argument(
        "--constitution-yaml",
        default="",
        help="Optional path; default from MLBOT_STRATEGIES_ROOT/../constitution/constitution.yaml",
    )
    p.add_argument(
        "--allow-shared-account",
        action="store_true",
        help=(
            "testnet: fallback to BINANCE_FUTURES_TESTNET_* when MULTI_LEG_* unset. "
            "mainnet: fallback to BINANCE_API_KEY/SECRET. Not recommended if you "
            "need strict account isolation."
        ),
    )
    p.add_argument(
        "--chop-grid-config",
        default="live/highcap/config/strategies/chop_grid",
    )
    p.add_argument(
        "--dual-add-config",
        default="config/strategies/dual_add_trend/research/calibrate_roll.default.yaml",
    )
    return p.parse_args()


async def async_main() -> None:
    args = parse_args()
    metrics_port = int(os.getenv("MLBOT_METRICS_PORT", "9090"))
    start_metrics_server(port=metrics_port)
    daemon, exchange_api, storage, run_id = build_daemon(args)
    bar_provider = daemon.bar_provider

    user_stream: BinanceUserStream | None = None
    if isinstance(exchange_api, BinanceAPI):

        def on_execution_report(exec_report: Dict[str, Any]) -> None:
            sym = str(exec_report.get("symbol") or "").upper().strip()
            if not sym:
                return
            for rt in daemon.runtimes:
                if rt.symbol.upper() == sym:
                    rt.orchestrator.on_execution_report(exec_report)
                    if storage is not None:
                        storage.record_execution_report(
                            {
                                **dict(exec_report),
                                "run_id": run_id,
                                "strategy": rt.name,
                                "raw": dict(exec_report),
                            }
                        )
                    try:
                        METRICS.multi_leg_user_stream_events_total.labels(
                            strategy=rt.name, symbol=sym
                        ).inc(1)
                    except Exception:
                        pass

        user_stream = BinanceUserStream(exchange_api, on_execution_report)
        await user_stream.start()

    try:
        start = getattr(bar_provider, "start", None)
        if callable(start):
            await start()
        if args.once:
            if args.bar_source == "websocket" and args.websocket_prime_seconds > 0:
                await asyncio.sleep(args.websocket_prime_seconds)
            report = daemon.run_once()
            print(report)
        else:
            max_iterations = args.max_iterations if args.max_iterations > 0 else None
            await daemon.run_forever(max_iterations=max_iterations)
    finally:
        stop = getattr(bar_provider, "stop", None)
        if callable(stop):
            await stop()
        if user_stream is not None:
            await user_stream.stop()
        if storage is not None and run_id is not None:
            storage.finish_run(run_id)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
