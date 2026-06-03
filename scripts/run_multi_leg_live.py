#!/usr/bin/env python3
"""Shadow/testnet daemon for standalone multi-leg strategies.

This runner is separate from ``scripts/run_live.py`` because ``chop_grid`` and
``trend_scalp`` own multi-leg inventory instead of single ``TradeIntent``
positions.

**Parallel with directional trend/fat-tail live (BPC / TPC / ME / SRB, etc.)**

- Feature-bus production: ``scripts/run_market_feature_publisher.py`` owns the
  **Binance market WebSocket** and writes disk Feature Bus snapshots.
- Classic production: ``scripts/run_live.py`` consumes that disk Feature Bus →
  ``MultiSymbolManager`` → ``GenericLiveStrategy`` → ``OrderManager``.
- Multi-leg: this script — ``--bar-source feature-store`` for production bus
  consumption, or ``--bar-source parquet`` for offline replay/shadow. In
  ``--mode testnet`` or ``--mode mainnet``, it also starts **Futures User
  Data Stream WebSocket** (``BinanceUserStream``) for fills/order updates into
  ``MultiLegLiveOrchestrator.on_execution_report``.

**Second Binance account (recommended)**

- Prefer ``MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY`` /
  ``MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET`` for this process so it never
  shares keys with ``run_live`` / ``OrderManager`` (which use
  ``BINANCE_FUTURES_TESTNET_*`` or ``BINANCE_API_KEY`` depending on config).
- If the ``MULTI_LEG_*`` vars are unset, this script refuses to start unless
  ``--allow-shared-account`` is explicitly set.

**Process output:** ``logging`` writes to **stderr** (captured by Docker/systemd).
By default this runner **also** appends the same INFO lines to a **hourly-rotated** (default)
file under ``{--state-dir}/logs/multi_leg_audit.log`` so restarts/crashes do not
lose recent history as long as ``--state-dir`` points at a **mounted volume**
(e.g. Docker ``-v /opt/quant-engine/data:/app/data`` and
``--state-dir data/multi_leg_live/state``).

**Environment (audit file):**

- ``MLBOT_MULTI_LEG_AUDIT_LOG`` — override path (unset or ``default``: use
  ``{state-dir}/logs/multi_leg_audit.log``). Set to ``0`` / ``off`` to disable file logging.
- ``MLBOT_MULTI_LEG_AUDIT_DISABLE`` — if truthy, skip file handler (stderr only).
- ``MLBOT_MULTI_LEG_AUDIT_RETENTION_DAYS`` — rotated files older than this many
  days are deleted on startup (default ``30``). The active log uses
  ``TimedRotatingFileHandler`` (hourly by default, keep about that many hourly backups).
- ``MLBOT_MULTI_LEG_AUDIT_ROTATION`` — ``hour`` / ``hourly`` (default, via
  ``MLBOT_AUDIT_ROTATION``) or ``day`` / ``midnight`` for once-per-day rollover.
- ``MLBOT_HEDGE_BAR_TICK_METRICS`` — if truthy, record ``bar_tick`` on
  ``mlbot_strategy_event_total`` once per processed bar (default: **off**, less noise).
- ``MLBOT_MULTI_LEG_CHAIN_DEBUG`` or ``MLBOT_CHAIN_DEBUG`` — on each **new closed bar**
  with zero engine actions, log inventory/chop/trend snapshot (one line per bar).
- ``MLBOT_MULTI_LEG_RECONCILE_WARN_COOLDOWN_SECONDS`` — minimum seconds between
  WARNING logs for repeated ``reconcile not ok`` (default ``300``; set ``0`` to log
  every time at WARNING).
- ``MLBOT_MULTI_LEG_ORDER_BACKFILL_INTERVAL_SECONDS`` — periodic REST backfill for
  ``multi_leg_orders`` rows (default ``60`` when unset; ``0``/``off`` to disable).
- ``MLBOT_MULTI_LEG_ORDER_BACKFILL_LOOKBACK_HOURS`` / ``..._LIMIT`` — backfill
  candidate window + per-pass max rows (defaults ``168`` / ``200``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import time
from collections import Counter
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
from src.order_management.multi_leg_daemon import (  # noqa: E402
    MultiLegBarEvent,
    MultiLegLiveDaemon,
    StrategyRuntime,
)
from src.order_management.multi_leg_orchestrator import (
    MultiLegLiveOrchestrator,
)  # noqa: E402
from src.order_management.multi_leg_order_backfill import (  # noqa: E402
    multi_leg_backfill_enabled,
    multi_leg_backfill_interval_seconds,
    periodic_multi_leg_order_backfill,
)
from src.order_management.multi_leg_reconciliation import (  # noqa: E402
    MultiLegReconciler,
    ReconciliationPolicy,
)
from src.order_management.multi_leg_risk_governor import (  # noqa: E402
    MultiLegPortfolioRiskGovernor,
    MultiLegRiskLimits,
)
from src.order_management.chop_grid_concurrency import (  # noqa: E402
    ChopGridConcurrencyGate,
)
from src.live_data_stream.constitution_config import (  # noqa: E402
    apply_multi_leg_args_from_constitution,
    load_constitution_dict,
    multi_leg_section,
    resolve_constitution_yaml,
    resolve_multi_leg_risk_limits_from_constitution,
)
from src.time_series_model.core.constitution.account_risk_guard import (  # noqa: E402
    snapshot_from_binance_balance,
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
from src.time_series_model.live.non_trend_funnel import (  # noqa: E402
    FifteenMinFlusher,
    default_live_monitor_db_path,
)
from src.time_series_model.live.stats_collector import StatsCollector  # noqa: E402
from scripts.live_audit_file import configure_audit_from_env_defaults  # noqa: E402


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


_FALLBACK_MULTI_LEG_SYMBOLS = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT"


def _read_universe_yaml_symbols(universe: str) -> List[str]:
    """Load symbol keys from ``live/{universe}/universe.yaml`` (same keys as ``start_live.sh``)."""

    import yaml

    path = PROJECT_ROOT / "live" / str(universe).strip() / "universe.yaml"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    symbols = sorted((cfg.get("symbols") or {}).keys())
    out = [str(s).strip().upper() for s in symbols if str(s).strip()]
    return out


def resolve_multi_leg_base_symbols_csv(args: argparse.Namespace) -> str:
    """Resolve comma-separated base symbols.

    Mirrors ``live/scripts/start_live.sh`` defaults: optional explicit ``--symbols``,
    else keys from ``live/{universe}/universe.yaml``.
    """

    cli = getattr(args, "symbols", None)
    if cli is not None and str(cli).strip():
        return str(cli).strip()

    universe = (
        str(getattr(args, "universe", "highcap") or "highcap").strip() or "highcap"
    )
    try:
        uni = _read_universe_yaml_symbols(universe)
    except FileNotFoundError:
        uni = []
    if uni:
        return ",".join(uni)

    env_syms = os.environ.get("MLBOT_MULTI_LEG_SYMBOLS", "").strip()
    if env_syms:
        return env_syms

    return _FALLBACK_MULTI_LEG_SYMBOLS


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
            "from the trend/fat-tail live stack, pass --allow-shared-account."
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


def _multi_leg_account_snapshot_provider(api: Any):
    def provider():
        return snapshot_from_binance_balance(
            balance=api.get_account_balance(),
            positions=api.get_positions(),
        )

    return provider


def build_daemon(
    args: argparse.Namespace,
) -> Tuple[MultiLegLiveDaemon, Any, MultiLegStorage | None, str | None]:
    apply_multi_leg_args_from_constitution(args)
    resolved_csv = resolve_multi_leg_base_symbols_csv(args)
    setattr(args, "symbols", resolved_csv)
    symbols = _parse_symbols(resolved_csv)
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
        and not _env_truthy("MLBOT_MULTI_LEG_SKIP_HEDGE_ON_START")
    ):
        if api.hedge_mode_probe_error:
            logger.warning(
                "multi-leg hedge: skipping API switch until probe succeeds (%s)",
                api.hedge_mode_probe_error,
            )
        elif not api.hedge_mode:
            try:
                api.set_dual_side_position(True)
                logger.info(
                    "multi-leg hedge: POST /fapi/v1/positionSide/dual (dualSide=true)"
                )
            except Exception as exc:
                logger.warning(
                    "multi-leg hedge: switch failed (%s); close positions and cancel "
                    "orders before switching position mode.",
                    exc,
                )
            api.refresh_hedge_mode()
            if not api.hedge_mode_probe_error and not api.hedge_mode:
                for settle_attempt in range(5):
                    time.sleep(2.0)
                    api.refresh_hedge_mode()
                    if api.hedge_mode:
                        logger.info(
                            "multi-leg hedge: confirmed after settle (%s/5)",
                            settle_attempt + 1,
                        )
                        break
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
    _const_path = resolve_constitution_yaml(
        os.getenv("MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies"),
        override=getattr(args, "constitution_yaml", None),
    )
    _ml_limits = resolve_multi_leg_risk_limits_from_constitution(
        {"multi_leg": multi_leg_section(load_constitution_dict(_const_path))}
    )
    risk_limits = MultiLegRiskLimits(
        max_gross_notional=args.max_gross_notional,
        max_net_notional=args.max_net_notional,
        max_symbol_gross_notional=args.max_symbol_gross_notional,
        max_symbol_net_notional=args.max_symbol_net_notional,
        max_resting_orders=args.max_resting_orders,
        account_equity_usdt=getattr(args, "account_equity_usdt", None),
        max_drawdown_pct=getattr(args, "max_drawdown_pct", None),
        account_risk_limits=dict(_ml_limits.get("account_risk_limits") or {}),
    )
    _account_snapshot_provider = None
    if args.mode in ("mainnet", "testnet") and isinstance(api, BinanceAPI):
        _account_snapshot_provider = _multi_leg_account_snapshot_provider(api)

    def _drawdown_pct_from_env() -> float | None:
        raw = os.getenv("MULTI_LEG_CURRENT_DRAWDOWN_PCT", "")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    if args.bar_source == "feature-store":
        from src.live_data_stream.feature_bus import resolve_disk_primary_timeframe

        bus_tf, bus_legacy = resolve_disk_primary_timeframe(
            args.feature_bus_root,
            args.feature_store_timeframe,
        )
        if bus_legacy:
            logger.warning(
                "multi-leg feature bus: reading legacy features/primary/ "
                "(requested tf=%s)",
                args.feature_store_timeframe,
            )
        provider = FeatureStoreBarProvider(
            feature_bus_root=args.feature_bus_root,
            timeframe=bus_tf,
            execution_timeframe=args.feature_store_execution_timeframe,
            initial_backfill_bars=args.feature_store_initial_backfill_bars,
        )
    else:
        provider = ParquetFeatureBarProvider(
            data_dir=args.data_dir,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
        )
    runtimes: List[StrategyRuntime] = []
    max_cg = int(getattr(args, "max_concurrent_grid_symbols", 0) or 0)
    chop_grid_gate = ChopGridConcurrencyGate(max_cg) if max_cg > 0 else None
    multi_engine_symbols = {
        s
        for s, n in Counter(
            str(sym).strip().upper()
            for _st in strategies
            for sym in strategy_symbols.get(_st, [])
        ).items()
        if int(n or 0) >= 2
    }
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
            governor = MultiLegPortfolioRiskGovernor(
                risk_limits,
                account_snapshot_provider=_account_snapshot_provider,
            )
            sym_u = str(symbol).strip().upper()
            reconciler = MultiLegReconciler(
                ReconciliationPolicy(
                    client_id_prefixes={f"{prefix}_"},
                    cancel_orphan_exchange_orders=True,
                    skip_position_reconciliation=(sym_u in multi_engine_symbols),
                )
            )
            engine = _make_engine(strategy, symbol=symbol, args=args)
            if strategy == "chop_grid" and chop_grid_gate is not None:
                chop_grid_gate.register(symbol, engine)
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
    stats_collector: StatsCollector | None = None
    funnel_flusher: FifteenMinFlusher | None = None
    if not _env_truthy("MLBOT_MULTI_LEG_FUNNEL_DISABLE"):
        try:
            stats_collector = StatsCollector(
                db_path=str(default_live_monitor_db_path()),
                auto_cleanup=False,
            )
            funnel_flusher = FifteenMinFlusher(
                stats_collector,
                interval_s=float(
                    os.getenv("MLBOT_MULTI_LEG_FUNNEL_FLUSH_SECONDS", "900")
                ),
            )
            logger.info(
                "multi-leg funnel: writing 15min snapshots to %s",
                default_live_monitor_db_path(),
            )
        except Exception:
            logger.exception(
                "multi-leg funnel: StatsCollector init failed; funnel disabled"
            )
            stats_collector = None
            funnel_flusher = None

    return (
        MultiLegLiveDaemon(
            bar_provider=provider,
            runtimes=runtimes,
            poll_seconds=args.poll_seconds,
            reconcile_interval_seconds=getattr(
                args, "reconcile_interval_seconds", 60.0
            ),
            stats_collector=stats_collector,
            funnel_flusher=funnel_flusher,
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
    elif strategy in ("dual_add_trend", "trend_scalp"):
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
    units = getattr(args, "unit_notional_by_strategy", None) or {}
    if strategy == "chop_grid":
        unit = float(units.get("chop_grid", getattr(args, "unit_notional", 100.0)))
        return ChopGridLiveEngine(
            config_path=args.chop_grid_config,
            state_path=state_dir / f"chop_grid_{symbol}.json",
            level_notional=unit,
            metrics_strategy=strategy,
            bar_simulation=args.mode == "shadow",
        )
    if strategy in ("dual_add_trend", "trend_scalp"):
        unit = float(units.get("trend_scalp", getattr(args, "unit_notional", 100.0)))
        return DualAddTrendLiveEngine(
            config_path=args.dual_add_config,
            state_path=state_dir / f"trend_scalp_{symbol}.json",
            unit_notional=unit,
            metrics_strategy=strategy,
        )
    raise ValueError(f"unsupported strategy: {strategy}")


def _configure_multi_leg_file_logging(args: argparse.Namespace) -> None:
    configure_audit_from_env_defaults(
        default_log_file=Path(args.state_dir) / "logs" / "multi_leg_audit.log",
        disable_env="MLBOT_MULTI_LEG_AUDIT_DISABLE",
        path_env="MLBOT_MULTI_LEG_AUDIT_LOG",
        retention_env="MLBOT_MULTI_LEG_AUDIT_RETENTION_DAYS",
        rotation_env="MLBOT_MULTI_LEG_AUDIT_ROTATION",
        banner="multi-leg audit file",
    )


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
    p.add_argument("--strategies", default="chop_grid,trend_scalp")
    p.add_argument(
        "--universe",
        default="highcap",
        help=(
            "When --symbols is omitted: load sorted symbol keys from "
            "live/{universe}/universe.yaml (matches start_live.sh with no argv2)."
        ),
    )
    p.add_argument(
        "--symbols",
        default=None,
        help=(
            "Comma-separated base symbols. Default from universe.yaml, then optionally "
            "MLBOT_MULTI_LEG_SYMBOLS env if YAML is missing; else baked fallback."
        ),
    )
    p.add_argument("--data-dir", default="data/parquet_data")
    p.add_argument(
        "--bar-source",
        choices=["parquet", "feature-store"],
        default="parquet",
    )
    p.add_argument("--feature-bus-root", default="live/shared_feature_bus")
    p.add_argument("--feature-store-timeframe", default="2h")
    p.add_argument("--feature-store-execution-timeframe", default="1min")
    p.add_argument("--feature-store-initial-backfill-bars", type=int, default=1)
    p.add_argument("--timeframe", default="2h")
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--reconcile-interval-seconds", type=float, default=60.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--max-iterations", type=int, default=0)
    p.add_argument("--state-dir", default="data/multi_leg_live/state")
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
        "--max-concurrent-grid-symbols",
        type=int,
        default=0,
        help="Max chop_grid symbols with an active segment at once (0=constitution/default off)",
    )
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
        default="config/strategies/trend_scalp/research/calibrate_roll.default.yaml",
    )
    return p.parse_args()


async def _periodic_process_metrics() -> None:
    interval = int(os.getenv("MLBOT_MARKET_DATA_INTERVAL", "30"))
    loop = asyncio.get_running_loop()
    while True:
        try:
            await loop.run_in_executor(None, METRICS.update_system_health)
            await loop.run_in_executor(None, METRICS.update_account_data)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug("multi-leg metrics update skipped: %s", exc)
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break


async def async_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    _configure_multi_leg_file_logging(args)

    metrics_port = int(os.getenv("MLBOT_METRICS_PORT", "9090"))
    start_metrics_server(port=metrics_port)
    daemon, exchange_api, storage, run_id = build_daemon(args)
    bar_provider = daemon.bar_provider
    try:
        METRICS.publish_dashboard_catalog(
            strategies=sorted({str(rt.name).strip().lower() for rt in daemon.runtimes}),
            symbols=sorted({str(rt.symbol).strip().upper() for rt in daemon.runtimes}),
        )
    except Exception:
        logger.debug("dashboard catalog publish skipped", exc_info=True)

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

    metrics_task = asyncio.create_task(_periodic_process_metrics())
    backfill_task: asyncio.Task | None = None
    if multi_leg_backfill_interval_seconds() > 0 and multi_leg_backfill_enabled(
        exchange_api, storage
    ):
        backfill_task = asyncio.create_task(
            periodic_multi_leg_order_backfill(
                api=exchange_api,
                storage=storage,
                startup_delay_seconds=20.0,
            )
        )
    try:
        start = getattr(bar_provider, "start", None)
        if callable(start):
            await start()
        if args.once:
            report = daemon.run_once()
            print(report)
        else:
            max_iterations = args.max_iterations if args.max_iterations > 0 else None
            await daemon.run_forever(max_iterations=max_iterations)
    finally:
        metrics_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await metrics_task
        if backfill_task is not None:
            backfill_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await backfill_task
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
