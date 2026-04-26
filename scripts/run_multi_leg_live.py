#!/usr/bin/env python3
"""Shadow/testnet daemon for standalone multi-leg strategies.

This runner is separate from ``scripts/run_live.py`` because chop_grid and
dual_add_trend own multi-leg inventory instead of single TradeIntent positions.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_chop_grid import GridConfig, build_features  # noqa: E402
from scripts.diagnose_crf_edge import _load_symbol_1m, _resample_ohlcv  # noqa: E402
from scripts.diagnose_dual_add_trend import _add_trend_features  # noqa: E402
from src.order_management.binance_api import BinanceAPI  # noqa: E402
from src.order_management.grid_execution_adapter import (
    GridExecutionAdapter,
)  # noqa: E402
from src.order_management.mock_binance_api import MockBinanceAPI  # noqa: E402
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
            df = _add_trend_features(df)
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


def _make_api(mode: str) -> Any:
    if mode == "shadow":
        api = MockBinanceAPI()
        api.hedge_mode = True
        return api
    api_key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY", "")
    api_secret = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError(
            "testnet mode requires BINANCE_FUTURES_TESTNET_API_KEY/SECRET"
        )
    return BinanceAPI(
        api_key=api_key, api_secret=api_secret, testnet=True, use_proxy=None
    )


def build_daemon(args: argparse.Namespace) -> MultiLegLiveDaemon:
    symbols = _parse_symbols(args.symbols)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    api = _make_api(args.mode)
    risk_limits = MultiLegRiskLimits(
        max_gross_notional=args.max_gross_notional,
        max_net_notional=args.max_net_notional,
        max_symbol_gross_notional=args.max_symbol_gross_notional,
        max_symbol_net_notional=args.max_symbol_net_notional,
        max_resting_orders=args.max_resting_orders,
    )
    provider = ParquetFeatureBarProvider(
        data_dir=args.data_dir,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
    )
    runtimes: List[StrategyRuntime] = []
    for symbol in symbols:
        for strategy in strategies:
            prefix = "cg" if strategy == "chop_grid" else "dat"
            adapter = GridExecutionAdapter(
                api,
                require_hedge_mode=args.mode != "shadow",
                shadow=args.mode == "shadow",
                client_id_prefix=prefix,
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
            )
            runtimes.append(
                StrategyRuntime(
                    name=strategy,
                    symbol=symbol,
                    engine=engine,
                    orchestrator=orchestrator,
                )
            )
    return MultiLegLiveDaemon(
        bar_provider=provider,
        runtimes=runtimes,
        poll_seconds=args.poll_seconds,
    )


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["shadow", "testnet"], default="shadow")
    p.add_argument("--strategies", default="chop_grid,dual_add_trend")
    p.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    p.add_argument("--data-dir", default="data/parquet_data")
    p.add_argument("--timeframe", default="2h")
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--max-iterations", type=int, default=0)
    p.add_argument("--state-dir", default="results/multi_leg_live/state")
    p.add_argument("--unit-notional", type=float, default=100.0)
    p.add_argument("--max-gross-notional", type=float, default=2_000.0)
    p.add_argument("--max-net-notional", type=float, default=1_000.0)
    p.add_argument("--max-symbol-gross-notional", type=float, default=800.0)
    p.add_argument("--max-symbol-net-notional", type=float, default=400.0)
    p.add_argument("--max-resting-orders", type=int, default=60)
    p.add_argument(
        "--chop-grid-config", default="config/strategies/chop_grid/grid.yaml"
    )
    p.add_argument(
        "--dual-add-config", default="config/strategies/dual_add_trend/dual_add.yaml"
    )
    return p.parse_args()


async def async_main() -> None:
    args = parse_args()
    daemon = build_daemon(args)
    if args.once:
        report = daemon.run_once()
        print(report)
        return
    max_iterations = args.max_iterations if args.max_iterations > 0 else None
    await daemon.run_forever(max_iterations=max_iterations)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
