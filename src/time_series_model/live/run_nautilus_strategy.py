#!/usr/bin/env python3
"""
Run Nautilus Trader Strategy with Feature Engineering.

Usage (recommended):
    python -m time_series_model.live.run_nautilus_strategy \\
        --strategy sr_reversal \\
        --symbol BTCUSDT-PERP \\
        --timeframe 15T \\
        --testnet

Environment Variables:
    BINANCE_API_KEY: Binance API key
    BINANCE_API_SECRET: Binance API secret
    BINANCE_FUTURES_TESTNET_API_KEY: Binance Futures testnet API key (optional)
    BINANCE_FUTURES_TESTNET_API_SECRET: Binance Futures testnet API secret (optional)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path (needed when running as a module)
project_root = next(
    (p for p in Path(__file__).resolve().parents if (p / "setup.py").exists()),
    Path(__file__).resolve().parents[4],
)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from nautilus_trader.config import TradingNodeConfig
    from nautilus_trader.config import CacheConfig
    from nautilus_trader.config import DatabaseConfig
    from nautilus_trader.config import LiveDataEngineConfig
    from nautilus_trader.config import LiveExecEngineConfig
    from nautilus_trader.config import LiveRiskEngineConfig
    from nautilus_trader.config import PortfolioConfig
    from nautilus_trader.adapters.binance import BINANCE
    from nautilus_trader.adapters.binance import BinanceDataClientConfig
    from nautilus_trader.adapters.binance import BinanceExecClientConfig
    from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
    from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
    from nautilus_trader.adapters.binance import BinanceLiveExecClientFactory
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.model import InstrumentId
    from nautilus_trader.model import BarType
    from nautilus_trader.model import BarSpecification
    from nautilus_trader.model import BarAggregation
    from nautilus_trader.model import PriceType
    from nautilus_trader.model import AggregationSource

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    print("❌ Nautilus Trader is not installed.")
    print("Install it with: pip install nautilus-trader")
    sys.exit(1)

from src.time_series_model.live.nautilus_strategy_with_features import (
    NautilusStrategyWithFeatures,
)


def parse_timeframe(timeframe_str: str) -> tuple[int, BarAggregation]:
    """
    Parse timeframe string (e.g., "15T", "1H") into minutes and aggregation.

    Args:
        timeframe_str: Timeframe string (e.g., "15T", "1H", "1D")

    Returns:
        Tuple of (minutes, BarAggregation)
    """
    timeframe_str = timeframe_str.upper()

    if timeframe_str.endswith("T") or timeframe_str.endswith("MIN"):
        minutes = int(timeframe_str.rstrip("TMIN"))
        return minutes, BarAggregation.MINUTE
    elif timeframe_str.endswith("H"):
        hours = int(timeframe_str.rstrip("H"))
        return hours * 60, BarAggregation.HOUR
    elif timeframe_str.endswith("D"):
        days = int(timeframe_str.rstrip("D"))
        return days * 1440, BarAggregation.DAY
    else:
        raise ValueError(f"Invalid timeframe: {timeframe_str}")


def create_trading_node_config(
    strategy_name: str,
    testnet: bool = True,
    use_cache: bool = False,
) -> TradingNodeConfig:
    """
    Create TradingNode configuration.

    Args:
        strategy_name: Strategy name
        testnet: Whether to use testnet
        use_cache: Whether to use Redis cache

    Returns:
        TradingNodeConfig instance
    """
    # Determine account type (default to USDT Futures)
    account_type = BinanceAccountType.USDT_FUTURES

    # Get API credentials from environment
    if testnet:
        api_key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET")
    else:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError(
            f"API credentials not found in environment variables. "
            f"Please set {'BINANCE_FUTURES_TESTNET_API_KEY' if testnet else 'BINANCE_API_KEY'} "
            f"and {'BINANCE_FUTURES_TESTNET_API_SECRET' if testnet else 'BINANCE_API_SECRET'}"
        )

    # Build config
    config_dict = {
        "trader_id": f"MLTrader-{strategy_name}-001",
        "data_engine": LiveDataEngineConfig(),
        "exec_engine": LiveExecEngineConfig(
            reconciliation=True,
            reconciliation_lookback_mins=60,
        ),
        "risk_engine": LiveRiskEngineConfig(),
        "portfolio": PortfolioConfig(),
        "data_clients": {
            BINANCE: BinanceDataClientConfig(
                api_key=api_key,
                api_secret=api_secret,
                account_type=account_type,
                testnet=testnet,
            ),
        },
        "exec_clients": {
            BINANCE: BinanceExecClientConfig(
                api_key=api_key,
                api_secret=api_secret,
                account_type=account_type,
                testnet=testnet,
            ),
        },
    }

    # Add cache config if requested
    if use_cache:
        config_dict["cache"] = CacheConfig(
            database=DatabaseConfig(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                timeout=2.0,
            ),
            encoding="msgpack",
        )

    return TradingNodeConfig(**config_dict)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Nautilus Trader strategy with feature engineering"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy name (e.g., sr_reversal, sr_breakout)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., BTCUSDT-PERP)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15T",
        help="Bar timeframe (e.g., 15T, 1H, 1D)",
    )
    parser.add_argument(
        "--trade-size",
        type=float,
        default=0.001,
        help="Base trade size",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=1000,
        help="Historical data window size for features",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model file (optional)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Binance testnet",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use Redis cache for state persistence",
    )
    parser.add_argument(
        "--config-base-path",
        type=str,
        default="config/strategies",
        help="Base path for strategy configs",
    )

    args = parser.parse_args()

    print(f"🚀 Starting Nautilus Trader strategy: {args.strategy}")
    print(f"   Symbol: {args.symbol}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Testnet: {args.testnet}")

    try:
        # 1. Parse timeframe
        minutes, aggregation = parse_timeframe(args.timeframe)
        bar_spec = BarSpecification(minutes, aggregation, PriceType.LAST)

        # 2. Create instrument ID
        instrument_id = InstrumentId.from_str(f"{args.symbol}.BINANCE")

        # 3. Create bar type
        bar_type = BarType(
            instrument_id=instrument_id,
            bar_spec=bar_spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )

        # 4. Create trading node config
        config = create_trading_node_config(
            strategy_name=args.strategy,
            testnet=args.testnet,
            use_cache=args.use_cache,
        )

        # 5. Create trading node
        node = TradingNode(config=config)

        # 6. Register client factories
        node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
        node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)

        # 7. Build node
        node.build()

        # 8. Create strategy
        strategy = NautilusStrategyWithFeatures(
            strategy_name=args.strategy,
            instrument_id=instrument_id,
            bar_type=bar_type,
            trade_size=args.trade_size,
            config_base_path=args.config_base_path,
            history_window=args.history_window,
            model_path=args.model_path,
        )

        # 9. Add strategy to node
        node.trader.add_strategy(strategy)

        # 10. Run node
        print("✅ Trading node initialized. Starting...")
        print("   Press Ctrl+C to stop")

        try:
            node.run()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            node.stop()
            node.dispose()
            print("✅ Shutdown complete")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
