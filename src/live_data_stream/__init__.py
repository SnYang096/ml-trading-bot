"""
Smart money triggered event-driven utilities.

This package contains:
- WebSocket client for Binance trade streams
- 100ms tick aggregation helpers
- Order-flow signal calculation utilities
- Orchestration helpers for realtime + manual execution
- Order flow listener with Nautilus Trader integration
- Feature storage (4h, 15min, 1min ticks)
- Memory window management
- Data gap filling with Feature Store integration
"""

from .config_loader import SmartMoneySettings, load_settings
from .engine import SmartMoneyEngine
from .feature_storage import (
    StorageManager,
    Feature4HStorage,
    Feature15MinStorage,
    Tick1MinStorage,
)
from .memory_window import MemoryWindow
from .gap_filler import GapFiller
from .order_flow_listener import OrderFlowListener
from .listener_config import OrderFlowListenerConfig
from .multi_symbol_manager import MultiSymbolManager

try:
    from .live_test_strategy import LiveTestStrategy
    LIVE_TEST_STRATEGY_AVAILABLE = True
except ImportError:
    LIVE_TEST_STRATEGY_AVAILABLE = False
    LiveTestStrategy = None

__all__ = [
    "SmartMoneySettings",
    "load_settings",
    "SmartMoneyEngine",
    "StorageManager",
    "Feature4HStorage",
    "Feature15MinStorage",
    "Tick1MinStorage",
    "MemoryWindow",
    "GapFiller",
    "OrderFlowListener",
    "OrderFlowListenerConfig",
    "MultiSymbolManager",
]

if LIVE_TEST_STRATEGY_AVAILABLE:
    __all__.append("LiveTestStrategy")
