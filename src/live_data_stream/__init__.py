"""
Smart money triggered event-driven utilities.

This package contains:
- WebSocket client for Binance trade streams
- 100ms tick aggregation helpers
- Order-flow signal calculation utilities
- Orchestration helpers for realtime + manual execution
"""

from .config_loader import SmartMoneySettings, load_settings
from .engine import SmartMoneyEngine

__all__ = ["SmartMoneySettings", "load_settings", "SmartMoneyEngine"]

