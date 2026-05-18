"""Event-driven multi-strategy backtest (PCM + 1m bar simulation).

Subpackages:
  - ``spot/``: spot_accum budget, deploy decay, inventory KPI, BH benchmarks
  - ``modes/``: classify runs as SPOT vs TREND (multi-leg uses other scripts)

The monolithic implementation lives in ``engine.py``; prefer importing from this
package root for stable symbols.
"""

from scripts.event_backtest.engine import (
    BacktestResult,
    ClosedTrade,
    EventBacktester,
    PositionSimulator,
    _save_json,
    generate_trading_map_html,
    main,
)
from scripts.event_backtest.spot.budget import spot_regime_unit_multiplier
from scripts.event_backtest.spot.metrics import (
    compute_spot_buy_hold_benchmarks,
    compute_spot_inventory_metrics,
)

# Legacy private aliases used by unit tests
_spot_regime_unit_multiplier = spot_regime_unit_multiplier
_compute_spot_buy_hold_benchmarks = compute_spot_buy_hold_benchmarks
_compute_spot_inventory_metrics = compute_spot_inventory_metrics
_bucket_spot_accum_funnel_row = __import__(
    "scripts.event_backtest.spot.metrics", fromlist=["bucket_spot_accum_funnel_row"]
).bucket_spot_accum_funnel_row
_compute_spot_accum_accumulation_audit = __import__(
    "scripts.event_backtest.spot.metrics",
    fromlist=["compute_spot_accum_accumulation_audit"],
).compute_spot_accum_accumulation_audit
_compute_deploy_quote_pct_series = __import__(
    "scripts.event_backtest.spot.metrics",
    fromlist=["compute_deploy_quote_pct_series"],
).compute_deploy_quote_pct_series

__all__ = [
    "BacktestResult",
    "ClosedTrade",
    "EventBacktester",
    "PositionSimulator",
    "_save_json",
    "generate_trading_map_html",
    "main",
    "spot_regime_unit_multiplier",
]
