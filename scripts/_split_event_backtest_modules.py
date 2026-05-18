#!/usr/bin/env python3
"""Split engine.py into focused submodules under scripts/event_backtest/."""
from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "scripts" / "event_backtest"
lines = (PKG / "engine.py").read_text(encoding="utf-8").splitlines(keepends=True)


def sl(a: int, b: int) -> str:
    return "".join(lines[a - 1 : b])


# bootstrap
(PKG / "_bootstrap.py").write_text(
    '''"""Repo path + logger for event_backtest."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("event_backtest")
''',
    encoding="utf-8",
)

# types
(PKG / "types").mkdir(exist_ok=True)
(PKG / "types" / "trade.py").write_text(
    "from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom datetime import datetime\n\n\n"
    + sl(388, 416),
    encoding="utf-8",
)
(PKG / "types" / "stats.py").write_text(
    """from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from scripts.event_backtest.types.trade import ClosedTrade
from src.time_series_model.core.constitution.add_position_rules import (
    resolve_add_position_size_multiplier as shared_resolve_add_position_size_multiplier,
)


def resolve_add_position_size_multiplier(add_rules, add_number, signal=None):
    return shared_resolve_add_position_size_multiplier(add_rules, add_number, signal)


"""
    + sl(426, 442)
    .replace("_tail_contribution_rate", "tail_contribution_rate")
    .replace("_clamp01", "clamp01"),
    encoding="utf-8",
)
(PKG / "types" / "__init__.py").write_text(
    "from scripts.event_backtest.types.trade import ClosedTrade\n",
    encoding="utf-8",
)

# reporting audit (rename _json_safe -> json_safe)
audit_body = (
    sl(2463, 2890)
    .replace("def _json_safe", "def json_safe")
    .replace("def _apply_pcm", "def apply_pcm")
    .replace("def _extract_path", "def extract_path")
    .replace("def _safe_float", "def safe_float")
    .replace("def _trade_audit", "def trade_audit")
    .replace("def _add_attempt", "def add_attempt")
    .replace("def _er_pct", "def er_pct")
    .replace("def _format_er", "def format_er")
    .replace("_json_safe(", "json_safe(")
)
(PKG / "reporting").mkdir(exist_ok=True)
(PKG / "reporting" / "audit.py").write_text(
    """from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.event_backtest.types.trade import ClosedTrade

"""
    + audit_body
    + """

# Legacy private aliases (backtester body unchanged from monolith)
_apply_pcm_direction_ffill = apply_pcm_direction_ffill
_extract_path_efficiency_pct = extract_path_efficiency_pct
_safe_float_or_none = safe_float_or_none
_trade_audit_row_from_fill = trade_audit_row_from_fill
_add_attempt_snapshot = add_attempt_snapshot
_er_pct_numeric_summary = er_pct_numeric_summary
_er_pct_attempt_stats = er_pct_attempt_stats
_format_er_pct_summary_lines = format_er_pct_summary_lines
_json_safe = json_safe
""",
    encoding="utf-8",
)

# features timeline
(PKG / "features").mkdir(exist_ok=True)
(PKG / "features" / "timeline.py").write_text(
    """from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from scripts.event_backtest._bootstrap import logger

"""
    + sl(176, 381)
    + sl(2451, 2460)
    + "\n\n"
    + sl(2892, 2950),
    encoding="utf-8",
)
(PKG / "features" / "__init__.py").write_text(
    "from scripts.event_backtest.features.timeline import *  # noqa: F403\n",
    encoding="utf-8",
)

# results
(PKG / "results.py").write_text(
    """from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.event_backtest.reporting.audit import (
    format_er_pct_summary_lines,
    merge_closed_trades_with_audit_rows,
)
from scripts.event_backtest.types.stats import tail_contribution_rate
from scripts.event_backtest.types.trade import ClosedTrade

"""
    + sl(2093, 2443)
    .replace("_tail_contribution_rate", "tail_contribution_rate")
    .replace("_format_er_pct_summary_lines", "format_er_pct_summary_lines"),
    encoding="utf-8",
)

# simulator position + add helpers
(PKG / "simulator").mkdir(exist_ok=True)
(PKG / "simulator" / "position.py").write_text(
    """from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd

from scripts.account_ledger import AccountLedger
from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.spot.budget import (
    _allocate_spot_accum_leg,
    _spot_entry_fill_price,
    _spot_peer_sims,
    _spot_regime_leg_kwargs,
    _spot_symbol_deploy_legs_today,
    _utc_calendar_day_str,
)
from scripts.event_backtest.types.stats import resolve_add_position_size_multiplier
from scripts.event_backtest.types.trade import ClosedTrade
from src.time_series_model.core.constitution.add_position_rules import (
    add_regime_gate_allows as _shared_add_regime_gate_allows,
    resolve_add_position_max_times as _shared_resolve_add_position_max_times,
    resolve_add_position_min_current_r,
    resolve_float_r_ladder_only as _shared_resolve_float_r_ladder_only,
    validate_add_position_trigger as _shared_validate_add_position_trigger,
)
from src.time_series_model.core.constitution.runtime_state import AddPositionRecord
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.position_logic import build_position_dict, enforce_position
from src.time_series_model.live.spot_accum_simple import (
    apply_partial_sell_to_position,
    is_spot_accum_archetype,
    maybe_spot_simple_partial_sell,
)


def _resolve_add_position_size_multiplier(add_rules, add_number, signal=None):
    return resolve_add_position_size_multiplier(add_rules, add_number, signal)


def _json_safe(value):
    from scripts.event_backtest.reporting.audit import json_safe
    return json_safe(value)


"""
    + sl(449, 1989),
    encoding="utf-8",
)

(PKG / "simulator" / "om_bridge.py").write_text(
    """from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from src.order_management.mock_binance_api import MockBinanceAPI
    from src.order_management.storage import Storage as OMStorage
    from src.order_management.order_manager import OrderManager
    from src.order_management.position_manager import PositionManager
    from src.order_management.models import (
        PositionSide as OMPositionSide,
        OrderSide as OMOrderSide,
        OrderType as OMOrderType,
    )
    OM_AVAILABLE = True
except ImportError:
    OM_AVAILABLE = False

"""
    + sl(1995, 2089),
    encoding="utf-8",
)

(PKG / "simulator" / "__init__.py").write_text(
    """from scripts.event_backtest.simulator.om_bridge import OMBridge
from scripts.event_backtest.simulator.position import PositionSimulator

__all__ = ["OMBridge", "PositionSimulator"]
""",
    encoding="utf-8",
)

# trading map
(PKG / "reporting" / "trading_map.py").write_text(
    """from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.results import BacktestResult
from scripts.event_backtest.types.trade import ClosedTrade
from src.data_tools.data_handler import DataHandler

try:
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import (
        HoverTool,
        Div,
        Tabs,
        TabPanel,
        FixedTicker,
        ColumnDataSource,
        Span,
        Toggle,
        CustomJS,
    )
    from bokeh.layouts import column as bk_column
    from bokeh.resources import INLINE as BK_RESOURCES
    from bokeh.embed import file_html as bk_file_html
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

"""
    + sl(4858, 5819),
    encoding="utf-8",
)

# json export
(PKG / "reporting" / "json_export.py").write_text(
    """from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from scripts.event_backtest.reporting.audit import json_safe
from scripts.event_backtest.results import BacktestResult
from scripts.event_backtest.types.stats import tail_contribution_rate
from scripts.event_backtest.types.trade import ClosedTrade

"""
    + sl(6253, 6379)
    .replace("def _trade_to_dict", "def trade_to_dict")
    .replace("def _save_json", "def save_json")
    .replace("def _save_path_efficiency_sidecar", "def save_path_efficiency_sidecar")
    .replace("_json_safe", "json_safe")
    .replace("_tail_contribution_rate", "tail_contribution_rate")
    .replace("_trade_to_dict", "trade_to_dict"),
    encoding="utf-8",
)

(PKG / "reporting" / "__init__.py").write_text(
    """from scripts.event_backtest.reporting.json_export import save_json, save_path_efficiency_sidecar
from scripts.event_backtest.reporting.trading_map import generate_trading_map_html

__all__ = ["generate_trading_map_html", "save_json", "save_path_efficiency_sidecar"]
""",
    encoding="utf-8",
)

# backtester - massive import block from original engine lines 56-130 + 2957-4852
bt_imports = """from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from scripts.capital_report import write_capital_report_from_trades
from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.features.timeline import (
    _align_feature_index_to_bar_close,
    _feature_asof_from_sym_tf_features,
    _feature_row_asof_from_sym_tf_features,
    _get_bar_minutes,
    _get_timeframe,
    _iter_update_bars_1min,
    _sync_ema_1200_from_feature_row,
    _sync_macro_tp_vwap_from_feature_row,
    _timeframe_from_strategy_meta,
    _timeframe_to_timedelta,
    row_to_features,
)
from scripts.event_backtest.modes import BacktestMode, resolve_backtest_mode
from scripts.event_backtest.reporting.audit import (
    _add_attempt_snapshot,
    _er_pct_attempt_stats,
    _extract_path_efficiency_pct,
    _trade_audit_row_from_fill,
)
from scripts.event_backtest.results import BacktestResult
from scripts.event_backtest.simulator.om_bridge import OMBridge, OM_AVAILABLE
from scripts.event_backtest.simulator.position import PositionSimulator
from scripts.event_backtest.spot.budget import _build_spot_capital_budget_or_none
from scripts.event_backtest.spot.metrics import (
    _compute_deploy_quote_pct_series,
    _compute_spot_accum_accumulation_audit,
    _compute_spot_buy_hold_benchmarks,
    _compute_spot_inventory_metrics,
)
from scripts.event_backtest.types.trade import ClosedTrade
from src.data_tools.data_handler import DataHandler
from src.feature_store import FeatureStore, FeatureStoreSpec
from src.feature_store.layer_naming import detect_layer_for_strategy
from src.live_data_stream.feature_storage import StorageManager
from src.time_series_model.core.constitution.constitution_executor import ConstitutionExecutor
from src.time_series_model.core.constitution.runtime_state import (
    AddPositionRecord,
    ConstitutionRuntimeState,
)
from src.time_series_model.core.constitution.safety_runtime import (
    SafetyRuntimeState,
    evaluate_safety_state,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.event_backtest_srb_hooks import SrbEventBacktestHooks
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.live.incremental_feature_computer import IncrementalFeatureComputer
from src.time_series_model.portfolio.live_pcm import LivePCM
from src.features.cross_symbol.macro_tp_vwap_anchor import (
    ANCHOR_COLUMN,
    parse_macro_tp_vwap_anchor_config,
)

"""
(PKG / "backtester.py").write_text(bt_imports + sl(2957, 4852), encoding="utf-8")

# cli
(PKG / "cli.py").write_text(
    """from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.backtester import EventBacktester
from scripts.event_backtest.reporting.json_export import save_json, save_path_efficiency_sidecar
from scripts.event_backtest.reporting.trading_map import generate_trading_map_html
from scripts.event_backtest.results import BacktestResult
from scripts.capital_report import write_capital_report_from_trades

try:
    from src.order_management.mock_binance_api import MockBinanceAPI
    OM_AVAILABLE = True
except ImportError:
    OM_AVAILABLE = False

"""
    + sl(5825, 6251)
    .replace("def main", "def main")
    .replace("_save_json", "save_json")
    .replace("_save_path_efficiency_sidecar", "save_path_efficiency_sidecar")
    .replace("generate_trading_map_html", "generate_trading_map_html"),
    encoding="utf-8",
)

# engine.py aggregator
(PKG / "engine.py").write_text(
    '''"""Event backtest implementation (split across submodules; import from here)."""

from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.backtester import EventBacktester
from scripts.event_backtest.cli import main
from scripts.event_backtest.results import BacktestResult
from scripts.event_backtest.simulator.om_bridge import OMBridge
from scripts.event_backtest.simulator.position import PositionSimulator
from scripts.event_backtest.reporting.json_export import save_json, save_path_efficiency_sidecar
from scripts.event_backtest.reporting.trading_map import generate_trading_map_html
from scripts.event_backtest.spot.budget import _spot_regime_unit_multiplier
from scripts.event_backtest.spot.metrics import (
    _bucket_spot_accum_funnel_row,
    _compute_deploy_quote_pct_series,
    _compute_spot_accum_accumulation_audit,
    _compute_spot_buy_hold_benchmarks,
    _compute_spot_inventory_metrics,
    _ts_utc,
)
from scripts.event_backtest.types.trade import ClosedTrade

__all__ = [
    "BacktestResult",
    "ClosedTrade",
    "EventBacktester",
    "OMBridge",
    "PositionSimulator",
    "generate_trading_map_html",
    "logger",
    "main",
    "save_json",
]

# Legacy private aliases for tests
_save_json = save_json
_spot_regime_unit_multiplier = _spot_regime_unit_multiplier
_compute_spot_buy_hold_benchmarks = _compute_spot_buy_hold_benchmarks
_compute_spot_inventory_metrics = _compute_spot_inventory_metrics
_compute_spot_accum_accumulation_audit = _compute_spot_accum_accumulation_audit
_compute_deploy_quote_pct_series = _compute_deploy_quote_pct_series
_bucket_spot_accum_funnel_row = _bucket_spot_accum_funnel_row

if __name__ == "__main__":
    import sys
    sys.exit(main())
''',
    encoding="utf-8",
)

# update __init__.py
(PKG / "__init__.py").write_text(
    '''"""Event-driven backtest package."""

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

_spot_regime_unit_multiplier = spot_regime_unit_multiplier

__all__ = [
    "BacktestResult",
    "ClosedTrade",
    "EventBacktester",
    "PositionSimulator",
    "_save_json",
    "generate_trading_map_html",
    "main",
]
''',
    encoding="utf-8",
)

# remove bad types.py if exists at pkg root
(PKG / "types.py").unlink(missing_ok=True)

print("split complete; engine lines", len((PKG / "engine.py").read_text().splitlines()))
