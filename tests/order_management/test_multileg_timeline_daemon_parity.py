"""Timeline backtest vs live daemon parity for chop↔trend same-bar handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.order_management.chop_grid_concurrency import MultiLegConcurrencyGate
from src.order_management.grid_execution_adapter import GridExecutionResult
from src.order_management.multi_leg_daemon import (
    MultiLegBarEvent,
    MultiLegLiveDaemon,
    StrategyRuntime,
)
from src.order_management.multi_leg_orchestrator import MultiLegLiveOrchestrator
from src.order_management.multi_leg_reconciliation import MultiLegReconciler
from src.order_management.multi_leg_risk_governor import (
    MultiLegPortfolioRiskGovernor,
    MultiLegRiskLimits,
)
from src.time_series_model.live.chop_grid_live_engine import (
    ChopGridLiveEngine,
    GridPosition,
)
from src.time_series_model.live.dual_add_trend_live_engine import DualAddTrendLiveEngine
from src.time_series_model.live.segment_lifecycle import SegmentState


def _chop_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "chop.yaml"
    path.write_text(
        """
regime:
  entry_chop_min: 0.50
  exit_chop_below: 0.32
  exclude_box_prefilter: true
inventory:
  spacing:
    atr_mult: 0.50
    min_pct: 0.004
  max_levels_per_side: 1
risk:
  fee_bps: 4.0
  max_open_levels_total: 2
""",
        encoding="utf-8",
    )
    return path


def _trend_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "trend.yaml"
    path.write_text(
        """
regime:
  entry_min: 0.70
  exit_below: 0.40
  max_semantic_chop_entry: 0.25
  max_semantic_chop_hold: 0.40
  exclude_box_prefilter: true
inventory:
  initial_hedge: false
  flip_action: close_offside_all
  max_adds_per_side: 1
  max_gross_exposure_units: 2
  max_net_exposure_units: 1
add_spacing:
  atr_mult: 0.50
take_profit:
  atr_mult: 0.25
  min_pct: 0.0005
  min_abs: 0.0
risk:
  diagnostic_fee_bps: 4.0
  max_loss_per_segment: 0.01
order_model:
  entry_order_type: marketable_limit
  add_order_type: marketable_limit
  max_slippage_bps: 5.0
  pending_timeout_bars: 1
""",
        encoding="utf-8",
    )
    return path


def _handoff_features() -> Dict[str, Any]:
    return {
        "bpc_semantic_chop": 0.12,
        "semantic_chop": 0.12,
        "box_pos_60": 0.50,
        "box_prefilter": False,
        "trend_confidence": 1.0,
        "trend_direction": "UP",
    }


def _seed_chop_active(engine: ChopGridLiveEngine, *, symbol: str = "HYPEUSDT") -> None:
    engine.state.active = True
    engine.state.symbol = symbol
    engine.state.segment_state = SegmentState.ACTIVE.value
    engine.state.grid_id = f"{symbol}_2026-05-30T18:00:00Z"
    engine.state.center = 67.80
    engine.state.spacing = 0.22
    engine.state.inventory = [
        GridPosition(
            symbol=symbol,
            side="SHORT",
            level=1,
            entry_price=68.02,
            quantity=1.0,
            entry_quantity=1.0,
            entry_time="2026-05-30T20:00:00Z",
            leg_id=f"{symbol}_2026-05-30T18:00:00Z_S1",
        )
    ]
    engine.state.pending_orders = []


def _make_engine_pair(
    tmp_path: Path,
    *,
    symbol: str = "HYPEUSDT",
    tag: str = "default",
) -> tuple[ChopGridLiveEngine, DualAddTrendLiveEngine, MultiLegConcurrencyGate]:
    gate = MultiLegConcurrencyGate(max_symbols=6, cooldown_bars=0)
    chop = ChopGridLiveEngine(
        config_path=_chop_cfg(tmp_path),
        state_path=tmp_path / f"chop_state_{tag}.json",
        level_notional=100.0,
        bar_simulation=True,
    )
    trend = DualAddTrendLiveEngine(
        config_path=_trend_cfg(tmp_path),
        state_path=tmp_path / f"trend_state_{tag}.json",
        unit_notional=100.0,
    )
    chop.state.symbol = trend.state.symbol = symbol
    gate.register(symbol, chop, strategy="chop_grid")
    gate.register(symbol, trend, strategy="trend_scalp")
    _seed_chop_active(chop, symbol=symbol)
    return chop, trend, gate


def _bar_kwargs(symbol: str = "HYPEUSDT") -> Dict[str, Any]:
    return dict(
        symbol=symbol,
        timestamp="2026-05-31T20:00:00Z",
        high=71.50,
        low=71.20,
        close=71.46,
        atr=0.5,
        features=_handoff_features(),
    )


def _action_kinds(actions: List[Dict[str, Any]]) -> List[str]:
    return [str(a.get("action", "")).lower() for a in actions]


def _timeline_actions(
    chop: ChopGridLiveEngine, trend: DualAddTrendLiveEngine, **bar
) -> List[Dict[str, Any]]:
    return list(chop.on_bar(**bar) or []) + list(trend.on_bar(**bar) or [])


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.sync_open_orders.return_value = []
    adapter.sync_positions.return_value = []
    adapter.execute_actions.side_effect = lambda actions: [
        GridExecutionResult(
            action=a.get("action", ""),
            status="shadow",
            symbol=a.get("symbol", "HYPEUSDT"),
            raw=a,
        )
        for a in actions
    ]
    return adapter


def _runtime(
    name: str, symbol: str, engine: Any, adapter: MagicMock
) -> StrategyRuntime:
    orchestrator = MultiLegLiveOrchestrator(
        engine=engine,
        governor=MultiLegPortfolioRiskGovernor(
            MultiLegRiskLimits(
                max_gross_notional=1_000_000.0, max_net_notional=1_000_000.0
            )
        ),
        adapter=adapter,
        reconciler=MultiLegReconciler(),
    )
    return StrategyRuntime(
        name=name, symbol=symbol, engine=engine, orchestrator=orchestrator
    )


def test_timeline_and_daemon_agree_on_same_bar_chop_exit_trend_enter(
    tmp_path: Path,
) -> None:
    """E2E: real engines — timeline merge matches daemon routing on handoff bar."""
    chop_tl, trend_tl, _ = _make_engine_pair(tmp_path, symbol="HYPEUSDT", tag="tl")
    bar = _bar_kwargs()
    timeline = _timeline_actions(chop_tl, trend_tl, **bar)

    chop_dm, trend_dm, _ = _make_engine_pair(tmp_path, symbol="HYPEUSDT", tag="dm")
    adapter_a = _adapter()
    adapter_b = _adapter()
    daemon = MultiLegLiveDaemon(
        bar_provider=MagicMock(
            latest_closed_bars=lambda _syms: [
                MultiLegBarEvent(
                    symbol=bar["symbol"],
                    timestamp=bar["timestamp"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    atr=bar["atr"],
                    features=bar["features"],
                )
            ]
        ),
        runtimes=[
            _runtime("chop_grid", "HYPEUSDT", chop_dm, adapter_a),
            _runtime("trend_scalp", "HYPEUSDT", trend_dm, adapter_b),
        ],
    )
    report = daemon.run_once()

    assert "market_exit" in _action_kinds(timeline)
    assert "place" in _action_kinds(timeline)
    assert chop_tl.state.segment_state == SegmentState.IDLE.value
    assert trend_tl.state.segment_state == SegmentState.ENTERING.value

    assert report.rejected_count == 0
    assert adapter_a.execute_actions.called
    assert adapter_b.execute_actions.called
    daemon_exits: List[str] = []
    for adapter in (adapter_a, adapter_b):
        for call in adapter.execute_actions.call_args_list:
            daemon_exits.extend(_action_kinds(call.args[0]))
    assert "market_exit" in daemon_exits
    assert "place" in daemon_exits
    assert chop_dm.state.segment_state == SegmentState.IDLE.value
    assert trend_dm.state.segment_state == SegmentState.ENTERING.value


@pytest.mark.skipif(
    not Path("data/parquet_data/HYPEUSDT_2026-05.parquet").is_file(),
    reason="HYPE 2026-05 parquet required for replay backtest",
)
def test_hype_replay_handoff_bars_match_timeline_and_daemon(tmp_path: Path) -> None:
    """Replay May28–Jun1 HYPE with mock execution so timeline and daemon stay aligned."""
    from scripts.backtest_multileg_timeline import (
        _build_features,
        _load_1m_bars,
        _lookup,
    )

    import pandas as pd

    sym = "HYPEUSDT"
    start = pd.Timestamp("2026-05-28", tz="UTC")
    end = pd.Timestamp("2026-06-01", tz="UTC")
    data_dir = Path("data/parquet_data")
    bars_1m = _load_1m_bars(data_dir, [sym], start, end)
    feats = _build_features(data_dir, [sym])

    def _new_pair(tag: str) -> tuple[ChopGridLiveEngine, DualAddTrendLiveEngine]:
        gate = MultiLegConcurrencyGate(6, cooldown_bars=0)
        chop = ChopGridLiveEngine(
            config_path=_chop_cfg(tmp_path),
            state_path=tmp_path / f"chop_{tag}.json",
            level_notional=100.0,
            bar_simulation=True,
        )
        trend = DualAddTrendLiveEngine(
            config_path=_trend_cfg(tmp_path),
            state_path=tmp_path / f"trend_{tag}.json",
            unit_notional=100.0,
        )
        chop.state.symbol = trend.state.symbol = sym
        gate.register(sym, chop, strategy="chop_grid")
        gate.register(sym, trend, strategy="trend_scalp")
        return chop, trend

    def _execute_pair(
        chop: ChopGridLiveEngine,
        trend: DualAddTrendLiveEngine,
        actions: List[Dict[str, Any]],
    ) -> None:
        adapter = _adapter()
        results = adapter.execute_actions(actions)
        for eng in (chop, trend):
            if hasattr(eng, "on_execution_results"):
                eng.on_execution_results(results)

    chop_tl, trend_tl = _new_pair("tl")
    chop_dm, trend_dm = _new_pair("dm")
    adapter_a = _adapter()
    adapter_b = _adapter()
    daemon = MultiLegLiveDaemon(
        bar_provider=MagicMock(latest_closed_bars=lambda _s: []),
        runtimes=[
            _runtime("chop_grid", sym, chop_dm, adapter_a),
            _runtime("trend_scalp", sym, trend_dm, adapter_b),
        ],
    )

    last_2h = None
    handoffs_tl = handoffs_dm = mismatches = 0
    for idx, row in bars_1m.iterrows():
        bar_2h = idx.floor("2h")
        if last_2h is not None and bar_2h <= last_2h:
            continue
        last_2h = bar_2h
        f = _lookup(feats, sym, idx)
        if not f:
            continue
        bar = dict(
            symbol=sym,
            timestamp=str(idx),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            atr=float(row.get("atr14", row["close"] * 0.02)),
            features=f,
        )

        tl = _timeline_actions(chop_tl, trend_tl, **bar)
        _execute_pair(chop_tl, trend_tl, tl)

        bar_ev = MultiLegBarEvent(
            symbol=sym,
            timestamp=bar["timestamp"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            atr=bar["atr"],
            features=bar["features"],
        )
        daemon.bar_provider.latest_closed_bars = lambda _s, b=bar_ev: [b]
        daemon._last_processed.clear()
        report = daemon.run_once()
        dm_calls: List[Dict[str, Any]] = []
        if adapter_a.execute_actions.called:
            dm_calls.extend(adapter_a.execute_actions.call_args.args[0])
        if adapter_b.execute_actions.called:
            dm_calls.extend(adapter_b.execute_actions.call_args.args[0])
        _execute_pair(chop_dm, trend_dm, dm_calls)
        adapter_a.execute_actions.reset_mock()
        adapter_b.execute_actions.reset_mock()

        def _parity_sig(
            chop_st: str, trend_st: str, kinds: List[str]
        ) -> tuple[str, str, bool, bool]:
            return (
                chop_st,
                trend_st,
                "market_exit" in kinds,
                "place" in kinds,
            )

        tl_sig = _parity_sig(
            chop_tl.state.segment_state,
            trend_tl.state.segment_state,
            _action_kinds(tl),
        )
        dm_sig = _parity_sig(
            chop_dm.state.segment_state,
            trend_dm.state.segment_state,
            _action_kinds(dm_calls),
        )
        if tl_sig != dm_sig:
            mismatches += 1
        tl_handoff = tl_sig[2] and tl_sig[3]
        dm_handoff = dm_sig[2] and dm_sig[3]
        if tl_handoff:
            handoffs_tl += 1
        if dm_handoff:
            handoffs_dm += 1
        if tl_handoff != dm_handoff:
            mismatches += 1
        if tl_handoff and not dm_handoff:
            assert report.rejected_count == 0, f"unexpected reject at {idx}"

    assert mismatches == 0, (
        f"timeline/daemon diverged on {mismatches} bar(s); "
        f"handoffs tl={handoffs_tl} dm={handoffs_dm}"
    )
