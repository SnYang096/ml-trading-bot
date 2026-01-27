from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

try:
    from nautilus_trader.model import (
        Bar,
        InstrumentId,
        BarType,
        TradeTick,
        QuoteTick,
    )
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.trading import Strategy
    from nautilus_trader.trading.config import StrategyConfig

    NAUTILUS_AVAILABLE = True
except Exception:  # pragma: no cover
    NAUTILUS_AVAILABLE = False
    Strategy = object  # type: ignore
    Bar = None  # type: ignore
    InstrumentId = None  # type: ignore
    BarType = None  # type: ignore
    OrderSide = None  # type: ignore
    TradeTick = None  # type: ignore
    QuoteTick = None  # type: ignore

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
    load_evidence_quantiles,
)
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.live.execution_manager import (
    ExecutionManager,
    GuardedOrderContext,
)
from src.time_series_model.live.meta_router_config import (
    MetaRouterLiveConfig,
    load_meta_router_live_config,
    select_first_enabled_archetype,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
    resolve_execution_profile_paths,
)
from src.time_series_model.live.live_runtime_paths import resolve_live_runtime_paths
from src.time_series_model.diagnostics.execution_log import (
    build_decision_id,
    build_stage_record,
    ExecutionStageLogWriter,
)
from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    compute_mode_3action,
)
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.nnmh_live_inferencer import (
    NNMHLiveInferencer,
    NNMHLiveInferencerConfig,
)
from src.time_series_model.live.timers import compute_next_aligned_delay_seconds
from src.time_series_model.live.live_feature_contract import (
    load_live_feature_contract_v1,
    validate_live_features_v1,
)
from src.time_series_model.live.archetype_heuristics import (
    evaluate_required_conditions,
)
from src.time_series_model.live.direction_resolver import resolve_direction
from src.time_series_model.live.execution_rules import (
    apply_execution_rules,
    load_execution_rules,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
from src.time_series_model.live.execution_intelligence import build_execution_profile
from src.time_series_model.ops.state_snapshot import (
    SystemStateSnapshot,
    write_state_snapshot,
)
from src.time_series_model.live.observability_metrics import (
    compute_evidence_true_rate,
    compute_feature_missing_rate,
    compute_router_mode_entropy,
    compute_tick_gap_seconds,
)
from src.time_series_model.portfolio.pcm import (
    SymbolDecision as PCMSymbolDecision,
    compute_pcm_budget_for_decisions,
)
from src.time_series_model.execution.et_hedge import (
    compute_et_position,
    ETPositionPair,
    compute_et_risk_score,
)


def _infer_regime_placeholder() -> str:
    """
    Placeholder router hook:
    - live should later plug in real Router (nnmultihead outputs + rules/gates).
    """
    return str(os.getenv("MLBOT_LIVE_FORCE_REGIME", "NO_TRADE")).upper()


if NAUTILUS_AVAILABLE:

    class MetaRouterStrategyConfig(StrategyConfig):
        strategy_name: str
        instrument_id: InstrumentId
        bar_type: BarType
        trade_size: float
        data_client_id: str = "BINANCE"
        live_config_path: str = "config/nnmultihead/live/meta_router_live_config.yaml"
        constitution_yaml: Optional[str] = None
        archetype_registry_path: str = "config/nnmultihead/execution_archetypes.yaml"

    class MetaRouterStrategy(Strategy):
        """
        One Strategy process, multiple archetypes (per docs/live_stream/策略一起还是分开.md).

        NOTE: Router decision is currently a placeholder (env override). This class is primarily
        to enforce correct *architecture*: one account, one world-view, one order stream.
        """

        def __init__(self, config: MetaRouterStrategyConfig):
            super().__init__()
            self.strategy_name = str(config.strategy_name)
            self.instrument_id = config.instrument_id
            self.bar_type = config.bar_type
            self.trade_size = float(config.trade_size)

            self.live_config_path = str(config.live_config_path)
            from nautilus_trader.model.identifiers import ClientId

            self._data_client_id = ClientId(str(config.data_client_id))
            live_paths = resolve_live_runtime_paths()
            self.constitution_yaml = (
                config.constitution_yaml or live_paths["constitution_yaml"]
            )
            _, resolved_registry = resolve_execution_profile_paths(
                default_archetype_registry_path=str(config.archetype_registry_path)
            )
            self.archetype_registry_path = str(resolved_registry)

            self._cfg: Optional[MetaRouterLiveConfig] = None
            self._exec = None
            self._st = None
            self._arches = None
            self._feature_computer: Optional[IncrementalFeatureComputer] = None
            self._inferencer: Optional[NNMHLiveInferencer] = None
            self._last_order_time_ns: Optional[int] = None
            self._live_feature_contract_path = str(
                live_paths["live_feature_contract_yaml"]
            )
            self._live_feature_contract = None
            self._execution_rules_yaml = str(live_paths["execution_rules_yaml"])
            self._execution_rules = None
            self._mode_hist = deque(maxlen=200)
            self._exec_stage_writers: dict[str, ExecutionStageLogWriter] = {}
            # ET pairing tracking
            from time_series_model.execution.et_hedge import ETPositionPair

            self._et_position_pairs: Dict[str, ETPositionPair] = (
                {}
            )  # et_position_id -> pair
            self._active_tc_te_positions: Dict[str, str] = (
                {}
            )  # position_id -> archetype (TC/TE)

        def on_start(self) -> None:
            self._cfg = load_meta_router_live_config(self.live_config_path)
            self._arches = load_execution_archetypes_registry(
                self.archetype_registry_path
            )
            self._exec = ConstitutionExecutor(constitution_yaml=self.constitution_yaml)
            self._st = self._exec.load_runtime_state()
            self._xm = ExecutionManager(
                strategy=self, executor=self._exec, runtime_state=self._st
            )
            self._feature_computer = IncrementalFeatureComputer(bar_window_size=1000)
            log_dir = Path(os.getenv("MLBOT_EXECUTION_LOG_DIR", "results/live_logs"))
            for stage in [
                "features",
                "preds",
                "router",
                "gate",
                "evidence",
                "execution",
                "returns",
                "observability",
            ]:
                self._exec_stage_writers[stage] = ExecutionStageLogWriter(
                    base_dir=log_dir, stage=stage
                )
            # Optional: online nnmultihead inference
            nni = (self._cfg.nnmultihead_inference or {}) if self._cfg else {}
            if bool(nni.get("enabled", False)) and nni.get("model_path"):
                self._inferencer = NNMHLiveInferencer(
                    NNMHLiveInferencerConfig(
                        model_path=str(nni.get("model_path")),
                        config_dir=(
                            str(nni.get("config_dir"))
                            if nni.get("config_dir")
                            else None
                        ),
                        device=str(nni.get("device")) if nni.get("device") else None,
                    )
                )

            # Live feature contract (runtime input credibility gate)
            try:
                if (
                    self._live_feature_contract_path
                    and Path(self._live_feature_contract_path).exists()
                ):
                    self._live_feature_contract = load_live_feature_contract_v1(
                        self._live_feature_contract_path
                    )
                    self.log.info(
                        f"✅ Live feature contract loaded: {self._live_feature_contract_path}"
                    )
            except Exception as e:
                self.log.error(f"⚠️ Live feature contract init failed: {e}")

            # Optional: exported execution rules (tree-distilled / YAML-first), fail-closed.
            try:
                if (
                    self._execution_rules_yaml
                    and Path(self._execution_rules_yaml).exists()
                ):
                    self._execution_rules = load_execution_rules(
                        self._execution_rules_yaml
                    )
                    self.log.info(
                        f"✅ Execution rules loaded: {self._execution_rules_yaml}"
                    )
            except Exception as e:
                self.log.error(f"⚠️ Execution rules init failed: {e}")
            self.subscribe_bars(self.bar_type, client_id=self._data_client_id)
            # Orderflow must update on trade ticks (not bar)
            self.subscribe_trade_ticks(
                self.instrument_id, client_id=self._data_client_id
            )

            # Start timer loop (decision/inference should NOT run only on bar)
            self._schedule_next_check()

        def on_trade_tick(self, tick: TradeTick) -> None:
            # Orderflow features MUST update on trade ticks
            if self._feature_computer is None:
                return
            try:
                self._feature_computer.on_tick(tick)
            except Exception:
                return

        def on_tick(self, tick: QuoteTick) -> None:
            # Quote ticks not used for orderflow; ignore for now
            return

        def on_bar(self, bar: Bar) -> None:
            # Only update bar-based features; decision happens in timer callback
            if self._cfg is None or self._feature_computer is None:
                return
            tf = str((self._cfg.nnmultihead_inference or {}).get("timeframe") or "15T")
            try:
                self._feature_computer.on_bar(bar, timeframe=tf)
            except Exception:
                return

        def on_position_closed(self, position) -> None:
            """Handle position closed event - cleanup tracking"""
            try:
                position_id = str(position.id) if hasattr(position, "id") else ""
                if position_id in self._active_tc_te_positions:
                    archetype = self._active_tc_te_positions.pop(position_id)
                    # If TC/TE position closed, check if we need to update ET hedge
                    # This will be handled in the next signal check
                    self.log.debug(
                        f"Position closed: {position_id} (archetype: {archetype})"
                    )
            except Exception as e:
                self.log.warning(f"Error handling position closed: {e}")

        def _get_position_by_archetype(self, archetype: str) -> float:
            """
            Get current position size for a given archetype (TC, TE, etc.)

            This method attempts to get positions from the strategy's cache
            and filter by archetype. If positions are not available or archetype
            cannot be determined, returns 0.0.

            Args:
                archetype: Archetype name (TC, TE, etc.)

            Returns:
                Position size (positive for long, negative for short, 0.0 if not found)
            """
            try:
                if not hasattr(self, "cache") or self.cache is None:
                    return 0.0

                # Try to get position from cache
                # In Nautilus Trader, positions are accessed via cache.positions()
                positions = self.cache.positions(instrument_id=self.instrument_id)
                if not positions:
                    return 0.0

                # Sum positions that match the archetype
                # Note: We need to identify positions by archetype. This could be done via:
                # 1. Position tags (if we store archetype in tags)
                # 2. Position metadata
                # 3. Tracking positions ourselves
                # For now, we'll use a simple heuristic: check if position has matching tags
                total_position = 0.0
                archetype_upper = str(archetype).upper()

                for position in positions:
                    # Check position tags for archetype match
                    tags = getattr(position, "tags", []) or []
                    position_strategy = None
                    for tag in tags:
                        if isinstance(tag, str) and tag.upper() == archetype_upper:
                            position_strategy = tag.upper()
                            break
                        # Also check if tag contains archetype
                        if isinstance(tag, str) and archetype_upper in tag.upper():
                            position_strategy = archetype_upper
                            break

                    # If we found a matching archetype, add to total
                    if position_strategy == archetype_upper:
                        # Get position size (signed: positive for long, negative for short)
                        position_qty = float(position.quantity.as_double())
                        if position.is_short():
                            position_qty = -position_qty
                        total_position += position_qty

                return total_position
            except Exception:
                # If any error occurs, return 0 (fail-safe)
                return 0.0

        def _get_all_active_positions_by_archetype(
            self, archetype: str
        ) -> List[Dict[str, Any]]:
            """
            Get all active positions for a given archetype (TC, TE, etc.)

            Returns list of dicts with 'position_id' and 'size' keys.
            """
            positions = []
            try:
                if not hasattr(self, "cache") or self.cache is None:
                    return positions

                cache_positions = self.cache.positions(instrument_id=self.instrument_id)
                if not cache_positions:
                    return positions

                archetype_upper = str(archetype).upper()

                for position in cache_positions:
                    # Check position tags for archetype match
                    tags = getattr(position, "tags", []) or []
                    position_strategy = None
                    for tag in tags:
                        if isinstance(tag, str) and tag.upper() == archetype_upper:
                            position_strategy = tag.upper()
                            break
                        if isinstance(tag, str) and archetype_upper in tag.upper():
                            position_strategy = archetype_upper
                            break

                    if position_strategy == archetype_upper:
                        position_qty = float(position.quantity.as_double())
                        if position.is_short():
                            position_qty = -position_qty
                        positions.append(
                            {
                                "position_id": str(position.id),
                                "size": position_qty,
                            }
                        )
            except Exception:
                pass
            return positions

        def _get_active_et_position(self) -> Optional[ETPositionPair]:
            """Get the currently active ET hedge position pair"""
            # Return the most recent active ET position
            if not self._et_position_pairs:
                return None
            # Get pairs that are still in active tracking
            active_pairs = [
                pair
                for pair in self._et_position_pairs.values()
                if pair.et_position_id in self._active_tc_te_positions
            ]
            if not active_pairs:
                # If no active ET in tracking, return the most recent one
                return max(
                    self._et_position_pairs.values(), key=lambda p: p.created_at_ns
                )
            return max(active_pairs, key=lambda p: p.created_at_ns)

        def _check_and_update_et_hedge(
            self,
            feats: Dict[str, Any],
            now_ns: int,
            regime: str,
            evidence: Optional[Dict[str, bool]],
        ) -> None:
            """Check and update ET hedge (independent of TC/TE order submission)"""
            # 1. Get all active TC/TE positions
            tc_positions = self._get_all_active_positions_by_archetype("TC")
            te_positions = self._get_all_active_positions_by_archetype("TE")

            # 2. Calculate total directional exposure
            tc_total = sum(p["size"] for p in tc_positions)
            te_total = sum(p["size"] for p in te_positions)

            # 3. If no TC/TE positions, close all ET hedges
            if len(tc_positions) == 0 and len(te_positions) == 0:
                self._close_all_et_hedges(now_ns)
                return

            # 4. Get reflexivity features for ET calculation
            ofci_p = float(feats.get("ofci_pct", 0.0))
            shd_p = float(feats.get("shd_pct", 0.0))
            vol_spike_p = float(feats.get("atr_percentile", 0.0))

            # 5. Calculate ET position
            et_position = compute_et_position(
                tc_position=tc_total,
                te_position=te_total,
                ofci_p=ofci_p,
                shd_p=shd_p,
                vol_spike_p=vol_spike_p,
                k_max=0.8,
            )

            # 6. Check if we need to create/update/close ET hedge
            existing_et = self._get_active_et_position()
            if abs(et_position) > 1e-6:
                if existing_et is None:
                    # Create new ET hedge
                    self._create_et_hedge(
                        et_position=et_position,
                        tc_positions=tc_positions,
                        te_positions=te_positions,
                        feats=feats,
                        now_ns=now_ns,
                        regime=regime,
                        evidence=evidence,
                    )
                else:
                    # Update existing ET hedge if size changed significantly
                    self._update_et_hedge_if_needed(
                        existing_et=existing_et,
                        new_et_position=et_position,
                        tc_positions=tc_positions,
                        te_positions=te_positions,
                        feats=feats,
                        now_ns=now_ns,
                        regime=regime,
                        evidence=evidence,
                    )
            else:
                # Close ET hedge
                if existing_et is not None:
                    self._close_et_hedge(existing_et.et_position_id, now_ns)

        def _create_et_hedge(
            self,
            et_position: float,
            tc_positions: List[Dict[str, Any]],
            te_positions: List[Dict[str, Any]],
            feats: Dict[str, Any],
            now_ns: int,
            regime: str,
            evidence: Optional[Dict[str, bool]],
        ) -> None:
            """Create new ET hedge order and establish pairing relationship"""
            try:
                # Calculate metrics
                tc_total = sum(p["size"] for p in tc_positions)
                te_total = sum(p["size"] for p in te_positions)
                directional_exposure = abs(tc_total) + abs(te_total)

                ofci_p = float(feats.get("ofci_pct", 0.0))
                shd_p = float(feats.get("shd_pct", 0.0))
                vol_spike_p = float(feats.get("atr_percentile", 0.0))
                risk_score = compute_et_risk_score(ofci_p, shd_p, vol_spike_p)

                # Create ET order
                et_qty = self.instrument.make_qty(abs(et_position))
                et_side = OrderSide.SELL if et_position < 0 else OrderSide.BUY
                et_order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=et_side,
                    quantity=et_qty,
                )

                et_position_id = f"{self.strategy_name}:ET:{int(now_ns)}"

                # Submit order
                self._xm.submit_order_guarded(
                    order=et_order,
                    ctx=GuardedOrderContext(
                        position_id=et_position_id,
                        symbol=str(self.instrument_id),
                        archetype=str(regime),
                        execution_strategy="ET",
                        execution_tags=[str(self.strategy_name), "ET_HEDGE"],
                        execution_evidence=evidence,
                    ),
                )

                # Create pairing record
                pair = ETPositionPair(
                    et_position_id=et_position_id,
                    tc_position_ids=[p["position_id"] for p in tc_positions],
                    te_position_ids=[p["position_id"] for p in te_positions],
                    created_at_ns=now_ns,
                    directional_exposure=directional_exposure,
                    et_position_size=et_position,
                    risk_score=risk_score,
                    ofci_p=ofci_p,
                    shd_p=shd_p,
                    vol_spike_p=vol_spike_p,
                )
                self._et_position_pairs[et_position_id] = pair
                self._active_tc_te_positions[et_position_id] = "ET"

                self.log.info(
                    f"ET hedge created: position_id={et_position_id}, "
                    f"et_size={et_position:.4f}, tc_pos={tc_total:.4f}, te_pos={te_total:.4f}, "
                    f"risk_score={risk_score:.3f}"
                )
            except Exception as e:
                self.log.error(f"Failed to create ET hedge: {e}")

        def _update_et_hedge_if_needed(
            self,
            existing_et: ETPositionPair,
            new_et_position: float,
            tc_positions: List[Dict[str, Any]],
            te_positions: List[Dict[str, Any]],
            feats: Dict[str, Any],
            now_ns: int,
            regime: str,
            evidence: Optional[Dict[str, bool]],
        ) -> None:
            """Update existing ET hedge if size changed significantly (threshold: 5%)"""
            size_diff = abs(new_et_position - existing_et.et_position_size)
            size_threshold = abs(existing_et.et_position_size) * 0.05  # 5% threshold

            if size_diff > size_threshold:
                # Close old hedge and create new one
                self._close_et_hedge(existing_et.et_position_id, now_ns)
                self._create_et_hedge(
                    et_position=new_et_position,
                    tc_positions=tc_positions,
                    te_positions=te_positions,
                    feats=feats,
                    now_ns=now_ns,
                    regime=regime,
                    evidence=evidence,
                )

        def _close_et_hedge(self, et_position_id: str, now_ns: int) -> None:
            """Close specified ET hedge"""
            try:
                if et_position_id not in self._et_position_pairs:
                    return

                # Close position (submit opposite order to close)
                pair = self._et_position_pairs[et_position_id]
                close_qty = self.instrument.make_qty(abs(pair.et_position_size))
                close_side = (
                    OrderSide.BUY if pair.et_position_size < 0 else OrderSide.SELL
                )
                close_order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=close_side,
                    quantity=close_qty,
                )

                self._xm.submit_order_guarded(
                    order=close_order,
                    ctx=GuardedOrderContext(
                        position_id=f"{et_position_id}:CLOSE:{int(now_ns)}",
                        symbol=str(self.instrument_id),
                        archetype="MEAN",  # ET is mean-reversion
                        execution_strategy="ET",
                        execution_tags=[str(self.strategy_name), "ET_HEDGE_CLOSE"],
                        execution_evidence=None,
                    ),
                )

                # Remove from tracking
                self._et_position_pairs.pop(et_position_id, None)
                self._active_tc_te_positions.pop(et_position_id, None)

                self.log.info(f"ET hedge closed: position_id={et_position_id}")
            except Exception as e:
                self.log.error(f"Failed to close ET hedge {et_position_id}: {e}")

        def _close_all_et_hedges(self, now_ns: int) -> None:
            """Close all active ET hedges"""
            et_position_ids = list(self._et_position_pairs.keys())
            for et_position_id in et_position_ids:
                self._close_et_hedge(et_position_id, now_ns)

        def _cleanup_closed_positions(self) -> None:
            """Clean up tracking for positions that are no longer active"""
            try:
                if not hasattr(self, "cache") or self.cache is None:
                    return

                active_position_ids = set()
                cache_positions = self.cache.positions(instrument_id=self.instrument_id)
                if cache_positions:
                    for position in cache_positions:
                        active_position_ids.add(str(position.id))

                # Remove closed positions from tracking
                closed_positions = []
                for position_id in list(self._active_tc_te_positions.keys()):
                    if position_id not in active_position_ids:
                        closed_positions.append(position_id)

                for position_id in closed_positions:
                    archetype = self._active_tc_te_positions.pop(position_id, None)
                    # If it was an ET position, also remove from pairs
                    if archetype == "ET" and position_id in self._et_position_pairs:
                        self._et_position_pairs.pop(position_id)
            except Exception:
                pass

        def _schedule_next_check(self) -> None:
            if self._cfg is None:
                return
            dl = self._cfg.decision_loop or {}
            if not bool(dl.get("enabled", True)):
                return
            interval_min = int(dl.get("check_interval_minutes", 10))
            delay_sec = compute_next_aligned_delay_seconds(
                now_ns=int(self.clock.timestamp_ns()), interval_minutes=interval_min
            )
            try:
                self.clock.cancel_timer("meta_router_check")
            except Exception:
                pass
            self.clock.set_timer(
                name="meta_router_check",
                interval=timedelta(seconds=int(delay_sec)),
                callback=self._on_signal_check,
            )

        def _on_signal_check(self, event=None) -> None:
            if (
                self._cfg is None
                or self._arches is None
                or self._feature_computer is None
            ):
                return

            # Clean up closed positions from tracking
            self._cleanup_closed_positions()
            dl = self._cfg.decision_loop or {}
            interval_min = int(dl.get("check_interval_minutes", 10))
            orderflow_win = int(dl.get("orderflow_window_minutes", 15))
            min_order_interval_min = int(
                dl.get("min_order_interval_minutes", interval_min)
            )

            now_ns = int(self.clock.timestamp_ns())
            if (
                self._last_order_time_ns is not None
                and now_ns - self._last_order_time_ns
                < min_order_interval_min * 60 * 1_000_000_000
            ):
                self._schedule_next_check()
                return

            # Build streaming feature dict
            feats: Dict[str, Any] = dict(self._feature_computer.get_features() or {})
            feats.update(
                self._feature_computer.get_orderflow_features(
                    window_minutes=orderflow_win
                )
            )

            # Precompute observability (before early returns)
            last_tick_ts = self._feature_computer.get_last_tick_ts_ns()
            tick_gap_seconds = compute_tick_gap_seconds(
                now_ns=now_ns, last_tick_ts_ns=last_tick_ts
            )
            required_keys_for_missing = []
            if self._live_feature_contract is not None:
                required_keys_for_missing += list(
                    self._live_feature_contract.required_keys_any or []
                )
                if self._inferencer is not None:
                    required_keys_for_missing += list(
                        self._live_feature_contract.required_pred_keys or []
                    )
            feature_missing_rate = compute_feature_missing_rate(
                required_keys=required_keys_for_missing, features=feats
            )

            def _emit_stage(
                *,
                stage: str,
                decision_id: str,
                decision_ts_ns: int,
                data: dict[str, Any] | None,
            ) -> None:
                writer = self._exec_stage_writers.get(stage)
                if writer is None:
                    return
                record = build_stage_record(
                    stage=stage,
                    decision_id=decision_id,
                    decision_ts_ns=decision_ts_ns,
                    source="live",
                    run_id=(
                        str(os.getenv("MLBOT_RUN_ID"))
                        if os.getenv("MLBOT_RUN_ID")
                        else None
                    ),
                    symbol=str(self.instrument_id),
                    timeframe=str(self.bar_type),
                    strategy_name=str(self.strategy_name),
                    instrument_id=str(self.instrument_id),
                    data=data,
                )
                try:
                    writer.write(record, decision_ts_ns=decision_ts_ns)
                except Exception:
                    pass

            def _emit_log(
                *,
                router_mode: Optional[str],
                gate_blocked: bool,
                gate_decisions: list[str],
                gate_reasons: Optional[dict[str, list[str]]] = None,
                evidence: Optional[dict[str, bool]] = None,
                execution: Optional[dict[str, Any]] = None,
                observability: Optional[dict[str, Any]] = None,
            ) -> None:
                decision_id = build_decision_id(
                    strategy_name=str(self.strategy_name),
                    symbol=str(self.instrument_id),
                    decision_ts_ns=now_ns,
                )
                _emit_stage(
                    stage="features",
                    decision_id=decision_id,
                    decision_ts_ns=now_ns,
                    data=feats,
                )
                preds = {
                    k: feats.get(k)
                    for k in [
                        "pred_dir_prob",
                        "pred_mfe_atr",
                        "pred_mae_atr",
                        "pred_t_to_mfe",
                    ]
                    if k in feats
                }
                _emit_stage(
                    stage="preds",
                    decision_id=decision_id,
                    decision_ts_ns=now_ns,
                    data=preds or None,
                )
                router = None
                if router_mode is not None:
                    rt = self._cfg.router_thresholds or {}
                    router = {
                        "mode": str(router_mode),
                        "thresholds": dict(rt),
                        "scores": {
                            "head_dir_score": feats.get("head_dir_score"),
                            "head_mfe_atr": feats.get("head_mfe_atr"),
                            "head_mae_atr": feats.get("head_mae_atr"),
                            "head_t_to_mfe": feats.get("head_t_to_mfe"),
                        },
                    }
                _emit_stage(
                    stage="router",
                    decision_id=decision_id,
                    decision_ts_ns=now_ns,
                    data=router,
                )
                _emit_stage(
                    stage="gate",
                    decision_id=decision_id,
                    decision_ts_ns=now_ns,
                    data={
                        "blocked": bool(gate_blocked),
                        "decisions": gate_decisions,
                        "reasons": gate_reasons or {},
                    },
                )
                if evidence is not None:
                    _emit_stage(
                        stage="evidence",
                        decision_id=decision_id,
                        decision_ts_ns=now_ns,
                        data=evidence,
                    )
                if execution is not None:
                    _emit_stage(
                        stage="execution",
                        decision_id=decision_id,
                        decision_ts_ns=now_ns,
                        data=execution,
                    )
                if observability is not None:
                    _emit_stage(
                        stage="observability",
                        decision_id=decision_id,
                        decision_ts_ns=now_ns,
                        data=observability,
                    )

            # Online nnmultihead inference (optional)
            if self._inferencer is not None:
                try:
                    preds = self._inferencer.predict_one(feats)
                    feats.update(preds)
                except Exception as exc:
                    err_reason = f"inference_error:{type(exc).__name__}"
                    _emit_log(
                        router_mode="NO_TRADE",
                        gate_blocked=True,
                        gate_decisions=["live_feature_contract_violation"],
                        gate_reasons={"contract": [err_reason]},
                        evidence=None,
                        execution={"intent": False, "submit_order": False},
                        observability={
                            "tick_gap_seconds": tick_gap_seconds,
                            "feature_missing_rate": feature_missing_rate,
                        },
                    )
                    self._schedule_next_check()
                    return

            # Runtime validate live feature contract BEFORE any router/archetype decision.
            if self._live_feature_contract is not None:
                ok, reasons = validate_live_features_v1(
                    contract=self._live_feature_contract,
                    features=feats,
                    nn_inference_enabled=(self._inferencer is not None),
                )
                if not ok:
                    # Record reason for auditability (file output is optional but helpful).
                    try:
                        snap_dir = Path(
                            os.getenv(
                                "MLBOT_LIVE_SNAPSHOT_DIR",
                                "results/live_snapshots",
                            )
                        )
                        now_iso = datetime.fromtimestamp(
                            now_ns / 1e9, tz=timezone.utc
                        ).isoformat()
                        out_path = snap_dir / f"system_state_snapshot_{now_ns}.json"
                        meta = (
                            self._exec.meta()
                            if getattr(self, "_exec", None) is not None
                            else {}
                        )
                        active_slots = (
                            int(self._st.slots.active_count())
                            if getattr(self, "_st", None) is not None
                            else None
                        )
                        write_state_snapshot(
                            out_path=out_path,
                            snapshot=SystemStateSnapshot(
                                task_id=(
                                    str(os.getenv("MLBOT_TASK_ID"))
                                    if os.getenv("MLBOT_TASK_ID")
                                    else None
                                ),
                                timestamp=now_iso,
                                constitution_hash=(
                                    str(meta.get("constitution_hash"))
                                    if meta.get("constitution_hash")
                                    else None
                                ),
                                constitution_yaml=(
                                    str(meta.get("constitution_yaml"))
                                    if meta.get("constitution_yaml")
                                    else None
                                ),
                                router_mode="NO_TRADE",
                                gate_decisions={
                                    "live_feature_contract_violation": reasons
                                },
                                pcm_budget={},
                                active_slots=active_slots,
                                drawdown=None,
                            ),
                        )
                    except Exception:
                        pass
                    self.log.warning(
                        "⚠️ live_feature_contract_violation -> NO_TRADE | "
                        + "; ".join(reasons)
                    )
                    self._schedule_next_check()
                    return

            # Router decision (in-process function; NOT the CLI command)
            regime = None
            if all(
                k in feats
                for k in [
                    "pred_dir_prob",
                    "pred_mfe_atr",
                    "pred_mae_atr",
                    "pred_t_to_mfe",
                ]
            ):
                rt = self._cfg.router_thresholds or {}
                if "preds_in_log1p" in rt:
                    preds_in_log1p = bool(rt.get("preds_in_log1p"))
                elif self._inferencer is not None:
                    preds_in_log1p = bool(self._inferencer.preds_in_log1p())
                else:
                    preds_in_log1p = True
                cfg = Rule3ActionConfig(
                    mfe_min=float(rt.get("mfe_min", 0.4)),
                    eff_min=float(rt.get("eff_min", 1.05)),
                    dir_conf_trend_min=float(rt.get("dir_conf_trend_min", 0.25)),
                    mfe_trend_min=float(rt.get("mfe_trend_min", 0.8)),
                    ttm_trend_min=float(rt.get("ttm_trend_min", 8.0)),
                    eff_mean_min=float(rt.get("eff_mean_min", 1.15)),
                    ttm_mean_max=float(rt.get("ttm_mean_max", 12.0)),
                )
                calibration = None
                calib_path = (
                    rt.get("calibration_json") if isinstance(rt, dict) else None
                )
                if calib_path:
                    try:
                        calibration = json.loads(
                            Path(str(calib_path)).read_text(encoding="utf-8")
                        )
                    except Exception:
                        calibration = None
                mode_df = compute_mode_3action(
                    pd.DataFrame([feats]),
                    cfg=cfg,
                    preds_in_log1p=preds_in_log1p,
                    calibration=calibration,
                )
                regime = str(mode_df["mode"].iloc[0]).upper()
            if regime is None:
                regime = _infer_regime_placeholder()
            if regime == "NO_TRADE":
                self._mode_hist.append("NO_TRADE")
                mode_entropy = compute_router_mode_entropy(list(self._mode_hist))
                self.log.info(
                    f"OBS mode=NO_TRADE tick_gap_s={tick_gap_seconds} missing_rate={feature_missing_rate} mode_entropy={mode_entropy}"
                )
                _emit_log(
                    router_mode="NO_TRADE",
                    gate_blocked=False,
                    gate_decisions=[],
                    gate_reasons={},
                    evidence=None,
                    execution={"intent": False, "submit_order": False},
                    observability={
                        "tick_gap_seconds": tick_gap_seconds,
                        "feature_missing_rate": feature_missing_rate,
                        "router_mode_entropy": mode_entropy,
                    },
                )
                self._schedule_next_check()
                return
            self._mode_hist.append(str(regime).upper())

            archetype_id = select_first_enabled_archetype(self._cfg, regime=regime)
            if not archetype_id:
                _emit_log(
                    router_mode=str(regime),
                    gate_blocked=True,
                    gate_decisions=["evidence_dsl_error"],
                    gate_reasons={"evidence": ["evidence_dsl_error"]},
                    evidence=None,
                    execution={"intent": True, "submit_order": False},
                )
                self._schedule_next_check()
                return

            arch = self._arches.get(archetype_id)
            if arch is None:
                self._schedule_next_check()
                return

            if self._cfg.vol_mean.enabled and regime == "MEAN":
                overlay_id = self._cfg.vol_mean.archetype_id
                if overlay_id in self._arches:
                    arch = self._arches[overlay_id]

            try:
                quantiles = load_evidence_quantiles(
                    os.getenv("MLBOT_EVIDENCE_QUANTILES_JSON")
                )
                evidence = compute_execution_evidence(
                    features=feats,
                    rules=list(arch.evidence_rules or []),
                    quantiles=quantiles,
                )
            except Exception as e:
                # Fail-closed: evidence DSL config/key mismatch should block trading.
                self.log.error(f"❌ evidence_dsl_error -> NO_TRADE: {e}")
                self._mode_hist.append("NO_TRADE")
                self._schedule_next_check()
                return
            evidence_true_rate = compute_evidence_true_rate(evidence)
            mode_entropy = compute_router_mode_entropy(list(self._mode_hist))
            self.log.info(
                f"OBS mode={regime} arch={arch.name} tick_gap_s={tick_gap_seconds} missing_rate={feature_missing_rate} "
                f"evidence_true_rate={evidence_true_rate} mode_entropy={mode_entropy}"
            )

            # Gate rules (optional, defined per archetype)
            when_then = list(getattr(arch, "when_then_rules", []) or [])
            gate_cfg = (
                {
                    "when_then_rules": when_then,
                    "default_action": getattr(arch, "default_action", "deny"),
                }
                if when_then
                else (getattr(arch, "gate_rules", None) or {})
            )
            if gate_cfg:
                ok3, reasons3 = apply_gate_rules(
                    gate_rules=gate_cfg,
                    features=feats,
                    quantiles=quantiles,
                )
                if not ok3:
                    self.log.info(
                        f"ℹ️ gate_rules_veto: {arch.name} | " + "; ".join(reasons3[:6])
                    )
                    _emit_log(
                        router_mode=str(regime),
                        gate_blocked=True,
                        gate_decisions=["gate_rules_veto"],
                        gate_reasons={"gate_rules": list(reasons3 or [])},
                        evidence=evidence,
                        execution={"intent": True, "submit_order": False},
                    )
                    self._schedule_next_check()
                    return

            # Execution archetype heuristics (v1, fail-closed)
            bars = self._feature_computer.get_recent_bars(200)
            hd = evaluate_required_conditions(
                archetype_name=str(arch.name),
                regime=str(arch.regime),
                required_conditions=list(arch.required_conditions or []),
                feats=feats,
                bars=bars,
            )
            if not bool(hd.ok):
                self.log.info(
                    f"ℹ️ archetype_heuristics_blocked: {arch.name} | "
                    + "; ".join(hd.reasons[:6])
                )
                _emit_log(
                    router_mode=str(regime),
                    gate_blocked=True,
                    gate_decisions=["archetype_heuristics_blocked"],
                    gate_reasons={"heuristics": list(hd.reasons or [])},
                    evidence=evidence,
                    execution={"intent": True, "submit_order": False},
                )
                self._schedule_next_check()
                return

            # Optional exported execution rules veto (tree-distilled hook)
            if self._execution_rules is not None:
                ok2, reasons2 = apply_execution_rules(
                    rules=self._execution_rules,
                    archetype_name=str(arch.name),
                    features=feats,
                )
                if not ok2:
                    self.log.info(
                        f"ℹ️ execution_rules_veto: {arch.name} | "
                        + "; ".join(reasons2[:6])
                    )
                    _emit_log(
                        router_mode=str(regime),
                        gate_blocked=True,
                        gate_decisions=["execution_rules_veto"],
                        gate_reasons={"execution_rules": list(reasons2 or [])},
                        evidence=evidence,
                        execution={"intent": True, "submit_order": False},
                    )
                    self._schedule_next_check()
                    return

            # Structural direction resolution (per-archetype policy)
            direction_policy = dict(getattr(arch, "direction_policy", None) or {})
            direction = resolve_direction(
                archetype_name=str(arch.name),
                policy=direction_policy,
                feats=feats,
                bars=bars,
            )
            if not direction.ok or direction.side is None:
                self.log.info(
                    f"ℹ️ direction_unresolved: {arch.name} | {direction.reason}"
                )
                _emit_log(
                    router_mode=str(regime),
                    gate_blocked=True,
                    gate_decisions=["direction_unresolved"],
                    gate_reasons={"direction": [direction.reason]},
                    evidence=evidence,
                    execution={"intent": True, "submit_order": False},
                )
                self._schedule_next_check()
                return

            # FR/ET low-frequency constraint (if configured)
            constraints = getattr(arch, "execution_constraints", None) or {}
            min_interval_m = float(constraints.get("min_order_interval_minutes", 0.0))
            if min_interval_m > 0 and self._last_order_time_ns is not None:
                delta_sec = (now_ns - int(self._last_order_time_ns)) / 1e9
                if delta_sec < (min_interval_m * 60.0):
                    self.log.info(
                        f"ℹ️ execution_constraints_rate_limit: {arch.name} | "
                        f"min_interval_minutes={min_interval_m}"
                    )
                    _emit_log(
                        router_mode=str(regime),
                        gate_blocked=True,
                        gate_decisions=["execution_constraints_rate_limit"],
                        gate_reasons={
                            "execution_constraints": [
                                f"min_interval_minutes={min_interval_m}"
                            ]
                        },
                        evidence=evidence,
                        execution={"intent": True, "submit_order": False},
                    )
                    self._schedule_next_check()
                    return

            pcm_budget = {}
            try:
                pcm_result = compute_pcm_budget_for_decisions(
                    decisions=[
                        PCMSymbolDecision(
                            symbol=str(self.instrument_id),
                            mode=str(regime),
                            gated=True,
                            score=float(feats.get("pred_dir_prob", 0.5)),
                        )
                    ]
                )
                pcm_budget = {
                    "global_pause": bool(pcm_result.global_pause),
                    "per_mode_budget": dict(pcm_result.per_mode_budget or {}),
                    "per_symbol_budget": dict(pcm_result.per_symbol_budget or {}),
                    "reasons": list(pcm_result.reasons or []),
                }
            except Exception:
                pcm_budget = {}

            size_mult = float(self._cfg.size_multipliers.get(str(arch.name), 1.0))
            if self._cfg.vol_mean.enabled and str(arch.name) == str(
                self._cfg.vol_mean.archetype_id
            ):
                size_mult = float(self._cfg.vol_mean.size_multiplier)
            if pcm_budget:
                sym_key = str(self.instrument_id)
                sym_mult = float(
                    (pcm_budget.get("per_symbol_budget") or {}).get(sym_key, 1.0)
                )
                size_mult *= max(0.0, sym_mult)
            exec_profile = build_execution_profile(
                archetype_name=str(arch.name),
                feats=feats,
                constraints=constraints,
            )
            size_mult *= float(exec_profile.get("size_multiplier", 1.0))
            qty = self.instrument.make_qty(
                self.trade_size * max(0.0, size_mult) * float(hd.risk_multiplier)
            )
            # Include archetype in order tags for position tracking
            order_tags = [str(self.strategy_name), str(arch.name)]
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=(
                    OrderSide.BUY if str(direction.side) == "BUY" else OrderSide.SELL
                ),
                quantity=qty,
            )
            # Set order tags if supported
            if hasattr(order, "tags"):
                order.tags = order_tags
            self._xm.submit_order_guarded(
                order=order,
                ctx=GuardedOrderContext(
                    position_id=f"{self.strategy_name}:{int(now_ns)}",
                    symbol=str(self.instrument_id),
                    archetype=str(arch.regime),
                    execution_strategy=str(arch.name),
                    execution_tags=order_tags,
                    execution_evidence=evidence,
                ),
            )
            self._last_order_time_ns = now_ns

            # ET hedge pairing is disabled under 6-archetype routing.

            _emit_log(
                router_mode=str(regime),
                gate_blocked=False,
                gate_decisions=[],
                gate_reasons={},
                evidence=evidence,
                execution={
                    "intent": True,
                    "submit_order": True,
                    "side": str(direction.side),
                    "direction_source": str(direction.source),
                    "direction_method": str(direction.method),
                    "qty": float(qty),
                    "price": None,
                    "reason": str(arch.name),
                    "rr_constraints": exec_profile.get("rr_constraints"),
                    "execution_profile": exec_profile,
                },
                observability={
                    "tick_gap_seconds": tick_gap_seconds,
                    "feature_missing_rate": feature_missing_rate,
                    "evidence_true_rate": evidence_true_rate,
                    "router_mode_entropy": mode_entropy,
                },
            )

            # Persist latest snapshot for auditability (overwrite by default).
            try:
                snap_dir = Path(
                    os.getenv("MLBOT_LIVE_SNAPSHOT_DIR", "results/live_snapshots")
                )
                snap_dir.mkdir(parents=True, exist_ok=True)
                meta = (
                    self._exec.meta()
                    if getattr(self, "_exec", None) is not None
                    else {}
                )
                now_iso = datetime.fromtimestamp(
                    now_ns / 1e9, tz=timezone.utc
                ).isoformat()
                obs = {
                    "tick_gap_seconds": tick_gap_seconds,
                    "feature_missing_rate": feature_missing_rate,
                    "evidence_true_rate": evidence_true_rate,
                    "router_mode_entropy": mode_entropy,
                }
                snap = SystemStateSnapshot(
                    task_id=(
                        str(os.getenv("MLBOT_TASK_ID"))
                        if os.getenv("MLBOT_TASK_ID")
                        else None
                    ),
                    timestamp=now_iso,
                    constitution_hash=(
                        str(meta.get("constitution_hash"))
                        if meta.get("constitution_hash")
                        else None
                    ),
                    constitution_yaml=(
                        str(meta.get("constitution_yaml"))
                        if meta.get("constitution_yaml")
                        else None
                    ),
                    router_mode=str(regime),
                    gate_decisions={},
                    pcm_budget=pcm_budget,
                    active_slots=(
                        int(self._st.slots.active_count())
                        if self._st is not None
                        else None
                    ),
                    drawdown=None,
                    observability=obs,
                )
                write_state_snapshot(
                    out_path=snap_dir / "latest_system_state_snapshot.json",
                    snapshot=snap,
                )
            except Exception:
                pass
            self._schedule_next_check()

else:

    class MetaRouterStrategy:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Nautilus Trader is not installed. Install it with: pip install nautilus-trader"
            )
