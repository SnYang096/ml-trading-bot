from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

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
from src.time_series_model.live.execution_rules import (
    apply_execution_rules,
    load_execution_rules,
)
from src.time_series_model.live.tree_gate import apply_gate_rules
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
            gate_cfg = getattr(arch, "gate_rules", None) or {}
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
            qty = self.instrument.make_qty(
                self.trade_size * max(0.0, size_mult) * float(hd.risk_multiplier)
            )
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY if str(hd.side) == "BUY" else OrderSide.SELL,
                quantity=qty,
            )
            self._xm.submit_order_guarded(
                order=order,
                ctx=GuardedOrderContext(
                    position_id=f"{self.strategy_name}:{int(now_ns)}",
                    symbol=str(self.instrument_id),
                    mode=str(arch.regime),
                    execution_strategy=str(arch.name),
                    execution_tags=[str(self.strategy_name)],
                    execution_evidence=evidence,
                ),
            )
            self._last_order_time_ns = now_ns
            _emit_log(
                router_mode=str(regime),
                gate_blocked=False,
                gate_decisions=[],
                gate_reasons={},
                evidence=evidence,
                execution={
                    "intent": True,
                    "submit_order": True,
                    "side": str(hd.side),
                    "qty": float(qty),
                    "price": None,
                    "reason": str(arch.name),
                    "rr_constraints": (
                        constraints.get("fixed_rr") if constraints else None
                    ),
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
