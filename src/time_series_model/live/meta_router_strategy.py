from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from nautilus_trader.model import (
        Bar,
        InstrumentId,
        BarType,
        OrderSide,
        TradeTick,
        QuoteTick,
    )
    from nautilus_trader.trading import Strategy

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


def _infer_regime_placeholder() -> str:
    """
    Placeholder router hook:
    - live should later plug in real Router (nnmultihead outputs + rules/gates).
    """
    return str(os.getenv("MLBOT_LIVE_FORCE_REGIME", "NO_TRADE")).upper()


if NAUTILUS_AVAILABLE:

    class MetaRouterStrategy(Strategy):
        """
        One Strategy process, multiple archetypes (per docs/live_stream/策略一起还是分开.md).

        NOTE: Router decision is currently a placeholder (env override). This class is primarily
        to enforce correct *architecture*: one account, one world-view, one order stream.
        """

        def __init__(
            self,
            *,
            strategy_name: str,
            instrument_id: InstrumentId,
            bar_type: BarType,
            trade_size: float,
            live_config_path: str = "config/nnmultihead/live/meta_router_live_config_v1.yaml",
            constitution_yaml: Optional[str] = None,
            archetype_registry_path: str = "config/nnmultihead/execution_archetypes_v1.yaml",
        ):
            super().__init__()
            self.strategy_name = str(strategy_name)
            self.instrument_id = instrument_id
            self.bar_type = bar_type
            self.trade_size = float(trade_size)

            self.live_config_path = str(live_config_path)
            self.constitution_yaml = constitution_yaml or os.getenv(
                "MLBOT_CONSTITUTION_YAML", "config/constitution/constitution_v1.yaml"
            )
            self.archetype_registry_path = str(
                os.getenv("MLBOT_NNMH_EXEC_ARCHETYPE_REGISTRY", archetype_registry_path)
            )

            self._cfg: Optional[MetaRouterLiveConfig] = None
            self._exec = None
            self._st = None
            self._arches = None
            self._feature_computer: Optional[IncrementalFeatureComputer] = None
            self._inferencer: Optional[NNMHLiveInferencer] = None
            self._last_order_time_ns: Optional[int] = None

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
            self.subscribe_bars(self.bar_type)
            # Orderflow must update on trade ticks (not bar)
            self.subscribe_trade_ticks(self.instrument_id)

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
            self.clock.set_timer(
                name="meta_router_check",
                interval=int(delay_sec),
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

            # Online nnmultihead inference (optional)
            if self._inferencer is not None:
                try:
                    preds = self._inferencer.predict_one(feats)
                    feats.update(preds)
                except Exception:
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
                mode_df = compute_mode_3action(
                    pd.DataFrame([feats]), cfg=cfg, preds_in_log1p=preds_in_log1p
                )
                regime = str(mode_df["mode"].iloc[0]).upper()
            if regime is None:
                regime = _infer_regime_placeholder()
            if regime == "NO_TRADE":
                self._schedule_next_check()
                return

            archetype_id = select_first_enabled_archetype(self._cfg, regime=regime)
            if not archetype_id:
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

            evidence = compute_execution_evidence(
                features=feats, rules=list(arch.evidence_rules or [])
            )

            size_mult = float(self._cfg.size_multipliers.get(str(arch.name), 1.0))
            if self._cfg.vol_mean.enabled and str(arch.name) == str(
                self._cfg.vol_mean.archetype_id
            ):
                size_mult = float(self._cfg.vol_mean.size_multiplier)
            qty = self.instrument.make_qty(self.trade_size * max(0.0, size_mult))
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
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
            self._schedule_next_check()

else:

    class MetaRouterStrategy:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Nautilus Trader is not installed. Install it with: pip install nautilus-trader"
            )
