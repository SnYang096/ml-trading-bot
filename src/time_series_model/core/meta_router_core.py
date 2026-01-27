from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
    load_evidence_quantiles,
)
from src.time_series_model.live.direction_resolver import resolve_direction
from src.time_series_model.live.meta_router_config import load_meta_router_live_config
from src.time_series_model.live.tree_gate import apply_gate_rules
from src.time_series_model.nnmultihead.strategy_profile import (
    ExecutionArchetype,
    load_execution_archetypes_registry,
)
from src.time_series_model.portfolio.pcm import (
    SymbolDecision,
    compute_pcm_budget_for_decisions,
)
from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    compute_mode_3action,
)
from src.time_series_model.live.execution_intelligence import build_execution_profile


@dataclass(frozen=True)
class TradeIntent:
    action: str  # LONG|SHORT|NO_TRADE
    symbol: str
    archetype: str
    execution_strategy: Optional[str] = None
    confidence: Optional[float] = None
    quantity: Optional[float] = None
    size_multiplier: Optional[float] = None
    position_id: Optional[str] = None
    add_position: bool = False
    parent_position_id: Optional[str] = None
    current_r: Optional[float] = None
    locked_profit: Optional[bool] = None
    execution_tags: Optional[List[str]] = None
    execution_evidence: Optional[Dict[str, bool]] = None
    execution_profile: Optional[Dict[str, Any]] = None
    pcm_budget: Optional[Dict[str, Any]] = None


@dataclass
class MetaRouterCoreConfig:
    live_config_path: str = "config/nnmultihead/live/meta_router_live_config.yaml"
    archetype_registry_path: str = "config/nnmultihead/execution_archetypes.yaml"
    evidence_quantiles_path: Optional[str] = None
    enabled_archetypes: Optional[Dict[str, List[str]]] = None
    size_multipliers: Optional[Dict[str, float]] = None
    router_thresholds: Optional[Dict[str, Any]] = None
    preds_in_log1p: Optional[bool] = None
    gate_enabled: bool = True
    gate_fail_open_missing_quantiles: bool = True
    use_pcm: bool = False


class MetaRouterCore:
    """
    Pure-logic meta router core.

    Input: feature dict (live, incremental) + symbol
    Output: list of TradeIntent
    """

    def __init__(self, cfg: Optional[MetaRouterCoreConfig] = None) -> None:
        self.cfg = cfg or MetaRouterCoreConfig()
        self._live_cfg: Optional[Any] = None
        self._arches: Dict[str, ExecutionArchetype] = {}
        self._quantiles: Dict[str, Any] | None = None
        self._load_configs()

    def _load_configs(self) -> None:
        try:
            self._live_cfg = load_meta_router_live_config(self.cfg.live_config_path)
        except Exception:
            self._live_cfg = None
        try:
            self._arches = load_execution_archetypes_registry(
                self.cfg.archetype_registry_path
            )
        except Exception:
            self._arches = {}
        self._quantiles = load_evidence_quantiles(self.cfg.evidence_quantiles_path)

    def _resolve_enabled_archetypes(self) -> Dict[str, List[str]]:
        if self.cfg.enabled_archetypes is not None:
            return {
                str(k).upper(): [str(x) for x in v]
                for k, v in (self.cfg.enabled_archetypes or {}).items()
            }
        if self._live_cfg is not None:
            return self._live_cfg.enabled_archetypes
        return {}

    def _resolve_size_multipliers(self) -> Dict[str, float]:
        if self.cfg.size_multipliers is not None:
            return {
                str(k): float(v) for k, v in (self.cfg.size_multipliers or {}).items()
            }
        if self._live_cfg is not None:
            return self._live_cfg.size_multipliers
        return {}

    def _resolve_router_thresholds(self) -> Dict[str, Any]:
        if self.cfg.router_thresholds is not None:
            return dict(self.cfg.router_thresholds or {})
        if self._live_cfg is not None:
            return self._live_cfg.router_thresholds
        return {}

    def _resolve_quantiles(self, symbol: str) -> Dict[str, Any] | None:
        if not self._quantiles:
            return None
        if symbol in self._quantiles and isinstance(self._quantiles[symbol], dict):
            return self._quantiles[symbol]
        return self._quantiles

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TradeIntent]:
        feats = features or {}
        required_preds = [
            "pred_dir_prob",
            "pred_mfe_atr",
            "pred_mae_atr",
            "pred_t_to_mfe",
        ]
        if any(k not in feats for k in required_preds):
            return []

        rt = self._resolve_router_thresholds()
        preds_in_log1p = (
            bool(self.cfg.preds_in_log1p)
            if self.cfg.preds_in_log1p is not None
            else bool(rt.get("preds_in_log1p", True))
        )
        router_cfg = Rule3ActionConfig(
            mfe_min=float(rt.get("mfe_min", 0.4)),
            eff_min=float(rt.get("eff_min", 1.05)),
            dir_conf_trend_min=float(rt.get("dir_conf_trend_min", 0.25)),
            mfe_trend_min=float(rt.get("mfe_trend_min", 0.8)),
            ttm_trend_min=float(rt.get("ttm_trend_min", 8.0)),
            eff_mean_min=float(rt.get("eff_mean_min", 1.15)),
            ttm_mean_max=float(rt.get("ttm_mean_max", 12.0)),
            trend_confirm_mode=str(rt.get("trend_confirm_mode", "and")),
        )
        df = pd.DataFrame([feats])
        try:
            mode_df = compute_mode_3action(
                df, cfg=router_cfg, preds_in_log1p=preds_in_log1p
            )
        except Exception:
            return []
        mode = str(mode_df.iloc[0]["mode"] or "NO_TRADE").upper()
        if mode == "NO_TRADE":
            return []

        enabled = self._resolve_enabled_archetypes()
        archetype_list = enabled.get(str(mode).upper()) or []
        archetype_id = str(archetype_list[0]) if archetype_list else None
        if not archetype_id:
            return []
        arch = self._arches.get(archetype_id)
        if arch is None:
            return []

        quantiles = self._resolve_quantiles(symbol)
        if self.cfg.gate_enabled and arch.gate_rules:
            ok, reasons = apply_gate_rules(
                gate_rules=arch.gate_rules, features=feats, quantiles=quantiles
            )
            if (
                not ok
                and quantiles is None
                and self.cfg.gate_fail_open_missing_quantiles
            ):
                ok = True
            if not ok:
                return []

        bars = bars or []
        direction = resolve_direction(
            archetype_name=arch.name,
            policy=arch.direction_policy,
            feats=feats,
            bars=bars,
        )
        if not direction.ok or direction.side not in {"BUY", "SELL"}:
            return []

        exec_profile = build_execution_profile(
            archetype_name=arch.name,
            feats=feats,
            constraints=arch.execution_constraints,
        )
        size_multiplier = float(
            (self._resolve_size_multipliers().get(arch.name, 1.0))
            * float(exec_profile.get("size_multiplier", 1.0))
        )
        evidence = compute_execution_evidence(
            features=feats, rules=arch.evidence_rules, quantiles=quantiles
        )

        confidence = abs(float(feats.get("pred_dir_prob", 0.5)) - 0.5) * 2.0
        action = "LONG" if direction.side == "BUY" else "SHORT"

        pcm_budget = None
        if self.cfg.use_pcm:
            decision = SymbolDecision(
                symbol=str(symbol), mode=str(mode), gated=True, score=float(confidence)
            )
            pcm = compute_pcm_budget_for_decisions(decisions=[decision])
            pcm_budget = {
                "global_pause": bool(pcm.global_pause),
                "per_mode_budget": dict(pcm.per_mode_budget or {}),
                "per_symbol_budget": dict(pcm.per_symbol_budget or {}),
                "reasons": list(pcm.reasons or []),
            }

        return [
            TradeIntent(
                action=action,
                symbol=str(symbol),
                archetype=str(arch.name),
                execution_strategy=str(arch.name),
                confidence=float(confidence),
                size_multiplier=size_multiplier,
                execution_tags=[str(mode), str(arch.name), str(direction.side)],
                execution_evidence=evidence,
                execution_profile=exec_profile,
                pcm_budget=pcm_budget,
            )
        ]
