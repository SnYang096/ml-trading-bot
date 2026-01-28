from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
    load_evidence_quantiles,
)
from src.time_series_model.live.direction_resolver import resolve_direction
from src.time_series_model.live.tree_gate import apply_gate_rules
from src.time_series_model.nnmultihead.strategy_profile import (
    ExecutionArchetype,
    load_execution_archetypes_registry,
)
from src.time_series_model.portfolio.pcm import (
    SymbolDecision,
    compute_pcm_budget_for_decisions,
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
    archetype_registry_path: str = "config/nnmultihead/execution_archetypes.yaml"
    evidence_quantiles_path: Optional[str] = None
    enabled_archetypes: Optional[List[str]] = (
        None  # Changed: direct list, no mode grouping
    )
    size_multipliers: Optional[Dict[str, float]] = None
    preds_in_log1p: Optional[bool] = None
    gate_enabled: bool = True
    gate_fail_open_missing_quantiles: bool = True
    use_pcm: bool = False
    db_path: Optional[str] = None  # Optional: read config from database


class MetaRouterCore:
    """
    Pure-logic meta router core.

    Input: feature dict (live, incremental) + symbol
    Output: list of TradeIntent
    """

    def __init__(self, cfg: Optional[MetaRouterCoreConfig] = None) -> None:
        self.cfg = cfg or MetaRouterCoreConfig()
        self._arches: Dict[str, ExecutionArchetype] = {}
        self._quantiles: Dict[str, Any] | None = None
        self._db_storage: Optional[Any] = None
        self._load_configs()

    def _load_configs(self) -> None:
        # Try to load from database first
        if self.cfg.db_path:
            try:
                from src.order_management.storage import Storage

                self._db_storage = Storage(db_path=self.cfg.db_path)
            except Exception:
                self._db_storage = None

        try:
            self._arches = load_execution_archetypes_registry(
                self.cfg.archetype_registry_path
            )
        except Exception:
            self._arches = {}
        self._quantiles = load_evidence_quantiles(self.cfg.evidence_quantiles_path)

    def _resolve_enabled_archetypes(self) -> List[str]:
        """Resolve enabled archetypes as a direct list (no mode grouping)"""
        # Priority: 1) explicit config, 2) database, 3) all archetypes
        if self.cfg.enabled_archetypes is not None:
            return [str(x) for x in self.cfg.enabled_archetypes]

        if self._db_storage is not None:
            cfg = self._db_storage.get_live_config()
            if cfg is not None:
                enabled = cfg.get("enabled_archetypes")
                if isinstance(enabled, list):
                    return [str(x) for x in enabled]

        # Default: all registered archetypes
        return list(self._arches.keys())

    def _resolve_size_multipliers(self) -> Dict[str, float]:
        if self.cfg.size_multipliers is not None:
            return {
                str(k): float(v) for k, v in (self.cfg.size_multipliers or {}).items()
            }

        if self._db_storage is not None:
            cfg = self._db_storage.get_live_config()
            if cfg is not None:
                multipliers = cfg.get("size_multipliers")
                if isinstance(multipliers, dict):
                    return {str(k): float(v) for k, v in multipliers.items()}
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

        # Evaluate all enabled archetypes (no mode-based filtering)
        enabled_list = self._resolve_enabled_archetypes()
        if not enabled_list:
            return []

        quantiles = self._resolve_quantiles(symbol)
        size_multipliers = self._resolve_size_multipliers()
        bars = bars or []
        intents: List[TradeIntent] = []

        # Evaluate each enabled archetype
        for archetype_id in enabled_list:
            arch = self._arches.get(archetype_id)
            if arch is None:
                continue

            # Gate check
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
                    continue  # Skip this archetype

            # Direction resolution
            direction = resolve_direction(
                archetype_name=arch.name,
                policy=arch.direction_policy,
                feats=feats,
                bars=bars,
            )
            if not direction.ok or direction.side not in {"BUY", "SELL"}:
                continue  # Skip this archetype

            # Build execution profile
            exec_profile = build_execution_profile(
                archetype_name=arch.name,
                feats=feats,
                constraints=arch.execution_constraints,
            )
            size_multiplier = float(
                (size_multipliers.get(arch.name, 1.0))
                * float(exec_profile.get("size_multiplier", 1.0))
            )
            evidence = compute_execution_evidence(
                features=feats, rules=arch.evidence_rules, quantiles=quantiles
            )

            confidence = float(
                exec_profile.get("signals", {}).get(
                    "confidence",
                    abs(float(feats.get("pred_dir_prob", 0.5)) - 0.5) * 2.0,
                )
            )
            action = "LONG" if direction.side == "BUY" else "SHORT"

            pcm_budget = None
            if self.cfg.use_pcm:
                decision = SymbolDecision(
                    symbol=str(symbol), mode="", gated=True, score=float(confidence)
                )
                pcm = compute_pcm_budget_for_decisions(decisions=[decision])
                pcm_budget = {
                    "global_pause": bool(pcm.global_pause),
                    "per_mode_budget": dict(pcm.per_mode_budget or {}),
                    "per_symbol_budget": dict(pcm.per_symbol_budget or {}),
                    "reasons": list(pcm.reasons or []),
                }

            intents.append(
                TradeIntent(
                    action=action,
                    symbol=str(symbol),
                    archetype=str(arch.name),
                    execution_strategy=str(arch.name),
                    confidence=float(confidence),
                    size_multiplier=size_multiplier,
                    execution_tags=[str(arch.name), str(direction.side)],
                    execution_evidence=evidence,
                    execution_profile=exec_profile,
                    pcm_budget=pcm_budget,
                )
            )

        return intents
