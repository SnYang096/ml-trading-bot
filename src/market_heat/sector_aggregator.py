"""Aggregate symbol-level heat into sector and market scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .heat_calculator import HeatResult
from .sector_registry import SectorRegistry


@dataclass
class SectorHeat:
    name: str
    score: float
    state: str
    hot_count: int
    warm_count: int
    cold_count: int
    total_count: int
    member_heats: List[HeatResult] = field(default_factory=list)


@dataclass
class MarketHeat:
    score: float
    state: str
    sector_heats: Dict[str, SectorHeat] = field(default_factory=dict)
    symbol_heats: Dict[str, HeatResult] = field(default_factory=dict)


def _state_from_score(score: float) -> str:
    if score >= 0.5:
        return "HOT"
    if score >= 0.2:
        return "WARM"
    return "COLD"


def aggregate(
    symbol_heats: Dict[str, HeatResult],
    registry: SectorRegistry,
) -> MarketHeat:
    """Roll up symbol heats -> sector heats -> market heat.

    Sector score = simple mean of member symbol scores.
    Market score = weighted mean of proxy symbols (BTC/ETH by default).
    """
    sector_heats: Dict[str, SectorHeat] = {}

    for sector_name, sector_info in registry.sectors.items():
        members: List[HeatResult] = []
        for sym in sector_info.symbols:
            if sym in symbol_heats:
                members.append(symbol_heats[sym])

        if not members:
            sector_heats[sector_name] = SectorHeat(
                name=sector_name,
                score=0.0,
                state="COLD",
                hot_count=0,
                warm_count=0,
                cold_count=0,
                total_count=0,
            )
            continue

        avg_score = sum(m.score for m in members) / len(members)
        hot = sum(1 for m in members if m.state == "HOT")
        warm = sum(1 for m in members if m.state == "WARM")
        cold = sum(1 for m in members if m.state == "COLD")

        sector_heats[sector_name] = SectorHeat(
            name=sector_name,
            score=round(avg_score, 4),
            state=_state_from_score(avg_score),
            hot_count=hot,
            warm_count=warm,
            cold_count=cold,
            total_count=len(members),
            member_heats=members,
        )

    # Market score from proxy symbols
    market_score = 0.0
    total_weight = 0.0
    for sym, weight in zip(
        registry.market_proxy_symbols, registry.market_proxy_weights
    ):
        if sym in symbol_heats:
            market_score += symbol_heats[sym].score * weight
            total_weight += weight

    if total_weight > 0:
        market_score /= total_weight

    return MarketHeat(
        score=round(market_score, 4),
        state=_state_from_score(market_score),
        sector_heats=sector_heats,
        symbol_heats=symbol_heats,
    )
