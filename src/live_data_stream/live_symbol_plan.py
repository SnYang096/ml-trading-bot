"""Live symbol resolution: universe.yaml for bus, strategy meta for filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.config.strategy_layout import resolve_strategy_package_under_root
from src.live_data_stream.universe_symbols import (
    parse_symbols_csv,
    read_universe_symbols,
    resolve_symbols_csv,
)


@dataclass(frozen=True)
class LiveSymbolPlan:
    """Resolved symbols for a live consumer process."""

    universe: str
    bus_symbols: List[str]
    strategy_symbols: Dict[str, List[str]]

    @property
    def listener_symbols(self) -> List[str]:
        """Symbols that need bus listeners (full bus universe)."""
        return list(self.bus_symbols)

    @property
    def active_union(self) -> List[str]:
        """Union of symbols any enabled strategy may trade."""
        out: List[str] = []
        seen: set[str] = set()
        for syms in self.strategy_symbols.values():
            for s in syms:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return sorted(out)


def resolve_bus_symbols(
    *,
    universe: str = "highcap",
    cli_symbols: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> List[str]:
    """Symbols the feature-bus must publish (universe.yaml keys; CLI override only)."""
    csv = resolve_symbols_csv(
        cli_symbols=cli_symbols,
        universe=universe,
        env_symbols="",
        project_root=project_root,
    )
    return parse_symbols_csv(csv)


def resolve_live_classic_symbol_plan(
    *,
    universe: str,
    strategies_root: str | Path,
    enabled_archetypes: Sequence[str],
    cli_symbols: Optional[str] = None,
    env_symbols: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> LiveSymbolPlan:
    """Classic trend consumer (run_live): bus = universe; strategies filter via meta.yaml.

    Priority for *bus* symbol set:
      1. ``cli_symbols`` (explicit override)
      2. ``env_symbols`` (legacy ``MLBOT_LIVE_SYMBOLS`` override)
      3. ``live/{universe}/universe.yaml`` keys
    """
    from scripts.pipeline.strategy_symbols import resolve_strategy_symbols

    if cli_symbols is not None and str(cli_symbols).strip():
        bus = parse_symbols_csv(cli_symbols)
    elif env_symbols is not None and str(env_symbols).strip():
        bus = parse_symbols_csv(env_symbols)
    else:
        bus = read_universe_symbols(universe, project_root=project_root)

    root = Path(strategies_root)
    strategy_symbols: Dict[str, List[str]] = {}
    for arch in enabled_archetypes:
        name = str(arch or "").strip()
        if not name:
            continue
        pkg = resolve_strategy_package_under_root(
            root, name, allow_bad_candidates=False
        )
        sel = resolve_strategy_symbols(
            strategy=name,
            base_symbols=bus,
            strategy_config_dir=pkg if pkg.is_dir() else None,
        )
        strategy_symbols[name] = list(sel.resolved_symbols)

    return LiveSymbolPlan(
        universe=str(universe or "highcap").strip() or "highcap",
        bus_symbols=list(bus),
        strategy_symbols=strategy_symbols,
    )
