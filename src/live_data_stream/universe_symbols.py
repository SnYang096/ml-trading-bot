"""Resolve live trading symbols from ``live/{universe}/universe.yaml``.

Design (live production):
  1. ``universe.yaml`` defines the full symbol set the feature-bus publishes.
  2. Each strategy filters that set via ``meta.yaml`` ``symbol_include`` /
     ``symbol_exclude`` (see ``live_symbol_plan.resolve_live_classic_symbol_plan``).
  3. ``MLBOT_LIVE_SYMBOLS`` is a legacy override for the bus set only; prefer
     editing ``universe.yaml`` and per-strategy meta.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_symbols_csv(raw: str) -> List[str]:
    """Parse comma-separated symbols to sorted unique uppercase tokens."""
    out: List[str] = []
    seen: set[str] = set()
    for chunk in str(raw or "").replace("|", ",").replace(";", ",").split(","):
        token = chunk.strip().upper()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def universe_yaml_path(universe: str, *, project_root: Optional[Path] = None) -> Path:
    root = project_root or PROJECT_ROOT
    name = str(universe or "highcap").strip() or "highcap"
    return root / "live" / name / "universe.yaml"


def read_universe_symbols(
    universe: str = "highcap",
    *,
    project_root: Optional[Path] = None,
) -> List[str]:
    """Return sorted symbol keys from ``live/{universe}/universe.yaml``."""
    path = universe_yaml_path(universe, project_root=project_root)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    symbols = cfg.get("symbols") or {}
    if not isinstance(symbols, dict):
        return []
    return sorted(str(k).strip().upper() for k in symbols if str(k).strip())


def resolve_symbols_csv(
    *,
    cli_symbols: Optional[str] = None,
    universe: str = "highcap",
    env_symbols: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> str:
    """Resolve comma-separated symbols: CLI > universe.yaml > env > error."""
    if cli_symbols is not None and str(cli_symbols).strip():
        return str(cli_symbols).strip()
    try:
        uni = read_universe_symbols(universe, project_root=project_root)
    except FileNotFoundError:
        uni = []
    if uni:
        return ",".join(uni)
    env = (env_symbols or "").strip()
    if env:
        return env
    raise ValueError(
        f"no symbols: pass --symbols or create live/{universe}/universe.yaml"
    )


def resolve_bus_symbols_csv(
    *,
    cli_symbols: Optional[str] = None,
    universe: str = "highcap",
    project_root: Optional[Path] = None,
) -> str:
    """Feature-bus publisher: CLI > universe.yaml only (no env fallback)."""
    return resolve_symbols_csv(
        cli_symbols=cli_symbols,
        universe=universe,
        env_symbols="",
        project_root=project_root,
    )


def resolve_bus_symbols(
    *,
    cli_symbols: Optional[str] = None,
    universe: str = "highcap",
    project_root: Optional[Path] = None,
) -> List[str]:
    """Feature-bus publisher symbol list (universe.yaml keys; CLI override only)."""
    return parse_symbols_csv(
        resolve_bus_symbols_csv(
            cli_symbols=cli_symbols,
            universe=universe,
            project_root=project_root,
        )
    )
