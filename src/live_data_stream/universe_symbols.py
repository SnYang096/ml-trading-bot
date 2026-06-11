"""Resolve live trading symbols from ``live/{universe}/universe.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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
