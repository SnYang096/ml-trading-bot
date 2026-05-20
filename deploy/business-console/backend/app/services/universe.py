"""Load trading universe symbols from live/highcap/universe.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml


def load_universe_symbols(path: Path) -> List[str]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    symbols = raw.get("symbols") or {}
    if not isinstance(symbols, dict):
        return []
    return sorted(str(k).strip().upper() for k in symbols if str(k).strip())
