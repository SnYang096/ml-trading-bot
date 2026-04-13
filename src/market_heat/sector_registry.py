"""Load crypto sector taxonomy from YAML config.

Provides symbol-to-sector mapping and sector metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "market_heat" / "crypto_sectors.yaml"


@dataclass
class SectorInfo:
    name: str
    description: str
    symbols: List[str]


@dataclass
class SectorRegistry:
    sectors: Dict[str, SectorInfo] = field(default_factory=dict)
    symbol_to_sector: Dict[str, str] = field(default_factory=dict)
    market_proxy_symbols: List[str] = field(default_factory=list)
    market_proxy_weights: List[float] = field(default_factory=list)

    @property
    def all_symbols(self) -> List[str]:
        seen: set = set()
        out: list = []
        for si in self.sectors.values():
            for s in si.symbols:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        for s in self.market_proxy_symbols:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def sector_for(self, symbol: str) -> Optional[str]:
        base = symbol.replace("USDT", "").replace("/USDT:USDT", "")
        return self.symbol_to_sector.get(base)


def load_sector_registry(path: Optional[Path] = None) -> SectorRegistry:
    cfg_path = path or _DEFAULT_CONFIG
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    reg = SectorRegistry()

    for sector_name, sector_data in raw.get("sectors", {}).items():
        symbols = sector_data.get("symbols", [])
        desc = sector_data.get("description", "")
        reg.sectors[sector_name] = SectorInfo(
            name=sector_name,
            description=desc,
            symbols=symbols,
        )
        for sym in symbols:
            reg.symbol_to_sector[sym] = sector_name

    proxy = raw.get("market_proxy", {})
    reg.market_proxy_symbols = proxy.get("symbols", ["BTC", "ETH"])
    reg.market_proxy_weights = proxy.get("weights", [0.6, 0.4])

    return reg
