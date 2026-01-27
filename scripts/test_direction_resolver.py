#!/usr/bin/env python3
"""
Lightweight smoke check for direction resolver policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from src.time_series_model.live.direction_resolver import resolve_direction


def _mock_trend_bars() -> List[Dict[str, Any]]:
    bars = []
    price = 100.0
    for _ in range(20):
        price += 0.5
        bars.append(
            {
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price,
            }
        )
    return bars


def _mock_failed_breakout_bars() -> List[Dict[str, Any]]:
    bars = []
    price = 100.0
    for _ in range(18):
        price += 0.5
        bars.append(
            {
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price,
            }
        )
    # Last bar: sweep above HH then close back below HH
    hh = max(b["high"] for b in bars)
    bars.append(
        {
            "high": hh + 1.0,
            "low": hh - 0.2,
            "close": hh - 0.1,
        }
    )
    return bars


def _mock_feats() -> Dict[str, Any]:
    return {
        "atr": 1.0,
        "price_dir_consistency_pct": 0.7,
    }


def main() -> None:
    registry = load_execution_archetypes_registry(
        Path("config/nnmultihead/execution_archetypes.yaml")
    )
    for name, arch in registry.items():
        policy = dict(getattr(arch, "direction_policy", None) or {})
        method = str((policy.get("structure_direction") or {}).get("method") or "")
        feats = _mock_feats()
        if method == "failed_breakout":
            bars = _mock_failed_breakout_bars()
        else:
            bars = _mock_trend_bars()
        if (
            method == "reverse_of"
            and (policy.get("structure_direction") or {}).get("base") == "sweep_side"
        ):
            feats["sweep_side"] = "BUY"
        decision = resolve_direction(
            archetype_name=name,
            policy=policy,
            feats=feats,
            bars=bars,
        )
        print(f"{name}: ok={decision.ok} side={decision.side} method={decision.method}")


if __name__ == "__main__":
    main()
