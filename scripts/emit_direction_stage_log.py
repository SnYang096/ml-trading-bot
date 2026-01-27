#!/usr/bin/env python3
"""
Emit a minimal execution stage log entry to validate direction fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.diagnostics.execution_log import (
    ExecutionStageLogWriter,
    build_decision_id,
    build_stage_record,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)
from src.time_series_model.live.direction_resolver import resolve_direction


def _mock_trend_bars() -> list[dict[str, Any]]:
    bars = []
    price = 100.0
    for _ in range(20):
        price += 0.5
        bars.append({"high": price + 0.2, "low": price - 0.2, "close": price})
    return bars


def _mock_failed_breakout_bars() -> list[dict[str, Any]]:
    bars = []
    price = 100.0
    for _ in range(18):
        price += 0.5
        bars.append({"high": price + 0.2, "low": price - 0.2, "close": price})
    hh = max(b["high"] for b in bars)
    bars.append({"high": hh + 1.0, "low": hh - 0.2, "close": hh - 0.1})
    return bars


def _mock_feats() -> Dict[str, Any]:
    return {"atr": 1.0, "price_dir_consistency_pct": 0.7, "sweep_side": "BUY"}


def main() -> None:
    base_dir = Path("results/live_logs")
    writer = ExecutionStageLogWriter(base_dir=base_dir, stage="execution")
    arches = load_execution_archetypes_registry(
        Path("config/nnmultihead/execution_archetypes.yaml")
    )
    now_ns = int(time.time() * 1e9)

    for name, arch in arches.items():
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
        decision_id = build_decision_id(
            strategy_name="direction_smoke",
            symbol="BTCUSDT",
            decision_ts_ns=now_ns,
        )
        record = build_stage_record(
            stage="execution",
            decision_id=decision_id,
            decision_ts_ns=now_ns,
            source="smoke",
            symbol="BTCUSDT",
            timeframe="4H",
            strategy_name="direction_smoke",
            instrument_id="BTCUSDT",
            data={
                "intent": True,
                "submit_order": False,
                "archetype": str(name),
                "side": str(decision.side),
                "direction_source": str(decision.source),
                "direction_method": str(decision.method),
                "direction_reason": str(decision.reason),
            },
        )
        writer.write(record, decision_ts_ns=now_ns)
        now_ns += 1_000_000  # ensure unique decision id order

    print(f"Wrote execution stage logs to {base_dir}/execution")


if __name__ == "__main__":
    main()
