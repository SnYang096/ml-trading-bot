#!/usr/bin/env python3
"""Export a rolling feature-bus window to a single parquet for monitor scripts."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_data_stream.feature_bus import FeatureBusReader, normalize_timeframe

_DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
_DEFAULT_BUS = "live/shared_feature_bus"


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _resolve_symbols(
    reader: FeatureBusReader, *, timeframe: str, explicit: Optional[str]
) -> List[str]:
    if explicit:
        return _parse_symbols(explicit)
    from_bus = list(reader.list_available_symbols(timeframe=timeframe))
    if from_bus:
        return from_bus
    return _parse_symbols(_DEFAULT_SYMBOLS)


def export_feature_bus_window(
    *,
    bus_root: Path,
    timeframe: str,
    lookback_days: int,
    output: Path,
    symbols: Optional[str] = None,
) -> Path:
    reader = FeatureBusReader(bus_root)
    tf = normalize_timeframe(timeframe)
    sym_list = _resolve_symbols(reader, timeframe=tf, explicit=symbols)
    use_all_rows = int(lookback_days) <= 0
    cutoff: Optional[pd.Timestamp] = None
    if not use_all_rows:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(lookback_days))

    frames: List[pd.DataFrame] = []
    for sym in sym_list:
        path = bus_root / "features" / tf / f"{sym}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        if df.empty or "timestamp" not in df.columns:
            continue
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if cutoff is not None:
            df = df[df["timestamp"] >= cutoff]
        if df.empty:
            continue
        df["symbol"] = sym
        frames.append(df)

    if not frames:
        window_desc = "all rows" if use_all_rows else f"last {lookback_days}d"
        raise FileNotFoundError(
            f"no bus rows ({window_desc}) under {bus_root}/features/{tf} "
            f"(symbols tried: {', '.join(sym_list)})"
        )

    out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    return output


def main() -> int:
    p = argparse.ArgumentParser(description="Export feature-bus window parquet")
    p.add_argument(
        "--bus-root",
        default=os.environ.get("MLBOT_FEATURE_BUS_ROOT", _DEFAULT_BUS),
        help="Feature bus root (env: MLBOT_FEATURE_BUS_ROOT)",
    )
    p.add_argument("--timeframe", default="120T")
    p.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Days of history to keep (<=0: export full rolling bus snapshot)",
    )
    p.add_argument(
        "--symbols",
        default="",
        help=f"Comma-separated symbols (default: bus listing or {_DEFAULT_SYMBOLS})",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output parquet path (e.g. results/monitoring/window/<ts>/features_current_7d.parquet)",
    )
    args = p.parse_args()

    bus_root = Path(args.bus_root)
    if not bus_root.is_absolute():
        bus_root = (PROJECT_ROOT / bus_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()

    try:
        out_path = export_feature_bus_window(
            bus_root=bus_root,
            timeframe=str(args.timeframe),
            lookback_days=int(args.lookback_days),
            output=output,
            symbols=str(args.symbols).strip() or None,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    df = pd.read_parquet(out_path)
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "bus_root": str(bus_root),
        "timeframe": normalize_timeframe(args.timeframe),
        "lookback_days": int(args.lookback_days),
        "rows": int(len(df)),
        "symbols": (
            sorted(df["symbol"].unique().tolist()) if "symbol" in df.columns else []
        ),
    }
    sidecar = out_path.with_suffix(".json")
    import json

    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved: {out_path} ({meta['rows']} rows, {len(meta['symbols'])} symbols)")
    print(f"meta: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
