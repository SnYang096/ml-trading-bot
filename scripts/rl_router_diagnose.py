#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.rl.router_diagnostics import (  # noqa: E402
    RouterDiagnosticsConfig,
    diagnose_router_from_logs,
    write_router_diagnostics_artifacts,
)


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Router diagnostics on multi-symbol logs.")
    ap.add_argument(
        "--logs",
        required=True,
        help="Logs .csv/.parquet (symbol,timestamp,mode,ret_mean,ret_trend,...)",
    )
    ap.add_argument("--out", required=True, help="Output directory for artifacts.")
    ap.add_argument("--rolling-window", type=int, default=300)
    ap.add_argument("--rolling-min-periods", type=int, default=60)
    args = ap.parse_args()

    df = _read_any(args.logs)
    cfg = RouterDiagnosticsConfig(
        rolling_window=int(args.rolling_window),
        rolling_min_periods=int(args.rolling_min_periods),
    )
    meta, metrics, per_symbol, rolling = diagnose_router_from_logs(df, cfg=cfg)
    write_router_diagnostics_artifacts(
        out_dir=str(args.out),
        meta=meta,
        metrics=metrics,
        per_symbol=per_symbol,
        rolling=rolling,
    )
    print("✅ router diagnostics saved to:", args.out)
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
