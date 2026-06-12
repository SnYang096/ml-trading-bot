#!/usr/bin/env python3
"""Preflight: exported monitor parquet must contain TPC contract columns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# TPC monitoring contract (watchdog regime + PSI + E22 classify).
TPC_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ema_1200_position",
    "adx_50",
    "vol_persistence",
    "vol_leverage_asymmetry",
)


def required_monitor_columns(
    *,
    psi_features: Sequence[str] | None = None,
) -> List[str]:
    """Union of regime/PSI contract columns for TPC primary archetype."""
    cols = list(TPC_REQUIRED_COLUMNS)
    for name in psi_features or ():
        n = str(name).strip()
        if n and n not in cols:
            cols.append(n)
    return cols


def validate_monitor_parquet(
    parquet_path: Path,
    *,
    required_columns: Sequence[str] | None = None,
    min_non_nan_rows: int = 1,
) -> Dict[str, Any]:
    """Return report; raises ValueError with MISSING_FEATURE detail on failure."""
    path = Path(parquet_path)
    if not path.is_file():
        raise ValueError(f"MISSING_FEATURE: parquet not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"MISSING_FEATURE: parquet has zero rows: {path}")

    required = list(required_columns or TPC_REQUIRED_COLUMNS)
    missing: List[str] = []
    all_nan: List[str] = []
    for col in required:
        if col not in df.columns:
            missing.append(col)
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if int(series.notna().sum()) < int(min_non_nan_rows):
            all_nan.append(col)

    report: Dict[str, Any] = {
        "parquet": str(path),
        "n_rows": int(len(df)),
        "required_columns": required,
        "missing_columns": missing,
        "all_nan_columns": all_nan,
        "ok": not missing and not all_nan,
    }
    if missing or all_nan:
        parts: List[str] = ["MISSING_FEATURE"]
        if missing:
            parts.append(f"missing={missing}")
        if all_nan:
            parts.append(f"all_nan={all_nan}")
        raise ValueError("; ".join(parts))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet", required=True, help="Exported feature-bus window parquet"
    )
    p.add_argument(
        "--psi-features",
        default="",
        help="Comma-separated extra PSI columns (manifest watchdog_defaults)",
    )
    p.add_argument("--json-out", default="", help="Optional report JSON path")
    args = p.parse_args()

    psi = [s.strip() for s in str(args.psi_features).split(",") if s.strip()]
    try:
        report = validate_monitor_parquet(
            Path(args.parquet),
            required_columns=required_monitor_columns(psi_features=psi),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
