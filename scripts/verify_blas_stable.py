#!/usr/bin/env python3
"""Quick smoke test for macOS OpenBLAS / NumPy / parquet stability."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_line(name: str) -> str:
    return f"  {name}={os.environ.get(name, '(unset)')}"


def main() -> int:
    print("=== ml-trading-bot BLAS stability check ===\n")
    print("Thread env:")
    for k in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_MAIN_FREE",
    ):
        print(_env_line(k))

    import numpy as np

    print(f"\nNumPy {np.__version__}")
    np.show_config()

    # Matmul (exercises OpenBLAS gemm)
    a = np.random.default_rng(0).standard_normal((512, 512))
    b = a @ a.T
    print(f"matmul ok: shape={b.shape} trace={float(b.trace()):.4f}")

    # JSON load (same stack as funnel JSON parse)
    sample = ROOT / "results/event_backtest/srb_2024_2026/event_backtest_srb.json"
    if sample.is_file():
        data = json.loads(sample.read_text())
        print(f"json ok: funnel keys={len(data.get('funnel', {}))}")
    else:
        print("json skip: no sample backtest json")

    # Parquet read (tick schema)
    pq = ROOT / "data/parquet_data/BTCUSDT_2024-06.parquet"
    if pq.is_file():
        import pandas as pd

        df = pd.read_parquet(pq, columns=["timestamp", "price", "volume", "side"])
        print(f"parquet ok: rows={len(df)} cols={list(df.columns)}")
    else:
        print("parquet skip: no sample parquet")

    print("\nOK — if this finishes without 'Python 意外退出', BLAS env is working.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)
