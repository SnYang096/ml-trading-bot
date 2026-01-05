import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cs_nautilus_backtest_smoke(tmp_path: Path):
    # Minimal panel: 2 timestamps, 4 symbols, 1 feature and close
    ts = [pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-01-03T00:00:00Z")]
    syms = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for t in ts:
        for i, s in enumerate(syms):
            rows.append(
                {
                    "timestamp": t.isoformat(),
                    "symbol": s,
                    "close": 100.0 + float(i),
                    "f1": float(i),
                }
            )
    panel = pd.DataFrame(rows)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)

    # Run factor-combo mode (does not require a model file)
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "src/cross_sectional/scripts/nautilus_backtest.py",
        "--panel",
        str(panel_path),
        "--output-dir",
        str(out_dir),
        "--signal",
        "factor_combo",
        "--feature-cols",
        "f1",
        "--mode",
        "market_neutral",
        "--holding",
        "1",
        "--lag",
        "0",
        "--topk",
        "1",
        "--bottomk",
        "1",
        "--min-assets",
        "2",
        "--html",
        "report.html",
        "--max-trades",
        "50",
    ]
    subprocess.check_call(cmd)

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "equity.csv").exists()
    assert (out_dir / "rebalance_log.csv").exists()
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "report.html").exists()
