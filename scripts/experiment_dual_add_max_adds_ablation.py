"""Ablation: dual_add_trend — sweep max_adds_per_side with identical signal & exit semantics.

Runs ``scripts/diagnose_dual_add_trend.py`` once per grid value with ``--no-maps`` for speed,
then aggregates segment-level and portfolio metrics.

Example (trend-only open + basket TP + regime_only; align with stress-test notes)::

    python scripts/experiment_dual_add_max_adds_ablation.py \\
      --out-root results/dual_add_ablation_max_adds_2024q1 \\
      --max-adds-grid 0,1,2,3 \\
      -- \\
      --config config/strategies/dual_add_trend/research/calibrate_roll.default.yaml \\
      --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
      --start 2024-01-01 --end 2024-03-31 \\
      --timeframe 2h --execution-timeframe 1min \\
      --take-profit-mode basket --no-initial-hedge \\
      --risk-stop-mode regime_only --fee-bps 8

Forwarded arguments must appear after ``--`` and are identical across grid runs except
``--max-adds-per-side`` / ``--out-dir``, which this script injects.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGNOSE = PROJECT_ROOT / "scripts" / "diagnose_dual_add_trend.py"


def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return list(argv[:i]), list(argv[i + 1 :])
    return list(argv), []


def _portfolio_dd_from_segments(segments: pd.DataFrame) -> float:
    """Max drawdown on cumulative sum of segment pnl_per_capital ordered by segment end."""
    if segments.empty or "end" not in segments.columns:
        return 0.0
    df = segments[["end", "pnl_per_capital"]].copy()
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df = df.dropna(subset=["end"]).sort_values("end")
    if df.empty:
        return 0.0
    cum = df["pnl_per_capital"].cumsum().to_numpy(dtype=float)
    peak = np.maximum.accumulate(cum)
    trough = cum - peak
    return float(trough.min()) if trough.size else 0.0


def _segment_quantiles(segments: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if segments.empty or "pnl_per_capital" not in segments.columns:
        return {
            "mean_segment_pnl": float("nan"),
            "median_segment_pnl": float("nan"),
            "segment_pnl_p05": float("nan"),
            "segment_pnl_p25": float("nan"),
            "segment_pnl_p75": float("nan"),
            "segment_pnl_p95": float("nan"),
        }
    s = segments["pnl_per_capital"].astype(float)
    out["mean_segment_pnl"] = float(s.mean())
    out["median_segment_pnl"] = float(s.median())
    for q, name in [(5, "p05"), (25, "p25"), (75, "p75"), (95, "p95")]:
        out[f"segment_pnl_{name}"] = float(s.quantile(q / 100.0))
    return out


def _forward_blacklist(forward: List[str]) -> List[str]:
    blocked = {"--max-adds-per-side", "--out-dir"}
    out: List[str] = []
    for tok in forward:
        key = tok.split("=", 1)[0] if "=" in tok else tok
        if key in blocked:
            raise ValueError(
                f"Do not pass {key} in forwarded args; the ablation script sets it."
            )
        out.append(tok)
    return out


def main() -> None:
    ours, forward_all = _split_argv(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-adds-grid",
        default="0,1,2,3",
        help="Comma-separated integers for --max-adds-per-side.",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Directory containing one subdirectory per grid value.",
    )
    ap.add_argument(
        "--diagnose-script",
        type=Path,
        default=DIAGNOSE,
        help="Path to diagnose_dual_add_trend.py.",
    )
    ap.add_argument(
        "--keep-maps",
        action="store_true",
        help="Forward --no-maps is omitted so diagnose emits maps (slow).",
    )
    args = ap.parse_args(ours)

    grid = [int(x.strip()) for x in str(args.max_adds_grid).split(",") if x.strip()]
    if not grid:
        raise SystemExit("empty --max-adds-grid")

    forward = _forward_blacklist(forward_all)
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for m in grid:
        run_dir = args.out_root / f"max_adds_{m}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd: List[str] = [
            sys.executable,
            str(args.diagnose_script),
            "--max-adds-per-side",
            str(m),
            "--out-dir",
            str(run_dir),
        ]
        if not args.keep_maps:
            cmd.append("--no-maps")
        cmd.extend(forward)

        print("\n=== Running ===")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

        summary_path = run_dir / "summary.csv"
        seg_path = run_dir / "dual_add_segments.csv"
        cfg_path = run_dir / "config.json"

        summary_row: dict[str, object] = {"max_adds_per_side": m}
        if summary_path.exists():
            sdf = pd.read_csv(summary_path)
            if not sdf.empty:
                for c in sdf.columns:
                    summary_row[c] = sdf.iloc[0][c]

        segments = pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame()
        summary_row.update(_segment_quantiles(segments))
        summary_row["portfolio_cum_dd_per_capital"] = _portfolio_dd_from_segments(
            segments
        )

        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            summary_row["forward_initial_hedge"] = cfg.get("initial_hedge")
            summary_row["forward_take_profit_mode"] = cfg.get("take_profit_mode")
            summary_row["forward_risk_stop_mode"] = cfg.get("risk_stop_mode")

        rows.append(summary_row)

    agg = pd.DataFrame(rows)
    agg_path = args.out_root / "ablation_summary.csv"
    agg.to_csv(agg_path, index=False)
    meta = {
        "max_adds_grid": grid,
        "out_root": str(args.out_root),
        "forward_argv": forward,
    }
    (args.out_root / "ablation_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\n=== Ablation aggregate ===")
    print(agg.to_string(index=False))
    print(f"\nSaved -> {agg_path}")


if __name__ == "__main__":
    main()
