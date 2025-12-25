from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.rl.shadow_eval_3action import (
    ShadowEvalConfig,
    train_and_shadow_eval_bc3_from_logs,
)


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shadow evaluation for BC(3-action) using logs with mode column."
    )
    ap.add_argument(
        "--logs",
        required=True,
        help="Path to logs .csv/.parquet with columns: symbol,timestamp,mode,head_*",
    )
    ap.add_argument(
        "--out", required=True, help="Output directory for artifacts (html/json/csv)."
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train ratio per symbol (time-ordered).",
    )
    args = ap.parse_args()

    df = _read_any(args.logs)
    cfg = ShadowEvalConfig()
    cfg = ShadowEvalConfig(
        split_cfg=cfg.split_cfg.__class__(train_ratio=float(args.train_ratio))
    )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    _, _, metrics = train_and_shadow_eval_bc3_from_logs(
        df, cfg=cfg, out_dir=str(args.out)
    )
    print("shadow metrics:", metrics)
    print("saved to:", args.out)


if __name__ == "__main__":
    main()
