"""L4 cross-market monthly momentum — offline helper.

Reads `config/portfolio/l4_cross_market_momentum.yaml` and an optional CSV of
daily (or monthly) **adjusted** closes for the tickers in the universe.

Implementation matches the P1 recipe in `实施文档_04` §2.3:
  - 12-month absolute return per name
  - Hold top_n by score (default 2), equal-weight

Does **not** fetch quotes (no network); bring your own CSV.

CSV format (example):
    date,SPY,QQQ,GLD,TLT
    2020-01-02,323.14,...

Usage:
    python scripts/l4_cross_market_momentum.py --csv path/to/monthly.csv
    python scripts/l4_cross_market_momentum.py   # prints spec only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def monthly_last(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to calendar month-end last observation."""
    if df.index.freq is None and not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    # 'M' = month-end (pandas); avoid 'ME' for older pandas builds
    return df.resample("M").last().dropna(how="all")


def momentum_weights(
    monthly: pd.DataFrame,
    lookback_months: int,
    top_n: int,
    all_negative_fallback_cash: bool,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return latest period target weights Series and history of scores."""
    # 12m return from month-end closes: P_t / P_{t-12} - 1
    ret = monthly.pct_change(lookback_months)
    scores = ret.iloc[-1].dropna()
    if scores.empty:
        raise ValueError("Insufficient history for momentum calculation")

    ranked = scores.sort_values(ascending=False)
    if all_negative_fallback_cash and ranked.iloc[0] < 0:
        w = pd.Series({"CASH": 1.0})
        return w, ret.tail(3)

    pick = ranked.head(top_n).index.tolist()
    w = pd.Series({k: 1.0 / len(pick) for k in pick}, dtype=float)
    # pad with zeros for reporting
    full = pd.Series(0.0, index=monthly.columns)
    for k in pick:
        full[k] = w[k]
    return full[full > 0], ret.tail(min(36, len(ret)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(ROOT / "config/portfolio/l4_cross_market_momentum.yaml"),
        help="YAML spec path",
    )
    p.add_argument(
        "--csv", default=None, help="Wide CSV with date column or DatetimeIndex"
    )
    p.add_argument(
        "--date-column",
        default=None,
        help="If CSV has no DatetimeIndex, column name for dates (e.g. date)",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    uni = cfg["universe"]["tickers"]
    lb = int(cfg["signals"]["lookback_months"])
    top_n = int(cfg["portfolio"]["top_n"])
    cash_fallback = cfg["constraints"].get("all_negative_fallback") == "cash"

    print("=== L4 cross-market momentum (spec) ===")
    print(f"config: {cfg_path}")
    print(f"universe: {uni}")
    print(f"lookback_months: {lb}  top_n: {top_n}")
    print()

    if not args.csv:
        print(
            "No --csv provided; supply daily/monthly adjusted closes to compute weights."
        )
        print("Example CSV header: date,SPY,QQQ,GLD,TLT")
        sys.exit(0)

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    if args.date_column:
        df[args.date_column] = pd.to_datetime(df[args.date_column])
        df = df.set_index(args.date_column)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # assume first column is date
        c0 = df.columns[0]
        df[c0] = pd.to_datetime(df[c0])
        df = df.set_index(c0)

    missing = [t for t in uni if t not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    sub = df[uni].astype(float)
    monthly = monthly_last(sub)
    w, hist = momentum_weights(monthly, lb, top_n, cash_fallback)

    print(f"=== Latest month-end: {monthly.index.max().date()} ===")
    print("Target weights:")
    print(w.to_string())
    print()
    print("Recent score matrix (rows=months, cols=tickers) — last few:")
    print(hist.round(4).to_string())


if __name__ == "__main__":
    main()
