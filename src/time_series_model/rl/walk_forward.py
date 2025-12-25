from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplitConfig:
    """
    Time-ordered split for walk-forward evaluation.

    - Split is done per symbol to avoid leakage across time.
    - train_ratio applies within each symbol's ordered timeline.
    """

    train_ratio: float = 0.7
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"


def time_ordered_split_by_symbol(
    df: pd.DataFrame,
    *,
    cfg: WalkForwardSplitConfig = WalkForwardSplitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or len(df) == 0:
        return df.copy(), df.copy()

    if cfg.symbol_col not in df.columns:
        raise ValueError(f"Missing required column: {cfg.symbol_col}")
    if cfg.timestamp_col not in df.columns:
        raise ValueError(f"Missing required column: {cfg.timestamp_col}")

    work = df.copy()
    work["_ts"] = pd.to_datetime(work[cfg.timestamp_col], errors="coerce", utc=True)
    work = work.sort_values([cfg.symbol_col, "_ts"]).reset_index(drop=True)

    train_parts = []
    test_parts = []
    for _, g in work.groupby(cfg.symbol_col, sort=False):
        n = len(g)
        if n < 2:
            continue
        n_train = max(1, min(n - 1, int(n * float(cfg.train_ratio))))
        train_parts.append(g.iloc[:n_train])
        test_parts.append(g.iloc[n_train:])

    train_df = (
        pd.concat(train_parts, axis=0).drop(columns=["_ts"]).reset_index(drop=True)
        if train_parts
        else work.iloc[:0]
    )
    test_df = (
        pd.concat(test_parts, axis=0).drop(columns=["_ts"]).reset_index(drop=True)
        if test_parts
        else work.iloc[:0]
    )
    return train_df, test_df
