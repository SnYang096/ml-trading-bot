from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class FeatureStoreIOConfig:
    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"


def _read_any(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def load_feature_store(
    path: str, *, cfg: FeatureStoreIOConfig = FeatureStoreIOConfig()
) -> pd.DataFrame:
    """
    Load precomputed features from:
      - a single .parquet/.csv file, or
      - a directory containing per-symbol files: features_<SYMBOL>.parquet / *.parquet / *.csv

    Returns a concatenated dataframe. Ensures cfg.symbol_col exists.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    files: List[Path] = []
    if p.is_dir():
        files = sorted(p.glob("features_*.parquet"))
        if not files:
            files = sorted(p.glob("*.parquet"))
        if not files:
            files = sorted(p.glob("*.csv"))
    else:
        files = [p]

    parts: List[pd.DataFrame] = []
    for f in files:
        df = _read_any(f)
        if cfg.symbol_col not in df.columns:
            df = df.copy()
            # Infer symbol from filename like features_BTCUSDT.parquet
            stem = f.stem
            if stem.startswith("features_"):
                sym = stem.replace("features_", "")
            else:
                sym = stem
            df[cfg.symbol_col] = sym
        parts.append(df)

    out = pd.concat(parts, axis=0, ignore_index=False) if parts else pd.DataFrame()
    return out
