from __future__ import annotations

import pandas as pd

from src.features.time_series.baseline_features import BaselineFeatureEngineer


def compute_bb_width_features_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
    std_dev: int = 2,
    atr_window: int = 14,
) -> pd.DataFrame:
    """
    Compute BB width features without passing the full wide DataFrame.

    This wrapper exists to support `pass_full_df: false` + `column_mappings` in YAML,
    reducing memory usage in research/backtest runs.
    """
    # Build a minimal df (index preserved)
    df = pd.DataFrame({"close": close, "high": high, "low": low})
    out = BaselineFeatureEngineer.compute_bb_width_features(
        df, period=period, std_dev=std_dev, atr_window=atr_window
    )
    # Return only the columns this feature is supposed to output; the loader will
    # also trim by output_columns, but keeping it tight here avoids surprises.
    keep = ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_width_normalized"]
    return out[keep] if all(c in out.columns for c in keep) else out
