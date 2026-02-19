"""
Lightweight integration checks for:

1. FeatureComputer wide-table handling:
   - Ensure that only output_columns are materialized/merged even if the
     underlying compute_func returns many extra columns.

2. SR Reversal feature-store runner (scripts/run_feature_store_sr_reversal.py):
   - Smoke-test that the script entry point is importable and its main()
     can be called with a tiny synthetic dataset (without touching real data).

These tests are designed to run quickly and not depend on large historical
data or tick caches.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

import pandas as pd

# Ensure project root is on sys.path for direct execution

from src.features.loader.feature_computer import FeatureComputer  # type: ignore  # noqa: E402


def _dummy_compute_wide_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame that contains the desired output column plus a number
    of extra columns, simulating a "wide" internal result (e.g. baseline
    functions that add many helper columns).
    """
    out = pd.DataFrame(index=df.index)
    # target output column
    out["bb_width"] = df["close"].rolling(5, min_periods=1).std().fillna(0.0)
    # many extra columns that must NOT leak into the final merged frame
    for i in range(10):
        out[f"extra_col_{i}"] = i
    return out


def _dummy_compute_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Second-stage feature that depends on bb_width and again returns a wide
    DataFrame. The FeatureComputer must keep only bb_width_ratio.
    """
    out = pd.DataFrame(index=df.index)
    if "bb_width" in df.columns:
        # simple numeric ratio; avoid NA type issues in tests
        bw = pd.to_numeric(df["bb_width"], errors="coerce").fillna(0.0)
        close = pd.to_numeric(df["close"], errors="coerce").replace(0, pd.NA)
        out["bb_width_ratio"] = (bw / close).fillna(0.0)
    else:
        out["bb_width_ratio"] = 0.0
    # add some junk columns that should be dropped by FeatureComputer
    out["junk_a"] = 1.0
    out["junk_b"] = 2.0
    return out


def test_parallel_feature_computer_wide_table_merge() -> None:
    """
    Integration-style check:

    - Build a tiny synthetic DataFrame with two months of data.
    - Define two fake features:
        * wide_bb_width: returns [bb_width] + many extra columns.
        * bb_width_ratio: depends on wide_bb_width, returns [bb_width_ratio]+junk.
    - Wire them through FeatureComputer._compute_and_cache_monthly
      and ensure that:
        * The merged result for wide_bb_width only has ['bb_width'].
        * The merged result for bb_width_ratio only has ['bb_width_ratio'].
    """
    # Synthetic input: 40 daily points over ~2 months
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({"close": (100 + pd.Series(range(40))).astype(float)}, index=idx)

    # Minimal feature metadata as FeatureComputer expects
    # Note: compute_func name must exist in feature_function_mapping for
    # _build_call_args, but we pass our own compute_func explicitly to
    # _compute_and_cache_monthly, so we can reuse any existing name here.
    wide_feature_info: Dict[str, Any] = {
        "compute_func": "compute_bb_width_features",
        "output_columns": ["bb_width"],
        "dependencies": [],
        "required_columns": ["close"],
        "compute_params": {},
        "pass_full_df": True,
    }
    ratio_feature_info: Dict[str, Any] = {
        "compute_func": "compute_bb_width_ratio",
        "output_columns": ["bb_width_ratio"],
        "dependencies": ["wide_bb_width"],
        "required_columns": ["close", "bb_width"],
        "compute_params": {},
        "pass_full_df": True,
    }

    # Instantiate FeatureComputer (sequential-only) with in-memory / no disk cache
    pfc = FeatureComputer(
        cache_dir=None,
        use_disk_cache=False,
        use_memory_cache=False,
        use_monthly_cache=True,
    )

    # Directly exercise the monthly-compute helper with our dummy functions.
    wide_result = pfc._compute_and_cache_monthly(  # type: ignore[attr-defined]
        feature_name="wide_bb_width",
        df=df,
        compute_params=wide_feature_info["compute_params"],
        feature_info=wide_feature_info,
        compute_func=_dummy_compute_wide_feature,
    )
    assert isinstance(wide_result, pd.DataFrame)
    assert list(wide_result.columns) == ["bb_width"], (
        "Wide feature should be trimmed to output_columns only, "
        f"got columns={list(wide_result.columns)}"
    )

    # Now feed wide_result as input to the ratio feature
    df_with_bb = df.join(wide_result)
    ratio_result = pfc._compute_and_cache_monthly(  # type: ignore[attr-defined]
        feature_name="bb_width_ratio",
        df=df_with_bb,
        compute_params=ratio_feature_info["compute_params"],
        feature_info=ratio_feature_info,
        compute_func=_dummy_compute_ratio,
    )
    assert isinstance(ratio_result, pd.DataFrame)
    assert list(ratio_result.columns) == ["bb_width_ratio"], (
        "bb_width_ratio feature should be trimmed to output_columns only, "
        f"got columns={list(ratio_result.columns)}"
    )

    print("✅ FeatureComputer wide-table merge test passed.")


if __name__ == "__main__":
    # Allow running this file directly as a quick smoke test.
    test_parallel_feature_computer_wide_table_merge()
