import numpy as np
import pandas as pd

from cross_sectional.governance import (
    flag_unstable_factors,
    governance_report,
    holm_bonferroni,
    rolling_ic,
)


def _make_panel() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
    symbols = ["BTC", "ETH", "SOL"]
    index = pd.MultiIndex.from_product([timestamps, symbols],
                                       names=["timestamp", "symbol"])
    rng = np.random.default_rng(42)
    panel = pd.DataFrame(
        {
            "factor_a": rng.normal(0, 1, len(index)),
            "factor_b": rng.normal(0, 1, len(index)),
            "target": rng.normal(0, 1, len(index)),
            "dollar_volume": rng.uniform(1_000_000, 2_000_000, len(index)),
        },
        index=index,
    )
    return panel


def test_governance_report_and_flags():
    panel = _make_panel()
    report = governance_report(panel,
                               factor_cols=["factor_a", "factor_b"],
                               target_col="target",
                               window=5)
    assert {"mean_ic", "p_value", "p_value_holm"}.issubset(report.columns)
    unstable = flag_unstable_factors(report,
                                     min_ic=-1.0,
                                     min_ic_ir=-1.0,
                                     max_pvalue=1.0)
    assert isinstance(unstable, list)


def test_holm_bonferroni_adjustment():
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.05})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_rolling_ic_computation():
    panel = _make_panel()
    rolling = rolling_ic(panel,
                         factor_cols=["factor_a"],
                         target_col="target",
                         window=12)
    assert "factor_a" in rolling
    assert isinstance(rolling["factor_a"], pd.Series)
