import pandas as pd

from cross_sectional.processing import cross_sectional_rank


def test_cross_sectional_rank_is_scale_invariant():
    # Two timestamps, three assets; scaling a factor should not change cross-sectional ranks.
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True), ["A", "B", "C"]],
        names=["timestamp", "symbol"],
    )
    panel = pd.DataFrame(
        {
            "f": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "g": [3.0, 2.0, 1.0, 30.0, 20.0, 10.0],
        },
        index=idx,
    )
    ranked = cross_sectional_rank(panel.copy(), columns=["f", "g"], pct=True)

    panel2 = panel.copy()
    panel2["f"] = panel2["f"] * 1000.0 + 7.0
    # Positive scaling should preserve ranks; negative scaling flips ordering and should NOT be invariant.
    panel2["g"] = panel2["g"] * 2.0 + 5.0
    ranked2 = cross_sectional_rank(panel2.copy(), columns=["f", "g"], pct=True)

    pd.testing.assert_series_equal(ranked["f"], ranked2["f"])
    pd.testing.assert_series_equal(ranked["g"], ranked2["g"])
