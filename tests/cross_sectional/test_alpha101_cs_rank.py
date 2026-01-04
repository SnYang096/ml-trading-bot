import pandas as pd

from src.cross_sectional.alpha101_cs_rank import compute_alpha101_cs_rank_panel


def _make_frames():
    idx = pd.date_range("2025-01-01", periods=40, freq="4h", tz="UTC")
    frames = {}
    for i, sym in enumerate(["A", "B", "C"]):
        base = 100 + i * 10
        df = pd.DataFrame(
            {
                "open": base + (pd.Series(range(len(idx))) * 0.1).values,
                "high": base + (pd.Series(range(len(idx))) * 0.1).values + 1.0,
                "low": base + (pd.Series(range(len(idx))) * 0.1).values - 1.0,
                "close": base + (pd.Series(range(len(idx))) * 0.1).values + 0.2,
                "volume": 1000 + i * 100 + (pd.Series(range(len(idx))) % 5).values,
            },
            index=idx,
        )
        frames[sym] = df
    return frames


def test_compute_alpha101_cs_rank_panel_shape_and_columns():
    frames = _make_frames()
    panel = compute_alpha101_cs_rank_panel(frames, alpha_ids=[1, 2, 3, 101])
    assert isinstance(panel, pd.DataFrame)
    assert isinstance(panel.index, pd.MultiIndex)
    assert set(["timestamp", "symbol"]).issubset(set(panel.index.names))
    # Columns are prefixed
    assert "alpha101_cs_001" in panel.columns
    assert "alpha101_cs_002" in panel.columns
    assert "alpha101_cs_003" in panel.columns
    assert "alpha101_cs_101" in panel.columns
    # At least some rows exist
    assert len(panel) > 0
