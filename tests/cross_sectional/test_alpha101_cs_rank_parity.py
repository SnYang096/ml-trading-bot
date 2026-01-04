import copy

import numpy as np
import pandas as pd

from src.cross_sectional.alpha101_cs_rank import compute_alpha101_cs_rank_panel


def _make_frames(n: int = 120) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    frames: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(["A", "B", "C", "D"]):
        base = 100 + i * 10
        t = np.arange(n, dtype=float)
        df = pd.DataFrame(
            {
                "open": base + 0.1 * t,
                "high": base + 0.1 * t + 1.0,
                "low": base + 0.1 * t - 1.0,
                "close": base + 0.1 * t + 0.2,
                "volume": 1000 + i * 50 + (t % 7),
            },
            index=idx,
        )
        frames[sym] = df
    return frames


def test_alpha101_cs_no_future_leak():
    frames = _make_frames(140)
    alpha_ids = [2, 3, 101]
    full = compute_alpha101_cs_rank_panel(frames, alpha_ids=alpha_ids)

    cutoff = pd.Timestamp("2025-01-10T00:00:00Z")
    frames2 = copy.deepcopy(frames)
    # Perturb ONLY future data for one symbol after cutoff
    m = frames2["A"].index >= cutoff
    frames2["A"].loc[m, "close"] = frames2["A"].loc[m, "close"] * 5.0
    frames2["A"].loc[m, "volume"] = frames2["A"].loc[m, "volume"] * 10.0

    full2 = compute_alpha101_cs_rank_panel(frames2, alpha_ids=alpha_ids)

    # Compare only timestamps strictly before cutoff
    ts = full.index.get_level_values("timestamp")
    mask = ts < cutoff
    left = full.loc[mask].sort_index()
    right = full2.loc[mask].sort_index()

    # Exact equality is ok for deterministic ops; allow small NaN differences
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-12, rtol=0)


def test_alpha101_cs_streaming_batch_consistency():
    frames = _make_frames(200)
    alpha_ids = [2, 3, 101]
    batch = compute_alpha101_cs_rank_panel(frames, alpha_ids=alpha_ids).sort_index()

    # Stream in two chunks with overlap warmup to emulate month stitching
    split_ts = pd.Timestamp("2025-01-20T00:00:00Z")
    overlap = pd.Timedelta(days=10)
    chunk1 = {k: v.loc[v.index < split_ts].copy() for k, v in frames.items()}
    chunk2 = {
        k: v.loc[v.index >= (split_ts - overlap)].copy() for k, v in frames.items()
    }

    out1 = compute_alpha101_cs_rank_panel(chunk1, alpha_ids=alpha_ids)
    out2 = compute_alpha101_cs_rank_panel(chunk2, alpha_ids=alpha_ids)

    # Keep only post-split from chunk2 and compare with batch
    ts2 = out2.index.get_level_values("timestamp")
    out2_post = out2.loc[ts2 >= split_ts].sort_index()

    tsb = batch.index.get_level_values("timestamp")
    batch_post = batch.loc[tsb >= split_ts].sort_index()

    pd.testing.assert_frame_equal(
        batch_post, out2_post, check_exact=False, atol=1e-10, rtol=0
    )


def test_multi_asset_normalization_alpha101_cs_rank():
    """
    Multi-asset normalization / cross-asset comparability check:
    - rank-based outputs should be bounded (0..1 or -1..0)
    - per-timestamp mean should be around the expected center
    - invariance to global price unit scaling (e.g. USD -> cents) should hold
    """
    frames = _make_frames(180)
    # pick a mix: some rank outputs + one correlation output
    alpha_ids = [2, 3, 10, 13, 101]
    panel = compute_alpha101_cs_rank_panel(frames, alpha_ids=alpha_ids).sort_index()

    # Basic boundedness checks
    col_010 = "alpha101_cs_010"
    col_013 = "alpha101_cs_013"
    col_002 = "alpha101_cs_002"
    col_003 = "alpha101_cs_003"
    col_101 = "alpha101_cs_101"

    # 010 is rank(...) => [0,1]
    s010 = panel[col_010].dropna()
    assert (s010 >= 0.0).all() and (s010 <= 1.0).all()
    # 013 is -rank(...) => [-1,0]
    s013 = panel[col_013].dropna()
    assert (s013 >= -1.0).all() and (s013 <= 0.0).all()
    # correlations bounded [-1,1]
    for c in [col_002, col_003]:
        sc = panel[c].dropna()
        assert (sc >= -1.000001).all() and (sc <= 1.000001).all()
    # 101 is ratio of differences, should be bounded roughly [-1,1] (allow small eps)
    s101 = panel[col_101].dropna()
    assert (s101 >= -1.1).all() and (s101 <= 1.1).all()

    # Per-timestamp center check for rank-based outputs
    by_ts_010 = panel[col_010].groupby(level="timestamp").mean()
    assert by_ts_010.dropna().between(0.35, 0.65).all()
    by_ts_013 = panel[col_013].groupby(level="timestamp").mean()
    assert by_ts_013.dropna().between(-0.65, -0.35).all()

    # Global price scaling invariance (USD -> cents) should hold for rank-like outputs.
    # Build a variant dataset with *non-tied* cross-sectional values to avoid rank jitter from float ties.
    idx = pd.date_range("2025-01-01", periods=180, freq="4h", tz="UTC")
    frames_var = {}
    for i, sym in enumerate(["A", "B", "C", "D"]):
        base = 100 + i * 10
        t = np.arange(len(idx), dtype=float)
        # different slopes per symbol => no ties in delta/rank
        slope = 0.1 + i * 1e-3
        df = pd.DataFrame(
            {
                "open": base + slope * t,
                "high": base + slope * t + 1.0,
                "low": base + slope * t - 1.0,
                "close": base + slope * t + 0.2,
                "volume": 1000 + i * 50 + (t % 7),
            },
            index=idx,
        )
        frames_var[sym] = df

    panel_var = compute_alpha101_cs_rank_panel(
        frames_var, alpha_ids=alpha_ids
    ).sort_index()
    frames_scaled = {}
    for sym, df in frames_var.items():
        df2 = df.copy()
        for c in ["open", "high", "low", "close"]:
            df2[c] = df2[c] * 100.0
        frames_scaled[sym] = df2
    panel_var2 = compute_alpha101_cs_rank_panel(
        frames_scaled, alpha_ids=alpha_ids
    ).sort_index()

    for col in [col_010, col_013]:
        pd.testing.assert_series_equal(
            panel_var[col], panel_var2[col], check_exact=False, atol=1e-10, rtol=0
        )
