from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_tools import tick_loader


def _write_month_ticks(path: Path, start: str, n: int) -> None:
    ts = pd.date_range(start, periods=n, freq="min")
    # Alternate sides to avoid degenerate buckets
    side = np.where(np.arange(n) % 2 == 0, 1, -1)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "price": 100.0,
            "volume": 1.0,
            "side": side,
        }
    )
    df.to_parquet(path, index=False)


def test_vpin_monthly_cache_uses_prev_bucket_state(tmp_path: Path) -> None:
    """
    Validate Plan-A style incremental caching behavior:
    - Jan month is cached with a final_state (standard cache can be state-only)
    - Feb month cache key should incorporate prev_bucket_state (state cache),
      so Feb can be recomputed consistently across month boundaries.
    """
    cache_dir = tmp_path / "monthly_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jan = tmp_path / "BTCUSDT_2025-01.parquet"
    feb = tmp_path / "BTCUSDT_2025-02.parquet"
    _write_month_ticks(jan, "2025-01-01 00:00:00", n=50)
    _write_month_ticks(feb, "2025-02-01 00:00:00", n=50)

    # Fixed bucket volume small so we get some buckets and a non-empty final_state
    bucket_volume = 7.0

    # First, compute Jan buckets directly and save a "standard" cache entry (state-only).
    jan_buckets, jan_state = tick_loader._compute_vpin_buckets_for_month(  # type: ignore[attr-defined]
        jan, bucket_volume=bucket_volume, bucket_volume_usd=None, initial_state=None
    )
    assert isinstance(jan_state, dict)
    jan_key = tick_loader._get_monthly_vpin_cache_key(  # type: ignore[attr-defined]
        str(jan),
        bucket_volume=bucket_volume,
        bucket_volume_usd=None,
        prev_bucket_state=None,
    )
    tick_loader._save_monthly_vpin_cache(  # type: ignore[attr-defined]
        cache_dir,
        jan_key,
        buckets=jan_buckets,
        final_state=jan_state,
        save_buckets=False,
    )

    # Now run the public API over Feb range; it should load Jan state and create a state-aware cache key for Feb.
    out = tick_loader.compute_vpin_from_cached_ticks(
        cache_files=[str(jan), str(feb)],
        start_ts="2025-02-01 00:00:00",
        end_ts="2025-02-02 00:00:00",
        bucket_volume=bucket_volume,
        n_buckets=10,
        adaptive=False,
        lookback_minutes=0,
        monthly_cache_dir=str(cache_dir),
        bucket_volume_usd=None,
    )
    assert isinstance(out, pd.Series)

    # The Feb state-aware cache key should exist (it includes prev_bucket_state from Jan).
    feb_state_key = tick_loader._get_monthly_vpin_cache_key(  # type: ignore[attr-defined]
        str(feb),
        bucket_volume=bucket_volume,
        bucket_volume_usd=None,
        prev_bucket_state=jan_state,
    )
    assert (cache_dir / f"{feb_state_key}.pkl").exists()
