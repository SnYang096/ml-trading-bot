import pandas as pd

from src.cross_sectional.feature_store_builder import (
    CSFeatureStoreBuildConfig,
    build_feature_store_for_symbols,
)


def test_build_store_alpha101_cs_rank_smoke(tmp_path, monkeypatch):
    # Mock load_raw_data so build_feature_store_for_symbols doesn't hit disk.
    from src import data_tools as _dt  # noqa: F401
    import src.data_tools.data_utils as du

    idx = pd.date_range("2025-01-01", periods=20, freq="4h", tz="UTC")

    def _fake_load_raw_data(
        *, data_path, symbol, start_date=None, end_date=None, timeframe="240T"
    ):
        base = 100.0 if symbol == "A" else (110.0 if symbol == "B" else 120.0)
        return pd.DataFrame(
            {
                "open": base,
                "high": base + 1,
                "low": base - 1,
                "close": base + 0.2,
                "volume": 1000.0,
            },
            index=idx,
        )

    monkeypatch.setattr(du, "load_raw_data", _fake_load_raw_data)

    cfg = CSFeatureStoreBuildConfig(
        data_path="ignored",
        features_store_root=str(tmp_path / "feature_store"),
        features_store_layer="test_layer",
        timeframe="240T",
        start_date="2025-01-01",
        end_date="2025-01-31",
        warmup_bars=5,
        include_ohlcv=True,
        overwrite=True,
    )

    layer = build_feature_store_for_symbols(
        symbols=["A", "B", "C"],
        desired_output_cols=["alpha101_cs_001", "alpha101_cs_002", "alpha101_cs_101"],
        feature_deps_path="config/feature_dependencies.yaml",
        cfg=cfg,
    )
    assert layer == "test_layer"

    # Verify files exist for at least one month
    p = tmp_path / "feature_store" / "test_layer" / "A" / "240T" / "2025-01.parquet"
    assert p.exists()
    df = pd.read_parquet(p)
    # Should contain alpha columns
    assert "alpha101_cs_001" in df.columns
    assert "alpha101_cs_002" in df.columns
    assert "alpha101_cs_101" in df.columns
