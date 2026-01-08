import pandas as pd

from src.cross_sectional.feature_store_builder import (
    CSFeatureStoreBuildConfig,
    build_feature_store_for_symbols,
)


def test_build_store_alpha101_cs_rank_smoke(tmp_path, monkeypatch):
    # Mock load_raw_data so build_feature_store_for_symbols doesn't hit disk.
    from src import data_tools as _dt  # noqa: F401
    import src.data_tools.data_utils as du
    import src.cross_sectional.feature_store_builder as fsb

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

    # The default RSI implementation depends on TA-Lib in some environments.
    # For this unit test we patch the registry lookup to a pure-pandas stub so we can
    # verify mixed-mode store writing (alpha101_cs_* + non-alpha DAG features).
    _orig_get = fsb.get_feature_func

    def _fake_get_feature_func(name: str):
        if name == "compute_rsi_from_series":

            def _stub_rsi_from_series(
                close: pd.Series, period: int = 14
            ) -> pd.DataFrame:
                s = pd.to_numeric(close, errors="coerce").astype(float)
                # cheap deterministic signal; not a real RSI
                out = (
                    s.pct_change(fill_method=None).fillna(0.0).rename("rsi").to_frame()
                )
                return out

            return _stub_rsi_from_series
        return _orig_get(name)

    monkeypatch.setattr(fsb, "get_feature_func", _fake_get_feature_func)

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
        # Mixed mode should be supported: alpha101_cs_* + normal OHLCV-only factors via feature_dependencies DAG.
        desired_output_cols=[
            "alpha101_cs_001",
            "alpha101_cs_002",
            "alpha101_cs_101",
            "rsi",
        ],
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
    # Should also contain at least one non-alpha feature from feature_dependencies.yaml
    assert "rsi" in df.columns
