"""Chop grid ts_quantile path: strict FeatureStore-only feature merge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.diagnose_crf_edge import StudyConfig
from scripts.diagnose_chop_grid import (
    GridConfig,
    _materialize_chop_grid_from_store_columns,
    build_features,
    merge_chop_grid_yaml,
    resolve_optional_repo_path,
    should_compute_semantic_chop_ts_q,
)
from src.config.multileg_config import load_multileg_effective_config
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ts_quantile_requires_feature_store_config() -> None:
    cfg = GridConfig(chop_signal="ts_quantile")
    bars = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.05],
            "low": [0.95],
            "close": [1.0],
            "volume": [1000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-06-01", tz="UTC")]),
    )
    with pytest.raises(ValueError, match="requires FeatureStore settings"):
        build_features("BTCUSDT", bars, cfg, bars_timeframe="2h")


def test_ts_quantile_maps_parquet_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parquet-merge path delegates to StrategyFeatureLoader with strict=True."""

    captured: dict[str, object] = {}

    def _fake_load(
        self,
        df: pd.DataFrame,
        requested,
        fit: bool = True,
        *,
        feature_store_strict: bool = False,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured["strict"] = feature_store_strict
        captured["kwargs"] = kwargs
        out = df.copy()
        out["atr"] = 0.12
        out["bpc_semantic_chop"] = 0.55
        out["bpc_semantic_chop_ts_q"] = 0.61
        w = 120
        suf = f"_{w}"
        out[f"box_hi{suf}"] = 51000.0
        out[f"box_lo{suf}"] = 49000.0
        out[f"box_width_pct{suf}"] = 0.08
        out[f"box_pos{suf}"] = 0.5
        out[f"box_stability{suf}"] = 0.9
        out[f"box_touches_hi{suf}"] = 7.0
        out[f"box_touches_lo{suf}"] = 8.0
        return out

    monkeypatch.setattr(
        StrategyFeatureLoader,
        "load_features_from_requested",
        _fake_load,
    )

    cfg = GridConfig(
        chop_signal="ts_quantile",
        box_window=120,
        feature_store_dir="/tmp/fs-does-not-need-exist-with-fake-load",
        feature_store_layer="test_layer",
    )
    bars = pd.DataFrame(
        {
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50000.0],
            "volume": [1e6],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-06-01")]),
    )
    merged = build_features("BTCUSDT", bars, cfg, bars_timeframe="2h")

    assert captured.get("strict") is True
    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("feature_store_symbol") == "BTCUSDT"

    assert float(merged["semantic_chop"].iloc[0]) == pytest.approx(0.55)
    assert float(merged["semantic_chop_ts_q"].iloc[0]) == pytest.approx(0.61)
    assert float(merged["atr14"].iloc[0]) == pytest.approx(0.12)
    assert bool(merged["box_prefilter"].iloc[0])


def test_raw_chop_path_uses_feature_store_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw chop signal should still load FeatureStore columns for prefilter rules."""

    captured: dict[str, object] = {}

    def _fake_load(
        self,
        df: pd.DataFrame,
        requested,
        fit: bool = True,
        *,
        feature_store_strict: bool = False,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured["strict"] = feature_store_strict
        captured["requested"] = list(requested or [])
        captured["kwargs"] = kwargs
        out = df.copy()
        out["atr"] = 0.12
        out["bpc_semantic_chop"] = 0.55
        out["wpt_compression_score"] = 0.72
        out["hurst_price_rolling"] = 0.38
        out["hilbert_price_env"] = 0.50
        w = 120
        suf = f"_{w}"
        out[f"box_hi{suf}"] = 51000.0
        out[f"box_lo{suf}"] = 49000.0
        out[f"box_width_pct{suf}"] = 0.08
        out[f"box_pos{suf}"] = 0.5
        out[f"box_stability{suf}"] = 0.9
        out[f"box_touches_hi{suf}"] = 7.0
        out[f"box_touches_lo{suf}"] = 8.0
        return out

    monkeypatch.setattr(
        StrategyFeatureLoader,
        "load_features_from_requested",
        _fake_load,
    )

    cfg = GridConfig(
        chop_signal="raw",
        box_window=120,
        feature_store_dir="/tmp/fs-does-not-need-exist-with-fake-load",
        feature_store_layer="test_layer",
    )
    bars = pd.DataFrame(
        {
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50000.0],
            "volume": [1e6],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-06-01")]),
    )
    merged = build_features("BTCUSDT", bars, cfg, bars_timeframe="2h")

    assert captured.get("strict") is True
    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("feature_store_symbol") == "BTCUSDT"
    assert "wpt_scene_semantic_scores_f" in captured["requested"]

    assert float(merged["semantic_chop"].iloc[0]) == pytest.approx(0.55)
    assert "semantic_chop_ts_q" not in merged.columns
    assert float(merged["wpt_compression_score"].iloc[0]) == pytest.approx(0.72)
    assert float(merged["hurst_price_rolling"].iloc[0]) == pytest.approx(0.38)
    assert float(merged["hilbert_price_env"].iloc[0]) == pytest.approx(0.50)


def test_feature_store_strict_empty_month_range_raises(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec

    root = tmp_path / "empty_fs"
    root.mkdir(parents=False, exist_ok=True)
    store = FeatureStore(root)
    sym = "ZZTESTUSDT"
    spec = FeatureStoreSpec(layer="L", symbol=sym, timeframe="2h")
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-15")])
    df = pd.DataFrame({"atr": [1.0], "dummy": [0.0]}, index=idx)
    store.write_month(spec, "2024-01", df, overwrite=True)

    loader = StrategyFeatureLoader(
        feature_deps_path=str(REPO_ROOT / "config/feature_dependencies.yaml"),
        use_disk_cache=False,
        use_memory_cache=False,
        verbose=False,
    )
    bars = pd.DataFrame(
        {"close": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-03-01")]),
    )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        loader.load_features_from_requested(
            bars,
            ["atr_f"],
            fit=False,
            feature_store_dir=str(root),
            feature_store_layer="L",
            feature_store_symbol=sym,
            feature_store_timeframe="2h",
            feature_store_strict=True,
        )


def test_feature_store_strict_accepts_requested_output_columns(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec

    root = tmp_path / "fs_output_cols"
    root.mkdir(parents=False, exist_ok=True)
    store = FeatureStore(root)
    sym = "ZZTESTUSDT"
    spec = FeatureStoreSpec(layer="L", symbol=sym, timeframe="2h")
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-15")])
    store.write_month(
        spec,
        "2024-01",
        pd.DataFrame(
            {
                "box_hi_120": [2.0],
                "box_lo_120": [1.0],
            },
            index=idx,
        ),
        overwrite=True,
    )

    loader = StrategyFeatureLoader(
        feature_deps_path=str(REPO_ROOT / "config/feature_dependencies.yaml"),
        use_disk_cache=False,
        use_memory_cache=False,
        verbose=False,
    )
    bars = pd.DataFrame(
        {"close": [1.5]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-15")]),
    )
    out = loader.load_features_from_requested(
        bars,
        ["box_hi_120", "box_lo_120"],
        fit=False,
        feature_store_dir=str(root),
        feature_store_layer="L",
        feature_store_symbol=sym,
        feature_store_timeframe="2h",
        feature_store_strict=True,
    )

    assert out["box_hi_120"].iloc[0] == pytest.approx(2.0)
    assert out["box_lo_120"].iloc[0] == pytest.approx(1.0)


def test_resolve_optional_repo_path_relative_inside_repo():
    deps = resolve_optional_repo_path("config/feature_dependencies.yaml")
    assert deps is not None
    path = Path(deps)
    assert path.is_absolute()
    assert path.name == "feature_dependencies.yaml"


def test_resolve_optional_repo_path_none():
    assert resolve_optional_repo_path(None) is None
    assert resolve_optional_repo_path("") is None


def test_should_compute_semantic_chop_ts_q_ts_quantile_default():
    cfg = GridConfig(chop_signal="ts_quantile", compute_semantic_chop_ts_q=None)
    assert should_compute_semantic_chop_ts_q(cfg)


def test_materialize_chop_grid_from_store_raises_on_missing_box_suffix():
    study = StudyConfig(
        box_window=240,
        chop_min=0.4,
        stability_min=0.85,
        width_min=0.04,
        width_max=0.30,
        touches_min=5,
    )
    cfg = GridConfig(box_window=240)
    df = pd.DataFrame(
        {
            "bpc_semantic_chop": [0.5],
            "bpc_semantic_chop_ts_q": [0.55],
            "atr": [0.02],
            "close": [50000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-06-01")]),
    )
    with pytest.raises(KeyError, match="box_structure columns"):
        _materialize_chop_grid_from_store_columns(df, cfg, study)


def test_materialize_chop_grid_from_store_prefilter_respects_stability():
    w = 120
    suf = f"_{w}"
    study = StudyConfig(
        box_window=w,
        chop_min=0.4,
        stability_min=0.85,
        width_min=0.04,
        width_max=0.30,
        touches_min=5,
    )
    cfg = GridConfig(box_window=w)
    df = pd.DataFrame(
        {
            "bpc_semantic_chop": [0.5],
            "bpc_semantic_chop_ts_q": [0.55],
            "atr14": [0.03],
            f"box_hi{suf}": [51000.0],
            f"box_lo{suf}": [49000.0],
            f"box_width_pct{suf}": [0.06],
            f"box_pos{suf}": [0.5],
            f"box_stability{suf}": [0.84],
            f"box_touches_hi{suf}": [7.0],
            f"box_touches_lo{suf}": [7.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-06-01")]),
    )
    merged = _materialize_chop_grid_from_store_columns(df, cfg, study)
    assert not bool(merged["box_prefilter"].iloc[0])


def test_repo_chop_grid_research_turbo_yaml_multileg_no_live_section():
    prof = (
        REPO_ROOT / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
    )
    assert prof.exists()
    eff = load_multileg_effective_config(
        config_dir=REPO_ROOT / "config/strategies/chop_grid",
        strategy_type="grid",
        profile_path=prof,
    )
    assert "live" not in eff


def test_merge_chop_grid_yaml_repo_profile_grid_backtest_store_baselines():
    merged = merge_chop_grid_yaml(
        REPO_ROOT / "config/strategies/chop_grid/research/calibrate_roll.default.yaml",
    )
    assert Path(str(merged.get("feature_store_dir"))).name == "feature_store"
    assert merged.get("feature_store_timeframe") == "120T"


def test_repo_chop_grid_turbo_grid_backtest_rolling_aligned_costs_and_maps():
    import yaml

    raw = yaml.safe_load(
        (
            REPO_ROOT
            / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
        ).read_text(encoding="utf-8")
    )
    grid_bt = raw["grid_backtest"]
    assert "output_dir" not in grid_bt
    assert "map_months" not in grid_bt
    assert "continuous_map_months" not in grid_bt
    assert grid_bt["costs"] == {
        "fee_bps": 20.0,
        "maker_fee_bps": 20.0,
        "taker_fee_bps": 20.0,
        "forced_exit_slippage_bps": 20.0,
        "funding_cost_bps_per_8h": 20.0,
    }


def test_live_highcap_chop_grid_uses_strategy_package_layers():
    pkg = REPO_ROOT / "live/highcap/config/strategies/chop_grid"
    assert not (pkg / "grid.yaml").exists()
    assert not (pkg / "research/calibrate_roll.default.yaml").exists()
    assert not (pkg / "features.yaml").exists()
    assert (pkg / "archetypes/prefilter.yaml").exists()
    assert (pkg / "archetypes/execution.yaml").exists()

    eff = load_multileg_effective_config(
        config_dir=pkg,
        strategy_type="grid",
    )
    assert eff["regime"]["entry_feature"] == "bpc_semantic_chop"
    assert eff["inventory"]["spacing"]["atr_mult"] == pytest.approx(1.00)
    assert eff["risk"]["max_open_levels_total"] == 4


def test_merge_chop_grid_yaml_loads_live_highcap_package_dir():
    merged = merge_chop_grid_yaml(
        REPO_ROOT / "live/highcap/config/strategies/chop_grid",
    )
    assert merged["chop_signal"] == "raw"
    assert merged["grid_atr_mult"] == pytest.approx(1.00)
    assert merged["fee_bps"] == pytest.approx(4.0)


def test_live_highcap_tpc_deploy_package_contains_only_meta_and_archetypes():
    pkg = REPO_ROOT / "live/highcap/config/strategies/tpc"
    allowed_roots = {"meta.yaml", "archetypes"}
    assert {p.name for p in pkg.iterdir()} == allowed_roots
