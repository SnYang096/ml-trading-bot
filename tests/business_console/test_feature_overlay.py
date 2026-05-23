import pandas as pd
import pytest

from mlbot_console.services.feature_overlay import (
    list_feature_columns,
    load_feature_overlay,
    load_feature_overlays,
)


def test_load_weekly_ema_overlay(bus_root):
    data = load_feature_overlay(
        bus_root,
        "ETHUSDT",
        "2h",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-02", tz="UTC"),
    )
    assert data["available"] is True
    assert len(data["points"]) >= 1
    assert data["latest"] is not None


def test_list_feature_columns(bus_root):
    meta = list_feature_columns(bus_root, "ETHUSDT", "2h")
    assert meta["available"] is True
    assert "weekly_ema_200_position" in meta["columns"]
    assert "regime_score" in meta["columns"]


def test_resolve_weekly_ema_alias_column(tmp_path):
    import pandas as pd

    feat_dir = tmp_path / "features" / "primary"
    feat_dir.mkdir(parents=True)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = [
        {
            "timestamp": start + pd.Timedelta(hours=i * 2),
            "weekly_ema_200_position_f": -0.1 + i * 0.01,
        }
        for i in range(5)
    ]
    pd.DataFrame(rows).to_parquet(feat_dir / "ETHUSDT.parquet", index=False)

    overlays = load_feature_overlays(
        tmp_path,
        "ETHUSDT",
        "2h",
        ["weekly_ema_200_position"],
        start=start,
        end=start + pd.Timedelta(days=1),
    )
    data = overlays["weekly_ema_200_position"]
    assert data["available"] is True
    assert data["parquet_column"] == "weekly_ema_200_position_f"
    assert data["point_count"] >= 1


def test_align_feature_points_to_candles(tmp_path):
    import pandas as pd

    from mlbot_console.services.feature_overlay import (
        _align_points_to_candles,
        load_feature_overlays,
    )

    feat_dir = tmp_path / "features" / "primary"
    feat_dir.mkdir(parents=True)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = [
        {
            "timestamp": start + pd.Timedelta(hours=i * 2),
            "regime_score": 0.1 + i * 0.05,
        }
        for i in range(5)
    ]
    pd.DataFrame(rows).to_parquet(feat_dir / "ETHUSDT.parquet", index=False)

    candles = [
        {"time": int((start + pd.Timedelta(hours=i * 2)).timestamp()), "close": 100 + i}
        for i in range(5)
    ]
    overlays = load_feature_overlays(
        tmp_path,
        "ETHUSDT",
        "2h",
        ["regime_score"],
        start=start,
        end=start + pd.Timedelta(days=1),
        candles=candles,
    )
    data = overlays["regime_score"]
    assert data["aligned"] is True
    assert data["point_count"] == len(candles)
    assert len(data["points"]) == len(candles)


def test_single_feature_row_aligns_to_multiple_candles(tmp_path):
    feat_dir = tmp_path / "features" / "primary"
    feat_dir.mkdir(parents=True)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    pd.DataFrame([{"timestamp": start, "regime_score": 0.42}]).to_parquet(
        feat_dir / "ETHUSDT.parquet", index=False
    )

    candles = [
        {"time": int((start + pd.Timedelta(hours=i * 2)).timestamp()), "close": 100 + i}
        for i in range(4)
    ]
    overlays = load_feature_overlays(
        tmp_path,
        "ETHUSDT",
        "2h",
        ["regime_score"],
        start=start,
        end=start + pd.Timedelta(hours=8),
        candles=candles,
    )
    pts = overlays["regime_score"]["points"]
    assert len(pts) == len(candles)
    assert len(pts) > 1


def test_align_ffill_after_last_feature_row(tmp_path):
    """Candles after the last feature timestamp should show last known value."""
    feat_dir = tmp_path / "features" / "primary"
    feat_dir.mkdir(parents=True)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = [
        {
            "timestamp": start + pd.Timedelta(hours=i * 2),
            "regime_score": 0.2 + i * 0.1,
        }
        for i in range(3)
    ]
    pd.DataFrame(rows).to_parquet(feat_dir / "ETHUSDT.parquet", index=False)

    candles = [
        {"time": int((start + pd.Timedelta(hours=i * 2)).timestamp()), "close": 100 + i}
        for i in range(6)
    ]
    overlays = load_feature_overlays(
        tmp_path,
        "ETHUSDT",
        "2h",
        ["regime_score"],
        start=start,
        end=start + pd.Timedelta(hours=12),
        candles=candles,
    )
    pts = overlays["regime_score"]["points"]
    assert len(pts) == 6
    assert pts[-1]["value"] == pytest.approx(pts[2]["value"])
    assert pts[-2]["value"] == pytest.approx(pts[2]["value"])


def test_load_multiple_overlays(bus_root):
    overlays = load_feature_overlays(
        bus_root,
        "ETHUSDT",
        "2h",
        ["weekly_ema_200_position", "regime_score"],
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-02", tz="UTC"),
    )
    assert overlays["weekly_ema_200_position"]["available"] is True
    assert overlays["regime_score"]["available"] is True
