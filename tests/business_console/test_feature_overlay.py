import pandas as pd

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
