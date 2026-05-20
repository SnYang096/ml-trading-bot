import pandas as pd

from app.services.feature_overlay import (
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
