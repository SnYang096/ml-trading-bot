import pandas as pd

from app.services.feature_overlay import load_feature_overlay


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
