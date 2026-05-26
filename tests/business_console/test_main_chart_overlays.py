"""Main chart slow MA overlays (EMA1200, weekly EMA200)."""

from __future__ import annotations

import pandas as pd
import pytest

from mlbot_console.services.main_chart_overlays import (
    _align_ma_to_candles,
    _align_weekly_ema_seed_to_candles,
    _ema1200_from_candle_closes,
    _ema1200_points_local,
    _overlay_points_for_chart,
    _seed_ema_plausible,
    load_main_chart_overlays,
)
from src.live_data_stream.spot_weekly_ema_seed import seed_parquet_path
from mlbot_console.services.ohlcv_reader import resolve_trade_map_window


def test_resolve_trade_map_window_defaults(bus_root):
    start, end, full = resolve_trade_map_window("2h", full_range=False)
    assert full is False
    assert start is not None
    assert (end - start).days >= 59


def test_align_ma_to_candles_merges_second_and_nanosecond_timestamps() -> None:
    """Parquet features are ns; candle times from unit=s must still merge_asof."""
    t0 = pd.Timestamp("2024-06-01 00:00", tz="UTC")
    feat = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([t0, t0 + pd.Timedelta(hours=2)], utc=True),
            "close": [100.0, 101.0],
            "ema_1200_position": [0.01, 0.02],
        }
    )
    candles = [
        {"time": int(t0.timestamp())},
        {"time": int((t0 + pd.Timedelta(hours=2)).timestamp())},
    ]
    points = _align_ma_to_candles(feat, "ema_1200_position", candles)
    assert len(points) == 2
    assert points[0]["value"] == pytest.approx(99.0)
    assert points[1]["value"] == pytest.approx(98.98)


def test_ema1200_overlay_uses_candle_ewm(bus_root) -> None:
    t0 = pd.Timestamp("2024-01-01 02:00", tz="UTC")
    t1 = pd.Timestamp("2024-01-01 14:00", tz="UTC")
    candles = [
        {"time": int(t0.timestamp()), "close": 700.0},
        {"time": int(t1.timestamp()), "close": 710.0},
    ]
    out = load_main_chart_overlays(
        "ETHUSDT",
        candles,
        ["ema_1200"],
        chart_timeframe="2h",
        feature_bus_root=bus_root,
    )
    assert out["ema_1200"]["available"]
    assert out["ema_1200"]["source"] == "bars_1min_2h"
    assert out["ema_1200"]["latest"] is not None


def test_ema1200_overlay_curve_has_varying_values(bus_root) -> None:
    candles = [
        {
            "time": int(pd.Timestamp("2024-01-01 02:00", tz="UTC").timestamp()),
            "close": 100.0,
        },
        {
            "time": int(pd.Timestamp("2024-01-01 14:00", tz="UTC").timestamp()),
            "close": 110.0,
        },
    ]
    out = load_main_chart_overlays(
        "ETHUSDT",
        candles,
        ["ema_1200"],
        chart_timeframe="2h",
    )
    pts = out["ema_1200"]["points"]
    assert len(pts) == 2
    assert pts[0]["value"] != pts[1]["value"]


def test_ema1200_warmup_uses_extended_2h_history(monkeypatch, bus_root) -> None:
    t0 = pd.Timestamp("2024-06-01 00:00", tz="UTC")
    long_candles = [
        {
            "time": int((t0 + pd.Timedelta(hours=2 * i)).timestamp()),
            "close": 600.0 + i * 0.5,
        }
        for i in range(50)
    ]
    visible = long_candles[-2:]

    def _fake_fetch(*_args, **_kwargs):
        return long_candles

    monkeypatch.setattr(
        "mlbot_console.services.main_chart_overlays._fetch_2h_candles",
        _fake_fetch,
    )
    pts, source = _ema1200_points_local(
        "ETHUSDT",
        visible,
        chart_timeframe="2h",
        feature_bus_root=bus_root,
    )
    expected = _overlay_points_for_chart(
        _ema1200_from_candle_closes(long_candles),
        visible,
    )
    short_only = _overlay_points_for_chart(
        _ema1200_from_candle_closes(visible),
        visible,
    )
    assert source == "bars_1min_2h"
    assert len(pts) == 2
    assert pts[-1]["value"] == pytest.approx(expected[-1]["value"])
    assert pts[-1]["value"] != pytest.approx(short_only[-1]["value"])


def test_load_main_overlays_aligns_to_candles(bus_root):
    candles = [
        {
            "time": int(pd.Timestamp("2024-01-01 10:00", tz="UTC").timestamp()),
            "close": 100.0,
        },
        {
            "time": int(pd.Timestamp("2024-01-01 14:00", tz="UTC").timestamp()),
            "close": 105.0,
        },
    ]
    out = load_main_chart_overlays(
        "ETHUSDT",
        candles,
        ["ema_1200"],
        chart_timeframe="2h",
    )
    assert out["ema_1200"]["available"]
    assert len(out["ema_1200"]["points"]) == 2
    assert out["ema_1200"]["points"][0]["value"] > 90


def test_weekly_ema_seed_curve_steps_over_weeks(tmp_path) -> None:
    seed_root = tmp_path / "macro" / "spot_weekly_ema200"
    seed_root.mkdir(parents=True, exist_ok=True)
    weeks = pd.date_range("2024-01-01", periods=4, freq="W-MON", tz="UTC")
    pd.DataFrame(
        {
            "week_ts": weeks,
            "weekly_ema_200": [500.0, 520.0, 540.0, 565.0],
        }
    ).to_parquet(seed_parquet_path(seed_root, "ETHUSDT"), index=False)
    candles = [
        {"time": int((weeks[0] + pd.Timedelta(days=d)).timestamp())}
        for d in range(0, 22, 2)
    ]
    points = _align_weekly_ema_seed_to_candles(seed_root, "ETHUSDT", candles)
    assert len(points) == len(candles)
    assert len({p["value"] for p in points}) >= 2


def test_stale_weekly_seed_uses_spot_daily_not_flat_line(tmp_path) -> None:
    import io
    import zipfile

    seed_root = tmp_path / "seed"
    seed_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "week_ts": [pd.Timestamp("2025-01-05", tz="UTC")],
            "weekly_ema_200": [374.0],
        }
    ).to_parquet(seed_parquet_path(seed_root, "ETHUSDT"), index=False)

    kroot = tmp_path / "klines" / "ETHUSDT" / "monthly" / "1d"
    kroot.mkdir(parents=True)
    rows = []
    t0 = int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)
    for i in range(3100):
        t = t0 + i * 86_400_000
        price = 500.0 + i * 0.5
        rows.append([t, price, price + 1, price - 1, price, 1000.0] + [0] * 6)
    body = "\n".join(",".join(str(v) for v in r) for r in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("k.csv", body)
    (kroot / "ETHUSDT-1d-2018-01.zip").write_bytes(buf.getvalue())

    candles = [
        {
            "time": int((pd.Timestamp("2026-05-20 02:00", tz="UTC")).timestamp()),
            "close": 650.0,
        },
        {
            "time": int((pd.Timestamp("2026-05-23 02:00", tz="UTC")).timestamp()),
            "close": 655.0,
        },
    ]
    out = load_main_chart_overlays(
        "ETHUSDT",
        candles,
        ["weekly_ema_200"],
        macro_seed_root=seed_root,
        macro_spot_kline_root=tmp_path / "klines",
    )
    wk = out["weekly_ema_200"]
    assert wk["available"]
    assert wk["source"] == "spot_daily_weekly"
    vals = {round(p["value"], 1) for p in wk["points"]}
    assert 374.0 not in vals or len(vals) > 1
    assert max(vals) > 400


def test_weekly_ema_overlay_uses_macro_seed_not_bus_close(tmp_path, bus_root) -> None:
    seed_root = tmp_path / "macro" / "spot_weekly_ema200"
    seed_root.mkdir(parents=True, exist_ok=True)
    week = pd.Timestamp("2024-01-01", tz="UTC")
    pd.DataFrame(
        {
            "week_ts": [week],
            "weekly_ema_200": [565.0],
        }
    ).to_parquet(seed_parquet_path(seed_root, "ETHUSDT"), index=False)

    t_bar = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    candles = [{"time": int(t_bar.timestamp()), "close": 617.0}]
    points = _align_weekly_ema_seed_to_candles(seed_root, "ETHUSDT", candles)
    assert len(points) == 1
    assert points[0]["value"] == pytest.approx(565.0)

    out = load_main_chart_overlays(
        "ETHUSDT",
        candles,
        ["weekly_ema_200"],
        macro_seed_root=seed_root,
    )
    assert out["weekly_ema_200"]["available"]
    assert out["weekly_ema_200"]["source"] == "macro_seed"
    assert out["weekly_ema_200"]["latest"] == pytest.approx(565.0)


def test_weekly_ema_unavailable_without_macro(bus_root) -> None:
    t0 = pd.Timestamp("2024-01-01 02:00", tz="UTC")
    candles = [{"time": int(t0.timestamp()), "close": 617.0}]
    out = load_main_chart_overlays(
        "ETHUSDT",
        candles,
        ["weekly_ema_200"],
        macro_seed_root=None,
        macro_spot_kline_root=None,
    )
    assert not out["weekly_ema_200"]["available"]


def test_weekly_ema_falls_back_to_chart_candles_when_macro_stale(bus_root) -> None:
    """1w timeframe with multi-year history should still get an EMA200 line."""
    base = pd.Timestamp("2023-01-02", tz="UTC")
    candles = []
    for i in range(120):
        ts = base + pd.Timedelta(weeks=i)
        candles.append({"time": int(ts.timestamp()), "close": 500.0 + i * 1.5})
    out = load_main_chart_overlays(
        "BNBUSDT",
        candles,
        ["weekly_ema_200"],
        macro_seed_root=None,
        macro_spot_kline_root=None,
    )
    assert out["weekly_ema_200"]["available"]
    assert out["weekly_ema_200"]["source"] == "chart_resample_weekly"
    assert out["weekly_ema_200"]["point_count"] >= 50


def test_weekly_ema_1w_aligns_one_point_per_candle(bus_root) -> None:
    """Weekly chart overlays must cover every bar (LWC needs aligned times)."""
    base = pd.Timestamp("2023-01-02", tz="UTC")
    candles = []
    for i in range(120):
        ts = base + pd.Timedelta(weeks=i)
        candles.append({"time": int(ts.timestamp()), "close": 500.0 + i * 1.5})
    out = load_main_chart_overlays(
        "BNBUSDT",
        candles,
        ["weekly_ema_200"],
        chart_timeframe="1w",
        macro_seed_root=None,
        macro_spot_kline_root=None,
    )
    pts = out["weekly_ema_200"]["points"]
    assert len(pts) == len(candles)
    assert pts[0]["time"] == candles[0]["time"]
    assert pts[-1]["time"] == candles[-1]["time"]


def test_ema1200_fetch_caps_window_on_long_weekly_chart(monkeypatch, bus_root) -> None:
    """1w full history must not OhlcvWindowError and block other overlays."""
    base = pd.Timestamp("2020-01-05", tz="UTC")
    candles = [
        {
            "time": int((base + pd.Timedelta(weeks=i)).timestamp()),
            "close": 500.0 + i,
        }
        for i in range(120)
    ]

    def _boom(*_args, **_kwargs):
        from mlbot_console.services.ohlcv_reader import OhlcvWindowError

        raise OhlcvWindowError("range 520.0d exceeds max_ohlcv_days=180")

    monkeypatch.setattr(
        "mlbot_console.services.ohlcv_reader.fetch_ohlcv",
        _boom,
    )
    out = load_main_chart_overlays(
        "ETHUSDT",
        candles,
        ["ema_1200", "weekly_ema_200"],
        chart_timeframe="1w",
        feature_bus_root=bus_root,
    )
    assert out["weekly_ema_200"]["available"]
    assert len(out["weekly_ema_200"]["points"]) == len(candles)
    assert out["ema_1200"]["available"]
    assert out["ema_1200"]["source"] == "chart_resample_2h"


def test_weekly_ema_chart_fallback_skips_short_window(bus_root) -> None:
    """Short chart windows (<52 weekly bars) should NOT show a degenerate EMA."""
    base = pd.Timestamp("2026-01-05", tz="UTC")
    candles = []
    for i in range(20):
        ts = base + pd.Timedelta(weeks=i)
        candles.append({"time": int(ts.timestamp()), "close": 600.0})
    out = load_main_chart_overlays(
        "BNBUSDT",
        candles,
        ["weekly_ema_200"],
        macro_seed_root=None,
        macro_spot_kline_root=None,
    )
    assert not out["weekly_ema_200"]["available"]


def test_stale_seed_412_rejected_when_spot_daily_available(tmp_path) -> None:
    """BNB-like: stale seed ~412 must not win when Vision daily recomputes ~650+."""
    import io
    import zipfile

    seed_root = tmp_path / "seed"
    seed_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "week_ts": [pd.Timestamp("2026-05-18", tz="UTC")],
            "weekly_ema_200": [412.0],
        }
    ).to_parquet(seed_parquet_path(seed_root, "BNBUSDT"), index=False)

    kroot = tmp_path / "klines" / "BNBUSDT" / "monthly" / "1d"
    kroot.mkdir(parents=True)
    rows = []
    t0 = int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)
    for i in range(3100):
        t = t0 + i * 86_400_000
        price = 80.0 + i * 0.25
        rows.append([t, price, price + 1, price - 1, price, 1000.0] + [0] * 6)
    body = "\n".join(",".join(str(v) for v in r) for r in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("k.csv", body)
    (kroot / "BNBUSDT-1d-2018-01.zip").write_bytes(buf.getvalue())

    candles = [
        {
            "time": int(pd.Timestamp("2026-05-23 02:00", tz="UTC").timestamp()),
            "close": 658.0,
        },
    ]
    out = load_main_chart_overlays(
        "BNBUSDT",
        candles,
        ["weekly_ema_200"],
        macro_seed_root=seed_root,
        macro_spot_kline_root=tmp_path / "klines",
    )
    wk = out["weekly_ema_200"]
    assert wk["available"]
    assert wk["source"] == "spot_daily_weekly"
    assert wk["latest"] > 500.0
    assert wk["latest"] != pytest.approx(412.0, rel=0.02)


def test_seed_ema_plausible_rejects_flat_line_far_from_spot(tmp_path) -> None:
    seed_root = tmp_path / "seed"
    seed_root.mkdir(parents=True)
    week = pd.Timestamp("2026-05-19", tz="UTC")
    pd.DataFrame(
        {
            "week_ts": [week],
            "weekly_ema_200": [374.0],
        }
    ).to_parquet(seed_parquet_path(seed_root, "BNBUSDT"), index=False)
    candles = [
        {
            "time": int((pd.Timestamp("2026-05-23 02:00", tz="UTC")).timestamp()),
            "close": 650.0,
        }
    ]
    seed = pd.read_parquet(seed_parquet_path(seed_root, "BNBUSDT"))
    assert not _seed_ema_plausible(seed, candles)
    assert _align_weekly_ema_seed_to_candles(seed_root, "BNBUSDT", candles) == []
