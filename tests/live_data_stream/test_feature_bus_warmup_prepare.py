from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_prepare_warmup_dataset_full_daily_zip_when_coverage_missing(
    tmp_path: Path, monkeypatch
) -> None:
    import live.scripts.prepare_warmup_ticks as prep

    calls: list[object] = []

    monkeypatch.setattr(
        prep,
        "compute_date_ranges",
        lambda months: (2025, 11, 2026, 4, "2026-05-01", "2026-05-12"),
    )

    def _reject_monthly(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected monthly zip download (default is daily)")

    monkeypatch.setattr(prep, "download_monthly", _reject_monthly)
    monkeypatch.setattr(
        prep,
        "download_daily",
        lambda *args, **kwargs: calls.append(("daily", args[1], args[2])),
    )
    monkeypatch.setattr(
        prep,
        "convert_and_split",
        lambda *args, **kwargs: calls.append("convert") or {"BTCUSDT": 1},
    )

    stats = prep.prepare_warmup_dataset(
        symbols=["BTCUSDT"],
        months=6,
        ticks_dir=tmp_path / "ticks",
        bars_dir=tmp_path / "bars",
        zip_dir=tmp_path / "raw",
    )

    assert stats == {"BTCUSDT": 1}
    assert calls[:-1] == [
        ("daily", "2025-11-01", "2026-05-12"),
    ]
    assert calls[-1] == "convert"


def test_prepare_warmup_dataset_monthly_zip_legacy_chain(
    tmp_path: Path, monkeypatch
) -> None:
    import live.scripts.prepare_warmup_ticks as prep

    seq: list[str] = []

    monkeypatch.setattr(
        prep,
        "compute_date_ranges",
        lambda months: (2025, 11, 2026, 4, "2026-05-01", "2026-05-12"),
    )
    monkeypatch.setattr(
        prep, "download_monthly", lambda *args, **kwargs: seq.append("monthly")
    )
    monkeypatch.setattr(
        prep, "download_daily", lambda *args, **kwargs: seq.append("daily")
    )
    monkeypatch.setattr(
        prep,
        "convert_and_split",
        lambda *args, **kwargs: seq.append("convert") or {"BTCUSDT": 1},
    )

    stats = prep.prepare_warmup_dataset(
        symbols=["BTCUSDT"],
        months=6,
        ticks_dir=tmp_path / "ticks",
        bars_dir=tmp_path / "bars",
        zip_dir=tmp_path / "raw",
        use_monthly_zip=True,
    )

    assert stats == {"BTCUSDT": 1}
    assert seq == ["monthly", "daily", "convert"]


def test_prepare_warmup_dataset_gap_only_when_coverage_sufficient(
    tmp_path: Path, monkeypatch
) -> None:
    import live.scripts.prepare_warmup_ticks as prep

    ticks_dir = tmp_path / "ticks"
    _touch(ticks_dir / "BTCUSDT" / "2026-05-01.parquet")
    _touch(ticks_dir / "BTCUSDT" / "2026-05-02.parquet")

    monkeypatch.setattr(
        prep,
        "compute_date_ranges",
        lambda months: (2026, 5, 2026, 5, "2026-05-02", "2026-05-02"),
    )
    monkeypatch.setattr(
        prep,
        "fill_gap",
        lambda *args, **kwargs: {"BTCUSDT": 0},
    )
    monkeypatch.setattr(
        prep,
        "download_monthly",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("no full download")
        ),
    )

    stats = prep.prepare_warmup_dataset(
        symbols=["BTCUSDT"],
        months=6,
        ticks_dir=ticks_dir,
        bars_dir=tmp_path / "bars",
        zip_dir=tmp_path / "raw",
    )

    assert stats == {"BTCUSDT": 0}


def test_feature_bus_prepare_uses_live_storage_paths(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_market_feature_publisher as publisher

    captured = {}

    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        publisher,
        "prepare_warmup_dataset",
        lambda **kwargs: captured.update(kwargs) or {"BTCUSDT": 1},
    )

    args = Namespace(
        skip_warmup_prepare=False,
        warmup_months=6,
        live_storage_base="live/highcap/data",
        warmup_raw_dir="data/warmup_raw/highcap",
        symbols="BTCUSDT,ETHUSDT",
    )

    publisher._prepare_live_warmup(args)

    assert captured["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert captured["months"] == 6
    assert captured["ticks_dir"] == tmp_path / "live/highcap/data/ticks"
    assert captured["bars_dir"] == tmp_path / "live/highcap/data/bars"
    assert captured["zip_dir"] == tmp_path / "data/warmup_raw/highcap"
    assert captured["force_full"] is False
