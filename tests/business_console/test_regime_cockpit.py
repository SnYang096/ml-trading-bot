"""Regime Cockpit + rebalance advisor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlbot_console.services.rebalance_advisor import (
    build_allocation,
    build_regime_cockpit,
    compute_composite,
    load_rebalance_config,
    _contradiction_alert,
    _eval_when,
)
from mlbot_console.services.regime_live import build_live_layers


def _write_rebalance_config(repo: Path) -> None:
    src = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "monitoring"
        / "rebalance_targets.yaml"
    )
    dest_dir = repo / "config" / "monitoring"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "rebalance_targets.yaml"
    if src.is_file():
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        dest.write_text("version: 1\nbands: {}\ncomposite: {}\n", encoding="utf-8")


def _seed_btc_features(bus_root: Path) -> None:
    feat_dir = bus_root / "features" / "120T"
    rows = []
    start = pd.Timestamp("2024-06-01", tz="UTC")
    for i in range(14):
        ts = start + pd.Timedelta(hours=2 * i)
        rows.append(
            {
                "timestamp": ts,
                "close": 65000.0 + i * 10,
                "weekly_ema_200_position": 0.08,
                "abc_macro_regime_score": 4.0,
                "adx_50": 28.0,
                "ema_1200_position": 0.15,
                "bpc_semantic_chop": 0.45,
                "trend_confidence": 0.75,
            }
        )
    pd.DataFrame(rows).to_parquet(feat_dir / "BTCUSDT.parquet", index=False)
    latest = bus_root / "latest" / "features" / "120T"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "BTCUSDT.json").write_text(
        json.dumps({"timestamp": rows[-1]["timestamp"].isoformat(), "rows": len(rows)}),
        encoding="utf-8",
    )


def test_eval_when_and_composite_risk_on():
    config = load_rebalance_config(Path(__file__).resolve().parents[2])
    ctx = {
        "abc_macro_regime_score": 4,
        "weekly_ema_200_position": 0.1,
        "tpc_bull_share_7d": 0.3,
        "tpc_bull_label": "bull",
        "chop_semantic": 0.4,
        "trend_confidence": 0.8,
    }
    assert _eval_when("abc_macro_regime_score >= 4", ctx)
    assert _eval_when("tpc_bull_label == bull", ctx)
    comp = compute_composite(ctx, config)
    assert comp["label"] == "risk_on"
    assert comp["total_score"] > 0


def test_build_live_layers(bus_root: Path, tmp_path: Path) -> None:
    _seed_btc_features(bus_root)
    strategies = Path(__file__).resolve().parents[2] / "config" / "strategies"
    live = build_live_layers(
        strategies_root=strategies,
        project_root=Path(__file__).resolve().parents[2],
        feature_bus_root=bus_root,
        symbol="BTCUSDT",
        window_days=7,
    )
    assert live["layers"]["a_spot"]["deploy_state"] == "ABOVE_EMA200"
    assert live["layers"]["b_trend"]["current_label"] in {"bull", "bear", "neutral"}
    assert live["layers"]["c_multileg"]["router_hint"] in {
        "chop_favored",
        "momentum_favored",
        "chop_neutral",
        "neutral",
    }


def test_feature_bus_stale_uses_120t_not_1min(bus_root: Path) -> None:
    """1min bar fresh + 120T stale → cockpit marks feature bus stale."""
    _seed_btc_features(bus_root)
    old_ts = pd.Timestamp("2024-01-01", tz="UTC")
    feat_dir = bus_root / "features" / "120T"
    rows = [{"timestamp": old_ts, "weekly_ema_200_position": 0.08}]
    pd.DataFrame(rows).to_parquet(feat_dir / "BTCUSDT.parquet", index=False)
    (bus_root / "latest" / "features" / "120T" / "BTCUSDT.json").write_text(
        json.dumps({"timestamp": old_ts.isoformat(), "rows": 1}),
        encoding="utf-8",
    )
    bars_dir = bus_root / "latest" / "bars_1min"
    bars_dir.mkdir(parents=True, exist_ok=True)
    fresh = pd.Timestamp.now(tz="UTC").isoformat()
    (bars_dir / "BTCUSDT.json").write_text(
        json.dumps({"timestamp": fresh, "rows": 1}),
        encoding="utf-8",
    )
    strategies = Path(__file__).resolve().parents[2] / "config" / "strategies"
    live = build_live_layers(
        strategies_root=strategies,
        project_root=Path(__file__).resolve().parents[2],
        feature_bus_root=bus_root,
        symbol="BTCUSDT",
        stale_minutes=60,
    )
    assert live["feature_bus"]["stale"] is True


def test_missing_scope_nav_pct_is_null(tmp_path: Path) -> None:
    _write_rebalance_config(tmp_path)
    config = load_rebalance_config(tmp_path)
    ledger = {
        "accounts": [
            {"scope": "spot", "ok": True, "equity_usdt": 1000.0},
            {"scope": "trend", "ok": False, "error": "auth"},
            {"scope": "multi_leg", "ok": True, "equity_usdt": 500.0},
        ]
    }
    alloc = build_allocation(ledger=ledger, composite_label="neutral", config=config)
    trend = next(r for r in alloc["scopes"] if r["scope"] == "trend")
    assert trend["status"] == "MISSING"
    assert trend["nav_pct"] is None


def test_contradiction_alert_risk_on_low_beta(tmp_path: Path) -> None:
    _write_rebalance_config(tmp_path)
    config = load_rebalance_config(tmp_path)
    composite = {"label": "risk_on"}
    ledger = {
        "accounts": [
            {"scope": "spot", "ok": True, "equity_usdt": 500.0},
            {"scope": "trend", "ok": True, "equity_usdt": 4500.0},
            {"scope": "multi_leg", "ok": True, "equity_usdt": 500.0},
        ]
    }
    alloc = build_allocation(ledger=ledger, composite_label="risk_on", config=config)
    assert _contradiction_alert(composite, alloc) == "REBALANCE_SUGGEST"


def test_regime_cockpit_api(
    client, bus_root: Path, tmp_path: Path, monkeypatch
) -> None:
    _seed_btc_features(bus_root)
    _write_rebalance_config(tmp_path)

    def _fake_ledger(**kwargs):
        return {
            "accounts": [
                {
                    "scope": "spot",
                    "ok": True,
                    "equity_usdt": 2500.0,
                    "label": "A·Spot",
                },
                {
                    "scope": "trend",
                    "ok": True,
                    "equity_usdt": 5000.0,
                    "label": "B·Trend",
                },
                {
                    "scope": "multi_leg",
                    "ok": True,
                    "equity_usdt": 1500.0,
                    "label": "C·Multi-leg",
                },
            ],
            "totals": {"equity_usdt": 9000.0},
        }

    monkeypatch.setattr(
        "mlbot_console.services.rebalance_advisor.build_exchange_ledger",
        _fake_ledger,
    )

    r = client.get("/api/regime/cockpit", params={"symbol": "BTCUSDT"})
    assert r.status_code == 200, r.text[:500]
    body = r.json()
    assert body["ok"] is True
    data = body["data"]
    assert data["layers"]["a_spot"]
    assert data["layers"]["b_trend"]
    assert data["layers"]["c_multileg"]
    assert data["allocation"]["total_nav_usdt"] == pytest.approx(9000.0)
    assert data["allocation"]["alert"] in {"OK", "WATCH", "REBALANCE_SUGGEST"}
    assert isinstance(data["ops"], list)


def test_build_regime_cockpit_integration(
    bus_root: Path, tmp_path: Path, monkeypatch
) -> None:
    _seed_btc_features(bus_root)
    _write_rebalance_config(tmp_path)
    monkeypatch.setattr(
        "mlbot_console.services.rebalance_advisor.build_exchange_ledger",
        lambda **kw: {
            "accounts": [
                {"scope": "spot", "ok": True, "equity_usdt": 1000.0},
                {"scope": "trend", "ok": True, "equity_usdt": 4000.0},
                {"scope": "multi_leg", "ok": True, "equity_usdt": 500.0},
            ],
            "totals": {"equity_usdt": 5500.0},
        },
    )
    strategies = Path(__file__).resolve().parents[2] / "config" / "strategies"
    payload = build_regime_cockpit(
        strategies_root=strategies,
        project_root=tmp_path,
        feature_bus_root=bus_root,
        symbol="BTCUSDT",
    )
    assert payload["composite"]["label"] in {"risk_on", "neutral", "risk_off"}
    assert len(payload["allocation"]["scopes"]) >= 3


def test_rebalance_cockpit_check_persists(
    bus_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from mlbot_console.config import ConsoleSettings
    from src.monitoring.rebalance_cockpit_run import run_rebalance_cockpit_check

    _seed_btc_features(bus_root)
    _write_rebalance_config(tmp_path)
    db = tmp_path / "rd_registry.sqlite"

    monkeypatch.setattr(
        "mlbot_console.services.rebalance_advisor.build_exchange_ledger",
        lambda **kw: {
            "accounts": [
                {"scope": "spot", "ok": True, "equity_usdt": 1000.0},
                {"scope": "trend", "ok": True, "equity_usdt": 3000.0},
                {"scope": "multi_leg", "ok": True, "equity_usdt": 500.0},
            ],
            "totals": {"equity_usdt": 4500.0},
        },
    )
    monkeypatch.setattr(
        "src.monitoring.rebalance_cockpit_run.send_telegram_message",
        lambda *a, **k: False,
    )

    settings = ConsoleSettings(
        repo_root=tmp_path,
        feature_bus_root=bus_root,
        live_data_root=tmp_path,
        engine_data_root=tmp_path,
        live_root=tmp_path,
        constitution_yaml=tmp_path / "c.yaml",
        universe_yaml=tmp_path / "u.yaml",
        trend_order_db=tmp_path / "t.db",
        live_monitor_db=tmp_path / "m.db",
        account_snapshot_db=tmp_path / "a.db",
        spot_order_db=tmp_path / "s.db",
        spot_ledger_db=tmp_path / "l.db",
        multi_leg_db=tmp_path / "ml.db",
        max_ohlcv_days=7,
        live_storage_bars_root=tmp_path / "bars",
        stitch_live_storage=False,
        macro_spot_kline_root=tmp_path / "mk",
        macro_weekly_ema_seed_root=tmp_path / "mw",
        daily_ohlcv_start=__import__("datetime").date(2020, 1, 1),
        max_daily_ohlcv_days=365,
        map_poll_seconds=10.0,
        grafana_url="",
        rolling_backtest_url="",
        basic_auth_user=None,
        basic_auth_password=None,
        strategies_root=Path(__file__).resolve().parents[2] / "config" / "strategies",
    )

    summary = run_rebalance_cockpit_check(
        settings=settings,
        symbol="BTCUSDT",
        registry_db=db,
        output_root=tmp_path / "monitoring" / "rebalance_4h",
        skip_telegram=True,
    )
    assert summary["run_ts"]
    out = tmp_path / "monitoring" / "rebalance_4h" / summary["run_ts"] / "cockpit.json"
    assert out.is_file()
    idx = json.loads((tmp_path / "results" / "monitoring" / "index.json").read_text())
    assert "rebalance_4h" in idx.get("cadences", {})

    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT status, source FROM monitor_event WHERE cadence = 'rebalance_4h'"
        ).fetchone()
        assert row is not None
        assert row[1] == "rebalance_cockpit"
    finally:
        conn.close()
