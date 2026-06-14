"""T2d rebalance cockpit scheduled check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.monitoring.rebalance_cockpit_run import (
    _compact_detail,
    _exit_code,
    _monitor_status,
    format_rebalance_telegram_message,
)


def test_monitor_status_and_exit_code():
    assert _monitor_status("OK") == "OK"
    assert _monitor_status("WATCH") == "ALERT"
    assert _exit_code("OK") == 0
    assert _exit_code("WATCH") == 1
    assert _exit_code("REBALANCE_SUGGEST") == 2


def test_format_rebalance_telegram_message():
    payload = {
        "symbol": "BTCUSDT",
        "composite": {"label_title": "risk-on"},
        "feature_bus": {"stale": False},
        "allocation": {
            "total_nav_usdt": 9000,
            "scopes": [{"label": "A·Spot", "nav_pct": 0.1, "status": "WATCH"}],
            "suggestions": ["考虑增加 beta"],
        },
    }
    msg = format_rebalance_telegram_message(
        payload=payload, alert="WATCH", run_ts="20260612_120000"
    )
    assert "WATCH" in msg
    assert "BTCUSDT" in msg
    assert "考虑增加 beta" in msg


def test_rebalance_check_exits_zero_on_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.monitoring.rebalance_cockpit_run import run_rebalance_cockpit_check

    monkeypatch.setattr(
        "src.monitoring.rebalance_cockpit_run.build_regime_cockpit",
        lambda **kw: {
            "symbol": "BTCUSDT",
            "composite": {"label": "neutral"},
            "feature_bus": {"stale": False},
            "allocation": {"alert": "WATCH", "scopes": [], "suggestions": []},
        },
    )
    summary = run_rebalance_cockpit_check(dry_run=True)
    assert summary["exit_code"] == 1

    import scripts.monitoring.rebalance_cockpit_check as chk

    monkeypatch.setattr(chk, "run_rebalance_cockpit_check", lambda **kw: summary)
    monkeypatch.setattr(chk.sys, "argv", ["rebalance_cockpit_check.py"])
    assert chk.main() == 0


def test_compact_detail():
    payload = {
        "symbol": "BTCUSDT",
        "as_of": "2026-06-12T00:00:00Z",
        "composite": {"label": "neutral"},
        "feature_bus": {"stale": False},
        "allocation": {
            "alert": "WATCH",
            "total_nav_usdt": 1000,
            "scopes": [
                {
                    "scope": "spot",
                    "nav_pct": 0.2,
                    "status": "OK",
                    "band": {"target": 0.25},
                }
            ],
            "suggestions": ["ok"],
        },
    }
    d = _compact_detail(payload)
    assert d["alert"] == "WATCH"
    assert d["scopes"][0]["scope"] == "spot"


def test_composite_rules_cover_risk_on_boundary():
    """total_score=10 (macro≥4 + weekly≥0 + bull≥0.25 + trend≥0.7) → risk_on."""
    from mlbot_console.services.rebalance_advisor import (
        compute_composite,
        load_rebalance_config,
    )

    config = load_rebalance_config(Path(__file__).resolve().parents[2])
    assert config, "rebalance_targets.yaml not found"

    # Full risk-on: all inputs at max score
    ctx = {
        "abc_macro_regime_score": 5,
        "weekly_ema_200_position": 0.02,
        "tpc_bull_share_7d": 0.30,
        "tpc_bull_label": "bull",
        "chop_semantic": 0.30,
        "trend_confidence": 0.75,
    }
    result = compute_composite(ctx, config)
    assert (
        result["label"] == "risk_on"
    ), f"total={result['total_score']}, expected risk_on"
    assert (
        result["total_score"] >= 7
    ), f"total should be >=7 for risk_on, got {result['total_score']}"

    # Borderline: just above risk_off threshold
    ctx_mid = {
        "abc_macro_regime_score": 3,
        "weekly_ema_200_position": 0.01,
        "tpc_bull_share_7d": 0.12,
        "chop_semantic": 0.40,
        "trend_confidence": 0.65,
    }
    result_mid = compute_composite(ctx_mid, config)
    assert result_mid["label"] in {
        "neutral",
        "risk_on",
    }, f"total={result_mid['total_score']}, expected neutral or risk_on"

    # Risk-off: all inputs low
    ctx_off = {
        "abc_macro_regime_score": 2,
        "weekly_ema_200_position": -0.10,
        "tpc_bull_share_7d": 0.01,
        "chop_semantic": 0.55,
        "trend_confidence": 0.30,
    }
    result_off = compute_composite(ctx_off, config)
    assert (
        result_off["label"] == "risk_off"
    ), f"total={result_off['total_score']}, expected risk_off"


def test_map_covers_max_possible_total():
    """The last map entry cap must be >= theoretical max (11)."""
    from mlbot_console.services.rebalance_advisor import load_rebalance_config

    config = load_rebalance_config(Path(__file__).resolve().parents[2])
    map_rules = (config.get("composite") or {}).get("map") or []
    assert map_rules, "map is empty"
    last_cap = int(map_rules[-1].get("max_total") or 0)
    assert (
        last_cap >= 11
    ), f"last map max_total={last_cap} must cover theoretical max of 11"
