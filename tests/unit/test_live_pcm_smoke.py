"""
冒烟测试: LivePCM 与 run_live 多策略入口

``_setup_bpc`` 已移除；经典实盘仅 ``_setup_three_strategies`` + Feature Bus。
"""

from unittest.mock import MagicMock

from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.portfolio.live_pcm import LivePCM


def test_setup_three_strategies_is_callable() -> None:
    from scripts import run_live

    assert callable(run_live._setup_three_strategies)


def test_run_live_pcm_registration_uses_enabled_archetypes_only() -> None:
    """No separate pcm_registered key; LivePCM loop follows ``enabled_archetypes``."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "run_live.py").read_text(encoding="utf-8")
    assert "for arch in enabled_archetypes" in text
    assert "pcm.register(_name" in text
    assert "pcm_registered_archetypes" not in text
    assert "get_open_slot_count=lambda: runtime_st.slots.active_count()" in text


def test_live_pcm_decide_delegates_to_bpc() -> None:
    """PCM.decide() 正确委托给注册的 BPC 策略"""
    intent = TradeIntent(
        action="LONG",
        symbol="BTCUSDT",
        archetype="bpc",
        confidence=0.8,
    )

    mock_bpc = MagicMock()
    mock_bpc.decide.return_value = [intent]

    pcm = LivePCM(max_slots=2)
    pcm.register("bpc", mock_bpc)

    result = pcm.decide(
        features={"close": 50000.0},
        symbol="BTCUSDT",
    )

    assert len(result) == 1
    assert result[0] is intent
    mock_bpc.decide.assert_called_once()


class _FakeTracker:
    def __init__(self, positions):
        self._positions = positions

    def all_positions(self):
        return self._positions


class _FakeListener:
    def __init__(self, positions):
        self._position_tracker = _FakeTracker(positions)


class _FakeManager:
    def __init__(self, by_symbol):
        self._by_symbol = by_symbol

    def get_listener(self, symbol):
        return self._by_symbol.get(symbol)


def test_run_live_open_trend_positions_snapshot_includes_breakeven_and_stop_flags() -> (
    None
):
    from scripts import run_live

    manager = _FakeManager(
        {
            "BTCUSDT": _FakeListener(
                {
                    "p1": {
                        "symbol": "BTCUSDT",
                        "archetype": "BPC",
                        "side": "LONG",
                        "entry_price": 100.0,
                        "stop_loss_price": 101.0,
                        "breakeven_locked": True,
                    }
                }
            ),
            "ETHUSDT": _FakeListener(
                {
                    "p2": {
                        "symbol": "ETHUSDT",
                        "archetype": "TPC",
                        "side": "short",
                        "entry_price": 200.0,
                        "stop_loss_price": 199.0,
                        "breakeven_locked": False,
                    }
                }
            ),
        }
    )

    rows = run_live._open_trend_positions_snapshot_from_manager(
        manager, ["BTCUSDT", "ETHUSDT"]
    )

    assert len(rows) == 2
    got = {(r["symbol"], r["archetype"]): r for r in rows}
    assert got[("BTCUSDT", "bpc")]["breakeven_locked"] is True
    assert got[("BTCUSDT", "bpc")]["stop_risk_nonnegative"] is True
    assert got[("ETHUSDT", "tpc")]["breakeven_locked"] is False
    assert got[("ETHUSDT", "tpc")]["stop_risk_nonnegative"] is True


def test_run_live_open_trend_positions_snapshot_skips_invalid_rows() -> None:
    from scripts import run_live

    manager = _FakeManager(
        {
            "BTCUSDT": _FakeListener(
                {
                    "bad1": {"archetype": "", "symbol": "BTCUSDT"},
                    "bad2": "not-a-dict",
                }
            )
        }
    )

    rows = run_live._open_trend_positions_snapshot_from_manager(manager, ["BTCUSDT"])
    assert rows == []
