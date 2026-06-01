"""Unit tests for monitor Telegram helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.monitoring.telegram import (
    format_alert_message,
    notify_cadence_result,
    send_telegram_message,
    should_notify_cadence_result,
)


def test_should_notify_on_exit_or_business_alert():
    assert should_notify_cadence_result(exit_code=1, index_row={"status": "OK"}) is True
    assert (
        should_notify_cadence_result(exit_code=0, index_row={"status": "ALERT"}) is True
    )
    assert (
        should_notify_cadence_result(
            exit_code=0, index_row={"status": "OK", "drift_any_alert": True}
        )
        is True
    )
    assert (
        should_notify_cadence_result(exit_code=0, index_row={"status": "OK"}) is False
    )


@patch("src.monitoring.telegram._cooldown_ok", return_value=True)
@patch("src.monitoring.telegram._load_telegram_creds", return_value=("tok", "chat"))
@patch("urllib.request.urlopen")
def test_send_telegram_message(mock_urlopen, mock_creds, mock_cd):
    mock_urlopen.return_value.__enter__ = MagicMock(
        return_value=MagicMock(read=MagicMock(return_value=b"{}"))
    )
    mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
    assert send_telegram_message("hello", stamp_key="test", cooldown_sec=0) is True


@patch("src.monitoring.telegram.send_telegram_message", return_value=True)
def test_notify_cadence_result_skips_ok(mock_send, tmp_path):
    db = tmp_path / "registry.sqlite"
    db.touch()
    assert (
        notify_cadence_result(
            cadence="weekly",
            exit_code=0,
            index_row={"status": "OK", "run_ts": "20260101_1200"},
            registry_db=db,
        )
        is False
    )
    mock_send.assert_not_called()


def test_format_alert_message_contains_strategy():
    msg = format_alert_message(
        cadence="weekly",
        card={"display_status": "ALERT", "run_ts": "t", "exit_code": 1},
        alert_events=[{"source": "drift", "strategy": "tpc", "status": "ALERT"}],
        host="h",
    )
    assert "drift/tpc" in msg
