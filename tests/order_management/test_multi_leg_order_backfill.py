from __future__ import annotations

import pytest

from src.order_management.multi_leg_order_backfill import (
    multi_leg_backfill_enabled,
    multi_leg_backfill_interval_seconds,
    normalize_rest_order_status,
    run_multi_leg_backfill_once,
)
from src.order_management.multi_leg_storage import MultiLegStorage


def test_backfill_interval_default_and_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLBOT_MULTI_LEG_ORDER_BACKFILL_INTERVAL_SECONDS", raising=False)
    assert multi_leg_backfill_interval_seconds() == 60.0
    monkeypatch.setenv("MLBOT_MULTI_LEG_ORDER_BACKFILL_INTERVAL_SECONDS", "off")
    assert multi_leg_backfill_interval_seconds() == 0.0


def test_backfill_interval_invalid_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLBOT_MULTI_LEG_ORDER_BACKFILL_INTERVAL_SECONDS", "bad")
    assert multi_leg_backfill_interval_seconds() == 60.0


def test_backfill_enabled_requires_api_and_storage() -> None:
    class _API:
        def get_order(self, *_args):
            return {}

    class _Storage:
        def get_recent_orders_for_backfill(self, **_kwargs):
            return []

        def apply_execution_report(self, _payload):
            return 0

    assert multi_leg_backfill_enabled(_API(), _Storage()) is True
    assert multi_leg_backfill_enabled(None, _Storage()) is False
    assert multi_leg_backfill_enabled(_API(), None) is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("closed", "filled"),
        ("FILLED", "filled"),
        ("cancelled", "canceled"),
        ("canceled", "canceled"),
        ("open", "open"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_normalize_rest_order_status(raw, expected: str) -> None:
    assert normalize_rest_order_status(raw) == expected


def test_run_backfill_once_updates_rows() -> None:
    class _API:
        def get_order(self, order_id: str, symbol: str):
            assert order_id == "111"
            assert symbol == "ETHUSDT"
            return {
                "order_id": "111",
                "client_order_id": "cg_xxx",
                "status": "closed",
                "filled": 0.12,
                "average_price": 3245.1,
                "timestamp": 1710000000,
                "update_time": 1710000060,
                "reject_reason": None,
                "error_message": None,
            }

    class _Storage:
        def __init__(self) -> None:
            self.payloads = []

        def get_recent_orders_for_backfill(self, **_kwargs):
            return [
                {
                    "run_id": "mlr_1",
                    "strategy": "chop_grid",
                    "symbol": "ETHUSDT",
                    "exchange_order_id": "111",
                    "client_order_id": "cg_xxx",
                }
            ]

        def apply_execution_report(self, payload):
            self.payloads.append(dict(payload))
            return 1

    storage = _Storage()
    changed = run_multi_leg_backfill_once(
        api=_API(),
        storage=storage,
        lookback_hours=24,
        limit=100,
    )
    assert changed == 1
    assert storage.payloads and storage.payloads[0]["status"] == "filled"


def test_run_backfill_once_skips_bad_candidate_and_continues() -> None:
    class _API:
        def get_order(self, order_id: str, _symbol: str):
            if order_id == "bad":
                raise RuntimeError("temporary exchange error")
            return {"order_id": order_id, "status": "open", "filled": 0.0}

    class _Storage:
        def __init__(self) -> None:
            self.payloads = []

        def get_recent_orders_for_backfill(self, **_kwargs):
            return [
                {"symbol": "ETHUSDT", "exchange_order_id": "bad"},
                {"symbol": "ETHUSDT", "exchange_order_id": "ok"},
            ]

        def apply_execution_report(self, payload):
            self.payloads.append(dict(payload))
            return 1

    storage = _Storage()
    changed = run_multi_leg_backfill_once(
        api=_API(),
        storage=storage,
        lookback_hours=24,
        limit=100,
    )
    assert changed == 1
    assert [p["order_id"] for p in storage.payloads] == ["ok"]


def test_run_backfill_marks_stale_missing_order_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLBOT_MULTI_LEG_STALE_OPEN_GRACE_SECONDS", "0")

    class _API:
        def get_order(self, _order_id: str, _symbol: str):
            return None

        def get_open_orders(self, _symbol: str):
            return []

    class _Storage:
        def __init__(self) -> None:
            self.payloads = []

        def get_recent_orders_for_backfill(self, **_kwargs):
            return [
                {
                    "run_id": "mlr_1",
                    "strategy": "chop_grid",
                    "symbol": "BNBUSDT",
                    "exchange_order_id": "90414532831",
                    "client_order_id": "cg_abc",
                    "status": "open",
                    "updated_at": "2026-05-19 13:41:12",
                    "created_at": "2026-05-19 13:41:12",
                }
            ]

        def apply_execution_report(self, payload):
            self.payloads.append(dict(payload))
            return 1

    storage = _Storage()
    changed = run_multi_leg_backfill_once(
        api=_API(),
        storage=storage,
        lookback_hours=24,
        limit=100,
    )
    assert changed == 1
    assert storage.payloads
    assert storage.payloads[0]["status"] == "expired"
    assert storage.payloads[0]["reject_reason"] == "exchange_order_missing"


def test_run_backfill_skips_stale_when_get_open_orders_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLBOT_MULTI_LEG_STALE_OPEN_GRACE_SECONDS", "0")

    class _API:
        def get_order(self, _order_id: str, _symbol: str):
            return None

        def get_open_orders(self, _symbol: str):
            raise RuntimeError("rate limit")

    class _Storage:
        def __init__(self) -> None:
            self.payloads = []

        def get_recent_orders_for_backfill(self, **_kwargs):
            return [
                {
                    "run_id": "mlr_1",
                    "strategy": "chop_grid",
                    "symbol": "BNBUSDT",
                    "exchange_order_id": "90414532831",
                    "client_order_id": "cg_abc",
                    "status": "open",
                    "updated_at": "2026-05-19 13:41:12",
                    "created_at": "2026-05-19 13:41:12",
                }
            ]

        def apply_execution_report(self, payload):
            self.payloads.append(dict(payload))
            return 1

    storage = _Storage()
    changed = run_multi_leg_backfill_once(
        api=_API(),
        storage=storage,
        lookback_hours=24,
        limit=100,
    )
    assert changed == 0
    assert storage.payloads == []


def test_run_backfill_repairs_open_row_from_open_snapshot_when_get_order_empty() -> (
    None
):
    class _API:
        def get_order(self, _order_id: str, _symbol: str):
            return None

        def get_open_orders_for_sl_cleanup(self, _symbol: str):
            return [
                {
                    "order_id": "90489849398",
                    "client_order_id": "cg_16738f8fae98",
                    "status": "open",
                    "filled": 0.0,
                    "average_price": 643.0,
                    "timestamp": 1710000000,
                }
            ]

    class _Storage:
        def __init__(self) -> None:
            self.payloads = []

        def get_recent_orders_for_backfill(self, **_kwargs):
            return [
                {
                    "run_id": "mlr_1",
                    "strategy": "chop_grid",
                    "symbol": "BNBUSDT",
                    "exchange_order_id": "90489849398",
                    "client_order_id": "cg_16738f8fae98",
                    "status": "open",
                }
            ]

        def apply_execution_report(self, payload):
            self.payloads.append(dict(payload))
            return 1

    storage = _Storage()
    changed = run_multi_leg_backfill_once(
        api=_API(),
        storage=storage,
        lookback_hours=24,
        limit=100,
    )
    assert changed == 1
    assert storage.payloads[0]["status"] == "open"


def test_multi_leg_storage_backfill_candidates(tmp_path) -> None:
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    storage.upsert_order(
        {
            "local_order_id": "open_1",
            "run_id": "run",
            "strategy": "chop_grid",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "order_type": "limit",
            "quantity": 0.1,
            "exchange_order_id": "111",
            "client_order_id": "cg_111",
            "status": "open",
        }
    )
    storage.upsert_order(
        {
            "local_order_id": "filled_missing",
            "run_id": "run",
            "strategy": "chop_grid",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "order_type": "limit",
            "quantity": 0.1,
            "exchange_order_id": "222",
            "client_order_id": "cg_222",
            "status": "filled",
            "filled_quantity": 0.1,
        }
    )
    storage.upsert_order(
        {
            "local_order_id": "complete",
            "run_id": "run",
            "strategy": "chop_grid",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "order_type": "limit",
            "quantity": 0.1,
            "exchange_order_id": "333",
            "client_order_id": "cg_333",
            "status": "filled",
            "filled_quantity": 0.1,
            "average_price": 3200,
            "filled_at": "2026-05-16T00:00:00",
        }
    )
    storage.upsert_order(
        {
            "local_order_id": "canceled_no_reason",
            "run_id": "run",
            "strategy": "chop_grid",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "order_type": "limit",
            "quantity": 0.1,
            "exchange_order_id": "444",
            "client_order_id": "cg_444",
            "status": "canceled",
        }
    )

    rows = storage.get_recent_orders_for_backfill(lookback_hours=24, limit=10)
    ids = {row["local_order_id"] for row in rows}
    assert ids == {"open_1", "filled_missing"}


def test_apply_execution_report_skips_unchanged_row(tmp_path) -> None:
    storage = MultiLegStorage(str(tmp_path / "multi_leg.db"))
    storage.upsert_order(
        {
            "local_order_id": "open_1",
            "run_id": "run",
            "strategy": "chop_grid",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "order_type": "limit",
            "quantity": 0.1,
            "exchange_order_id": "111",
            "client_order_id": "cg_111",
            "status": "open",
        }
    )
    assert (
        storage.apply_execution_report(
            {
                "order_id": "111",
                "client_order_id": "cg_111",
                "status": "open",
                "filled_qty": 0.0,
            }
        )
        == 0
    )
    assert (
        storage.apply_execution_report(
            {
                "order_id": "111",
                "client_order_id": "cg_111",
                "status": "partially_filled",
                "filled_qty": 0.05,
            }
        )
        == 1
    )
    assert (
        storage.apply_execution_report(
            {
                "order_id": "111",
                "client_order_id": "cg_111",
                "status": "partially_filled",
                "filled_qty": 0.05,
            }
        )
        == 0
    )
