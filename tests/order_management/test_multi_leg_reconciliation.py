from __future__ import annotations

from src.order_management.multi_leg_reconciliation import (
    LocalOrderSnapshot,
    LocalPositionSnapshot,
    MultiLegReconciler,
    ReconciliationPolicy,
)


def test_reports_missing_exchange_order_for_local_pending_order() -> None:
    reconciler = MultiLegReconciler()

    report = reconciler.reconcile(
        local_orders=[
            LocalOrderSnapshot(
                order_id="local_1",
                client_order_id="cg_abc",
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.01,
                price=60_000.0,
            )
        ],
        exchange_orders=[],
    )

    assert not report.ok
    assert report.missing_exchange_orders[0].client_order_id == "cg_abc"


def test_reports_and_suggests_cancel_for_orphan_exchange_order() -> None:
    reconciler = MultiLegReconciler(
        ReconciliationPolicy(
            client_id_prefixes={"dat_"}, cancel_orphan_exchange_orders=True
        )
    )

    report = reconciler.reconcile(
        local_orders=[],
        exchange_orders=[
            {
                "order_id": "ex_1",
                "client_order_id": "dat_orphan",
                "symbol": "BTCUSDT",
            },
            {
                "order_id": "ex_2",
                "client_order_id": "other_strategy",
                "symbol": "ETHUSDT",
            },
        ],
    )

    assert len(report.orphan_exchange_orders) == 1
    assert report.orphan_exchange_orders[0]["order_id"] == "ex_1"
    assert report.suggested_actions == [
        {
            "action": "cancel",
            "symbol": "BTCUSDT",
            "exchange_order_id": "ex_1",
            "reason": "orphan_exchange_order",
            "is_algo_order": False,
        }
    ]


def test_orphan_algo_order_suggest_cancel_with_algo_flag() -> None:
    reconciler = MultiLegReconciler(
        ReconciliationPolicy(
            client_id_prefixes={"cg_"}, cancel_orphan_exchange_orders=True
        )
    )

    report = reconciler.reconcile(
        local_orders=[],
        exchange_orders=[
            {
                "order_id": "2000000972847548",
                "client_order_id": "cg_orphan_tp",
                "symbol": "BNBUSDT",
                "_is_algo_order": True,
            },
        ],
    )

    assert len(report.orphan_exchange_orders) == 1
    assert report.suggested_actions == [
        {
            "action": "cancel",
            "symbol": "BNBUSDT",
            "exchange_order_id": "2000000972847548",
            "reason": "orphan_exchange_order",
            "is_algo_order": True,
        }
    ]


def test_position_mismatch_detects_exchange_inventory_drift() -> None:
    reconciler = MultiLegReconciler()

    report = reconciler.reconcile(
        local_positions=[LocalPositionSnapshot("BTCUSDT", "LONG", 0.02)],
        exchange_positions=[
            {
                "symbol": "BTCUSDT",
                "position_side": "LONG",
                "position_amount": 0.03,
            }
        ],
    )

    assert len(report.position_mismatches) == 1
    mismatch = report.position_mismatches[0]
    assert mismatch.symbol == "BTCUSDT"
    assert mismatch.side == "LONG"
    assert mismatch.local_quantity == 0.02
    assert mismatch.exchange_quantity == 0.03


def test_position_mismatch_respects_tolerance() -> None:
    reconciler = MultiLegReconciler(ReconciliationPolicy(quantity_tolerance=0.001))

    report = reconciler.reconcile(
        local_positions=[LocalPositionSnapshot("BTCUSDT", "SHORT", 0.0200)],
        exchange_positions=[
            {
                "symbol": "BTCUSDT",
                "position_side": "SHORT",
                "position_amount": 0.0205,
            }
        ],
    )

    assert report.position_mismatches == []


def test_skips_position_check_when_policy_opted_out() -> None:
    reconciler = MultiLegReconciler(
        ReconciliationPolicy(skip_position_reconciliation=True)
    )
    report = reconciler.reconcile(
        local_positions=[LocalPositionSnapshot("BTCUSDT", "LONG", 0.02)],
        exchange_positions=[
            {
                "symbol": "BTCUSDT",
                "position_side": "LONG",
                "position_amount": 0.0,
            }
        ],
    )
    assert report.position_mismatches == []
    assert report.ok


def test_matching_client_order_id_is_clean() -> None:
    reconciler = MultiLegReconciler(ReconciliationPolicy(client_id_prefixes={"cg_"}))

    report = reconciler.reconcile(
        local_orders=[
            LocalOrderSnapshot(
                order_id="local_1",
                client_order_id="cg_abc",
                symbol="BTCUSDT",
                side="SELL",
                quantity=0.01,
                price=61_000.0,
            )
        ],
        exchange_orders=[
            {
                "order_id": "ex_1",
                "client_order_id": "cg_abc",
                "symbol": "BTCUSDT",
            }
        ],
    )

    assert report.ok
