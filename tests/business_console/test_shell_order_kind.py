"""Mirror of displayOrderKind rules for CMS order type column."""

from typing import Optional


def _display_order_kind(
    order_type: str = "",
    purpose: str = "",
    *,
    stop_price: Optional[float] = None,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
) -> str:
    ot = (order_type or purpose or "").lower()
    if (
        "stop_market" in ot
        or "take_profit_market" in ot
        or "trailing_stop" in ot
        or "stop" in ot
        or "take_profit" in ot
    ):
        return "条件单"
    for key in (stop_price, stop_loss_price, take_profit_price):
        n = float(key) if key is not None else float("nan")
        if n == n and n > 0:  # finite and > 0
            return "条件单"
    if "limit" in ot or ot == "marketable_limit":
        return "限价"
    if "market" in ot:
        return "市价"
    return ot or "—"


def test_stop_market_labeled_conditional():
    assert _display_order_kind("stop_market") == "条件单"
    assert _display_order_kind("limit") == "限价"


def test_stop_price_fallback_labels_conditional():
    assert _display_order_kind("limit", stop_price=3500.0) == "条件单"
    assert _display_order_kind("", stop_loss_price=3400.0) == "条件单"
    assert _display_order_kind("limit", take_profit_price=3600.0) == "条件单"


def test_zero_stop_prices_do_not_force_conditional():
    assert _display_order_kind("limit", stop_price=0.0) == "限价"
    assert _display_order_kind("limit", stop_price=None) == "限价"
