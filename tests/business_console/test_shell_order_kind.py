"""Mirror of displayOrderKind rules for stop_market labeling."""


def _display_order_kind(order_type: str, purpose: str = "") -> str:
    ot = (order_type or purpose or "").lower()
    if "stop_market" in ot or "take_profit_market" in ot:
        return "条件单"
    if "limit" in ot:
        return "限价"
    if "market" in ot:
        return "市价"
    return ot or "—"


def test_stop_market_labeled_conditional():
    assert _display_order_kind("stop_market") == "条件单"
    assert _display_order_kind("limit") == "限价"
