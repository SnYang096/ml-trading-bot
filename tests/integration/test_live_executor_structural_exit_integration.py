"""模拟实盘链路：TradeExecutor → PositionTracker → enforce_position（结构出场 + 加仓继承）。

说明：
- 使用仓库内真实 constitution.yaml 校验加仓宪法逻辑（validate/record_add_position）。
- patch enforce_before_order：避免集成测试依赖 kill-switch / slot 落盘等副作用。
- patch ConstitutionExecutor.save_runtime_state：不写磁盘状态文件。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.order_management.position_tracker import PositionTracker
from src.order_management.trade_executor import TradeExecutor
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
)
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSTITUTION_YAML = PROJECT_ROOT / "config" / "constitution" / "constitution.yaml"


def _mock_order(oid: str = "ORD1") -> MagicMock:
    o = MagicMock()
    o.order_id = oid
    return o


def _features(
    *,
    close: float = 100.0,
    atr: float = 1.0,
    equity: float = 10_000.0,
    macro_pv: float | None = None,
) -> dict:
    out = {
        "close": close,
        "240T_atr": atr,
        "equity": equity,
        "timestamp": datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
    }
    if macro_pv is not None:
        out["macro_tp_vwap_1200_position"] = macro_pv
    return out


@pytest.mark.integration
@pytest.mark.skipif(
    not CONSTITUTION_YAML.is_file(),
    reason="constitution.yaml not found",
)
def test_live_like_parent_add_child_inherits_structural_exit_then_vwap_exit() -> None:
    """父仓带 vwap1200 → 加仓 intent 无 rr_constraints.structural_exit 仍继承 → enforce 触发结构平。"""
    ce = ConstitutionExecutor(constitution_yaml=str(CONSTITUTION_YAML))
    ce.save_runtime_state = lambda _st: None  # type: ignore[method-assign]

    rs = ConstitutionRuntimeState()
    om = MagicMock()
    om.place_order.return_value = _mock_order()

    pt = PositionTracker(order_manager=om, symbol="BTCUSDT", default_bar_minutes=240)
    ex = TradeExecutor(
        order_manager=om,
        constitution_executor=ce,
        runtime_state=rs,
        position_tracker=pt,
        symbol="BTCUSDT",
        bar_minutes=240,
        risk_per_slot=0.01,
        trade_size=0.01,
    )

    parent_pid = "BTCUSDT:parent-live-int"
    parent_ep = {
        "rr_constraints": {
            "stop_loss_r": 2.0,
            "take_profit_r": 0.0,
            "structural_exit": "vwap1200",
        },
    }
    parent_intent = TradeIntent(
        action="LONG",
        symbol="BTCUSDT",
        archetype="bpc-long-120T",
        position_id=parent_pid,
        execution_profile=parent_ep,
    )

    with patch("src.order_management.trade_executor.enforce_before_order"):
        assert (
            ex.execute(parent_intent, features=_features(close=100.0, atr=1.0)) is True
        )

    parent_pos = pt.get(parent_pid)
    assert parent_pos is not None
    assert parent_pos.get("structural_exit") == "vwap1200"

    add_intent = TradeIntent(
        action="LONG",
        symbol="BTCUSDT",
        archetype="bpc-long-120T",
        position_id="BTCUSDT:add-live-int",
        add_position=True,
        execution_profile={
            "rr_constraints": {"stop_loss_r": 2.0, "take_profit_r": 0.0},
            "add_position": {
                "trigger": {"type": "float_r_ladder_only"},
            },
        },
    )

    with patch("src.order_management.trade_executor.enforce_before_order"):
        assert ex.execute(add_intent, features=_features(close=100.5, atr=1.0)) is True

    add_pos = pt.get("BTCUSDT:add-live-int")
    assert add_pos is not None
    assert add_pos.get("_is_add_position") is True
    assert add_pos.get("_parent_pid") == parent_pid
    assert add_pos.get("structural_exit") == "vwap1200"

    rec = rs.add_position.positions.get(parent_pid)
    assert rec is not None
    assert rec.add_count == 1

    # LONG + vwap1200：macro_tp_vwap_1200_position 为负的 pv → 触发结构出场
    closed = pt.enforce_all(_features(close=100.0, atr=1.0, macro_pv=-0.02))
    assert parent_pid in closed
    assert "BTCUSDT:add-live-int" in closed
    assert len(pt) == 0


@pytest.mark.integration
def test_structural_exit_mode_comes_from_rr_constraints_not_archetype_name() -> None:
    """structural_exit 由 execution_profile.rr_constraints（及 YAML 合并结果）决定，与 archetype 字符串无硬编码绑定。"""
    now = datetime.now(timezone.utc)
    arch = "same-archetype-placeholder"

    p_vwap = build_position_dict(
        TradeIntent(
            action="LONG",
            symbol="X",
            archetype=arch,
            execution_profile={
                "rr_constraints": {
                    "stop_loss_r": 2.0,
                    "structural_exit": "vwap1200",
                },
            },
        ),
        entry_price=100.0,
        atr=1.0,
        bar_minutes=240,
        entry_time=now,
    )
    p_ema = build_position_dict(
        TradeIntent(
            action="LONG",
            symbol="X",
            archetype=arch,
            execution_profile={
                "rr_constraints": {
                    "stop_loss_r": 2.0,
                    "structural_exit": "ema200",
                },
            },
        ),
        entry_price=100.0,
        atr=1.0,
        bar_minutes=240,
        entry_time=now,
    )
    assert p_vwap.get("structural_exit") == "vwap1200"
    assert p_ema.get("structural_exit") == "ema200"

    r_v, _ = enforce_position(
        p_vwap,
        price_high=100.0,
        price_low=100.0,
        price_close=100.0,
        now=now,
        macro_tp_vwap_position=-0.01,
    )
    assert r_v == "structural_exit_vwap1200"

    r_e, _ = enforce_position(
        p_ema,
        price_high=99.0,
        price_low=99.0,
        price_close=99.0,
        now=now,
        structural_price=100.0,
    )
    assert r_e == "structural_exit_ema200"
