"""Tests for risk-based position sizing.

Coverage:
  1. slot_sizing.compute_slot_size_from_risk — 宪法风险反算仓位
  2. order_flow_listener._execute_intent — 仓位计算优先级链
  3. backtest_execution_layer.compute_risk_equity_curve — 回测权益曲线
  4. live_pcm._load_constitution_constraints — 宪法加载 risk_per_slot
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Dict, Any

import pandas as pd
import pytest

from src.time_series_model.portfolio.slot_sizing import (
    compute_slot_size_from_risk,
    SlotSizingResult,
)
from src.time_series_model.core.trade_intent import TradeIntent
from scripts.backtest_execution_layer import compute_risk_equity_curve


# ──────────────────────────────────────────────────────────────────────────────
# 1. compute_slot_size_from_risk — 宪法风险反算
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeSlotSizeFromRisk:
    """Test risk-based slot sizing with 1% risk_per_slot."""

    def test_normal_1pct_risk(self):
        """equity=10000, risk=1%, price=100, atr=2, SL=1R
        risk_usd=100, stop_ret=0.02, notional=5000, qty=50
        """
        r = compute_slot_size_from_risk(
            equity_usd=10_000,
            risk_frac=0.01,
            price=100.0,
            atr=2.0,
            stop_atr=1.0,
            max_leverage=10.0,
        )
        assert r.qty == pytest.approx(50.0)
        assert r.notional_usd == pytest.approx(5000.0)
        assert r.stop_return_frac == pytest.approx(0.02)

    def test_wider_stop_reduces_qty(self):
        """SL=2R => stop_ret=0.04, notional=2500, qty=25 (更宽止损 → 更小仓位)"""
        r = compute_slot_size_from_risk(
            equity_usd=10_000,
            risk_frac=0.01,
            price=100.0,
            atr=2.0,
            stop_atr=2.0,
            max_leverage=10.0,
        )
        assert r.qty == pytest.approx(25.0)

    def test_leverage_cap(self):
        """max_leverage=0.3 cap notional to 3000 => qty=30"""
        r = compute_slot_size_from_risk(
            equity_usd=10_000,
            risk_frac=0.01,
            price=100.0,
            atr=2.0,
            stop_atr=1.0,
            max_leverage=0.3,
        )
        assert r.notional_usd == pytest.approx(3000.0)
        assert r.qty == pytest.approx(30.0)

    def test_zero_equity_returns_zero(self):
        r = compute_slot_size_from_risk(
            equity_usd=0,
            risk_frac=0.01,
            price=100.0,
            atr=2.0,
            stop_atr=1.0,
        )
        assert r.qty == 0.0

    def test_zero_atr_returns_zero(self):
        r = compute_slot_size_from_risk(
            equity_usd=10_000,
            risk_frac=0.01,
            price=100.0,
            atr=0.0,
            stop_atr=1.0,
        )
        assert r.qty == 0.0

    def test_zero_price_returns_zero(self):
        r = compute_slot_size_from_risk(
            equity_usd=10_000,
            risk_frac=0.01,
            price=0.0,
            atr=2.0,
            stop_atr=1.0,
        )
        assert r.qty == 0.0

    def test_min_qty_enforced(self):
        """min_qty 比风险反算更大时，使用 min_qty"""
        r = compute_slot_size_from_risk(
            equity_usd=10_000,
            risk_frac=0.01,
            price=100.0,
            atr=2.0,
            stop_atr=1.0,
            max_leverage=10.0,
            min_qty=100.0,
        )
        assert r.qty == pytest.approx(100.0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. _execute_intent — 仓位计算优先级链
# ──────────────────────────────────────────────────────────────────────────────


def _make_listener(**overrides):
    """创建一个最小化的 OrderFlowListener mock 用于测试 _execute_intent."""
    from src.live_data_stream.order_flow_listener import OrderFlowListener

    storage = MagicMock()
    listener = OrderFlowListener.__new__(OrderFlowListener)
    # 必要属性
    listener.symbol = "BTCUSDT"
    listener.order_manager = MagicMock()
    listener.trade_size = 0.0
    listener.risk_per_trade = 0.0
    listener.risk_per_slot = 0.0
    listener.per_strategy_limits = {}  # NEW: per-strategy risk/add config
    listener.constitution_executor = MagicMock()
    listener.runtime_state = MagicMock()
    listener.stats_collector = None
    listener.memory_window = MagicMock()
    listener.current_1min_bar = None
    listener._open_positions = {}

    for k, v in overrides.items():
        setattr(listener, k, v)

    return listener


def _make_intent(
    action: str = "LONG",
    quantity=None,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    archetype: str = "BPC",
    **kwargs,
) -> TradeIntent:
    ep = {
        "rr_constraints": {
            "stop_loss_r": stop_loss_r,
            "take_profit_r": take_profit_r,
        }
    }
    return TradeIntent(
        action=action,
        symbol="BTCUSDT",
        archetype=archetype,
        quantity=quantity,
        execution_profile=ep,
        **kwargs,
    )


def _base_features(equity=10_000, close=100.0, atr=2.0) -> Dict[str, Any]:
    return {"equity": equity, "close": close, "atr": atr}


class TestExecuteIntentPriorityChain:
    """Test 仓位计算优先级: intent.quantity > constitution risk > fixed USD > trade_size."""

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_priority_1_intent_quantity(self, mock_enforce):
        """intent.quantity 直接使用，不走风险反算"""
        listener = _make_listener(risk_per_slot=0.01)
        intent = _make_intent(quantity=0.5)
        features = _base_features()

        listener._execute_intent(intent, features)

        # 应该下了单，数量是 0.5
        call_args = listener.order_manager.place_order.call_args_list
        assert len(call_args) >= 1
        first_order = call_args[0]
        assert first_order.kwargs["quantity"] == pytest.approx(0.5)

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_priority_2_constitution_risk(self, mock_enforce):
        """equity=10000, risk=1%, price=100, atr=2, SL=1R => qty=50"""
        listener = _make_listener(risk_per_slot=0.01)
        intent = _make_intent()  # no quantity
        features = _base_features(equity=10_000, close=100.0, atr=2.0)

        listener._execute_intent(intent, features)

        call_args = listener.order_manager.place_order.call_args_list
        assert len(call_args) >= 1
        qty = call_args[0].kwargs["quantity"]
        assert qty == pytest.approx(50.0)

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_priority_3_fixed_usd_risk(self, mock_enforce):
        """无 equity 时 fallback 到 risk_per_trade (固定美元风险)"""
        listener = _make_listener(
            risk_per_slot=0.01,
            risk_per_trade=10.0,  # $10 固定风险
        )
        intent = _make_intent()
        features = _base_features(equity=0, close=100.0, atr=2.0)

        listener._execute_intent(intent, features)

        call_args = listener.order_manager.place_order.call_args_list
        assert len(call_args) >= 1
        qty = call_args[0].kwargs["quantity"]
        # risk_per_trade=10, equity=10/0.01=1000, risk_usd=10
        # stop_ret=0.02, notional=500, qty=5
        assert qty == pytest.approx(5.0)

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_priority_4_trade_size_fallback(self, mock_enforce):
        """无风险参数时 fallback 到 trade_size"""
        listener = _make_listener(
            risk_per_slot=0.0,
            risk_per_trade=0.0,
            trade_size=0.001,
        )
        intent = _make_intent()
        features = _base_features(equity=0, close=100.0, atr=2.0)

        listener._execute_intent(intent, features)

        call_args = listener.order_manager.place_order.call_args_list
        assert len(call_args) >= 1
        qty = call_args[0].kwargs["quantity"]
        assert qty == pytest.approx(0.001)

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_min_qty_fallback_to_trade_size(self, mock_enforce):
        """风险反算 qty 太小时 fallback 到 trade_size"""
        # 小 equity 导致 qty 非常小
        listener = _make_listener(
            risk_per_slot=0.01,
            trade_size=0.001,
        )
        intent = _make_intent()
        # equity=10 => risk_usd=0.1, stop_ret=0.02, notional=5, qty=0.05
        # 但 atr 很大时 qty 会很小
        # equity=1, risk_usd=0.01, notional=0.5, qty=0.005
        # 用 equity=0.05 => risk_usd=0.0005, notional=0.025, qty=0.00025 < trade_size
        features = _base_features(equity=0.05, close=100.0, atr=2.0)

        listener._execute_intent(intent, features)

        call_args = listener.order_manager.place_order.call_args_list
        assert len(call_args) >= 1
        qty = call_args[0].kwargs["quantity"]
        assert qty == pytest.approx(0.001)

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_no_trade_does_nothing(self, mock_enforce):
        """NO_TRADE intent 不下单"""
        listener = _make_listener(risk_per_slot=0.01)
        intent = _make_intent(action="NO_TRADE")
        features = _base_features()

        listener._execute_intent(intent, features)

        listener.order_manager.place_order.assert_not_called()

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_zero_everything_does_not_place_order(self, mock_enforce):
        """所有来源都是 0 时不下单"""
        listener = _make_listener(
            risk_per_slot=0.0,
            risk_per_trade=0.0,
            trade_size=0.0,
        )
        intent = _make_intent()
        features = _base_features(equity=0)

        listener._execute_intent(intent, features)

        listener.order_manager.place_order.assert_not_called()

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_missing_atr_fallback_to_trade_size(self, mock_enforce):
        """无 ATR 时风险反算跳过，fallback 到 trade_size"""
        listener = _make_listener(
            risk_per_slot=0.01,
            trade_size=0.002,
        )
        intent = _make_intent()
        features = _base_features(equity=10_000, close=100.0, atr=0.0)

        listener._execute_intent(intent, features)

        call_args = listener.order_manager.place_order.call_args_list
        assert len(call_args) >= 1
        qty = call_args[0].kwargs["quantity"]
        assert qty == pytest.approx(0.002)

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_per_strategy_risk_cap(self, mock_enforce):
        """LV 策略 max_risk_per_trade=0.005 cap 风险
        equity=10000, risk=0.5% (capped from 1%), SL=1R
        risk_usd=50, stop_ret=0.02, notional=2500, qty=25
        """
        listener = _make_listener(
            risk_per_slot=0.01,
            per_strategy_limits={
                "lv": {"max_risk_per_trade": 0.005, "allow_add_position": False},
                "bpc": {"allow_add_position": True},
            },
        )
        intent = _make_intent(archetype="LV")  # LV strategy
        features = _base_features(equity=10_000, close=100.0, atr=2.0)

        listener._execute_intent(intent, features)

        call_args = listener.order_manager.place_order.call_args_list
        assert len(call_args) >= 1
        qty = call_args[0].kwargs["quantity"]
        assert qty == pytest.approx(25.0)  # 0.5% risk, not 1%

    @patch("src.live_data_stream.order_flow_listener.enforce_before_order")
    def test_per_strategy_risk_no_cap_uses_global(self, mock_enforce):
        """BPC 策略没有 max_risk_per_trade → 使用全局 risk_per_slot
        equity=10000, risk=1%, SL=1R => qty=50
        """
        listener = _make_listener(
            risk_per_slot=0.01,
            per_strategy_limits={
                "lv": {"max_risk_per_trade": 0.005},
                "bpc": {},
            },
        )
        intent = _make_intent(archetype="BPC")
        features = _base_features(equity=10_000, close=100.0, atr=2.0)

        listener._execute_intent(intent, features)

        call_args = listener.order_manager.place_order.call_args_list
        assert len(call_args) >= 1
        qty = call_args[0].kwargs["quantity"]
        assert qty == pytest.approx(50.0)  # 1% risk (global)


# ──────────────────────────────────────────────────────────────────────────────
# 3. compute_risk_equity_curve — 回测 R-multiple 权益曲线
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeRiskEquityCurve:
    """Test backtest equity curve from R-returns."""

    def test_single_win(self):
        """一笔赢利交易: equity=1000, risk=1%, rr=2.0, SL=1R
        risk_usd=10, pnl=10*2.0/1.0=20, final=1020
        """
        r_returns = pd.Series([2.0])
        result = compute_risk_equity_curve(r_returns, initial_cash=1000.0)
        assert result["final_equity"] == pytest.approx(1020.0)
        assert result["total_return_pct"] == pytest.approx(2.0)
        assert result["max_dd"] == pytest.approx(0.0)
        assert len(result["equity_curve"]) == 2

    def test_single_loss(self):
        """一笔止损: rr=-1.0, SL=1R => pnl=-10, final=990"""
        r_returns = pd.Series([-1.0])
        result = compute_risk_equity_curve(r_returns, initial_cash=1000.0)
        assert result["final_equity"] == pytest.approx(990.0)
        assert result["max_dd"] == pytest.approx(10.0 / 1000.0)

    def test_compound_growth(self):
        """多笔交易复利: 每笔风险 1% of current equity"""
        # 3 笔 +2R 交易: 复利计算
        r_returns = pd.Series([2.0, 2.0, 2.0])
        result = compute_risk_equity_curve(r_returns, initial_cash=1000.0)
        # 1000 -> 1020 -> 1040.4 -> 1061.208
        assert result["final_equity"] == pytest.approx(1061.208, rel=1e-3)

    def test_empty_returns(self):
        """无交易"""
        r_returns = pd.Series([], dtype=float)
        result = compute_risk_equity_curve(r_returns, initial_cash=1000.0)
        assert result["final_equity"] == pytest.approx(1000.0)
        assert result["max_dd"] == 0.0
        assert result["equity_curve"] == []

    def test_all_nan_returns(self):
        """全部 NaN"""
        r_returns = pd.Series([float("nan"), float("nan")])
        result = compute_risk_equity_curve(r_returns, initial_cash=1000.0)
        assert result["final_equity"] == pytest.approx(1000.0)

    def test_equity_never_negative(self):
        """连续大亏，equity 不会低于 0"""
        r_returns = pd.Series([-1.0] * 200)  # 200 笔止损
        result = compute_risk_equity_curve(r_returns, initial_cash=100.0)
        assert result["final_equity"] >= 0.0

    def test_max_drawdown_tracking(self):
        """验证 max_dd 正确追踪"""
        # +2R, +2R (peak), -1R, -1R (trough)
        r_returns = pd.Series([2.0, 2.0, -1.0, -1.0])
        result = compute_risk_equity_curve(r_returns, initial_cash=1000.0)
        assert result["max_dd"] > 0.0
        # peak at ~1040.4, then drawdown
        assert result["max_dd"] < 0.05  # < 5% with 1% risk

    def test_custom_risk_per_slot(self):
        """自定义 risk_per_slot = 2%"""
        r_returns = pd.Series([2.0])
        result = compute_risk_equity_curve(
            r_returns,
            initial_cash=1000.0,
            risk_per_slot=0.02,
        )
        # risk_usd=20, pnl=40, final=1040
        assert result["final_equity"] == pytest.approx(1040.0)

    def test_custom_stop_loss_r(self):
        """自定义 stop_loss_r = 1.5R"""
        r_returns = pd.Series([3.0])  # 3R win
        result = compute_risk_equity_curve(
            r_returns,
            initial_cash=1000.0,
            stop_loss_r=1.5,
        )
        # risk_usd=10, pnl=10*3.0/1.5=20, final=1020
        assert result["final_equity"] == pytest.approx(1020.0)

    def test_per_trade_risk_series(self):
        """每笔交易不同风险: trade1=1%, trade2=0.5%"""
        r_returns = pd.Series([2.0, -1.0])
        risk_series = pd.Series([0.01, 0.005])
        result = compute_risk_equity_curve(
            r_returns,
            initial_cash=1000.0,
            risk_per_trade_series=risk_series,
        )
        # trade1: risk=1000*0.01=10, pnl=+20, equity=1020
        # trade2: risk=1020*0.005=5.1, pnl=-5.1, equity=1014.9
        assert result["final_equity"] == pytest.approx(1014.9)

    def test_per_trade_risk_series_fallback_to_global(self):
        """risk_per_trade_series 有 NaN 时 fallback 到 risk_per_slot"""
        r_returns = pd.Series([2.0, 2.0])
        risk_series = pd.Series([0.005, float("nan")])
        result = compute_risk_equity_curve(
            r_returns,
            initial_cash=1000.0,
            risk_per_slot=0.01,
            risk_per_trade_series=risk_series,
        )
        # trade1: risk=1000*0.005=5, pnl=+10, equity=1010
        # trade2: risk=1010*0.01=10.1 (fallback), pnl=+20.2, equity=1030.2
        assert result["final_equity"] == pytest.approx(1030.2)


# ──────────────────────────────────────────────────────────────────────────────
# 4. _load_constitution_constraints — 宪法加载
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadConstitutionConstraints:
    """Test constitution loading reads risk_per_slot correctly."""

    def test_load_from_real_file(self, tmp_path):
        """从 YAML 文件加载 risk_per_slot"""
        from src.time_series_model.portfolio.live_pcm import (
            _load_constitution_constraints,
        )

        yaml_content = """\
slots:
  enabled: true
  slot_count: 2
  risk_per_slot: 0.01

resource_allocation:
  per_strategy_limits:
    bpc:
      max_slots: 2
      allow_add_position: true
    lv:
      max_slots: 1
      max_risk_per_trade: 0.005
      allow_add_position: false
"""
        yaml_file = tmp_path / "constitution.yaml"
        yaml_file.write_text(yaml_content)

        result = _load_constitution_constraints(str(yaml_file))
        assert result["slot_count"] == 2
        assert result["risk_per_slot"] == pytest.approx(0.01)
        assert result["per_strategy_limits"]["lv"]["max_risk_per_trade"] == 0.005
        assert "max_risk_per_trade" not in result["per_strategy_limits"]["bpc"]

    def test_default_when_no_file(self):
        """无文件时使用默认值"""
        from src.time_series_model.portfolio.live_pcm import (
            _load_constitution_constraints,
        )

        result = _load_constitution_constraints(None)
        assert result["risk_per_slot"] == pytest.approx(0.01)

    def test_default_when_file_missing(self, tmp_path):
        """文件不存在时使用默认值"""
        from src.time_series_model.portfolio.live_pcm import (
            _load_constitution_constraints,
        )

        result = _load_constitution_constraints(str(tmp_path / "nonexist.yaml"))
        assert result["risk_per_slot"] == pytest.approx(0.01)

    def test_load_add_position_rules(self, tmp_path):
        """从新结构 add_position_rules 加载"""
        from src.time_series_model.portfolio.live_pcm import (
            _load_constitution_constraints,
        )

        yaml_content = """\
slots:
  risk_per_slot: 0.01
resource_allocation:
  per_strategy_limits:
    bpc:
      allow_add_position: true
    fer:
      allow_add_position: false
  add_position_rules:
    max_add_times: 1
    require_locked_profit: true
    lock_profit_breakeven_trigger_r: 1.0
"""
        yaml_file = tmp_path / "constitution.yaml"
        yaml_file.write_text(yaml_content)
        result = _load_constitution_constraints(str(yaml_file))
        assert result["add_position_rules"]["max_add_times"] == 1
        assert result["per_strategy_limits"]["bpc"]["allow_add_position"] is True
        assert result["per_strategy_limits"]["fer"]["allow_add_position"] is False


# ──────────────────────────────────────────────────────────────────────────────
# 5. ConstitutionExecutor — 按策略加仓控制 + 风险解析
# ──────────────────────────────────────────────────────────────────────────────


class TestConstitutionExecutorPerStrategy:
    """Test validate_add_position per-strategy and resolve_risk_for_strategy."""

    def _make_executor(self, tmp_path, yaml_content: str):
        from src.time_series_model.core.constitution.constitution_executor import (
            ConstitutionExecutor,
        )

        yaml_file = tmp_path / "constitution.yaml"
        yaml_file.write_text(yaml_content)
        return ConstitutionExecutor(constitution_yaml=str(yaml_file))

    _YAML = """\
version: 1
name: "C_TEST"
kill_switch:
  enabled: true
  max_dd: 0.5
  daily_loss_limit: 1.0
  weekly_loss_limit: 1.0
  monthly_loss_limit: 1.0
  kill_on_any_hard_violation: true
slots:
  enabled: true
  slot_count: 2
  risk_per_slot: 0.01
resource_allocation:
  per_strategy_limits:
    bpc:
      max_slots: 2
      allow_add_position: true
    lv:
      max_slots: 1
      max_risk_per_trade: 0.005
      allow_add_position: false
    fer:
      max_slots: 2
      allow_add_position: false
  add_position_rules:
    max_add_times: 1
    require_locked_profit: true
    lock_profit_breakeven_trigger_r: 1.0
"""

    def test_resolve_risk_bpc_inherits_global(self, tmp_path):
        """BPC 没有 max_risk_per_trade → 返回 risk_per_slot 0.01"""
        ex = self._make_executor(tmp_path, self._YAML)
        assert ex.resolve_risk_for_strategy("bpc") == pytest.approx(0.01)

    def test_resolve_risk_lv_capped(self, tmp_path):
        """LV 有 max_risk_per_trade=0.005 → 返回 0.005"""
        ex = self._make_executor(tmp_path, self._YAML)
        assert ex.resolve_risk_for_strategy("lv") == pytest.approx(0.005)

    def test_resolve_risk_unknown_strategy(self, tmp_path):
        """未知策略 → 返回 risk_per_slot"""
        ex = self._make_executor(tmp_path, self._YAML)
        assert ex.resolve_risk_for_strategy("xyz") == pytest.approx(0.01)

    def test_validate_add_position_bpc_allowed(self, tmp_path):
        """BPC allow_add_position=true → 不抛异常"""
        from src.time_series_model.core.constitution.runtime_state import (
            ConstitutionRuntimeState,
        )

        ex = self._make_executor(tmp_path, self._YAML)
        st = ConstitutionRuntimeState()
        # Should not raise (BPC is allowed, has locked profit)
        ex.validate_add_position(
            st=st,
            position_id="p1",
            archetype="bpc",
            current_r=1.5,
            locked_profit=True,
        )

    def test_validate_add_position_lv_forbidden(self, tmp_path):
        """LV allow_add_position=false → 抛 ConstitutionViolation"""
        from src.time_series_model.core.constitution.runtime_state import (
            ConstitutionRuntimeState,
        )
        from src.time_series_model.core.constitution.violation import (
            ConstitutionViolation,
        )

        ex = self._make_executor(tmp_path, self._YAML)
        st = ConstitutionRuntimeState()
        with pytest.raises(ConstitutionViolation) as exc_info:
            ex.validate_add_position(
                st=st,
                position_id="p1",
                archetype="lv",
                current_r=2.0,
                locked_profit=True,
            )
        assert exc_info.value.code == "ADD_POSITION_STRATEGY_FORBIDDEN"

    def test_validate_add_position_fer_forbidden(self, tmp_path):
        """FER allow_add_position=false → 抛 ConstitutionViolation"""
        from src.time_series_model.core.constitution.runtime_state import (
            ConstitutionRuntimeState,
        )
        from src.time_series_model.core.constitution.violation import (
            ConstitutionViolation,
        )

        ex = self._make_executor(tmp_path, self._YAML)
        st = ConstitutionRuntimeState()
        with pytest.raises(ConstitutionViolation) as exc_info:
            ex.validate_add_position(
                st=st,
                position_id="p1",
                archetype="fer",
                current_r=2.0,
                locked_profit=True,
            )
        assert exc_info.value.code == "ADD_POSITION_STRATEGY_FORBIDDEN"

    def test_validate_add_position_global_max_add_times(self, tmp_path):
        """BPC 允许加仓但超过 max_add_times → 抛异常"""
        from src.time_series_model.core.constitution.runtime_state import (
            ConstitutionRuntimeState,
            AddPositionRecord,
        )
        from src.time_series_model.core.constitution.violation import (
            ConstitutionViolation,
        )

        ex = self._make_executor(tmp_path, self._YAML)
        st = ConstitutionRuntimeState()
        st.add_position.positions["p1"] = AddPositionRecord(
            position_id="p1",
            add_count=1,
        )
        with pytest.raises(ConstitutionViolation) as exc_info:
            ex.validate_add_position(
                st=st,
                position_id="p1",
                archetype="bpc",
                current_r=2.0,
                locked_profit=True,
            )
        assert exc_info.value.code == "ADD_POSITION_MAX_TIMES"

    def test_validate_add_position_bare_bpc_matches_directional_limits(self, tmp_path):
        """constitution 仅有 bpc-long 时，家族名 bpc + LONG 应命中加仓上限（与实盘 intent 一致）。"""
        from src.time_series_model.core.constitution.runtime_state import (
            AddPositionRecord,
            ConstitutionRuntimeState,
        )

        yaml_content = """\
version: 1
name: "C_DIR"
kill_switch:
  enabled: false
slots:
  risk_per_slot: 0.01
resource_allocation:
  per_strategy_limits:
    bpc-long:
      allow_add_position: true
      max_add_times: 3
"""
        ex = self._make_executor(tmp_path, yaml_content)
        st = ConstitutionRuntimeState()
        st.add_position.positions["p1"] = AddPositionRecord(
            position_id="p1",
            add_count=1,
        )
        ex.validate_add_position(
            st=st,
            position_id="p1",
            archetype="bpc",
            current_r=2.0,
            locked_profit=True,
            position_action="LONG",
        )
        assert (
            ex.resolve_add_position_for_strategy("bpc", position_action="LONG")[
                "max_add_times"
            ]
            == 3
        )
