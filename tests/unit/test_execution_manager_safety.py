import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.live.execution_manager import (
    ExecutionManager,
    GuardedOrderContext,
)
from src.order_management.storage import Storage


@pytest.mark.unit
def test_execution_manager_blocks_on_safety_halt(tmp_path):
    """Test that ExecutionManager blocks orders when Safety is halted."""
    # Use unique database path for this test and ensure it's clean
    import uuid

    test_db_path = tmp_path / f"test_safety_halt_{uuid.uuid4().hex[:8]}.db"
    cy = tmp_path / "constitution.yaml"
    cy.write_text(
        f"""
version: 1
name: "C_TEST"
kill_switch:
  enabled: true
  max_dd: 0.2
  daily_loss_limit: 0.04
  weekly_loss_limit: 0.08
  monthly_loss_limit: 0.12
  max_turnover_mean: 0.35
  max_cost_mean: 0.002
  kill_on_any_hard_violation: true
  safety_state:
    persist_to: "{test_db_path.as_posix()}"
    cooldown_minutes: 60
    daily_reset_tz: "UTC"
slots:
  enabled: true
  slot_count: 10
  risk_per_slot: 0.01
  slot_state_tracking:
    persist_to: "{tmp_path.as_posix()}/state/slots.json"
add_position:
  enabled: false
replacement_policy:
  enabled: false
capital_escalation:
  enabled: false
""",
        encoding="utf-8",
    )

    executor = ConstitutionExecutor(constitution_yaml=str(cy))
    runtime_state = executor.load_runtime_state()
    mock_strategy = Mock()
    mock_order = Mock()

    # Ensure database is clean before test - initialize with clean state
    storage = Storage(str(test_db_path))
    storage.upsert_safety_state(
        state_id="global", payload={"halted": False, "halt_reason": []}
    )

    manager = ExecutionManager(
        strategy=mock_strategy,
        executor=executor,
        runtime_state=runtime_state,
    )

    # Normal case: should pass
    ctx_ok = GuardedOrderContext(
        position_id="p1",
        symbol="BTCUSDT",
        archetype="TREND",
        execution_strategy="TrendContinuationTC",
        equity=10000.0,
        drawdown=0.1,  # 0.1 < 0.2, should not trigger halt
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
    )
    manager.submit_order_guarded(order=mock_order, ctx=ctx_ok)
    mock_strategy.submit_order.assert_called_once_with(mock_order)

    # Reset mock
    mock_strategy.reset_mock()

    # Safety halt case: should raise ConstitutionViolation
    ctx_halted = GuardedOrderContext(
        position_id="p2",
        symbol="ETHUSDT",
        archetype="MEAN",
        execution_strategy="FailureReversionFR",
        equity=10000.0,
        drawdown=0.25,  # Exceeds 0.2 limit
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
    )
    with pytest.raises(ConstitutionViolation) as exc_info:
        manager.submit_order_guarded(order=mock_order, ctx=ctx_halted)
    assert "KILL_SWITCH" in str(exc_info.value.code) or "max_dd" in str(
        exc_info.value.message
    )
    mock_strategy.submit_order.assert_not_called()


@pytest.mark.unit
def test_execution_manager_passes_daily_cost_turnover(tmp_path):
    """Test that ExecutionManager correctly passes daily_cost_mean and daily_turnover_mean."""
    # Use unique database path for this test and ensure it's clean
    import uuid

    test_db_path = tmp_path / f"test_cost_turnover_{uuid.uuid4().hex[:8]}.db"
    cy = tmp_path / "constitution.yaml"
    cy.write_text(
        f"""
version: 1
name: "C_TEST"
kill_switch:
  enabled: true
  max_dd: 0.2
  daily_loss_limit: 0.04
  weekly_loss_limit: 0.08
  monthly_loss_limit: 0.12
  max_turnover_mean: 0.35
  max_cost_mean: 0.002
  kill_on_any_hard_violation: true
  safety_state:
    persist_to: "{test_db_path.as_posix()}"
    cooldown_minutes: 60
    daily_reset_tz: "UTC"
slots:
  enabled: true
  slot_count: 10
  risk_per_slot: 0.01
  slot_state_tracking:
    persist_to: "{tmp_path.as_posix()}/state/slots.json"
add_position:
  enabled: false
replacement_policy:
  enabled: false
capital_escalation:
  enabled: false
""",
        encoding="utf-8",
    )

    executor = ConstitutionExecutor(constitution_yaml=str(cy))
    runtime_state = executor.load_runtime_state()
    mock_strategy = Mock()
    mock_order = Mock()

    # Ensure database is clean before test - initialize with clean state
    storage = Storage(str(test_db_path))
    storage.upsert_safety_state(
        state_id="global", payload={"halted": False, "halt_reason": []}
    )

    manager = ExecutionManager(
        strategy=mock_strategy,
        executor=executor,
        runtime_state=runtime_state,
    )

    # Test with daily_cost_mean and daily_turnover_mean
    ctx_with_metrics = GuardedOrderContext(
        position_id="p1",
        symbol="BTCUSDT",
        archetype="TREND",
        execution_strategy="TrendContinuationTC",
        equity=10000.0,
        drawdown=0.1,  # 0.1 < 0.2, should not trigger halt
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        daily_cost_mean=0.001,  # 0.001 < 0.002, should not trigger halt
        daily_turnover_mean=0.2,  # 0.2 < 0.35, should not trigger halt
    )
    manager.submit_order_guarded(order=mock_order, ctx=ctx_with_metrics)
    mock_strategy.submit_order.assert_called_once_with(mock_order)

    # Reset mock
    mock_strategy.reset_mock()

    # Test with excessive daily_cost_mean -> should halt
    ctx_cost_violation = GuardedOrderContext(
        position_id="p2",
        symbol="ETHUSDT",
        archetype="MEAN",
        execution_strategy="FailureReversionFR",
        equity=10000.0,
        drawdown=0.1,
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        daily_cost_mean=0.003,  # Exceeds 0.002 limit
        daily_turnover_mean=0.2,
    )
    with pytest.raises(ConstitutionViolation):
        manager.submit_order_guarded(order=mock_order, ctx=ctx_cost_violation)
    mock_strategy.submit_order.assert_not_called()


@pytest.mark.unit
def test_execution_manager_passes_evt_risk_flag(tmp_path):
    """Test that ExecutionManager correctly passes evt_risk_flag (soft alert only)."""
    # Use unique database path for this test and ensure it's clean
    import uuid

    test_db_path = tmp_path / f"test_evt_risk_{uuid.uuid4().hex[:8]}.db"
    cy = tmp_path / "constitution.yaml"
    cy.write_text(
        f"""
version: 1
name: "C_TEST"
kill_switch:
  enabled: true
  max_dd: 0.2
  daily_loss_limit: 0.04
  weekly_loss_limit: 0.08
  monthly_loss_limit: 0.12
  max_turnover_mean: 0.35
  max_cost_mean: 0.002
  kill_on_any_hard_violation: true
  safety_state:
    persist_to: "{test_db_path.as_posix()}"
    cooldown_minutes: 60
    daily_reset_tz: "UTC"
slots:
  enabled: true
  slot_count: 10
  risk_per_slot: 0.01
  slot_state_tracking:
    persist_to: "{tmp_path.as_posix()}/state/slots.json"
add_position:
  enabled: false
replacement_policy:
  enabled: false
capital_escalation:
  enabled: false
""",
        encoding="utf-8",
    )

    executor = ConstitutionExecutor(constitution_yaml=str(cy))
    runtime_state = executor.load_runtime_state()
    mock_strategy = Mock()
    mock_order = Mock()

    # Ensure database is clean before test - initialize with clean state
    storage = Storage(str(test_db_path))
    storage.upsert_safety_state(
        state_id="global", payload={"halted": False, "halt_reason": []}
    )

    manager = ExecutionManager(
        strategy=mock_strategy,
        executor=executor,
        runtime_state=runtime_state,
    )

    # EVT risk flag should not block (soft alert only)
    ctx_evt_risk = GuardedOrderContext(
        position_id="p1",
        symbol="BTCUSDT",
        archetype="TREND",
        execution_strategy="TrendContinuationTC",
        equity=10000.0,
        drawdown=0.1,  # 0.1 < 0.2, should not trigger halt
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        evt_risk_flag=True,  # Soft alert, should not block
    )
    manager.submit_order_guarded(order=mock_order, ctx=ctx_evt_risk)
    mock_strategy.submit_order.assert_called_once_with(mock_order)
