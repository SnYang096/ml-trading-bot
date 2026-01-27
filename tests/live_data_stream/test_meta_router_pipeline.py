import pandas as pd
import pytest

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.meta_router_core import (
    MetaRouterCore,
    MetaRouterCoreConfig,
)
from src.live_data_stream.order_flow_listener import OrderFlowListener


class DummyOrderManager:
    def __init__(self):
        self.calls = []

    def place_order(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


@pytest.mark.unit
def test_meta_router_pipeline_places_order(tmp_path):
    cy = tmp_path / "constitution.yaml"
    db_path = tmp_path / "om.db"
    cy.write_text(
        f"""
version: 1
name: "C_PIPE"
kill_switch:
  enabled: true
  max_dd: 0.5
  daily_loss_limit: 1.0
  weekly_loss_limit: 1.0
  monthly_loss_limit: 1.0
  max_turnover_mean: 0.35
  max_cost_mean: 0.002
  kill_on_any_hard_violation: true
safety_state:
  persist_to: "{db_path.as_posix()}"
slots:
  enabled: true
  slot_count: 1
  risk_per_slot: 0.01
  slot_state_tracking:
    persist_to: "{db_path.as_posix()}"
add_position:
  enabled: false
capital_escalation:
  enabled: false
""",
        encoding="utf-8",
    )

    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ex.load_runtime_state()
    core = MetaRouterCore(
        MetaRouterCoreConfig(
            enabled_archetypes={
                "TREND": ["BreakoutPullbackContinuation"],
                "MEAN": ["FailureReversionFR"],
                "NO_TRADE": [],
            },
            gate_enabled=True,
            gate_fail_open_missing_quantiles=True,
        )
    )
    om = DummyOrderManager()

    listener = OrderFlowListener(
        symbol="BTCUSDT",
        storage_manager=None,  # not used for this test
        feature_computer=None,
        meta_router_core=core,
        constitution_executor=ex,
        runtime_state=st,
        order_manager=om,
        trade_size=1.0,
    )

    ts = pd.Timestamp.now(tz="UTC")
    for idx in range(6):
        price = 100.0 + idx * 0.5
        listener.memory_window.add(
            {
                "timestamp": ts - pd.Timedelta(minutes=6 - idx),
                "open": price - 0.2,
                "high": price + 0.5,
                "low": price - 0.4,
                "close": price,
            }
        )

    listener._handle_features(
        {
            "pred_dir_prob": 0.72,
            "pred_mfe_atr": 1.2,
            "pred_mae_atr": 0.4,
            "pred_t_to_mfe": 10.0,
            "price_dir_consistency_pct": 0.9,
        }
    )

    assert len(om.calls) == 1
    assert st.slots.active_count() == 1

    with pytest.raises(Exception):
        listener._handle_features(
            {
                "pred_dir_prob": 0.72,
                "pred_mfe_atr": 1.2,
                "pred_mae_atr": 0.4,
                "pred_t_to_mfe": 10.0,
                "price_dir_consistency_pct": 0.9,
            }
        )
