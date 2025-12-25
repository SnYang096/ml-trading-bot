import numpy as np

from src.time_series_model.rl.bc_dataset import (
    BCDataset,
    BCPolicySchema,
    BCStateSchema,
    Router3Action,
    Router3ActionInferConfig,
    infer_router3_action,
)


def test_bc_dataset_shapes_and_decode() -> None:
    transitions = [
        {
            "state": {"head_dir_score": 0.1, "head_mfe_atr": 1.0, "drawdown": 0.0},
            "action": {
                "global_pause": False,
                "router_enabled": {"sr": True, "trend": True},
                "capital_multiplier": {"sr": 1.0, "trend": 0.5},
            },
        },
        {
            "state": {"head_dir_score": -0.2, "head_mfe_atr": 0.7, "drawdown": 0.2},
            "action": {
                "global_pause": True,
                "router_enabled": {"sr": False, "trend": True},
                "capital_multiplier": {"sr": 0.0, "trend": 1.5},
            },
        },
    ]

    state_schema = BCStateSchema(keys=["head_dir_score", "head_mfe_atr", "drawdown"])
    policy_schema = BCPolicySchema(
        router_names=["sr", "trend"], min_mult=0.0, max_mult=2.0
    )

    ds = BCDataset(
        transitions=transitions, state_schema=state_schema, policy_schema=policy_schema
    )
    assert len(ds) == 2

    item0 = ds[0]
    assert tuple(item0["x"].shape) == (3,)
    assert tuple(item0["y"].shape) == (policy_schema.action_dim,)

    # decode roundtrip sanity
    y0 = item0["y"].numpy()
    act0 = policy_schema.decode_action(y0)
    assert act0["global_pause"] is False
    assert act0["router_enabled"]["sr"] is True
    assert np.isfinite(act0["capital_multiplier"]["trend"])


def test_infer_router3_action() -> None:
    cfg = Router3ActionInferConfig(mean_routers=["sr"], trend_routers=["trend"])

    a0 = {
        "global_pause": True,
        "router_enabled": {"sr": True},
        "capital_multiplier": {"sr": 1.0},
    }
    assert infer_router3_action(a0, cfg=cfg) == Router3Action.NO_TRADE

    a1 = {
        "global_pause": False,
        "router_enabled": {"sr": True},
        "capital_multiplier": {"sr": 1.0},
    }
    assert infer_router3_action(a1, cfg=cfg) == Router3Action.MEAN

    a2 = {
        "global_pause": False,
        "router_enabled": {"trend": True},
        "capital_multiplier": {"trend": 1.0},
    }
    assert infer_router3_action(a2, cfg=cfg) == Router3Action.TREND

    # New-system direct mode label: no grouping required
    a3 = {"mode": "MEAN"}
    assert infer_router3_action(a3, cfg=cfg) == Router3Action.MEAN
