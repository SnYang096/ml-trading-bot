import numpy as np

from src.time_series_model.rl.bc_dataset import BCStateSchema, Router3ActionInferConfig
from src.time_series_model.rl.bc_trainer_3action import (
    BC3TrainConfig,
    train_bc_router3_policy,
)


def test_bc_trainer_3action_smoke() -> None:
    rng = np.random.default_rng(0)
    transitions = []
    for i in range(300):
        # make a learnable pattern: positive dir -> TREND, negative dir -> MEAN, low mfe -> NO_TRADE
        dir_score = float(rng.normal())
        mfe = float(abs(rng.normal()))
        state = {"head_dir_score": dir_score, "head_mfe_atr": mfe, "drawdown": 0.0}
        if mfe < 0.3:
            action = {"mode": "NO_TRADE"}
        elif dir_score > 0:
            action = {"mode": "TREND"}
        else:
            action = {"mode": "MEAN"}
        transitions.append({"state": state, "action": action})

    state_schema = BCStateSchema(keys=["head_dir_score", "head_mfe_atr", "drawdown"])
    # In the new system, mode is logged directly; router name grouping is unused but still required by signature.
    infer_cfg = Router3ActionInferConfig(mean_routers=[], trend_routers=[])

    model, meta = train_bc_router3_policy(
        transitions=transitions,
        state_schema=state_schema,
        infer_cfg=infer_cfg,
        cfg=BC3TrainConfig(
            epochs=2, batch_size=64, hidden=32, depth=2, dropout=0.0, device="cpu"
        ),
    )

    assert meta["n_samples"] == len(transitions)
    assert "history" in meta
    assert model is not None
