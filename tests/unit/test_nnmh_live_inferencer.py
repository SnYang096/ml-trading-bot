from pathlib import Path

import pytest
import torch

from src.time_series_model.live.nnmh_live_inferencer import (
    NNMHLiveInferencer,
    NNMHLiveInferencerConfig,
)
from src.time_series_model.models.nn.path_primitives_model import (
    MultiHeadPathPrimitivesMLP,
    PathPrimitivesModelConfig,
)


@pytest.mark.unit
def test_nnmh_live_inferencer_can_load_and_predict(tmp_path: Path):
    # Build a tiny synthetic model artifact
    model = MultiHeadPathPrimitivesMLP(
        cfg=PathPrimitivesModelConfig(d_in=2, hidden=8, depth=1, dropout=0.0)
    )
    payload = {
        "model": model.export_state(),
        "meta": {
            "feature_cols": ["f1", "f2"],
            "dataset_cfg": {"log1p_targets": True},
        },
    }
    p = tmp_path / "model.pt"
    torch.save(payload, p)

    inf = NNMHLiveInferencer(
        NNMHLiveInferencerConfig(model_path=str(p), config_dir=None, device="cpu")
    )
    out = inf.predict_one({"f1": 1.0, "f2": -2.0})

    assert "pred_dir_prob" in out
    assert "pred_mfe_atr" in out
    assert "pred_mae_atr" in out
    assert "pred_t_to_mfe" in out
    assert 0.0 <= out["pred_dir_prob"] <= 1.0
    assert out["pred_mfe_atr"] >= 0.0
    assert out["pred_mae_atr"] >= 0.0
    assert out["pred_t_to_mfe"] >= 0.0
