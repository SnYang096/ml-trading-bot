from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.time_series_model.models.nn.path_primitives_artifact import (
    PathPrimitivesModelArtifact,
)
from src.time_series_model.models.nn.path_primitives_reporting import (
    predict_path_primitives,
)


@dataclass(frozen=True)
class NNMHLiveInferencerConfig:
    model_path: str
    config_dir: Optional[str] = None
    device: Optional[str] = None
    fill_nan_value: float = 0.0


class NNMHLiveInferencer:
    """
    Minimal online nnmultihead inference wrapper for live.
    """

    def __init__(self, cfg: NNMHLiveInferencerConfig):
        self.cfg = cfg
        self.artifact = PathPrimitivesModelArtifact.load(
            model_path=str(cfg.model_path),
            config_dir=str(cfg.config_dir) if cfg.config_dir else None,
        )

    def preds_in_log1p(self) -> bool:
        return bool(self.artifact.preds_in_log1p(True))

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, float]:
        df = pd.DataFrame([dict(features or {})])
        if not self.artifact.feature_cols:
            raise ValueError(
                "Model artifact missing feature_cols; cannot run live inference."
            )
        preds = predict_path_primitives(
            model=self.artifact.model,
            df=df,
            feature_cols=list(self.artifact.feature_cols),
            fill_nan_value=float(self.cfg.fill_nan_value),
            block_cols_by_name=self.artifact.block_cols_by_name,
            append_block_mask=bool(self.artifact.append_block_mask),
            device=self.cfg.device,
            feature_scaler=self.artifact.feature_scaler,
        )
        row = preds.iloc[0].to_dict()
        return {str(k): float(v) for k, v in row.items()}
