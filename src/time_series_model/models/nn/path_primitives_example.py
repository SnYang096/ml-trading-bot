"""
Minimal usage example for the multi-head path primitives MLP.

This module is intentionally importable (not a CLI) so you can run it from notebooks/tests
or wire it into an existing pipeline script later.

Typical usage flow:
1) You already have df_features from the feature pipeline (must include open/high/low/close/atr).
2) Choose feature_cols (the columns you want to feed to the MLP).
3) Configure horizon_bars (IMPORTANT: 80H at 4H bars => 20 bars).
4) Train and save the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .path_primitives_labels import PathPrimitivesLabelConfig
from .path_primitives_trainer import TrainConfig, train_path_primitives_mlp


def train_example(
    df_features: pd.DataFrame,
    *,
    feature_cols: List[str],
    horizon_bars: int,
    out_path: str = "models/path_primitives_mlp.pt",
) -> None:
    cfg = TrainConfig(
        label_cfg=PathPrimitivesLabelConfig(
            horizon_bars=horizon_bars,
            entry_offset=1,
            entry_price_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            atr_col="atr",
        ),
        epochs=30,
        batch_size=512,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model, meta = train_path_primitives_mlp(
        df_features,
        feature_cols=feature_cols,
        cfg=cfg,
        save_path=out_path,
    )
    print(
        f"✅ Saved model to {out_path} (n_samples={meta['n_samples']}, device={meta['device']})"
    )
