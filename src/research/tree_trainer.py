"""LightGBM research trainer (layer-agnostic)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    model_path: Path
    feature_cols: List[str]
    metrics: Dict[str, Any]


def train_lightgbm_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    output_dir: Path,
    *,
    seed: int = 42,
) -> FitResult:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    output_dir.mkdir(parents=True, exist_ok=True)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=False
    )
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": seed,
        "num_leaves": 31,
        "learning_rate": 0.05,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=200,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    model_path = output_dir / "model.txt"
    booster.save_model(str(model_path))
    pred = booster.predict(X_val)
    auc = float(
        __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
            y_val, pred
        )
    )
    metrics = {"val_auc": auc, "n_train": len(X_train), "n_val": len(X_val)}
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return FitResult(model_path=model_path, feature_cols=feature_cols, metrics=metrics)
