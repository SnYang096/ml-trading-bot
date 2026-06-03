#!/usr/bin/env python3
"""Predict tree scores from a frozen ModelArtifact using saved feature_config."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.tree_holdout_tau_rr_scan import (  # noqa: E402
    _ensure_atr,
    _inject_feature_freq,
)
from scripts.train_strategy_pipeline import (  # noqa: E402
    _ensure_ticks_configured,
    generate_predictions,
    run_feature_pipeline,
)
from src.data_tools.data_handler import DataHandler  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategies.models.model_artifact import (
    ModelArtifact,
)  # noqa: E402
from src.time_series_model.strategy_config.loader import (  # noqa: E402
    FeaturePipelineConfig,
    StrategyConfigLoader,
)


DEFAULT_GATE_COLS = [
    "atr",
    "trend_confidence",
    "bpc_semantic_chop_ts_q",
    "me_accel_5k",
    "vol_accel",
    "macro_tp_vwap_1200_position",
]

# Gate model features are NOT IC-pruned direction-model columns — they are a fixed
# orthogonal set chosen for adverse excursion prediction (see train_tree_adverse_gate.py).
# Map inject-column names → tier nodes to supplement artifact feature pipelines.
GATE_FEATURE_PIPELINE_NODES: dict[str, list[str]] = {
    "trend_confidence": ["trend_confidence_f"],
    "bpc_semantic_chop_ts_q": ["bpc_soft_phase_f"],
    "me_accel_5k": ["me_accel_5k_split_f"],
    "vol_accel": ["extended_volatility_features_f"],
    "macro_tp_vwap_1200_position": ["macro_tp_vwap_1200_position_f"],
}


def supplement_pipeline_for_gate_features(
    pipeline_cfg: FeaturePipelineConfig,
    gate_feature_names: list[str] | None = None,
) -> FeaturePipelineConfig:
    """Union gate tier nodes into pipeline so event inject parquet has gate inputs."""
    names = list(gate_feature_names or DEFAULT_GATE_COLS)
    extra_nodes: list[str] = []
    for name in names:
        if name == "atr":
            continue
        extra_nodes.extend(GATE_FEATURE_PIPELINE_NODES.get(name, []))
    if not extra_nodes:
        return pipeline_cfg
    merged = list(
        dict.fromkeys(list(pipeline_cfg.requested_features or []) + extra_nodes)
    )
    return FeaturePipelineConfig(
        requested_features=merged,
        forbidden_requested_features=list(
            pipeline_cfg.forbidden_requested_features or []
        ),
        invert_features=list(pipeline_cfg.invert_features or []),
        exclude_columns=list(pipeline_cfg.exclude_columns or []),
        post_processors=list(pipeline_cfg.post_processors or []),
        selector=pipeline_cfg.selector,
        ensure_signal=pipeline_cfg.ensure_signal,
    )


def pipeline_cfg_from_artifact(artifact: ModelArtifact) -> FeaturePipelineConfig:
    raw = dict(artifact.feature_config or {})
    if not raw.get("requested_features"):
        raise ValueError(
            "artifact missing feature_config.requested_features; "
            "cannot reproduce training feature pipeline"
        )
    return FeaturePipelineConfig(
        requested_features=list(raw.get("requested_features") or []),
        forbidden_requested_features=list(
            raw.get("forbidden_requested_features") or []
        ),
        invert_features=list(raw.get("invert_features") or []),
        exclude_columns=list(raw.get("exclude_columns") or []),
        post_processors=[],
        selector=None,
        ensure_signal=raw.get("ensure_signal"),
    )


def predict_from_artifact(
    *,
    artifact_dir: Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
    data_path: str = "data/parquet_data",
    timeframe: str = "120T",
    feature_store_layer: str = "features",
    config_dir: Path | None = None,
    include_gate_features: bool = False,
    gate_feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Run inference with artifact.feature_config (not deploy IC-pruned yaml)."""
    artifact = ModelArtifact.load(artifact_dir)
    pipeline_cfg = pipeline_cfg_from_artifact(artifact)
    if include_gate_features:
        pipeline_cfg = supplement_pipeline_for_gate_features(
            pipeline_cfg, gate_feature_names
        )
    cfg_dir = config_dir or (
        PROJECT_ROOT / "config/strategies/tree_strategies/fast_scalp"
    )
    strategy_config = StrategyConfigLoader(cfg_dir).load()
    feature_loader = StrategyFeatureLoader()
    _inject_feature_freq(feature_loader, strategy_config, timeframe=timeframe)
    data_handler = DataHandler(data_path=data_path)

    parts: list[pd.DataFrame] = []
    for sym in symbols:
        df_raw = data_handler.load_ohlcv(symbol=sym, timeframe=timeframe)
        df_raw = df_raw.loc[start_date:end_date]
        if df_raw.empty:
            continue
        df_raw = df_raw.copy()
        df_raw["_symbol"] = sym
        df_raw["symbol"] = sym
        start_ts = str(df_raw.index.min())
        end_ts = str(df_raw.index.max())
        _ensure_ticks_configured(
            feature_loader,
            sym,
            data_path,
            start_ts,
            end_ts,
            pipeline_cfg.requested_features,
        )
        df_feat = run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=pipeline_cfg,
            fit=False,
            feature_store_dir="feature_store",
            feature_store_layer=feature_store_layer,
            feature_store_symbol=sym,
            feature_store_timeframe=timeframe,
        )
        missing = [c for c in artifact.used_features if c not in df_feat.columns]
        if missing:
            raise ValueError(
                f"{sym}: missing {len(missing)} model features after pipeline, "
                f"e.g. {missing[:5]}"
            )
        model_obj = artifact.model
        models = model_obj if isinstance(model_obj, list) else [model_obj]
        X = artifact.preprocessor.transform(
            df_feat, feature_cols=artifact.used_features
        )
        preds = generate_predictions(
            models,
            model_type=strategy_config.model.trainer.params.get(
                "model_type", "lightgbm"
            ),
            task_type=strategy_config.model.trainer.params.get(
                "task_type", "regression"
            ),
            X=X,
        )
        out = df_feat.copy()
        out["pred"] = np.asarray(preds, dtype=float)
        if "timestamp" not in out.columns:
            out = out.reset_index().rename(columns={"index": "timestamp"})
        parts.append(out)

    if not parts:
        raise ValueError(f"No rows for {start_date}→{end_date}")
    merged = pd.concat(parts, axis=0, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    if "atr" not in merged.columns:
        atr = _ensure_atr(
            merged,
            atr_col="atr",
            price_col="close",
            high_col="high",
            low_col="low",
            atr_window=14,
        )
        merged["atr"] = atr.values
    return merged


def validate_score_distribution(
    df: pd.DataFrame,
    *,
    score_col: str = "pred",
    short_entry: float = -0.0074,
    min_frac_short: float = 0.001,
) -> dict[str, Any]:
    scores = pd.to_numeric(df[score_col], errors="coerce")
    stats = {
        "n": int(scores.notna().sum()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "frac_le_short_entry": float((scores <= short_entry).mean()),
    }
    if stats["frac_le_short_entry"] < min_frac_short:
        raise ValueError(
            "degenerate score distribution: "
            f"frac<={short_entry}={stats['frac_le_short_entry']:.4f} "
            f"(min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f})"
        )
    return stats
