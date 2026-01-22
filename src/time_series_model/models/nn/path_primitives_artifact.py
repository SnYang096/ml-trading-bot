"""
PathPrimitivesModelArtifact: unified loader for nnmultihead (path primitives) models.

Motivation:
- Avoid repeating fragile glue code across scripts:
  - feature_cols selection must match training
  - feature_scaler must be applied consistently
  - optional-block mask settings must match contract

This mirrors the tree-side ModelArtifact idea, but for PyTorch nnmultihead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.time_series_model.models.nn.feature_contract import (
    FeatureContract,
    load_feature_contract,
)
from src.time_series_model.models.nn.path_primitives_dataset import (
    resolve_block_cols_by_name,
)
from src.time_series_model.models.nn.path_primitives_model import (
    MultiHeadPathPrimitivesMLP,
)


@dataclass(frozen=True)
class PathPrimitivesModelArtifact:
    """
    Unifies everything required for consistent inference/eval:
    - model (torch module)
    - feature_cols (exact training-time feature ordering)
    - feature_scaler (z-score params fitted on training data)
    - contract + block mask behavior (optional blocks)
    - dataset_cfg (e.g., log1p_targets)
    """

    model: MultiHeadPathPrimitivesMLP
    meta: Dict[str, Any]
    feature_cols: Optional[List[str]]
    feature_scaler: Optional[Dict[str, Any]]
    contract: Optional[FeatureContract]
    block_cols_by_name: Optional[Dict[str, List[str]]]
    append_block_mask: bool

    @classmethod
    def load(
        cls,
        *,
        model_path: str | Path,
        config_dir: Optional[str | Path] = None,
    ) -> "PathPrimitivesModelArtifact":
        payload = torch.load(str(model_path), map_location="cpu")
        if "model" not in payload:
            raise ValueError("Invalid model payload: missing 'model' key")
        model = MultiHeadPathPrimitivesMLP.from_export(payload["model"])
        meta = payload.get("meta", {}) or {}

        feature_cols = meta.get("feature_cols", None)
        if feature_cols is not None:
            feature_cols = [str(c) for c in list(feature_cols)]

        feature_scaler = meta.get("feature_scaler", None)

        contract = None
        # Priority: use feature_contract from model meta (training-time config)
        # Only fallback to config_dir if model doesn't have it saved
        try:
            fc = meta.get("feature_contract", None)
            if isinstance(fc, dict):
                contract = FeatureContract.from_dict(fc)  # type: ignore[attr-defined]
        except Exception:
            contract = None

        # Fallback to config_dir if model doesn't have feature_contract saved
        if contract is None and config_dir is not None:
            contract = load_feature_contract(Path(config_dir).resolve())

        block_cols_by_name = None
        append_block_mask = False
        if (
            contract is not None
            and contract.optional_blocks
            and feature_cols is not None
        ):
            block_cols_by_name = resolve_block_cols_by_name(
                feature_cols, optional_blocks=contract.optional_blocks
            )
            append_block_mask = bool(
                (contract.missingness_policy or {}).get("append_block_mask", False)
            )
            if not block_cols_by_name:
                append_block_mask = False

        # Safety check: ensure expected_feature_dim matches model.input_dim
        # If model was trained without block masks, disable append_block_mask even if contract says otherwise
        if feature_cols is not None:
            base_dim = len(feature_cols)
            if append_block_mask and block_cols_by_name:
                expected_with_mask = base_dim + len(block_cols_by_name)
            else:
                expected_with_mask = base_dim

            # Try to get model input dim from backbone first layer
            try:
                model_input_dim = None
                if hasattr(model, "input_dim"):
                    model_input_dim = model.input_dim
                elif hasattr(model, "backbone"):
                    layers = list(model.backbone.children())
                    if layers and hasattr(layers[0], "in_features"):
                        model_input_dim = layers[0].in_features

                if (
                    model_input_dim is not None
                    and expected_with_mask != model_input_dim
                ):
                    # Model expects different dimension, disable append_block_mask to match training
                    append_block_mask = False
                    block_cols_by_name = None
            except Exception:
                pass

        return cls(
            model=model,
            meta=meta,
            feature_cols=feature_cols,
            feature_scaler=feature_scaler,
            contract=contract,
            block_cols_by_name=block_cols_by_name,
            append_block_mask=append_block_mask,
        )

    def preds_in_log1p(self, default: bool = True) -> bool:
        ds_cfg = self.meta.get("dataset_cfg") or {}
        return bool(ds_cfg.get("log1p_targets", default))

    def input_dim(self) -> Optional[int]:
        try:
            return int(self.model.input_dim)
        except Exception:
            return None

    def expected_feature_dim(self) -> Optional[int]:
        if self.feature_cols is None:
            return None
        base = len(self.feature_cols)
        extra = 0
        if self.append_block_mask and self.block_cols_by_name:
            # block mask dims == number of blocks resolved
            extra = len(list(self.block_cols_by_name.keys()))
        return int(base + extra)
