from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fnmatch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetConfig:
    # Which columns to use for multi-head labels (already computed)
    dir_y_col: str = "dir_y"
    mfe_atr_col: str = "mfe_atr"
    mae_atr_col: str = "mae_atr"
    t_to_mfe_col: str = "t_to_mfe"
    mfe_valid_col: str = "mfe_valid"

    # Transform stability
    log1p_targets: bool = True
    clamp_targets: bool = True
    cap_mfe_atr: float = 10.0
    cap_mae_atr: float = 10.0

    # Feature preprocessing for torch (trees can keep NaN; torch generally can't)
    fill_nan_value: float = 0.0


def resolve_block_cols_by_name(
    feature_cols: List[str],
    *,
    optional_blocks: object,
) -> Dict[str, List[str]]:
    """
    Resolve FeatureContract optional blocks into concrete column lists.

    Supported formats:
    - legacy: List[str] (block names only) -> returns {}
    - new: Dict[str, List[str]] where values are patterns (fnmatch) or exact columns
    """
    if not feature_cols:
        return {}

    if isinstance(optional_blocks, dict):
        out: Dict[str, List[str]] = {}
        for bname, patterns in optional_blocks.items():
            if not isinstance(bname, str) or not bname.strip():
                continue
            pats = patterns if isinstance(patterns, list) else []
            pats = [str(p).strip() for p in pats if str(p).strip()]
            if not pats:
                continue
            matched: List[str] = []
            for col in feature_cols:
                for pat in pats:
                    if col == pat or fnmatch.fnmatch(col, pat):
                        matched.append(col)
                        break
            # De-dup while preserving order
            seen = set()
            matched = [c for c in matched if not (c in seen or seen.add(c))]
            if matched:
                out[str(bname)] = matched
        return out

    # legacy list[str] does not provide column mapping, so no block features can be resolved
    return {}


def compute_block_feature_indices(
    feature_cols: List[str], *, block_cols_by_name: Dict[str, List[str]]
) -> Tuple[List[str], List[List[int]]]:
    """
    Convert block->cols mapping into:
    - block_names: list[str] (stable order)
    - block_feature_indices: list[list[int]] indices into feature_cols for each block
    """
    if not feature_cols or not block_cols_by_name:
        return [], []
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    block_names: List[str] = []
    block_indices: List[List[int]] = []
    for bname in sorted(block_cols_by_name.keys()):
        cols = block_cols_by_name.get(bname) or []
        idxs = [col_to_idx[c] for c in cols if c in col_to_idx]
        if idxs:
            block_names.append(bname)
            block_indices.append(idxs)
    return block_names, block_indices


def _compute_block_availability_mask(
    df: pd.DataFrame,
    *,
    block_cols_by_name: Dict[str, List[str]],
    block_names: List[str],
) -> np.ndarray:
    """
    Returns float32 matrix of shape (n_rows, n_blocks) with values in {0,1}.
    A block is available at a row if ANY column in that block is finite (notna and not inf).
    """
    n = len(df)
    if n == 0 or not block_names:
        return np.zeros((n, 0), dtype=np.float32)

    mask = np.zeros((n, len(block_names)), dtype=np.float32)
    for j, bname in enumerate(block_names):
        cols = block_cols_by_name.get(bname) or []
        if not cols:
            continue
        # Only consider columns that exist; missing columns => unavailable.
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        block_df = (
            df[present]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        avail = block_df.notna().any(axis=1).to_numpy(dtype=bool)
        mask[:, j] = avail.astype(np.float32)
    return mask


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    fill_nan_value: float = 0.0,
    block_cols_by_name: Optional[Dict[str, List[str]]] = None,
    append_block_mask: bool = False,
) -> np.ndarray:
    """
    Minimal feature matrix builder for torch.

    - Converts to numeric
    - Replaces inf/-inf with NaN
    - Fills NaN with fill_nan_value

    (Tree pipeline keeps NaN; torch pipeline needs a deterministic numeric matrix.)
    """
    if not feature_cols:
        return np.zeros((len(df), 0), dtype=np.float32)

    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col not in df.columns:
            X[col] = np.nan
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        X[col] = s.astype(float)

    arr = X[feature_cols].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(
        arr,
        nan=float(fill_nan_value),
        posinf=float(fill_nan_value),
        neginf=float(fill_nan_value),
    )

    if append_block_mask and block_cols_by_name:
        block_names, _ = compute_block_feature_indices(
            feature_cols, block_cols_by_name=block_cols_by_name
        )
        block_mask = _compute_block_availability_mask(
            df, block_cols_by_name=block_cols_by_name, block_names=block_names
        )
        # Concatenate mask as extra input dims (model can learn to ignore missing blocks)
        if block_mask.shape[1] > 0:
            arr = np.concatenate([arr, block_mask.astype(np.float32)], axis=1)
    return arr


class PathPrimitivesDataset(Dataset):
    """
    Torch dataset for path primitives multi-head training.

    Each sample corresponds to a timestamp t (aligned with features and labels).
    """

    def __init__(
        self,
        X: np.ndarray,
        labels: Dict[str, np.ndarray],
        *,
        cfg: DatasetConfig,
    ):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.cfg = cfg

        # Required label keys
        self.dir_y = torch.as_tensor(labels[cfg.dir_y_col], dtype=torch.float32)
        self.mfe_atr = torch.as_tensor(labels[cfg.mfe_atr_col], dtype=torch.float32)
        self.mae_atr = torch.as_tensor(labels[cfg.mae_atr_col], dtype=torch.float32)
        self.t_to_mfe = torch.as_tensor(labels[cfg.t_to_mfe_col], dtype=torch.float32)
        self.mfe_valid = torch.as_tensor(labels[cfg.mfe_valid_col], dtype=torch.float32)

        # Optional transforms
        if cfg.clamp_targets:
            self.mfe_atr = self.mfe_atr.clamp(0.0, float(cfg.cap_mfe_atr))
            self.mae_atr = self.mae_atr.clamp(0.0, float(cfg.cap_mae_atr))
        if cfg.log1p_targets:
            self.mfe_atr = torch.log1p(self.mfe_atr)
            self.mae_atr = torch.log1p(self.mae_atr)
            self.t_to_mfe = torch.log1p(self.t_to_mfe.clamp_min(0.0))

        # Build a valid sample mask:
        # - dir_y must be finite
        # - mae_atr must be finite
        # (mfe_atr and t_to_mfe may be masked by mfe_valid in loss)
        valid = torch.isfinite(self.dir_y) & torch.isfinite(self.mae_atr)
        self.valid_idx = torch.where(valid)[0]

    def __len__(self) -> int:
        return int(self.valid_idx.numel())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(self.valid_idx[idx].item())
        return {
            "x": self.X[i],
            "dir_y": self.dir_y[i],
            "mfe_atr": self.mfe_atr[i],
            "mae_atr": self.mae_atr[i],
            "t_to_mfe": self.t_to_mfe[i],
            "mfe_valid": self.mfe_valid[i],
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}
