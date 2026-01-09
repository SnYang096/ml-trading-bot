from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .path_primitives_dataset import (
    DatasetConfig,
    PathPrimitivesDataset,
    build_feature_matrix,
    collate_fn,
)


@dataclass
class FeatureScaler:
    """Simple z-score scaler for features."""

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / (self.std + self.eps)

    def to_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps}

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureScaler":
        return cls(
            mean=np.array(d["mean"]), std=np.array(d["std"]), eps=d.get("eps", 1e-8)
        )

    @classmethod
    def fit(cls, X: np.ndarray) -> "FeatureScaler":
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        # Avoid division by zero for constant features
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)


from .path_primitives_dataset import compute_block_feature_indices
from .path_primitives_labels import (
    PathPrimitivesLabelConfig,
    compute_path_primitives_labels,
)
from .path_primitives_loss import (
    LossWeights,
    default_loss_weights,
    path_primitives_loss,
)
from .path_primitives_model import MultiHeadPathPrimitivesMLP, PathPrimitivesModelConfig


def _is_verbose() -> bool:
    v = str(os.environ.get("MLBOT_TRAIN_VERBOSE", "")).strip().lower()
    return v not in ("", "0", "false", "no", "off")


def _log(msg: str) -> None:
    if _is_verbose():
        print(msg, flush=True)


@dataclass(frozen=True)
class TrainConfig:
    # Label config
    label_cfg: PathPrimitivesLabelConfig

    # Dataset config
    dataset_cfg: DatasetConfig = DatasetConfig()

    # Model config (d_in is set dynamically)
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.1

    # Training
    batch_size: int = 512
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    val_ratio: float = 0.2  # last 20% as validation (time-ordered)
    seed: int = 42
    device: Optional[str] = None  # "cuda" or "cpu"

    # Loss weights
    use_weight_schedule: bool = True
    fixed_weights: LossWeights = LossWeights()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_path_primitives_mlp(
    df: pd.DataFrame,
    *,
    feature_cols: List[str],
    cfg: TrainConfig,
    save_path: Optional[str] = None,
    group_col: Optional[str] = None,
    block_cols_by_name: Optional[Dict[str, List[str]]] = None,
    append_block_mask: bool = False,
    block_dropout_p: float = 0.0,
) -> Tuple[MultiHeadPathPrimitivesMLP, Dict]:
    """
    Minimal trainer for the multi-head path primitives MLP.

    This intentionally does NOT integrate with `strategy_trainer.py` yet, because:
    - current strategy configs assume a single target_col
    - multi-head needs multi-target labels + masks

    Returns:
      (trained_model, metadata)
    """

    _set_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    _log(
        f"[train_path_primitives_mlp] device={device} seed={cfg.seed} epochs={cfg.epochs}"
    )

    block_names: List[str] = []
    block_feature_indices: List[List[int]] = []
    if append_block_mask and block_cols_by_name:
        block_names, block_feature_indices = compute_block_feature_indices(
            feature_cols, block_cols_by_name=block_cols_by_name
        )
        _log(
            f"[train_path_primitives_mlp] block_mask=on blocks={len(block_names)} "
            f"(append_block_mask={append_block_mask}, block_dropout_p={block_dropout_p})"
        )
    else:
        _log(
            f"[train_path_primitives_mlp] block_mask=off (append_block_mask={append_block_mask})"
        )

    # 1) Labels + dataset (group-safe)
    # IMPORTANT: if multi-symbol, compute labels/datasets per symbol to avoid horizon leakage.
    train_subsets = []
    val_subsets = []
    n_samples_total = 0
    n_train_total = 0
    n_val_total = 0

    if group_col is not None and group_col in df.columns:
        groups = list(df.groupby(group_col, sort=False))
    else:
        groups = [(None, df)]
    _log(
        f"[train_path_primitives_mlp] n_groups={len(groups)} group_col={group_col} "
        f"n_rows_input={len(df)} n_feature_cols={len(feature_cols)}"
    )

    # First pass: collect all raw feature matrices for scaler fitting
    all_X_raw = []
    all_labels_list = []
    all_work_list = []

    for gname, gdf in groups:
        _log(
            f"[train_path_primitives_mlp] build labels+X group={gname} rows={len(gdf)}"
        )
        df_labels = compute_path_primitives_labels(
            gdf, cfg=cfg.label_cfg, out_prefix="", group_col=None
        )
        work = gdf.join(df_labels)
        X_raw = build_feature_matrix(
            work,
            feature_cols,
            fill_nan_value=cfg.dataset_cfg.fill_nan_value,
            block_cols_by_name=block_cols_by_name,
            append_block_mask=append_block_mask,
        )
        labels = {
            cfg.dataset_cfg.dir_y_col: work[cfg.dataset_cfg.dir_y_col].to_numpy(
                dtype=float
            ),
            cfg.dataset_cfg.mfe_atr_col: work[cfg.dataset_cfg.mfe_atr_col].to_numpy(
                dtype=float
            ),
            cfg.dataset_cfg.mae_atr_col: work[cfg.dataset_cfg.mae_atr_col].to_numpy(
                dtype=float
            ),
            cfg.dataset_cfg.t_to_mfe_col: work[cfg.dataset_cfg.t_to_mfe_col].to_numpy(
                dtype=float
            ),
            cfg.dataset_cfg.mfe_valid_col: work[cfg.dataset_cfg.mfe_valid_col].to_numpy(
                dtype=float
            ),
        }
        all_X_raw.append(X_raw)
        all_labels_list.append(labels)
        all_work_list.append(work)

    # Fit scaler on all training data (only on feature columns, not block masks)
    n_feature_cols = len(feature_cols)
    _log("[train_path_primitives_mlp] fitting scaler on concatenated X ...")
    all_X_concat = np.vstack(all_X_raw)
    # Only scale feature columns, not block mask columns (last n_blocks dims if append_block_mask)
    feature_scaler = FeatureScaler.fit(all_X_concat[:, :n_feature_cols])
    _log(
        f"[train_path_primitives_mlp] scaler fit done X_shape={tuple(all_X_concat.shape)} "
        f"(features={n_feature_cols}, extra={int(all_X_concat.shape[1] - n_feature_cols)})"
    )

    # Second pass: apply scaler and create datasets
    for i, (X_raw, labels) in enumerate(zip(all_X_raw, all_labels_list)):
        # Scale feature columns
        X_scaled = X_raw.copy()
        X_scaled[:, :n_feature_cols] = feature_scaler.transform(
            X_raw[:, :n_feature_cols]
        )

        ds = PathPrimitivesDataset(X_scaled, labels, cfg=cfg.dataset_cfg)
        if len(ds) < 10:
            # Skip tiny groups (not enough to form windows)
            continue

        n = len(ds)
        n_val = max(1, int(n * cfg.val_ratio))
        n_train = n - n_val

        train_subsets.append(torch.utils.data.Subset(ds, list(range(0, n_train))))
        val_subsets.append(torch.utils.data.Subset(ds, list(range(n_train, n))))
        n_samples_total += n
        n_train_total += n_train
        n_val_total += n_val

    if n_samples_total < 100:
        raise ValueError(f"Not enough valid samples for training: {n_samples_total}")
    _log(
        f"[train_path_primitives_mlp] dataset ready n_samples={n_samples_total} "
        f"n_train={n_train_total} n_val={n_val_total} batch_size={cfg.batch_size}"
    )

    train_dataset = torch.utils.data.ConcatDataset(train_subsets)
    val_dataset = torch.utils.data.ConcatDataset(val_subsets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,  # shuffle within train to stabilize SGD
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 4) Model
    model_cfg = PathPrimitivesModelConfig(
        # d_in is consistent across groups because feature_cols fixed
        d_in=int(train_subsets[0].dataset.X.shape[1]) if train_subsets else 0,
        hidden=cfg.hidden,
        depth=cfg.depth,
        dropout=cfg.dropout,
        with_persistence=False,
    )
    model = MultiHeadPathPrimitivesMLP(cfg=model_cfg).to(device)
    _log(
        f"[train_path_primitives_mlp] model created d_in={model_cfg.d_in} hidden={model_cfg.hidden} "
        f"depth={model_cfg.depth} dropout={model_cfg.dropout}"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    history = {"train": [], "val": []}

    def _eval(loader: DataLoader, *, epoch: int) -> Dict[str, float]:
        model.eval()
        comps_sum = {"dir": 0.0, "mfe": 0.0, "mae": 0.0, "t": 0.0, "total": 0.0}
        n_batches = 0
        w = (
            cfg.fixed_weights
            if not cfg.use_weight_schedule
            else default_loss_weights(epoch)
        )
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(batch["x"])
                loss, comps = path_primitives_loss(pred, batch, w=w)
                for k in comps_sum:
                    comps_sum[k] += comps[k]
                n_batches += 1
        if n_batches == 0:
            return {k: 0.0 for k in comps_sum}
        return {k: v / n_batches for k, v in comps_sum.items()}

    # 5) Train loop
    best_val = float("inf")
    best_state = None

    for epoch in range(int(cfg.epochs)):
        t_ep = time.time()
        model.train()
        w = (
            cfg.fixed_weights
            if not cfg.use_weight_schedule
            else default_loss_weights(epoch)
        )

        comps_sum = {"dir": 0.0, "mfe": 0.0, "mae": 0.0, "t": 0.0, "total": 0.0}
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch["x"]
            # Optional: per-batch block dropout for robustness to block on/off toggles.
            # This only applies during training; val/eval sees deterministic availability masks.
            if (
                append_block_mask
                and block_feature_indices
                and float(block_dropout_p) > 0.0
                and len(block_names) > 0
            ):
                base_d = int(len(feature_cols))
                n_blocks = int(len(block_names))
                # last n_blocks dims are block availability masks
                rand = torch.rand((x.shape[0], n_blocks), device=x.device)
                avail = x[:, base_d : base_d + n_blocks] > 0.5
                drop = (rand < float(block_dropout_p)) & avail
                for b, idxs in enumerate(block_feature_indices):
                    if not idxs:
                        continue
                    rows = drop[:, b]  # Boolean tensor of shape (batch_size,)
                    if rows.any():
                        # Get row indices to drop
                        row_indices = torch.where(rows)[0]
                        # Zero out feature columns for these rows
                        for col_idx in idxs:
                            x[row_indices, col_idx] = 0.0
                        # Zero out block mask for these rows
                        x[row_indices, base_d + b] = 0.0
                batch["x"] = x

            pred = model(batch["x"])
            loss, comps = path_primitives_loss(pred, batch, w=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            for k in comps_sum:
                comps_sum[k] += comps[k]
            n_batches += 1

        train_metrics = {k: v / max(1, n_batches) for k, v in comps_sum.items()}
        val_metrics = _eval(val_loader, epoch=epoch)

        history["train"].append({"epoch": epoch, **train_metrics, "weights": asdict(w)})
        history["val"].append({"epoch": epoch, **val_metrics})

        if val_metrics["total"] < best_val:
            best_val = float(val_metrics["total"])
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        _log(
            f"[train_path_primitives_mlp] epoch={epoch+1}/{cfg.epochs} "
            f"train_total={float(train_metrics.get('total', 0.0)):.6g} "
            f"val_total={float(val_metrics.get('total', 0.0)):.6g} "
            f"dt_s={time.time()-t_ep:.1f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    _log(
        f"[train_path_primitives_mlp] done dt_s={time.time()-t0:.1f} best_val={best_val:.6g}"
    )

    meta = {
        "feature_cols": feature_cols,
        "label_cfg": asdict(cfg.label_cfg),
        "dataset_cfg": asdict(cfg.dataset_cfg),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(cfg),
        "device": device,
        "history": history,
        "group_col": group_col,
        "n_samples": n_samples_total,
        "n_train": n_train_total,
        "n_val": n_val_total,
        "append_block_mask": bool(append_block_mask),
        "block_dropout_p": float(block_dropout_p),
        "block_mask_names": list(block_names),
        "feature_scaler": feature_scaler.to_dict(),
    }

    if save_path:
        payload = {"meta": meta, "model": model.export_state()}
        torch.save(payload, save_path)

    return model, meta
