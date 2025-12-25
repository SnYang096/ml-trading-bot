from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .bc_dataset import (
    BCRouter3ActionDataset,
    BCStateSchema,
    Router3ActionInferConfig,
    bc3_collate_fn,
)


@dataclass(frozen=True)
class BC3TrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.1
    val_ratio: float = 0.2
    device: Optional[str] = None


class BC3PolicyMLP(nn.Module):
    def __init__(
        self, *, d_in: int, hidden: int = 128, depth: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        layers = []
        d = d_in
        for _ in range(int(depth)):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = hidden
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(d, 3)  # NO_TRADE/MEAN/TREND

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_bc_router3_policy(
    *,
    transitions: Sequence[Dict[str, Any]],
    state_schema: BCStateSchema,
    infer_cfg: Router3ActionInferConfig,
    cfg: BC3TrainConfig = BC3TrainConfig(),
) -> Tuple[BC3PolicyMLP, Dict[str, Any]]:
    """
    Minimal supervised BC trainer for 3-action Router.
    """
    _set_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = BCRouter3ActionDataset(
        transitions=transitions, state_schema=state_schema, infer_cfg=infer_cfg
    )
    if len(ds) < 50:
        raise ValueError(f"Not enough samples for BC training: {len(ds)}")

    n = len(ds)
    n_val = max(1, int(n * cfg.val_ratio))
    n_train = n - n_val
    train_subset = torch.utils.data.Subset(ds, list(range(0, n_train)))
    val_subset = torch.utils.data.Subset(ds, list(range(n_train, n)))

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=bc3_collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=bc3_collate_fn,
    )

    model = BC3PolicyMLP(
        d_in=state_schema.obs_dim,
        hidden=cfg.hidden,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
        pred = torch.argmax(logits, dim=-1)
        return float((pred == y).float().mean().item())

    history = {"train": [], "val": []}
    for epoch in range(int(cfg.epochs)):
        model.train()
        loss_sum = 0.0
        acc_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            loss_sum += float(loss.item())
            acc_sum += _acc(logits.detach(), y)
            n_batches += 1
        history["train"].append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(1, n_batches),
                "acc": acc_sum / max(1, n_batches),
            }
        )

        model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            acc_sum = 0.0
            n_batches = 0
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss_sum += float(loss.item())
                acc_sum += _acc(logits, y)
                n_batches += 1
        history["val"].append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(1, n_batches),
                "acc": acc_sum / max(1, n_batches),
            }
        )

    meta = {
        "state_schema": {"keys": list(state_schema.keys)},
        "infer_cfg": asdict(infer_cfg),
        "train_cfg": asdict(cfg),
        "history": history,
        "n_samples": len(ds),
        "n_train": n_train,
        "n_val": n_val,
    }
    return model, meta
