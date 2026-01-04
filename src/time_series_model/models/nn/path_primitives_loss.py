from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    dir: float = 1.0
    mfe: float = 0.5
    mae: float = 0.5
    t: float = 0.3


def default_loss_weights(epoch: int) -> LossWeights:
    """
    Simple, practical schedule:
    - early: learn direction first
    - mid: ramp up mfe/mae
    - late: increase time-scale head
    """
    if epoch < 5:
        return LossWeights(dir=1.0, mfe=0.2, mae=0.2, t=0.1)
    if epoch < 20:
        return LossWeights(dir=0.8, mfe=0.6, mae=0.6, t=0.2)
    return LossWeights(dir=0.6, mfe=1.0, mae=1.0, t=0.5)


def path_primitives_loss(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    w: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-head loss:
    - dir: BCEWithLogits (binary)
    - mfe/t_to_mfe: masked by mfe_valid (when no upside excursion exists)
    - mae: always supervised (risk exists even if no upside)

    Assumes batch targets are already transformed consistently (e.g. log1p done in dataset).
    """

    dir_y = batch["dir_y"].float()
    mfe_valid = batch["mfe_valid"].float().clamp(0.0, 1.0)

    loss_dir = F.binary_cross_entropy_with_logits(pred["dir_logit"], dir_y)

    # Regression heads use smooth_l1 (Huber)
    loss_mfe_raw = F.smooth_l1_loss(
        pred["mfe_atr"], batch["mfe_atr"].float(), reduction="none"
    )
    loss_t_raw = F.smooth_l1_loss(
        pred["t_to_mfe"], batch["t_to_mfe"].float(), reduction="none"
    )

    denom = mfe_valid.sum().clamp_min(1.0)
    loss_mfe = (loss_mfe_raw * mfe_valid).sum() / denom
    loss_t = (loss_t_raw * mfe_valid).sum() / denom

    loss_mae = F.smooth_l1_loss(pred["mae_atr"], batch["mae_atr"].float())

    total = (
        float(w.dir) * loss_dir
        + float(w.mfe) * loss_mfe
        + float(w.mae) * loss_mae
        + float(w.t) * loss_t
    )

    return total, {
        "dir": float(loss_dir.detach().cpu().item()),
        "mfe": float(loss_mfe.detach().cpu().item()),
        "mae": float(loss_mae.detach().cpu().item()),
        "t": float(loss_t.detach().cpu().item()),
        "total": float(total.detach().cpu().item()),
    }
