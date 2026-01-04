from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PathPrimitivesModelConfig:
    d_in: int
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.1

    # Optional extra head
    with_persistence: bool = False


class MultiHeadPathPrimitivesMLP(nn.Module):
    """
    Multi-head MLP that predicts path primitives:
      - dir_logit (binary classification)
      - mfe_atr (>=0, regression target usually log1p-scaled)
      - mae_atr (>=0, regression target usually log1p-scaled)
      - t_to_mfe (>=0, regression target usually log1p-scaled)
      - persistence (optional, [0,1] regression)
    """

    def __init__(self, cfg: Optional[PathPrimitivesModelConfig] = None, **kwargs):
        # Backward compatible: allow either cfg object or kwargs
        super().__init__()
        if cfg is None:
            cfg = PathPrimitivesModelConfig(**kwargs)
        self.cfg = cfg

        layers = []
        d = cfg.d_in
        for _ in range(int(cfg.depth)):
            layers.append(nn.Linear(d, cfg.hidden))
            layers.append(nn.ReLU())
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            d = cfg.hidden
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        self.dir_logit = nn.Linear(cfg.hidden if layers else cfg.d_in, 1)
        self.mfe = nn.Linear(cfg.hidden if layers else cfg.d_in, 1)
        self.mae = nn.Linear(cfg.hidden if layers else cfg.d_in, 1)
        self.t_to_mfe = nn.Linear(cfg.hidden if layers else cfg.d_in, 1)

        self.persistence = None
        if cfg.with_persistence:
            self.persistence = nn.Linear(cfg.hidden if layers else cfg.d_in, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        out: Dict[str, torch.Tensor] = {
            "dir_logit": self.dir_logit(h).squeeze(-1),
            # Use softplus for strictly-positive regression heads
            "mfe_atr": F.softplus(self.mfe(h)).squeeze(-1),
            "mae_atr": F.softplus(self.mae(h)).squeeze(-1),
            "t_to_mfe": F.softplus(self.t_to_mfe(h)).squeeze(-1),
        }
        if self.persistence is not None:
            out["persistence"] = torch.sigmoid(self.persistence(h)).squeeze(-1)
        return out

    def export_state(self) -> Dict:
        return {"config": asdict(self.cfg), "state_dict": self.state_dict()}

    @staticmethod
    def from_export(payload: Dict) -> "MultiHeadPathPrimitivesMLP":
        cfg = PathPrimitivesModelConfig(**payload["config"])
        model = MultiHeadPathPrimitivesMLP(cfg=cfg)
        model.load_state_dict(payload["state_dict"])
        return model
