from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class Router3Action(IntEnum):
    """
    3-action Router mode (recommended action space for sample efficiency).
    """

    NO_TRADE = 0
    MEAN = 1
    TREND = 2


@dataclass(frozen=True)
class Router3ActionInferConfig:
    """
    Map fine-grained router knobs into 3-action structural modes.

    - mean_routers: routers considered mean-reversion family (e.g., sr_reversal, compression_fade)
    - trend_routers: routers considered trend-follow family (e.g., trend, breakout)
    """

    mean_routers: Sequence[str]
    trend_routers: Sequence[str]
    min_effective_mult: float = 1e-6


def infer_router3_action(
    action: Dict[str, Any],
    *,
    cfg: Router3ActionInferConfig,
) -> Router3Action:
    """
    Infer 3-action label from a RouterAction-like dict.

    Rules:
    - global_pause => NO_TRADE
    - if neither mean nor trend has any effective enabled*mult => NO_TRADE
    - else choose argmax(sum effective mult) between mean and trend (ties -> MEAN)
    """
    action = action or {}

    # If new system logs a first-class mode label, use it directly (no router-name grouping required).
    if "mode" in action and action.get("mode") is not None:
        m = str(action.get("mode")).upper()
        if m in {"NO_TRADE", "NOTRADE", "OFF", "OBSERVE", "PAUSE"}:
            return Router3Action.NO_TRADE
        if m in {"MEAN", "MEAN_REVERT", "MEANREVERT"}:
            return Router3Action.MEAN
        if m in {"TREND", "TREND_FOLLOW", "TRENDFOLLOW"}:
            return Router3Action.TREND

    if bool(action.get("global_pause", False)):
        return Router3Action.NO_TRADE

    enabled = action.get("router_enabled") or {}
    mult = action.get("capital_multiplier") or {}

    def _eff(name: str) -> float:
        if not bool(enabled.get(name, True)):
            return 0.0
        try:
            return float(mult.get(name, 0.0) or 0.0)
        except Exception:
            return 0.0

    mean_w = float(sum(max(0.0, _eff(r)) for r in cfg.mean_routers))
    trend_w = float(sum(max(0.0, _eff(r)) for r in cfg.trend_routers))

    if max(mean_w, trend_w) <= float(cfg.min_effective_mult):
        return Router3Action.NO_TRADE
    if trend_w > mean_w:
        return Router3Action.TREND
    return Router3Action.MEAN


@dataclass(frozen=True)
class BCPolicySchema:
    """
    Fixed, RL-ready action vector schema for supervised BC.

    We encode RouterAction dicts into a single vector:
      [global_pause,
       enabled[r0..rN-1],
       capital_multiplier[r0..rN-1]]
    """

    router_names: Sequence[str]
    min_mult: float = 0.0
    max_mult: float = 2.0

    @property
    def action_dim(self) -> int:
        r = len(self.router_names)
        return 1 + r + r

    def encode_action(self, action: Dict[str, Any]) -> np.ndarray:
        action = action or {}
        gp = 1.0 if bool(action.get("global_pause", False)) else 0.0
        enabled = action.get("router_enabled") or {}
        mult = action.get("capital_multiplier") or {}

        en = []
        cm = []
        for rn in self.router_names:
            en.append(1.0 if bool(enabled.get(rn, True)) else 0.0)
            try:
                v = float(mult.get(rn, 1.0))
            except Exception:
                v = 0.0
            v = min(max(v, float(self.min_mult)), float(self.max_mult))
            cm.append(v)

        return np.asarray([gp, *en, *cm], dtype=np.float32)

    def decode_action(self, vec: np.ndarray) -> Dict[str, Any]:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        r = len(self.router_names)
        if v.shape[0] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {v.shape[0]}")
        gp = bool(v[0] >= 0.5)
        enabled = {self.router_names[i]: bool(v[1 + i] >= 0.5) for i in range(r)}
        cm0 = v[1 + r : 1 + r + r]
        cm = {self.router_names[i]: float(cm0[i]) for i in range(r)}
        return {"global_pause": gp, "router_enabled": enabled, "capital_multiplier": cm}


@dataclass(frozen=True)
class BCStateSchema:
    """
    Deterministic flattening of state dict into a vector.

    - keys define ordering
    - missing values default to 0.0
    """

    keys: Sequence[str]

    @property
    def obs_dim(self) -> int:
        return int(len(self.keys))

    def encode_state(self, state: Dict[str, Any]) -> np.ndarray:
        s = state or {}
        x = np.zeros(self.obs_dim, dtype=np.float32)
        for i, k in enumerate(self.keys):
            try:
                v = float(s.get(k, 0.0) or 0.0)
                if not np.isfinite(v):
                    v = 0.0
                # Prevent float32 overflow downstream (e.g. extreme expm1 values).
                v = float(max(-1e6, min(1e6, v)))
                x[i] = v
            except Exception:
                x[i] = 0.0
        return x


class BCDataset(Dataset):
    """
    Offline behavior cloning dataset: (obs_vec, action_vec).
    """

    def __init__(
        self,
        *,
        transitions: Sequence[Dict[str, Any]],
        state_schema: BCStateSchema,
        policy_schema: BCPolicySchema,
    ) -> None:
        self.state_schema = state_schema
        self.policy_schema = policy_schema

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for t in transitions:
            state = dict(t.get("state") or {})
            action = dict(t.get("action") or {})
            xs.append(self.state_schema.encode_state(state))
            ys.append(self.policy_schema.encode_action(action))

        self._x = (
            np.stack(xs, axis=0).astype(np.float32)
            if xs
            else np.zeros((0, state_schema.obs_dim))
        )
        self._y = (
            np.stack(ys, axis=0).astype(np.float32)
            if ys
            else np.zeros((0, policy_schema.action_dim))
        )

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self._x[idx]),
            "y": torch.from_numpy(self._y[idx]),
        }


def bc_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {"x": x, "y": y}


class BCRouter3ActionDataset(Dataset):
    """
    Offline BC dataset for 3-action classification: (obs_vec, action_class).
    """

    def __init__(
        self,
        *,
        transitions: Sequence[Dict[str, Any]],
        state_schema: BCStateSchema,
        infer_cfg: Router3ActionInferConfig,
    ) -> None:
        self.state_schema = state_schema
        self.infer_cfg = infer_cfg

        xs: List[np.ndarray] = []
        ys: List[int] = []
        for t in transitions:
            state = dict(t.get("state") or {})
            action = dict(t.get("action") or {})
            xs.append(self.state_schema.encode_state(state))
            ys.append(int(infer_router3_action(action, cfg=infer_cfg)))

        self._x = (
            np.stack(xs, axis=0).astype(np.float32)
            if xs
            else np.zeros((0, state_schema.obs_dim))
        )
        self._y = (
            np.asarray(ys, dtype=np.int64) if ys else np.zeros((0,), dtype=np.int64)
        )

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self._x[idx]),
            "y": torch.as_tensor(int(self._y[idx]), dtype=torch.long),
        }


def bc3_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {"x": x, "y": y}
