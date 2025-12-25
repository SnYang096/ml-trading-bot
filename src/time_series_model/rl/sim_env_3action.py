from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .bc_dataset import Router3Action
from .reward import RewardConfig, compute_router_reward_from_step


@dataclass(frozen=True)
class SimEnvConfig:
    """
    A minimal, production-oriented simulation environment for offline RL/BC at the Router layer.

    Key design choices:
    - Action space is 3-action structural mode: NO_TRADE / MEAN / TREND
    - The environment does NOT model entry/exit micro-mechanics. Instead, it consumes
      per-step "next return if in mode" series precomputed by your execution/backtest layer.
    - Costs, slippage, turnover, drawdown guards are applied consistently and deterministically.
    """

    # Required market-return columns (interpreted as next-step returns for unit exposure)
    ret_mean_col: str = "ret_mean"
    ret_trend_col: str = "ret_trend"
    realized_vol_col: Optional[str] = (
        None  # if provided, passed into reward as realized_vol
    )

    # Execution timing: action at t sets exposure for t+delay return.
    entry_delay: int = 1

    # Exposure / risk budget
    base_exposure: float = 1.0
    mean_exposure: float = 0.8
    trend_exposure: float = 1.0
    max_abs_exposure: float = 1.0

    # Costs
    cost_per_turnover: float = 0.0002  # 2 bps per 1.0 turnover
    slippage_bps: float = 0.0  # extra cost proportional to abs(exposure) in bps

    # Risk control
    max_drawdown_stop: Optional[float] = (
        None  # e.g. 0.2 => 20% peak-to-trough forces NO_TRADE
    )
    cooldown_steps: int = 0  # steps to stay in NO_TRADE after stop triggers

    # Episode state
    initial_equity: float = 1.0


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


def _action_to_target_exposure(
    action: int | Router3Action, *, cfg: SimEnvConfig
) -> float:
    a = int(action)
    if a == int(Router3Action.NO_TRADE):
        return 0.0
    if a == int(Router3Action.MEAN):
        return _clip(
            cfg.base_exposure * cfg.mean_exposure,
            -cfg.max_abs_exposure,
            cfg.max_abs_exposure,
        )
    if a == int(Router3Action.TREND):
        return _clip(
            cfg.base_exposure * cfg.trend_exposure,
            -cfg.max_abs_exposure,
            cfg.max_abs_exposure,
        )
    raise ValueError(f"Unknown action: {action}")


def simulate_3action_episode(
    df: pd.DataFrame,
    *,
    actions: Sequence[int],
    cfg: SimEnvConfig = SimEnvConfig(),
) -> pd.DataFrame:
    """
    Deterministic simulation given a full action sequence aligned to df index.

    Interpretation:
    - At step t, we apply action[t] to set exposure for t+entry_delay.
    - PnL at time t is exposure[t] * return[t] (where return[t] is the mode return series).
      Therefore if entry_delay=1, action[t] affects PnL starting at t+1.

    Returns a DataFrame with:
      equity, drawdown, exposure, turnover, cost, pnl, mode_action
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if len(actions) != len(df):
        raise ValueError(
            f"actions length {len(actions)} must match df length {len(df)}"
        )
    if cfg.ret_mean_col not in df.columns:
        raise ValueError(f"Missing required return column: {cfg.ret_mean_col}")
    if cfg.ret_trend_col not in df.columns:
        raise ValueError(f"Missing required return column: {cfg.ret_trend_col}")

    ret_mean = (
        pd.to_numeric(df[cfg.ret_mean_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    ret_trend = (
        pd.to_numeric(df[cfg.ret_trend_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    T = len(df)
    exposure = np.zeros(T, dtype=float)
    desired = np.zeros(T, dtype=float)
    action_arr = np.asarray([int(a) for a in actions], dtype=int)

    cooldown = 0
    peak = float(cfg.initial_equity)
    equity = float(cfg.initial_equity)

    out_equity = np.zeros(T, dtype=float)
    out_dd = np.zeros(T, dtype=float)
    out_turnover = np.zeros(T, dtype=float)
    out_cost = np.zeros(T, dtype=float)
    out_pnl = np.zeros(T, dtype=float)
    out_cooldown = np.zeros(T, dtype=int)

    # Pre-compute desired exposure based on action (risk control may override later)
    for t in range(T):
        desired[t] = _action_to_target_exposure(action_arr[t], cfg=cfg)

    for t in range(T):
        # apply drawdown stop / cooldown before setting exposure
        if cfg.max_drawdown_stop is not None and peak > 0:
            dd = (peak - equity) / peak
            if dd + 1e-12 >= float(cfg.max_drawdown_stop):
                cooldown = max(cooldown, int(cfg.cooldown_steps))

        if cooldown > 0:
            target = 0.0
            cooldown -= 1
        else:
            target = float(desired[t])

        prev = float(exposure[t - 1]) if t > 0 else 0.0
        turnover = abs(target - prev)
        cost = float(cfg.cost_per_turnover) * turnover
        cost += abs(target) * float(cfg.slippage_bps) * 1e-4  # bps to fraction

        # Set exposure for future return realization (entry_delay)
        if cfg.entry_delay <= 0:
            exposure[t] = target
        else:
            exposure[t] = (
                prev  # current exposure unchanged; target becomes effective later
            )
            eff_t = t + int(cfg.entry_delay)
            if eff_t < T:
                # store as "desired exposure at eff_t", but in this simple model we directly
                # set exposure[eff_t] when we reach it by using prev-based recurrence:
                # we need a separate buffer. Use desired_effective array.
                pass

        # We need a buffer for delayed exposure changes
        out_turnover[t] = turnover
        out_cost[t] = cost
        out_cooldown[t] = cooldown

        # Realize pnl using current exposure and the return for current mode label at t.
        # In "delayed" mode, this effectively means action impacts later steps.
        if int(action_arr[t]) == int(Router3Action.MEAN):
            r = float(ret_mean[t])
        elif int(action_arr[t]) == int(Router3Action.TREND):
            r = float(ret_trend[t])
        else:
            r = 0.0

        pnl = float(exposure[t]) * r - cost
        equity = equity * (1.0 + pnl)
        peak = max(peak, equity)
        dd_now = (peak - equity) / peak if peak > 0 else 0.0

        out_equity[t] = equity
        out_dd[t] = dd_now
        out_pnl[t] = pnl

    # Delayed exposure changes require a second pass buffer. Implement properly:
    if cfg.entry_delay > 0:
        exposure = np.zeros(T, dtype=float)
        desired_eff = np.zeros(T, dtype=float)
        cooldown = 0
        peak = float(cfg.initial_equity)
        equity = float(cfg.initial_equity)

        for t in range(T):
            # schedule desired exposure for t+delay
            eff_t = t + int(cfg.entry_delay)
            if eff_t < T:
                desired_eff[eff_t] = desired[t]

        for t in range(T):
            if cfg.max_drawdown_stop is not None and peak > 0:
                dd = (peak - equity) / peak
                if dd + 1e-12 >= float(cfg.max_drawdown_stop):
                    cooldown = max(cooldown, int(cfg.cooldown_steps))

            if cooldown > 0:
                target = 0.0
                cooldown -= 1
            else:
                target = float(desired_eff[t])

            prev = float(exposure[t - 1]) if t > 0 else 0.0
            turnover = abs(target - prev)
            cost = float(cfg.cost_per_turnover) * turnover
            cost += abs(target) * float(cfg.slippage_bps) * 1e-4

            exposure[t] = target

            # realize return based on *effective* action mode at t
            # We don't have action_eff, so we use whichever return is chosen by scheduled desired:
            # If target==0 => NO_TRADE; if target==mean_exposure => MEAN; else TREND.
            # This is stable only if mean/trend exposures are distinct.
            if abs(target) <= 1e-12:
                r = 0.0
                mode_eff = int(Router3Action.NO_TRADE)
            else:
                mean_tgt = _clip(
                    cfg.base_exposure * cfg.mean_exposure,
                    -cfg.max_abs_exposure,
                    cfg.max_abs_exposure,
                )
                if abs(target - mean_tgt) < 1e-12:
                    r = float(ret_mean[t])
                    mode_eff = int(Router3Action.MEAN)
                else:
                    r = float(ret_trend[t])
                    mode_eff = int(Router3Action.TREND)

            pnl = float(exposure[t]) * r - cost
            equity = equity * (1.0 + pnl)
            peak = max(peak, equity)
            dd_now = (peak - equity) / peak if peak > 0 else 0.0

            out_turnover[t] = turnover
            out_cost[t] = cost
            out_cooldown[t] = cooldown
            out_equity[t] = equity
            out_dd[t] = dd_now
            out_pnl[t] = pnl
            action_arr[t] = mode_eff

    out = pd.DataFrame(
        {
            "mode_action": action_arr,
            "exposure": exposure,
            "turnover": out_turnover,
            "cost": out_cost,
            "pnl": out_pnl,
            "equity": out_equity,
            "drawdown": out_dd,
            "cooldown": out_cooldown,
        },
        index=df.index,
    )
    return out


class TradingSimEnv3Action:
    """
    Minimal gym-like environment (no external gym dependency).

    - reset() returns initial obs
    - step(action) -> (obs, reward, done, info)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        cfg: SimEnvConfig = SimEnvConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
        obs_keys: Sequence[str] = (
            "head_dir_score",
            "head_mfe_atr",
            "head_mae_atr",
            "head_t_to_mfe",
            "drawdown",
            "exposure",
        ),
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.reward_cfg = reward_cfg
        self.obs_keys = tuple(obs_keys)
        self._t = 0
        self._equity = float(cfg.initial_equity)
        self._peak = float(cfg.initial_equity)
        self._cooldown = 0
        self._exposure = 0.0
        self._desired_eff = np.zeros(len(self.df), dtype=float)
        self._prev_action: Optional[Dict[str, Any]] = None

        if (
            cfg.ret_mean_col not in self.df.columns
            or cfg.ret_trend_col not in self.df.columns
        ):
            raise ValueError("df must include return columns for mean and trend modes")

        # schedule buffer is filled during step() by applying entry_delay

    def reset(self) -> Dict[str, float]:
        self._t = 0
        self._equity = float(self.cfg.initial_equity)
        self._peak = float(self.cfg.initial_equity)
        self._cooldown = 0
        self._exposure = 0.0
        self._desired_eff = np.zeros(len(self.df), dtype=float)
        self._prev_action = None
        return self._obs()

    def _obs(self) -> Dict[str, float]:
        if self._t >= len(self.df):
            return {k: 0.0 for k in self.obs_keys}
        row = self.df.iloc[self._t].to_dict()
        dd = (self._peak - self._equity) / self._peak if self._peak > 0 else 0.0
        row["drawdown"] = dd
        row["exposure"] = float(self._exposure)
        obs = {}
        for k in self.obs_keys:
            try:
                obs[k] = float(row.get(k, 0.0) or 0.0)
            except Exception:
                obs[k] = 0.0
        return obs

    def step(
        self, action: int | Router3Action
    ) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        if self._t >= len(self.df):
            return self._obs(), 0.0, True, {}

        # schedule desired exposure for t+delay
        desired = _action_to_target_exposure(action, cfg=self.cfg)
        eff_t = self._t + int(self.cfg.entry_delay)
        if eff_t < len(self.df):
            self._desired_eff[eff_t] = desired

        # risk control: dd stop + cooldown
        dd_now = (self._peak - self._equity) / self._peak if self._peak > 0 else 0.0
        if self.cfg.max_drawdown_stop is not None and dd_now + 1e-12 >= float(
            self.cfg.max_drawdown_stop
        ):
            self._cooldown = max(self._cooldown, int(self.cfg.cooldown_steps))

        if self._cooldown > 0:
            target = 0.0
            self._cooldown -= 1
        else:
            target = float(self._desired_eff[self._t])

        turnover = abs(target - float(self._exposure))
        cost = float(self.cfg.cost_per_turnover) * turnover
        cost += abs(target) * float(self.cfg.slippage_bps) * 1e-4

        # realize return based on effective exposure target
        if abs(target) <= 1e-12:
            r = 0.0
            mode_eff = "NO_TRADE"
        else:
            mean_tgt = _clip(
                self.cfg.base_exposure * self.cfg.mean_exposure,
                -self.cfg.max_abs_exposure,
                self.cfg.max_abs_exposure,
            )
            if abs(target - mean_tgt) < 1e-12:
                r = float(self.df.at[self._t, self.cfg.ret_mean_col])
                mode_eff = "MEAN"
            else:
                r = float(self.df.at[self._t, self.cfg.ret_trend_col])
                mode_eff = "TREND"

        pnl = float(target) * r - cost
        prev_equity = float(self._equity)
        self._equity = self._equity * (1.0 + pnl)
        self._peak = max(self._peak, self._equity)
        dd_ratio = (self._peak - self._equity) / self._peak if self._peak > 0 else 0.0
        if (
            self.cfg.max_drawdown_stop is not None
            and float(self.cfg.max_drawdown_stop) > 0
        ):
            dd_ratio = dd_ratio / float(self.cfg.max_drawdown_stop)

        realized_vol = 0.0
        if self.cfg.realized_vol_col and self.cfg.realized_vol_col in self.df.columns:
            try:
                realized_vol = float(
                    self.df.at[self._t, self.cfg.realized_vol_col] or 0.0
                )
            except Exception:
                realized_vol = 0.0

        # reward uses action dict; in 3-action we record mode only
        action_dict = {"mode": mode_eff}
        reward = compute_router_reward_from_step(
            pnl=float(self._equity / prev_equity - 1.0),
            cost=float(cost),
            turnover=float(turnover),
            dd_ratio=float(dd_ratio),
            realized_vol=float(realized_vol),
            action_prev=self._prev_action,
            action_next=action_dict,
            cfg=self.reward_cfg,
        )
        self._prev_action = action_dict
        self._exposure = float(target)

        self._t += 1
        done = bool(self._t >= len(self.df))
        info = {
            "equity": float(self._equity),
            "drawdown": float(
                (self._peak - self._equity) / self._peak if self._peak > 0 else 0.0
            ),
            "turnover": float(turnover),
            "cost": float(cost),
            "pnl": float(pnl),
            "mode_eff": mode_eff,
        }
        return self._obs(), float(reward), done, info
