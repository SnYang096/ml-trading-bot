from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .reward import RewardConfig, compute_router_reward_from_step


@dataclass(frozen=True)
class ReplayTransition:
    """
    One transition for offline RL / BC.

    This is intentionally close to the doc schema:
      (state, action, reward, next_state, done, symbol, timestamp)
    """

    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    done: bool
    symbol: str
    timestamp: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": float(self.reward),
            "next_state": self.next_state,
            "done": bool(self.done),
            "symbol": str(self.symbol),
            "timestamp": str(self.timestamp),
        }


def parse_action_json(action_json: str | None) -> Dict[str, Any]:
    if not action_json:
        return {}
    try:
        obj = json.loads(action_json)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def default_state_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a minimal, RL-safe state dict from a RouterEpisodeLogger row.

    Notes:
    - We include heads and a small amount of account/risk state (equity/drawdown if present).
    - We intentionally exclude `decision` fields (router_name/score/position_size) because those are
      policy outputs in most setups.
    """
    state: Dict[str, Any] = {}

    # Context
    for k in ["timeframe", "regime_score"]:
        if k in row and row[k] is not None:
            state[k] = row[k]

    # Heads (prefixed by router_logging)
    for k in [
        "head_dir_score",
        "head_mfe_atr",
        "head_mae_atr",
        "head_t_to_mfe",
        "head_persistence",
    ]:
        if k in row and row[k] is not None:
            state[k] = float(row[k])

    # Optional account state (if logged)
    for k in ["equity", "drawdown"]:
        if k in row and row[k] is not None:
            state[k] = float(row[k])

    # Optional realized vol or risk proxy (if present in logs)
    for k in ["rolling_vol", "realized_vol", "dd_ratio"]:
        if k in row and row[k] is not None:
            state[k] = float(row[k])

    return state


def build_replay_transitions_from_router_logs(
    df: pd.DataFrame,
    *,
    reward_cfg: RewardConfig = RewardConfig(),
    state_builder=default_state_from_row,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    action_json_col: str = "action_json",
    mode_col: str = "mode",
) -> List[ReplayTransition]:
    """
    Convert RouterEpisodeLogger output (wide DataFrame) into per-step transitions.
    """
    if df is None or len(df) == 0:
        return []

    work = df.copy()
    if timestamp_col in work.columns:
        # robust ordering; keep original timestamp string for output
        work["_ts"] = pd.to_datetime(work[timestamp_col], errors="coerce", utc=True)
    else:
        work["_ts"] = pd.NaT

    if symbol_col not in work.columns:
        raise ValueError(f"Missing required column: {symbol_col}")
    if timestamp_col not in work.columns:
        raise ValueError(f"Missing required column: {timestamp_col}")

    work = work.sort_values([symbol_col, "_ts"]).reset_index(drop=True)

    transitions: List[ReplayTransition] = []
    for sym, g in work.groupby(symbol_col, sort=False):
        rows = g.to_dict(orient="records")
        if len(rows) < 2:
            continue

        prev_action: Optional[Dict[str, Any]] = None
        for i in range(len(rows) - 1):
            r0 = rows[i]
            r1 = rows[i + 1]

            # Prefer high-level mode label for scheme-B (NO_TRADE/MEAN/TREND).
            # This removes any need for router-name grouping in the dataset.
            if mode_col in r0 and r0.get(mode_col) is not None:
                action = {"mode": str(r0.get(mode_col))}
            else:
                action = parse_action_json(r0.get(action_json_col))

            # Compute reward for the step using realized outcomes logged at r0
            pnl = float(r0.get("pnl") or 0.0)
            cost = float(r0.get("cost") or 0.0)
            turnover = float(r0.get("turnover") or 0.0)

            # Prefer explicit risk proxy cols if present
            realized_vol = float(r0.get("realized_vol") or r0.get("rolling_vol") or 0.0)
            dd_ratio = float(r0.get("dd_ratio") or r0.get("drawdown") or 0.0)

            reward = compute_router_reward_from_step(
                pnl=pnl,
                cost=cost,
                turnover=turnover,
                dd_ratio=dd_ratio,
                realized_vol=realized_vol,
                action_prev=prev_action,
                action_next=action,
                cfg=reward_cfg,
            )

            state = state_builder(r0)
            next_state = state_builder(r1)
            done = bool(i == (len(rows) - 2))

            transitions.append(
                ReplayTransition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    done=done,
                    symbol=str(sym),
                    timestamp=str(r0.get(timestamp_col)),
                )
            )

            prev_action = action

    return transitions


def save_transitions_jsonl(transitions: Iterable[ReplayTransition], path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for t in transitions:
            f.write(json.dumps(t.as_dict(), ensure_ascii=False, default=str) + "\n")
    return str(p)


def load_transitions_jsonl(path: str) -> List[ReplayTransition]:
    p = Path(path)
    out: List[ReplayTransition] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                ReplayTransition(
                    state=dict(obj.get("state") or {}),
                    action=dict(obj.get("action") or {}),
                    reward=float(obj.get("reward") or 0.0),
                    next_state=dict(obj.get("next_state") or {}),
                    done=bool(obj.get("done")),
                    symbol=str(obj.get("symbol") or ""),
                    timestamp=str(obj.get("timestamp") or ""),
                )
            )
    return out
