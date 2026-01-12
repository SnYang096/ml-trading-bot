"""
RL utilities for the time-series trading system.

Design principle:
- RL operates ONLY at the Router/Allocator layer (decision management).
- RL must NOT directly predict prices, nor directly decide entry/exit mechanics.
"""

from .router_types import RouterAction, RouterContext, RouterDecision, RouterHeads
from .router_logging import RouterStepLog, RouterEpisodeLogger
from .reward import RewardConfig, compute_step_reward
from .router_policy import (
    AppliedDecision,
    apply_action_to_decision,
    apply_action_to_decisions,
)
from .replay_buffer import ReplayTransition, build_replay_transitions_from_router_logs
from .bc_dataset import (
    BCPolicySchema,
    BCStateSchema,
    BCDataset,
    bc_collate_fn,
    BCRouter3ActionDataset,
    Router3Action,
    Router3ActionInferConfig,
    bc3_collate_fn,
    infer_router3_action,
)
from .walk_forward import WalkForwardSplitConfig, time_ordered_split_by_symbol
from .shadow_eval_3action import ShadowEvalConfig, train_and_shadow_eval_bc3_from_logs
from .sim_env_3action import (
    SimEnvConfig,
    TradingSimEnv3Action,
    simulate_3action_episode,
)
from .fallback_fsm import (
    FallbackFSM,
    GateConfig,
    GateInputs,
    RouterControlState,
    evaluate_gates,
)
