from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.rl.fallback_fsm import (
    FallbackFSM,
    GateConfig,
    GateInputs,
    RouterControlState,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply RL fallback FSM decision using counterfactual metrics.json."
    )
    ap.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics.json produced by rl_counterfactual_eval_3action.py",
    )
    ap.add_argument(
        "--state",
        default="RL_CANDIDATE",
        help="Initial FSM state: RULE/RL_CANDIDATE/RL_ACTIVE/RL_SUSPENDED",
    )
    ap.add_argument(
        "--promote_days",
        type=int,
        default=10,
        help="Consecutive ok windows required to promote.",
    )
    ap.add_argument(
        "--cooldown_days",
        type=int,
        default=20,
        help="Cooldown windows after suspension.",
    )
    ap.add_argument("--out", default=None, help="Optional path to write decision json.")
    args = ap.parse_args()

    m = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    eps = 1e-9
    dd_rule = max(float(m.get("rule_avg_max_dd", 0.0)), eps)
    dd_rl = max(float(m.get("pred_avg_max_dd", 0.0)), eps)
    pnl_dd_rule = float(m.get("rule_avg_total_return", 0.0)) / dd_rule
    pnl_dd_rl = float(m.get("pred_avg_total_return", 0.0)) / dd_rl

    inp = GateInputs(
        max_dd_rule=dd_rule,
        max_dd_rl=dd_rl,
        switch_rate_rule=float(m.get("rule_avg_switch_rate", 0.0)),
        switch_rate_rl=float(m.get("pred_avg_switch_rate", 0.0)),
        pnl_dd_rule=pnl_dd_rule,
        pnl_dd_rl=pnl_dd_rl,
    )

    fsm = FallbackFSM(
        cfg=GateConfig(
            promote_min_days=int(args.promote_days),
            cooldown_days=int(args.cooldown_days),
        )
    )
    fsm.state = RouterControlState(str(args.state))
    out = fsm.step(inp)
    payload = json.dumps(out, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(str(args.out)).parent.mkdir(parents=True, exist_ok=True)
        Path(str(args.out)).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
