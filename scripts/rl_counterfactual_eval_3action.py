from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.rl.counterfactual_eval_3action import (
    CounterfactualEvalConfig,
    train_and_counterfactual_eval_bc3,
)
from src.time_series_model.rl.sim_env_3action import SimEnvConfig
from src.time_series_model.rl.walk_forward import WalkForwardSplitConfig
from src.time_series_model.rule.router_3action import Rule3ActionConfig
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.state import ConstitutionState
from src.time_series_model.diagnostics.kpi_gate import run_kpi_gate
from src.time_series_model.ops.state_snapshot import (
    HumanOverride,
    SystemStateSnapshot,
    write_state_snapshot,
)


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Counterfactual eval Rule vs BC(3-action) using ret_mean/ret_trend columns."
    )
    ap.add_argument(
        "--logs",
        required=True,
        help="Path to logs .csv/.parquet with mode + ret_mean/ret_trend + head_*",
    )
    ap.add_argument(
        "--out", required=True, help="Output directory for report artifacts."
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train ratio per symbol (time-ordered).",
    )
    ap.add_argument(
        "--entry_delay", type=int, default=0, help="Entry delay steps for sim."
    )
    ap.add_argument(
        "--cost_per_turnover", type=float, default=0.0, help="Cost per turnover unit."
    )
    ap.add_argument(
        "--slippage_bps", type=float, default=0.0, help="Slippage bps per abs exposure."
    )
    ap.add_argument(
        "--preds-in-log1p",
        type=int,
        default=1,
        help="Whether head_mfe/head_mae/head_t_to_mfe are in log1p space (1=yes, 0=no).",
    )
    ap.add_argument(
        "--router-mfe-min",
        type=float,
        default=None,
        help="Router threshold override: mfe_min (for report diagnostics).",
    )
    ap.add_argument(
        "--router-eff-min",
        type=float,
        default=None,
        help="Router threshold override: eff_min (for report diagnostics).",
    )
    ap.add_argument(
        "--router-dir-conf-trend-min",
        type=float,
        default=None,
        help="Router threshold override: dir_conf_trend_min (for report diagnostics).",
    )
    ap.add_argument(
        "--survival-preds",
        default=None,
        help="Optional survival preds parquet (must contain symbol,timestamp,survival_prob). If provided, will be merged into logs for report + extra baseline.",
    )
    ap.add_argument(
        "--survival-prob-col",
        default="survival_prob",
        help="Column name for survival probability after merging (default: survival_prob).",
    )
    ap.add_argument(
        "--ood-score-col",
        default="ood_score",
        help="Optional ood score column name in logs (default: ood_score).",
    )
    ap.add_argument(
        "--ood-config",
        default="config/ood/ood_config_v1.yaml",
        help="OOD config YAML used to map (ood,survival)->size cap for the extra baseline.",
    )
    args = ap.parse_args()

    df = _read_any(args.logs)
    if args.survival_preds:
        sp = _read_any(args.survival_preds)
        need = {"symbol", "timestamp", str(args.survival_prob_col)}
        if not need.issubset(set(sp.columns)):
            raise ValueError(
                f"--survival-preds missing required cols: {sorted(list(need))}, got {list(sp.columns)}"
            )
        sp["timestamp"] = pd.to_datetime(sp["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.merge(
            sp[["symbol", "timestamp", str(args.survival_prob_col)]],
            on=["symbol", "timestamp"],
            how="left",
        )
    router_cfg = Rule3ActionConfig()
    if args.router_mfe_min is not None:
        router_cfg = Rule3ActionConfig(
            mfe_min=float(args.router_mfe_min),
            eff_min=(
                float(args.router_eff_min)
                if args.router_eff_min is not None
                else float(router_cfg.eff_min)
            ),
            dir_conf_trend_min=(
                float(args.router_dir_conf_trend_min)
                if args.router_dir_conf_trend_min is not None
                else float(router_cfg.dir_conf_trend_min)
            ),
        )
    cfg = CounterfactualEvalConfig(
        split_cfg=WalkForwardSplitConfig(train_ratio=float(args.train_ratio)),
        sim_cfg=SimEnvConfig(
            entry_delay=int(args.entry_delay),
            cost_per_turnover=float(args.cost_per_turnover),
            slippage_bps=float(args.slippage_bps),
        ),
        router_cfg=router_cfg,
        preds_in_log1p=bool(int(args.preds_in_log1p)),
        portfolio_assets_yaml=str(
            os.environ.get("MLBOT_PORTFOLIO_ASSETS_YAML") or ""
        ).strip()
        or None,
    )
    # Optional: let library compute extra baseline if cols exist
    try:
        cfg = CounterfactualEvalConfig(
            **{
                **cfg.__dict__,
                "survival_prob_col": str(args.survival_prob_col),
                "ood_score_col": str(args.ood_score_col),
                "ood_config_yaml": str(args.ood_config),
            }
        )
    except Exception:
        pass

    Path(args.out).mkdir(parents=True, exist_ok=True)
    _, metrics, _ = train_and_counterfactual_eval_bc3(
        df, cfg=cfg, out_dir=str(args.out)
    )

    # -----------------------------------------------------------------------------
    # V1.1 enforcement hooks (opt-in via env) — prevents "PPT-only" KPIs/constitution
    # -----------------------------------------------------------------------------
    task_id = str(os.environ.get("MLBOT_TASK_ID") or "").strip() or None
    constitution_yaml = (
        str(os.environ.get("MLBOT_CONSTITUTION_YAML") or "").strip() or None
    )
    kpi_gate_yaml = str(os.environ.get("MLBOT_KPI_GATE_YAML") or "").strip() or None

    override_tag = str(os.environ.get("MLBOT_HUMAN_OVERRIDE_TAG") or "").strip() or None
    override_reason = (
        str(os.environ.get("MLBOT_HUMAN_OVERRIDE_REASON") or "").strip() or None
    )
    overrides = (
        [HumanOverride(tag=override_tag, reason=override_reason)]
        if override_tag and override_reason
        else []
    )

    # Constitution check (kill-switch style) using counterfactual dd metric.
    constitution_meta = {}
    if constitution_yaml:
        ex = ConstitutionExecutor(constitution_yaml=constitution_yaml)
        constitution_meta = ex.meta()
        st = ConstitutionState(
            task_id=task_id,
            # best-effort: use rule-side mean max dd as a proxy
            drawdown=float(metrics.get("rule_avg_max_dd", 0.0)),
        )
        ex.validate_drawdown(state=st)

    # KPI gate: hard-fail CI if gate says no.
    kpi_gate_res = None
    if kpi_gate_yaml:
        metrics_json = str(Path(args.out) / "metrics.json")
        out_json = str(Path(args.out) / "kpi_gate_result.json")
        rc, res = run_kpi_gate(
            metrics_json=metrics_json, gate_yaml=kpi_gate_yaml, out_json=out_json
        )
        kpi_gate_res = res.as_dict()
        if rc != 0:
            raise SystemExit(int(rc))

    # Always write a minimal snapshot for attribution/replay.
    snap = SystemStateSnapshot(
        task_id=task_id,
        timestamp=None,
        constitution_hash=(
            str(constitution_meta.get("constitution_hash"))
            if constitution_meta
            else None
        ),
        constitution_yaml=constitution_yaml,
        router_mode=None,
        gate_decisions={},
        pcm_budget={},
        active_slots=None,
        drawdown=float(metrics.get("rule_avg_max_dd", 0.0)),
        kpi_gate=kpi_gate_res,
        overrides=overrides,
    )
    # If portfolio assets summary exists, include it for attribution.
    try:
        pa_path = Path(args.out) / "portfolio_assets_summary.json"
        if pa_path.exists():
            snap.pcm_budget.update(
                json.loads(pa_path.read_text(encoding="utf-8")) or {}
            )
    except Exception:
        pass
    write_state_snapshot(
        out_path=str(Path(args.out) / "system_state_snapshot.json"), snapshot=snap
    )

    print("counterfactual metrics:", metrics)
    print("saved to:", args.out)


if __name__ == "__main__":
    main()
