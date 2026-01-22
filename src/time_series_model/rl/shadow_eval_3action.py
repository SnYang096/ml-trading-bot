from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .bc_dataset import (
    BCStateSchema,
    Router3Action,
    Router3ActionInferConfig,
    infer_router3_action,
)
from .bc_trainer_3action import BC3PolicyMLP, BC3TrainConfig, train_bc_router3_policy
from .walk_forward import WalkForwardSplitConfig, time_ordered_split_by_symbol


@dataclass(frozen=True)
class ShadowEvalConfig:
    """
    Offline shadow evaluation for BC(3-action) against rule-logged archetype labels.

    This is *not* a counterfactual PnL simulator. It is a deployment gate:
    - does BC reproduce rule behavior?
    - is the action distribution stable?
    - does it collapse to a single mode?

    Note: Mode is inferred from archetype (TC/TE → TREND, FR/ET → MEAN, others → NO_TRADE).
    """

    archetype_col: str = "gate_archetype"  # Primary: use gate_archetype if available
    regime_col: str = "regime"  # Fallback: use regime if archetype not available
    mode_col: str = (
        "mode"  # Legacy fallback: use mode if archetype/regime not available
    )
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"

    # Which columns to use as BC observation vector
    state_keys: Sequence[str] = (
        "head_dir_score",
        "head_mfe_atr",
        "head_mae_atr",
        "head_t_to_mfe",
        "drawdown",
    )

    split_cfg: WalkForwardSplitConfig = WalkForwardSplitConfig()
    bc_cfg: BC3TrainConfig = BC3TrainConfig(
        epochs=5, batch_size=256, hidden=128, depth=2, dropout=0.1
    )


def _infer_mode_from_archetype(archetype: Any, regime: Any = None) -> str:
    """
    Infer mode (TREND/MEAN/NO_TRADE) from archetype or regime.

    Rules:
    - TC/TE → TREND
    - FR/ET → MEAN
    - Others or missing → NO_TRADE
    - If archetype not available, use regime (TC_REGIME/TE_REGIME → TREND, MEAN_REGIME → MEAN)
    """
    if archetype is not None:
        arch_str = str(archetype).upper()
        if "TC" in arch_str or "TE" in arch_str:
            return "TREND"
        elif "FR" in arch_str or "ET" in arch_str:
            return "MEAN"

    # Fallback to regime
    if regime is not None:
        reg_str = str(regime).upper()
        if "TC_REGIME" in reg_str or "TE_REGIME" in reg_str or reg_str == "TREND":
            return "TREND"
        elif "MEAN_REGIME" in reg_str or reg_str == "MEAN":
            return "MEAN"

    return "NO_TRADE"


def _as_transition_rows(
    df: pd.DataFrame,
    *,
    state_keys: Sequence[str],
    archetype_col: str,
    regime_col: str,
    mode_col: str,
) -> List[Dict[str, Any]]:
    """
    Convert dataframe rows to transition rows for BC training.
    Infers mode from archetype (preferred) or regime (fallback) or mode (legacy fallback).
    """
    rows = []
    for r in df.to_dict(orient="records"):
        state = {k: r.get(k, 0.0) for k in state_keys}

        # Try to infer mode from archetype/regime/mode in order of preference
        archetype = r.get(archetype_col)
        regime = r.get(regime_col)
        mode = r.get(mode_col)

        if archetype is not None or regime is not None:
            inferred_mode = _infer_mode_from_archetype(archetype, regime)
        elif mode is not None:
            inferred_mode = str(mode).upper()
        else:
            inferred_mode = "NO_TRADE"

        action = {"mode": inferred_mode}
        rows.append({"state": state, "action": action})
    return rows


def _entropy_from_counts(counts: Dict[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = float(c) / total
        if p > 0:
            ent -= p * math.log(p)
    return float(ent)


def _switch_rate(series: Sequence[int]) -> float:
    if not series:
        return 0.0
    switches = 0
    for i in range(1, len(series)):
        if int(series[i]) != int(series[i - 1]):
            switches += 1
    return float(switches) / float(max(1, len(series) - 1))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(yt) <= 2 and 0 <= int(yp) <= 2:
            cm[int(yt), int(yp)] += 1
    return cm


def _render_html_report(
    *,
    meta: Dict[str, Any],
    metrics: Dict[str, float],
    cm: np.ndarray,
    sample: pd.DataFrame,
) -> str:
    def esc(x: Any) -> str:
        s = "" if x is None else str(x)
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    rows = []
    for k in sorted(metrics.keys()):
        rows.append(
            f"<tr><td><code>{esc(k)}</code></td><td>{esc(f'{metrics[k]:.6g}')}</td></tr>"
        )

    cm_df = pd.DataFrame(
        cm, index=["NO_TRADE", "MEAN", "TREND"], columns=["NO_TRADE", "MEAN", "TREND"]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shadow Eval - BC(3-action)</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color: #111; }}
    code {{ background: #f4f4f5; padding: 2px 6px; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 16px; background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #eee; text-align: left; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #fafafa; position: sticky; top: 0; }}
    .muted {{ color: #6b7280; }}
    details > summary {{ cursor: pointer; font-weight: 600; margin: 6px 0; }}
  </style>
</head>
<body>
  <h1>Shadow Eval - BC(3-action)</h1>
  <p class="muted">Offline gate: behavior match + stability. Not a counterfactual PnL simulator.</p>

  <div class="card" style="margin:16px 0;">
    <h2 style="margin-top:0;">How to read this report</h2>
    <ul style="margin:0; padding-left:18px;">
      <li>This checks whether a small BC Router can reproduce the logged <code>mode</code> decisions from the rule router.</li>
      <li><b>acc_vs_rule_mode</b>: fraction of test steps where BC predicted the same mode as rule.</li>
      <li><b>confusion matrix</b>: where BC disagrees (e.g., MEAN → NO_TRADE collapse).</li>
      <li><b>mode_entropy_pred</b> near 0 and <b>pred_rate_no_trade</b> near 1 means mode collapse (bad for deployment).</li>
      <li>This report is a stability gate only; it does <b>not</b> measure trading PnL.</li>
    </ul>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Metrics</h2>
      <table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </div>
    <div class="card">
      <h2>Confusion matrix</h2>
      {cm_df.to_html(escape=True)}
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Sample (test)</h2>
    {sample.to_html(index=False, escape=True) if len(sample) else "<em>No sample.</em>"}
  </div>

  <details style="margin-top:16px;">
    <summary>Raw meta (json)</summary>
    <pre>{esc(json.dumps(meta, ensure_ascii=False, indent=2, default=str))}</pre>
  </details>
</body>
</html>
"""


def train_and_shadow_eval_bc3_from_logs(
    df_logs: pd.DataFrame,
    *,
    cfg: ShadowEvalConfig = ShadowEvalConfig(),
    out_dir: Optional[str] = None,
) -> Tuple[BC3PolicyMLP, Dict[str, Any], Dict[str, float]]:
    """
    Train BC(3-action) on train split and evaluate on test split.

    Returns (model, meta, metrics). If out_dir is provided, also saves:
      - meta.json
      - metrics.json
      - confusion.csv
      - shadow_sample.csv
      - shadow_report.html
    """
    # Check for at least one of archetype/regime/mode columns
    has_archetype = cfg.archetype_col in df_logs.columns
    has_regime = cfg.regime_col in df_logs.columns
    has_mode = cfg.mode_col in df_logs.columns

    if not (has_archetype or has_regime or has_mode):
        raise ValueError(
            f"Missing required columns. Need at least one of: "
            f"'{cfg.archetype_col}' (preferred), '{cfg.regime_col}' (fallback), "
            f"or '{cfg.mode_col}' (legacy fallback) in logs."
        )

    train_df, test_df = time_ordered_split_by_symbol(df_logs, cfg=cfg.split_cfg)
    state_schema = BCStateSchema(keys=list(cfg.state_keys))

    # In new system, mode is inferred from archetype; infer_cfg is unused but required by trainer signature.
    infer_cfg = Router3ActionInferConfig(mean_routers=[], trend_routers=[])
    transitions_train = _as_transition_rows(
        train_df,
        state_keys=cfg.state_keys,
        archetype_col=cfg.archetype_col,
        regime_col=cfg.regime_col,
        mode_col=cfg.mode_col,
    )

    model, meta = train_bc_router3_policy(
        transitions=transitions_train,
        state_schema=state_schema,
        infer_cfg=infer_cfg,
        cfg=cfg.bc_cfg,
    )

    # Evaluate on test
    test_rows = _as_transition_rows(
        test_df,
        state_keys=cfg.state_keys,
        archetype_col=cfg.archetype_col,
        regime_col=cfg.regime_col,
        mode_col=cfg.mode_col,
    )
    y_true = np.asarray(
        [int(infer_router3_action(r["action"], cfg=infer_cfg)) for r in test_rows],
        dtype=np.int64,
    )
    X = (
        np.stack(
            [state_schema.encode_state(r["state"]) for r in test_rows], axis=0
        ).astype(np.float32)
        if test_rows
        else np.zeros((0, state_schema.obs_dim), dtype=np.float32)
    )

    device = cfg.bc_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = (
            model(torch.from_numpy(X).to(device))
            if len(X)
            else torch.zeros((0, 3), device=device)
        )
        y_pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)

    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    cm = _confusion(y_true, y_pred)

    # behavior metrics per symbol
    pred_series = (
        pd.Series(y_pred, index=test_df.index) if len(test_df) else pd.Series(dtype=int)
    )
    true_series = (
        pd.Series(y_true, index=test_df.index) if len(test_df) else pd.Series(dtype=int)
    )

    switch_pred = 0.0
    switch_true = 0.0
    n_sym = 0
    for sym, g in test_df.groupby(cfg.symbol_col, sort=False):
        idx = g.index
        sp = [int(x) for x in pred_series.loc[idx].tolist()]
        st = [int(x) for x in true_series.loc[idx].tolist()]
        switch_pred += _switch_rate(sp)
        switch_true += _switch_rate(st)
        n_sym += 1
    if n_sym > 0:
        switch_pred /= n_sym
        switch_true /= n_sym

    # mode distribution (test)
    def _counts(arr: np.ndarray) -> Dict[str, int]:
        names = {0: "NO_TRADE", 1: "MEAN", 2: "TREND"}
        c: Dict[str, int] = {v: 0 for v in names.values()}
        for x in arr.tolist():
            c[names.get(int(x), "NO_TRADE")] += 1
        return c

    pred_counts = _counts(y_pred)
    true_counts = _counts(y_true)

    metrics: Dict[str, float] = {
        "test_n": float(len(test_df)),
        "acc_vs_rule_mode": float(acc),
        "switch_rate_pred": float(switch_pred),
        "switch_rate_rule": float(switch_true),
        "mode_entropy_pred": float(_entropy_from_counts(pred_counts)),
        "mode_entropy_rule": float(_entropy_from_counts(true_counts)),
        "pred_rate_no_trade": float(pred_counts["NO_TRADE"])
        / float(max(1, len(y_pred))),
        "pred_rate_mean": float(pred_counts["MEAN"]) / float(max(1, len(y_pred))),
        "pred_rate_trend": float(pred_counts["TREND"]) / float(max(1, len(y_pred))),
    }

    meta_out = {
        "shadow_cfg": {
            "mode_col": cfg.mode_col,
            "state_keys": list(cfg.state_keys),
            "split_cfg": asdict(cfg.split_cfg),
        },
        "train_meta": meta,
    }

    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "meta.json").write_text(
            json.dumps(meta_out, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (p / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        pd.DataFrame(
            cm,
            index=["NO_TRADE", "MEAN", "TREND"],
            columns=["NO_TRADE", "MEAN", "TREND"],
        ).to_csv(p / "confusion.csv")

        sample = test_df.copy()
        if len(sample):
            sample = sample.assign(
                rule_mode=true_series.values, pred_mode=pred_series.values
            ).tail(50)
        sample.to_csv(p / "shadow_sample.csv", index=False)

        html = _render_html_report(
            meta=meta_out, metrics=metrics, cm=cm, sample=sample.tail(20)
        )
        (p / "shadow_report.html").write_text(html, encoding="utf-8")

    return model, meta_out, metrics
