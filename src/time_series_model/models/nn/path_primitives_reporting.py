from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .path_primitives_dataset import DatasetConfig, build_feature_matrix
from .path_primitives_eval import EvalConfig, evaluate_path_primitives
from .path_primitives_labels import (
    PathPrimitivesLabelConfig,
    compute_path_primitives_labels,
)
from .path_primitives_model import MultiHeadPathPrimitivesMLP
from .path_primitives_conditions import SRFuseConditionConfig, compute_near_sr_mask


def _html_escape(s: object) -> str:
    text = "" if s is None else str(s)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _metrics_to_html_table(metrics: Dict[str, float]) -> str:
    rows = []
    for k in sorted(metrics.keys()):
        v = metrics[k]
        try:
            v_str = f"{float(v):.6g}"
        except Exception:
            v_str = _html_escape(v)
        rows.append(
            f"<tr><td><code>{_html_escape(k)}</code></td><td>{_html_escape(v_str)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>metric</th><th>value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _history_to_html(meta: Dict) -> str:
    hist = (meta or {}).get("history") or {}
    train = hist.get("train") or []
    val = hist.get("val") or []
    if not train and not val:
        return "<p><em>No training history found in meta.</em></p>"

    # show last 10 rows for readability
    def _df(rows):
        if not rows:
            return None
        df = pd.DataFrame(rows)
        keep = [
            c for c in ["epoch", "total", "dir", "mfe", "mae", "t"] if c in df.columns
        ]
        if keep:
            df = df[keep]
        return df.tail(10)

    parts = []
    dft = _df(train)
    if dft is not None and len(dft) > 0:
        parts.append("<h4>train (last 10)</h4>")
        parts.append(dft.to_html(index=False, escape=True))
    dfv = _df(val)
    if dfv is not None and len(dfv) > 0:
        parts.append("<h4>val (last 10)</h4>")
        parts.append(dfv.to_html(index=False, escape=True))
    return "\n".join(parts)


def _fmt(x: Any, *, digits: int = 4) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "N/A"
        return f"{v:.{digits}f}"
    except Exception:
        return "N/A"


def _metrics_summary_md(metrics: Dict[str, float]) -> str:
    """
    Human-facing short summary of key training metrics.
    This is intentionally heuristic (not a scientific proof).
    """
    m = metrics or {}
    lines: List[str] = []

    # Direction head
    dir_acc = m.get("dir_acc")
    dir_auc = m.get("dir_auc")
    lines.append(f"- **dir_acc**: {_fmt(dir_acc)} (0.50≈随机，>0.55 才算有一点信息量)")
    lines.append(f"- **dir_auc**: {_fmt(dir_auc)} (0.50≈随机；越接近 1 越好)")

    # Regression heads: rank-IC style is more robust than absolute MAE
    for head in ["mfe_atr", "mae_atr", "t_to_mfe"]:
        sp = m.get(f"{head}_spearman")
        mae = m.get(f"{head}_mae")
        lines.append(
            f"- **{head}**: spearman={_fmt(sp)} (0≈无信息) / mae={_fmt(mae, digits=3)} (量纲敏感，用于 sanity)"
        )

    mask_rate = m.get("mask_rate")
    if mask_rate is not None:
        lines.append(
            f"- **mask_rate**: {_fmt(mask_rate)} (有效标签比例；越低代表样本稀疏/标签不可用)"
        )

    # Rolling IC/ICIR
    if "roll_icir__dir" in m:
        lines.append(
            f"- **roll_icir__dir**: {_fmt(m.get('roll_icir__dir'))} (滚动 Rank-ICIR；>0 表示有轻微稳定相关)"
        )
    if "roll_icir__mfe_atr" in m:
        lines.append(
            f"- **roll_icir__mfe_atr**: {_fmt(m.get('roll_icir__mfe_atr'))} (滚动 Rank-ICIR)"
        )
    if "roll_icir__mae_atr" in m:
        lines.append(
            f"- **roll_icir__mae_atr**: {_fmt(m.get('roll_icir__mae_atr'))} (滚动 Rank-ICIR；负值通常表示反相关或不稳定)"
        )
    if "roll_icir__t_to_mfe" in m:
        lines.append(
            f"- **roll_icir__t_to_mfe**: {_fmt(m.get('roll_icir__t_to_mfe'))} (滚动 Rank-ICIR)"
        )

    # Conditional slice: trend_high
    if "trend_high__rate" in m:
        lines.append(
            f"- **trend_high__rate**: {_fmt(m.get('trend_high__rate'))} (trend_high 切片占比；用于检查分层样本量)"
        )
        if "trend_high__roll_icir__dir" in m:
            lines.append(
                f"- **trend_high__roll_icir__dir**: {_fmt(m.get('trend_high__roll_icir__dir'))} (trend_high 内滚动 ICIR)"
            )

    return "\n".join(lines)


def _metrics_summary_html(metrics: Dict[str, float]) -> str:
    md = _metrics_summary_md(metrics or {})
    # very small markdown->html: bullets only
    items = []
    for line in md.splitlines():
        if line.strip().startswith("- "):
            items.append(f"<li>{_html_escape(line.strip()[2:])}</li>")
    return (
        '<div class="card" style="margin-top:16px;">'
        "<h2>Summary (how to interpret key metrics)</h2>"
        '<p class="muted">这些是启发式解读，用于快速发现：是否接近随机、是否存在量纲/标签/归一化问题、以及滚动稳定性。</p>'
        f"<ul>{''.join(items)}</ul>"
        "</div>"
    )


def render_html_dashboard(
    *,
    meta: Dict,
    metrics: Optional[Dict[str, float]],
    df_pred_sample: Optional[pd.DataFrame],
    title: str = "Path Primitives - Report",
) -> str:
    """
    Render a single-file, dependency-free HTML dashboard.
    Designed for offline environments (no CDN / no JS dependencies).
    """
    metrics = metrics or {}
    # split conditional metrics (e.g. near_sr__)
    cond = {k: v for k, v in metrics.items() if "__" in k}
    base = {k: v for k, v in metrics.items() if "__" not in k}

    sample_html = "<p><em>No pred sample provided.</em></p>"
    if df_pred_sample is not None and len(df_pred_sample) > 0:
        sample_html = df_pred_sample.to_html(index=True, escape=True)

    model_cfg = (meta or {}).get("model_cfg") or {}
    label_cfg = (meta or {}).get("label_cfg") or {}
    dataset_cfg = (meta or {}).get("dataset_cfg") or {}
    n_samples = (meta or {}).get("n_samples")
    rolling_ic = (meta or {}).get("rolling_ic") or {}
    rolling_cfg = rolling_ic.get("cfg") or {}
    preview_by_slice = rolling_ic.get("preview_by_slice") or {}

    rolling_html = "<p><em>No rolling IC preview.</em></p>"
    if isinstance(preview_by_slice, dict) and preview_by_slice:
        parts = []
        parts.append(
            '<p class="muted">Rolling Rank-IC (Spearman) computed per symbol, then averaged by timestamp. '
            "Shown as a tail preview for quick drift inspection.</p>"
        )
        if rolling_cfg:
            parts.append(
                f"<p><b>rolling_window</b>: <code>{_html_escape(rolling_cfg.get('window'))}</code> &nbsp; "
                f"<b>min_periods</b>: <code>{_html_escape(rolling_cfg.get('min_periods'))}</code></p>"
            )
        for name, rows in preview_by_slice.items():
            try:
                dfp = pd.DataFrame(rows)
                parts.append(f"<details open><summary>{_html_escape(name)}</summary>")
                if len(dfp) > 0:
                    parts.append(dfp.to_html(index=False, escape=True))
                else:
                    parts.append("<p><em>Empty preview.</em></p>")
                parts.append("</details>")
            except Exception:
                continue
        rolling_html = "\n".join(parts)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_html_escape(title)}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color: #111; }}
    h1,h2,h3,h4 {{ margin: 0.6em 0 0.4em; }}
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
  <h1>{_html_escape(title)}</h1>
  <p class="muted">Generated by <code>save_train_artifacts</code>. Single-file report; open directly in browser.</p>

  {_metrics_summary_html(metrics)}

  <div class="grid">
    <div class="card">
      <h2>Run info</h2>
      <table>
        <tbody>
          <tr><td><code>n_samples</code></td><td>{_html_escape(n_samples)}</td></tr>
          <tr><td><code>model_cfg</code></td><td><pre style="margin:0; white-space:pre-wrap;">{_html_escape(json.dumps(model_cfg, ensure_ascii=False, indent=2, default=str))}</pre></td></tr>
          <tr><td><code>label_cfg</code></td><td><pre style="margin:0; white-space:pre-wrap;">{_html_escape(json.dumps(label_cfg, ensure_ascii=False, indent=2, default=str))}</pre></td></tr>
          <tr><td><code>dataset_cfg</code></td><td><pre style="margin:0; white-space:pre-wrap;">{_html_escape(json.dumps(dataset_cfg, ensure_ascii=False, indent=2, default=str))}</pre></td></tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Metrics (global)</h2>
      {_metrics_to_html_table(base) if base else "<p><em>No global metrics.</em></p>"}

      <details style="margin-top:10px;" open>
        <summary>Conditional metrics</summary>
        {_metrics_to_html_table(cond) if cond else "<p><em>No conditional metrics.</em></p>"}
      </details>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Rolling IC/ICIR (preview)</h2>
    {rolling_html}
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Training history</h2>
    {_history_to_html(meta)}
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Prediction sample</h2>
    {sample_html}
  </div>

  <details style="margin-top:16px;">
    <summary>Raw meta (json)</summary>
    <pre>{_html_escape(json.dumps(meta, ensure_ascii=False, indent=2, default=str))}</pre>
  </details>
</body>
</html>
"""


def predict_path_primitives(
    *,
    model: MultiHeadPathPrimitivesMLP,
    df: pd.DataFrame,
    feature_cols: List[str],
    fill_nan_value: float = 0.0,
    block_cols_by_name: Optional[Dict[str, List[str]]] = None,
    append_block_mask: bool = False,
    device: Optional[str] = None,
    feature_scaler: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Run model inference on a dataframe and return prediction columns aligned to df.index.

    Note: regression heads are in the same space as training targets (often log1p).

    Args:
        feature_scaler: Optional dict with 'mean', 'std', 'eps' for z-score normalization.
                       If provided, features are scaled before inference.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    X = build_feature_matrix(
        df,
        feature_cols,
        fill_nan_value=fill_nan_value,
        block_cols_by_name=block_cols_by_name,
        append_block_mask=append_block_mask,
    )

    # Apply feature scaling if scaler is provided
    if feature_scaler is not None:
        mean = np.array(feature_scaler["mean"])
        std = np.array(feature_scaler["std"])
        eps = feature_scaler.get("eps", 1e-8)
        n_features = len(mean)
        # Only scale feature columns (not block mask columns)
        X[:, :n_features] = (X[:, :n_features] - mean) / (std + eps)

    x = torch.as_tensor(X, dtype=torch.float32, device=dev)
    with torch.no_grad():
        out = model(x)
        dir_prob = torch.sigmoid(out["dir_logit"]).detach().cpu().numpy()

        preds = pd.DataFrame(
            {
                "pred_dir_prob": dir_prob,
                "pred_mfe_atr": out["mfe_atr"].detach().cpu().numpy(),
                "pred_mae_atr": out["mae_atr"].detach().cpu().numpy(),
                "pred_t_to_mfe": out["t_to_mfe"].detach().cpu().numpy(),
            },
            index=df.index,
        )
        if "persistence" in out:
            preds["pred_persistence"] = out["persistence"].detach().cpu().numpy()
    return preds


def evaluate_model_on_df(
    *,
    model: MultiHeadPathPrimitivesMLP,
    df_features: pd.DataFrame,
    feature_cols: List[str],
    label_cfg: PathPrimitivesLabelConfig,
    dataset_cfg: DatasetConfig = DatasetConfig(),
    eval_cfg: EvalConfig = EvalConfig(),
    group_col: Optional[str] = None,
    block_cols_by_name: Optional[Dict[str, List[str]]] = None,
    append_block_mask: bool = False,
    feature_scaler: Optional[Dict] = None,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Any]]:
    """
    Compute labels, run predictions, and return (metrics, merged_frame).
    """
    df_labels = compute_path_primitives_labels(
        df_features, cfg=label_cfg, group_col=group_col
    )
    work = df_features.join(df_labels)

    preds = predict_path_primitives(
        model=model,
        df=work,
        feature_cols=feature_cols,
        fill_nan_value=dataset_cfg.fill_nan_value,
        block_cols_by_name=block_cols_by_name,
        append_block_mask=append_block_mask,
        feature_scaler=feature_scaler,
    )
    work = work.join(preds)

    # For fair evaluation, transform true labels to match training space (log1p by default)
    true_mfe = pd.to_numeric(work[dataset_cfg.mfe_atr_col], errors="coerce")
    true_mae = pd.to_numeric(work[dataset_cfg.mae_atr_col], errors="coerce")
    true_t = pd.to_numeric(work[dataset_cfg.t_to_mfe_col], errors="coerce")

    if dataset_cfg.clamp_targets:
        true_mfe = true_mfe.clip(lower=0.0, upper=float(dataset_cfg.cap_mfe_atr))
        true_mae = true_mae.clip(lower=0.0, upper=float(dataset_cfg.cap_mae_atr))
        true_t = true_t.clip(lower=0.0)
    if dataset_cfg.log1p_targets:
        work["_true_mfe_atr_tr"] = np.log1p(true_mfe)
        work["_true_mae_atr_tr"] = np.log1p(true_mae)
        work["_true_t_to_mfe_tr"] = np.log1p(true_t)
    else:
        work["_true_mfe_atr_tr"] = true_mfe
        work["_true_mae_atr_tr"] = true_mae
        work["_true_t_to_mfe_tr"] = true_t

    metrics_all = evaluate_path_primitives(
        df=work,
        pred_cols={
            "dir": "pred_dir_prob",
            "mfe_atr": "pred_mfe_atr",
            "mae_atr": "pred_mae_atr",
            "t_to_mfe": "pred_t_to_mfe",
        },
        true_cols={
            "dir_y": dataset_cfg.dir_y_col,
            "mfe_atr": "_true_mfe_atr_tr",
            "mae_atr": "_true_mae_atr_tr",
            "t_to_mfe": "_true_t_to_mfe_tr",
        },
        mask_col=(
            dataset_cfg.mfe_valid_col
            if dataset_cfg.mfe_valid_col in work.columns
            else None
        ),
        cfg=eval_cfg,
    )

    metrics: Dict[str, float] = dict(metrics_all)

    # Conditional evaluation for SR reversal (near SR) if required columns exist.
    # This is a "Router-like" subset evaluation and is usually more informative than global metrics.
    near_mask = compute_near_sr_mask(work, cfg=SRFuseConditionConfig())
    work["cond_near_sr"] = near_mask
    if bool(near_mask.any()):
        metrics_sr = evaluate_path_primitives(
            df=work.loc[near_mask],
            pred_cols={
                "dir": "pred_dir_prob",
                "mfe_atr": "pred_mfe_atr",
                "mae_atr": "pred_mae_atr",
                "t_to_mfe": "pred_t_to_mfe",
            },
            true_cols={
                "dir_y": dataset_cfg.dir_y_col,
                "mfe_atr": "_true_mfe_atr_tr",
                "mae_atr": "_true_mae_atr_tr",
                "t_to_mfe": "_true_t_to_mfe_tr",
            },
            mask_col=(
                dataset_cfg.mfe_valid_col
                if dataset_cfg.mfe_valid_col in work.columns
                else None
            ),
            cfg=eval_cfg,
        )
        # Prefix keys to avoid collisions
        for k, v in metrics_sr.items():
            metrics[f"near_sr__{k}"] = float(v)
        metrics["near_sr__rate"] = float(near_mask.mean())

    extra: Dict[str, Any] = {}

    # Rolling IC/ICIR monitoring (per symbol, then averaged by timestamp)
    if bool(getattr(eval_cfg, "rolling_enabled", True)):
        window = int(getattr(eval_cfg, "rolling_window", 300))
        minp = int(getattr(eval_cfg, "rolling_min_periods", 60))
        tail_n = int(getattr(eval_cfg, "rolling_tail_points", 120))

        # timestamp source for previews
        if isinstance(work.index, pd.DatetimeIndex):
            ts = work.index
        elif "timestamp" in work.columns:
            ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        elif "datetime" in work.columns:
            ts = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(
                pd.Series(np.arange(len(work))), utc=True, errors="coerce"
            )
        work["_ts_for_roll"] = ts

        def _spearman_no_scipy(a: np.ndarray, b: np.ndarray) -> float:
            if a.size < 2:
                return float("nan")
            s1 = pd.Series(a).rank(method="average")
            s2 = pd.Series(b).rank(method="average")
            c = s1.corr(s2, method="pearson")
            return float(c) if c is not None and not np.isnan(c) else float("nan")

        def _rolling_rank_ic(
            p: np.ndarray, y: np.ndarray, valid: np.ndarray
        ) -> np.ndarray:
            n = int(len(p))
            out = np.full((n,), np.nan, dtype=float)
            for i in range(n):
                lo = max(0, i - window + 1)
                sel = valid[lo : i + 1]
                if int(np.sum(sel)) < minp:
                    continue
                out[i] = _spearman_no_scipy(p[lo : i + 1][sel], y[lo : i + 1][sel])
            return out

        def _build_roll_df(
            df_slice: pd.DataFrame, *, slice_name: str
        ) -> Tuple[Dict[str, float], pd.DataFrame]:
            rows = []
            for sym, g in (
                df_slice.groupby(group_col, sort=False)
                if group_col and group_col in df_slice.columns
                else [("__single__", df_slice)]
            ):
                g = g.sort_values("_ts_for_roll").reset_index(drop=True)
                p_dir = pd.to_numeric(g["pred_dir_prob"], errors="coerce").to_numpy(
                    dtype=float
                )
                y_dir = pd.to_numeric(
                    g[dataset_cfg.dir_y_col], errors="coerce"
                ).to_numpy(dtype=float)
                valid_dir = np.isfinite(p_dir) & np.isfinite(y_dir)
                ic_dir = _rolling_rank_ic(p_dir, y_dir, valid_dir)

                # Continuous heads (use transformed labels already aligned to training space)
                out_cols = {
                    "dir": ic_dir,
                }
                for head, pcol, ycol in [
                    ("mfe_atr", "pred_mfe_atr", "_true_mfe_atr_tr"),
                    ("mae_atr", "pred_mae_atr", "_true_mae_atr_tr"),
                    ("t_to_mfe", "pred_t_to_mfe", "_true_t_to_mfe_tr"),
                ]:
                    if pcol not in g.columns or ycol not in g.columns:
                        continue
                    p = pd.to_numeric(g[pcol], errors="coerce").to_numpy(dtype=float)
                    y = pd.to_numeric(g[ycol], errors="coerce").to_numpy(dtype=float)
                    v = np.isfinite(p) & np.isfinite(y)
                    if dataset_cfg.mfe_valid_col in g.columns:
                        m = (
                            pd.to_numeric(g[dataset_cfg.mfe_valid_col], errors="coerce")
                            .fillna(0.0)
                            .to_numpy(dtype=float)
                        )
                        v = v & (m > 0.5)
                    out_cols[head] = _rolling_rank_ic(p, y, v)

                for i in range(len(g)):
                    t = (
                        g["_ts_for_roll"].iloc[i]
                        if "_ts_for_roll" in g.columns
                        else None
                    )
                    if t is None or (isinstance(t, float) and np.isnan(t)):
                        ts_str = ""
                    else:
                        try:
                            ts_str = pd.Timestamp(t).isoformat()
                        except Exception:
                            ts_str = str(t)
                    rec = {
                        "slice": str(slice_name),
                        "symbol": str(sym),
                        "timestamp": ts_str,
                    }
                    for k, arr in out_cols.items():
                        rec[f"roll_ic_{k}"] = (
                            float(arr[i]) if np.isfinite(arr[i]) else np.nan
                        )
                    rows.append(rec)
            df_roll = pd.DataFrame(rows)
            if df_roll.empty:
                return {}, df_roll
            # aggregate across symbols by timestamp
            agg = (
                df_roll.groupby(["slice", "timestamp"], sort=True)
                .mean(numeric_only=True)
                .reset_index()
            )
            # summary metrics (mean/std across time of roll_ic, then ICIR)
            summ: Dict[str, float] = {}
            for k in ["dir", "mfe_atr", "mae_atr", "t_to_mfe"]:
                col = f"roll_ic_{k}"
                if col not in agg.columns:
                    continue
                s = pd.to_numeric(agg[col], errors="coerce")
                mu = float(s.mean()) if s.notna().any() else 0.0
                sd = float(s.std(ddof=1)) if s.notna().sum() > 1 else 0.0
                summ[f"roll_ic__{k}_mean"] = mu
                summ[f"roll_ic__{k}_std"] = sd
                summ[f"roll_icir__{k}"] = float(mu / sd) if sd > 1e-12 else 0.0
                # a handy tail value
                last = s.dropna().iloc[-1] if s.notna().any() else np.nan
                if np.isfinite(last):
                    summ[f"roll_ic__{k}_last"] = float(last)
            return summ, agg

        preview_by_slice: Dict[str, List[Dict[str, Any]]] = {}

        # Global slice
        summ_g, agg_g = _build_roll_df(work, slice_name="global")
        for k, v in summ_g.items():
            metrics[k] = float(v)
        if not agg_g.empty:
            preview_by_slice["global"] = agg_g.tail(tail_n).to_dict(orient="records")

        # near_sr slice
        if "cond_near_sr" in work.columns and bool(work["cond_near_sr"].any()):
            summ_s, agg_s = _build_roll_df(
                work.loc[work["cond_near_sr"]].copy(), slice_name="near_sr"
            )
            for k, v in summ_s.items():
                metrics[f"near_sr__{k}"] = float(v)
            if not agg_s.empty:
                preview_by_slice["near_sr"] = agg_s.tail(tail_n).to_dict(
                    orient="records"
                )

        # Optional: trend_high / compression_high if columns exist
        if "trend_r2_20" in work.columns:
            mask = pd.to_numeric(work["trend_r2_20"], errors="coerce") >= 0.7
            if bool(mask.any()):
                summ_t, agg_t = _build_roll_df(
                    work.loc[mask].copy(), slice_name="trend_high"
                )
                for k, v in summ_t.items():
                    metrics[f"trend_high__{k}"] = float(v)
                if not agg_t.empty:
                    preview_by_slice["trend_high"] = agg_t.tail(tail_n).to_dict(
                        orient="records"
                    )
                metrics["trend_high__rate"] = float(mask.mean())

        if "compression_score" in work.columns:
            mask = pd.to_numeric(work["compression_score"], errors="coerce") >= 0.7
            if bool(mask.any()):
                summ_c, agg_c = _build_roll_df(
                    work.loc[mask].copy(), slice_name="compression_high"
                )
                for k, v in summ_c.items():
                    metrics[f"compression_high__{k}"] = float(v)
                if not agg_c.empty:
                    preview_by_slice["compression_high"] = agg_c.tail(tail_n).to_dict(
                        orient="records"
                    )
                metrics["compression_high__rate"] = float(mask.mean())

        extra["rolling_ic"] = {
            "cfg": {"window": window, "min_periods": minp, "tail_points": tail_n},
            "preview_by_slice": preview_by_slice,
        }

    return metrics, work, extra


def save_train_artifacts(
    *,
    out_dir: str,
    model_path: str,
    meta: Dict,
    metrics: Optional[Dict[str, float]] = None,
    df_pred_sample: Optional[pd.DataFrame] = None,
) -> str:
    """
    Save metadata/metrics/pred samples in a stable layout.
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    # meta
    (p / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    if metrics is not None:
        (p / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        # also write a short human-readable summary
        (p / "metrics_summary.md").write_text(
            _metrics_summary_md(metrics),
            encoding="utf-8",
        )

    if df_pred_sample is not None:
        df_pred_sample.to_csv(p / "pred_sample.csv", index=True)

    # html dashboard (single-file, dependency-free)
    html = render_html_dashboard(
        meta=meta, metrics=metrics, df_pred_sample=df_pred_sample
    )
    (p / "report.html").write_text(html, encoding="utf-8")

    # keep a pointer file for model artifact
    (p / "model_path.txt").write_text(str(model_path), encoding="utf-8")
    return str(p)
