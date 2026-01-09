from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalConfig:
    """
    Evaluation config for path primitives.
    Keep it dependency-light (no scipy required).
    """

    # For binary direction
    dir_threshold: float = 0.5  # for prob

    # Small epsilon for numerical stability
    eps: float = 1e-12

    # Rolling Rank-IC/ICIR monitoring (time-series rolling, per-symbol then aggregated)
    rolling_enabled: bool = True
    rolling_window: int = 300
    rolling_min_periods: int = 60
    rolling_tail_points: int = 120


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman correlation using pandas rank (no scipy dependency).
    Returns 0.0 when not computable.
    """
    if a.size == 0:
        return 0.0
    s1 = pd.Series(a).rank(method="average")
    s2 = pd.Series(b).rank(method="average")
    corr = s1.corr(s2, method="pearson")
    return float(0.0 if corr is None or np.isnan(corr) else corr)


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Simple AUC without sklearn:
    Uses rank-based U-statistic. Returns 0.5 when undefined.
    """
    y_true = y_true.astype(int)
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return 0.5
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_pos_ranks = float(ranks[pos].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Average precision (area under precision-recall curve) without sklearn.
    Returns positive-class prevalence when undefined.
    """
    y_true = y_true.astype(int)
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return 0.0
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0
    if n_pos == int(y_true.size):
        return 1.0

    order = np.argsort(-y_score)  # descending
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    # AP = mean(precision at each positive hit)
    ap = float(np.sum(precision[y_sorted == 1]) / float(n_pos))
    return float(max(0.0, min(1.0, ap)))


def binary_threshold_metrics(
    *,
    y_true_cont: np.ndarray,
    y_score_cont: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Evaluate how well y_score ranks samples that exceed a threshold on y_true.
    Returns AUC and AP (average precision).
    """
    y = y_true_cont
    s = y_score_cont
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if y.size == 0:
        return {"auc": 0.5, "ap": 0.0, "pos_rate": 0.0}
    y_bin = (y > float(threshold)).astype(int)
    pos_rate = float(np.mean(y_bin))
    return {
        "auc": _binary_auc(y_bin, s),
        "ap": _average_precision(y_bin, s),
        "pos_rate": pos_rate,
    }


def evaluate_path_primitives(
    *,
    df: pd.DataFrame,
    pred_cols: Dict[str, str],
    true_cols: Dict[str, str],
    mask_col: Optional[str] = None,
    cfg: EvalConfig = EvalConfig(),
) -> Dict[str, float]:
    """
    Compute basic head metrics on a dataframe that contains:
    - prediction columns (pred_cols)
    - ground-truth label columns (true_cols)
    - optional mask column (e.g., mfe_valid)
    """
    out: Dict[str, float] = {}

    def _get(col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    # Direction: supports either prob (0..1) or score (-1..1).
    y_dir = _get(true_cols["dir_y"])
    p_dir = _get(pred_cols["dir"])

    # Convert 0/1 labels possibly stored as float
    y_dir_bin = (y_dir > 0.5).astype(int)

    # If p_dir looks like signed score, convert to prob
    p_dir_prob = p_dir
    if np.nanmin(p_dir) < 0.0 and np.nanmax(p_dir) <= 1.0:
        p_dir_prob = (p_dir + 1.0) / 2.0
    p_dir_prob = np.clip(p_dir_prob, 0.0, 1.0)

    y_hat = (p_dir_prob >= cfg.dir_threshold).astype(int)
    acc = float(np.mean(y_hat == y_dir_bin)) if y_hat.size else 0.0
    auc = _binary_auc(y_dir_bin, p_dir_prob)

    out["dir_acc"] = acc
    out["dir_auc"] = auc

    # Continuous heads: spearman (optionally under mask)
    if mask_col is not None and mask_col in df.columns:
        m = (
            pd.to_numeric(df[mask_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        mask = m > 0.5
    else:
        mask = np.ones(len(df), dtype=bool)

    for head in ("mfe_atr", "mae_atr", "t_to_mfe"):
        if head not in pred_cols or head not in true_cols:
            continue
        y = _get(true_cols[head])
        p = _get(pred_cols[head])
        sel = mask & np.isfinite(y) & np.isfinite(p)
        out[f"{head}_spearman"] = _spearman(p[sel], y[sel]) if sel.any() else 0.0
        # error magnitude (MAE) for sanity
        out[f"{head}_mae"] = (
            float(np.mean(np.abs(p[sel] - y[sel]))) if sel.any() else 0.0
        )

    # Mask hit-rate for monitoring
    if mask_col is not None and mask_col in df.columns:
        out["mask_rate"] = float(np.mean(mask.astype(float)))

    return out
