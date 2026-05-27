"""Snotio (mean R-multiple) KPI and entry-filter plateau detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def compute_snotio(r_returns: pd.Series) -> float:
    """snotio = mean(R-multiples) per trade."""
    valid = pd.to_numeric(r_returns, errors="coerce").dropna()
    if valid.empty:
        return 0.0
    return float(valid.mean())


def width_to_confidence(width: float) -> str:
    """Map plateau width to deploy confidence tier."""
    if width >= 0.3:
        return "HIGH"
    if width >= 0.15:
        return "MEDIUM"
    return "LOW"


def find_snotio_plateau(
    results: List[Dict[str, Any]],
    *,
    window: int = 5,
    operator: str = ">=",
    snotio_cv_max: float = 0.3,
    trades_cv_max: float = 0.4,
) -> Dict[str, Any]:
    """Find stable snotio plateau from threshold scan results (entry filter KPI)."""
    valid = [r for r in results if not r.get("too_few")]
    if len(valid) < window:
        return {
            "is_plateau": False,
            "reason": f"有效点不足 ({len(valid)} < {window})",
        }

    best_plateau: Optional[Dict[str, Any]] = None
    for i in range(len(valid) - window + 1):
        w = valid[i : i + window]
        snotios = [r["snotio"] for r in w]
        trades_list = [r["trades"] for r in w]
        mean_sn = float(np.mean(snotios))
        std_sn = float(np.std(snotios))
        cv_snotio = std_sn / mean_sn if mean_sn > 1e-8 else 999.0
        mean_tr = float(np.mean(trades_list))
        std_tr = float(np.std(trades_list))
        cv_trades = std_tr / mean_tr if mean_tr > 1e-8 else 999.0

        if cv_snotio < snotio_cv_max and cv_trades < trades_cv_max and mean_sn > 0:
            start_t = w[0]["threshold"]
            end_t = w[-1]["threshold"]
            plateau_width = abs(end_t - start_t)
            robustness = mean_sn * plateau_width
            if best_plateau is None or robustness > best_plateau.get("_robustness", -1.0):
                bias = 0.2 if plateau_width < 0.25 else 0.1
                if operator in (">=", ">"):
                    rec_val = start_t + bias * plateau_width
                else:
                    rec_val = end_t - bias * plateau_width
                rec_idx = int(np.argmin([abs(r["threshold"] - rec_val) for r in w]))
                best_plateau = {
                    "is_plateau": True,
                    "_robustness": float(robustness),
                    "start_threshold": start_t,
                    "end_threshold": end_t,
                    "plateau_width": float(plateau_width),
                    "confidence": width_to_confidence(plateau_width),
                    "mean_snotio": mean_sn,
                    "cv_snotio": float(cv_snotio),
                    "cv_trades": float(cv_trades),
                    "mean_trades": float(mean_tr),
                    "recommended": float(w[rec_idx]["threshold"]),
                }

    if best_plateau is None:
        for i in range(len(valid) - window + 1):
            w = valid[i : i + window]
            snotios = [r["snotio"] for r in w]
            mean_sn = float(np.mean(snotios))
            cv_snotio = float(np.std(snotios) / mean_sn) if mean_sn > 1e-8 else 999.0
            if cv_snotio < snotio_cv_max and mean_sn > 0:
                start_t = w[0]["threshold"]
                end_t = w[-1]["threshold"]
                pw = abs(end_t - start_t)
                robustness = mean_sn * pw
                if best_plateau is None or robustness > best_plateau.get(
                    "_robustness", -1.0
                ):
                    trades_list = [r["trades"] for r in w]
                    cv_trades = (
                        float(np.std(trades_list) / np.mean(trades_list))
                        if np.mean(trades_list) > 0
                        else 999.0
                    )
                    bias = 0.2 if pw < 0.25 else 0.1
                    if operator in (">=", ">"):
                        rec_val = start_t + bias * pw
                    else:
                        rec_val = end_t - bias * pw
                    rec_idx = int(np.argmin([abs(r["threshold"] - rec_val) for r in w]))
                    best_plateau = {
                        "is_plateau": True,
                        "_robustness": float(robustness),
                        "start_threshold": start_t,
                        "end_threshold": end_t,
                        "plateau_width": float(pw),
                        "confidence": width_to_confidence(pw),
                        "mean_snotio": mean_sn,
                        "cv_snotio": cv_snotio,
                        "cv_trades": cv_trades,
                        "cv_trades_warning": True,
                        "mean_trades": float(np.mean(trades_list)),
                        "recommended": float(w[rec_idx]["threshold"]),
                    }

    if best_plateau is None:
        snotios = [r["snotio"] for r in valid]
        best_idx = int(np.argmax(snotios))
        return {
            "is_plateau": False,
            "reason": "无 CV<0.3 的稳定窗口",
            "best_single": {
                "threshold": valid[best_idx]["threshold"],
                "snotio": valid[best_idx]["snotio"],
                "trades": valid[best_idx]["trades"],
            },
        }

    best_plateau.pop("_robustness", None)
    return best_plateau
