"""Plateau stability contract — 防止阈值在 calibration 周期之间漂移。

跨调参周期使用：
    每条 locked 阈值（regime / prefilter / gate / entry）在 YAML 里保留
    ``last_calibration.plateau`` 区间。下一轮调参出新 plateau 时调用
    ``decide_plateau_update`` 决定是 ADOPT / KEEP / ALERT。

合约（默认 keep_if_overlaps）：
    - 新 plateau 与旧 plateau **完全无交集** → ALERT；阈值保留旧值（人工复核）
    - 新 plateau 至少与旧 plateau 部分重叠 → ADOPT；写入新 plateau_mid
    - 没有旧 plateau（首轮）→ ADOPT；记录新 plateau 作为基准

仅做纯函数运算 + 小工具，调用方负责 IO。便于在
``optimize_gate_unified.py`` / ``optimize_entry_filter_plateau.py`` /
``locked_prefilter_parquet_tune.py`` / ``regime_threshold_calibrate.py`` 中
统一接入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = [
    "PlateauRange",
    "ranges_overlap",
    "decide_plateau_update",
    "plateau_range_from_dict",
    "plateau_range_to_dict",
]


@dataclass(frozen=True)
class PlateauRange:
    """单个 plateau 区间 (闭区间 [start, end])，mid 通常是 (start+end)/2。"""

    start: float
    end: float
    mid: float

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"PlateauRange.start({self.start}) > end({self.end})")
        if not (self.start <= self.mid <= self.end):
            raise ValueError(
                f"PlateauRange.mid({self.mid}) outside [{self.start}, {self.end}]"
            )

    @property
    def width(self) -> float:
        return self.end - self.start


def ranges_overlap(a: PlateauRange, b: PlateauRange, *, tol: float = 0.0) -> bool:
    """两段区间是否相交（允许 ``tol`` 数值容差）。"""
    return not (a.end < b.start - tol or b.end < a.start - tol)


def plateau_range_from_dict(d: Optional[Dict[str, Any]]) -> Optional[PlateauRange]:
    """从 YAML dict 还原 PlateauRange（缺字段返回 None）。"""
    if not isinstance(d, dict):
        return None
    start = d.get("start")
    end = d.get("end")
    mid = d.get("mid")
    if start is None or end is None:
        return None
    if mid is None:
        mid = (float(start) + float(end)) / 2.0
    try:
        return PlateauRange(start=float(start), end=float(end), mid=float(mid))
    except (TypeError, ValueError):
        return None


def plateau_range_to_dict(r: PlateauRange) -> Dict[str, float]:
    return {"start": float(r.start), "end": float(r.end), "mid": float(r.mid)}


def decide_plateau_update(
    *,
    old: Optional[PlateauRange],
    new: PlateauRange,
    current_value: float,
    policy: str = "keep_if_no_overlap",
    overlap_tol: float = 0.0,
) -> Dict[str, Any]:
    """决定本轮 plateau calibration 的处置。

    Returns dict 字段:
        action       : "ADOPT" | "KEEP" | "ALERT"
        chosen_value : float (写回 YAML 的阈值)
        reason       : str (人类可读)
        old_range    : dict | None
        new_range    : dict
        overlap      : bool
    """
    policy_norm = (policy or "keep_if_no_overlap").lower().strip()
    new_dict = plateau_range_to_dict(new)

    if old is None:
        return {
            "action": "ADOPT",
            "chosen_value": float(new.mid),
            "reason": "no_prior_plateau (first calibration)",
            "old_range": None,
            "new_range": new_dict,
            "overlap": True,
        }

    old_dict = plateau_range_to_dict(old)
    overlap = ranges_overlap(old, new, tol=overlap_tol)

    if overlap:
        return {
            "action": "ADOPT",
            "chosen_value": float(new.mid),
            "reason": "new_plateau_overlaps_old (stability contract satisfied)",
            "old_range": old_dict,
            "new_range": new_dict,
            "overlap": True,
        }

    # 无交集 → 漂移
    if policy_norm == "keep_if_no_overlap":
        return {
            "action": "ALERT",
            "chosen_value": float(current_value),
            "reason": (
                f"plateau_drift_detected: new=[{new.start:.6g},{new.end:.6g}] "
                f"disjoint from old=[{old.start:.6g},{old.end:.6g}]; "
                "keep current value, escalate to human review."
            ),
            "old_range": old_dict,
            "new_range": new_dict,
            "overlap": False,
        }
    if policy_norm == "adopt_anyway":
        return {
            "action": "ADOPT",
            "chosen_value": float(new.mid),
            "reason": (
                f"plateau_drift_adopt_anyway: new=[{new.start:.6g},{new.end:.6g}] "
                f"disjoint from old=[{old.start:.6g},{old.end:.6g}]; "
                "policy=adopt_anyway."
            ),
            "old_range": old_dict,
            "new_range": new_dict,
            "overlap": False,
        }
    raise ValueError(f"unknown policy: {policy!r}")
