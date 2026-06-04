"""
Entry Filter 公共模块 — backtest 和 live 共用。

职责：
  - 加载 entry_filters.yaml 配置
  - 计算衍生 entry filter 特征（批量 DataFrame / 单 bar dict）
  - 构建条件 mask / 检查单 bar 条件
  - 并联组合：默认 OR；可选 ``combination_mode: and``（见 ``check_entry_filters_or_single``）

数据流：
  backtest: DataFrame 批量 → compute_derived_entry_features() → apply_entry_filters_or()
  live:     Dict 单 bar  → compute_derived_entry_features_single() → check_entry_filters_or_single()
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.config.strategy_layout import resolve_strategy_package_under_root

# ================================================================
# Operator mapping (条件运算符 → 函数)
# ================================================================

_OP_MAP: Dict[str, Callable] = {
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
}

# Scalar version for single-bar checks
_OP_MAP_SCALAR: Dict[str, Callable[[float, float], bool]] = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


# ================================================================
# Config loading
# ================================================================


def load_entry_filters_config(
    strategy: str,
    strategies_root: str = "config/strategies",
    *,
    research: bool = False,
    live_layout: bool = False,
) -> Dict[str, Any]:
    """加载 entry_filters.yaml 配置。

    Args:
        research: True → 读根目录研究文件 (含全部候选 + disabled);
                  False → 读 archetypes/ 生产文件 (默认, backtest/live 用).
        live_layout: ``True`` 不向 ``bad-candidates/`` 回退（与实盘磁盘布局一致）。
    """
    pkg = resolve_strategy_package_under_root(
        Path(strategies_root),
        strategy,
        allow_bad_candidates=not live_layout,
    )
    if research:
        # 研究文件: config/strategies/{strategy}/entry_filters.yaml
        path = pkg / "entry_filters.yaml"
        if not path.exists():
            # fallback to archetypes
            path = pkg / "archetypes" / "entry_filters.yaml"
    else:
        path = pkg / "archetypes" / "entry_filters.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_available_filters(
    entry_cfg: Dict[str, Any],
) -> List[str]:
    """返回 entry_filters.yaml 中所有 enabled=true 的 filter id 列表"""
    ids = ["none"]
    for f in entry_cfg.get("filters") or []:
        if f.get("enabled", True):
            ids.append(f["id"])
    return ids


# ================================================================
# Derived entry features — batch (DataFrame)
# ================================================================


def compute_derived_entry_features(df: pd.DataFrame) -> None:
    """计算衍生 entry filter 特征（从已有列派生的正交维度）。

    三个正交维度：
    1. ef_vol_regime_shift:   波动率 regime 切换（bb_width 5-bar 变化率）
       负值 = 波动率正在下降 → squeeze 中 → 爆发前兆
    2. ef_liquidity_silence:  流动性沉默分位（成交量历史百分位）
       低值 = 市场极度安静 → 参与者在等待
    3. ef_consolidation_bars: 回踩盘整持续 bar 数（连续 was_in_pullback==1 的 bar 数）
       高值 = 盘整越久，蓄势越充分

    NOTE: 这些特征直接添加为 df 的新列（in-place）。
    """
    # --- 1. vol_regime_shift: bb_width 5-bar 变化 ---
    if "bb_width_normalized_pct" in df.columns:
        if "symbol" in df.columns:
            df["ef_vol_regime_shift"] = df.groupby("symbol")[
                "bb_width_normalized_pct"
            ].diff(5)
        else:
            df["ef_vol_regime_shift"] = df["bb_width_normalized_pct"].diff(5)
        df["ef_vol_regime_shift"] = df["ef_vol_regime_shift"].fillna(0.0)

    # --- 2. liquidity_silence: 直接用 vol_percentile_approx ---
    # vol_percentile_approx [0,1], 低 = 成交量极低 = 流动性沉默
    if "vol_percentile_approx" in df.columns:
        df["ef_liquidity_silence"] = df[
            "vol_percentile_approx"
        ]  # NaN = warmup 不足，禁止静默降级为 0.5
    elif "bpc_vol_ratio" in df.columns:
        # fallback: vol_ratio < 0.6 → 低于均量 60% → 沉默
        df["ef_liquidity_silence"] = df["bpc_vol_ratio"].fillna(1.0)

    # --- 3. consolidation_bars: 连续 was_in_pullback==1 的 bar 数 ---
    if "bpc_was_in_pullback" in df.columns:
        wip = df["bpc_was_in_pullback"].astype(float)
        if "symbol" in df.columns:
            cons = []
            for sym, grp in df.groupby("symbol"):
                vals = grp["bpc_was_in_pullback"].values
                cnt = np.zeros(len(vals), dtype=float)
                c = 0
                for i, v in enumerate(vals):
                    if v == 1:
                        c += 1
                    else:
                        c = 0
                    cnt[i] = c
                cons.append(pd.Series(cnt, index=grp.index))
            df["ef_consolidation_bars"] = pd.concat(cons).reindex(df.index)
        else:
            vals = wip.values
            cnt = np.zeros(len(vals), dtype=float)
            c = 0
            for i, v in enumerate(vals):
                if v == 1:
                    c += 1
                else:
                    c = 0
                cnt[i] = c
            df["ef_consolidation_bars"] = cnt
        # 归一化到 [0, 1]（上限 40 bar ≈ 近 7 天 4H bar）
        df["ef_consolidation_bars"] = (df["ef_consolidation_bars"] / 40.0).clip(0, 1)


# ================================================================
# Derived entry features — live (single bar, stateful)
# ================================================================


class DerivedEntryFeatureState:
    """Live 用有状态的衍生特征计算器。

    每收到一根新 bar 的特征 dict，更新内部状态并返回 ef_* 衍生值。

    用法::

        state = DerivedEntryFeatureState()
        # 每根新 bar:
        ef_features = state.update(features_dict)
        # ef_features = {"ef_vol_regime_shift": ..., "ef_liquidity_silence": ..., "ef_consolidation_bars": ...}
    """

    def __init__(self, diff_window: int = 5, consolidation_cap: int = 40):
        self._diff_window = diff_window
        self._consolidation_cap = consolidation_cap
        # bb_width 历史 buffer（最近 diff_window+1 值）
        self._bb_width_history: deque = deque(maxlen=diff_window + 1)
        # 连续 was_in_pullback==1 的计数
        self._consolidation_count: int = 0

    def update(self, features: Dict[str, float]) -> Dict[str, float]:
        """接收当前 bar 特征 dict，返回 ef_* 衍生特征 dict。"""
        result: Dict[str, float] = {}

        # --- 1. ef_vol_regime_shift ---
        bb_val = features.get("bb_width_normalized_pct")
        if bb_val is not None:
            self._bb_width_history.append(float(bb_val))
            if len(self._bb_width_history) > self._diff_window:
                result["ef_vol_regime_shift"] = (
                    self._bb_width_history[-1]
                    - self._bb_width_history[-1 - self._diff_window]
                )
            else:
                result["ef_vol_regime_shift"] = 0.0
        else:
            result["ef_vol_regime_shift"] = 0.0

        # --- 2. ef_liquidity_silence ---
        vol_pct = features.get("vol_percentile_approx")
        if vol_pct is not None:
            result["ef_liquidity_silence"] = float(vol_pct)
        else:
            vol_ratio = features.get("bpc_vol_ratio")
            if vol_ratio is not None:
                result["ef_liquidity_silence"] = float(vol_ratio)
            else:
                result["ef_liquidity_silence"] = 0.5

        # --- 3. ef_consolidation_bars ---
        wip = features.get("bpc_was_in_pullback")
        if wip is not None and float(wip) == 1:
            self._consolidation_count += 1
        else:
            self._consolidation_count = 0
        result["ef_consolidation_bars"] = min(
            self._consolidation_count / self._consolidation_cap, 1.0
        )

        return result

    def reset(self) -> None:
        """重置状态（例如换 symbol 时）。"""
        self._bb_width_history.clear()
        self._consolidation_count = 0


# ================================================================
# Per-filter direction scope (long / short / both)
# ================================================================


def _normalize_filter_direction(filt: Dict[str, Any]) -> Optional[str]:
    """Return 'long', 'short', or None if filter applies to both sides."""
    raw = filt.get("direction")
    if raw is None:
        return None
    side = str(raw).strip().lower()
    if side in ("long", "buy", "1", "+1"):
        return "long"
    if side in ("short", "sell", "-1"):
        return "short"
    return None


def entry_filter_applies_to_direction(
    filt: Dict[str, Any],
    direction: Optional[int],
) -> bool:
    """True when filter should be evaluated for this direction (+1 long / -1 short)."""
    side = _normalize_filter_direction(filt)
    if side is None or direction is None:
        return True
    if direction > 0:
        return side == "long"
    if direction < 0:
        return side == "short"
    return True


def _direction_vacuous_pass_mask(
    df: pd.DataFrame,
    filt: Dict[str, Any],
) -> pd.Series:
    """Rows where a direction-scoped filter does not apply → vacuous pass."""
    side = _normalize_filter_direction(filt)
    if side is None or "entry_direction" not in df.columns:
        return pd.Series(False, index=df.index)
    ed = pd.to_numeric(df["entry_direction"], errors="coerce").fillna(0.0)
    if side == "long":
        return ed != 1.0
    return ed != -1.0


# ================================================================
# Condition mask — batch (DataFrame)
# ================================================================


def _build_mask_from_conditions(
    df: pd.DataFrame,
    conditions: List[Dict[str, Any]],
    silent: bool = False,
) -> pd.Series:
    """从 conditions 列表构建 AND 组合的 boolean mask"""
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        feat = cond["feature"]
        op_str = cond["operator"]
        val = cond["value"]
        if feat not in df.columns:
            if not silent:
                print(f"   ⚠️  Missing feature '{feat}', condition skipped")
            continue
        op_fn = _OP_MAP.get(op_str)
        if op_fn is None:
            if not silent:
                print(f"   ⚠️  Unknown operator '{op_str}', condition skipped")
            continue
        mask = mask & op_fn(df[feat].astype(float), float(val))
    return mask


# ================================================================
# Condition check — live (single bar dict)
# ================================================================


def check_conditions_single(
    features: Dict[str, float],
    conditions: List[Dict[str, Any]],
) -> bool:
    """检查单个 bar 的特征 dict 是否满足所有 conditions (AND)。

    Returns:
        True if ALL conditions are met, False otherwise.
        Missing features → condition treated as not met.
    """
    for cond in conditions:
        feat = cond["feature"]
        op_str = cond["operator"]
        threshold = float(cond["value"])

        val = features.get(feat)
        if val is None:
            return False  # missing feature → fail

        op_fn = _OP_MAP_SCALAR.get(op_str)
        if op_fn is None:
            return False

        if not op_fn(float(val), threshold):
            return False

    return True


def check_entry_filters_or_single(
    features: Dict[str, float],
    entry_cfg: Dict[str, Any],
    *,
    direction: Optional[int] = None,
) -> bool:
    """Live 单 bar entry_filters.yaml 顶层组合。

    每个 filter：其 ``conditions`` 始终 **内部 AND**（见 ``check_conditions_single``）。
    **多个 filter**：由 ``combination_mode`` 控制并联语义（缺省 ``or``）：

    - ``or`` — 任一 enabled filter（且含非空 conditions）通过 ⇒ 允许入场
    - ``and`` — 所有这类 filter 均需通过 ⇒ 允许入场

    若无 enabled filter、或均无有效 conditions ⇒ 放行（等价无门）。

    Args:
        features: 当前 bar 的特征字典（含 ef_* 衍生特征）
        entry_cfg: load_entry_filters_config() 返回的配置

    Returns:
        True = 允许入场, False = 等待
    """
    filters_list = entry_cfg.get("filters") or []
    enabled = [f for f in filters_list if f.get("enabled", False)]

    if not enabled:
        return True  # no filters → all pass

    mode = str(entry_cfg.get("combination_mode") or "or").strip().lower()
    if mode not in {"or", "and"}:
        mode = "or"

    per_filter_ok: List[bool] = []
    for filt in enabled:
        conditions = filt.get("conditions", [])
        if not conditions:
            continue
        if not entry_filter_applies_to_direction(filt, direction):
            per_filter_ok.append(True)
            continue
        per_filter_ok.append(check_conditions_single(features, conditions))

    if not per_filter_ok:
        return True  # enabled filters exist but none had conditions → pass

    if mode == "and":
        return all(per_filter_ok)
    return any(per_filter_ok)


# ================================================================
# Apply filters — batch (DataFrame, in-place mutation)
# ================================================================


def apply_entry_filter(
    df: pd.DataFrame,
    filter_name: str,
    entry_cfg: Optional[Dict[str, Any]] = None,
    silent: bool = False,
) -> int:
    """
    Config-driven 入场时机过滤器（单个 filter）。

    从 entry_filters.yaml 读取 filter 定义（conditions），
    将不满足条件的 bar 的 entry_direction 置 0。

    Returns:
        过滤后剩余的入场信号数
    """
    if filter_name == "none":
        return int((df["entry_direction"] != 0).sum())

    n_before = int((df["entry_direction"] != 0).sum())

    # 查找 filter 定义
    filter_def = None
    if entry_cfg:
        for f in entry_cfg.get("filters") or []:
            if f.get("id") == filter_name:
                filter_def = f
                break

    if filter_def is None:
        if not silent:
            print(f"   ⚠️  Filter '{filter_name}' not found in entry_filters.yaml")
        return n_before

    conditions = filter_def.get("conditions", [])
    if not conditions:
        if not silent:
            print(f"   ⚠️  Filter '{filter_name}' has no conditions")
        return n_before

    # 构建 mask
    mask = _build_mask_from_conditions(df, conditions, silent=silent)

    # 应用: 不满足 mask 的行 entry_direction → 0
    df.loc[~mask, "entry_direction"] = 0.0
    n_after = int((df["entry_direction"] != 0).sum())

    if not silent:
        pct = n_after / n_before * 100 if n_before > 0 else 0
        desc = filter_def.get("description", "")
        print(
            f"   🔍 Entry filter '{filter_name}': {n_before} → {n_after} entries ({pct:.1f}% pass)"
        )
        print(f"      {desc}")

    return n_after


def apply_entry_filters_or(
    df: pd.DataFrame,
    entry_cfg: Dict[str, Any],
    silent: bool = False,
) -> int:
    """
    对所有 enabled=true 的 entry filter 并联应用顶层 ``combination_mode``。

    - 每个 filter 内部的 conditions：**AND**
    - 多个 filter：**OR（默认）** 或 **AND**（``combination_mode: and``）

    Returns:
        过滤后剩余的入场信号数
    """
    filters_list = entry_cfg.get("filters") or []
    enabled = [f for f in filters_list if f.get("enabled", False)]

    if not enabled:
        if not silent:
            print("   ℹ️  No enabled entry filters, all entries pass")
        return int((df["entry_direction"] != 0).sum())

    mode = str(entry_cfg.get("combination_mode") or "or").strip().lower()
    if mode not in {"or", "and"}:
        mode = "or"

    n_before = int((df["entry_direction"] != 0).sum())

    filter_masks: List[pd.Series] = []
    filter_stats = []
    for filt in enabled:
        conditions = filt.get("conditions", [])
        if not conditions:
            continue
        filt_mask = _build_mask_from_conditions(df, conditions, silent=True)
        vacuous = _direction_vacuous_pass_mask(df, filt)
        filt_mask = filt_mask | vacuous
        filt_pass = int((filt_mask & (df["entry_direction"] != 0)).sum())
        filter_stats.append((filt["id"], filt_pass))
        filter_masks.append(filt_mask)

    if not filter_masks:
        return n_before

    if mode == "and":
        combo_mask = filter_masks[0]
        for m in filter_masks[1:]:
            combo_mask = combo_mask & m
    else:
        combo_mask = pd.Series(False, index=df.index)
        for m in filter_masks:
            combo_mask = combo_mask | m

    # 应用: 不满足并联 mask 的 bar → entry_direction = 0
    df.loc[~combo_mask, "entry_direction"] = 0.0
    n_after = int((df["entry_direction"] != 0).sum())

    if not silent:
        pct = n_after / n_before * 100 if n_before > 0 else 0
        top = "AND" if mode == "and" else "OR"
        print(
            f"   🔍 Entry Filter ({top}, {len(filter_masks)} filters): "
            f"{n_before} → {n_after} entries ({pct:.1f}% pass)"
        )
        for fid, fpass in filter_stats:
            print(f"      ✅ {fid}: {fpass} entries")

    return n_after
