#!/usr/bin/env python3
"""
统一 Meta-Algorithm 核心模块 — Prefilter 与 Gate 共用

共享流程:
  1. Train/Holdout 时间分割 (支持 research=3段 / deploy=2段)
  2. LightGBM 训练 (regression 或 classification)
  3. SHAP∩Gain 特征发现
  4. 百分位阈值扫描 (单阈值 gt/lt + range)
  5. Plateau 稳定性验证
  6. Holdout KPI 验证 + Fold Robustness
  7. 相关性剪枝
  8. 规则输出 (YAML)

参数区别:
  | 参数               | Prefilter                        | Gate                                        |
  | 训练数据           | 全量 features_labeled.parquet    | prefilter 过滤后的 features_labeled.parquet |
  | 特征文件           | features_prefilter.yaml          | features.yaml (or _shap)                    |
  | 训练标签           | forward_rr (regression)          | success_no_rr_extreme (classification)      |
  | scoring_method     | bad_rate_lift / positive_rr / ks | 同上（可不同值）                            |
  | 输出               | archetypes/prefilter.yaml        | archetypes/gate.yaml                        |
"""
from __future__ import annotations

import operator as op_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ====================================================================
# Configuration
# ====================================================================


@dataclass
class MetaAlgorithmConfig:
    """统一 Meta-Algorithm 配置."""

    # ── 层级 ──
    layer: str = "prefilter"  # "prefilter" | "gate"

    # ── 数据分割 ──
    mode: str = "research"  # "research" (3-way) | "deploy" (2-way)
    holdout_ratio: float = 0.30  # research: test + oos; deploy: oos only
    test_ratio: float = 0.10  # research mode: test 占全量比例
    oos_ratio: float = 0.10  # research mode: oos 占全量比例

    # ── LightGBM ──
    objective: str = "regression"  # "regression" | "binary"
    metric: str = "mse"  # "mse" | "binary_logloss"
    num_leaves: int = 31
    learning_rate: float = 0.05
    min_child_samples: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    n_estimators: int = 200
    seed: int = 42

    # ── SHAP∩Gain ──
    shap_top_n: int = 8
    compute_interactions: bool = True

    # ── 阈值扫描 ──
    quantiles: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95]
    )
    deny_rate_min: float = 0.03
    deny_rate_max: float = 0.40
    # Range 规则（同 feature 上 lo <= x <= hi 的 deny）最小宽度，单位 = σ（feature std）。
    # 目的：防止「针尖 range」过拟合 —— holdout KS 看似高，但语义完全无法解释。
    # 2026-04-22 BPC 坑示例：bpc_impulse_return_atr ∈ [-0.877, -0.669]（宽 ≈0.2σ）
    #                         bpc_dir_flip_count    ∈ [0.35, 0.65]     （宽 ≈0.6σ）
    # 建议值：1.0（保底 1σ 宽；若需研究期探索，可降到 0.5）
    min_range_width_sigma: float = 1.0

    # ── KPI 门禁 ──
    scoring_method: str = "ks"  # "ks" | "bad_rate_lift" | "positive_rr" | "combined"
    min_ks_statistic: float = 0.05
    max_ks_pvalue: float = 0.01
    min_lift: float = 1.05
    min_bad_rate_lift: float = 1.05
    min_effect: float = 0.02
    min_robustness: float = 0.3
    min_positive_lift: float = 1.20
    positive_threshold: float = 0.8

    # ── Plateau ──
    min_plateau_width: float = 0.05
    max_lift_std_ratio: float = 0.3

    # ── 规则输出 ──
    max_rules: int = 4
    correlation_threshold: float = 0.80
    n_folds: int = 5

    # ── 策略 ──
    strategy: str = ""


# ====================================================================
# 0. Semantic Polarity Registry (per-strategy, no shared fallback)
# ====================================================================
#
# 每个策略在自己目录下声明单调特征的方向语义:
#   config/strategies/{strategy}/semantic_polarity.yaml
#
# 不共享: 同名特征在不同策略语义可能不同 (e.g. dollar_volume_over_mcap
# 在 ME 语义里是 higher_is_better, 在低波动轮动策略里可能是 lower_is_better),
# 所以每个策略需要在自己的文件里显式声明。
#
# 三档:
#   higher_is_better → skip direction="gt" (deny high) + skip range
#   lower_is_better  → skip direction="lt" (deny low)  + skip range
#   unknown / 未列入 → 两侧方向 + range 都扫 (保留原自由度)

_POLARITY_CACHE: Dict[str, Dict[str, str]] = {}  # key: strategy name
_POLARITY_VALID = {"higher_is_better", "lower_is_better", "unknown"}


def _find_config_root() -> Optional[Path]:
    """Return the repo's config/ directory, or None if not found."""
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent / "config", Path.cwd() / "config"):
        if candidate.is_dir():
            return candidate
    return None


def _read_polarity_file(path: Path) -> Dict[str, str]:
    """Parse a single semantic_polarity.yaml file. Returns empty dict on failure."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        raw = data.get("polarity", {}) or {}
        out: Dict[str, str] = {}
        for k, v in raw.items():
            if not isinstance(v, str):
                continue
            v_norm = v.strip().lower()
            if v_norm in _POLARITY_VALID:
                out[str(k)] = v_norm
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  {path.name} 加载失败, 忽略: {exc}")
        return {}


def _load_semantic_polarity(strategy: Optional[str] = None) -> Dict[str, str]:
    """Load per-strategy polarity map (cached).

    Args:
        strategy: strategy name (e.g. "me", "bpc", "tpc"). None/empty → no polarity map.

    Returns:
        Dict[feature_name, polarity]. Empty if strategy not given or file missing.
    """
    key = (strategy or "").strip().lower()
    if key in _POLARITY_CACHE:
        return _POLARITY_CACHE[key]

    mapping: Dict[str, str] = {}
    config_root = _find_config_root()
    if config_root is not None and key:
        mapping = _read_polarity_file(
            config_root / "strategies" / key / "semantic_polarity.yaml"
        )

    _POLARITY_CACHE[key] = mapping
    return mapping


def get_feature_polarity(feature: str, strategy: Optional[str] = None) -> str:
    """Return polarity for a feature name under the given strategy.

    No strategy (or strategy has no polarity file) → always 'unknown' (free sweep).
    """
    return _load_semantic_polarity(strategy).get(feature, "unknown")


# ====================================================================
# 1. Time Split
# ====================================================================


def find_time_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the time column in a DataFrame."""
    for col in ["timestamp", "datetime", "date", "time", "ts"]:
        if col in df.columns:
            return col
    # Check index
    if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64):
        return "__index__"
    return None


def time_split(
    df: pd.DataFrame,
    mode: str = "research",
    test_ratio: float = 0.10,
    oos_ratio: float = 0.10,
    holdout_ratio: float = 0.30,
) -> Dict[str, pd.DataFrame]:
    """时间有序分割数据.

    research mode: Train / Test / OOS (3 段)
    deploy mode:   Train+Test / OOS (2 段)

    Returns:
        {"train": df_train, "test": df_test, "oos": df_oos}
        deploy mode: test == train (合并)
    """
    time_col = find_time_column(df)
    if time_col is not None:
        if time_col == "__index__":
            times = df.index
        else:
            times = pd.to_datetime(df[time_col], errors="coerce")
        sort_order = times.values.argsort()
        df_sorted = df.iloc[sort_order].reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    n = len(df_sorted)

    if mode == "deploy":
        n_oos = int(n * oos_ratio)
        n_train = n - n_oos
        return {
            "train": df_sorted.iloc[:n_train].copy(),
            "test": df_sorted.iloc[:n_train].copy(),  # same as train
            "oos": df_sorted.iloc[n_train:].copy(),
        }
    else:  # research
        n_oos = int(n * oos_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_test - n_oos
        return {
            "train": df_sorted.iloc[:n_train].copy(),
            "test": df_sorted.iloc[n_train : n_train + n_test].copy(),
            "oos": df_sorted.iloc[n_train + n_test :].copy(),
        }


# ====================================================================
# 2. LightGBM Training
# ====================================================================


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    cfg: MetaAlgorithmConfig,
):
    """Train a LightGBM model with the given config.

    Returns:
        lgb.Booster model
    """
    import lightgbm as lgb

    params = {
        "objective": cfg.objective,
        "metric": cfg.metric,
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_child_samples": cfg.min_child_samples,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "verbose": -1,
        "seed": cfg.seed,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    model = lgb.train(params, train_data, num_boost_round=cfg.n_estimators)
    return model


# ====================================================================
# 3. SHAP∩Gain Feature Discovery (delegates to existing)
# ====================================================================


def discover_features(
    df: pd.DataFrame,
    feature_names: List[str],
    model,
    top_n: int = 8,
    compute_interactions: bool = True,
    use_shap: bool = True,
) -> Tuple[List[str], Dict[str, float], List[Tuple]]:
    """SHAP∩Gain 特征发现 — 委托给 export_lightgbm_rules_to_readme._compute_shap_gain_features."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_lightgbm_rules_to_readme import _compute_shap_gain_features

    return _compute_shap_gain_features(
        df,
        feature_names,
        model,
        top_n=top_n,
        compute_interactions=compute_interactions if use_shap else False,
        use_shap=use_shap,
    )


# ====================================================================
# 4. Threshold Sweep
# ====================================================================


def sweep_thresholds(
    df: pd.DataFrame,
    features: List[str],
    rr_col: str,
    label_col: str,
    cfg: MetaAlgorithmConfig,
) -> List[Dict[str, Any]]:
    """Per-feature 百分位阈值扫描 (单阈值 gt/lt + range).

    Returns list of candidate dicts with keys:
      feature, op, threshold, threshold_low, threshold_high,
      lift, robustness, deny_rate, effect_size, score, _deny_mask
    """
    quantiles = cfg.quantiles
    n_total = len(df)
    n_folds = cfg.n_folds
    fold_size = n_total // n_folds

    # Build bad label
    rr_vals = df[rr_col].values.astype(float) if rr_col in df.columns else None
    if rr_vals is None:
        return []

    if label_col in df.columns:
        label_vals = df[label_col].values
        if label_col == "success_no_rr_extreme":
            bad = (label_vals < 0.5).astype(int)
        else:
            bad = (label_vals == 0).astype(int)
    else:
        q30 = np.nanpercentile(rr_vals[~np.isnan(rr_vals)], 30)
        bad = (rr_vals < q30).astype(int)

    overall_bad_rate = float(np.nanmean(bad))
    if overall_bad_rate < 0.05 or overall_bad_rate > 0.95:
        print(f"   ⚠️  bad_rate={overall_bad_rate:.1%} 极端，跳过扫描")
        return []

    candidates = []
    polarity_skipped = 0
    narrow_range_skipped = 0
    for feat in features:
        if feat not in df.columns:
            continue
        col = df[feat].values.astype(float)
        valid = ~np.isnan(col)
        if valid.sum() < 100:
            continue

        thresholds = np.unique(np.quantile(col[valid], quantiles))

        # ── Semantic polarity filter ──
        # higher_is_better → only allow deny_low (direction="lt", flips to ">=" pass)
        # lower_is_better  → only allow deny_high (direction="gt", flips to "<=" pass)
        # unknown          → both directions + range (original behavior)
        feat_polarity = get_feature_polarity(feat, cfg.strategy)
        allow_gt = feat_polarity in ("unknown", "lower_is_better")
        allow_lt = feat_polarity in ("unknown", "higher_is_better")
        allow_range = feat_polarity == "unknown"

        rule_specs = []  # (deny_mask, op_str, thr_val, thr_low, thr_high)

        # (A) Single threshold rules
        for thr in thresholds:
            for direction in ["gt", "lt"]:
                if direction == "gt" and not allow_gt:
                    polarity_skipped += 1
                    continue
                if direction == "lt" and not allow_lt:
                    polarity_skipped += 1
                    continue
                dm = (col > thr) if direction == "gt" else (col < thr)
                dm = dm & valid
                op_s = ">" if direction == "gt" else "<"
                rule_specs.append((dm, op_s, float(thr), None, None))

        # (B) Range rules: deny = in-range (skip for polarity-defined features)
        if allow_range:
            # Range-width 守门：禁止「针尖 range」过拟合规则。
            # width (hi - lo) 必须 >= min_range_width_sigma * std(col)，否则拒收。
            col_std = float(np.nanstd(col[valid])) if valid.any() else 0.0
            min_width_abs = cfg.min_range_width_sigma * col_std if col_std > 0 else 0.0
            for i_lo, thr_lo in enumerate(thresholds):
                for thr_hi in thresholds[i_lo + 1 :]:
                    width = float(thr_hi) - float(thr_lo)
                    if min_width_abs > 0 and width < min_width_abs:
                        narrow_range_skipped += 1
                        continue
                    dm = ((col >= thr_lo) & (col <= thr_hi)) & valid
                    rule_specs.append(
                        (dm, "range_deny", None, float(thr_lo), float(thr_hi))
                    )
        else:
            # counts of range candidates we would have emitted
            n_thr = len(thresholds)
            polarity_skipped += n_thr * (n_thr - 1) // 2

        for deny_mask, _op_str, _thr_val, _thr_low, _thr_high in rule_specs:
            deny_rate = float(deny_mask.mean())
            if deny_rate < cfg.deny_rate_min or deny_rate > cfg.deny_rate_max:
                continue

            # Lift
            bad_in_deny = float(bad[deny_mask].mean()) if deny_mask.any() else 0
            lift = bad_in_deny / overall_bad_rate if overall_bad_rate > 0 else 0
            if lift <= cfg.min_lift:
                continue

            # Effect size
            mean_allow = float(np.nanmean(rr_vals[~deny_mask]))
            mean_deny = float(np.nanmean(rr_vals[deny_mask]))
            effect = mean_allow - mean_deny
            if effect < cfg.min_effect:
                continue

            # Robustness (time-ordered folds)
            fold_lifts = []
            for fi in range(n_folds):
                s = fi * fold_size
                e = (fi + 1) * fold_size if fi < n_folds - 1 else n_total
                fb = bad[s:e]
                fd = deny_mask[s:e]
                fbr = float(fb.mean())
                if fbr > 0 and fd.any():
                    fl = float(fb[fd].mean()) / fbr
                    fold_lifts.append(fl)

            rob_time = sum(1 for fl in fold_lifts if fl > 1.0) / max(len(fold_lifts), 1)
            if len(fold_lifts) >= 2:
                rob_cross = (
                    min(fold_lifts) / max(fold_lifts) if max(fold_lifts) > 0 else 0
                )
                robustness = rob_time * 0.6 + rob_cross * 0.4
            else:
                robustness = rob_time

            if robustness < cfg.min_robustness:
                continue

            score = lift * 0.4 + robustness * 0.4 + (1 - deny_rate) * 0.2

            candidates.append(
                {
                    "feature": feat,
                    "op": _op_str,
                    "threshold": _thr_val,
                    "threshold_low": _thr_low,
                    "threshold_high": _thr_high,
                    "lift": lift,
                    "robustness": robustness,
                    "deny_rate": deny_rate,
                    "effect_size": effect,
                    "score": score,
                    "_deny_mask": deny_mask,
                }
            )

    candidates.sort(key=lambda x: -x["score"])
    if polarity_skipped > 0:
        print(f"   🛡️  semantic-polarity: 跳过 {polarity_skipped} 条反向/range 候选规则")
    if narrow_range_skipped > 0:
        print(
            f"   🛡️  range-width (< {cfg.min_range_width_sigma}σ): 跳过 {narrow_range_skipped} 条针尖 range 候选"
        )
    return candidates


# ====================================================================
# 5. Plateau Validation
# ====================================================================


def find_plateau(
    candidates: List[Dict[str, Any]],
    feature: str,
    min_width: float = 0.05,
    max_lift_std_ratio: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """在同一 feature 的候选中寻找 lift 平台区间.

    Args:
        candidates: 同一 feature 的候选 (已按 threshold 排序)
        feature: 特征名
        min_width: 最小 plateau 宽度
        max_lift_std_ratio: lift 变化容忍度

    Returns:
        plateau info dict or None
    """
    feat_cands = [
        c for c in candidates if c["feature"] == feature and c["threshold"] is not None
    ]
    if len(feat_cands) < 2:
        return None

    feat_cands.sort(key=lambda x: x["threshold"])
    lifts = [c["lift"] for c in feat_cands]
    thresholds = [c["threshold"] for c in feat_cands]

    # Find stable intervals
    best_plateau = None
    best_score = -1

    for i in range(len(feat_cands)):
        anchor_lift = lifts[i]
        j = i + 1
        while j < len(feat_cands):
            lift_change = abs(lifts[j] - anchor_lift)
            if lift_change > max_lift_std_ratio * abs(anchor_lift):
                break
            j += 1

        if j - i >= 2:
            interval = feat_cands[i:j]
            width = thresholds[j - 1] - thresholds[i]
            if width >= min_width:
                int_lifts = [c["lift"] for c in interval]
                lift_mean = np.mean(int_lifts)
                lift_std = np.std(int_lifts)
                plateau_score = lift_mean * (1 - lift_std / max(lift_mean, 0.2))

                if plateau_score > best_score:
                    best_score = plateau_score
                    mid_th = (thresholds[i] + thresholds[j - 1]) / 2
                    best_plateau = {
                        "feature": feature,
                        "plateau_start": thresholds[i],
                        "plateau_end": thresholds[j - 1],
                        "plateau_mid": mid_th,
                        "lift_mean": float(lift_mean),
                        "lift_std": float(lift_std),
                        "width": width,
                        "num_points": j - i,
                        "plateau_score": float(plateau_score),
                    }

    return best_plateau


# ====================================================================
# 6. Correlation Pruning
# ====================================================================


def prune_correlated(
    candidates: List[Dict[str, Any]],
    df: pd.DataFrame,
    max_rules: int = 4,
    corr_threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    """Remove correlated candidates (Spearman > threshold).

    Greedy forward selection: pick top-scored, remove correlated.
    """
    from scipy.stats import spearmanr

    if not candidates:
        return []

    selected = []
    for cand in candidates:
        if len(selected) >= max_rules:
            break

        feat = cand["feature"]
        if feat not in df.columns:
            continue

        # Check correlation with already selected
        is_redundant = False
        for sel in selected:
            sel_feat = sel["feature"]
            if sel_feat not in df.columns:
                continue
            try:
                corr, _ = spearmanr(
                    df[feat].fillna(0).values,
                    df[sel_feat].fillna(0).values,
                )
                if abs(corr) > corr_threshold:
                    is_redundant = True
                    break
            except Exception:
                pass

        if not is_redundant:
            selected.append(cand)

    return selected


# ====================================================================
# 7. Rule Output
# ====================================================================

# Deny → Pass operator flip for prefilter (prefilter.yaml uses PASS semantics)
_DENY_TO_PASS_OP = {
    ">": "<=",
    "<": ">=",
    ">=": "<",
    "<=": ">",
}


def format_rules_prefilter(
    candidates: List[Dict[str, Any]],
    existing_rules: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Format selected candidates into prefilter.yaml structure.

    Prefilter uses PASS semantics: rules describe which samples to KEEP.
    Internal scanning uses deny_mask, so operators need to be flipped.
    """
    rules = list(existing_rules or [])

    for cand in candidates:
        op = cand["op"]
        if op == "range_deny":
            # Range deny → pass = outside range
            # deny = (>= low AND <= high) → pass = (< low OR > high)
            rules.append(
                {
                    "any_of": [
                        {
                            "feature": cand["feature"],
                            "operator": "<",
                            "value": round(cand["threshold_low"], 6),
                        },
                        {
                            "feature": cand["feature"],
                            "operator": ">",
                            "value": round(cand["threshold_high"], 6),
                        },
                    ],
                    "_meta": {
                        "source": "meta_algorithm_unified",
                        "lift": round(cand["lift"], 4),
                        "robustness": round(cand["robustness"], 4),
                        "score": round(cand["score"], 4),
                    },
                }
            )
        else:
            pass_op = _DENY_TO_PASS_OP.get(op, op)
            rules.append(
                {
                    "feature": cand["feature"],
                    "operator": pass_op,
                    "value": round(cand["threshold"], 6),
                    "_meta": {
                        "source": "meta_algorithm_unified",
                        "lift": round(cand["lift"], 4),
                        "robustness": round(cand["robustness"], 4),
                        "score": round(cand["score"], 4),
                    },
                }
            )

    return {"rules": rules}


def format_rules_gate(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Format selected candidates into gate rule format.

    Gate uses DENY semantics: rules describe which samples to REJECT.
    Returns list suitable for _generate_gate_rules_statistical output format.
    """
    rules = []
    for cand in candidates:
        op = cand["op"]
        if op == "range_deny":
            conditions = [
                (cand["feature"], ">=", cand["threshold_low"]),
                (cand["feature"], "<=", cand["threshold_high"]),
            ]
        else:
            conditions = [(cand["feature"], op, cand["threshold"])]

        rules.append(
            {
                "conditions": conditions,
                "coef": cand["lift"],
                "lift": cand["lift"],
                "robustness": cand["robustness"],
                "deny_rate": cand["deny_rate"],
                "effect_size": cand["effect_size"],
                "score": cand["score"],
                "_meta": {"source": "meta_algorithm_unified"},
            }
        )

    return rules


# ====================================================================
# Main Orchestrator
# ====================================================================


def run_meta_algorithm(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    rr_col: str,
    cfg: MetaAlgorithmConfig,
    existing_rules: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """统一 Meta-Algorithm 核心流程.

    Args:
        df: 输入数据 (已经过 prefilter subset 等预处理)
        features: 候选特征列名列表
        label_col: 标签列名 (Gate: success_no_rr_extreme, Prefilter: forward_rr)
        rr_col: 回报列名 (forward_rr)
        cfg: MetaAlgorithmConfig 配置
        existing_rules: 已有的语义规则 (仅 prefilter 用, 会保留在输出中)

    Returns:
        {
            "model": lgb.Booster,
            "top_features": List[str],
            "shap_importance": Dict[str, float],
            "interaction_pairs": List[Tuple],
            "candidates": List[Dict],  # 扫描后全量候选
            "selected": List[Dict],    # 剪枝后最终选中
            "plateaus": Dict[str, Dict],  # per-feature plateau info
            "splits": {"train": df, "test": df, "oos": df},
            "rules_yaml": Dict,  # prefilter.yaml or gate rules
        }
    """
    print(f"\n{'='*110}")
    print(f"🔬 Meta-Algorithm Unified ({cfg.layer.upper()}, {cfg.strategy})")
    print(f"{'='*110}")

    # ── 0. Validate features ──
    avail_features = [f for f in features if f in df.columns]
    if len(avail_features) < 2:
        print(f"❌ 可用特征不足 ({len(avail_features)} < 2)")
        return {"error": "insufficient_features"}

    print(f"📊 特征: {len(avail_features)} 列, 数据: {len(df)} 行")

    # ── 1. Time split ──
    splits = time_split(
        df,
        mode=cfg.mode,
        test_ratio=cfg.test_ratio,
        oos_ratio=cfg.oos_ratio,
        holdout_ratio=cfg.holdout_ratio,
    )
    df_train = splits["train"]
    df_test = splits["test"]
    df_oos = splits["oos"]

    print(f"📐 数据分割 ({cfg.mode} mode):")
    print(f"   Train:  {len(df_train)} 行")
    if cfg.mode == "research":
        print(f"   Test:   {len(df_test)} 行")
    print(f"   OOS:    {len(df_oos)} 行")

    # ── 2. LightGBM training ──
    X_train = df_train[avail_features].fillna(0).values.astype(float)

    if cfg.objective == "binary" or cfg.scoring_method == "positive_rr":
        if cfg.scoring_method == "positive_rr":
            y_train = (
                (df_train[rr_col] > cfg.positive_threshold)
                .fillna(False)
                .astype(float)
                .values
            )
            cfg.objective = "binary"
            cfg.metric = "binary_logloss"
        else:
            y_train = df_train[label_col].fillna(0).values.astype(float)
    else:
        y_train = df_train[rr_col].clip(-2, 2).fillna(0).values.astype(float)

    print(
        f"\n🌲 LightGBM 训练: {len(avail_features)} 特征, {len(df_train)} 样本, {cfg.n_estimators} 轮"
    )
    model = train_lightgbm(X_train, y_train, avail_features, cfg)
    print(f"   ✅ 训练完成")

    # ── 3. SHAP∩Gain feature discovery ──
    print(f"\n🔍 SHAP∩Gain 特征发现 (top_n={cfg.shap_top_n})")
    top_features, shap_importance, interaction_pairs = discover_features(
        df_train,
        avail_features,
        model,
        top_n=cfg.shap_top_n,
        compute_interactions=cfg.compute_interactions,
    )
    print(f"   Top features: {top_features}")

    # ── 4. Threshold sweep on test/holdout ──
    # Research: sweep on test, validate on OOS
    # Deploy: sweep on OOS (train+test merged)
    if cfg.mode == "research":
        df_sweep = df_test
        df_validate = df_oos
    else:
        df_sweep = df_oos
        df_validate = df_oos  # same set for deploy

    print(f"\n📊 阈值扫描 ({len(df_sweep)} 行, {len(top_features)} 特征)")
    candidates = sweep_thresholds(df_sweep, top_features, rr_col, label_col, cfg)
    print(f"   {len(candidates)} 候选规则通过筛选")

    # ── 5. Plateau validation per feature ──
    plateaus = {}
    for feat in top_features:
        plateau = find_plateau(
            candidates,
            feat,
            min_width=cfg.min_plateau_width,
            max_lift_std_ratio=cfg.max_lift_std_ratio,
        )
        if plateau is not None:
            plateaus[feat] = plateau
            print(
                f"   📈 {feat}: plateau [{plateau['plateau_start']:.4f}, "
                f"{plateau['plateau_end']:.4f}] lift={plateau['lift_mean']:.3f}"
            )

    # ── 6. Correlation pruning ──
    selected = prune_correlated(
        candidates,
        df_sweep,
        max_rules=cfg.max_rules,
        corr_threshold=cfg.correlation_threshold,
    )
    print(f"\n✂️  相关性剪枝: {len(candidates)} → {len(selected)} 规则")

    # ── 7. OOS validation (research mode) ──
    if cfg.mode == "research" and len(df_validate) > 0 and selected:
        print(f"\n🔍 OOS 验证 ({len(df_validate)} 行)")
        _validate_on_oos(selected, df_validate, rr_col, label_col)

    # ── 8. Format rules ──
    if cfg.layer == "prefilter":
        rules_yaml = format_rules_prefilter(selected, existing_rules)
    else:
        rules_yaml = {"rules": format_rules_gate(selected)}

    for s in selected:
        s.pop("_deny_mask", None)  # clean up non-serializable

    return {
        "model": model,
        "top_features": top_features,
        "shap_importance": shap_importance,
        "interaction_pairs": interaction_pairs,
        "candidates": [
            {k: v for k, v in c.items() if k != "_deny_mask"} for c in candidates
        ],
        "selected": selected,
        "plateaus": plateaus,
        "splits": splits,
        "rules_yaml": rules_yaml,
    }


def _validate_on_oos(
    selected: List[Dict[str, Any]],
    df_oos: pd.DataFrame,
    rr_col: str,
    label_col: str,
):
    """Validate selected rules on OOS data, print diagnostics."""
    if rr_col not in df_oos.columns:
        print("   ⚠️  OOS 中无 rr_col, 跳过验证")
        return

    rr_vals = df_oos[rr_col].values.astype(float)
    overall_mean_rr = float(np.nanmean(rr_vals))
    overall_hit_rate = float((rr_vals > 0).mean())

    # Build bad label for OOS
    if label_col in df_oos.columns:
        label_vals = df_oos[label_col].values
        if label_col == "success_no_rr_extreme":
            bad = (label_vals < 0.5).astype(int)
        else:
            bad = (label_vals == 0).astype(int)
    else:
        q30 = np.nanpercentile(rr_vals[~np.isnan(rr_vals)], 30)
        bad = (rr_vals < q30).astype(int)

    overall_bad_rate = float(np.nanmean(bad))

    print(
        f"   OOS baseline: mean_rr={overall_mean_rr:+.4f}, hit_rate={overall_hit_rate:.1%}, bad_rate={overall_bad_rate:.1%}"
    )

    for i, cand in enumerate(selected):
        feat = cand["feature"]
        if feat not in df_oos.columns:
            continue

        col = df_oos[feat].values.astype(float)
        valid = ~np.isnan(col)
        op = cand["op"]
        thr = cand.get("threshold")

        if op == "range_deny":
            deny_mask = (
                (col >= cand["threshold_low"]) & (col <= cand["threshold_high"]) & valid
            )
        elif op == ">":
            deny_mask = (col > thr) & valid
        elif op == "<":
            deny_mask = (col < thr) & valid
        else:
            continue

        if deny_mask.sum() == 0:
            continue

        oos_deny_rate = float(deny_mask.mean())
        oos_bad_deny = float(bad[deny_mask].mean()) if deny_mask.any() else 0
        oos_lift = oos_bad_deny / overall_bad_rate if overall_bad_rate > 0 else 0
        oos_mean_rr_allow = float(np.nanmean(rr_vals[~deny_mask]))
        oos_effect = oos_mean_rr_allow - float(np.nanmean(rr_vals[deny_mask]))

        status = "✅" if oos_lift > 1.0 and oos_effect > 0 else "⚠️"
        print(
            f"   {status} Rule {i+1}: {feat} {op} {thr or ''}"
            f" → OOS lift={oos_lift:.2f}, effect={oos_effect:+.4f}, deny_rate={oos_deny_rate:.1%}"
        )
