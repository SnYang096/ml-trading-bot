#!/usr/bin/env python3
"""
Direction.yaml 严格验证 — 覆盖率 + 方向质量

验证每个策略的 predictions 数据是否能通过 direction.yaml 规则确定方向，
以及方向模块是否优于随机方向基线。

验证项:
  Phase 1 - 覆盖率验证:
    - 非零方向占比 (目标 100%)
    - 命中规则统计
    - 方向分布 (Long / Short / Zero)
  Phase 2 - 方向质量验证:
    - median(rr × direction) > 0 → 方向信号优于随机 (置换检验 p<0.05)
    - bad rate(按方向) vs bad rate(随机)
    - 每个方向子集的样本量校验 (>= 1080)

用法:
    python scripts/direction_strict_validation.py
    python scripts/direction_strict_validation.py --logs path/to/predictions.parquet --strategy me
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.locked_direction_utils import merge_direction_rules_for_promote
from src.time_series_model.live.direction_rule_ops import (
    direction_rule_ft_key,
    dual_position_agree_deadband_series,
    is_direction_rule_enabled,
    parse_dual_rule,
    parse_signal_match_position_band_rule,
    parse_single_position_band_rule,
    signal_match_position_band_series,
    single_position_band_series,
)

PREDICTIONS = {
    "bpc": "results/train_final_20260208_220616_return_tree/bpc/predictions.parquet",
    "me": "results/train_final_20260215_234211_return_tree/me/predictions_fixed.parquet",
    "fer": "results/train_final_20260216_184525_return_tree/fer/predictions_fixed.parquet",
}

STRATEGIES_ROOT = PROJECT_ROOT / "config" / "strategies"

# 方向「候选 + 评估回写」工作区文件名 (与 gate 的 gate_draft / prefilter 的 features_prefilter 同级叙事)
# promote 时合并 locked(来自 archetypes) + 非 locked(来自工作区 direction_rules) 写入 archetypes/direction.yaml
DEFAULT_DIRECTION_WORKSPACE = "features_direction.yaml"

# 方向质量验证的最小可信样本量
MIN_CREDIBLE_SAMPLES = 1080

# forward_rr 候选列名 (按优先级)
RR_COLUMN_CANDIDATES = ["forward_rr", "bpc_impulse_return_atr", "rr", "return_atr"]

# failure 阈值
FAILURE_RR_THRESHOLD = -0.8


def load_direction_config(strategy: str) -> dict:
    """加载实盘/验证用的方向规则 (archetypes/direction.yaml)。"""
    path = STRATEGIES_ROOT / strategy / "archetypes" / "direction.yaml"
    if not path.exists():
        raise FileNotFoundError(f"direction.yaml 不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _direction_workspace_path(strategy: str, basename: str) -> Path:
    return STRATEGIES_ROOT / strategy / basename


def _direction_config_write_path(strategy: str, workspace_basename: str) -> Path:
    """回写 candidates / last_evaluation / 草稿 direction_rules — 仅方向工作区文件。"""
    return _direction_workspace_path(strategy, workspace_basename)


def validate_archetype(arch_name: str, df: pd.DataFrame) -> dict:
    """严格用 direction.yaml 规则确定方向，不做任何兜底"""
    cfg = load_direction_config(arch_name)
    rules = cfg.get("direction_rules", [])
    if not rules:
        raise ValueError(f"{arch_name}: direction.yaml 无 direction_rules")

    n = len(df)
    direction = pd.Series(0.0, index=df.index)
    assigned = pd.Series(False, index=df.index)
    rule_hits = {}  # rule_feature -> count

    for rule in rules:
        if not is_direction_rule_enabled(rule):
            continue
        dual = parse_dual_rule(rule)
        band = parse_single_position_band_rule(rule)
        rk = str(rule.get("id", "")) or ""
        if dual is not None:
            col_a, col_b, eps = dual
            rk = rk or f"dual_{col_a}_{col_b}"
            if col_a not in df.columns or col_b not in df.columns:
                rule_hits[rk] = {"status": "NOT_IN_DATA", "count": 0}
                continue
            vals = dual_position_agree_deadband_series(df, col_a, col_b, eps)
            unassigned = ~assigned
            direction.loc[unassigned] = vals.loc[unassigned]
            newly = unassigned & (direction != 0)
            count = int(newly.sum())
            assigned = assigned | newly
            rule_hits[rk] = {
                "status": "HIT",
                "transform": "dual_position_agree_deadband",
                "count": count,
                "epsilon": eps,
            }
            if assigned.all():
                break
            continue

        if band is not None:
            fcol, inner_a, outer_a = band
            rk = rk or fcol
            if fcol not in df.columns:
                rule_hits[rk] = {"status": "NOT_IN_DATA", "count": 0}
                continue
            vals = single_position_band_series(df, fcol, inner_a, outer_a)
            unassigned = ~assigned
            direction.loc[unassigned] = vals.loc[unassigned]
            newly = unassigned & (direction != 0)
            count = int(newly.sum())
            assigned = assigned | newly
            rule_hits[rk] = {
                "status": "HIT",
                "transform": "single_position_band",
                "inner_abs": inner_a,
                "outer_abs": outer_a,
                "count": count,
            }
            if assigned.all():
                break
            continue

        feature = rule.get("feature", "")
        transform = rule.get("transform", "raw")

        if feature not in df.columns:
            rule_hits[feature] = {"status": "NOT_IN_DATA", "count": 0}
            continue

        series = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
        unassigned = ~assigned

        if transform == "raw":
            vals = series
        elif transform == "sign":
            vals = np.sign(series)
        elif transform == "negate_sign":
            vals = -np.sign(series)
        elif transform == "center_sign":
            vals = np.sign(series - 0.5)
        else:
            vals = series

        direction.loc[unassigned] = vals.loc[unassigned]
        newly = unassigned & (direction != 0)
        count = int(newly.sum())
        assigned = assigned | newly
        rule_hits[feature] = {
            "status": "HIT",
            "transform": transform,
            "count": count,
        }

        if assigned.all():
            break

    # 统计
    n_nonzero = int((direction != 0).sum())
    n_long = int((direction > 0).sum())
    n_short = int((direction < 0).sum())
    n_zero = int((direction == 0).sum())
    coverage = n_nonzero / n * 100 if n > 0 else 0

    return {
        "archetype": arch_name,
        "causal_source": cfg.get("causal_source", "?"),
        "rows": n,
        "coverage_pct": round(coverage, 2),
        "long": n_long,
        "short": n_short,
        "zero": n_zero,
        "rule_hits": rule_hits,
    }


def _find_rr_column(df: pd.DataFrame) -> Optional[str]:
    """在 DataFrame 中查找 forward_rr 列。"""
    for col in RR_COLUMN_CANDIDATES:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def compute_direction_series_from_rules(df: pd.DataFrame, rules: list) -> pd.Series:
    """Apply direction_rules to df (same cascade as live/backtest)."""
    if not rules:
        raise ValueError("direction_rules 为空")

    direction = pd.Series(0.0, index=df.index)
    assigned = pd.Series(False, index=df.index)

    for rule in rules:
        if not is_direction_rule_enabled(rule):
            continue
        cmp = parse_signal_match_position_band_rule(rule)
        if cmp is not None:
            need = [cmp["band_feature"]]
            for sr in cmp["signal_rules"]:
                if not isinstance(sr, dict):
                    continue
                dr = parse_dual_rule(sr)
                if dr is not None:
                    need.extend([dr[0], dr[1]])
                    continue
                b2 = parse_single_position_band_rule(sr)
                if b2 is not None:
                    need.append(b2[0])
                    continue
                f2 = sr.get("feature")
                if f2:
                    need.append(str(f2))
            if any(c not in df.columns for c in need):
                continue
            vals = signal_match_position_band_series(
                df,
                signal_rules=cmp["signal_rules"],
                band_feature=cmp["band_feature"],
                inner_abs=float(cmp["inner_abs"]),
                outer_abs=float(cmp["outer_abs"]),
                consensus_mode=cmp.get("consensus_mode", "first"),
            )
            unassigned = ~assigned
            direction.loc[unassigned] = vals.loc[unassigned]
            newly = unassigned & (direction != 0)
            assigned = assigned | newly
            if assigned.all():
                break
            continue

        dual = parse_dual_rule(rule)
        band = parse_single_position_band_rule(rule)
        if dual is not None:
            col_a, col_b, eps = dual
            if col_a not in df.columns or col_b not in df.columns:
                continue
            vals = dual_position_agree_deadband_series(df, col_a, col_b, eps)
            unassigned = ~assigned
            direction.loc[unassigned] = vals.loc[unassigned]
            newly = unassigned & (direction != 0)
            assigned = assigned | newly
            if assigned.all():
                break
            continue

        if band is not None:
            fcol, inner_a, outer_a = band
            if fcol not in df.columns:
                continue
            vals = single_position_band_series(df, fcol, inner_a, outer_a)
            unassigned = ~assigned
            direction.loc[unassigned] = vals.loc[unassigned]
            newly = unassigned & (direction != 0)
            assigned = assigned | newly
            if assigned.all():
                break
            continue

        feature = rule.get("feature", "")
        transform = rule.get("transform", "raw")
        if feature not in df.columns:
            continue

        series = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
        unassigned = ~assigned

        if transform == "sign":
            vals = np.sign(series)
        elif transform == "negate_sign":
            vals = -np.sign(series)
        elif transform == "center_sign":
            vals = np.sign(series - 0.5)
        else:
            vals = series

        direction.loc[unassigned] = vals.loc[unassigned]
        newly = unassigned & (direction != 0)
        assigned = assigned | newly
        if assigned.all():
            break

    return direction


def compute_direction_series(arch_name: str, df: pd.DataFrame) -> pd.Series:
    """从 direction.yaml 规则计算 per-bar 方向 Series。

    复用 validate_archetype 的逻辑，但只返回方向 Series。
    """
    cfg = load_direction_config(arch_name)
    rules = cfg.get("direction_rules", [])
    if not rules:
        raise ValueError(f"{arch_name}: direction.yaml 无 direction_rules")
    return compute_direction_series_from_rules(df, rules)


def _permutation_p_value(
    rr_arr: np.ndarray,
    dir_arr: np.ndarray,
    observed_median: float,
    n_perm: int = 200,
) -> float:
    """置换检验: 方向信号是否优于随机方向。

    H0: direction 与 forward_rr 无关 (随机方向, 期望 median ≈ 0)
    H1: median(rr × direction) > 0 (方向信号有信息)

    方法: 保持 +1/-1 比例不变, 随机打乱 direction 的行分配,
    计算 n_perm 次 median(rr × shuffled_dir)。
    p = (#{perm_median >= observed} + 1) / (n_perm + 1)
    """
    count_ge = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        perm_dir = rng.permutation(dir_arr)
        perm_med = float(np.median(rr_arr * perm_dir))
        if perm_med >= observed_median:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)


def validate_direction_quality(
    arch_name: str,
    df: pd.DataFrame,
    direction: pd.Series,
) -> Optional[Dict[str, Any]]:
    """Phase 2: 方向质量验证 — 方向信号是否优于随机方向。

    核心公式:
        rr_in_direction = forward_rr_long × direction
        随机方向基线: median(rr × random_direction) ≈ 0

    通过标准:
        1. median(rr × direction) > 0  → 方向信号优于随机
        2. 置换检验 p < 0.05             → 统计显著
        3. short 子集样本量 >= MIN_CREDIBLE_SAMPLES → 统计可信
    """
    rr_col = _find_rr_column(df)
    if rr_col is None:
        return None

    # 对齐: 只用 forward_rr 和 direction 都有效的行
    valid_mask = df[rr_col].notna() & (direction != 0)
    rr = df.loc[valid_mask, rr_col].astype(float)
    dir_s = direction[valid_mask]

    if len(rr) < MIN_CREDIBLE_SAMPLES:
        return {
            "status": "INSUFFICIENT_DATA",
            "n_valid": len(rr),
            "min_required": MIN_CREDIBLE_SAMPLES,
        }

    # ── 核心指标 ──────────────────────────────────────────
    rr_in_dir = rr * dir_s  # 按方向调整后的 RR

    # 中位数 / 均值 (vs 随机基线 = 0)
    med_in_dir = float(rr_in_dir.median())
    mean_in_dir = float(rr_in_dir.mean())

    # Always-long 参考 (仅供对比, 不作为通过标准)
    med_long = float(rr.median())
    mean_long = float(rr.mean())

    # Bad rate: failure_rr_extreme (rr < -0.8)
    bad_in_dir = float((rr_in_dir < FAILURE_RR_THRESHOLD).mean())
    bad_long = float((rr < FAILURE_RR_THRESHOLD).mean())

    # 置换检验: 方向信号 vs 随机方向
    p_random = _permutation_p_value(rr.values, dir_s.values, med_in_dir, n_perm=500)

    # ── Long / Short 子集分析 ──────────────────────────────
    long_mask = dir_s > 0
    short_mask = dir_s < 0
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())

    long_med_rr = float(rr[long_mask].median()) if n_long > 0 else float("nan")
    short_med_rr = float((-rr[short_mask]).median()) if n_short > 0 else float("nan")

    long_bad = (
        float((rr[long_mask] < FAILURE_RR_THRESHOLD).mean())
        if n_long > 0
        else float("nan")
    )
    short_bad = (
        float((-rr[short_mask] < FAILURE_RR_THRESHOLD).mean())
        if n_short > 0
        else float("nan")
    )

    # ── 判定 (基线: 随机方向) ──────────────────────────────
    direction_lift_bad = bad_long - bad_in_dir  # 正 = 方向模块降低了 bad rate

    passes = []
    passes.append(("median_rr_positive", med_in_dir > 0, med_in_dir))
    passes.append(("perm_significant", p_random < 0.05, p_random))
    passes.append(("mean_positive", mean_in_dir > 0, mean_in_dir))
    passes.append(("short_credible", n_short >= MIN_CREDIBLE_SAMPLES, n_short))
    passes.append(
        (
            "short_ratio",
            n_short / (n_long + n_short) > 0.15 if (n_long + n_short) > 0 else False,
            (
                round(n_short / (n_long + n_short) * 100, 1)
                if (n_long + n_short) > 0
                else 0
            ),
        )
    )

    return {
        "status": "OK",
        "rr_col": rr_col,
        "n_valid": len(rr),
        # vs 随机基线 (主要标准)
        "median_in_direction": round(med_in_dir, 4),
        "p_random": round(p_random, 4),
        "mean_in_direction": round(mean_in_dir, 4),
        "bad_rate_in_direction": round(bad_in_dir, 4),
        # vs always-long (仅参考)
        "median_always_long": round(med_long, 4),
        "lift_vs_long": round(med_in_dir - med_long, 4),
        "mean_always_long": round(mean_long, 4),
        "bad_rate_always_long": round(bad_long, 4),
        "bad_rate_reduction": round(direction_lift_bad, 4),
        # 子集
        "n_long": n_long,
        "n_short": n_short,
        "short_pct": (
            round(n_short / (n_long + n_short) * 100, 1)
            if (n_long + n_short) > 0
            else 0
        ),
        "long_median_rr": round(long_med_rr, 4),
        "short_median_rr": round(short_med_rr, 4),
        "long_bad_rate": round(long_bad, 4),
        "short_bad_rate": round(short_bad, 4),
        # 判定
        "checks": passes,
    }


def _print_coverage(arch_name: str, r: dict) -> bool:
    """打印 Phase 1 覆盖率验证结果，返回是否通过。"""
    print(f"\n{'─' * 70}")
    print(f"  {arch_name.upper()} | causal_source={r['causal_source']}")
    print(f"  \n  Phase 1: 覆盖率验证")
    print(f"  行数: {r['rows']}")
    print(f"  覆盖率: {r['coverage_pct']}%")
    print(f"  方向分布: Long={r['long']}  Short={r['short']}  Zero={r['zero']}")
    if (r["long"] + r["short"]) > 0:
        short_pct = r["short"] / (r["long"] + r["short"]) * 100
        print(f"  Short 占比: {short_pct:.1f}%")
    print(f"  规则命中:")
    for feat, info in r["rule_hits"].items():
        if info["status"] == "NOT_IN_DATA":
            print(f"    ⬜ {feat}: 列不存在 (跳过)")
        else:
            print(
                f"    ✅ {feat} (transform={info['transform']}): "
                f"命中 {info['count']} 行"
            )

    passed = True
    if r["coverage_pct"] < 100:
        print(f"  ⚠️  覆盖率不足 100% — {r['zero']} 行无方向")
        if r["zero"] > 0:
            passed = False
    else:
        print(f"  ✅ 覆盖率 100%")
    return passed


def _print_quality(arch_name: str, q: Optional[Dict]) -> bool:
    """打印 Phase 2 方向质量验证结果，返回是否通过。"""
    print(f"  \n  Phase 2: 方向质量验证 (基线: 随机方向)")

    if q is None:
        print(f"  ⬜ 无 forward_rr 列，跳过质量验证")
        return True  # 不阻断

    if q["status"] == "INSUFFICIENT_DATA":
        print(f"  ❌ 数据量不足: {q['n_valid']} < {q['min_required']}")
        return False

    print(f"  RR 列: {q['rr_col']}  |  有效行: {q['n_valid']}")
    print(f"")
    print(f"  ┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
    print(f"  │ 指标                │ 按方向交易   │ 随机方向     │ Always-Long  │")
    print(f"  ├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(
        f"  │ Median RR           │ {q['median_in_direction']:>+11.4f} │      ≈0      │ {q['median_always_long']:>+11.4f} │"
    )
    print(
        f"  │ Mean RR             │ {q['mean_in_direction']:>+11.4f} │      ≈0      │ {q['mean_always_long']:>+11.4f} │"
    )
    print(
        f"  │ Bad Rate (<-0.8R)   │ {q['bad_rate_in_direction']:>10.1%}  │     ~50%     │ {q['bad_rate_always_long']:>10.1%}  │"
    )
    print(f"  └─────────────────────┴──────────────┴──────────────┴──────────────┘")
    p_str = f"{q['p_random']:.4f}" if "p_random" in q else "N/A"
    print(
        f"  置换检验 p-value: {p_str} {'✅ 显著' if q.get('p_random', 1) < 0.05 else '❌ 不显著'}"
    )
    print(f"")
    print(
        f"  Long 子集:  n={q['n_long']:>6d}  median_rr={q['long_median_rr']:>+.4f}  bad_rate={q['long_bad_rate']:.1%}"
    )
    print(
        f"  Short 子集: n={q['n_short']:>6d}  median_rr={q['short_median_rr']:>+.4f}  bad_rate={q['short_bad_rate']:.1%}"
    )
    print(f"  Short 占比: {q['short_pct']:.1f}%")
    print(f"  (参考) vs Always-Long lift: {q.get('lift_vs_long', 0):+.4f}")
    print(f"")

    all_ok = True
    for name, passed, value in q["checks"]:
        icon = "✅" if passed else "❌"
        if name == "median_rr_positive":
            desc = f"Median RR > 0 (优于随机): {value:+.4f}"
        elif name == "perm_significant":
            desc = f"置换检验 p<0.05: p={value:.4f}"
        elif name == "mean_positive":
            desc = f"Mean RR > 0: {value:+.4f}"
        elif name == "short_credible":
            desc = f"Short 样本量 >= {MIN_CREDIBLE_SAMPLES}: n={value}"
        elif name == "short_ratio":
            desc = f"Short 占比 > 15%: {value}%"
        else:
            desc = f"{name}: {value}"
        print(f"  {icon} {desc}")
        if not passed:
            all_ok = False

    return all_ok


# ── Compare-features 模式 ─────────────────────────────────────────


def _load_direction_candidates(
    strategy: str, workspace_basename: str
) -> Optional[list]:
    """读取方向工作区 YAML 的 candidates（无回退路径）。

    返回 [(feature, transform_or_None), ...] 或 None (文件不存在/无 candidates)。
    """
    path = _direction_workspace_path(strategy, workspace_basename)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("candidates", [])
    if not raw:
        return None
    result = []
    for item in raw:
        if isinstance(item, dict):
            feat = item.get("feature", "")
            transform = item.get("transform", None)  # None = auto-detect
        elif isinstance(item, str):
            feat = item
            transform = None
        else:
            continue
        if feat:
            result.append((feat, transform))
    return result if result else None


def _detect_transform(series: pd.Series) -> str:
    """自动检测合适的 transform: 全非负 → center_sign, 否则 sign。"""
    valid = series.dropna()
    if len(valid) == 0:
        return "sign"
    if valid.min() >= 0 and valid.max() <= 1.0:
        return "center_sign"
    return "sign"


def _apply_direction_transform(series: pd.Series, transform: str) -> pd.Series:
    """对一列特征应用方向变换，返回 +1/-1/0 Series。"""
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if transform == "center_sign":
        return np.sign(s - 0.5)
    return np.sign(s)


def compare_direction_features(
    df: pd.DataFrame,
    rr_col: str,
    candidates: Optional[list] = None,
    config_candidates: Optional[list] = None,
) -> list:
    """对比多个候选特征的方向质量，返回排名列表。

    候选来源 (优先级):
      1. config_candidates: 从 direction.yaml 读取的 [(feature, transform), ...]
      2. candidates: 列名列表 (auto-detect transform)
      3. None + None: 自动扫描全部数值特征
    """
    rr = df[rr_col].astype(float)
    rr_long_median = float(rr.median())
    rr_long_mean = float(rr.mean())
    bad_long = float((rr < FAILURE_RR_THRESHOLD).mean())

    # Build feature list with transforms
    feat_transform_pairs: list = []

    if config_candidates is not None:
        # Mode: config-driven (from direction.yaml)
        for feat, transform in config_candidates:
            if feat in df.columns:
                t = transform if transform else _detect_transform(df[feat])
                feat_transform_pairs.append((feat, t))
            # else: silently skip missing features
    elif candidates is not None:
        for feat in candidates:
            if feat in df.columns:
                feat_transform_pairs.append((feat, _detect_transform(df[feat])))
    else:
        # Auto-discover: scan all numeric features
        _exclude = {
            rr_col,
            "forward_rr",
            "success_no_rr_extreme",
            "failure_rr_extreme",
            "target",
            "sample_weight",
            "timestamp",
            "datetime",
            "date",
            "symbol",
            "_symbol",
        }
        for col in sorted(df.columns):
            if col in _exclude:
                continue
            if df[col].dtype not in ["float64", "float32", "int64", "int32"]:
                continue
            s = df[col].dropna()
            if len(s) < MIN_CREDIBLE_SAMPLES:
                continue
            neg_pct = (s < 0).sum() / len(s)
            if 0.15 < neg_pct < 0.85:
                feat_transform_pairs.append((col, "sign"))
            elif neg_pct == 0 and s.min() >= 0 and s.max() <= 1.0 and s.mean() > 0.1:
                feat_transform_pairs.append((col, "center_sign"))

    results = []
    for feat, transform in feat_transform_pairs:
        direction = _apply_direction_transform(df[feat], transform)

        # Skip features with too few short
        n_short = int((direction < 0).sum())
        n_long = int((direction > 0).sum())
        n_total = n_long + n_short
        if n_short < MIN_CREDIBLE_SAMPLES or n_long < MIN_CREDIBLE_SAMPLES:
            continue

        rr_in_dir = rr * direction
        valid = direction != 0
        rr_dir_valid = rr_in_dir[valid]

        med_dir = float(rr_dir_valid.median())  # median(rr × direction), vs random ≈ 0
        mean_dir = float(rr_dir_valid.mean())
        bad_dir = float((rr_dir_valid < FAILURE_RR_THRESHOLD).mean())

        # 置换检验: 方向信号 vs 随机方向
        p_rand = _permutation_p_value(
            rr[valid].values, direction[valid].values, med_dir, n_perm=200
        )

        results.append(
            {
                "feature": feat,
                "transform": transform,
                "n_long": n_long,
                "n_short": n_short,
                "short_pct": round(n_short / n_total * 100, 1),
                "median_rr_in_dir": round(med_dir, 4),  # 主指标 (vs random=0)
                "p_random": round(p_rand, 4),  # 置换检验 p-value
                "lift_vs_long": round(
                    med_dir - rr_long_median, 4
                ),  # 参考: vs always-long
                "mean_rr": round(mean_dir, 4),
                "bad_rate_reduction": round((bad_long - bad_dir) * 100, 2),
            }
        )

    # Sort by median_rr_in_dir descending (vs random baseline)
    results.sort(key=lambda x: x["median_rr_in_dir"], reverse=True)
    return results


def _print_compare_table(results: list, rr_col: str) -> None:
    """打印方向特征对比排名表 (基线: 随机方向)。"""
    print(f"\n{'=' * 110}")
    print(f"Direction 特征对比排名 (RR列: {rr_col}, 基线: 随机方向, 按 Med.RR 排序)")
    print(f"{'=' * 110}")
    print(
        f"{'Rank':>4s}  {'特征':<32s} {'Transform':<13s} "
        f"{'Short%':>7s} {'Med.RR':>9s} {'p_rand':>7s} {'vsLong':>9s} {'MeanRR':>9s} {'BadR↓%':>7s}"
    )
    print(f"{'-' * 110}")

    for i, r in enumerate(results, 1):
        p_str = f"{r['p_random']:.3f}" if r["p_random"] >= 0.001 else "<.001"
        sig = "*" if r["p_random"] < 0.05 else " "
        marker = " 🏆" if i == 1 else ""
        # 标注常数偏差警告: short% > 80% 或 < 20%
        bias_warn = ""
        if r["short_pct"] > DIR_RULE_MAX_SHORT_PCT:
            bias_warn = " ⚠️常数做空"
        elif r["short_pct"] < DIR_RULE_MIN_SHORT_PCT:
            bias_warn = " ⚠️常数做多"
        print(
            f"{i:>4d}  {r['feature']:<32s} {r['transform']:<13s} "
            f"{r['short_pct']:>6.1f}% {r['median_rr_in_dir']:>+8.4f} "
            f"{p_str:>6s}{sig} {r['lift_vs_long']:>+8.4f} "
            f"{r['mean_rr']:>+8.4f} {r['bad_rate_reduction']:>+6.2f}%{marker}{bias_warn}"
        )

    print(f"{'-' * 110}")
    print(
        f"  共 {len(results)} 个候选特征通过双向分布筛选 (Short >= {MIN_CREDIBLE_SAMPLES})"
    )
    print(f"  Med.RR = median(rr × direction) — 随机方向基线 ≈ 0, >0 即优于随机")
    print(f"  p_rand = 置换检验 p-value (*=显著 p<0.05)")
    print(f"  vsLong = Med.RR - median(rr_always_long) — 参考, 非主要标准")
    print(f"  BadR↓% = bad_rate_always_long - bad_rate_in_direction (正=更好)")
    if results:
        best = results[0]
        sig_str = "显著" if best["p_random"] < 0.05 else "不显著"
        print(f"\n  🏆 推荐: {best['feature']} ({best['transform']})")
        print(
            f"     Med.RR={best['median_rr_in_dir']:+.4f} (p={best['p_random']:.4f} {sig_str}), "
            f"Short={best['short_pct']:.1f}%, vsLong={best['lift_vs_long']:+.4f}"
        )
    print(f"{'=' * 110}")


# ── Temporal stability for compare-features ───────────────────────

TEMPORAL_WINDOW_MONTHS = [2, 3, 4, 6]
TEMPORAL_MIN_SAMPLES = 1080
TEMPORAL_TOP_N = 20  # 只对 Top N 做时间分析


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    """找到时间列。"""
    for col in ["timestamp", "datetime", "date"]:
        if col in df.columns:
            return col
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return "__index__"
    return None


def _get_times(df: pd.DataFrame, time_col: str) -> pd.Series:
    """获取时间序列。"""
    if time_col == "__index__":
        return pd.to_datetime(df.index)
    return pd.to_datetime(df[time_col])


def _compute_window_lift(
    rr: pd.Series,
    direction: pd.Series,
    min_samples: int = TEMPORAL_MIN_SAMPLES,
) -> Optional[float]:
    """计算单个窗口的 median(rr × direction)。

    随机方向基线 = 0, 所以 median(rr × direction) 本身就是 lift。
    """
    valid = direction != 0
    rr_dir = (rr * direction)[valid]
    if len(rr_dir) < min_samples:
        return None
    return float(rr_dir.median())


def temporal_direction_stability(
    df: pd.DataFrame,
    rr_col: str,
    top_results: list,
) -> Dict[str, Dict]:
    """对 Top N 方向特征做 rolling Median Lift 时间稳定性分析。

    对齐 prefilter --temporal 的模式:
      - 多窗口 [2m, 3m, 4m, 6m], 选最优窗口 (avg CV 最小)
      - 每个特征: rolling Median Lift + CV
      - 判定: CV < 0.3 稳定, < 0.5 一般, >= 0.5 不稳
    """
    time_col = _find_time_column(df)
    if time_col is None:
        print("\n⚠️  无时间列, 跳过 --temporal 分析")
        return {}

    times = _get_times(df, time_col)
    t_min, t_max = times.min(), times.max()
    total_months = (t_max.year - t_min.year) * 12 + (t_max.month - t_min.month)
    rr = df[rr_col].astype(float)

    print(f"\n{'=' * 110}")
    print(f"🕰️  方向特征时间稳定性分析 (--temporal)")
    print(
        f"   时间范围: {t_min.strftime('%Y-%m')} → {t_max.strftime('%Y-%m')}, 共 {total_months} 个月"
    )
    print(f"   分析 Top {len(top_results)} 个特征")
    print(f"   候选窗口: {', '.join(f'{w}m' for w in TEMPORAL_WINDOW_MONTHS)}")

    # 对每个窗口大小，对每个特征计算 rolling median_lift
    window_results: Dict[int, list] = {}

    for wm in TEMPORAL_WINDOW_MONTHS:
        window_results[wm] = []

        # 生成窗口 (步长 1 个月)
        window_start = t_min
        windows = []
        while True:
            window_end = window_start + pd.DateOffset(months=wm)
            if window_end > t_max + pd.Timedelta(days=1):
                break
            windows.append((window_start, window_end))
            window_start += pd.DateOffset(months=1)

        if len(windows) < 3:
            continue

        for r in top_results:
            feat = r["feature"]
            transform = r["transform"]
            raw_series = df[feat]
            direction_full = _apply_direction_transform(raw_series, transform)

            # NaN/0 = "不发信号" → 回退到 always-long (direction=+1)
            # 这样 NaN 时段的 rr_dir = rr，与基线相同，lift 贡献 = 0
            # 特征的实际贡献被覆盖率自然稀释，反映真实影响
            coverage_rate = float((direction_full != 0).mean())
            direction_eval = direction_full.copy()
            direction_eval[direction_eval == 0] = 1.0  # 无信号 → always-long

            lifts = []
            window_details = []

            for w_start, w_end in windows:
                mask = (times >= w_start) & (times < w_end)
                w_rr = rr[mask.values]
                w_dir = direction_eval[mask.values]

                lift = _compute_window_lift(w_rr, w_dir)
                if lift is None:
                    continue

                lifts.append(lift)
                window_details.append(
                    {
                        "period": f"{w_start.strftime('%Y-%m')}→{w_end.strftime('%Y-%m')}",
                        "lift": round(lift, 4),
                    }
                )

            if len(lifts) < 3:
                continue

            arr = np.array(lifts)
            mean_lift = float(np.mean(arr))
            std_lift = float(np.std(arr))
            cv = abs(std_lift / mean_lift) if abs(mean_lift) > 1e-6 else float("inf")

            # 信号反转检测: 最近 3 个窗口中有负 Lift 吗?
            recent_negative = sum(1 for x in lifts[-3:] if x < 0)

            window_results[wm].append(
                {
                    "feature": feat,
                    "transform": transform,
                    "full_lift": r.get("median_rr_in_dir", r.get("median_lift", 0)),
                    "mean_lift": round(mean_lift, 4),
                    "std_lift": round(std_lift, 4),
                    "cv": round(cv, 2),
                    "latest_lift": round(lifts[-1], 4),
                    "n_windows": len(lifts),
                    "recent_negative": recent_negative,
                    "coverage": round(coverage_rate * 100, 0),
                    "windows": window_details,
                }
            )

    # 找最优窗口
    best_window = None
    best_avg_cv = float("inf")
    window_summary: Dict[int, Dict] = {}

    for wm, results in window_results.items():
        if not results:
            continue
        cvs = [r["cv"] for r in results if r["cv"] < float("inf")]
        if not cvs:
            continue
        avg_cv = float(np.mean(cvs))
        window_summary[wm] = {"avg_cv": round(avg_cv, 2), "n_features": len(results)}
        if avg_cv < best_avg_cv:
            best_avg_cv = avg_cv
            best_window = wm

    # 输出窗口对比
    print(f"\n   窗口对比:")
    for wm in sorted(window_summary.keys()):
        ws = window_summary[wm]
        marker = " ← 最优" if wm == best_window else ""
        print(
            f"     {wm}m: avg CV={ws['avg_cv']:.2f}, {ws['n_features']} 个特征{marker}"
        )

    if best_window is None:
        print("   ❌ 无有效窗口")
        return {}

    print(f"\n   ✅ 最优窗口: {best_window} 个月 (avg CV={best_avg_cv:.2f})")

    # 详细表格 (按 CV 排序)
    best = window_results[best_window]
    best.sort(key=lambda x: x["cv"])

    print(f"\n{'─' * 120}")
    print(
        f"{'  特征':<34s} {'Transform':<13s} {'覆盖率':>6s} {'全周期Lift':>10s} "
        f"{'最近窗口':>10s} {'CV':>8s} {'近3反转':>8s} {'判定':>8s}"
    )
    print(f"{'─' * 120}")

    for r in best:
        if r["cv"] < 0.3:
            verdict = "✅ 稳定"
        elif r["cv"] < 0.5:
            verdict = "⚠️  一般"
        else:
            verdict = "❌ 不稳"

        rev_str = f"{r['recent_negative']}/3"
        if r["recent_negative"] >= 2:
            verdict = "🚫 衰减"  # 最近 3 窗口中 >=2 个负 Lift → 信号衰减

        cov_str = f"{r['coverage']:.0f}%"
        print(
            f"  {r['feature']:<32s} {r['transform']:<13s} "
            f"{cov_str:>6s} {r['full_lift']:>+9.4f} {r['latest_lift']:>+9.4f} "
            f"{r['cv']:>7.2f} {rev_str:>8s} {verdict}"
        )

    # Rolling 曲线: Top-5 by CV + Lift 第 1 名 (确保选中特征总被画出)
    chart_feats = list(best[:5])
    # Lift 排名第 1 的特征 (top_results[0]) 一定画出, 即使 CV 不在前 5
    if top_results:
        lift_top1_name = top_results[0]["feature"]
        if not any(r["feature"] == lift_top1_name for r in chart_feats):
            lift_top1 = next((r for r in best if r["feature"] == lift_top1_name), None)
            if lift_top1:
                chart_feats.append(lift_top1)
    print(
        f"\n📈 Rolling Median Lift 曲线 ({best_window}m 窗口, {len(chart_feats)} 个特征):"
    )
    for r in chart_feats:
        print(f"\n  {r['feature']} ({r['transform']}) [CV={r['cv']:.2f}]:")
        # 自适应缩放: 每个特征的最大 |lift| 映射到 30 格
        max_abs = max((abs(w["lift"]) for w in r["windows"]), default=1.0) or 0.001
        for w in r["windows"]:
            lift = w["lift"]
            bar_len = int(abs(lift) / max_abs * 30)
            bar = "█" * min(bar_len, 30)
            sign_char = "+" if lift >= 0 else "-"
            print(f"    {w['period']}: {lift:>+.4f} {sign_char}{bar}")

    # 综合推荐
    stable = [
        r
        for r in best
        if r["cv"] < 0.5 and r["recent_negative"] < 2 and r["full_lift"] > 0
    ]
    if stable:
        rec = stable[0]
        print(
            f"\n  🏆 综合推荐 (Lift>0 + 稳定 + 无衰减): {rec['feature']} ({rec['transform']})"
        )
        print(
            f"     全周期 Lift={rec['full_lift']:+.4f}, "
            f"最近={rec['latest_lift']:+.4f}, CV={rec['cv']:.2f}"
        )
    else:
        positive = [r for r in best if r["full_lift"] > 0]
        if positive:
            rec = positive[0]
            print(
                f"\n  ⚠️  无稳定+正Lift特征, 最佳: {rec['feature']} (CV={rec['cv']:.2f})"
            )
        else:
            print(f"\n  ❌ 无正 Lift 特征")
    print(f"{'=' * 110}")

    # Return temporal data for writeback
    temporal_map: Dict[str, Dict] = {}
    for r in best:
        temporal_map[r["feature"]] = {
            "cv": r["cv"],
            "recent_decay": f"{r['recent_negative']}/3",
            "latest_lift": r["latest_lift"],
            "mean_lift": r["mean_lift"],
        }
    return temporal_map


# ── 自动回写 last_evaluation ─────────────────────────────────────


def _write_direction_evaluation(
    strategy: str,
    results: list,
    temporal_data: Optional[Dict[str, Dict]] = None,
    data_source: str = "",
    n_rows: int = 0,
    baseline_median_rr: float = 0.0,
    baseline_bad_rate: float = 0.0,
    *,
    workspace_basename: str = DEFAULT_DIRECTION_WORKSPACE,
) -> None:
    """回写 --compare-features 结果到方向工作区 YAML 的 last_evaluation。

    保留 candidates 段和注释，仅替换 last_evaluation 段。
    """
    path = _direction_config_write_path(strategy, workspace_basename)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            'description: ""\n' "candidates: []\n\n",
            encoding="utf-8",
        )
        print(f"\nℹ️  已创建方向工作区: {path}")

    # ── Build last_evaluation YAML text ──
    lines: list[str] = []
    lines.append("last_evaluation:")
    lines.append(f"  # ── 自动生成 by --compare-features ({date.today()}) ──")
    lines.append(f'  timestamp: "{date.today()}"')
    lines.append(f'  data_source: "{data_source}"')
    lines.append(f"  n_rows: {n_rows}")
    lines.append(f"  baseline: random_direction  # 随机方向 (期望 median ≈ 0)")
    lines.append(f"  always_long_median_rr: {baseline_median_rr:+.4f}  # 参考")
    lines.append(f"  always_long_bad_rate: {baseline_bad_rate:.4f}  # 参考")
    lines.append("")

    positive = [r for r in results if r.get("median_rr_in_dir", 0) > 0]
    negative = [r for r in results if r.get("median_rr_in_dir", 0) <= 0]

    lines.append(f"  # ── Med.RR > 0 (优于随机, {len(positive)} 个) ──")
    lines.append("  positive_lift:")
    if positive:
        for r in positive:
            lines.append(f"    - feature: {r['feature']}")
            lines.append(f"      transform: {r['transform']}")
            lines.append(f"      median_rr_in_dir: {r.get('median_rr_in_dir', 0):+.4f}")
            lines.append(f"      p_random: {r.get('p_random', 1):.4f}")
            lines.append(f"      lift_vs_long: {r.get('lift_vs_long', 0):+.4f}")
            lines.append(
                f"      bad_rate_reduction_pct: {r['bad_rate_reduction']:+.2f}"
            )
            lines.append(f"      short_pct: {r['short_pct']}")
            if temporal_data and r["feature"] in temporal_data:
                td = temporal_data[r["feature"]]
                lines.append(f"      temporal_cv: {td['cv']}")
                lines.append(f"      recent_decay: \"{td['recent_decay']}\"")
    else:
        lines.append("    []")
    lines.append("")

    lines.append(f"  # ── Med.RR <= 0 (不优于随机, 前 5) ──")
    lines.append("  fallback:")
    top_neg = negative[:5]
    if top_neg:
        for r in top_neg:
            lines.append(f"    - feature: {r['feature']}")
            lines.append(f"      transform: {r['transform']}")
            lines.append(f"      median_rr_in_dir: {r.get('median_rr_in_dir', 0):+.4f}")
            lines.append(f"      p_random: {r.get('p_random', 1):.4f}")
            lines.append(f"      lift_vs_long: {r.get('lift_vs_long', 0):+.4f}")
            lines.append(
                f"      bad_rate_reduction_pct: {r['bad_rate_reduction']:+.2f}"
            )
            lines.append(f"      short_pct: {r['short_pct']}")
            if temporal_data and r["feature"] in temporal_data:
                td = temporal_data[r["feature"]]
                lines.append(f"      temporal_cv: {td['cv']}")
                lines.append(f"      recent_decay: \"{td['recent_decay']}\"")
    else:
        lines.append("    []")
    lines.append("")

    eval_text = "\n".join(lines) + "\n"

    # ── Read & replace ──
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    marker = "\nlast_evaluation:"
    idx = content.find(marker)
    if idx >= 0:
        new_content = content[: idx + 1] + eval_text
    else:
        new_content = content.rstrip() + "\n\n" + eval_text

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"\n💾 已回写 last_evaluation → {path}")
    print(f"   优于随机: {len(positive)}, 兜底: {len(top_neg)}")


# ── 自动生成 direction_rules ─────────────────────────────────

# 选择门槛
DIR_RULE_MIN_LIFT = 0.0  # median_rr_in_dir 必须 > 0 (优于随机方向)
DIR_RULE_MIN_SHORT_PCT = 20.0  # short_pct 必须 >= 20% (不能是假方向)
DIR_RULE_MAX_SHORT_PCT = 80.0  # short_pct 必须 <= 80% (不能是常数偏差)
DIR_RULE_MAX_CV = 1.5  # temporal_cv < 1.5 (时间稳定性)
DIR_RULE_MAX_DECAY = "3/3"  # recent_decay 不能是 3/3 (完全衰减)
DIR_RULE_MAX_RULES = 3  # 最多 3 条级联规则


def _auto_generate_direction_rules(
    strategy: str,
    results: list,
    temporal_data: Optional[Dict[str, Dict]] = None,
) -> list:
    """从 compare-features 结果自动选择 top 候选生成 direction_rules.

    选择标准 (全部必须满足):
      1. median_rr_in_dir > 0 — 优于随机方向基线
      2. p_random < 0.05       — 置换检验统计显著
      3. 20% <= short_pct <= 80% — 不能是假方向或常数偏差
      4. temporal_cv < 1.5     — 时间稳定 (如有 temporal 数据)
      5. recent_decay != 3/3   — 信号未完全衰减 (如有 temporal 数据)
      6. 最多 3 条, 按 median_rr_in_dir 降序

    Returns:
        list of direction rule dicts (FER 格式), 空则无合格候选
    """
    qualified = []
    for r in results:
        # 必须优于随机方向 (median > 0)
        if r.get("median_rr_in_dir", 0) <= 0:
            continue
        # 置换检验必须显著
        if r.get("p_random", 1.0) >= 0.05:
            continue
        # short_pct 不能太低 (几乎全 long)
        if r.get("short_pct", 0) < DIR_RULE_MIN_SHORT_PCT:
            continue
        # short_pct 不能太高 (常数做空偏差, 如 center_sign 加在强度分数上)
        if r.get("short_pct", 100) > DIR_RULE_MAX_SHORT_PCT:
            continue
        # temporal 质量门槛 (如果有 temporal 数据)
        if temporal_data and r["feature"] in temporal_data:
            td = temporal_data[r["feature"]]
            cv = td.get("cv", 0)
            decay = td.get("recent_decay", "0/3")
            if cv >= DIR_RULE_MAX_CV:
                continue
            if decay == DIR_RULE_MAX_DECAY:
                continue
        qualified.append(r)

    # 按 median_rr_in_dir 降序, 取前 N
    qualified.sort(key=lambda x: x.get("median_rr_in_dir", 0), reverse=True)
    selected = qualified[:DIR_RULE_MAX_RULES]

    if not selected:
        return []

    # 生成 direction_rules 格式 (FER 兼容)
    rules = []
    for i, r in enumerate(selected, 1):
        feat = r["feature"]
        transform = r.get("transform", "sign")
        med_rr = r.get("median_rr_in_dir", 0)
        p_val = r.get("p_random", 1)
        short_pct = r.get("short_pct", 0)
        cv_info = ""
        if temporal_data and feat in temporal_data:
            cv_info = f", CV={temporal_data[feat]['cv']:.2f}"
        rules.append(
            {
                "method": "feature_sign",
                "feature": feat,
                "transform": transform,
                "description": f"规则{i}: {feat} (med_rr={med_rr:+.4f}, p={p_val:.3f}, short={short_pct:.0f}%{cv_info})",
            }
        )

    return rules


def _direction_rule_yaml_lines(r: dict) -> list:
    """单行 direction 规则 → YAML 文本行 (含可选 locked / lock_reason / id)."""
    _band = parse_single_position_band_rule(r)
    if _band is not None:
        fcol, inn, out = _band
        lines = [
            f"  - method: single_position_band",
        ]
        if r.get("id"):
            lines.append(f"    id: {r['id']}")
        lines.append(f"    feature: {fcol}")
        lines.append(f"    inner_abs: {float(inn)}")
        lines.append(f"    outer_abs: {float(out)}")
        lines.append(f'    description: "{r.get("description", "")}"')
        if r.get("locked"):
            lines.append("    locked: true")
        lr = r.get("lock_reason")
        if lr:
            lines.append(f'    lock_reason: "{lr}"')
        if "enabled" in r:
            lines.append(f"    enabled: {str(r['enabled']).lower()}")
        return lines
    if parse_dual_rule(r) is not None:
        feats = r.get("features") or []
        lines = [
            f"  - method: {r.get('method', 'dual_position_agree_deadband')}",
        ]
        if r.get("id"):
            lines.append(f"    id: {r['id']}")
        lines.append("    features:")
        for f in feats:
            lines.append(f"      - {f}")
        lines.append(f"    epsilon: {float(r.get('epsilon', 0.0))}")
        lines.append(f'    description: "{r.get("description", "")}"')
        if r.get("locked"):
            lines.append("    locked: true")
        lr = r.get("lock_reason")
        if lr:
            lines.append(f'    lock_reason: "{lr}"')
        if "enabled" in r:
            lines.append(f"    enabled: {str(r['enabled']).lower()}")
        return lines

    m = r.get("method", "feature_sign")
    lines = [
        f"  - method: {m}",
        f"    feature: {r.get('feature', '')}",
        f"    transform: {r.get('transform', 'raw')}",
        f'    description: "{r.get("description", "")}"',
    ]
    if r.get("locked"):
        lines.append("    locked: true")
    lr = r.get("lock_reason")
    if lr:
        lines.append(f'    lock_reason: "{lr}"')
    if "enabled" in r:
        lines.append(f"    enabled: {str(r['enabled']).lower()}")
    return lines


def _write_direction_rules(
    strategy: str,
    rules: list,
    *,
    workspace_basename: str = DEFAULT_DIRECTION_WORKSPACE,
) -> bool:
    """将自动生成的 direction_rules 写入方向工作区 YAML.

    写入位置: 在 last_evaluation: 之前、candidates: 之后.
    保留已有 ``locked: true`` 的规则在前排，避免自动生成整段覆盖语义锚点；
    自动规则按 (feature, transform) 与 locked 去重后追加在后。
    """
    path = _direction_config_write_path(strategy, workspace_basename)
    if not path.exists():
        print(f"\n⚠️  {path} 不存在, 无法写入 direction_rules")
        return False

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    full_data = yaml.safe_load(content) or {}
    old_rules = full_data.get("direction_rules") or []
    if not isinstance(old_rules, list):
        old_rules = []

    locked_first = [
        copy.deepcopy(r) for r in old_rules if isinstance(r, dict) and r.get("locked")
    ]
    locked_keys = {
        k for k in (direction_rule_ft_key(r) for r in locked_first) if k != (None, None)
    }

    tail = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        k = direction_rule_ft_key(r)
        if k in locked_keys:
            continue
        tail.append(r)

    merged = locked_first + tail

    # 生成 direction_rules YAML 文本
    lines = []
    lines.append(f"# ── direction_rules (自动生成 by --promote {date.today()}) ──")
    lines.append(
        f"# 选择标准: med_rr>0 + p<0.05 + short%>=20% + temporal CV<1.5 + 未衰减"
    )
    lines.append(f"# 级联规则: 规则1覆盖的行不再用规则2, 依此类推")
    if locked_first:
        lines.append(
            "# locked: true 的规则已保留在列表前部 (与 prefilter locked 语义对齐)"
        )
    lines.append("direction_rules:")
    for r in merged:
        lines.extend(_direction_rule_yaml_lines(r))
    lines.append("")
    rules_text = "\n".join(lines) + "\n"

    # 写入位置: 替换现有 direction_rules 或插入在 last_evaluation 之前
    marker_existing = "\ndirection_rules:"
    marker_eval = "\nlast_evaluation:"
    idx_existing = content.find(marker_existing)
    idx_eval = content.find(marker_eval)

    if idx_existing >= 0:
        # 替换现有 direction_rules 段 (到下一个顶层 key 为止)
        after = content[idx_existing + 1 :]
        # 找下一个非缩进的 key (或 last_evaluation / candidates)
        end_idx = None
        for candidate_marker in [
            "\nlast_evaluation:",
            "\ncandidates:",
            "\ndescription:",
        ]:
            pos = after.find(candidate_marker)
            if pos >= 0 and (end_idx is None or pos < end_idx):
                end_idx = pos
        if end_idx is not None:
            new_content = (
                content[: idx_existing + 1] + rules_text + after[end_idx + 1 :]
            )
        else:
            new_content = content[: idx_existing + 1] + rules_text
    elif idx_eval >= 0:
        # 插入在 last_evaluation 之前
        new_content = content[: idx_eval + 1] + rules_text + content[idx_eval + 1 :]
    else:
        # 追加到末尾
        new_content = content.rstrip() + "\n\n" + rules_text

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return True


def _promote_direction_archetypes(
    strategy: str, workspace_basename: str = DEFAULT_DIRECTION_WORKSPACE
) -> bool:
    """合并写入 archetypes/direction.yaml: locked 保留自当前 archetypes, 非 locked 来自工作区。"""
    src = _direction_workspace_path(strategy, workspace_basename)
    if not src.exists():
        print(f"\n⚠️  {strategy}: 方向工作区不存在 ({src}), 跳过 promote")
        return False
    dst = STRATEGIES_ROOT / strategy / "archetypes" / "direction.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "direction_rules" not in data:
        print(
            f"\n⚠️  {src}: 工作区未包含 direction_rules 键, 跳过 promote (不写入 {dst})"
        )
        return False

    ws_rules = data["direction_rules"]
    if ws_rules is None:
        print(f"\n⚠️  {src}: direction_rules 为 null, 跳过写入 {dst}")
        return False
    if not isinstance(ws_rules, list):
        print(f"\n⚠️  {src}: direction_rules 非列表, 跳过写入 {dst}")
        return False

    arch_meta: Dict[str, Any] = {}
    arch_rules: list = []
    if dst.exists():
        arch_meta = yaml.safe_load(dst.read_text(encoding="utf-8")) or {}
        ar = arch_meta.get("direction_rules")
        if isinstance(ar, list):
            arch_rules = ar

    merged = merge_direction_rules_for_promote(arch_rules, ws_rules)

    out: Dict[str, Any] = {}
    desc = arch_meta.get("description") or data.get("description")
    if desc:
        out["description"] = desc
    for key in ("fixed_direction", "direction_filter", "version", "validation"):
        if key in arch_meta:
            out[key] = arch_meta[key]
        elif key in data:
            out[key] = data[key]
    out["direction_rules"] = merged

    hdr = (
        "# 生产方向规则 (由 scripts/direction_strict_validation.py --promote 合并写入)\n"
        "# locked 来自当前 archetypes；非 locked 来自方向工作区的 direction_rules\n"
        f"# 候选与 last_evaluation 见同目录 {workspace_basename}\n\n"
    )
    dst.write_text(
        hdr + yaml.safe_dump(out, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\n📦 Promoted (merged) direction_rules → {dst}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Direction.yaml 严格验证 — 覆盖率 + 方向质量"
    )
    parser.add_argument(
        "--logs",
        default=None,
        help="单个 predictions.parquet 路径 (指定后只验证该文件)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="策略名称 (与 --logs 配合使用)",
    )
    parser.add_argument(
        "--compare-features",
        action="store_true",
        help=(
            "对比模式: 读方向工作区 (默认 features_direction.yaml) 的 candidates, "
            "按 Med.RR 排名 (vs 随机方向)"
        ),
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="发现模式: 扫描全部数值特征 (需配合 --compare-features)",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="时间稳定性分析: 对 Top 20 特征做 rolling Median Lift + CV (需配合 --compare-features)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "验证通过后: 合并写入 archetypes/direction.yaml — 保留 locked，"
            "非 locked 来自工作区 direction_rules（无该键则跳过）"
        ),
    )
    parser.add_argument(
        "--direction-workspace",
        default=DEFAULT_DIRECTION_WORKSPACE,
        help=(
            "方向工作区 YAML 文件名 (默认 features_direction.yaml), "
            "存放 candidates / last_evaluation; 生产规则在 promote 时写入 archetypes/"
        ),
    )
    parser.add_argument(
        "--strategies-root",
        default=None,
        help="策略配置根目录 (默认: config/strategies, 实验隔离时传入实验目录)",
    )
    args = parser.parse_args()

    # 支持实验目录隔离: 覆盖 STRATEGIES_ROOT
    if args.strategies_root:
        global STRATEGIES_ROOT
        sr = Path(args.strategies_root)
        STRATEGIES_ROOT = sr if sr.is_absolute() else (PROJECT_ROOT / sr)

    print("=" * 70)
    print("Direction.yaml 验证 (Phase 1: 覆盖率 + Phase 2: 方向质量)")
    print("=" * 70)

    # 确定要验证的策略列表
    if args.logs and args.strategy:
        targets = {args.strategy: args.logs}
    elif args.logs or args.strategy:
        print("❌ --logs 和 --strategy 必须同时指定")
        return 1
    else:
        targets = PREDICTIONS

    all_pass = True

    for arch_name, pred_path in targets.items():
        full_path = (
            PROJECT_ROOT / pred_path
            if not Path(pred_path).is_absolute()
            else Path(pred_path)
        )
        if not full_path.exists():
            print(f"\n❌ {arch_name}: {full_path} 不存在")
            all_pass = False
            continue

        df = pd.read_parquet(full_path)
        if "_symbol" in df.columns and "symbol" not in df.columns:
            df["symbol"] = df["_symbol"]

        # 检查 direction_rules 是否存在
        _dir_cfg = load_direction_config(arch_name)
        _has_rules = bool(_dir_cfg.get("direction_rules", []))

        if _has_rules:
            # Phase 1: 覆盖率
            r = validate_archetype(arch_name, df)
            p1_pass = _print_coverage(arch_name, r)

            # Phase 2: 方向质量
            direction = compute_direction_series(arch_name, df)
            q = validate_direction_quality(arch_name, df, direction)
            p2_pass = _print_quality(arch_name, q)

            if not (p1_pass and p2_pass):
                all_pass = False
        else:
            print(
                f"\nℹ️  {arch_name}: direction.yaml 无 direction_rules, 跳过 Phase 1/2"
            )
            if not getattr(args, "compare_features", False):
                print(f"   ⚠️  无规则且未指定 --compare-features, 无法验证")
                all_pass = False
                continue

        # Compare-features 模式: 对比方向候选特征
        if getattr(args, "compare_features", False):
            rr_col = _find_rr_column(df)
            if rr_col is None:
                print("\n⚠️  --compare-features 需要 forward_rr 列，当前数据中不存在")
            else:
                # 确定候选来源
                cfg_candidates = None
                if not getattr(args, "all_features", False):
                    cfg_candidates = _load_direction_candidates(
                        arch_name, args.direction_workspace
                    )
                    if cfg_candidates:
                        feats_str = ", ".join(f[0] for f in cfg_candidates)
                        print(
                            f"\n📖 读取方向候选 {args.direction_workspace}: {len(cfg_candidates)} 个"
                        )
                        print(f"   [{feats_str}]")
                    else:
                        print(
                            f"\n⚠️  {arch_name}: {args.direction_workspace} 无 candidates, 回退到全扫描"
                        )
                else:
                    print(f"\n🔍 --all-features: 扫描全部数值特征")

                results = compare_direction_features(
                    df, rr_col, config_candidates=cfg_candidates
                )
                _print_compare_table(results, rr_col)

                # Temporal 稳定性分析
                temporal_data = None
                if getattr(args, "temporal", False) and results:
                    top_n = results[:TEMPORAL_TOP_N]
                    temporal_data = temporal_direction_stability(df, rr_col, top_n)

                # 回写 last_evaluation
                rr_baseline = float(df[rr_col].median())
                bad_baseline = float((df[rr_col] < FAILURE_RR_THRESHOLD).mean())
                _write_direction_evaluation(
                    strategy=arch_name,
                    results=results,
                    temporal_data=temporal_data,
                    data_source=str(full_path.name),
                    n_rows=len(df),
                    baseline_median_rr=rr_baseline,
                    baseline_bad_rate=bad_baseline,
                    workspace_basename=args.direction_workspace,
                )

                # 自动生成 direction_rules (如果当前没有)
                if not _has_rules and getattr(args, "promote", False):
                    auto_rules = _auto_generate_direction_rules(
                        arch_name, results, temporal_data
                    )
                    if auto_rules:
                        ok = _write_direction_rules(
                            arch_name,
                            auto_rules,
                            workspace_basename=args.direction_workspace,
                        )
                        if ok:
                            print(
                                f"\n✨ 自动生成 {len(auto_rules)} 条 direction_rules:"
                            )
                            for ar in auto_rules:
                                print(
                                    f"   → {ar['feature']} ({ar['transform']}) — {ar['description']}"
                                )
                        # 重新加载并验证
                        _dir_cfg2 = load_direction_config(arch_name)
                        if _dir_cfg2.get("direction_rules"):
                            print(f"\n🔄 重新验证自动生成的 direction_rules...")
                            r2 = validate_archetype(arch_name, df)
                            p1 = _print_coverage(arch_name, r2)
                            d2 = compute_direction_series(arch_name, df)
                            q2 = validate_direction_quality(arch_name, df, d2)
                            p2 = _print_quality(arch_name, q2)
                            if not (p1 and p2):
                                all_pass = False
                    else:
                        print(f"\n⚠️  无候选满足自动 direction_rules 标准")
                        print(
                            f"   (需要: med_rr>0 + p<0.05 + short%>=20% + CV<1.5 + 未衰减)"
                        )
                        # 无规则不阻止 promote, 只是无 direction 功能

    print(f"\n{'=' * 70}")
    if all_pass:
        print("✅ 全部通过")
        # --promote: 合并工作区非 locked + archetypes locked → archetypes/direction.yaml
        if getattr(args, "promote", False):
            for arch_name in targets:
                _promote_direction_archetypes(arch_name, args.direction_workspace)
    else:
        print("❌ 存在问题，请检查上方详情")
        if getattr(args, "promote", False):
            print("⚠️  --promote 跳过: 验证未通过")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
