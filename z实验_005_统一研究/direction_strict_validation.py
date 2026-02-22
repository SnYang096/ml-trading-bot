#!/usr/bin/env python3
"""
Direction.yaml 严格验证 — 覆盖率 + 方向质量

验证每个策略的 predictions 数据是否能通过 direction.yaml 规则确定方向，
以及方向模块是否优于 always-long 基线。

验证项:
  Phase 1 - 覆盖率验证:
    - 非零方向占比 (目标 100%)
    - 命中规则统计
    - 方向分布 (Long / Short / Zero)
  Phase 2 - 方向质量验证:
    - median(rr_in_direction) vs median(rr_always_long)
    - bad rate(按方向) vs bad rate(always-long)
    - 每个方向子集的样本量校验 (>= 1080)

用法:
    python z实验_005_统一研究/direction_strict_validation.py
    python z实验_005_统一研究/direction_strict_validation.py --logs path/to/predictions.parquet --strategy me
"""
from __future__ import annotations

import argparse
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

PREDICTIONS = {
    "bpc": "results/train_final_20260208_220616_return_tree/bpc/predictions.parquet",
    "me": "results/train_final_20260215_234211_return_tree/me/predictions_fixed.parquet",
    "fer": "results/train_final_20260216_184525_return_tree/fer/predictions_fixed.parquet",
}

STRATEGIES_ROOT = PROJECT_ROOT / "config" / "strategies"

# 方向质量验证的最小可信样本量
MIN_CREDIBLE_SAMPLES = 1080

# forward_rr 候选列名 (按优先级)
RR_COLUMN_CANDIDATES = ["forward_rr", "bpc_impulse_return_atr", "rr", "return_atr"]

# failure 阈值
FAILURE_RR_THRESHOLD = -0.8


def load_direction_config(strategy: str) -> dict:
    path = STRATEGIES_ROOT / strategy / "archetypes" / "direction.yaml"
    if not path.exists():
        raise FileNotFoundError(f"direction.yaml 不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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

        # 只对未赋值的行赋值
        direction[unassigned] = vals[unassigned]
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


def compute_direction_series(
    arch_name: str, df: pd.DataFrame
) -> pd.Series:
    """从 direction.yaml 规则计算 per-bar 方向 Series。

    复用 validate_archetype 的逻辑，但只返回方向 Series。
    """
    cfg = load_direction_config(arch_name)
    rules = cfg.get("direction_rules", [])
    if not rules:
        raise ValueError(f"{arch_name}: direction.yaml 无 direction_rules")

    direction = pd.Series(0.0, index=df.index)
    assigned = pd.Series(False, index=df.index)

    for rule in rules:
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

        direction[unassigned] = vals[unassigned]
        newly = unassigned & (direction != 0)
        assigned = assigned | newly
        if assigned.all():
            break

    return direction


def validate_direction_quality(
    arch_name: str,
    df: pd.DataFrame,
    direction: pd.Series,
) -> Optional[Dict[str, Any]]:
    """Phase 2: 方向质量验证 — 方向模块是否优于 always-long。

    核心公式:
        rr_in_direction = forward_rr_long * direction
        (direction=-1 时, rr = -forward_rr_long = forward_rr_short)

    通过标准:
        1. median(rr_in_direction) > median(forward_rr_long)  → 方向有正贡献
        2. bad_rate(按方向) < bad_rate(always-long)           → 失败率更低
        3. short 子集样本量 >= MIN_CREDIBLE_SAMPLES            → 统计可信
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
    rr_long = rr  # always-long 基线

    # 中位数 / 均值
    med_in_dir = float(rr_in_dir.median())
    med_long = float(rr_long.median())
    mean_in_dir = float(rr_in_dir.mean())
    mean_long = float(rr_long.mean())

    # Bad rate: failure_rr_extreme (rr < -0.8)
    bad_in_dir = float((rr_in_dir < FAILURE_RR_THRESHOLD).mean())
    bad_long = float((rr_long < FAILURE_RR_THRESHOLD).mean())

    # ── Long / Short 子集分析 ──────────────────────────────
    long_mask = dir_s > 0
    short_mask = dir_s < 0
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())

    long_med_rr = float(rr[long_mask].median()) if n_long > 0 else float("nan")
    short_med_rr = float((-rr[short_mask]).median()) if n_short > 0 else float("nan")
    # short_med_rr: -forward_rr_long = forward_rr_short，正值表示做空盈利

    long_bad = float((rr[long_mask] < FAILURE_RR_THRESHOLD).mean()) if n_long > 0 else float("nan")
    short_bad = float((-rr[short_mask] < FAILURE_RR_THRESHOLD).mean()) if n_short > 0 else float("nan")

    # ── 判定 ──────────────────────────────────────────────
    direction_lift_median = med_in_dir - med_long
    direction_lift_bad = bad_long - bad_in_dir  # 正 = 方向模块降低了 bad rate

    passes = []
    passes.append(("median_lift", direction_lift_median > 0, direction_lift_median))
    passes.append(("mean_positive", mean_in_dir > 0, mean_in_dir))
    passes.append(("bad_rate_reduction", direction_lift_bad > 0, direction_lift_bad))
    passes.append(("short_credible", n_short >= MIN_CREDIBLE_SAMPLES, n_short))
    passes.append(("short_ratio", n_short / (n_long + n_short) > 0.15 if (n_long + n_short) > 0 else False,
                   round(n_short / (n_long + n_short) * 100, 1) if (n_long + n_short) > 0 else 0))

    return {
        "status": "OK",
        "rr_col": rr_col,
        "n_valid": len(rr),
        # 全局对比
        "median_in_direction": round(med_in_dir, 4),
        "median_always_long": round(med_long, 4),
        "direction_lift_median": round(direction_lift_median, 4),
        "mean_in_direction": round(mean_in_dir, 4),
        "mean_always_long": round(mean_long, 4),
        "bad_rate_in_direction": round(bad_in_dir, 4),
        "bad_rate_always_long": round(bad_long, 4),
        "bad_rate_reduction": round(direction_lift_bad, 4),
        # 子集
        "n_long": n_long,
        "n_short": n_short,
        "short_pct": round(n_short / (n_long + n_short) * 100, 1) if (n_long + n_short) > 0 else 0,
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
    if (r['long'] + r['short']) > 0:
        short_pct = r['short'] / (r['long'] + r['short']) * 100
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
    print(f"  \n  Phase 2: 方向质量验证")

    if q is None:
        print(f"  ⬜ 无 forward_rr 列，跳过质量验证")
        return True  # 不阻断

    if q["status"] == "INSUFFICIENT_DATA":
        print(f"  ❌ 数据量不足: {q['n_valid']} < {q['min_required']}")
        return False

    print(f"  RR 列: {q['rr_col']}  |  有效行: {q['n_valid']}")
    print(f"")
    print(f"  ┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
    print(f"  │ 指标                │ 按方向交易   │ Always-Long  │  Lift        │")
    print(f"  ├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"  │ Median RR           │ {q['median_in_direction']:>+11.4f} │ {q['median_always_long']:>+11.4f} │ {q['direction_lift_median']:>+11.4f} │")
    print(f"  │ Mean RR             │ {q['mean_in_direction']:>+11.4f} │ {q['mean_always_long']:>+11.4f} │ {q['mean_in_direction'] - q['mean_always_long']:>+11.4f} │")
    print(f"  │ Bad Rate (<-0.8R)   │ {q['bad_rate_in_direction']:>10.1%}  │ {q['bad_rate_always_long']:>10.1%}  │ {-q['bad_rate_reduction']:>+10.1%}  │")
    print(f"  └─────────────────────┴──────────────┴──────────────┴──────────────┘")
    print(f"")
    print(f"  Long 子集:  n={q['n_long']:>6d}  median_rr={q['long_median_rr']:>+.4f}  bad_rate={q['long_bad_rate']:.1%}")
    print(f"  Short 子集: n={q['n_short']:>6d}  median_rr={q['short_median_rr']:>+.4f}  bad_rate={q['short_bad_rate']:.1%}")
    print(f"  Short 占比: {q['short_pct']:.1f}%")
    print(f"")

    all_ok = True
    for name, passed, value in q["checks"]:
        icon = "✅" if passed else "❌"
        if name == "median_lift":
            desc = f"Median Lift > 0: {value:+.4f}"
        elif name == "mean_positive":
            desc = f"Mean RR > 0: {value:+.4f}"
        elif name == "bad_rate_reduction":
            desc = f"Bad Rate 降低: {value:+.1%}"
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

def _load_direction_candidates(strategy: str) -> Optional[list]:
    """读取 config/strategies/{strategy}/direction.yaml 的 candidates 列表。

    返回 [(feature, transform_or_None), ...] 或 None (文件不存在/无 candidates)。
    """
    path = STRATEGIES_ROOT / strategy / "direction.yaml"
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
            rr_col, "forward_rr", "success_no_rr_extreme", "failure_rr_extreme",
            "target", "sample_weight", "timestamp", "datetime", "date",
            "symbol", "_symbol",
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

        med_dir = float(rr_dir_valid.median())
        mean_dir = float(rr_dir_valid.mean())
        bad_dir = float((rr_dir_valid < FAILURE_RR_THRESHOLD).mean())

        results.append({
            "feature": feat,
            "transform": transform,
            "n_long": n_long,
            "n_short": n_short,
            "short_pct": round(n_short / n_total * 100, 1),
            "median_lift": round(med_dir - rr_long_median, 4),
            "mean_rr": round(mean_dir, 4),
            "bad_rate_reduction": round((bad_long - bad_dir) * 100, 2),
        })

    # Sort by median_lift descending
    results.sort(key=lambda x: x["median_lift"], reverse=True)
    return results


def _print_compare_table(results: list, rr_col: str) -> None:
    """打印方向特征对比排名表。"""
    print(f"\n{'=' * 90}")
    print(f"Direction 特征对比排名 (RR列: {rr_col}, 按 Median Lift 排序)")
    print(f"{'=' * 90}")
    print(f"{'Rank':>4s}  {'特征':<32s} {'Transform':<13s} {'Short%':>7s} {'Med.Lift':>9s} {'MeanRR':>9s} {'BadR↓%':>7s}")
    print(f"{'-' * 90}")

    for i, r in enumerate(results, 1):
        marker = " 🏆" if i == 1 else ""
        print(
            f"{i:>4d}  {r['feature']:<32s} {r['transform']:<13s} "
            f"{r['short_pct']:>6.1f}% {r['median_lift']:>+8.4f} "
            f"{r['mean_rr']:>+8.4f} {r['bad_rate_reduction']:>+6.2f}%{marker}"
        )

    print(f"{'-' * 90}")
    print(f"  共 {len(results)} 个候选特征通过双向分布筛选 (Short >= {MIN_CREDIBLE_SAMPLES})")
    print(f"  Median Lift = median(rr*direction) - median(rr_always_long)")
    print(f"  BadR↓% = bad_rate_always_long - bad_rate_in_direction (正=更好)")
    if results:
        best = results[0]
        print(f"\n  🏆 推荐: {best['feature']} ({best['transform']})")
        print(f"     Median Lift={best['median_lift']:+.4f}, Short={best['short_pct']:.1f}%, BadRate降低={best['bad_rate_reduction']:+.2f}%")
    print(f"{'=' * 90}")


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
    rr_long_median: float,
    min_samples: int = TEMPORAL_MIN_SAMPLES,
) -> Optional[float]:
    """计算单个窗口的 Median Lift。"""
    valid = direction != 0
    rr_dir = (rr * direction)[valid]
    if len(rr_dir) < min_samples:
        return None
    return float(rr_dir.median()) - rr_long_median


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
    print(f"   时间范围: {t_min.strftime('%Y-%m')} → {t_max.strftime('%Y-%m')}, 共 {total_months} 个月")
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

                # always-long baseline per window
                rr_long_med_w = float(w_rr.median()) if len(w_rr) > 0 else 0.0
                lift = _compute_window_lift(w_rr, w_dir, rr_long_med_w)
                if lift is None:
                    continue

                lifts.append(lift)
                window_details.append({
                    "period": f"{w_start.strftime('%Y-%m')}→{w_end.strftime('%Y-%m')}",
                    "lift": round(lift, 4),
                })

            if len(lifts) < 3:
                continue

            arr = np.array(lifts)
            mean_lift = float(np.mean(arr))
            std_lift = float(np.std(arr))
            cv = abs(std_lift / mean_lift) if abs(mean_lift) > 1e-6 else float("inf")

            # 信号反转检测: 最近 3 个窗口中有负 Lift 吗?
            recent_negative = sum(1 for x in lifts[-3:] if x < 0)

            window_results[wm].append({
                "feature": feat,
                "transform": transform,
                "full_lift": r["median_lift"],
                "mean_lift": round(mean_lift, 4),
                "std_lift": round(std_lift, 4),
                "cv": round(cv, 2),
                "latest_lift": round(lifts[-1], 4),
                "n_windows": len(lifts),
                "recent_negative": recent_negative,
                "coverage": round(coverage_rate * 100, 0),
                "windows": window_details,
            })

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
        print(f"     {wm}m: avg CV={ws['avg_cv']:.2f}, {ws['n_features']} 个特征{marker}")

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
    print(f"\n📈 Rolling Median Lift 曲线 ({best_window}m 窗口, {len(chart_feats)} 个特征):")
    for r in chart_feats:
        print(
            f"\n  {r['feature']} ({r['transform']}) [CV={r['cv']:.2f}]:"
        )
        # 自适应缩放: 每个特征的最大 |lift| 映射到 30 格
        max_abs = max((abs(w["lift"]) for w in r["windows"]), default=1.0) or 0.001
        for w in r["windows"]:
            lift = w["lift"]
            bar_len = int(abs(lift) / max_abs * 30)
            bar = "█" * min(bar_len, 30)
            sign_char = "+" if lift >= 0 else "-"
            print(f"    {w['period']}: {lift:>+.4f} {sign_char}{bar}")

    # 综合推荐
    stable = [r for r in best if r["cv"] < 0.5 and r["recent_negative"] < 2 and r["full_lift"] > 0]
    if stable:
        rec = stable[0]
        print(f"\n  🏆 综合推荐 (Lift>0 + 稳定 + 无衰减): {rec['feature']} ({rec['transform']})")
        print(
            f"     全周期 Lift={rec['full_lift']:+.4f}, "
            f"最近={rec['latest_lift']:+.4f}, CV={rec['cv']:.2f}"
        )
    else:
        positive = [r for r in best if r["full_lift"] > 0]
        if positive:
            rec = positive[0]
            print(f"\n  ⚠️  无稳定+正Lift特征, 最佳: {rec['feature']} (CV={rec['cv']:.2f})")
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
) -> None:
    """回写 --compare-features 结果到 config/strategies/{strategy}/direction.yaml 的 last_evaluation。

    保留 candidates 段和注释，仅替换 last_evaluation 段。
    """
    path = STRATEGIES_ROOT / strategy / "direction.yaml"
    if not path.exists():
        print(f"\n⚠️  {path} 不存在, 跳过回写")
        return

    # ── Build last_evaluation YAML text ──
    lines: list[str] = []
    lines.append("last_evaluation:")
    lines.append(f"  # ── 自动生成 by --compare-features ({date.today()}) ──")
    lines.append(f'  timestamp: "{date.today()}"')
    lines.append(f'  data_source: "{data_source}"')
    lines.append(f"  n_rows: {n_rows}")
    lines.append(f"  baseline_median_rr: {baseline_median_rr:+.4f}")
    lines.append(f"  baseline_bad_rate: {baseline_bad_rate:.4f}")
    lines.append("")

    positive = [r for r in results if r["median_lift"] > 0]
    negative = [r for r in results if r["median_lift"] <= 0]

    lines.append(f"  # ── Median Lift > 0 ({len(positive)} 个) ──")
    lines.append("  positive_lift:")
    if positive:
        for r in positive:
            lines.append(f"    - feature: {r['feature']}")
            lines.append(f"      transform: {r['transform']}")
            lines.append(f"      median_lift: {r['median_lift']:+.4f}")
            lines.append(f"      bad_rate_reduction_pct: {r['bad_rate_reduction']:+.2f}")
            lines.append(f"      short_pct: {r['short_pct']}")
            if temporal_data and r["feature"] in temporal_data:
                td = temporal_data[r["feature"]]
                lines.append(f"      temporal_cv: {td['cv']}")
                lines.append(f"      recent_decay: \"{td['recent_decay']}\"")
    else:
        lines.append("    []")
    lines.append("")

    lines.append(f"  # ── Lift <= 0 (兜底候选, 前 5) ──")
    lines.append("  fallback:")
    top_neg = negative[:5]
    if top_neg:
        for r in top_neg:
            lines.append(f"    - feature: {r['feature']}")
            lines.append(f"      transform: {r['transform']}")
            lines.append(f"      median_lift: {r['median_lift']:+.4f}")
            lines.append(f"      bad_rate_reduction_pct: {r['bad_rate_reduction']:+.2f}")
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
        new_content = content[:idx + 1] + eval_text
    else:
        new_content = content.rstrip() + "\n\n" + eval_text

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"\n💾 已回写 last_evaluation → {path}")
    print(f"   正 Lift: {len(positive)}, 兜底: {len(top_neg)}")


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
        help="对比模式: 读 config/strategies/{strategy}/direction.yaml 候选, 按 Median Lift 排名",
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
        help="验证通过后自动复制 direction.yaml 到 archetypes/direction.yaml",
    )
    args = parser.parse_args()

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
        full_path = PROJECT_ROOT / pred_path if not Path(pred_path).is_absolute() else Path(pred_path)
        if not full_path.exists():
            print(f"\n❌ {arch_name}: {full_path} 不存在")
            all_pass = False
            continue

        df = pd.read_parquet(full_path)
        if "_symbol" in df.columns and "symbol" not in df.columns:
            df["symbol"] = df["_symbol"]

        # Phase 1: 覆盖率
        r = validate_archetype(arch_name, df)
        p1_pass = _print_coverage(arch_name, r)

        # Phase 2: 方向质量
        direction = compute_direction_series(arch_name, df)
        q = validate_direction_quality(arch_name, df, direction)
        p2_pass = _print_quality(arch_name, q)

        if not (p1_pass and p2_pass):
            all_pass = False

        # Compare-features 模式: 对比方向候选特征
        if getattr(args, "compare_features", False):
            rr_col = _find_rr_column(df)
            if rr_col is None:
                print("\n⚠️  --compare-features 需要 forward_rr 列，当前数据中不存在")
            else:
                # 确定候选来源
                cfg_candidates = None
                if not getattr(args, "all_features", False):
                    cfg_candidates = _load_direction_candidates(arch_name)
                    if cfg_candidates:
                        feats_str = ", ".join(f[0] for f in cfg_candidates)
                        print(f"\n📖 读取 config/strategies/{arch_name}/direction.yaml: {len(cfg_candidates)} 个候选")
                        print(f"   [{feats_str}]")
                    else:
                        print(f"\n⚠️  config/strategies/{arch_name}/direction.yaml 无 candidates, 回退到全扫描")
                else:
                    print(f"\n🔍 --all-features: 扫描全部数值特征")

                results = compare_direction_features(df, rr_col, config_candidates=cfg_candidates)
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
                )

    print(f"\n{'=' * 70}")
    if all_pass:
        print("✅ 全部通过")
        # --promote: 复制 direction.yaml 到 archetypes/
        if getattr(args, "promote", False):
            for arch_name in targets:
                src = STRATEGIES_ROOT / arch_name / "direction.yaml"
                dst = STRATEGIES_ROOT / arch_name / "archetypes" / "direction.yaml"
                if src.exists():
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"\U0001f4e6 Promoted direction.yaml → {dst}")
                else:
                    print(f"\u26a0\ufe0f  Cannot promote: {src} not found")
    else:
        print("❌ 存在问题，请检查上方详情")
        if getattr(args, "promote", False):
            print("⚠️  --promote 跳过: 验证未通过")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
