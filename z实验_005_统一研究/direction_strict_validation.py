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

    print(f"\n{'=' * 70}")
    if all_pass:
        print("✅ 全部通过")
    else:
        print("❌ 存在问题，请检查上方详情")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
