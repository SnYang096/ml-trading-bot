#!/usr/bin/env python3
"""
Direction.yaml 严格验证 — 纯 direction.yaml 逻辑，零兼容

验证每个策略的 predictions 数据是否能通过 direction.yaml 规则确定方向:
  - 覆盖率: 非零方向占比 (目标 100%)
  - 命中规则: 实际命中的 direction_rules 条目
  - 方向分布: Long / Short / Zero

用法:
    python z实验_005_统一研究/direction_strict_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

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


def main():
    print("=" * 80)
    print("Direction.yaml 严格验证 (零兼容/零兜底)")
    print("=" * 80)

    all_pass = True

    for arch_name, pred_path in PREDICTIONS.items():
        full_path = PROJECT_ROOT / pred_path
        if not full_path.exists():
            print(f"\n❌ {arch_name}: {full_path} 不存在")
            all_pass = False
            continue

        df = pd.read_parquet(full_path)
        if "_symbol" in df.columns and "symbol" not in df.columns:
            df["symbol"] = df["_symbol"]

        r = validate_archetype(arch_name, df)

        print(f"\n{'─' * 60}")
        print(f"  {arch_name.upper()} | causal_source={r['causal_source']}")
        print(f"  行数: {r['rows']}")
        print(f"  覆盖率: {r['coverage_pct']}%")
        print(f"  方向分布: Long={r['long']}  Short={r['short']}  Zero={r['zero']}")
        print(f"  规则命中:")
        for feat, info in r["rule_hits"].items():
            if info["status"] == "NOT_IN_DATA":
                print(f"    ⬜ {feat}: 列不存在 (跳过)")
            else:
                print(
                    f"    ✅ {feat} (transform={info['transform']}): "
                    f"命中 {info['count']} 行"
                )

        # 判定
        if r["coverage_pct"] < 100:
            print(f"  ⚠️  覆盖率不足 100% — {r['zero']} 行无方向")
            # 对于 ME 全做多的情况，zero=0 其实也算正常
            if r["zero"] > 0:
                all_pass = False
        else:
            print(f"  ✅ 覆盖率 100%")

    print(f"\n{'=' * 80}")
    if all_pass:
        print("✅ 全部通过: 纯 direction.yaml 逻辑可完全确定方向")
    else:
        print("❌ 存在问题: 部分策略覆盖率不足")
    print("=" * 80)


if __name__ == "__main__":
    main()
