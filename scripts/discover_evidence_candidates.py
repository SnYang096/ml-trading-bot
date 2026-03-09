#!/usr/bin/env python3
"""
Evidence Candidate Discovery — Spearman + Quintile Monotonicity

替代旧的 LightGBM regression → SHAP∩Gain 方法。
直接在 gate 放行子集上测试环境/微结构类特征对 trade quality 的分层作用。

方法论：
  1. 加载 logs_gated.parquet (gate 放行后的交易数据)
  2. 对 evidence 候选特征逐一计算:
     - Spearman 相关性 (与 forward_rr)
     - 分位数 5 组的 winrate / avgR / tail% 单调性
     - 首尾组 t-test
  3. 筛选条件: p < 0.01, 单调性 >= 0.6, 方向一致
  4. 取 top 4-5 个, 自动生成 evidence.yaml
  5. --promote 直接写入 config/strategies/{strat}/archetypes/evidence.yaml

用法:
    python scripts/discover_evidence_candidates.py \\
        --logs results/train_final_.../me/logs_gated.parquet \\
        --strategy me \\
        --strategies-root config/strategies \\
        --promote
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ================================================================
# Evidence 候选自动发现
# ================================================================
# 默认: 扫描 logs_gated.parquet 全部数值列
# 自动排除: gate/prefilter/direction 特征 + 元数据 + 前瞻标签
# evidence_candidates.yaml 可选: 提供额外 exclude 和 category_hints

N_BINS = 5
TAIL_R = 2.0
MAX_EVIDENCE_FEATURES = 3
MIN_SAMPLES = 50
CORR_DEDUP_THRESHOLD = 0.7  # 相关性去重阈值

# 元数据/前瞻列 — 永远不参与 evidence 筛选
_META_EXCLUDE = {
    "symbol",
    "_symbol",
    "timestamp",
    "date",
    "datetime",
    "time",
    "ts",
    "entry_direction",
    "bpc_breakout_direction",
    "gate_decision",
    "gate_ok",
    "gate_label",
    "gate_passed",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr",
    "_entry_quality",
    "exec_r",
    "target",
    "label",
    "signal",
    "success_no_rr_extreme",
    "rr_extreme",
    "forward_rr",
    "forward_return",
    "forward_r",
    "path_extreme",
    "path_extreme_r",
}


def _load_gate_prefilter_direction_features(strategy: str, strategies_root: str) -> set:
    """从 gate/prefilter/direction 配置提取已使用的特征名, 自动排除 (防 double counting)."""
    exclude: set = set()
    root = Path(strategies_root) / strategy

    # Prefilter rules
    for pf_path in [root / "archetypes" / "prefilter.yaml", root / "prefilter.yaml"]:
        if pf_path.exists():
            cfg = yaml.safe_load(pf_path.read_text(encoding="utf-8")) or {}
            for rule in cfg.get("rules", []):
                if "feature" in rule:
                    exclude.add(rule["feature"])
            break

    # Gate hard_gates + guardrails
    for gate_path in [root / "archetypes" / "gate.yaml", root / "gate_draft.yaml"]:
        if gate_path.exists():
            cfg = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}
            for g in cfg.get("hard_gates", []):
                for feat in g.get("when", {}):
                    exclude.add(feat)
            for g in cfg.get("guardrails", []):
                for feat in g.get("when", {}):
                    exclude.add(feat)
            break

    # Direction rules
    for dir_path in [root / "archetypes" / "direction.yaml", root / "direction.yaml"]:
        if dir_path.exists():
            cfg = yaml.safe_load(dir_path.read_text(encoding="utf-8")) or {}
            for rule in cfg.get("direction_rules", []):
                if "feature" in rule:
                    exclude.add(rule["feature"])
            break

    return exclude


def _load_evidence_config(
    strategy: str, strategies_root: str = "config/strategies"
) -> set:
    """加载 evidence_candidates.yaml (可选): 额外排除列表.

    Returns:
        extra_exclude_set
    """
    cfg_path = Path(strategies_root) / strategy / "evidence_candidates.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        extra_exclude = set(cfg.get("exclude", []))
        return extra_exclude
    return set()


def _discover_all_numeric_features(df: pd.DataFrame, exclude: set) -> List[str]:
    """扫描 DataFrame 全部数值列, 排除元数据/前瞻/指定列."""
    features = []
    for col in df.columns:
        if col in _META_EXCLUDE or col in exclude:
            continue
        if (
            col.startswith("gate_")
            or col.startswith("__")
            or col.startswith("forward_")
        ):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique() < 3:
            continue
        features.append(col)
    return sorted(features)


# ================================================================
# 筛选阈值
# ================================================================
P_THRESHOLD = 0.01  # Spearman p-value 门槛
MIN_MONOTONICITY = 0.6  # WR 或 tail 单调性 (Spearman of bin vs metric)


def detect_outcome(df: pd.DataFrame) -> Optional[str]:
    """自动检测 outcome 列."""
    for col in ["forward_rr", "rr", "bpc_impulse_return_atr", "realized_rr"]:
        if col in df.columns and df[col].notna().sum() > 10:
            return col
    return None


def _compute_monotonicity(bins: pd.Series, values: pd.Series) -> float:
    """计算 bin 序列 vs value 序列的 Spearman 单调性 (绝对值)."""
    if len(bins) < 3:
        return 0.0
    try:
        r, _ = stats.spearmanr(bins, values)
        return abs(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


def analyze_feature(
    df: pd.DataFrame,
    feat: str,
    outcome_col: str,
) -> Optional[Dict[str, Any]]:
    """对单个特征做 Spearman + 分位数分层分析.

    Returns:
        dict with sp_r, sp_p, wr_mono, tail_mono, direction, groups, etc.
        或 None (数据不足).
    """
    sub = df[[feat, outcome_col]].dropna()
    if len(sub) < MIN_SAMPLES:
        return None

    try:
        sub = sub.copy()
        sub["bin"] = pd.qcut(sub[feat], N_BINS, labels=False, duplicates="drop")
    except Exception:
        return None
    if sub["bin"].nunique() < 3:
        return None

    # Spearman correlation
    sp_r, sp_p = stats.spearmanr(sub[feat], sub[outcome_col])
    if not np.isfinite(sp_r):
        return None

    # Per-bin stats
    groups = []
    for b in sorted(sub["bin"].unique()):
        g = sub[sub["bin"] == b]
        rr = g[outcome_col]
        wr = float((rr > 0).mean())
        avg_r = float(rr.mean())
        tail_pct = float((rr >= TAIL_R).mean())
        # expectancy = avg_R per trade (captures both WR and magnitude)
        groups.append(
            {
                "bin": int(b),
                "n": len(g),
                "feat_mean": float(g[feat].mean()),
                "winrate": wr,
                "avg_R": avg_r,
                "tail_pct": tail_pct,
                "expectancy": avg_r,  # = mean(rr), same as avg_R but explicit
            }
        )
    gdf = pd.DataFrame(groups)

    # Monotonicity checks
    wr_mono = _compute_monotonicity(gdf["bin"], gdf["winrate"])
    tail_mono = _compute_monotonicity(gdf["bin"], gdf["tail_pct"])
    avgr_mono = _compute_monotonicity(gdf["bin"], gdf["avg_R"])
    exp_mono = _compute_monotonicity(gdf["bin"], gdf["expectancy"])

    # T-test between lowest and highest bin
    low_rr = sub[sub["bin"] == gdf["bin"].min()][outcome_col]
    hi_rr = sub[sub["bin"] == gdf["bin"].max()][outcome_col]
    t_stat, t_p = (
        stats.ttest_ind(hi_rr, low_rr, equal_var=False)
        if len(low_rr) > 5 and len(hi_rr) > 5
        else (np.nan, np.nan)
    )

    # Direction: positive = higher feature → better outcome
    direction = "positive" if sp_r > 0 else "negative"

    # WR direction consistency check
    wr_direction = gdf.iloc[-1]["winrate"] - gdf.iloc[0]["winrate"]
    direction_consistent = (sp_r > 0 and wr_direction > 0) or (
        sp_r < 0 and wr_direction < 0
    )

    return {
        "feature": feat,
        "n": len(sub),
        "sp_r": float(sp_r),
        "sp_p": float(sp_p),
        "wr_mono": float(wr_mono),
        "tail_mono": float(tail_mono),
        "avgr_mono": float(avgr_mono),
        "exp_mono": float(exp_mono),
        "direction": direction,
        "direction_consistent": direction_consistent,
        "t_p": float(t_p) if np.isfinite(t_p) else 1.0,
        "wr_lo": float(gdf.iloc[0]["winrate"]),
        "wr_hi": float(gdf.iloc[-1]["winrate"]),
        "tail_lo": float(gdf.iloc[0]["tail_pct"]),
        "tail_hi": float(gdf.iloc[-1]["tail_pct"]),
        "exp_lo": float(gdf.iloc[0]["expectancy"]),
        "exp_hi": float(gdf.iloc[-1]["expectancy"]),
        "groups": groups,
        # For quantile_mapping generation
        "quantile_bins": _compute_quantile_bins(sub[feat]),
    }


def _compute_quantile_bins(series: pd.Series, bins: List[float] = None) -> List[float]:
    """计算特征的分位数 bins (用于 evidence.yaml quantile_mapping)."""
    if bins is None:
        bins = [0.2, 0.4, 0.6, 0.8]
    result = []
    for q in bins:
        result.append(float(series.quantile(q)))
    return result


def select_candidates(
    results: Dict[str, Dict[str, Any]],
    exclude_set: set,
    df: pd.DataFrame,
    max_features: int = MAX_EVIDENCE_FEATURES,
    p_threshold: float = P_THRESHOLD,
    corr_threshold: float = CORR_DEDUP_THRESHOLD,
) -> List[Dict[str, Any]]:
    """从分析结果中筛选 evidence 候选.

    筛选条件:
      1. 排除 exclude_set 中的特征
      2. Spearman p < p_threshold
      3. WR 或 tail 或 expectancy 单调性 >= MIN_MONOTONICITY
      4. 方向一致 (sp_r 符号与 wr 趋势一致)
      5. 相关性去重: |corr| > corr_threshold 的特征只保留 |sp_r| 最大的
    """
    candidates = []
    for feat, r in results.items():
        if r is None:
            continue
        if feat in exclude_set:
            continue
        if r["sp_p"] >= p_threshold:
            continue
        # 放宽: WR / tail / expectancy 任一单调性够即可
        mono_ok = (
            r["wr_mono"] >= MIN_MONOTONICITY
            or r["tail_mono"] >= MIN_MONOTONICITY
            or r.get("exp_mono", 0) >= MIN_MONOTONICITY
        )
        if not mono_ok:
            continue
        if not r["direction_consistent"]:
            continue
        candidates.append(r)

    # Sort by |sp_r| descending
    candidates.sort(key=lambda x: abs(x["sp_r"]), reverse=True)

    # Correlation-based dedup: 高相关特征只保留最强的
    selected = []
    selected_vectors: Dict[str, np.ndarray] = {}
    for c in candidates:
        if len(selected) >= max_features:
            break
        feat = c["feature"]
        if feat not in df.columns:
            continue
        col_vals = df[feat].fillna(0).values.astype(float)
        # 检查与已选特征的相关性
        correlated = False
        for sel_feat, sel_vals in selected_vectors.items():
            try:
                r_val = np.corrcoef(col_vals, sel_vals)[0, 1]
                if np.isfinite(r_val) and abs(r_val) > corr_threshold:
                    correlated = True
                    break
            except Exception:
                continue
        if correlated:
            continue
        selected.append(c)
        selected_vectors[feat] = col_vals

    return selected


def generate_evidence_yaml(
    candidates: List[Dict[str, Any]],
    strategy: str,
) -> Dict[str, Any]:
    """生成 evidence.yaml 格式的配置."""
    evidence_list = []
    for rank, c in enumerate(candidates, start=1):
        feat = c["feature"]
        direction = c["direction"]

        # quantile_mapping: 根据方向决定标签顺序
        if direction == "positive":
            labels = ["suppress", "downweight", "neutral", "favor", "amplify"]
        else:
            labels = ["amplify", "favor", "neutral", "downweight", "suppress"]

        evidence_list.append(
            {
                "id": f"evidence_{feat}",
                "feature": feat,
                "rank": rank,
                "direction": direction,
                "usage_hint": (
                    f"sp_r={c['sp_r']:+.3f} " f"WR:{c['wr_lo']:.1%}→{c['wr_hi']:.1%}"
                ),
                "affects": ["position_size", "tp_range"],
                "quantile_mapping": {
                    "bins": c["quantile_bins"],
                    "labels": labels,
                },
                "split_count": N_BINS,
            }
        )

    return {
        "schema": {
            "label_semantics": {
                "suppress": "强烈不利 - 极度限制仓位/信心",
                "downweight": "不利 - 降低信心/仓位",
                "neutral": "中性 - 标准执行",
                "favor": "有利 - 提高信心/仓位",
                "amplify": "强烈有利 - 最大化执行",
            }
        },
        "evidence": evidence_list,
        "min_score": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evidence Candidate Discovery (Spearman + Quintile Monotonicity)"
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="Path to logs_gated.parquet",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy name (bpc, me, fer, lv)",
    )
    parser.add_argument(
        "--strategies-root",
        default="config/strategies",
        help="Strategies config root directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path for analysis results (optional)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Write evidence.yaml to config/strategies/{strat}/archetypes/",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=MAX_EVIDENCE_FEATURES,
        help=f"Max evidence features to select (default: {MAX_EVIDENCE_FEATURES})",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=P_THRESHOLD,
        help=f"Spearman p-value threshold (default: {P_THRESHOLD})",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Only use data before this date (IS cutoff, avoid OOS lookahead)",
    )
    args = parser.parse_args()

    # ── 0. Build exclude set ──
    # 自动排除: gate/prefilter/direction 特征 (防 double counting)
    auto_exclude = _load_gate_prefilter_direction_features(
        args.strategy, args.strategies_root
    )
    # 可选配置: 额外排除
    extra_exclude = _load_evidence_config(args.strategy, args.strategies_root)
    exclude_set = auto_exclude | extra_exclude
    print(f"🚫 Auto-exclude: {sorted(auto_exclude)} (gate/prefilter/direction)")
    if extra_exclude:
        print(f"   Extra exclude: {sorted(extra_exclude)} (from config)")

    # ── 1. Load data ──
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ File not found: {logs_path}")
        sys.exit(1)

    df = pd.read_parquet(logs_path)
    print(f"📊 Loaded {len(df)} rows from {logs_path}")

    # Apply cutoff date (IS only — avoid OOS lookahead)
    if args.cutoff_date:
        ts_col = "timestamp" if "timestamp" in df.columns else None
        if ts_col is None and df.index.name == "timestamp":
            df = df.reset_index()
            ts_col = "timestamp"
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col])
            n_before = len(df)
            df = df[df[ts_col] < args.cutoff_date]
            print(f"   IS cutoff {args.cutoff_date}: {n_before} → {len(df)} rows")

    # Apply gate filter
    mask = pd.Series(True, index=df.index)
    if "gate_decision" in df.columns:
        mask &= df["gate_decision"] == "allow"
    elif "gate_passed" in df.columns:
        mask &= df["gate_passed"] == True  # noqa: E712
    dff = df[mask]
    print(f"   Gate-passed: {len(dff)} rows ({len(dff)/max(1,len(df))*100:.1f}%)")

    if len(dff) < MIN_SAMPLES:
        print(f"⚠️  Only {len(dff)} gate-passed rows, using all {len(df)} rows")
        dff = df

    # Detect outcome
    oc = detect_outcome(dff)
    if not oc:
        print(f"❌ No outcome column found in {list(dff.columns)[:20]}...")
        sys.exit(1)
    print(f"   Outcome: {oc}")

    # ── 2. Auto-discover all numeric features ──
    all_features = _discover_all_numeric_features(dff, exclude_set)
    print(
        f"   Numeric features scanned: {len(all_features)} (after excluding {len(exclude_set)} features)"
    )

    results = {}
    for feat in all_features:
        r = analyze_feature(dff, feat, oc)
        if r is not None:
            results[feat] = r

    # ── 3. Select top candidates (correlation dedup) ──
    selected = select_candidates(
        results,
        exclude_set=exclude_set,
        df=dff,
        max_features=args.max_features,
        p_threshold=args.p_threshold,
    )

    # ── 4. Report ──
    print(f"\n{'='*80}")
    print(f"Evidence Discovery: {args.strategy.upper()}")
    print(f"{'='*80}")
    print(
        f"{'feature':<40} {'sp_r':>7} {'sp_p':>9} "
        f"{'wr_m':>5} {'tl_m':>5} {'ex_m':>5} {'dir':>4}"
    )
    print("-" * 80)

    for feat in sorted(results, key=lambda f: results[f]["sp_p"]):
        r = results[feat]
        sig = "★" if r["sp_p"] < args.p_threshold else " "
        sel = "→" if r in selected else " "
        excl = "✗" if feat in exclude_set else " "
        print(
            f"{sel}{sig}{excl}{feat:<37} "
            f"{r['sp_r']:+.3f} {r['sp_p']:.6f} "
            f"{r['wr_mono']:.2f} {r['tail_mono']:.2f} {r.get('exp_mono', 0):.2f} "
            f"{'↑' if r['direction'] == 'positive' else '↓'}"
        )

    print(f"\n✅ Selected {len(selected)}/{len(results)} candidates:")
    for i, c in enumerate(selected, 1):
        print(
            f"  {i}. {c['feature']} "
            f"sp_r={c['sp_r']:+.3f} "
            f"WR:{c['wr_lo']:.1%}→{c['wr_hi']:.1%} "
            f"tail:{c['tail_lo']:.1%}→{c['tail_hi']:.1%} "
            f"E[R]:{c.get('exp_lo', 0):.3f}→{c.get('exp_hi', 0):.3f}"
        )

    # ── 5. Generate evidence.yaml ──
    evidence_yaml = generate_evidence_yaml(selected, args.strategy)

    if args.promote:
        evidence_path = (
            Path(args.strategies_root) / args.strategy / "archetypes" / "evidence.yaml"
        )
        evidence_path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            f"# {args.strategy.upper()} Evidence (auto-discovered)\n"
            f"# 来源: {logs_path}\n"
            f"# 方法: Spearman + Quintile Monotonicity (p < {args.p_threshold})\n"
            f"# 候选: {len(selected)}/{len(results)} 通过筛选\n\n"
        )
        yaml_content = yaml.dump(
            evidence_yaml, allow_unicode=True, default_flow_style=False, sort_keys=False
        )
        evidence_path.write_text(header + yaml_content, encoding="utf-8")
        print(f"\n📝 Promoted to {evidence_path}")
    else:
        print("\n📋 Evidence YAML (use --promote to write):")
        print(yaml.dump(evidence_yaml, allow_unicode=True, default_flow_style=False))

    # ── 6. Save analysis results ──
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize: remove groups for compactness, convert numpy types
        serializable = {}
        for feat, r in results.items():
            sr = {}
            for k, v in r.items():
                if k == "groups":
                    continue
                if isinstance(v, (np.integer,)):
                    sr[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    sr[k] = float(v)
                elif isinstance(v, (np.bool_,)):
                    sr[k] = bool(v)
                else:
                    sr[k] = v
            sr["selected"] = feat in {c["feature"] for c in selected}
            serializable[feat] = sr
        out_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"📊 Analysis saved to {out_path}")

    return 0 if selected else 1


if __name__ == "__main__":
    sys.exit(main())
