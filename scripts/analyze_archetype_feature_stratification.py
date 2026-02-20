#!/usr/bin/env python3
"""分位数分层分析：验证 archetype 语义特征的预测力。

对指定策略的候选特征，按百分位阈值切分数据，
对比「有语义信号」vs「无语义信号」的 bad rate 和 median RR。

用法:
    # 配置驱动 (推荐: 从 prefilter.yaml 读取候选特征)
    python scripts/analyze_archetype_feature_stratification.py \
        --logs results/train_final_xxx/bpc/predictions.parquet \
        --strategy bpc \
        --config config/strategies/bpc/prefilter.yaml

    # 前缀模式 (fallback: 硬编码前缀匹配)
    python scripts/analyze_archetype_feature_stratification.py \
        --logs results/train_final_xxx/me/predictions.parquet \
        --strategy me --prefix me_

    # 指定阈值百分位
    python scripts/analyze_archetype_feature_stratification.py \
        --logs results/train_final_xxx/fer/predictions.parquet \
        --strategy fer --percentiles 5,10,80,90,95 \
        --config config/strategies/fer/prefilter.yaml

    # 输出 JSON 报告
    python scripts/analyze_archetype_feature_stratification.py \
        --logs results/train_final_xxx/bpc/predictions.parquet \
        --strategy bpc --output results/bpc_stratification.json \
        --config config/strategies/bpc/prefilter.yaml

算法:
    对每个候选特征:
      1. 取全量数据 (predictions.parquet)
      2. 按 P5 / P10 / P80 / P90 / P95 阈值切分数据为两组
      3. 计算两组的:
         - bad rate = failure_rr_extreme 占比 (forward_rr < -0.8R)
         - median forward_rr
      4. 差异越大 → 该特征在此阈值处有区分力

特征来源 (二选一):
    --config: 读 prefilter.yaml 的 candidates 列表 → 查 feature_dependencies.yaml
             的 output_columns 解析实际列名 → 匹配 parquet 中存在的列
    --prefix: 按前缀匹配 parquet 中的列 (旧方案, fallback)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import pandas as pd

# ── 默认前缀映射 (fallback, 无 --config 时使用) ─────────────────
STRATEGY_PREFIX_MAP = {
    "bpc": ["bpc_"],
    "me": ["me_"],
    "fer": ["fer_"],
    "lv": ["oi_", "funding_rate_abs_zscore", "funding_rate_zscore"],
}

# ── 默认百分位 ────────────────────────────────────────────────
DEFAULT_PERCENTILES = [5, 10, 20, 80, 90, 95]

# ── 默认最小样本量 ─────────────────────────────────────────
DEFAULT_MIN_SAMPLES = 30

# ── 默认 feature_dependencies.yaml 路径 ───────────────────────
DEFAULT_DEPS_PATH = "config/feature_dependencies.yaml"


def _resolve_features_from_config(
    config_path: str,
    deps_path: str,
    available_columns: List[str],
) -> List[str]:
    """从 prefilter.yaml + feature_dependencies.yaml 解析候选特征列名。

    链路: prefilter.yaml (candidates) → feature_dependencies.yaml (output_columns)
          → 匹配 parquet 中存在的列
    """
    # 1. 读 prefilter.yaml
    with open(config_path, "r", encoding="utf-8") as f:
        prefilter_cfg = yaml.safe_load(f)

    candidates = prefilter_cfg.get("candidates", [])
    if not candidates:
        print(f"❌ prefilter.yaml 中没有 candidates: {config_path}")
        return []

    # 2. 读 feature_dependencies.yaml
    with open(deps_path, "r", encoding="utf-8") as f:
        deps_cfg = yaml.safe_load(f)

    all_features_def = deps_cfg.get("features", {})

    # 3. 解析 _f → output_columns
    resolved_columns = []
    available_set = set(available_columns)

    for feat_f in candidates:
        if feat_f not in all_features_def:
            print(f"⚠️  _f '{feat_f}' 在 feature_dependencies.yaml 中未找到, 跳过")
            continue

        output_cols = all_features_def[feat_f].get("output_columns", [])
        matched = [c for c in output_cols if c in available_set]
        skipped = [c for c in output_cols if c not in available_set]

        if matched:
            resolved_columns.extend(matched)
            suffix = "..." if len(matched) > 5 else ""
            print(
                f"  \u2705 {feat_f}: {len(matched)} \u5217\u5339\u914d ({', '.join(matched[:5])}{suffix})"
            )
        if skipped:
            print(f"  ℹ️  {feat_f}: {len(skipped)} 列不在 parquet 中: {skipped[:5]}")

    return sorted(set(resolved_columns))


def _resolve_prefixes(strategy: str, prefix_override: Optional[str]) -> List[str]:
    """解析特征前缀列表 (fallback 模式)。"""
    if prefix_override:
        return [p.strip() for p in prefix_override.split(",")]
    return STRATEGY_PREFIX_MAP.get(strategy.lower(), [f"{strategy.lower()}_"])


def _find_archetype_features(columns: List[str], prefixes: List[str]) -> List[str]:
    """从 DataFrame 列中找出匹配前缀的 archetype 专属特征 (fallback 模式)。"""
    features = []
    for col in sorted(columns):
        if any(col.startswith(p) for p in prefixes):
            features.append(col)
    return features


def _compute_stratification(
    df: pd.DataFrame,
    feature: str,
    threshold: float,
    operator: str,  # "high" (>= threshold) or "low" (<= threshold)
    rr_col: str,
    label_col: str,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Optional[Dict[str, Any]]:
    """
    按阈值分层，计算两组的 bad rate 和 median RR。

    operator="high": 分析「高端信号」→ df[feature >= threshold] vs 其余
    operator="low":  分析「低端信号」→ df[feature <= threshold] vs 其余
    """
    # 避免重复列名导致 DataFrame 而非 Series
    cols_needed = [feature, label_col]
    if rr_col in df.columns and df[rr_col].notna().any():
        cols_needed.append(rr_col)
        has_rr = True
    else:
        has_rr = False
    # 去重列名
    cols_needed = list(dict.fromkeys(cols_needed))
    valid = df[cols_needed].copy()
    if valid.columns.duplicated().any():
        valid = valid.loc[:, ~valid.columns.duplicated()]
    valid = valid.dropna().reset_index(drop=True)
    if len(valid) < min_samples * 2:
        return None

    if operator == "high":
        signal_mask = (valid[feature] >= threshold).values
    else:
        signal_mask = (valid[feature] <= threshold).values

    signal_df = valid.loc[signal_mask]
    rest_df = valid.loc[~signal_mask]

    if len(signal_df) < min_samples or len(rest_df) < min_samples:
        return None

    # bad rate = 标签为 0 的比例 (label_col = success_no_rr_extreme, 0 = 踩坑)
    signal_bad_rate = (signal_df[label_col] == 0).mean()
    rest_bad_rate = (rest_df[label_col] == 0).mean()

    # median forward_rr
    signal_med_rr = (
        float(signal_df[rr_col].median())
        if rr_col in signal_df.columns
        else float("nan")
    )
    rest_med_rr = (
        float(rest_df[rr_col].median()) if rr_col in rest_df.columns else float("nan")
    )

    return {
        "n_signal": len(signal_df),
        "n_rest": len(rest_df),
        "bad_rate_signal": round(signal_bad_rate, 4),
        "bad_rate_rest": round(rest_bad_rate, 4),
        "bad_rate_diff": round(signal_bad_rate - rest_bad_rate, 4),
        "bad_rate_diff_abs": round(abs(signal_bad_rate - rest_bad_rate), 4),
        "median_rr_signal": round(signal_med_rr, 2),
        "median_rr_rest": round(rest_med_rr, 2),
        "threshold": round(threshold, 4),
    }


def analyze_feature(
    df: pd.DataFrame,
    feature: str,
    percentiles: List[int],
    rr_col: str,
    label_col: str,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> List[Dict[str, Any]]:
    """分析单个特征在多个百分位阈值下的分层效果。"""
    valid = df[feature].dropna()
    if len(valid) < min_samples * 2:
        return []

    results = []

    for pct in percentiles:
        threshold = float(np.percentile(valid, pct))

        # 高端分析 (P80, P90, P95): feature >= threshold → "有信号"
        if pct >= 50:
            r = _compute_stratification(
                df, feature, threshold, "high", rr_col, label_col, min_samples
            )
            if r:
                r["percentile"] = f"P{pct}"
                r["direction"] = "high"
                r["feature"] = feature
                results.append(r)

        # 低端分析 (P5, P10, P20): feature <= threshold → "低端/缺失"
        if pct <= 50:
            r = _compute_stratification(
                df, feature, threshold, "low", rr_col, label_col, min_samples
            )
            if r:
                r["percentile"] = f"P{pct}"
                r["direction"] = "low"
                r["feature"] = feature
                results.append(r)

    return results


def _format_table(results: List[Dict[str, Any]], title: str) -> str:
    """格式化输出表格。"""
    if not results:
        return f"\n{title}\n  (无有效结果)\n"

    lines = [
        f"\n{title}",
        f"{'特征':<35s} {'阈值':<12s} {'方向':>4s} {'n':>6s} {'bad_rate':>10s} {'vs其余':>10s} {'差异':>8s} {'medRR':>8s} {'vs其余':>8s}",
        "-" * 110,
    ]

    for r in results:
        op_str = ">=" if r["direction"] == "high" else "<="
        threshold_str = f"{r['percentile']}({op_str}{r['threshold']:.3f})"
        diff_str = f"{r['bad_rate_diff']:+.1%}"
        lines.append(
            f"{r['feature']:<35s} {threshold_str:<12s} {r['direction']:>4s} "
            f"{r['n_signal']:>6d} {r['bad_rate_signal']:>9.1%} {r['bad_rate_rest']:>9.1%} "
            f"{diff_str:>8s} {r['median_rr_signal']:>+7.2f} {r['median_rr_rest']:>+7.2f}"
        )

    return "\n".join(lines)


def _classify_results(
    results: List[Dict[str, Any]],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    分类结果为三组:
    - 正信号 (high=good): 高端有信号 + bad_rate 降低
    - 反信号 (high=bad): 高端有信号 + bad_rate 升高
    - 低端信号 (absence=bad): 低端缺失信号 + bad_rate 升高
    """
    positive = []  # high=good (bad_rate_signal < bad_rate_rest)
    anti = []  # high=bad (bad_rate_signal > bad_rate_rest)
    absence = []  # low, absence=bad (bad_rate_signal > bad_rate_rest)

    for r in results:
        if r["direction"] == "high":
            if r["bad_rate_diff"] < -0.02:  # 信号组 bad rate 低 2% 以上 → 正信号
                positive.append(r)
            elif r["bad_rate_diff"] > 0.02:  # 信号组 bad rate 高 2% 以上 → 反信号
                anti.append(r)
        elif r["direction"] == "low":
            if r["bad_rate_diff"] > 0.02:  # 低端 bad rate 高 → absence=bad
                absence.append(r)
            elif r["bad_rate_diff"] < -0.02:
                positive.append(r)

    # 按差异绝对值排序
    positive.sort(key=lambda x: x["bad_rate_diff_abs"], reverse=True)
    anti.sort(key=lambda x: x["bad_rate_diff_abs"], reverse=True)
    absence.sort(key=lambda x: x["bad_rate_diff_abs"], reverse=True)

    return positive, anti, absence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="分位数分层分析：验证 archetype 语义特征的预测力"
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="predictions.parquet 路径 (训练管线输出)",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["bpc", "me", "fer", "lv"],
        help="策略名称",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="特征前缀 (逗号分隔, fallback 模式: 未指定 --config 时使用)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="prefilter.yaml 路径 (推荐: 配置驱动特征发现)",
    )
    parser.add_argument(
        "--deps",
        default=DEFAULT_DEPS_PATH,
        help=f"feature_dependencies.yaml 路径 (默认: {DEFAULT_DEPS_PATH})",
    )
    parser.add_argument(
        "--percentiles",
        default=",".join(str(p) for p in DEFAULT_PERCENTILES),
        help=f"百分位列表 (逗号分隔, 默认: {DEFAULT_PERCENTILES})",
    )
    parser.add_argument(
        "--rr-col",
        default="forward_rr",
        help="forward_rr 列名 (默认: forward_rr)",
    )
    parser.add_argument(
        "--label-col",
        default="success_no_rr_extreme",
        help="标签列名 (默认: success_no_rr_extreme, 1=好 0=坏)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSON 路径 (可选)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"每组最小样本量 (默认: {DEFAULT_MIN_SAMPLES})",
    )
    args = parser.parse_args()

    min_samples = args.min_samples

    # ── 1. 加载数据 ──────────────────────────────────────────
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ 文件不存在: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"✅ 加载 {len(df)} 行, {len(df.columns)} 列 from {logs_path}")

    # ── 2. 检查标签列 ────────────────────────────────────────
    rr_col = args.rr_col
    label_col = args.label_col

    # 自动生成标签 (与 optimize_gate_unified.py 一致)
    if label_col not in df.columns and rr_col in df.columns:
        df[label_col] = (df[rr_col] >= -0.8).astype(int)
        print(f"ℹ️  自动生成 '{label_col}' from '{rr_col}' (threshold: -0.8R)")

    if label_col not in df.columns:
        print(f"❌ 标签列 '{label_col}' 不存在")
        print(
            f"   可用列: {[c for c in df.columns if 'success' in c or 'failure' in c or 'rr' in c][:20]}"
        )
        return 1

    if rr_col not in df.columns:
        # 尝试常见 fallback
        for candidate in ["bpc_impulse_return_atr", "rr", "return_atr"]:
            if candidate in df.columns:
                rr_col = candidate
                break
        if rr_col not in df.columns:
            print(f"⚠️  RR 列 '{args.rr_col}' 不存在, median RR 将显示 NaN")
            df[rr_col] = np.nan

    # 统计概览
    n_total = len(df)
    n_valid = df[label_col].notna().sum()
    bad_rate = (df[label_col] == 0).mean()
    med_rr = df[rr_col].median() if rr_col in df.columns else float("nan")

    symbols = df["symbol"].nunique() if "symbol" in df.columns else "?"
    print(f"   {n_valid} valid rows, {symbols} symbols")
    print(f"   全局 bad rate: {bad_rate:.1%}, median RR: {med_rr:+.2f}")

    # ── 3. 找候选特征 (config 驱动 / prefix fallback) ────────
    if args.config:
        # 配置驱动模式: prefilter.yaml → feature_dependencies.yaml → parquet 列
        config_path = Path(args.config)
        deps_path = Path(args.deps)
        if not config_path.exists():
            print(f"❌ prefilter.yaml 不存在: {config_path}")
            return 1
        if not deps_path.exists():
            print(f"❌ feature_dependencies.yaml 不存在: {deps_path}")
            return 1
        print(f"\n🔧 配置驱动模式: {config_path}")
        features = _resolve_features_from_config(
            str(config_path), str(deps_path), list(df.columns)
        )
        feature_mode = "config"
    else:
        # Fallback: 前缀模式
        prefixes = _resolve_prefixes(args.strategy, args.prefix)
        features = _find_archetype_features(list(df.columns), prefixes)
        feature_mode = "prefix"

    if not features:
        if feature_mode == "config":
            print(f"❌ 从 prefilter.yaml 解析后无匹配列")
            print(f"   提示: 检查 prefilter.yaml 的 candidates 是否与 parquet 列名对应")
        else:
            print(f"❌ 未找到前缀为 {prefixes} 的特征")
            print(
                f"   提示: 使用 --prefix 手动指定, 或使用 --config 指定 prefilter.yaml"
            )
        return 1

    mode_desc = (
        f"config: {args.config}" if feature_mode == "config" else f"prefix: {prefixes}"
    )
    print(f"\n📊 找到 {len(features)} 个候选特征 ({mode_desc})")
    for f in features:
        n_valid_f = df[f].notna().sum()
        n_nonzero = (df[f] != 0).sum() if df[f].notna().any() else 0
        print(f"   {f}: {n_valid_f} valid, {n_nonzero} nonzero")

    # ── 4. 逐特征分层分析 ────────────────────────────────────
    percentiles = [int(p) for p in args.percentiles.split(",")]
    all_results = []

    for feat in features:
        feat_results = analyze_feature(
            df, feat, percentiles, rr_col, label_col, min_samples
        )
        all_results.extend(feat_results)

    if not all_results:
        print("\n⚠️  所有特征均无有效分层结果 (样本不足)")
        return 0

    # ── 5. 分类和输出 ────────────────────────────────────────
    positive, anti, absence = _classify_results(all_results)

    print(f"\n{'='*110}")
    print(f"📊 {args.strategy.upper()} 语义特征分位数分层分析 (mode={feature_mode})")
    print(
        f"   数据: {n_valid} rows, {symbols} symbols, bad rate={bad_rate:.1%}, medRR={med_rr:+.2f}"
    )
    print(f"{'='*110}")

    print(_format_table(positive, "✅ 正信号 (高端有信号 → bad rate 降低)"))
    print(_format_table(anti, "⚠️  反信号 (高端有信号 → bad rate 升高)"))
    print(_format_table(absence, "🔻 低端信号 (缺失信号 → bad rate 升高)"))

    # ── 6. 总结 ──────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("📋 总结")
    print(f"   正信号: {len(positive)} 条 (可做 guardrail / pre_filter 的正向条件)")
    print(f"   反信号: {len(anti)} 条 (高值=坏, 可做反向 hard_gate)")
    print(f"   低端信号: {len(absence)} 条 (缺失=坏, 可做最低门槛)")

    if positive:
        best = positive[0]
        print(
            f"\n   最强正信号: {best['feature']} {best['percentile']}, "
            f"bad_rate: {best['bad_rate_signal']:.1%} vs {best['bad_rate_rest']:.1%} "
            f"(差异 {best['bad_rate_diff']:+.1%})"
        )
    if anti:
        best = anti[0]
        print(
            f"   最强反信号: {best['feature']} {best['percentile']}, "
            f"bad_rate: {best['bad_rate_signal']:.1%} vs {best['bad_rate_rest']:.1%} "
            f"(差异 {best['bad_rate_diff']:+.1%})"
        )
    if absence:
        best = absence[0]
        print(
            f"   最强低端信号: {best['feature']} {best['percentile']}, "
            f"bad_rate: {best['bad_rate_signal']:.1%} vs {best['bad_rate_rest']:.1%} "
            f"(差异 {best['bad_rate_diff']:+.1%})"
        )

    # ── 7. 输出 JSON ─────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "strategy": args.strategy,
            "feature_mode": feature_mode,
            "config_path": args.config,
            "data": {
                "n_rows": n_total,
                "n_valid": int(n_valid),
                "n_symbols": (
                    int(symbols) if isinstance(symbols, (int, np.integer)) else symbols
                ),
                "global_bad_rate": round(bad_rate, 4),
                "global_median_rr": round(med_rr, 2),
            },
            "features_analyzed": len(features),
            "features_list": features,
            "percentiles_used": percentiles,
            "summary": {
                "n_positive": len(positive),
                "n_anti": len(anti),
                "n_absence": len(absence),
            },
            "positive_signals": positive,
            "anti_signals": anti,
            "absence_signals": absence,
            "all_results": all_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 报告已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
