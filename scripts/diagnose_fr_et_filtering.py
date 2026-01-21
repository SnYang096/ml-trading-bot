#!/usr/bin/env python3
"""
诊断 FR/ET 交易缺失的原因，检查每一层的过滤情况。

检查层级：
1. Regime 分类：MEAN_REGIME 的数量
2. Router：MEAN mode 的输出
3. Gate：FR/ET archetype 的过滤
4. Execution：最终交易数量
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.rule.regime import PhysicsRegimeConfig, classify_regime
from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    QualityScoreConfig,
    compute_mode_3action_regime_aware,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose FR/ET filtering at each layer.")
    p.add_argument(
        "--preds", required=True, help="Predictions parquet directory or file"
    )
    p.add_argument("--output-md", default=None, help="Output Markdown report path")
    p.add_argument(
        "--relax-router",
        action="store_true",
        help="Relax router thresholds for testing",
    )
    p.add_argument(
        "--relax-regime",
        action="store_true",
        help="Relax regime classification for testing",
    )
    args = p.parse_args()

    # Load predictions
    preds_path = Path(args.preds)
    if preds_path.is_dir():
        pred_files = list(preds_path.glob("preds_*.parquet"))
        if not pred_files:
            print(f"Error: No preds files found in {preds_path}", file=sys.stderr)
            return 1
        preds_df = pd.concat(
            [pd.read_parquet(f) for f in pred_files], ignore_index=True
        )
    else:
        preds_df = pd.read_parquet(preds_path)

    # Ensure required columns
    required_cols = ["pred_dir_prob", "pred_mfe_atr", "pred_mae_atr", "pred_t_to_mfe"]
    missing = [c for c in required_cols if c not in preds_df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}", file=sys.stderr)
        return 1

    # Ensure timestamp
    if "timestamp" not in preds_df.columns:
        if isinstance(preds_df.index, pd.DatetimeIndex):
            preds_df = preds_df.copy()
            preds_df["timestamp"] = preds_df.index
        else:
            print("Error: No timestamp column found", file=sys.stderr)
            return 1

    # Ensure symbol
    if "symbol" not in preds_df.columns:
        print("Warning: No symbol column, using 'UNKNOWN'", file=sys.stderr)
        preds_df["symbol"] = "UNKNOWN"

    # Ensure required features for regime classification
    required_features = ["atr", "atr_percentile", "high", "low", "close"]
    missing_features = [f for f in required_features if f not in preds_df.columns]
    if missing_features:
        print(
            f"Warning: Missing features {missing_features}, using defaults",
            file=sys.stderr,
        )
        for f in missing_features:
            if f == "atr":
                preds_df["atr"] = 1.0
            elif f == "atr_percentile":
                preds_df["atr_percentile"] = 0.5
            else:
                preds_df[f] = preds_df["close"] if "close" in preds_df.columns else 0.0

    print("=" * 80)
    print("FR/ET 交易缺失诊断报告")
    print("=" * 80)
    print(f"\n总行数: {len(preds_df)}")

    # Layer 1: Regime Classification
    print("\n" + "=" * 80)
    print("Layer 1: Regime 分类")
    print("=" * 80)

    regime_cfg = PhysicsRegimeConfig()
    if args.relax_regime:
        # 放松 Regime 分类条件
        regime_cfg = PhysicsRegimeConfig(
            mean_deviation_z_min=2.0,  # 从 2.5 降到 2.0
            mean_path_length_min_pct=0.7,  # 从 0.8 降到 0.7
            mean_dir_sign_consistency_max_pct=0.5,  # 从 0.4 提高到 0.5
            mean_atr_percentile_min=0.8,  # 从 0.9 降到 0.8
        )
        print("⚠️  使用放松的 Regime 分类条件")

    regime_df = classify_regime(preds_df, cfg=regime_cfg)
    regime_counts = regime_df["regime"].value_counts()
    print("\nRegime 分布:")
    for regime, count in regime_counts.items():
        pct = count / len(regime_df) * 100
        print(f"  {regime}: {count} ({pct:.2f}%)")

    mean_regime_count = regime_counts.get("MEAN_REGIME", 0)
    print(
        f"\n✅ MEAN_REGIME 数量: {mean_regime_count} ({mean_regime_count/len(regime_df)*100:.2f}%)"
    )

    # Layer 2: Router (mode-3action)
    print("\n" + "=" * 80)
    print("Layer 2: Router (mode-3action)")
    print("=" * 80)

    router_cfg = Rule3ActionConfig()
    quality_cfg = QualityScoreConfig()
    if args.relax_router:
        # 放松 Router 阈值
        router_cfg = Rule3ActionConfig(
            eff_mean_min=1.05,  # 从 1.15 降到 1.05
            ttm_mean_max=15.0,  # 从 12.0 提高到 15.0
        )
        quality_cfg = QualityScoreConfig(
            quality_mean_min=0.5,  # 从 0.8 降到 0.5
        )
        print("⚠️  使用放松的 Router 阈值")

    router_df = compute_mode_3action_regime_aware(
        preds_df,
        rule_cfg=router_cfg,
        score_cfg=quality_cfg,
        use_physics_regime=True,
        physics_regime_cfg=regime_cfg,
    )

    mode_counts = router_df["mode"].value_counts()
    print("\nMode 分布 (总体):")
    for mode, count in mode_counts.items():
        pct = count / len(router_df) * 100
        print(f"  {mode}: {count} ({pct:.2f}%)")

    # 检查 MEAN_REGIME 中的 mode 分布
    # 避免列名冲突：只选择需要的列
    regime_cols = ["regime"]
    router_cols = ["mode"]
    pred_cols = ["timestamp", "symbol"]

    # 合并时使用 suffixes 避免冲突
    merged = preds_df[pred_cols].copy()
    merged = merged.merge(
        regime_df[regime_cols],
        left_index=True,
        right_index=True,
        how="left",
        suffixes=("", "_regime"),
    )
    merged = merged.merge(
        router_df[router_cols],
        left_index=True,
        right_index=True,
        how="left",
        suffixes=("", "_router"),
    )
    mean_regime_df = merged[merged["regime"] == "MEAN_REGIME"]
    if len(mean_regime_df) > 0:
        print(f"\n在 MEAN_REGIME 中的 Mode 分布 ({len(mean_regime_df)} 行):")
        mean_mode_counts = mean_regime_df["mode"].value_counts()
        for mode, count in mean_mode_counts.items():
            pct = count / len(mean_regime_df) * 100
            print(f"  {mode}: {count} ({pct:.2f}%)")
    else:
        print("\n⚠️  MEAN_REGIME 中没有数据")

    mean_mode_count = merged[
        (merged["regime"] == "MEAN_REGIME") & (merged["mode"] == "MEAN")
    ].shape[0]
    print(f"\n✅ MEAN_REGIME + MEAN mode 数量: {mean_mode_count}")

    # Layer 3: Semantic Scores
    print("\n" + "=" * 80)
    print("Layer 3: Semantic Scores (FR/ET)")
    print("=" * 80)

    if (
        "fr_semantic_score" in regime_df.columns
        and "et_semantic_score" in regime_df.columns
    ):
        mean_regime_semantic = regime_df[regime_df["regime"] == "MEAN_REGIME"]
        if len(mean_regime_semantic) > 0:
            fr_scores = pd.to_numeric(
                mean_regime_semantic["fr_semantic_score"], errors="coerce"
            )
            et_scores = pd.to_numeric(
                mean_regime_semantic["et_semantic_score"], errors="coerce"
            )

            print(
                f"\nFR Semantic Score (MEAN_REGIME, {fr_scores.notna().sum()} 有效值):"
            )
            if fr_scores.notna().any():
                print(f"  均值: {fr_scores.mean():.4f}")
                print(f"  中位数: {fr_scores.median():.4f}")
                print(f"  P10: {fr_scores.quantile(0.1):.4f}")
                print(f"  P90: {fr_scores.quantile(0.9):.4f}")

            print(
                f"\nET Semantic Score (MEAN_REGIME, {et_scores.notna().sum()} 有效值):"
            )
            if et_scores.notna().any():
                print(f"  均值: {et_scores.mean():.4f}")
                print(f"  中位数: {et_scores.median():.4f}")
                print(f"  P10: {et_scores.quantile(0.1):.4f}")
                print(f"  P90: {et_scores.quantile(0.9):.4f}")

            # 语义分桶
            print("\n语义分桶 (5 个桶):")
            for score_col, name in [
                ("fr_semantic_score", "FR"),
                ("et_semantic_score", "ET"),
            ]:
                scores = pd.to_numeric(mean_regime_semantic[score_col], errors="coerce")
                if scores.notna().any():
                    try:
                        buckets = pd.qcut(scores, q=5, duplicates="drop")
                        print(f"\n{name} Semantic Score 分桶:")
                        for bucket in buckets.cat.categories:
                            count = (buckets == bucket).sum()
                            mean_score = scores[buckets == bucket].mean()
                            print(f"  {bucket}: {count} 行, 平均分数 {mean_score:.4f}")
                    except ValueError:
                        print(f"\n{name}: 无法分桶（数据不足或重复值）")
    else:
        print("⚠️  语义分数列不存在")

    # 生成报告
    if args.output_md:
        lines = []
        lines.append("# FR/ET 交易缺失诊断报告\n\n")
        lines.append(f"- 总行数: {len(preds_df)}\n")
        lines.append(f"- 使用放松的 Regime 条件: {args.relax_regime}\n")
        lines.append(f"- 使用放松的 Router 条件: {args.relax_router}\n\n")

        lines.append("## Layer 1: Regime 分类\n\n")
        lines.append("| Regime | 数量 | 百分比 |\n|---|---|---|\n")
        for regime, count in regime_counts.items():
            pct = count / len(regime_df) * 100
            lines.append(f"| {regime} | {count} | {pct:.2f}% |\n")

        lines.append("\n## Layer 2: Router (mode-3action)\n\n")
        lines.append("| Mode | 数量 | 百分比 |\n|---|---|---|\n")
        for mode, count in mode_counts.items():
            pct = count / len(router_df) * 100
            lines.append(f"| {mode} | {count} | {pct:.2f}% |\n")

        if len(mean_regime_df) > 0:
            lines.append("\n### MEAN_REGIME 中的 Mode 分布\n\n")
            lines.append("| Mode | 数量 | 百分比 |\n|---|---|---|\n")
            for mode, count in mean_mode_counts.items():
                pct = count / len(mean_regime_df) * 100
                lines.append(f"| {mode} | {count} | {pct:.2f}% |\n")

        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(lines), encoding="utf-8")
        print(f"\n✅ 报告已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
