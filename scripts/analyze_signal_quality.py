#!/usr/bin/env python3
"""
P0.2: 信号质量分析 — 研究回测中 top N% 信号贡献了多少 Sharpe

目的: 搞清楚研究 Sharpe 是由少数精英信号撑起来的，还是普遍盈利。
      如果 top 30% 信号贡献 80%+ 的 R，说明 slot 约束下只要选对信号，
      Sharpe 可以很高。

用法:
  python scripts/analyze_signal_quality.py \
    --logs results/train_final_.../bpc/logs_gated.parquet \
    --strategy bpc

  # 多策略合并分析
  python scripts/analyze_signal_quality.py \
    --logs bpc:results/.../bpc/logs_gated.parquet \
          fer:results/.../fer/logs_gated.parquet \
          me:results/.../me/logs_gated.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 加载 execution 配置 (读 tier params)
# ---------------------------------------------------------------------------
def _load_execution_config(strategy: str) -> dict:
    """加载策略的 execution.yaml"""
    import yaml

    path = Path(f"config/strategies/{strategy}/execution.yaml")
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _compute_rr_from_logs(df: pd.DataFrame, strategy: str) -> pd.Series:
    """从 logs_gated 计算每个信号的理论 R-multiple (使用实际 execution 参数)"""
    exec_cfg = _load_execution_config(strategy)
    tiers = exec_cfg.get("tiers", [])

    # 获取默认 initial_r 和 target_r
    default_initial_r = exec_cfg.get("initial_risk_r", 1.5)
    default_target_r = exec_cfg.get("take_profit_r", 2.5)

    # 简化: 用 label 列 (rr_extreme / return_tree 的结果) 作为实际 R
    if "pnl_r" in df.columns:
        return df["pnl_r"]
    if "label" in df.columns:
        return df["label"]
    if "return" in df.columns:
        return df["return"]

    print(f"  ⚠️  No R-multiple column found for {strategy}")
    return pd.Series(0.0, index=df.index)


def analyze_single_strategy(df: pd.DataFrame, strategy: str, silent: bool = False):
    """分析单策略信号质量分布"""
    # 只看 gate_decision=allow 的信号
    if "gate_decision" in df.columns:
        mask = df["gate_decision"].astype(str).str.lower().isin(["allow", "1", "true"])
        df_active = df[mask].copy()
    else:
        df_active = df.copy()

    n_total = len(df)
    n_active = len(df_active)

    r_values = _compute_rr_from_logs(df_active, strategy)
    r_values = r_values.dropna()

    if len(r_values) == 0:
        print(f"  ❌ No valid R values for {strategy}")
        return None

    # 基本统计
    mean_r = r_values.mean()
    win_rate = (r_values > 0).mean()
    total_r = r_values.sum()
    sharpe_r = r_values.mean() / r_values.std() if r_values.std() > 0 else 0

    if not silent:
        print(f"\n{'='*70}")
        print(f"  📊 {strategy.upper()} 信号质量分析")
        print(f"{'='*70}")
        print(
            f"  总信号: {n_total}  Gate Allow: {n_active} ({n_active/n_total*100:.1f}%)"
        )
        print(
            f"  Mean R: {mean_r:.4f}  Win%: {win_rate:.1%}  Total R: {total_r:.2f}  Sharpe(R): {sharpe_r:.4f}"
        )

    # evidence score 分布
    ev_col = None
    for col in ["evidence_score", "evidence", "ev_score"]:
        if col in df_active.columns:
            ev_col = col
            break

    if ev_col and not silent:
        ev = df_active[ev_col].dropna()
        print(f"\n  Evidence Score 分布:")
        print(
            f"    min={ev.min():.3f}  25%={ev.quantile(0.25):.3f}  50%={ev.quantile(0.5):.3f}  75%={ev.quantile(0.75):.3f}  max={ev.max():.3f}"
        )

    # 按 R 排序，分析 top N% 贡献
    sorted_r = r_values.sort_values(ascending=False)
    n = len(sorted_r)

    if not silent:
        print(f"\n  信号质量分层 (按 R-multiple 排序):")
        print(
            f"  {'分层':<15} {'信号数':<8} {'占比':<8} {'Mean R':<10} {'Total R':<10} {'贡献占比':<10}"
        )
        print(f"  {'-'*65}")

    results = {}
    for pct in [10, 20, 30, 50, 70, 100]:
        cutoff = int(n * pct / 100)
        subset = sorted_r.iloc[:cutoff]
        subset_mean = subset.mean()
        subset_total = subset.sum()
        contrib = subset_total / total_r * 100 if total_r != 0 else 0

        results[f"top_{pct}"] = {
            "count": cutoff,
            "mean_r": subset_mean,
            "total_r": subset_total,
            "contribution_pct": contrib,
        }

        if not silent:
            print(
                f"  Top {pct:>3}%       {cutoff:<8} {pct:>5}%   {subset_mean:>8.4f}  {subset_total:>9.2f}  {contrib:>8.1f}%"
            )

    # 正/负信号占比
    n_positive = (r_values > 0).sum()
    n_negative = (r_values <= 0).sum()
    positive_total = r_values[r_values > 0].sum()
    negative_total = r_values[r_values <= 0].sum()

    if not silent:
        print(f"\n  盈亏分布:")
        print(
            f"    盈利信号: {n_positive} ({n_positive/n*100:.1f}%)  Total R: +{positive_total:.2f}"
        )
        print(
            f"    亏损信号: {n_negative} ({n_negative/n*100:.1f}%)  Total R: {negative_total:.2f}"
        )
        print(
            f"    盈亏比: {abs(positive_total/negative_total):.2f}x"
            if negative_total != 0
            else ""
        )

    # 如果有 evidence score，按分位数分层看质量
    if ev_col:
        ev = df_active[ev_col]
        r_with_ev = pd.DataFrame({"r": r_values, "ev": ev}).dropna()

        if len(r_with_ev) > 20 and not silent:
            print(f"\n  按 Evidence Score 分层:")
            print(
                f"  {'Evidence 区间':<20} {'信号数':<8} {'Mean R':<10} {'Win%':<8} {'Total R':<10}"
            )
            print(f"  {'-'*60}")

            # 四分位分层
            quartiles = [0, 0.25, 0.5, 0.75, 1.0]
            for i in range(len(quartiles) - 1):
                lo = r_with_ev["ev"].quantile(quartiles[i])
                hi = r_with_ev["ev"].quantile(quartiles[i + 1])
                mask = (r_with_ev["ev"] >= lo) & (r_with_ev["ev"] <= hi)
                if i < len(quartiles) - 2:
                    mask = (r_with_ev["ev"] >= lo) & (r_with_ev["ev"] < hi)
                sub = r_with_ev[mask]
                if len(sub) == 0:
                    continue
                label = f"Q{i+1} [{lo:.3f},{hi:.3f}]"
                print(
                    f"  {label:<20} {len(sub):<8} {sub['r'].mean():>8.4f}  {(sub['r']>0).mean():>6.1%}  {sub['r'].sum():>9.2f}"
                )

    return {
        "strategy": strategy,
        "n_total": n_total,
        "n_active": n_active,
        "mean_r": mean_r,
        "win_rate": win_rate,
        "total_r": total_r,
        "sharpe_r": sharpe_r,
        "quality_layers": results,
    }


def main():
    parser = argparse.ArgumentParser(description="信号质量分析")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="strategy:path 或 单个 path (需配合 --strategy)",
    )
    parser.add_argument(
        "--strategy", default=None, help="单策略名 (当 --logs 无前缀时)"
    )
    args = parser.parse_args()

    all_results = []

    for spec in args.logs:
        if ":" in spec:
            strategy, path = spec.split(":", 1)
        else:
            strategy = args.strategy or Path(spec).parent.name
            path = spec

        if not Path(path).exists():
            print(f"❌ File not found: {path}")
            continue

        df = pd.read_parquet(path)
        result = analyze_single_strategy(df, strategy)
        if result:
            all_results.append(result)

    # 多策略汇总
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  📊 多策略汇总")
        print(f"{'='*70}")
        print(
            f"  {'策略':<8} {'Active':<8} {'Mean R':<10} {'Win%':<8} {'Total R':<10} {'Sharpe(R)':<10}"
        )
        print(f"  {'-'*58}")
        total_active = 0
        total_total_r = 0
        for r in all_results:
            print(
                f"  {r['strategy']:<8} {r['n_active']:<8} {r['mean_r']:>8.4f}  {r['win_rate']:>6.1%}  {r['total_r']:>9.2f}  {r['sharpe_r']:>8.4f}"
            )
            total_active += r["n_active"]
            total_total_r += r["total_r"]

        print(f"  {'-'*58}")
        print(
            f"  {'TOTAL':<8} {total_active:<8} {'':>8}  {'':>6}  {total_total_r:>9.2f}"
        )

        # 关键洞察
        print(f"\n  💡 关键洞察:")
        for r in all_results:
            top30 = r["quality_layers"].get("top_30", {})
            contrib = top30.get("contribution_pct", 0)
            print(f"    {r['strategy']}: Top 30% 信号贡献 {contrib:.0f}% 的 Total R")
            if contrib > 70:
                print(
                    f"    → 信号质量高度集中! 只要选对 top 信号, slot 约束下 Sharpe 可保持高位"
                )
            else:
                print(f"    → 信号质量较均匀, 需要更多 slot 才能捕获收益")


if __name__ == "__main__":
    main()
