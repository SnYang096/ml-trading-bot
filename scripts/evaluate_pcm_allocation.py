#!/usr/bin/env python3
"""
PCM (Portfolio Control Manager) KPI 评估脚本

评估 PCM 仲裁层的有效性。

KPI 指标:
  1. conflict_rate:           冲突信号占比（低=策略互补）
  2. override_accuracy:       Override 后的 R vs 被覆盖信号的 R
  3. regime_switch_frequency: Regime 切换频率（不宜过高）
  4. per_archetype_contribution: 各策略对总 Sharpe 的贡献
  5. counterfactual_loss:     被拒信号的事后表现（反事实分析）

用法:
  python scripts/evaluate_pcm_allocation.py \\
    --pcm-report <pcm_backtest_report.html or .csv> \\
    [--output-dir results/pcm_kpi]

输入: PCM 联合回测结果（来自 backtest_execution_layer.py --pcm）
输出: KPI 报告（console + optional HTML）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# KPI 计算函数
# ================================================================


def compute_conflict_rate(df: pd.DataFrame) -> Dict[str, Any]:
    """计算冲突信号占比。

    冲突 = 同一 bar 多个 archetype 同时触发信号。
    低冲突率 = 策略互补性好。

    Returns:
        {"conflict_rate": float, "conflict_count": int, "total_bars": int}
    """
    if "archetype" not in df.columns:
        return {"conflict_rate": 0.0, "conflict_count": 0, "total_bars": 0}

    # Group by timestamp (or index) and count archetypes
    ts_col = None
    for col in ["timestamp", "datetime", "bar_index"]:
        if col in df.columns:
            ts_col = col
            break

    if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["_ts"] = df.index
        ts_col = "_ts"

    if ts_col is None:
        return {"conflict_rate": 0.0, "conflict_count": 0, "total_bars": 0}

    group_cols = [ts_col]
    if "symbol" in df.columns:
        group_cols.append("symbol")

    grouped = df.groupby(group_cols)["archetype"].nunique()
    conflict_count = int((grouped > 1).sum())
    total_bars = len(grouped)

    return {
        "conflict_rate": conflict_count / total_bars if total_bars > 0 else 0.0,
        "conflict_count": conflict_count,
        "total_bars": total_bars,
    }


def compute_per_archetype_contribution(
    df: pd.DataFrame,
    r_col: str = "r_multiple",
) -> Dict[str, Dict[str, float]]:
    """计算各 archetype 对总 Sharpe 的贡献。

    Returns:
        {archetype: {"count": N, "mean_r": float, "sharpe": float, "contribution_pct": float}}
    """
    if "archetype" not in df.columns or r_col not in df.columns:
        return {}

    result = {}
    total_pnl = df[r_col].sum()

    for arch, grp in df.groupby("archetype"):
        r = grp[r_col].dropna()
        mean_r = float(r.mean()) if len(r) > 0 else 0.0
        std_r = float(r.std(ddof=1)) if len(r) > 1 else 1.0
        sharpe = mean_r / std_r if std_r > 1e-8 else 0.0
        arch_pnl = float(r.sum())

        result[arch] = {
            "count": len(r),
            "mean_r": mean_r,
            "std_r": std_r,
            "sharpe": round(sharpe, 3),
            "total_r": round(arch_pnl, 3),
            "contribution_pct": round(
                arch_pnl / total_pnl * 100 if abs(total_pnl) > 1e-8 else 0.0, 1
            ),
        }

    return result


def compute_regime_stats(
    df: pd.DataFrame,
    r_col: str = "r_multiple",
) -> Dict[str, Dict[str, Any]]:
    """按 regime 统计各策略表现。

    需要 df 中有 'pcm_regime' 列（来自回测时的 regime 标注）。

    Returns:
        {regime: {"count": N, "archetypes": {arch: stats}, "sharpe": float}}
    """
    if "pcm_regime" not in df.columns or r_col not in df.columns:
        return {}

    result = {}
    for regime, grp in df.groupby("pcm_regime"):
        r = grp[r_col].dropna()
        mean_r = float(r.mean()) if len(r) > 0 else 0.0
        std_r = float(r.std(ddof=1)) if len(r) > 1 else 1.0

        arch_stats = {}
        if "archetype" in grp.columns:
            for arch, adf in grp.groupby("archetype"):
                ar = adf[r_col].dropna()
                arch_stats[arch] = {
                    "count": len(ar),
                    "mean_r": round(float(ar.mean()) if len(ar) > 0 else 0.0, 4),
                }

        result[regime] = {
            "count": len(r),
            "mean_r": round(mean_r, 4),
            "sharpe": round(mean_r / std_r if std_r > 1e-8 else 0.0, 3),
            "archetypes": arch_stats,
        }

    return result


def compute_counterfactual_loss(
    df: pd.DataFrame,
    r_col: str = "r_multiple",
) -> Dict[str, Any]:
    """反事实分析：被 PCM 拒绝的信号事后表现如何？

    需要 df 中有 'pcm_accepted' 列（True=执行, False=被拒）
    和 'counterfactual_r' 列（被拒信号的假设 R-multiple）。

    Returns:
        {"rejected_mean_r": float, "accepted_mean_r": float, "pcm_advantage": float}
    """
    if "pcm_accepted" not in df.columns:
        return {}

    accepted = df[df["pcm_accepted"] == True]
    rejected = df[df["pcm_accepted"] == False]

    cf_col = "counterfactual_r" if "counterfactual_r" in df.columns else r_col

    acc_mean = float(accepted[r_col].mean()) if len(accepted) > 0 else 0.0
    rej_mean = float(rejected[cf_col].mean()) if len(rejected) > 0 else 0.0

    return {
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_mean_r": round(acc_mean, 4),
        "rejected_mean_r": round(rej_mean, 4),
        "pcm_advantage": round(acc_mean - rej_mean, 4),
    }


def compute_regime_switch_frequency(
    regime_series: pd.Series,
) -> Dict[str, Any]:
    """计算 regime 切换频率。

    Args:
        regime_series: 每根 bar 的 regime 标注

    Returns:
        {"total_bars": int, "switch_count": int, "switch_rate": float,
         "avg_bars_per_regime": float, "regime_distribution": dict}
    """
    if len(regime_series) == 0:
        return {"total_bars": 0, "switch_count": 0, "switch_rate": 0.0}

    switches = (regime_series != regime_series.shift(1)).sum() - 1  # -1 for first bar
    switches = max(0, int(switches))

    dist = regime_series.value_counts(normalize=True).to_dict()
    dist = {k: round(v, 3) for k, v in dist.items()}

    n = len(regime_series)
    return {
        "total_bars": n,
        "switch_count": switches,
        "switch_rate": round(switches / n, 4) if n > 0 else 0.0,
        "avg_bars_per_regime": round(n / (switches + 1), 1) if switches >= 0 else n,
        "regime_distribution": dist,
    }


# ================================================================
# 主报告生成
# ================================================================


def generate_pcm_kpi_report(
    df: pd.DataFrame,
    r_col: str = "r_multiple",
) -> Dict[str, Any]:
    """生成完整 PCM KPI 报告。

    Args:
        df: PCM 联合回测结果 DataFrame。
            必须列: archetype, r_multiple
            可选列: pcm_regime, pcm_accepted, counterfactual_r, timestamp, symbol

    Returns:
        嵌套 dict KPI 报告
    """
    report: Dict[str, Any] = {}

    # 1. 冲突率
    report["conflict"] = compute_conflict_rate(df)

    # 2. 各策略贡献
    report["per_archetype"] = compute_per_archetype_contribution(df, r_col)

    # 3. 分 regime 统计
    report["regime_stats"] = compute_regime_stats(df, r_col)

    # 4. 反事实分析
    report["counterfactual"] = compute_counterfactual_loss(df, r_col)

    # 5. Regime 切换频率
    if "pcm_regime" in df.columns:
        report["regime_switches"] = compute_regime_switch_frequency(df["pcm_regime"])

    # 6. 总体统计
    r = df[r_col].dropna() if r_col in df.columns else pd.Series(dtype=float)
    if len(r) > 1:
        report["overall"] = {
            "total_trades": len(r),
            "mean_r": round(float(r.mean()), 4),
            "std_r": round(float(r.std(ddof=1)), 4),
            "sharpe": round(
                float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 1e-8 else 0.0, 3
            ),
            "win_rate": round(float((r > 0).mean()), 3),
            "max_drawdown_r": round(
                float((r.cumsum().expanding().max() - r.cumsum()).max()), 3
            ),
        }

    return report


def print_kpi_report(report: Dict[str, Any]) -> None:
    """美观打印 KPI 报告到 console。"""
    print("\n" + "=" * 60)
    print("  PCM KPI 评估报告")
    print("=" * 60)

    # Overall
    if "overall" in report:
        ov = report["overall"]
        print(f"\n📊 总体表现:")
        print(f"   总交易数: {ov['total_trades']}")
        print(f"   Mean R: {ov['mean_r']:.4f}")
        print(f"   Sharpe: {ov['sharpe']:.3f}")
        print(f"   胜率: {ov['win_rate']:.1%}")
        print(f"   最大回撤(R): {ov.get('max_drawdown_r', 'N/A')}")

    # Conflict
    if "conflict" in report:
        cf = report["conflict"]
        print(f"\n🔀 冲突率:")
        print(f"   冲突 bar: {cf['conflict_count']}/{cf['total_bars']}")
        print(f"   冲突率: {cf['conflict_rate']:.2%}")

    # Per archetype
    if "per_archetype" in report and report["per_archetype"]:
        print(f"\n📈 各 Archetype 贡献:")
        for arch, stats in sorted(
            report["per_archetype"].items(),
            key=lambda x: x[1].get("total_r", 0),
            reverse=True,
        ):
            print(
                f"   {arch:>5s}: trades={stats['count']:>4d}, "
                f"mean_R={stats['mean_r']:+.4f}, "
                f"sharpe={stats['sharpe']:.3f}, "
                f"贡献={stats['contribution_pct']:>5.1f}%"
            )

    # Regime stats
    if "regime_stats" in report and report["regime_stats"]:
        print(f"\n🌡️ 分 Regime 统计:")
        for regime, stats in report["regime_stats"].items():
            archs = ", ".join(
                f"{a}({s['count']})" for a, s in stats.get("archetypes", {}).items()
            )
            print(
                f"   {regime:>15s}: trades={stats['count']:>4d}, "
                f"sharpe={stats['sharpe']:.3f}, "
                f"archetypes=[{archs}]"
            )

    # Regime switches
    if "regime_switches" in report:
        rs = report["regime_switches"]
        print(f"\n🔄 Regime 切换:")
        print(f"   切换次数: {rs['switch_count']}")
        print(f"   切换率: {rs['switch_rate']:.4f} (每 bar)")
        print(f"   平均持续: {rs['avg_bars_per_regime']:.1f} bars/regime")
        if "regime_distribution" in rs:
            dist_str = ", ".join(
                f"{k}={v:.1%}" for k, v in rs["regime_distribution"].items()
            )
            print(f"   分布: {dist_str}")

    # Counterfactual
    if "counterfactual" in report and report["counterfactual"]:
        ct = report["counterfactual"]
        print(f"\n🔮 反事实分析:")
        print(
            f"   执行: {ct.get('accepted_count', 0)} 笔, mean_R={ct.get('accepted_mean_r', 0):.4f}"
        )
        print(
            f"   拒绝: {ct.get('rejected_count', 0)} 笔, mean_R={ct.get('rejected_mean_r', 0):.4f}"
        )
        advantage = ct.get("pcm_advantage", 0)
        emoji = "✅" if advantage > 0 else "❌"
        print(f"   PCM 优势: {emoji} {advantage:+.4f} R")

    print("\n" + "=" * 60)


# ================================================================
# CLI
# ================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PCM KPI 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pcm-report",
        type=str,
        required=True,
        help="PCM 联合回测结果文件 (CSV/Parquet)",
    )
    parser.add_argument(
        "--r-col",
        type=str,
        default="r_multiple",
        help="R-multiple 列名 (default: r_multiple)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (可选)",
    )

    args = parser.parse_args()

    # Load data
    path = Path(args.pcm_report)
    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        print(f"❌ 不支持的文件格式: {path.suffix} (支持 .csv, .parquet)")
        sys.exit(1)

    print(f"📂 加载数据: {path} ({len(df)} 行)")

    # Generate report
    report = generate_pcm_kpi_report(df, r_col=args.r_col)
    print_kpi_report(report)

    # Save if output dir specified
    if args.output_dir:
        import json

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pcm_kpi_report.json"

        # Make report JSON-serializable
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=_convert)
        print(f"\n💾 报告已保存: {out_path}")


if __name__ == "__main__":
    main()
