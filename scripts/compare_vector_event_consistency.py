#!/usr/bin/env python3
"""
向量回测 vs 事件回测 一致性对比报告

用法:
    # Step 1: 跑向量回测 (PCM 模式) 并导出交易明细
    python scripts/backtest_execution_layer.py \
        --pcm bpc:results/.../logs_gated.parquet \
              fer:results/.../logs_gated.parquet \
              me:results/.../logs_gated.parquet \
        --use-1min --export-trades /tmp/trades_vector.csv

    # Step 2: 跑事件回测并导出交易明细
    python scripts/event_backtest.py \
        --strategy bpc,fer,me --days 180 \
        --export /tmp/trades_event.csv

    # Step 3: 对比
    python scripts/compare_vector_event_consistency.py \
        --vector /tmp/trades_vector.csv \
        --event /tmp/trades_event.csv

对比维度:
    1. 总交易数偏差 (pass < 10%)
    2. Sharpe 偏差 (pass < 0.5x)
    3. 胜率偏差 (pass < 5pp)
    4. 出场原因分布 (结构一致)
    5. Per-archetype 拆解
    6. Kill switch 统计 (如提供)
    7. Per-symbol 交易数分布
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════
# 统计计算
# ═════════════════════════════════════════════════════════════════════════


def _sharpe(pnl_r: np.ndarray) -> float:
    if len(pnl_r) < 2:
        return 0.0
    s = float(np.std(pnl_r, ddof=1))
    return float(np.mean(pnl_r) / s) if s > 1e-8 else 0.0


def _win_rate(pnl_r: np.ndarray) -> float:
    if len(pnl_r) == 0:
        return 0.0
    return float(np.mean(pnl_r > 0))


def _compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """从 trades CSV 计算核心统计指标"""
    pnl = df["pnl_r"].values.astype(float)
    stats: Dict[str, Any] = {
        "n_trades": len(df),
        "sharpe": _sharpe(pnl),
        "win_rate": _win_rate(pnl),
        "mean_r": float(np.mean(pnl)) if len(pnl) > 0 else 0.0,
        "median_r": float(np.median(pnl)) if len(pnl) > 0 else 0.0,
        "max_r": float(np.max(pnl)) if len(pnl) > 0 else 0.0,
        "min_r": float(np.min(pnl)) if len(pnl) > 0 else 0.0,
    }

    # 出场原因分布
    if "exit_reason" in df.columns:
        reason_counts = df["exit_reason"].value_counts(normalize=True)
        stats["exit_reasons"] = reason_counts.to_dict()

    # Per-symbol 统计
    if "symbol" in df.columns:
        sym_counts = df.groupby("symbol").size().to_dict()
        stats["per_symbol_trades"] = sym_counts
        sym_sharpes = {}
        for sym, grp in df.groupby("symbol"):
            sym_sharpes[sym] = _sharpe(grp["pnl_r"].values.astype(float))
        stats["per_symbol_sharpe"] = sym_sharpes

    # Per-archetype 统计
    arch_col = "archetype" if "archetype" in df.columns else None
    if arch_col:
        stats["per_archetype"] = {}
        for arch, grp in df.groupby(arch_col):
            if not arch or str(arch).strip() == "":
                continue
            pnl_a = grp["pnl_r"].values.astype(float)
            stats["per_archetype"][str(arch)] = {
                "n_trades": len(grp),
                "sharpe": _sharpe(pnl_a),
                "win_rate": _win_rate(pnl_a),
                "mean_r": float(np.mean(pnl_a)),
            }

    # Per-side 统计
    if "side" in df.columns:
        stats["per_side"] = {}
        for side, grp in df.groupby("side"):
            pnl_s = grp["pnl_r"].values.astype(float)
            stats["per_side"][str(side)] = {
                "n_trades": len(grp),
                "sharpe": _sharpe(pnl_s),
                "win_rate": _win_rate(pnl_s),
            }

    return stats


# ═════════════════════════════════════════════════════════════════════════
# 一致性检查
# ═════════════════════════════════════════════════════════════════════════


def _check_consistency(
    v_stats: Dict[str, Any],
    e_stats: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """对比两侧统计，返回逐项检查结果"""
    t = thresholds or {}
    max_trade_diff = t.get("max_trade_diff_pct", 0.10)  # 10%
    max_sharpe_diff = t.get("max_sharpe_diff", 0.30)  # 收紧: 0.5 → 0.3
    max_winrate_diff = t.get("max_winrate_diff_pp", 0.05)  # 5pp
    max_mean_r_diff = t.get("max_mean_r_diff", 0.05)  # 新增: mean_r 偏差 < 0.05R

    checks: Dict[str, Dict[str, Any]] = {}

    # 1. 交易数
    vn, en = v_stats["n_trades"], e_stats["n_trades"]
    denom = max(vn, en, 1)
    diff_pct = abs(vn - en) / denom
    checks["trade_count"] = {
        "vector": vn,
        "event": en,
        "diff_pct": diff_pct,
        "threshold": max_trade_diff,
        "pass": diff_pct <= max_trade_diff,
    }

    # 2. Sharpe
    vs, es = v_stats["sharpe"], e_stats["sharpe"]
    sd = abs(vs - es)
    checks["sharpe"] = {
        "vector": round(vs, 4),
        "event": round(es, 4),
        "diff": round(sd, 4),
        "threshold": max_sharpe_diff,
        "pass": sd <= max_sharpe_diff,
    }

    # 3. 胜率
    vw, ew = v_stats["win_rate"], e_stats["win_rate"]
    wd = abs(vw - ew)
    checks["win_rate"] = {
        "vector": f"{vw:.1%}",
        "event": f"{ew:.1%}",
        "diff_pp": f"{wd:.1%}",
        "threshold": f"{max_winrate_diff:.0%}",
        "pass": wd <= max_winrate_diff,
    }

    # 4. Mean R (带阈值检查)
    vm, em = v_stats["mean_r"], e_stats["mean_r"]
    mean_r_diff = abs(vm - em)
    checks["mean_r"] = {
        "vector": round(vm, 4),
        "event": round(em, 4),
        "diff": round(mean_r_diff, 4),
        "threshold": max_mean_r_diff,
        "pass": mean_r_diff <= max_mean_r_diff,
    }

    # 5. 出场原因分布
    v_reasons = v_stats.get("exit_reasons", {})
    e_reasons = e_stats.get("exit_reasons", {})
    all_reasons = set(v_reasons.keys()) | set(e_reasons.keys())
    reason_diffs = {}
    max_reason_diff = 0.0
    for reason in sorted(all_reasons):
        vr = v_reasons.get(reason, 0.0)
        er = e_reasons.get(reason, 0.0)
        d = abs(vr - er)
        reason_diffs[reason] = {
            "vector": f"{vr:.1%}",
            "event": f"{er:.1%}",
            "diff": f"{d:.1%}",
        }
        max_reason_diff = max(max_reason_diff, d)
    checks["exit_reasons"] = {
        "details": reason_diffs,
        "max_diff": f"{max_reason_diff:.1%}",
        "pass": max_reason_diff < 0.15,  # 任一原因占比偏差 < 15pp
    }

    # 6. Per-archetype
    v_arch = v_stats.get("per_archetype", {})
    e_arch = e_stats.get("per_archetype", {})
    all_archs = set(v_arch.keys()) | set(e_arch.keys())
    arch_checks = {}
    for arch in sorted(all_archs):
        va = v_arch.get(arch, {})
        ea = e_arch.get(arch, {})
        arch_checks[arch] = {
            "trades": f"{va.get('n_trades', 0)} vs {ea.get('n_trades', 0)}",
            "sharpe": f"{va.get('sharpe', 0):.4f} vs {ea.get('sharpe', 0):.4f}",
            "win_rate": f"{va.get('win_rate', 0):.1%} vs {ea.get('win_rate', 0):.1%}",
            "mean_r": f"{va.get('mean_r', 0):.4f} vs {ea.get('mean_r', 0):.4f}",
        }
    checks["per_archetype"] = arch_checks

    # 7. Per-symbol trades 分布
    v_sym = v_stats.get("per_symbol_trades", {})
    e_sym = e_stats.get("per_symbol_trades", {})
    all_syms = set(v_sym.keys()) | set(e_sym.keys())
    sym_diffs = {}
    for sym in sorted(all_syms):
        vsc = v_sym.get(sym, 0)
        esc = e_sym.get(sym, 0)
        d = abs(vsc - esc)
        sym_diffs[sym] = {"vector": vsc, "event": esc, "diff": d}
    checks["per_symbol"] = sym_diffs

    return checks


# ═════════════════════════════════════════════════════════════════════════
# 报告输出
# ═════════════════════════════════════════════════════════════════════════


def _print_report(checks: Dict[str, Dict[str, Any]]) -> int:
    """打印一致性报告，返回 fail 数量"""
    print("\n" + "=" * 80)
    print("   向量回测 vs 事件回测 — 一致性对比报告")
    print("=" * 80)

    fail_count = 0

    # 1. 核心指标
    print("\n┌─ 1. 核心指标对比 ─────────────────────────────────────────┐")
    for key in ["trade_count", "sharpe", "win_rate", "mean_r"]:
        c = checks.get(key, {})
        passed = c.get("pass", True)
        icon = "✅" if passed else "❌"
        if not passed:
            fail_count += 1

        if key == "trade_count":
            print(
                f"  {icon} 交易数:  向量 {c['vector']}  |  事件 {c['event']}  (偏差 {c['diff_pct']:.1%}, 阈值 {c['threshold']:.0%})"
            )
        elif key == "sharpe":
            print(
                f"  {icon} Sharpe:  向量 {c['vector']}  |  事件 {c['event']}  (偏差 {c['diff']}, 阈值 {c['threshold']})"
            )
        elif key == "win_rate":
            print(
                f"  {icon} 胜率:    向量 {c['vector']}  |  事件 {c['event']}  (偏差 {c['diff_pp']}, 阈值 {c['threshold']})"
            )
        elif key == "mean_r":
            threshold_str = f", 阈值 {c['threshold']}R" if "threshold" in c else ""
            print(
                f"  {icon} Mean R:  向量 {c['vector']}  |  事件 {c['event']}  (偏差 {c['diff']}{threshold_str})"
            )
    print("└──────────────────────────────────────────────────────────┘")

    # 2. 出场原因分布
    exit_c = checks.get("exit_reasons", {})
    exit_pass = exit_c.get("pass", True)
    icon = "✅" if exit_pass else "❌"
    if not exit_pass:
        fail_count += 1
    print(
        f"\n┌─ 2. 出场原因分布 {icon} (max偏差 {exit_c.get('max_diff', '?')}) ──────────────────┐"
    )
    details = exit_c.get("details", {})
    print(f"  {'原因':<20} {'向量':>8} {'事件':>8} {'偏差':>8}")
    print(f"  {'-' * 48}")
    for reason, rd in details.items():
        print(f"  {reason:<20} {rd['vector']:>8} {rd['event']:>8} {rd['diff']:>8}")
    print("└──────────────────────────────────────────────────────────┘")

    # 3. Per-archetype
    arch_c = checks.get("per_archetype", {})
    if arch_c:
        print(f"\n┌─ 3. Per-Archetype 拆解 ──────────────────────────────────┐")
        for arch, ac in arch_c.items():
            print(f"  📊 {arch}:")
            print(f"     Trades:  {ac['trades']}")
            print(f"     Sharpe:  {ac['sharpe']}")
            print(f"     Win%:    {ac['win_rate']}")
            print(f"     Mean R:  {ac['mean_r']}")
        print("└──────────────────────────────────────────────────────────┘")

    # 4. Per-symbol
    sym_c = checks.get("per_symbol", {})
    if sym_c:
        print(f"\n┌─ 4. Per-Symbol 交易数 ───────────────────────────────────┐")
        print(f"  {'Symbol':<14} {'向量':>8} {'事件':>8} {'偏差':>6}")
        print(f"  {'-' * 40}")
        for sym, sd in sym_c.items():
            diff_mark = (
                "⚠️" if sd["diff"] > max(sd["vector"], sd["event"], 1) * 0.2 else "  "
            )
            print(
                f"  {sym:<14} {sd['vector']:>8} {sd['event']:>8} {sd['diff']:>6} {diff_mark}"
            )
        print("└──────────────────────────────────────────────────────────┘")

    # 5. 总结
    total_checks = 0
    for key in ["trade_count", "sharpe", "win_rate", "mean_r"]:
        if "pass" in checks.get(key, {}):
            total_checks += 1
    if "pass" in exit_c:
        total_checks += 1

    pass_count = total_checks - fail_count
    print(f"\n{'=' * 80}")
    if fail_count == 0:
        print(f"  ✅ 一致性通过: {pass_count}/{total_checks} 项检查全部 PASS")
    else:
        print(f"  ❌ 一致性未通过: {pass_count}/{total_checks} PASS, {fail_count} FAIL")
    print(f"{'=' * 80}\n")

    return fail_count


# ═════════════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="向量回测 vs 事件回测一致性对比")
    parser.add_argument(
        "--vector",
        "-v",
        required=True,
        help="向量回测 trades CSV (backtest_execution_layer.py --export-trades 输出)",
    )
    parser.add_argument(
        "--event",
        "-e",
        required=True,
        help="事件回测 trades CSV (event_backtest.py --export 输出)",
    )
    parser.add_argument(
        "--trade-diff-pct",
        type=float,
        default=0.10,
        help="交易数偏差阈值 (默认 0.10 = 10%%)",
    )
    parser.add_argument(
        "--sharpe-diff",
        type=float,
        default=0.5,
        help="Sharpe 偏差阈值 (默认 0.5, 绝对值)",
    )
    parser.add_argument(
        "--winrate-diff-pp",
        type=float,
        default=0.05,
        help="胜率偏差阈值 (默认 0.05 = 5pp)",
    )
    args = parser.parse_args()

    # 读取数据
    v_path = Path(args.vector)
    e_path = Path(args.event)
    if not v_path.exists():
        print(f"❌ 向量回测文件不存在: {v_path}")
        return 1
    if not e_path.exists():
        print(f"❌ 事件回测文件不存在: {e_path}")
        return 1

    v_df = pd.read_csv(v_path)
    e_df = pd.read_csv(e_path)
    print(f"📥 向量回测: {len(v_df)} trades from {v_path}")
    print(f"📥 事件回测: {len(e_df)} trades from {e_path}")

    # 标准化列名 (事件回测用 'tier' 而非 'archetype')
    if "archetype" not in e_df.columns and "tier" in e_df.columns:
        e_df["archetype"] = e_df["tier"]

    # 计算统计
    v_stats = _compute_stats(v_df)
    e_stats = _compute_stats(e_df)

    # 一致性检查
    thresholds = {
        "max_trade_diff_pct": args.trade_diff_pct,
        "max_sharpe_diff": args.sharpe_diff,
        "max_winrate_diff_pp": args.winrate_diff_pp,
    }
    checks = _check_consistency(v_stats, e_stats, thresholds)

    # 打印报告
    fail_count = _print_report(checks)

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
