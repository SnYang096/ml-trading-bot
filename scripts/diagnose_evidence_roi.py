#!/usr/bin/env python3
"""
Evidence ROI 快速诊断 — 回答两个关键问题:

  Q1: Evidence score 是否真的能区分信号好坏？
      → 按 evidence 四分位分层，看 Mean R / Win% / Sharpe 是否单调递增
      → 如果不单调 → evidence 模型本身需要修

  Q2: 如果用 evidence 做 slot 竞争排序，Sharpe 能提升多少？
      → 模拟 "先到先得" vs "evidence 竞争" 两种 slot 分配策略
      → 对比 Sharpe 差异 → 量化 P1 的预期收益

用法:
  # 三策略合并诊断 (推荐)
  python scripts/diagnose_evidence_roi.py \
    --logs bpc:results/.../bpc/logs_gated.parquet \
          fer:results/.../fer/logs_gated.parquet \
          me:results/.../me/logs_gated.parquet

  # 单策略
  python scripts/diagnose_evidence_roi.py \
    --logs results/.../bpc/logs_gated.parquet --strategy bpc
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest_execution_layer import (
    load_evidence_config,
    compute_evidence_quantiles,
    compute_evidence_scores,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. 数据加载 + Evidence Score 动态计算
# ═══════════════════════════════════════════════════════════════════════════


def _compute_evidence_for_strategy(
    df: pd.DataFrame, strategy: str, strategies_root: str = "config/strategies"
) -> pd.Series:
    """从 evidence.yaml + 原始特征列计算 evidence score

    与 backtest_execution_layer.py 完全一致的逻辑:
    1. 加载 evidence.yaml
    2. 用前 60% 数据作为校准集计算 quantile 阈值
    3. 对全量数据计算 evidence score
    """
    ev_cfg = load_evidence_config(strategy, strategies_root)
    ev_list = ev_cfg.get("evidence", [])
    if not ev_list:
        print(f"    ⚠️  {strategy}: evidence.yaml 无 evidence 规则")
        return pd.Series(np.nan, index=df.index)

    # 用前 60% 作为校准集 (避免 look-ahead)
    n_calib = max(50, int(len(df) * 0.6))
    calib_df = df.iloc[:n_calib]
    quantiles = compute_evidence_quantiles(calib_df, ev_cfg, silent=True)

    if not quantiles:
        print(f"    ⚠️  {strategy}: 无法计算 quantile (特征缺失或数据不足)")
        return pd.Series(np.nan, index=df.index)

    scores = compute_evidence_scores(
        df, ev_cfg, precomputed_quantiles=quantiles, silent=True
    )
    n_valid = scores.notna().sum()
    n_non_default = (scores != 0.5).sum()
    print(
        f"    📊 {strategy}: evidence 计算完成, mean={scores.mean():.3f}, "
        f"非默认值比例={n_non_default}/{len(scores)} ({n_non_default/len(scores)*100:.1f}%)"
    )
    return scores


def _filter_gate_and_direction(df: pd.DataFrame) -> pd.DataFrame:
    """先 gate 过滤 + 方向过滤 (与实盘一致: Direction → Gate → Evidence)"""
    n_before = len(df)

    # gate filter
    if "gate_decision" in df.columns:
        mask = df["gate_decision"].astype(str).str.lower().isin(["allow", "1", "true"])
        df = df[mask].copy()

    # direction filter — evidence 只对有方向的信号计算
    dir_col = None
    for col in ["entry_direction", "bpc_breakout_direction", "me_delta_net_flow"]:
        if col in df.columns:
            dir_col = col
            break
    if dir_col:
        df = df[df[dir_col] != 0].copy()

    return df


def load_logs(specs: List[str], default_strategy: str = None) -> pd.DataFrame:
    """加载 logs_gated.parquet，先 gate+方向过滤，再算 evidence (与实盘一致)"""
    frames = []
    for spec in specs:
        if ":" in spec:
            strategy, path = spec.split(":", 1)
        else:
            strategy = default_strategy or Path(spec).parent.name
            path = spec

        p = Path(path)
        if not p.exists():
            print(f"❌ File not found: {path}")
            continue

        df = pd.read_parquet(p)
        df["_strategy"] = strategy
        n_raw = len(df)

        # 1. 先 gate + 方向过滤 (与实盘一致)
        df = _filter_gate_and_direction(df)
        print(f"  ✅ {strategy}: {n_raw} rows → {len(df)} gate-allowed+有方向")

        # 2. 再在过滤后的数据上计算 evidence score
        df["evidence_score"] = _compute_evidence_for_strategy(df, strategy)

        frames.append(df)

    if not frames:
        print("❌ No data loaded")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def prepare_signals(df: pd.DataFrame) -> pd.DataFrame:
    """准备分析所需的标准列 (数据已在 load_logs 中完成 gate 过滤)"""
    # R-multiple
    r_col = None
    for col in ["pnl_r", "label", "forward_rr", "return"]:
        if col in df.columns:
            r_col = col
            break
    if r_col is None:
        print("❌ No R-multiple column found (tried: pnl_r, label, forward_rr, return)")
        sys.exit(1)
    df["_r"] = df[r_col].astype(float)

    # Evidence score (已在 load_logs 中计算)
    if "evidence_score" in df.columns:
        df["_ev"] = df["evidence_score"].astype(float)
    else:
        df["_ev"] = np.nan

    # Symbol
    sym_col = None
    for col in ["_symbol", "symbol"]:
        if col in df.columns:
            sym_col = col
            break
    if sym_col:
        df["_sym"] = df[sym_col]
    else:
        df["_sym"] = "UNKNOWN"

    # Timestamp
    ts_col = None
    for col in ["timestamp", "date", "datetime"]:
        if col in df.columns:
            ts_col = col
            break
    if ts_col:
        df["_ts"] = pd.to_datetime(df[ts_col])
    elif df.index.dtype == "datetime64[ns]" or "datetime" in str(df.index.dtype):
        df["_ts"] = df.index
    else:
        df["_ts"] = range(len(df))

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Q1: Evidence Predictiveness — 分位数分层分析
# ═══════════════════════════════════════════════════════════════════════════


def diagnose_evidence_predictiveness(df: pd.DataFrame):
    """按 evidence score 四分位分层，检查是否单调递增"""
    print(f"\n{'='*70}")
    print(f"  🔬 Q1: Evidence Score 能否区分信号好坏？")
    print(f"{'='*70}")

    has_ev = df["_ev"].notna().sum()
    total = len(df)
    print(f"  有 evidence score 的信号: {has_ev}/{total} ({has_ev/total*100:.1f}%)")

    if has_ev < 20:
        print(f"  ⚠️  Evidence 数据不足，无法分析")
        print(f"  → logs_gated.parquet 可能没有 evidence_score 列")
        print(f"  → 需要检查 pipeline Step 5-6 是否正常输出 evidence")
        return None

    valid = df.dropna(subset=["_ev", "_r"])
    print(f"  有效样本: {len(valid)}")
    print(
        f"  Evidence 分布: min={valid['_ev'].min():.3f}  "
        f"median={valid['_ev'].median():.3f}  max={valid['_ev'].max():.3f}"
    )

    # 四分位分层
    quartile_edges = valid["_ev"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    print(f"  分位边界: {[f'{x:.3f}' for x in quartile_edges]}")

    print(
        f"\n  {'分层':<20} {'N':<6} {'Mean R':<10} {'Win%':<8} {'Sharpe(R)':<10} {'Total R':<10}"
    )
    print(f"  {'-'*68}")

    layer_stats = []
    for i in range(4):
        lo = quartile_edges[i]
        hi = quartile_edges[i + 1]
        if i < 3:
            mask = (valid["_ev"] >= lo) & (valid["_ev"] < hi)
        else:
            mask = (valid["_ev"] >= lo) & (valid["_ev"] <= hi)
        sub = valid[mask]
        if len(sub) == 0:
            continue

        mean_r = sub["_r"].mean()
        win_pct = (sub["_r"] > 0).mean()
        sharpe = mean_r / sub["_r"].std() if sub["_r"].std() > 0 else 0
        total_r = sub["_r"].sum()

        label = f"Q{i+1} [{lo:.3f},{hi:.3f}]"
        print(
            f"  {label:<20} {len(sub):<6} {mean_r:>8.4f}  {win_pct:>6.1%}  {sharpe:>8.4f}  {total_r:>9.2f}"
        )

        layer_stats.append(
            {
                "quartile": i + 1,
                "n": len(sub),
                "mean_r": mean_r,
                "win_pct": win_pct,
                "sharpe": sharpe,
                "total_r": total_r,
            }
        )

    # 单调性检查
    if len(layer_stats) >= 3:
        mean_rs = [s["mean_r"] for s in layer_stats]
        is_monotonic = all(
            mean_rs[i] <= mean_rs[i + 1] for i in range(len(mean_rs) - 1)
        )
        q4_vs_q1 = mean_rs[-1] - mean_rs[0]

        print(f"\n  📊 诊断结果:")
        if is_monotonic and q4_vs_q1 > 0.05:
            print(f"  ✅ Evidence 有预测力! Q4-Q1 Mean R 差异: {q4_vs_q1:+.4f}")
            print(f"     → 高分信号显著好于低分信号")
            print(f"     → Slot 竞争排序 (P1) 有 ROI!")
        elif q4_vs_q1 > 0.02:
            print(f"  ⚠️  Evidence 有弱预测力. Q4-Q1 差异: {q4_vs_q1:+.4f}")
            print(f"     → 趋势正确但区分度弱")
            print(f"     → P1 可能有微小 ROI，但 evidence 模型值得改进")
        else:
            print(f"  ❌ Evidence 无预测力! Q4-Q1 差异: {q4_vs_q1:+.4f}")
            print(f"     → 高分信号并不比低分信号好")
            print(f"     → Slot 竞争排序无意义，需先修 evidence 模型")

    return layer_stats


# ═══════════════════════════════════════════════════════════════════════════
# 3. Q2: Slot 竞争模拟 — FCFS vs Evidence Ranking
# ═══════════════════════════════════════════════════════════════════════════


def simulate_slot_competition(df: pd.DataFrame, capacity_limit: int = 3):
    """模拟两种 slot 分配策略的 Sharpe 差异"""
    print(f"\n{'='*70}")
    print(f"  🎰 Q2: Evidence Slot 竞争 vs 先到先得 (capacity_limit={capacity_limit})")
    print(f"{'='*70}")

    has_ev = df["_ev"].notna().sum()
    if has_ev < 20:
        print(f"  ⚠️  Evidence 数据不足，跳过 slot 模拟")
        return None

    valid = df.dropna(subset=["_ev", "_r", "_ts"]).sort_values("_ts")

    # 按时间分组（同一 bar/时间窗口的信号需要竞争 slot）
    # 使用时间戳分组，同一时间的多个信号 = 竞争者
    if valid["_ts"].dtype == "int64":
        valid["_group"] = valid["_ts"]
    else:
        # 按 4H 窗口分组
        valid["_group"] = valid["_ts"].dt.floor("4h")

    results = {}
    for mode in ["fcfs", "evidence_rank", "random"]:
        selected_r = []
        for _, group in valid.groupby("_group"):
            if len(group) <= capacity_limit:
                # slot 够用，全部通过
                selected_r.extend(group["_r"].tolist())
            else:
                # slot 不够，需要选择
                if mode == "fcfs":
                    # 先到先得（按原始顺序取前 capacity_limit）
                    chosen = group.head(capacity_limit)
                elif mode == "evidence_rank":
                    # 按 evidence score 排序取 top
                    chosen = group.nlargest(capacity_limit, "_ev")
                elif mode == "random":
                    # 随机选择 (baseline)
                    chosen = group.sample(
                        min(capacity_limit, len(group)), random_state=42
                    )
                selected_r.extend(chosen["_r"].tolist())

        r_arr = np.array(selected_r)
        mean_r = r_arr.mean() if len(r_arr) > 0 else 0
        win_pct = (r_arr > 0).mean() if len(r_arr) > 0 else 0
        sharpe = mean_r / r_arr.std() if len(r_arr) > 1 and r_arr.std() > 0 else 0
        total_r = r_arr.sum()

        results[mode] = {
            "trades": len(r_arr),
            "mean_r": mean_r,
            "win_pct": win_pct,
            "sharpe": sharpe,
            "total_r": total_r,
        }

    print(
        f"\n  {'策略':<20} {'Trades':<8} {'Mean R':<10} {'Win%':<8} {'Sharpe(R)':<10} {'Total R':<10}"
    )
    print(f"  {'-'*70}")
    for mode, stats in results.items():
        label = {
            "fcfs": "先到先得 (现状)",
            "evidence_rank": "Evidence竞争 (P1)",
            "random": "随机选择 (baseline)",
        }[mode]
        print(
            f"  {label:<20} {stats['trades']:<8} {stats['mean_r']:>8.4f}  "
            f"{stats['win_pct']:>6.1%}  {stats['sharpe']:>8.4f}  {stats['total_r']:>9.2f}"
        )

    # ROI 评估
    ev_sharpe = results["evidence_rank"]["sharpe"]
    fc_sharpe = results["fcfs"]["sharpe"]
    ev_total_r = results["evidence_rank"]["total_r"]
    fc_total_r = results["fcfs"]["total_r"]
    sharpe_delta = ev_sharpe - fc_sharpe

    # 用 Total R 改善比例衡量 ROI (避免负 Sharpe 除法问题)
    if abs(fc_total_r) > 1e-6:
        total_r_improvement = (ev_total_r - fc_total_r) / abs(fc_total_r) * 100
    else:
        total_r_improvement = 0

    print(f"\n  📊 模拟结果:")
    print(f"  Sharpe 变化: {fc_sharpe:.4f} → {ev_sharpe:.4f} (Δ={sharpe_delta:+.4f})")
    print(
        f"  Total R 变化: {fc_total_r:.1f} → {ev_total_r:.1f} ({total_r_improvement:+.1f}%)"
    )
    has_slot_roi = sharpe_delta > 0.005 or total_r_improvement > 10
    if has_slot_roi and sharpe_delta > 0.01:
        print(f"  ✅ Evidence slot 竞争有显著 ROI! → 推荐实施 P1")
    elif has_slot_roi:
        print(f"  ⚠️  有微弱 ROI (Sharpe Δ={sharpe_delta:+.4f}) → 可以做，但优先级不高")
    else:
        print(f"  ❌ 无 ROI → Evidence 模型需先改进")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. Q3: Per-Strategy Evidence 分析
# ═══════════════════════════════════════════════════════════════════════════


def per_strategy_analysis(df: pd.DataFrame):
    """每个策略独立的 evidence 分析"""
    strategies = df["_strategy"].unique()
    if len(strategies) <= 1:
        return

    print(f"\n{'='*70}")
    print(f"  📋 Q3: Per-Strategy Evidence 分析")
    print(f"{'='*70}")

    for strat in sorted(strategies):
        sub = df[df["_strategy"] == strat]
        has_ev = sub["_ev"].notna().sum()
        has_r = sub["_r"].notna().sum()
        valid = sub.dropna(subset=["_ev", "_r"])

        if len(valid) < 10:
            print(f"\n  {strat.upper()}: 有效样本 {len(valid)} < 10, 跳过")
            continue

        # Evidence-R 相关性
        corr = valid["_ev"].corr(valid["_r"])

        # Q4 vs Q1
        q1_mask = valid["_ev"] <= valid["_ev"].quantile(0.25)
        q4_mask = valid["_ev"] >= valid["_ev"].quantile(0.75)
        q1_mean = valid.loc[q1_mask, "_r"].mean()
        q4_mean = valid.loc[q4_mask, "_r"].mean()

        emoji = (
            "✅" if q4_mean > q1_mean + 0.02 else ("⚠️" if q4_mean > q1_mean else "❌")
        )
        print(
            f"\n  {strat.upper()}: N={len(valid)}  "
            f"Corr(ev,R)={corr:+.3f}  "
            f"Q1_MeanR={q1_mean:.4f}  Q4_MeanR={q4_mean:.4f}  "
            f"Δ={q4_mean-q1_mean:+.4f} {emoji}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Evidence ROI 快速诊断",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 三策略诊断
  python scripts/diagnose_evidence_roi.py \\
    --logs bpc:results/.../bpc/logs_gated.parquet \\
          fer:results/.../fer/logs_gated.parquet \\
          me:results/.../me/logs_gated.parquet

  # 单策略
  python scripts/diagnose_evidence_roi.py \\
    --logs results/.../bpc/logs_gated.parquet --strategy bpc
        """,
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="strategy:path 格式，或单个 path (需配合 --strategy)",
    )
    parser.add_argument("--strategy", default=None, help="默认策略名")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  Evidence ROI 快速诊断")
    print(f"{'='*70}")

    # 加载数据
    df = load_logs(args.logs, args.strategy)
    df = prepare_signals(df)
    print(f"\n  有效信号数: {len(df)}")

    # Q1: Evidence Predictiveness
    layer_stats = diagnose_evidence_predictiveness(df)

    # Q2: Slot Competition Simulation
    slot_results = simulate_slot_competition(df, capacity_limit=3)

    # Q3: Per-Strategy
    per_strategy_analysis(df)

    # 综合建议
    print(f"\n{'='*70}")
    print(f"  💡 综合建议")
    print(f"{'='*70}")

    ev_has_power = False
    slot_has_roi = False
    if layer_stats and len(layer_stats) >= 3:
        q4_q1 = layer_stats[-1]["mean_r"] - layer_stats[0]["mean_r"]
        ev_has_power = q4_q1 > 0.02

    if slot_results:
        ev_sharpe = slot_results.get("evidence_rank", {}).get("sharpe", 0)
        fc_sharpe = slot_results.get("fcfs", {}).get("sharpe", 0)
        ev_total_r = slot_results.get("evidence_rank", {}).get("total_r", 0)
        fc_total_r = slot_results.get("fcfs", {}).get("total_r", 0)
        sharpe_delta = ev_sharpe - fc_sharpe
        total_r_imp = (
            (ev_total_r - fc_total_r) / abs(fc_total_r) * 100
            if abs(fc_total_r) > 1e-6
            else 0
        )
        slot_has_roi = sharpe_delta > 0.005 or total_r_imp > 10

    if ev_has_power and slot_has_roi:
        print(f"  ✅ 推荐路径 A: 实施 P1 (Evidence → Slot 竞争排序)")
        print(f"     Evidence 有预测力 + Slot 竞争有 ROI")
        print(f"     改动点: live_pcm.py → slot 满时按 evidence 排序替换")
    elif ev_has_power and not slot_has_roi:
        print(f"  ⚠️  推荐路径 B: Evidence → 连续仓位缩放")
        print(f"     Evidence 有预测力但 slot 竞争 ROI 不大")
        print(f"     说明瓶颈不在 slot 选择，而在执行层")
        print(f"     改动点: size_multiplier = f(evidence_score)")
    elif not ev_has_power:
        print(f"  ❌ 推荐路径 C: 先修 Evidence 模型")
        print(f"     当前 Evidence 无法区分信号好坏")
        print(f"     可能原因: quantile 校准过期 / 特征选择不当 / 评分权重需调优")
        print(f"     改动点: Step 5-6 Evidence 训练 + 评估")

    print()


if __name__ == "__main__":
    main()
