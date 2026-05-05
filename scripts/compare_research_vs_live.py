#!/usr/bin/env python3
"""
研究路径 vs 实盘路径 — 全面对比脚本

对比项目:
  1. 特征级: 逐列对比重叠区间内的特征值
  2. 信号级: direction → gate → entry_filter → evidence → tier 逐步对比

研究路径: predictions.parquet (FeatureStore 批量计算)
实盘路径: IncrementalFeatureComputer (从 bars/ticks 增量计算)

用法:
    python scripts/compare_research_vs_live.py
    python scripts/compare_research_vs_live.py --symbol BTCUSDT --last-n 50
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_data_stream.feature_storage import StorageManager
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy


# ================================================================
# 辅助函数
# ================================================================


def row_to_features(row: pd.Series) -> Dict[str, float]:
    features = {}
    for k, v in row.items():
        try:
            if v is not None and np.isscalar(v) and not pd.isna(v):
                features[str(k)] = float(v)
        except (ValueError, TypeError):
            continue
    return features


def compare_series(
    research: pd.Series,
    live: pd.Series,
    name: str,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """对比两个 Series，返回统计信息"""
    both_valid = research.notna() & live.notna()
    n_valid = int(both_valid.sum())
    if n_valid == 0:
        return {"name": name, "n_valid": 0, "match_pct": 0, "note": "no overlap"}

    r = research[both_valid].values.astype(float)
    l = live[both_valid].values.astype(float)
    diff = np.abs(r - l)
    match_mask = diff <= tolerance
    match_pct = float(match_mask.mean() * 100)

    return {
        "name": name,
        "n_valid": n_valid,
        "match_pct": match_pct,
        "avg_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "r_mean": float(r.mean()),
        "l_mean": float(l.mean()),
        "r_std": float(r.std()),
        "l_std": float(l.std()),
    }


def print_feature_table(results: List[Dict], title: str):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(
        f"  {'Feature':<35} {'N':>5} {'Match%':>8} {'AvgDiff':>10} "
        f"{'MaxDiff':>10} {'R_mean':>9} {'L_mean':>9}"
    )
    print(f"  {'-'*87}")
    for r in sorted(results, key=lambda x: x.get("match_pct", 0)):
        if r["n_valid"] == 0:
            print(f"  {r['name']:<35} {'---':>5} {'N/A':>8}  {r.get('note','')}")
            continue
        print(
            f"  {r['name']:<35} {r['n_valid']:>5} {r['match_pct']:>7.1f}% "
            f"{r['avg_diff']:>10.6f} {r['max_diff']:>10.6f} "
            f"{r['r_mean']:>9.4f} {r['l_mean']:>9.4f}"
        )


# ================================================================
# 主逻辑
# ================================================================


def main():
    p = argparse.ArgumentParser(description="研究 vs 实盘 全面对比")
    p.add_argument(
        "--predictions",
        default="results/train_final_20260208_220616_return_tree/bpc/predictions.parquet",
        help="研究路径 predictions.parquet",
    )
    p.add_argument("--live-root", default="live/highcap", help="实盘数据根目录")
    p.add_argument(
        "--strategies-root",
        default="config/strategies",
        help="策略配置 (用于 gate/entry_filter/evidence)",
    )
    p.add_argument("--symbol", default=None, help="只对比单个 symbol (默认全部)")
    p.add_argument("--last-n", type=int, default=0, help="只对比最后 N 个重叠 bar")
    p.add_argument("--tolerance", type=float, default=1e-4, help="数值匹配容差")
    p.add_argument("--export-csv", default=None, help="导出对比结果 CSV")
    args = p.parse_args()

    print("=" * 90)
    print("  研究路径 vs 实盘路径 — 全面对比")
    print("=" * 90)

    # ── 1. 加载研究数据 ──
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"  predictions.parquet 不存在: {pred_path}")
        return 1

    research_df = pd.read_parquet(pred_path)
    print(f"\n  研究数据: {len(research_df)} 行")
    print(f"  列: {len(research_df.columns)}")

    # 处理 timestamp 和 symbol
    if "timestamp" in research_df.columns:
        research_df["timestamp"] = pd.to_datetime(research_df["timestamp"], utc=True)
    elif isinstance(research_df.index, pd.DatetimeIndex):
        research_df["timestamp"] = research_df.index
    if "_symbol" in research_df.columns and "symbol" not in research_df.columns:
        research_df["symbol"] = research_df["_symbol"]

    symbols = sorted(research_df["symbol"].unique())
    print(f"  Symbols: {symbols}")
    print(
        f"  时间: {research_df['timestamp'].min()} ~ {research_df['timestamp'].max()}"
    )

    # ── 2. 加载实盘数据并计算特征 ──
    storage = StorageManager(f"{args.live_root}/data")
    archetypes_dir = str(Path(args.strategies_root) / "bpc" / "archetypes")

    if args.symbol:
        symbols = [s for s in symbols if s == args.symbol]
    if not symbols:
        print("  没有匹配的 symbol")
        return 1

    # 对每个 symbol 计算实盘特征
    live_features_all = {}
    for sym in symbols:
        print(f"\n  计算实盘特征: {sym} ...")
        # 加载尽可能多的数据用于 warmup
        end_date = datetime.now().strftime("%Y-%m-%d")
        warmup_start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")

        bars = storage.bar_1min.load_range(sym, warmup_start, end_date)
        ticks = storage.ticks.load_range(sym, warmup_start, end_date)
        print(f"    bars: {len(bars)}, ticks: {len(ticks)}")

        if len(bars) == 0:
            print(f"    跳过 {sym}: 无 bars")
            continue

        fc = IncrementalFeatureComputer(
            primary_timeframe="240T",
            archetypes_dir=archetypes_dir,
        )
        features_df = fc.compute_features_dataframe(
            bars_1min=bars,
            ticks_1min=ticks,
            primary_timeframe="240T",
        )
        if features_df is not None and not features_df.empty:
            features_df.index = pd.to_datetime(features_df.index, utc=True)
            live_features_all[sym] = features_df
            print(f"    特征: {len(features_df)} 行 x {len(features_df.columns)} 列")
        else:
            print(f"    跳过 {sym}: 特征为空")

    if not live_features_all:
        print("  没有可用的实盘特征")
        return 1

    # ── 3. 构建重叠区间 ──
    all_feature_results = []
    all_signal_records = []

    # 初始化 BPC (用于信号级对比)
    bpc_research = GenericLiveStrategy(
        strategy_name="bpc",
        strategies_root=args.strategies_root,
        primary_timeframe="240T",
        bar_minutes=240,
    )

    bpc_live = GenericLiveStrategy(
        strategy_name="bpc",
        strategies_root=args.strategies_root,
        primary_timeframe="240T",
        bar_minutes=240,
    )

    for sym in symbols:
        if sym not in live_features_all:
            continue

        research_sym = research_df[research_df["symbol"] == sym].copy()
        research_sym = research_sym.set_index("timestamp").sort_index()
        live_sym = live_features_all[sym].sort_index()

        # 找重叠时间
        overlap_idx = research_sym.index.intersection(live_sym.index)
        if len(overlap_idx) == 0:
            print(f"\n  {sym}: 无时间重叠")
            continue

        if args.last_n > 0:
            overlap_idx = overlap_idx[-args.last_n :]

        print(f"\n{'='*90}")
        print(f"  {sym}: 重叠 {len(overlap_idx)} 个 bar")
        print(f"  时间: {overlap_idx.min()} ~ {overlap_idx.max()}")

        r_overlap = research_sym.loc[overlap_idx]
        l_overlap = live_sym.loc[overlap_idx]

        # ── 3a. 特征级对比 ──
        # 找共同的 BPC 相关特征列
        bpc_features = [
            "close",
            "high",
            "low",
            "open",
            "atr",
            "bpc_breakout_direction",
            "bpc_bb_compression",
            "bpc_score_breakout",
            "bpc_score_pullback",
            "bpc_recent_breakout_strength",
            "bpc_was_in_pullback",
            "bpc_pullback_depth",
            "bpc_direction_confidence",
            "bpc_phase",
            "bpc_volume_compression_pct",
            "bb_width_normalized_pct",
            "wpt_ignition_score",
            "wpt_exhaustion_score",
            "price_position",
            "ef_liquidity_silence",
        ]
        common_cols = [
            c for c in bpc_features if c in r_overlap.columns and c in l_overlap.columns
        ]

        sym_feature_results = []
        for col in common_cols:
            r_vals = pd.to_numeric(r_overlap[col], errors="coerce")
            l_vals = pd.to_numeric(l_overlap[col], errors="coerce")
            result = compare_series(r_vals, l_vals, col, tolerance=args.tolerance)
            result["symbol"] = sym
            sym_feature_results.append(result)
            all_feature_results.append(result)

        # 缺失特征诊断
        r_only = [
            c
            for c in bpc_features
            if c in r_overlap.columns and c not in l_overlap.columns
        ]
        l_only = [
            c
            for c in bpc_features
            if c not in r_overlap.columns and c in l_overlap.columns
        ]
        if r_only:
            print(f"    仅研究有: {r_only}")
        if l_only:
            print(f"    仅实盘有: {l_only}")

        print_feature_table(
            sym_feature_results, f"{sym} 特征对比 ({len(overlap_idx)} bars)"
        )

        # 后 50 bar 对比
        last_50_idx = overlap_idx[-min(50, len(overlap_idx)) :]
        if len(last_50_idx) >= 10:
            last50_results = []
            for col in common_cols:
                r_vals = pd.to_numeric(r_overlap.loc[last_50_idx, col], errors="coerce")
                l_vals = pd.to_numeric(l_overlap.loc[last_50_idx, col], errors="coerce")
                result = compare_series(r_vals, l_vals, col, tolerance=args.tolerance)
                last50_results.append(result)
            print_feature_table(
                last50_results, f"{sym} 特征对比 (最后 {len(last_50_idx)} bars)"
            )

        # ── 3b. 信号级对比 ──
        # 从实盘特征计算 evidence quantiles
        bpc_live.set_quantiles_from_df(live_features_all[sym])
        # 从研究特征计算 evidence quantiles (用重叠前的数据)
        research_full = research_df[research_df["symbol"] == sym].copy()
        research_full = research_full.set_index(
            pd.to_datetime(research_full["timestamp"], utc=True)
        ).sort_index()
        calib_end = overlap_idx.min()
        calib_data = research_full[research_full.index < calib_end]
        if len(calib_data) >= 50:
            bpc_research.set_quantiles_from_df(calib_data)
        else:
            bpc_research.set_quantiles_from_df(research_full)

        print(f"\n  {sym} 信号级对比:")
        print(f"  {'─'*86}")
        print(
            f"  {'Timestamp':<22} {'R_dir':>5} {'L_dir':>5} "
            f"{'R_gate':>6} {'L_gate':>6} {'R_ef':>5} {'L_ef':>5} "
            f"{'R_ev':>6} {'L_ev':>6} {'R_sig':>5} {'L_sig':>5} {'Match':>5}"
        )

        n_dir_match = 0
        n_gate_match = 0
        n_ef_match = 0
        n_signal_match = 0
        n_total = 0
        signal_diffs = []

        for ts in overlap_idx:
            r_row = r_overlap.loc[ts]
            l_row = l_overlap.loc[ts]
            r_feat = row_to_features(r_row)
            l_feat = row_to_features(l_row)

            # Direction
            r_dir = int(float(r_feat.get("bpc_breakout_direction", 0)))
            l_dir = int(float(l_feat.get("bpc_breakout_direction", 0)))

            # Gate (live path)
            r_gate = "—"
            l_gate = "—"
            if r_dir != 0 and bpc_research._archetype is not None:
                g_pass, g_reasons, g_w = bpc_research._archetype.apply_gate(
                    r_feat, bpc_research._quantiles or None
                )
                r_gate = "PASS" if g_pass else "DENY"
            if l_dir != 0 and bpc_live._archetype is not None:
                g_pass, g_reasons, g_w = bpc_live._archetype.apply_gate(
                    l_feat, bpc_live._quantiles or None
                )
                l_gate = "PASS" if g_pass else "DENY"

            # Entry filter
            r_ef = "—"
            l_ef = "—"
            if r_dir != 0 and r_gate == "PASS":
                r_should, r_info = bpc_research._evaluate_entry_signal(r_feat)
                r_ef = (
                    "PASS"
                    if r_should or r_info.get("reject_reason") != "entry_filter_deny"
                    else "DENY"
                )
                if r_should:
                    r_ef = "PASS"
                elif r_info.get("reject_reason") == "entry_filter_deny":
                    r_ef = "DENY"
            if l_dir != 0 and l_gate == "PASS":
                l_should, l_info = bpc_live._evaluate_entry_signal(l_feat)
                l_ef = (
                    "PASS"
                    if l_should or l_info.get("reject_reason") != "entry_filter_deny"
                    else "DENY"
                )
                if l_should:
                    l_ef = "PASS"
                elif l_info.get("reject_reason") == "entry_filter_deny":
                    l_ef = "DENY"

            # Evidence
            r_ev = "—"
            l_ev = "—"
            r_signal = 0
            l_signal = 0
            if r_dir != 0 and r_gate == "PASS" and r_ef == "PASS":
                r_should, r_info = bpc_research._evaluate_entry_signal(r_feat)
                r_ev = f"{r_info.get('evidence_score', 0):.2f}" if r_should else "—"
                r_signal = r_dir if r_should else 0
            if l_dir != 0 and l_gate == "PASS" and l_ef == "PASS":
                l_should, l_info = bpc_live._evaluate_entry_signal(l_feat)
                l_ev = f"{l_info.get('evidence_score', 0):.2f}" if l_should else "—"
                l_signal = l_dir if l_should else 0

            # Stats
            n_total += 1
            dir_match = r_dir == l_dir
            gate_match = r_gate == l_gate
            ef_match = r_ef == l_ef
            sig_match = r_signal == l_signal

            if dir_match:
                n_dir_match += 1
            if gate_match:
                n_gate_match += 1
            if ef_match:
                n_ef_match += 1
            if sig_match:
                n_signal_match += 1

            match_str = "YES" if sig_match else "NO"

            # Only print mismatches or signals
            if not sig_match or r_signal != 0 or l_signal != 0:
                ts_str = str(ts)[:19]
                print(
                    f"  {ts_str:<22} {r_dir:>5} {l_dir:>5} "
                    f"{r_gate:>6} {l_gate:>6} {r_ef:>5} {l_ef:>5} "
                    f"{r_ev:>6} {l_ev:>6} {r_signal:>5} {l_signal:>5} {match_str:>5}"
                )

                if not sig_match:
                    signal_diffs.append(
                        {
                            "timestamp": str(ts),
                            "symbol": sym,
                            "r_dir": r_dir,
                            "l_dir": l_dir,
                            "r_gate": r_gate,
                            "l_gate": l_gate,
                            "r_ef": r_ef,
                            "l_ef": l_ef,
                            "r_signal": r_signal,
                            "l_signal": l_signal,
                        }
                    )

            all_signal_records.append(
                {
                    "timestamp": str(ts),
                    "symbol": sym,
                    "r_dir": r_dir,
                    "l_dir": l_dir,
                    "dir_match": dir_match,
                    "r_gate": r_gate,
                    "l_gate": l_gate,
                    "gate_match": gate_match,
                    "r_ef": r_ef,
                    "l_ef": l_ef,
                    "ef_match": ef_match,
                    "r_signal": r_signal,
                    "l_signal": l_signal,
                    "signal_match": sig_match,
                }
            )

        # 信号对比汇总
        print(f"\n  {sym} 信号对比汇总:")
        print(f"    总 bar:             {n_total}")
        print(
            f"    方向一致:           {n_dir_match}/{n_total} ({n_dir_match/n_total*100:.1f}%)"
        )
        print(
            f"    Gate 一致:          {n_gate_match}/{n_total} ({n_gate_match/n_total*100:.1f}%)"
        )
        print(
            f"    Entry Filter 一致:  {n_ef_match}/{n_total} ({n_ef_match/n_total*100:.1f}%)"
        )
        print(
            f"    最终信号一致:       {n_signal_match}/{n_total} ({n_signal_match/n_total*100:.1f}%)"
        )

        if signal_diffs:
            # 分析差异来源
            diff_at_dir = sum(1 for d in signal_diffs if d["r_dir"] != d["l_dir"])
            diff_at_gate = sum(
                1
                for d in signal_diffs
                if d["r_dir"] == d["l_dir"] and d["r_gate"] != d["l_gate"]
            )
            diff_at_ef = sum(
                1
                for d in signal_diffs
                if d["r_dir"] == d["l_dir"]
                and d["r_gate"] == d["l_gate"]
                and d["r_ef"] != d["l_ef"]
            )
            print(f"\n    差异来源分析 ({len(signal_diffs)} 个不一致):")
            print(f"      方向不同:         {diff_at_dir}")
            print(f"      Gate 不同:        {diff_at_gate}")
            print(f"      Entry Filter 不同: {diff_at_ef}")

    # ── 4. 全局汇总 ──
    print(f"\n{'='*90}")
    print("  全局汇总")
    print(f"{'='*90}")

    if all_feature_results:
        # 按特征聚合
        feat_agg = defaultdict(list)
        for r in all_feature_results:
            feat_agg[r["name"]].append(r)

        agg_results = []
        for name, items in feat_agg.items():
            n_valid = sum(r["n_valid"] for r in items)
            if n_valid == 0:
                continue
            # 加权平均
            w_match = sum(r["match_pct"] * r["n_valid"] for r in items) / n_valid
            w_diff = sum(r.get("avg_diff", 0) * r["n_valid"] for r in items) / n_valid
            agg_results.append(
                {
                    "name": name,
                    "n_valid": n_valid,
                    "match_pct": w_match,
                    "avg_diff": w_diff,
                    "max_diff": max(r.get("max_diff", 0) for r in items),
                    "r_mean": sum(r.get("r_mean", 0) * r["n_valid"] for r in items)
                    / n_valid,
                    "l_mean": sum(r.get("l_mean", 0) * r["n_valid"] for r in items)
                    / n_valid,
                }
            )
        print_feature_table(agg_results, "全局特征对比 (所有 symbol 加权平均)")

    if all_signal_records:
        total = len(all_signal_records)
        dir_ok = sum(1 for r in all_signal_records if r["dir_match"])
        gate_ok = sum(1 for r in all_signal_records if r["gate_match"])
        ef_ok = sum(1 for r in all_signal_records if r["ef_match"])
        sig_ok = sum(1 for r in all_signal_records if r["signal_match"])

        print(f"\n  信号对比 (全局):")
        print(f"    总 bar:             {total}")
        print(f"    方向一致:           {dir_ok}/{total} ({dir_ok/total*100:.1f}%)")
        print(f"    Gate 一致:          {gate_ok}/{total} ({gate_ok/total*100:.1f}%)")
        print(f"    Entry Filter 一致:  {ef_ok}/{total} ({ef_ok/total*100:.1f}%)")
        print(f"    最终信号一致:       {sig_ok}/{total} ({sig_ok/total*100:.1f}%)")

        # 研究产生信号 vs 实盘产生信号
        r_signals = sum(1 for r in all_signal_records if r["r_signal"] != 0)
        l_signals = sum(1 for r in all_signal_records if r["l_signal"] != 0)
        both_signals = sum(
            1 for r in all_signal_records if r["r_signal"] != 0 and r["l_signal"] != 0
        )
        print(f"\n    研究信号数:         {r_signals}")
        print(f"    实盘信号数:         {l_signals}")
        print(f"    两者都有信号:       {both_signals}")
        if r_signals > 0:
            recall = both_signals / r_signals * 100
            print(f"    实盘召回率:         {recall:.1f}% ({both_signals}/{r_signals})")

    # ── 5. 导出 CSV ──
    if args.export_csv and all_signal_records:
        export_df = pd.DataFrame(all_signal_records)
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(export_path, index=False)
        print(f"\n  导出: {len(export_df)} 行 → {export_path}")

    # ── 6. 关键诊断建议 ──
    print(f"\n{'='*90}")
    print("  诊断建议")
    print(f"{'='*90}")

    # 检查特征差异严重的情况
    if all_feature_results:
        bad_features = [
            r
            for r in agg_results
            if r["match_pct"] < 90 and r["name"] not in ("close", "high", "low", "open")
        ]
        if bad_features:
            print("\n  特征匹配率 < 90% 的列:")
            for r in sorted(bad_features, key=lambda x: x["match_pct"]):
                print(
                    f"    {r['name']:<35} {r['match_pct']:.1f}%  (avg_diff={r['avg_diff']:.6f})"
                )
            print("\n  可能原因:")
            print(
                "    - percentile 类特征: 历史数据长度不同 (研究 18 个月 vs 实盘 6 个月)"
            )
            print("    - tick 依赖特征: 数据来源/聚合方式可能不同")
            print(
                "    - warmup 不足: 实盘 bars 从 2025-08 开始，percentile_window=540 需要 ~90 天"
            )
        else:
            print("\n  所有特征匹配率 >= 90%")

    if all_signal_records:
        if sig_ok < total:
            diff_bars = total - sig_ok
            print(f"\n  {diff_bars} 个 bar 信号不一致。")
            if dir_ok < total:
                print(f"    主要差异在 direction 层: {total - dir_ok} 个不同")
                print("    → 检查 bpc_breakout_direction 的计算 (bpc_soft_phase_f)")
        else:
            print("\n  所有信号完全一致!")

    print(f"\n{'='*90}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
