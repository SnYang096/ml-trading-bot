#!/usr/bin/env python3
"""
相同数据集对比：研究路径 vs 实盘路径

两条路径共用同一份 data/parquet_data 原始数据，消除数据差异，
验证纯代码计算的一致性。

研究路径: DataHandler → 4H bars (含 orderflow) → StrategyFeatureLoader → features
实盘路径: DataHandler → 1min bars → IncrementalFeatureComputer → 4H bars → features

用法:
    python scripts/compare_same_data.py
    python scripts/compare_same_data.py --symbol BTCUSDT --start-date 2025-02-01 --end-date 2026-01-31
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_tools.data_handler import DataHandler
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.live.live_feature_plan import (
    extract_features_from_archetypes,
)


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
    both_valid = research.notna() & live.notna()
    n_valid = int(both_valid.sum())
    if n_valid == 0:
        return {"name": name, "n_valid": 0, "match_pct": 0, "note": "no overlap"}
    r = research[both_valid].values.astype(float)
    l = live[both_valid].values.astype(float)
    diff = np.abs(r - l)
    # For relative comparison on features with large values (like close/atr)
    denom = np.maximum(np.abs(r), np.abs(l))
    denom[denom == 0] = 1.0
    rel_diff = diff / denom
    match_mask = rel_diff <= tolerance
    match_pct = float(match_mask.mean() * 100)
    return {
        "name": name,
        "n_valid": n_valid,
        "match_pct": match_pct,
        "avg_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "avg_rel_diff": float(rel_diff.mean()),
        "r_mean": float(r.mean()),
        "l_mean": float(l.mean()),
    }


def print_feature_table(results: List[Dict], title: str):
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    print(
        f"  {'Feature':<35} {'N':>5} {'Match%':>8} {'AvgDiff':>10} "
        f"{'MaxDiff':>10} {'AvgRelDiff':>10} {'R_mean':>9} {'L_mean':>9}"
    )
    print(f"  {'-'*92}")
    for r in sorted(results, key=lambda x: x.get("match_pct", 0)):
        if r["n_valid"] == 0:
            print(f"  {r['name']:<35} {'---':>5} {'N/A':>8}  {r.get('note','')}")
            continue
        print(
            f"  {r['name']:<35} {r['n_valid']:>5} {r['match_pct']:>7.1f}% "
            f"{r['avg_diff']:>10.6f} {r['max_diff']:>10.6f} "
            f"{r.get('avg_rel_diff', 0):>10.6f} "
            f"{r['r_mean']:>9.4f} {r['l_mean']:>9.4f}"
        )


# ================================================================
# 主逻辑
# ================================================================


def main():
    p = argparse.ArgumentParser(description="相同数据集对比: 研究 vs 实盘路径")
    p.add_argument(
        "--data-path", default="data/parquet_data", help="原始 tick parquet 数据目录"
    )
    p.add_argument("--symbol", default="BTCUSDT", help="交易对（单个，向后兼容）")
    p.add_argument(
        "--symbols", default=None, help="多币种对比，逗号分隔 (e.g. BTCUSDT,ETHUSDT)"
    )
    p.add_argument("--start-date", default="2025-02-01", help="起始日期")
    p.add_argument("--end-date", default="2026-01-31", help="结束日期")
    p.add_argument(
        "--strategies-root", default="config/strategies", help="策略配置目录"
    )
    p.add_argument("--tolerance", type=float, default=1e-4, help="相对匹配容差")
    p.add_argument("--skip-signals", action="store_true", help="跳过信号级对比")
    args = p.parse_args()

    # 确定币种列表
    if args.symbols:
        symbol_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbol_list = [args.symbol]

    archetypes_dir = str(Path(args.strategies_root) / "bpc" / "archetypes")
    live_feature_set, live_feature_nodes = extract_features_from_archetypes(
        archetypes_dir
    )

    # 多币种循环
    all_results = []
    for sym in symbol_list:
        args.symbol = sym
        try:
            result = compare_one_symbol(
                args, archetypes_dir, live_feature_set, live_feature_nodes
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ❌ {sym} 对比失败: {e}")
            all_results.append({"symbol": sym, "error": str(e)})

    # 多币种汇总
    if len(symbol_list) > 1:
        print(f"\n{'='*95}")
        print(f"  多币种汇总 ({len(symbol_list)} 个币种)")
        print(f"{'='*95}")
        print(
            f"  {'Symbol':<12} {'Bars':>6} {'信号一致':>10} {'完美特征':>10} {'良好':>6} {'较差':>6} {'状态':>8}"
        )
        print(f"  {'-'*60}")
        for r in all_results:
            if "error" in r:
                print(
                    f"  {r['symbol']:<12} {'ERROR':>6} {'-':>10} {'-':>10} {'-':>6} {'-':>6} {'❌':>8}"
                )
            else:
                status = "✅" if r.get("signal_match_rate", 0) == 100.0 else "⚠️"
                print(
                    f"  {r['symbol']:<12} {r.get('bars',0):>6} {r.get('signal_match_rate',0):>9.1f}% {r.get('perfect',0):>10} {r.get('good',0):>6} {r.get('bad',0):>6} {status:>8}"
                )
        all_ok = all(
            r.get("signal_match_rate", 0) == 100.0
            for r in all_results
            if "error" not in r
        )
        no_errors = all("error" not in r for r in all_results)
        if all_ok and no_errors:
            print(f"\n  🎉 全部 {len(symbol_list)} 个币种信号 100% 一致!")
        print(f"{'='*95}")

    return 0


def compare_one_symbol(args, archetypes_dir, live_feature_set, live_feature_nodes):
    print("=" * 95)
    print(f"  相同数据集对比: 研究路径 vs 实盘路径 [{args.symbol}]")
    print("=" * 95)
    print(f"  数据源: {args.data_path}")
    print(f"  Symbol: {args.symbol}")
    print(f"  时间: {args.start_date} ~ {args.end_date}")
    print()

    print(f"  BPC feature_nodes: {len(live_feature_nodes)} 个")
    print(f"  BPC feature_set: {len(live_feature_set)} 列")

    # ================================================================
    # 1. 研究路径: tick parquet → 4H bars → StrategyFeatureLoader
    # ================================================================
    print(f"\n{'─'*95}")
    print("  [1/2] 研究路径: DataHandler → 4H bars → StrategyFeatureLoader")
    print(f"{'─'*95}")
    t0 = time.time()

    dh = DataHandler(args.data_path)
    df_4h = dh.load_ohlcv(
        symbol=args.symbol,
        timeframe="240T",
        start_date=args.start_date,
        end_date=args.end_date,
    )
    df_4h.index = pd.to_datetime(df_4h.index, utc=True)
    print(f"  4H bars: {len(df_4h)} rows, {df_4h.index.min()} ~ {df_4h.index.max()}")
    print(f"  4H columns: {sorted(df_4h.columns.tolist())[:15]}...")

    # 排除 tick 依赖节点（无 tick 数据时无法计算）
    tick_dependent_nodes = {
        "trade_cluster_base_aligned_features_f",
        "vpin_base_aligned_features_f",
        "footprint_base_features_f",
    }
    feature_deps_cfg = None
    try:
        import yaml

        with open("config/feature_dependencies.yaml") as f:
            feature_deps_cfg = yaml.safe_load(f).get("features", {})
    except Exception:
        pass

    def _has_tick_dep(node_name, visited=None):
        if visited is None:
            visited = set()
        if node_name in visited:
            return False
        visited.add(node_name)
        if node_name in tick_dependent_nodes:
            return True
        if feature_deps_cfg:
            info = feature_deps_cfg.get(node_name)
            if isinstance(info, dict):
                for dep in info.get("dependencies") or []:
                    if _has_tick_dep(dep, visited):
                        return True
        return False

    research_nodes = [n for n in live_feature_nodes if not _has_tick_dep(n)]
    skipped_nodes = [n for n in live_feature_nodes if _has_tick_dep(n)]
    if skipped_nodes:
        print(f"  跳过 tick 依赖节点 (第一轮): {skipped_nodes}")

    # 将跳过节点的非-tick 依赖也加入 research_nodes，
    # 确保 load_features_from_requested 保留它们的输出列（如 atr）
    research_set = set(research_nodes)
    extra_deps = []
    for n in skipped_nodes:
        if n in tick_dependent_nodes:
            continue
        info = feature_deps_cfg.get(n) if feature_deps_cfg else None
        if isinstance(info, dict):
            for dep in info.get("dependencies") or []:
                if dep not in research_set and not _has_tick_dep(dep):
                    research_nodes.append(dep)
                    research_set.add(dep)
                    extra_deps.append(dep)
    if extra_deps:
        print(f"  额外保留依赖节点: {extra_deps}")
    print(f"  第一轮计算节点: {len(research_nodes)} 个")

    # 用 StrategyFeatureLoader 计算研究路径特征
    feature_loader = StrategyFeatureLoader(use_monthly_cache=False)
    research_df = feature_loader.load_features_from_requested(
        df_4h, requested_features=research_nodes, fit=True
    )
    research_df.index = pd.to_datetime(research_df.index, utc=True)
    t1 = time.time()
    print(
        f"  研究特征 (第一轮): {len(research_df)} rows × {len(research_df.columns)} cols [{t1-t0:.1f}s]"
    )
    print(f"  → 第二轮待实盘路径计算完成后注入 tick 特征")

    # ================================================================
    # 2. 实盘路径: tick parquet → 1min bars → IncrementalFeatureComputer
    # ================================================================
    print(f"\n{'─'*95}")
    print("  [2/2] 实盘路径: DataHandler → 1min bars → IncrementalFeatureComputer")
    print(f"{'─'*95}")
    t0 = time.time()

    dh_1min = DataHandler(args.data_path)
    df_1min = dh_1min.load_ohlcv(
        symbol=args.symbol,
        timeframe="1T",
        start_date=args.start_date,
        end_date=args.end_date,
    )
    df_1min.index = pd.to_datetime(df_1min.index, utc=True)
    print(
        f"  1min bars: {len(df_1min)} rows, {df_1min.index.min()} ~ {df_1min.index.max()}"
    )

    # Rename columns: buy_qty → buy_volume, sell_qty → sell_volume
    col_rename = {"buy_qty": "buy_volume", "sell_qty": "sell_volume"}
    df_1min = df_1min.rename(
        columns={k: v for k, v in col_rename.items() if k in df_1min.columns}
    )

    # Ensure timestamp column exists
    if "timestamp" not in df_1min.columns:
        df_1min["timestamp"] = df_1min.index

    print(f"  1min columns: {sorted(df_1min.columns.tolist())[:15]}...")

    # 加载 tick 数据（用于 VPIN / trade clustering / bpc_soft_phase）
    print("  加载 tick 数据...")
    tick_frames = []
    data_root = Path(args.data_path)
    for fp in sorted(data_root.glob(f"{args.symbol}_*.parquet")):
        df_tick = pd.read_parquet(fp)
        if "price" in df_tick.columns and "volume" in df_tick.columns:
            tick_frames.append(df_tick)
    if tick_frames:
        all_ticks = pd.concat(tick_frames, ignore_index=True)
        all_ticks["timestamp"] = pd.to_datetime(all_ticks["timestamp"], utc=True)
        start_ts = pd.to_datetime(args.start_date, utc=True)
        end_ts = pd.to_datetime(args.end_date, utc=True)
        all_ticks = all_ticks[
            (all_ticks["timestamp"] >= start_ts) & (all_ticks["timestamp"] <= end_ts)
        ]
        print(f"  ticks: {len(all_ticks)} rows")
    else:
        all_ticks = pd.DataFrame()
        print("  ticks: 0 rows (跳过 tick 依赖特征)")

    # IncrementalFeatureComputer 计算实盘路径特征
    fc = IncrementalFeatureComputer(
        primary_timeframe="240T",
        archetypes_dir=archetypes_dir,
    )
    live_df = fc.compute_features_dataframe(
        bars_1min=df_1min,
        ticks_1min=all_ticks,
        primary_timeframe="240T",
    )
    if live_df is None or live_df.empty:
        print("  ERROR: IncrementalFeatureComputer 返回空 DataFrame")
        return 1

    live_df.index = pd.to_datetime(live_df.index, utc=True)
    t1 = time.time()
    print(
        f"  实盘特征: {len(live_df)} rows × {len(live_df.columns)} cols [{t1-t0:.1f}s]"
    )

    # ================================================================
    # 2.5 研究路径第二轮: 用相同的 tick 数据计算订单流特征 + bpc_soft_phase_f
    # ================================================================
    print(f"\n{'─'*95}")
    print("  [2.5] 研究路径第二轮: 计算 tick 特征 + 跳过的节点")
    print(f"{'─'*95}")
    t0 = time.time()

    # 直接在研究 4H bars 上计算订单流特征（和 IncrementalFeatureComputer step2 相同）
    if not all_ticks.empty:
        from src.features.time_series.utils_order_flow_features import (
            extract_order_flow_features,
            compute_vpin_derived_features_from_base,
        )

        try:
            ticks_copy = all_ticks.copy()
            if not isinstance(ticks_copy.index, pd.DatetimeIndex):
                if "timestamp" in ticks_copy.columns:
                    ticks_copy.index = pd.to_datetime(ticks_copy["timestamp"], utc=True)
            if ticks_copy.index.tz is None:
                ticks_copy.index = ticks_copy.index.tz_localize("UTC")

            of_df = extract_order_flow_features(
                df_4h,
                ticks=ticks_copy,
                freq="240T",
                include_trade_clustering=True,
                compute_vpin_derived=True,
            )
            of_new = [c for c in of_df.columns if c not in research_df.columns]
            if of_new:
                research_df = research_df.join(of_df[of_new], how="left")
            print(f"  订单流特征: +{len(of_new)} 列")

            # VPIN 衍生特征
            try:
                vpin_derived = compute_vpin_derived_features_from_base(research_df)
                for c in vpin_derived.columns:
                    if c not in research_df.columns:
                        research_df[c] = vpin_derived[c]
            except Exception:
                pass
        except Exception as e:
            print(f"  订单流特征计算失败: {e}")
    else:
        print("  无 tick 数据，跳过订单流特征")

    # 第二轮: 计算之前跳过的非叶子节点
    if skipped_nodes and feature_deps_cfg:
        bar_cols_updated = set(research_df.columns)
        second_pass = []
        for n in skipped_nodes:
            if n in tick_dependent_nodes:
                continue
            info = feature_deps_cfg.get(n)
            if not isinstance(info, dict):
                continue
            # 只检查 required_columns（必须列），column_mappings 可能包含可选参数
            req_cols = set(info.get("required_columns") or [])
            if req_cols.issubset(bar_cols_updated):
                second_pass.append(n)
            else:
                missing = req_cols - bar_cols_updated
                print(f"    {n}: ✗ 仍缺少必须列: {missing}")

        if second_pass:
            print(f"  第二轮计算节点: {second_pass}")
            from src.features.registry import get_compute_func
            from src.features.loader.feature_computer import _build_call_args
            import inspect

            for n in second_pass:
                try:
                    info = feature_deps_cfg[n]
                    compute_func_name = info.get("compute_func", n)
                    cfn = get_compute_func(compute_func_name)
                    if cfn is None:
                        continue
                    # 过滤 column_mappings: 只保留实际存在的列
                    info_filtered = dict(info)
                    raw_mappings = info.get("column_mappings") or {}
                    if raw_mappings:
                        avail_mappings = {}
                        for param, src in raw_mappings.items():
                            if isinstance(src, str) and src in bar_cols_updated:
                                avail_mappings[param] = src
                            elif isinstance(src, list) and all(
                                s in bar_cols_updated for s in src
                            ):
                                avail_mappings[param] = src
                        info_filtered["column_mappings"] = avail_mappings
                    call_args, call_kwargs = _build_call_args(
                        info_filtered, research_df, n
                    )
                    sig = inspect.signature(cfn)
                    accepts_var_kw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    )
                    if not accepts_var_kw and call_kwargs:
                        allowed = set(sig.parameters.keys())
                        call_kwargs = {
                            k: v for k, v in call_kwargs.items() if k in allowed
                        }
                    result = cfn(*call_args, **call_kwargs)
                    output_cols = info.get("output_columns", [n])
                    if isinstance(result, tuple):
                        if len(result) == len(output_cols):
                            result = pd.DataFrame(dict(zip(output_cols, result)))
                    if isinstance(result, pd.DataFrame):
                        new_cols = [
                            c for c in result.columns if c not in research_df.columns
                        ]
                        if new_cols:
                            aligned = result[new_cols].reindex(research_df.index)
                            research_df = pd.concat([research_df, aligned], axis=1)
                        print(f"    {n}: +{len(result.columns)} cols OK")
                    elif isinstance(result, pd.Series):
                        name = result.name or n
                        if name not in research_df.columns:
                            research_df[name] = result.reindex(research_df.index)
                        print(f"    {n}: +1 col OK")
                except Exception as e:
                    print(f"    {n}: FAILED - {e}")
        else:
            print(f"  第二轮: 无可计算节点")

    t1 = time.time()
    print(
        f"  研究特征 (最终): {len(research_df)} rows × "
        f"{len(research_df.columns)} cols [{t1-t0:.1f}s]"
    )
    if "bpc_breakout_direction" in research_df.columns:
        n_nonzero = (research_df["bpc_breakout_direction"].fillna(0) != 0).sum()
        print(f"  bpc_breakout_direction: {n_nonzero}/{len(research_df)} non-zero")
    else:
        print(f"  WARNING: bpc_breakout_direction 仍不可用")

    # ================================================================
    # 3. 对齐时间索引
    # ================================================================
    overlap_idx = research_df.index.intersection(live_df.index)
    print(f"\n  重叠: {len(overlap_idx)} 个 4H bar")
    print(f"  研究独有: {len(research_df.index.difference(live_df.index))}")
    print(f"  实盘独有: {len(live_df.index.difference(research_df.index))}")

    if len(overlap_idx) < 10:
        print("  ERROR: 重叠 bar 太少，无法比较")
        return 1

    r_overlap = research_df.loc[overlap_idx]
    l_overlap = live_df.loc[overlap_idx]

    # ================================================================
    # 4. 特征级对比
    # ================================================================
    # 找共同列（排除元数据列）
    meta_cols = {"_symbol", "symbol", "timestamp", "datetime", "date"}
    r_feature_cols = set(r_overlap.columns) - meta_cols
    l_feature_cols = set(l_overlap.columns) - meta_cols
    common_cols = sorted(r_feature_cols & l_feature_cols)

    print(f"\n  共同特征列: {len(common_cols)}")
    print(f"  仅研究: {len(r_feature_cols - l_feature_cols)}")
    if r_feature_cols - l_feature_cols:
        r_only = sorted(r_feature_cols - l_feature_cols)
        print(f"    {r_only[:20]}{'...' if len(r_only) > 20 else ''}")
    print(f"  仅实盘: {len(l_feature_cols - r_feature_cols)}")
    if l_feature_cols - r_feature_cols:
        l_only = sorted(l_feature_cols - r_feature_cols)
        print(f"    {l_only[:20]}{'...' if len(l_only) > 20 else ''}")

    # 计算每列的匹配度
    feature_results = []
    for col in common_cols:
        r_vals = pd.to_numeric(r_overlap[col], errors="coerce")
        l_vals = pd.to_numeric(l_overlap[col], errors="coerce")
        result = compare_series(r_vals, l_vals, col, tolerance=args.tolerance)
        feature_results.append(result)

    # 输出: 所有特征
    print_feature_table(feature_results, f"全部特征对比 ({len(overlap_idx)} bars)")

    # BPC 关键特征单独列出
    bpc_key_features = [
        "bpc_breakout_direction",
        "bpc_bb_compression",
        "bpc_was_in_pullback",
        "bpc_pullback_depth",
        "bpc_volume_compression_pct",
        "bpc_dir_consistency_long",
        "bpc_score_breakout",
        "macd_signal_atr",
        "sma_200_position",
        "wpt_ignition_score",
        "wpt_exhaustion_score",
        "price_position",
        "sr_strength_max",
        "vol_percentile_approx",
        "atr",
        "close",
        "open",
        "high",
        "low",
        "volume",
    ]
    bpc_results = [r for r in feature_results if r["name"] in bpc_key_features]
    if bpc_results:
        print_feature_table(bpc_results, f"BPC 关键特征 ({len(overlap_idx)} bars)")

    # 后 200 bar 对比 (warmup 之后)
    n_tail = min(200, len(overlap_idx))
    tail_idx = overlap_idx[-n_tail:]
    if n_tail >= 50:
        tail_results = []
        for col in common_cols:
            r_vals = pd.to_numeric(r_overlap.loc[tail_idx, col], errors="coerce")
            l_vals = pd.to_numeric(l_overlap.loc[tail_idx, col], errors="coerce")
            result = compare_series(r_vals, l_vals, col, tolerance=args.tolerance)
            tail_results.append(result)

        bad_tail = [r for r in tail_results if r["n_valid"] > 0 and r["match_pct"] < 99]
        if bad_tail:
            print_feature_table(bad_tail, f"最后 {n_tail} bar 中匹配 < 99% 的特征")
        else:
            print(f"\n  最后 {n_tail} bar: 所有特征匹配率 >= 99%")

    # ================================================================
    # 5. 信号级对比 (direction → gate → entry_filter → evidence)
    # ================================================================
    if args.skip_signals:
        print("\n  [跳过信号级对比]")
        return 0

    print(f"\n{'='*95}")
    print("  信号级对比")
    print(f"{'='*95}")

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

    # 用后半段做信号比较，避免把重叠区首段异常放大
    calib_cutoff = overlap_idx[len(overlap_idx) // 2]

    # 只比较后半部分（校准之后）
    eval_idx = overlap_idx[overlap_idx >= calib_cutoff]
    if len(eval_idx) < 10:
        eval_idx = overlap_idx

    print(f"\n  信号比较窗口: >= {calib_cutoff}")
    print(f"  评估: {len(eval_idx)} bars (>= {calib_cutoff})")

    print(
        f"\n  {'Timestamp':<22} {'R_dir':>5} {'L_dir':>5} "
        f"{'R_gate':>6} {'L_gate':>6} {'R_ef':>5} {'L_ef':>5} "
        f"{'R_ev':>6} {'L_ev':>6} {'R_sig':>5} {'L_sig':>5} {'Match':>5}"
    )

    n_total = 0
    n_dir_match = 0
    n_gate_match = 0
    n_ef_match = 0
    n_signal_match = 0
    signal_diffs = []

    for ts in eval_idx:
        r_feat = row_to_features(r_overlap.loc[ts])
        l_feat = row_to_features(l_overlap.loc[ts])

        r_dir = int(float(r_feat.get("bpc_breakout_direction", 0)))
        l_dir = int(float(l_feat.get("bpc_breakout_direction", 0)))

        # Gate
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

        # Entry filter + Evidence
        r_ef, l_ef = "—", "—"
        r_ev, l_ev = "—", "—"
        r_signal, l_signal = 0, 0

        if r_dir != 0 and r_gate == "PASS":
            r_should, r_info = bpc_research._evaluate_entry_signal(r_feat)
            if r_should:
                r_ef = "PASS"
                r_ev = f"{r_info.get('evidence_score', 0):.2f}"
                r_signal = r_dir
            elif r_info.get("reject_reason") == "entry_filter_deny":
                r_ef = "DENY"
            else:
                r_ef = "PASS"  # passed filter but failed evidence

        if l_dir != 0 and l_gate == "PASS":
            l_should, l_info = bpc_live._evaluate_entry_signal(l_feat)
            if l_should:
                l_ef = "PASS"
                l_ev = f"{l_info.get('evidence_score', 0):.2f}"
                l_signal = l_dir
            elif l_info.get("reject_reason") == "entry_filter_deny":
                l_ef = "DENY"
            else:
                l_ef = "PASS"

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

        # 只打印有信号或不匹配的行
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
                        "r_dir": r_dir,
                        "l_dir": l_dir,
                        "r_gate": r_gate,
                        "l_gate": l_gate,
                        "r_signal": r_signal,
                        "l_signal": l_signal,
                    }
                )

    # ── 汇总 ──
    print(f"\n{'='*95}")
    print("  汇总")
    print(f"{'='*95}")
    print(f"  数据源:           {args.data_path} (相同数据!)")
    print(f"  时间范围:         {args.start_date} ~ {args.end_date}")
    print(f"  重叠 bars:        {len(overlap_idx)}")
    print(f"  评估 bars:        {n_total}")
    print()

    if n_total > 0:
        print(
            f"  方向一致:         {n_dir_match}/{n_total} ({n_dir_match/n_total*100:.1f}%)"
        )
        print(
            f"  Gate 一致:        {n_gate_match}/{n_total} ({n_gate_match/n_total*100:.1f}%)"
        )
        print(
            f"  Entry Filter 一致: {n_ef_match}/{n_total} ({n_ef_match/n_total*100:.1f}%)"
        )
        print(
            f"  最终信号一致:     {n_signal_match}/{n_total} ({n_signal_match/n_total*100:.1f}%)"
        )

    # 特征匹配汇总
    if feature_results:
        all_match = [r for r in feature_results if r["n_valid"] > 0]
        perfect = [r for r in all_match if r["match_pct"] >= 99.9]
        good = [r for r in all_match if 90 <= r["match_pct"] < 99.9]
        bad = [r for r in all_match if r["match_pct"] < 90]

        print(f"\n  特征匹配分布:")
        print(f"    完美 (>=99.9%):  {len(perfect)}/{len(all_match)}")
        print(f"    良好 (90-99.9%): {len(good)}/{len(all_match)}")
        print(f"    较差 (<90%):     {len(bad)}/{len(all_match)}")

        if bad:
            print(f"\n  匹配率 < 90% 的特征:")
            for r in sorted(bad, key=lambda x: x["match_pct"]):
                print(f"    {r['name']:<35} {r['match_pct']:.1f}%")

    if signal_diffs:
        diff_at_dir = sum(1 for d in signal_diffs if d["r_dir"] != d["l_dir"])
        diff_at_gate = sum(
            1
            for d in signal_diffs
            if d["r_dir"] == d["l_dir"] and d["r_gate"] != d["l_gate"]
        )
        print(f"\n  信号不一致来源:")
        print(f"    方向不同:       {diff_at_dir}")
        print(f"    Gate 不同:      {diff_at_gate}")
        print(f"    其他:           {len(signal_diffs) - diff_at_dir - diff_at_gate}")
    elif n_total > 0:
        print(f"\n  所有信号完全一致!")

    print(f"\n{'='*95}")

    # 返回结果字典（多币种汇总用）
    all_match_r = (
        [r for r in feature_results if r["n_valid"] > 0] if feature_results else []
    )
    n_perfect = len([r for r in all_match_r if r["match_pct"] >= 99.9])
    n_good = len([r for r in all_match_r if 90 <= r["match_pct"] < 99.9])
    n_bad = len([r for r in all_match_r if r["match_pct"] < 90])
    sig_rate = (n_signal_match / n_total * 100) if n_total > 0 else 0.0
    return {
        "symbol": args.symbol,
        "bars": len(overlap_idx),
        "eval_bars": n_total,
        "signal_match_rate": sig_rate,
        "perfect": n_perfect,
        "good": n_good,
        "bad": n_bad,
    }


if __name__ == "__main__":
    sys.exit(main())
