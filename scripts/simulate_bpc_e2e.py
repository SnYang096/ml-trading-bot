#!/usr/bin/env python3
"""
BPC 端到端模拟测试 — 用磁盘历史数据回放，验证完整链路

优化版：
  1. 一次性批量计算全部特征（~6秒）
  2. 逐行喂给 BPC decide()，毫秒级
  3. 详细诊断每一步（direction/gate/entry_filter/evidence）

链路: 磁盘 bars/ticks → 特征计算(全量) → 逐行 GenericLiveStrategy decide → TradeIntent 输出

用法:
    python scripts/simulate_bpc_e2e.py --symbol BTCUSDT --days 7
    python scripts/simulate_bpc_e2e.py --days 3 --interval 60  # 快速
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_data_stream.feature_storage import StorageManager
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_sim")


# ============================================================
# 辅助：从 DataFrame 行提取特征 dict
# ============================================================
def row_to_features(row: pd.Series) -> Dict[str, float]:
    """把 DataFrame 的一行转为策略需要的 features dict"""
    features = {}
    for k, v in row.items():
        try:
            if v is not None and np.isscalar(v) and not pd.isna(v):
                features[str(k)] = float(v)
        except (ValueError, TypeError):
            continue
    return features


# ============================================================
# 主逻辑
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="BPC 端到端模拟测试 (优化版)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=7, help="测试天数")
    parser.add_argument(
        "--interval", type=int, default=240, help="触发间隔(分钟)，240=每个4h bar"
    )
    parser.add_argument("--live-root", default="live/highcap", help="实盘数据根目录")
    parser.add_argument(
        "--strategies-root",
        default="live/highcap/config/strategies",
        help="策略配置目录",
    )
    parser.add_argument("--verbose", action="store_true", help="打印每行特征详情")
    parser.add_argument(
        "--export-signals",
        type=str,
        default=None,
        help="导出逐 bar 信号决策 CSV，用于与 backtest_execution_layer.py 对比验证信号对齐。",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  BPC 端到端模拟测试 (优化版)")
    print("=" * 70)
    print(f"  Symbol: {args.symbol}")
    print(f"  测试: 最近 {args.days} 天, 间隔 {args.interval} min")
    print(f"  策略: {args.strategies_root}")
    print()

    # ── 1. 加载数据 ──
    storage = StorageManager(f"{args.live_root}/data")
    end_date = datetime.now().strftime("%Y-%m-%d")
    warmup_start = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

    logger.info("加载 bars 和 ticks ...")
    bars = storage.bar_1min.load_range(args.symbol, warmup_start, end_date)
    ticks = storage.ticks.load_range(args.symbol, warmup_start, end_date)
    logger.info(f"数据: {len(bars)} bars, {len(ticks)} ticks")

    if len(bars) == 0:
        print("  没有 bars 数据，退出")
        return 1

    # ── 2. 初始化 ──
    archetypes_dir = str(Path(args.strategies_root) / "bpc" / "archetypes")
    feature_computer = IncrementalFeatureComputer(
        primary_timeframe="240T",
        archetypes_dir=archetypes_dir,
    )

    bpc = GenericLiveStrategy(
        strategy_name="bpc",
        strategies_root=args.strategies_root,
        primary_timeframe="240T",
        bar_minutes=240,
    )

    # ── 3. 一次性计算全部特征 ──
    logger.info("一次性批量计算全部特征 ...")
    t0 = time.time()
    features_df = feature_computer.compute_features_dataframe(
        bars_1min=bars,
        ticks_1min=ticks,
        primary_timeframe="240T",
    )
    elapsed = time.time() - t0
    logger.info(
        f"特征计算完成: {len(features_df)} 行 x {len(features_df.columns)} 列, 耗时 {elapsed:.1f}s"
    )

    if features_df.empty:
        print("  特征为空，退出")
        return 1

    # ── 4. 确定测试范围 ──
    features_df.index = pd.to_datetime(features_df.index, utc=True)
    data_end = features_df.index.max()
    test_start = data_end - timedelta(days=args.days)
    test_df = features_df[features_df.index >= test_start].copy()

    # 按 interval 采样（如果 interval < 240，则在 4h bar 之间插值无意义；如果 > 240 就跳过）
    if args.interval > 240:
        step = args.interval // 240
        test_df = test_df.iloc[::step]

    logger.info(f"测试范围: {test_start} ~ {data_end}")
    logger.info(f"测试行数: {len(test_df)} (每行 = 1 个 4h bar)")

    # ── 5. 逐行 BPC 决策 ──
    stats = {
        "total": len(test_df),
        "signals": [],
        "no_direction": 0,
        "gate_deny": 0,
        "entry_filter_deny": 0,
        "gate_reasons": Counter(),
        "reject_reasons": Counter(),
    }

    # 关键特征分布追踪
    key_features_track = defaultdict(list)
    evidence_scores_track = []  # 追踪所有 evidence score
    KEY_FEATURES = [
        "bpc_breakout_direction",
        "bpc_was_in_pullback",
        "bpc_pullback_depth",
        "bpc_bb_compression",
        "ef_liquidity_silence",
        "bpc_score_breakout",
        "bpc_dir_consistency_long",
        "wpt_ignition_score",
        "wpt_exhaustion_score",
        "price_position",
        "bpc_volume_compression_pct",
        "atr",
    ]

    # 信号导出记录（--export-signals）
    signal_records: List[Dict[str, Any]] = []

    for i, (ts, row) in enumerate(test_df.iterrows()):
        features = row_to_features(row)

        # 追踪关键特征
        for kf in KEY_FEATURES:
            key_features_track[kf].append(features.get(kf, float("nan")))

        # BPC 决策
        intents = bpc.decide(features=features, symbol=args.symbol)

        # 记录每个 bar 的决策（用于 --export-signals）
        direction = features.get("bpc_breakout_direction", 0)
        try:
            direction = int(float(direction))
        except (TypeError, ValueError):
            direction = 0

        if intents:
            intent = intents[0]
            sig = {
                "time": str(ts),
                "action": intent.action,
                "confidence": intent.confidence,
                "size_mult": intent.size_multiplier,
                "price": features.get("close", 0),
                "atr": features.get("atr", 0),
                "bpc_dir": features.get("bpc_breakout_direction", 0),
                "bpc_score": features.get("bpc_score_breakout", 0),
                "bpc_pullback_depth": features.get("bpc_pullback_depth", 0),
                "tier": (
                    bpc._last_tier_params.get("tier_name", "?")
                    if bpc._last_tier_params
                    else "?"
                ),
            }
            stats["signals"].append(sig)
            evidence_scores_track.append(intent.confidence)

            # 导出记录
            if direction != 0:
                signal_records.append(
                    {
                        "timestamp": str(ts),
                        "symbol": args.symbol,
                        "direction": direction,
                        "entry_direction": 1 if intent.action == "BUY" else -1,
                        "evidence_score": round(intent.confidence, 4),
                        "tier": sig["tier"],
                    }
                )

            logger.info(
                f"  SIGNAL [{i+1}/{len(test_df)}] @ {ts}: "
                f"{intent.action} conf={intent.confidence:.2f} "
                f"tier={sig['tier']} "
                f"price={sig['price']:.2f} depth={sig['bpc_pullback_depth']:.2f}"
            )
        else:
            # 详细诊断拒绝原因
            _, info = bpc._evaluate_entry_signal(features)
            reason = info.get("reject_reason", "unknown")
            stats["reject_reasons"][reason] += 1

            # 导出记录（被拒绝的 bar）
            if direction != 0:
                signal_records.append(
                    {
                        "timestamp": str(ts),
                        "symbol": args.symbol,
                        "direction": direction,
                        "entry_direction": 0,
                        "evidence_score": "",
                        "tier": "",
                    }
                )

            if reason == "no_direction":
                stats["no_direction"] += 1
            elif reason == "gate_deny":
                stats["gate_deny"] += 1
                gate_reasons = info.get("gate_reasons", [])
                for gr in gate_reasons:
                    stats["gate_reasons"][str(gr)] += 1
            elif reason == "entry_filter_deny":
                stats["entry_filter_deny"] += 1

            if args.verbose:
                logger.info(
                    f"  REJECT [{i+1}/{len(test_df)}] @ {ts}: {reason} "
                    f"dir={features.get('bpc_breakout_direction', '?')} "
                    f"pullback={features.get('bpc_was_in_pullback', '?')} "
                    f"depth={features.get('bpc_pullback_depth', '?'):.2f}"
                )

    # ── 6. 打印报告 ──
    print()
    print("=" * 70)
    print("  BPC 端到端模拟结果")
    print("=" * 70)

    total = stats["total"]
    n_sig = len(stats["signals"])

    print(f"\n  触发统计:")
    print(f"    总 4h bars: {total}")
    print(f"    产生信号:   {n_sig}")
    print(
        f"    无方向:     {stats['no_direction']} ({stats['no_direction']/total*100:.1f}%)"
    )
    print(f"    Gate 拒绝:  {stats['gate_deny']} ({stats['gate_deny']/total*100:.1f}%)")
    print(
        f"    Filter 拒绝:{stats['entry_filter_deny']} ({stats['entry_filter_deny']/total*100:.1f}%)"
    )

    if stats["gate_reasons"]:
        print(f"\n  Gate 拒绝明细:")
        for reason, count in stats["gate_reasons"].most_common():
            print(f"    {reason}: {count}")

    if stats["reject_reasons"]:
        print(f"\n  拒绝原因分布:")
        for reason, count in stats["reject_reasons"].most_common():
            pct = count / total * 100
            print(f"    {reason}: {count} ({pct:.1f}%)")

    # ── 7. 关键特征分布 ──
    print(f"\n  关键特征分布 (测试期间 {total} 个 4h bar):")
    for kf in KEY_FEATURES:
        vals = key_features_track[kf]
        vals_clean = [v for v in vals if not np.isnan(v)]
        if not vals_clean:
            print(f"    {kf:40s}: ALL NaN (缺失!)")
        else:
            arr = np.array(vals_clean)
            non_zero = np.count_nonzero(arr)
            print(
                f"    {kf:40s}: "
                f"min={arr.min():.4f}  "
                f"median={np.median(arr):.4f}  "
                f"max={arr.max():.4f}  "
                f"non-zero={non_zero}/{len(arr)}"
            )

    # ── 8. Entry Filter 通过条件分析 ──
    print(f"\n  Entry Filter 条件分析 (已通过 direction + gate 的 bars):")
    passed_dir_gate = 0
    ef_bb_pass = 0
    ef_liq_pass = 0
    ef_both_fail = 0
    for _, row in test_df.iterrows():
        features = row_to_features(row)
        direction = features.get("bpc_breakout_direction", 0)
        try:
            direction = int(float(direction))
        except (TypeError, ValueError):
            direction = 0
        if direction == 0:
            continue

        # Gate check
        if bpc._archetype is not None:
            gate_passed, _, _ = bpc._archetype.apply_gate(features)
            if not gate_passed:
                continue

        passed_dir_gate += 1

        # Check each entry filter condition manually
        in_pullback = features.get("bpc_was_in_pullback", 0) == 1
        depth_06 = features.get("bpc_pullback_depth", 0) >= 0.6
        depth_055 = features.get("bpc_pullback_depth", 0) >= 0.55
        bb_compress = features.get("bpc_bb_compression", 0) > 0.72
        liq_silence = features.get("ef_liquidity_silence", 1.0) < 0.2

        bb_filter = in_pullback and depth_06 and bb_compress
        liq_filter = in_pullback and depth_055 and liq_silence

        if bb_filter:
            ef_bb_pass += 1
        if liq_filter:
            ef_liq_pass += 1
        if not bb_filter and not liq_filter:
            ef_both_fail += 1

    print(f"    通过 direction+gate: {passed_dir_gate}")
    print(f"    通过 deep_pullback_bb: {ef_bb_pass}")
    print(f"    通过 liquidity_silence: {ef_liq_pass}")
    print(f"    两个都不通过: {ef_both_fail}")

    # entry filter 不通过的细分原因
    if passed_dir_gate > 0:
        not_in_pb = 0
        depth_too_shallow = 0
        bb_not_compressed = 0
        liq_not_silent = 0
        for _, row in test_df.iterrows():
            features = row_to_features(row)
            direction = features.get("bpc_breakout_direction", 0)
            try:
                direction = int(float(direction))
            except (TypeError, ValueError):
                direction = 0
            if direction == 0:
                continue
            if bpc._archetype is not None:
                gate_passed, _, _ = bpc._archetype.apply_gate(features)
                if not gate_passed:
                    continue

            in_pullback = features.get("bpc_was_in_pullback", 0) == 1
            if not in_pullback:
                not_in_pb += 1
                continue
            depth = features.get("bpc_pullback_depth", 0)
            if depth < 0.55:
                depth_too_shallow += 1
                continue
            bb = features.get("bpc_bb_compression", 0)
            liq = features.get("ef_liquidity_silence", 1.0)
            if bb <= 0.72:
                bb_not_compressed += 1
            if liq >= 0.2:
                liq_not_silent += 1

        print(f"\n    Entry Filter 不通过细分 (通过 dir+gate 后):")
        print(f"      不在回踩中 (bpc_was_in_pullback!=1): {not_in_pb}")
        print(f"      回踩太浅 (depth<0.55): {depth_too_shallow}")
        print(f"      BB未压缩 (bb_compression<=0.72): {bb_not_compressed}")
        print(f"      流动性未沉默 (ef_liq_silence>=0.2): {liq_not_silent}")

    # ── 9. 信号列表 ──
    if stats["signals"]:
        print(f"\n  产生的信号 ({n_sig} 个):")
        for sig in stats["signals"]:
            print(
                f"    {sig['time']}  {sig['action']}  "
                f"conf={sig['confidence']:.2f}  "
                f"tier={sig.get('tier', '?')}  "
                f"price={sig['price']:.2f}  "
                f"depth={sig['bpc_pullback_depth']:.2f}  "
                f"bpc_dir={sig['bpc_dir']}  "
                f"bpc_score={sig['bpc_score']:.3f}"
            )
        if evidence_scores_track:
            ev_arr = np.array(evidence_scores_track)
            print(
                f"\n  Evidence Score 分布 ({len(ev_arr)} signals):  "
                f"min={ev_arr.min():.3f}  median={np.median(ev_arr):.3f}  "
                f"max={ev_arr.max():.3f}  mean={ev_arr.mean():.3f}"
            )
    else:
        print(f"\n  没有产生信号")
        print(
            f"  这可能是正常的 — BPC 在 {args.days} 天内未发现 breakout+deep_pullback 形态"
        )
        print(f"  尝试增加 --days 14 或 --days 30 扩大搜索范围")

    print("\n" + "=" * 70)

    # ── Export signals CSV（信号对齐验证）──
    if args.export_signals and signal_records:
        export_path = Path(args.export_signals)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_df = pd.DataFrame(signal_records)
        export_df.to_csv(export_path, index=False)
        print(f"\n   📤 Signals exported: {len(export_df)} rows → {export_path}")

    return 0 if n_sig > 0 or stats["total"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
