#!/usr/bin/env python3
"""
向量回测 vs 事件回测 — 多维度对比诊断
对比: Gate / Entry Filter / Evidence / Slot / 加仓 / 宪法约束

用法:
  python scripts/compare_dimensions.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── 配置 ──
LOGS_GATED = {
    "bpc": "results/train_final_20260228_155016_return_tree/bpc/logs_gated.parquet",
    "fer": "results/train_final_20260228_155642_return_tree/fer/logs_gated.parquet",
    "me-long": "results/train_final_20260228_160556_return_tree/me/logs_gated.parquet",
}
VECTOR_CSV = "/tmp/trades_vector.csv"
EVENT_CSV = "/tmp/trades_event.csv"
TEST_START = "2025-08-01"
TEST_END = "2026-01-01"
STRATEGIES_ROOT = "config/strategies"


def load_logs_gated(arch: str) -> pd.DataFrame:
    p = LOGS_GATED[arch]
    df = pd.read_parquet(p)
    # timestamp 可能是列而非 index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] < TEST_END)
        return df[mask].copy()
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        mask = (df.index >= TEST_START) & (df.index < TEST_END)
        return df[mask].copy()


def replay_vector_pipeline(arch: str, df: pd.DataFrame) -> dict:
    """在 logs_gated 上重放向量回测管线, 统计每层通过率"""
    from src.time_series_model.execution.entry_filter import (
        apply_entry_filters_or,
        compute_derived_entry_features,
        load_entry_filters_config,
    )
    from scripts.backtest_execution_layer import (
        compute_evidence_scores,
        compute_evidence_quantiles,
    )
    import yaml

    stats = {}

    # 列名适配
    if "_symbol" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"_symbol": "symbol"})

    # 0. 总 bars
    stats["total_bars"] = len(df)

    # 1. Direction: pred 列 → entry_direction (threshold 0.5)
    if "pred" in df.columns:
        df["entry_direction"] = 0.0
        df.loc[df["pred"] > 0.5, "entry_direction"] = 1.0
        df.loc[df["pred"] < -0.5, "entry_direction"] = -1.0
    elif "entry_direction" in df.columns:
        pass
    else:
        # 从 breakout_direction 推断
        dir_col = f"{arch}_breakout_direction"
        if dir_col in df.columns:
            df["entry_direction"] = df[dir_col]
        else:
            df["entry_direction"] = 0.0

    stats["direction_nonzero"] = int((df["entry_direction"] != 0).sum())

    # 2. Gate
    if "gate_decision" in df.columns:
        veto = df["gate_decision"] != "allow"
        stats["gate_allow"] = int((~veto).sum())
        stats["gate_veto"] = int(veto.sum())
        # 统计 gate veto 中有 direction 的
        has_dir_and_veto = (df["entry_direction"] != 0) & veto
        stats["gate_blocked_signals"] = int(has_dir_and_veto.sum())
        df.loc[veto, "entry_direction"] = 0.0
    else:
        stats["gate_allow"] = stats["total_bars"]
        stats["gate_veto"] = 0
        stats["gate_blocked_signals"] = 0

    stats["after_gate"] = int((df["entry_direction"] != 0).sum())

    # 3. Entry Filter
    ef_cfg = load_entry_filters_config(arch, STRATEGIES_ROOT)
    if ef_cfg:
        compute_derived_entry_features(df)
        n_before_ef = int((df["entry_direction"] != 0).sum())
        n_after_ef = apply_entry_filters_or(df, ef_cfg, silent=True)
        stats["entry_filter_pass"] = n_after_ef
        stats["entry_filter_reject"] = n_before_ef - n_after_ef
    else:
        stats["entry_filter_pass"] = stats["after_gate"]
        stats["entry_filter_reject"] = 0

    stats["after_entry_filter"] = int((df["entry_direction"] != 0).sum())

    # 4. Evidence
    ev_path = Path(STRATEGIES_ROOT) / arch / "archetypes" / "evidence.yaml"
    if ev_path.exists():
        with open(ev_path) as f:
            ev_cfg = yaml.safe_load(f) or {}
        try:
            quantiles = compute_evidence_quantiles(df, ev_cfg)
            scores = compute_evidence_scores(
                df, ev_cfg, silent=True, precomputed_quantiles=quantiles
            )
            active = df["entry_direction"] != 0
            stats["evidence_mean"] = (
                float(scores[active].mean()) if active.sum() > 0 else 0.0
            )
            stats["evidence_std"] = (
                float(scores[active].std()) if active.sum() > 0 else 0.0
            )
            stats["evidence_min"] = (
                float(scores[active].min()) if active.sum() > 0 else 0.0
            )
            stats["evidence_max"] = (
                float(scores[active].max()) if active.sum() > 0 else 0.0
            )
        except Exception as e:
            stats["evidence_mean"] = f"error: {e}"
    else:
        stats["evidence_mean"] = "N/A"

    # Per-symbol breakdown
    if "_symbol" in df.columns:
        sym_col = "_symbol"
    elif "symbol" in df.columns:
        sym_col = "symbol"
    else:
        sym_col = None

    if sym_col:
        per_sym = {}
        for s, g in df.groupby(sym_col):
            per_sym[s] = int((g["entry_direction"] != 0).sum())
        stats["per_symbol_signals"] = per_sym

    return stats


def analyze_trade_csv(path: str) -> dict:
    """分析 trade CSV 统计"""
    df = pd.read_csv(path)
    stats = {
        "total_trades": len(df),
        "per_archetype": {},
    }

    for arch, g in df.groupby("archetype"):
        arch_stats = {
            "trades": len(g),
            "win_rate": float((g["pnl_r"] > 0).mean()) * 100,
            "mean_r": float(g["pnl_r"].mean()),
            "evidence_mean": float(g["evidence"].mean()),
            "evidence_std": float(g["evidence"].std()),
            "evidence_min": float(g["evidence"].min()),
            "evidence_max": float(g["evidence"].max()),
        }

        # 出场原因分布
        exit_dist = g["exit_reason"].value_counts(normalize=True).to_dict()
        arch_stats["exit_reasons"] = {
            k: round(v * 100, 1) for k, v in exit_dist.items()
        }

        # 加仓
        if "is_add_position" in g.columns:
            arch_stats["add_position_count"] = int(g["is_add_position"].sum())
        else:
            arch_stats["add_position_count"] = "N/A (no column)"

        # Per-symbol
        arch_stats["per_symbol"] = g.groupby("symbol").size().to_dict()

        stats["per_archetype"][arch] = arch_stats

    # 全局加仓统计
    if "is_add_position" in df.columns:
        stats["total_add_positions"] = int(df["is_add_position"].sum())
    else:
        stats["total_add_positions"] = "N/A"

    return stats


def replay_event_funnel():
    """重放事件回测的 decide() 调用, 获取 per-archetype gate/ef/evidence 统计"""
    from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy

    strats = {}
    tf_map = {"bpc": "240T", "fer": "240T", "me-long": "60T"}
    bm_map = {"bpc": 240, "fer": 240, "me-long": 60}

    for s in ["bpc", "fer", "me-long"]:
        strats[s] = GenericLiveStrategy(
            s,
            strategies_root=STRATEGIES_ROOT,
            primary_timeframe=tf_map[s],
            bar_minutes=bm_map[s],
        )

    # 加载特征并设置 quantiles
    from src.time_series_model.live.incremental_feature_computer import (
        IncrementalFeatureComputer,
    )
    from src.data_tools.data_handler import DataHandler

    dh = DataHandler("data/parquet_data")
    symbols = ["ADAUSDT", "BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

    # 按 timeframe 分组计算特征
    all_features_by_tf = defaultdict(list)
    sym_features = {}

    for tf in sorted(set(tf_map.values())):
        tf_strats = [s for s in ["bpc", "fer", "me-long"] if tf_map[s] == tf]
        first = tf_strats[0]
        archetypes_dir = str(Path(STRATEGIES_ROOT) / first / "archetypes")
        fc = IncrementalFeatureComputer(
            primary_timeframe=tf, archetypes_dir=archetypes_dir
        )

        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )

        for extra in tf_strats[1:]:
            extra_dir = str(Path(STRATEGIES_ROOT) / extra / "archetypes")
            try:
                extra_feat_set, extra_feat_nodes = extract_features_from_archetypes(
                    extra_dir
                )
                if fc.live_feature_set:
                    fc.live_feature_set |= extra_feat_set
                fc.live_feature_nodes = sorted(
                    set(fc.live_feature_nodes) | set(extra_feat_nodes)
                )
            except Exception:
                pass
        fc.live_feature_set = None

        for sym in symbols:
            warmup_start = "2025-05-01"
            bars_1min = dh.load_ohlcv(
                symbol=sym, timeframe="1T", start_date=warmup_start, end_date=TEST_END
            )
            if bars_1min.empty:
                continue
            bars_1min.index = pd.to_datetime(bars_1min.index, utc=True)
            col_rename = {"buy_qty": "buy_volume", "sell_qty": "sell_volume"}
            bars_1min = bars_1min.rename(
                columns={k: v for k, v in col_rename.items() if k in bars_1min.columns}
            )
            if "timestamp" not in bars_1min.columns:
                bars_1min["timestamp"] = bars_1min.index

            features_df = fc.compute_features_dataframe(
                bars_1min=bars_1min, ticks_1min=pd.DataFrame(), primary_timeframe=tf
            )
            if features_df.empty:
                continue
            features_df.index = pd.to_datetime(features_df.index, utc=True)
            all_features_by_tf[tf].append(features_df)

            test_df = features_df[
                (features_df.index >= TEST_START) & (features_df.index < TEST_END)
            ]
            if not test_df.empty:
                sym_features[(sym, tf)] = test_df
                print(f"  Loaded {sym} {tf}: {len(test_df)} test bars")

    # 设置 quantiles
    for s_name, s_obj in strats.items():
        tf = tf_map[s_name]
        if tf in all_features_by_tf:
            combined = pd.concat(all_features_by_tf[tf], axis=0)
            s_obj.set_quantiles_from_df(combined)

    # 逐行 decide, 统计 per-archetype funnel
    results = {
        s: {
            "total_checked": 0,
            "direction_nonzero": 0,
            "gate_pass": 0,
            "gate_deny": 0,
            "entry_filter_pass": 0,
            "entry_filter_deny": 0,
            "evidence_scores": [],
            "per_symbol": defaultdict(int),
        }
        for s in strats
    }

    def row_to_features(row):
        ft = {}
        for k, v in row.items():
            try:
                if v is not None and np.isscalar(v) and not pd.isna(v):
                    ft[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return ft

    for (sym, tf), test_df in sym_features.items():
        tf_strats = [s for s in strats if tf_map[s] == tf]
        for ts, row in test_df.iterrows():
            features = row_to_features(row)
            for s_name in tf_strats:
                s_obj = strats[s_name]
                results[s_name]["total_checked"] += 1

                intents = s_obj.decide(features=features, symbol=sym, bars=None)
                lf = getattr(s_obj, "_last_funnel", {})

                if intents:
                    results[s_name]["direction_nonzero"] += 1
                    results[s_name]["gate_pass"] += 1
                    results[s_name]["entry_filter_pass"] += 1
                    ev = getattr(intents[0], "confidence", 0.5)
                    results[s_name]["evidence_scores"].append(ev)
                    results[s_name]["per_symbol"][sym] += 1
                else:
                    # 检查漏斗拒绝原因
                    dir_val = lf.get("direction_value", 0)
                    if dir_val == 0:
                        pass  # no direction
                    else:
                        results[s_name]["direction_nonzero"] += 1
                        gate_ok = lf.get("gate", False)
                        if not gate_ok:
                            results[s_name]["gate_deny"] += 1
                        else:
                            results[s_name]["gate_pass"] += 1
                            ef_ok = lf.get("entry_filter", False)
                            if not ef_ok:
                                results[s_name]["entry_filter_deny"] += 1

    return results


def print_section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def main():
    print_section("维度 1: 向量侧 Pipeline 重放 (logs_gated)")
    vec_pipeline = {}
    for arch in ["bpc", "fer", "me-long"]:
        print(f"\n--- {arch.upper()} ---")
        df = load_logs_gated(arch)
        stats = replay_vector_pipeline(arch, df)
        vec_pipeline[arch] = stats
        print(f"  总 bars:           {stats['total_bars']}")
        print(f"  方向 != 0:         {stats['direction_nonzero']}")
        print(f"  Gate allow:        {stats['gate_allow']} (veto={stats['gate_veto']})")
        print(f"  Gate 拦截信号:     {stats['gate_blocked_signals']}")
        print(f"  After gate:        {stats['after_gate']}")
        print(
            f"  Entry filter pass: {stats['entry_filter_pass']} (reject={stats['entry_filter_reject']})"
        )
        print(f"  After EF:          {stats['after_entry_filter']}")
        if isinstance(stats.get("evidence_mean"), float):
            print(
                f"  Evidence (active): mean={stats['evidence_mean']:.4f} ± {stats['evidence_std']:.4f} [{stats['evidence_min']:.4f}, {stats['evidence_max']:.4f}]"
            )
        if "per_symbol_signals" in stats:
            for s, c in sorted(stats["per_symbol_signals"].items()):
                print(f"    {s}: {c} signals")

    print_section("维度 2: 事件侧 Pipeline 重放 (decide per bar)")
    print("  重放 decide() 调用...")
    evt_pipeline = replay_event_funnel()
    for arch in ["bpc", "fer", "me-long"]:
        s = evt_pipeline[arch]
        ev_arr = s["evidence_scores"]
        print(f"\n--- {arch.upper()} ---")
        print(f"  总 checked:        {s['total_checked']}")
        print(f"  方向 != 0:         {s['direction_nonzero']}")
        print(f"  Gate pass:         {s['gate_pass']} (deny={s['gate_deny']})")
        print(
            f"  Entry filter pass: {s['entry_filter_pass']} (deny={s['entry_filter_deny']})"
        )
        if ev_arr:
            print(
                f"  Evidence (pass):   mean={np.mean(ev_arr):.4f} ± {np.std(ev_arr):.4f} [{min(ev_arr):.4f}, {max(ev_arr):.4f}]"
            )
        for sym, cnt in sorted(s["per_symbol"].items()):
            print(f"    {sym}: {cnt} signals")

    print_section("维度 3: Trade CSV 对比 — Evidence / 加仓 / 出场")
    vec_trades = analyze_trade_csv(VECTOR_CSV)
    evt_trades = analyze_trade_csv(EVENT_CSV)

    for arch in ["bpc", "fer", "me-long"]:
        v = vec_trades["per_archetype"].get(arch, {})
        e = evt_trades["per_archetype"].get(arch, {})
        print(f"\n--- {arch.upper()} ---")
        print(f"  {'':20s} {'向量':>10s} {'事件':>10s} {'偏差':>10s}")
        vt = v.get("trades", 0)
        et = e.get("trades", 0)
        print(f"  {'Trades':20s} {vt:>10d} {et:>10d} {vt-et:>10d}")
        vw = v.get("win_rate", 0)
        ew = e.get("win_rate", 0)
        print(f"  {'Win Rate %':20s} {vw:>10.1f} {ew:>10.1f} {vw-ew:>10.1f}")
        vm = v.get("mean_r", 0)
        em = e.get("mean_r", 0)
        print(f"  {'Mean R':20s} {vm:>10.4f} {em:>10.4f} {vm-em:>10.4f}")
        ve = v.get("evidence_mean", 0)
        ee = e.get("evidence_mean", 0)
        print(f"  {'Evidence Mean':20s} {ve:>10.4f} {ee:>10.4f} {ve-ee:>10.4f}")
        vs = v.get("evidence_std", 0)
        es = e.get("evidence_std", 0)
        print(f"  {'Evidence Std':20s} {vs:>10.4f} {es:>10.4f} {vs-es:>10.4f}")
        va = v.get("add_position_count", 0)
        ea = e.get("add_position_count", 0)
        va_str = str(va)
        ea_str = str(ea)
        print(f"  {'Add Position':20s} {va_str:>10s} {ea_str:>10s}")

        # 出场原因
        v_exit = v.get("exit_reasons", {})
        e_exit = e.get("exit_reasons", {})
        all_reasons = sorted(set(list(v_exit.keys()) + list(e_exit.keys())))
        if all_reasons:
            print(f"  出场原因:")
            for r in all_reasons:
                vr = v_exit.get(r, 0)
                er = e_exit.get(r, 0)
                print(f"    {r:22s} {vr:>8.1f}% {er:>8.1f}% {vr-er:>8.1f}pp")

        # Per-symbol
        v_sym = v.get("per_symbol", {})
        e_sym = e.get("per_symbol", {})
        all_syms = sorted(set(list(v_sym.keys()) + list(e_sym.keys())))
        if all_syms:
            print(f"  Per-Symbol:")
            for s in all_syms:
                vs2 = v_sym.get(s, 0)
                es2 = e_sym.get(s, 0)
                print(f"    {s:12s} {vs2:>6d} {es2:>6d} {vs2-es2:>+6d}")

    print_section("维度 4: 宪法约束 (Kill Switch)")
    print("  向量回测: Kill Switch 未实现 (N/A)")
    print("  事件回测: Kill Switch 触发=0, 跳过=0 (从回测输出)")

    print_section("维度 5: 信号管线汇总对比")
    print(
        f"\n  {'Archetype':8s} {'向量信号':>10s} {'事件信号':>10s} {'向量Trades':>12s} {'事件Trades':>12s} {'信号差':>8s} {'Trade差':>8s}"
    )
    for arch in ["bpc", "fer", "me-long"]:
        v_sig = vec_pipeline[arch]["after_entry_filter"]
        e_sig = evt_pipeline[arch]["entry_filter_pass"]
        vt2 = vec_trades["per_archetype"].get(arch, {}).get("trades", 0)
        et2 = evt_trades["per_archetype"].get(arch, {}).get("trades", 0)
        print(
            f"  {arch.upper():8s} {v_sig:>10d} {e_sig:>10d} {vt2:>12d} {et2:>12d} {v_sig-e_sig:>+8d} {vt2-et2:>+8d}"
        )

    # 汇总
    v_total_sig = sum(
        vec_pipeline[a]["after_entry_filter"] for a in ["bpc", "fer", "me-long"]
    )
    e_total_sig = sum(
        evt_pipeline[a]["entry_filter_pass"] for a in ["bpc", "fer", "me-long"]
    )
    v_total_t = sum(
        vec_trades["per_archetype"].get(a, {}).get("trades", 0)
        for a in ["bpc", "fer", "me-long"]
    )
    e_total_t = sum(
        evt_trades["per_archetype"].get(a, {}).get("trades", 0)
        for a in ["bpc", "fer", "me-long"]
    )
    print(
        f"  {'TOTAL':8s} {v_total_sig:>10d} {e_total_sig:>10d} {v_total_t:>12d} {e_total_t:>12d} {v_total_sig-e_total_sig:>+8d} {v_total_t-e_total_t:>+8d}"
    )

    print(
        f"\n  加仓总计:  向量={vec_trades['total_add_positions']}  事件={evt_trades['total_add_positions']}"
    )


if __name__ == "__main__":
    main()
