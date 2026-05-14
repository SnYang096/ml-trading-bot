#!/usr/bin/env python3
"""诊断 BPC/ME 在事件回测中为什么不出信号"""
import sys, logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "/home/yin/trading/ml_trading_bot")

import pandas as pd, numpy as np
from pathlib import Path
from scripts.event_backtest import DataHandler
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy

# Load BTC data (shorter range for speed)
dh = DataHandler("data/parquet_data")
bars = dh.load_ohlcv(
    symbol="BTCUSDT", timeframe="1T", start_date="2024-10-01", end_date="2026-02-01"
)
bars.index = pd.to_datetime(bars.index, utc=True)
bars = bars.rename(columns={"buy_qty": "buy_volume", "sell_qty": "sell_volume"})
if "timestamp" not in bars.columns:
    bars["timestamp"] = bars.index

# Load ticks
data_root = Path("data/parquet_data")
tick_frames = []
for fp in sorted(data_root.glob("BTCUSDT_*.parquet")):
    try:
        df_tick = pd.read_parquet(fp)
        if "price" in df_tick.columns and "volume" in df_tick.columns:
            tick_frames.append(df_tick)
    except:
        pass
ticks = pd.concat(tick_frames, ignore_index=True) if tick_frames else pd.DataFrame()
if not ticks.empty:
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True)
print(f"Data: {len(bars)} bars, {len(ticks)} ticks")

# Compute features using same IFC setup as event_backtest.py
from src.time_series_model.live.live_feature_plan import (
    extract_features_from_archetypes,
)

strategies_root = "config/strategies"
tf_strats = {"240T": ["bpc", "fer"], "60T": ["me-long"]}
features_by_tf = {}

for tf, strat_list in tf_strats.items():
    first = strat_list[0]
    archetypes_dir = str(Path(strategies_root) / first / "archetypes")
    fc = IncrementalFeatureComputer(primary_timeframe=tf, archetypes_dir=archetypes_dir)
    for extra in strat_list[1:]:
        extra_dir = str(Path(strategies_root) / extra / "archetypes")
        try:
            extra_feat_set, extra_feat_nodes = extract_features_from_archetypes(
                extra_dir
            )
            if fc.live_feature_set:
                fc.live_feature_set |= extra_feat_set
            fc.live_feature_nodes = sorted(
                set(fc.live_feature_nodes) | set(extra_feat_nodes)
            )
        except Exception as e:
            print(f"  Feature merge for {extra} failed: {e}")
    fc.live_feature_set = None  # disable filtering, same as event backtest
    feat_df = fc.compute_features_dataframe(
        bars_1min=bars, ticks_1min=ticks, primary_timeframe=tf
    )
    feat_df.index = pd.to_datetime(feat_df.index, utc=True)
    features_by_tf[tf] = feat_df
    print(f"Features {tf}: {len(feat_df)} rows x {len(feat_df.columns)} cols")

feat_240t = features_by_tf["240T"]
feat_60t = features_by_tf["60T"]

# Check key features
print("\n=== Key Feature Presence ===")
bpc_feats = ["bpc_volume_compression_pct", "bpc_bb_compression", "bpc_cvd_z"]
me_feats = [
    "atr_percentile",
    "me_atr_pct",
    "me_cvd_alignment",
    "evt_scale",
    "evt_var_99",
    "vp_width_ratio",
]
for f in bpc_feats:
    if f in feat_240t.columns:
        v = feat_240t[f].dropna()
        print(f"  BPC 240T {f}: [{v.min():.4f}, {v.max():.4f}], mean={v.mean():.4f}")
    else:
        print(f"  BPC 240T {f}: *** MISSING ***")
for f in me_feats:
    if f in feat_60t.columns:
        v = feat_60t[f].dropna()
        print(f"  ME 60T {f}: [{v.min():.4f}, {v.max():.4f}], mean={v.mean():.4f}")
    else:
        print(f"  ME 60T {f}: *** MISSING ***")

# Test each strategy
for strat_name, feat_df, tf in [
    ("bpc", feat_240t, "240T"),
    ("me-long", feat_60t, "60T"),
]:
    strat = GenericLiveStrategy(
        strat_name,
        strategies_root="config/strategies",
        primary_timeframe=tf,
        bar_minutes=int(tf.replace("T", "")),
    )
    test = feat_df[feat_df.index >= "2025-08-01"]
    dir0 = 0
    dirP = 0
    gateP = 0
    efP = 0
    gr_ctr = {}

    for _, row in test.iterrows():
        ft = {}
        for k, v in row.items():
            try:
                if v is not None and np.isscalar(v) and not pd.isna(v):
                    ft[str(k)] = float(v)
            except:
                continue

        d, rid = strat.direction_evaluator.evaluate(ft)
        if d == 0:
            dir0 += 1
            continue
        dirP += 1

        gp, gr, gw = strat.gate_evaluator.evaluate(ft, strat._quantiles)
        if not gp:
            for r in gr:
                gr_ctr[r] = gr_ctr.get(r, 0) + 1
            continue
        gateP += 1

        if strat.entry_filter_checker:
            ef = strat.entry_filter_checker.check(ft)
            if not ef:
                continue
        efP += 1

    print(f"\n=== {strat_name.upper()} Diagnosis ({len(test)} bars, {tf}) ===")
    print(f"  Direction=0: {dir0}")
    print(f"  Direction pass: {dirP}")
    print(f"  Gate pass: {gateP}")
    print(f"  Entry filter pass: {efP}")
    if gr_ctr:
        print(f"  Gate rejection breakdown:")
        for r, c in sorted(gr_ctr.items(), key=lambda x: -x[1]):
            print(f"    {r}: {c}")

# Direct comparison: research pipeline vs IFC features at same timestamps
research_df = pd.read_parquet(
    "results/train_final_20260228_155016_return_tree/bpc/logs_gated.parquet"
)
research_df["timestamp"] = pd.to_datetime(research_df["timestamp"], utc=True)
research_allowed = research_df[research_df["gate_decision"] == "allow"]
research_btc = (
    research_allowed[research_allowed["_symbol"] == "BTCUSDT"]
    .set_index("timestamp")
    .sort_index()
)

# Get IFC features at same timestamps
ifc_btc = feat_240t.copy()
common_ts = research_btc.index.intersection(ifc_btc.index)
print(
    f"\n=== Direct Feature Comparison (BTCUSDT, {len(common_ts)} common timestamps) ==="
)
if len(common_ts) > 0:
    for f in ["bpc_volume_compression_pct", "bpc_bb_compression", "bpc_cvd_z"]:
        research_vals = research_btc.loc[common_ts, f]
        ifc_vals = ifc_btc.loc[common_ts, f]
        corr = research_vals.corr(ifc_vals)
        diff = (research_vals - ifc_vals).abs()
        print(f"  {f}:")
        print(
            f"    Corr: {corr:.4f}, MAE: {diff.mean():.4f}, Max diff: {diff.max():.4f}"
        )
        # Show first 3 examples
        for ts in common_ts[:3]:
            rv = research_vals.loc[ts]
            iv = ifc_vals.loc[ts]
            print(f"    {ts}: research={rv:.4f}, ifc={iv:.4f}, diff={abs(rv-iv):.4f}")
else:
    print("  No common timestamps found!")
    print(f"  Research BTC timestamps: {research_btc.index[:5].tolist()}")
    print(f"  IFC timestamps: {ifc_btc.index[:5].tolist()}")
