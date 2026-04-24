"""XRP 单独复盘：原 trailing vs 最优 dynamic trailing，逐笔对比。"""

import sys

sys.path.insert(0, ".")
import glob
import pandas as pd, numpy as np
from scripts.simulate_srb_l3_trailing import (
    SimParams,
    simulate_trailing,
    load_symbol_bars,
)

trades = pd.read_parquet(
    "reports/srb_break_level_attribution_v2_alltrades_trades.parquet"
)
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
xrp = trades[
    (trades.symbol == "XRPUSDT")
    & ~trades.is_add_position.fillna(False)
    & ~trades.is_reverse.fillna(False)
].copy()
bars = load_symbol_bars("feature_store/features_srb_120T_5643a66b47", "XRPUSDT")

best_params = SimParams(
    m_far=7.0,
    m_near=5.0,
    thr_l3_atr=2.0,
    activation_r=1.0,
    breakeven_lock_r=0.0,
    max_hold_bars=360,
)
baseline_like_prod = SimParams(
    m_far=5.0,
    m_near=5.0,
    thr_l3_atr=2.0,
    activation_r=6.0,
    breakeven_lock_r=0.0,
    max_hold_bars=360,
)

print(
    f"{'entry_time':16s}  {'side':5s}  {'orig_pnl':>8s}  {'sim_prod':>8s}  {'alt_pnl':>8s}  {'delta':>7s}  reason_alt   bars_alt   orig_exit   wide_dist"
)
rows = []
for _, t in xrp.iterrows():
    ep = float(t.entry_price)
    sp = float(t.effective_stop_pct)
    orig_pnl = float(t.pnl_r)
    prod_alt, prod_reason, prod_bars = simulate_trailing(
        bars, t.side, t.entry_time, ep, sp, baseline_like_prod
    )
    best_alt, best_reason, best_bars = simulate_trailing(
        bars, t.side, t.entry_time, ep, sp, best_params
    )
    print(
        f"{str(t.entry_time)[:16]}  {t.side:5s}  {orig_pnl:+8.2f}  {prod_alt:+8.2f}  {best_alt:+8.2f}  {best_alt-orig_pnl:+7.2f}  {best_reason:12s}  {best_bars:>4d}      {t.exit_reason:12s}  {t.f_wide_sr_dist_atr if pd.notna(t.f_wide_sr_dist_atr) else float('nan'):.2f}"
    )
    rows.append(
        {
            "entry_time": t.entry_time,
            "side": t.side,
            "orig_pnl": orig_pnl,
            "sim_prod": prod_alt,
            "alt_pnl": best_alt,
            "delta": best_alt - orig_pnl,
            "reason_alt": best_reason,
            "bars_alt": best_bars,
        }
    )
df = pd.DataFrame(rows).dropna()
print()
print(
    f"XRP 首单 total R: orig={xrp.pnl_r.sum():+.2f}  sim-prod={df.sim_prod.sum():+.2f}  best-dynamic={df.alt_pnl.sum():+.2f}"
)
print(
    f"  n improved (best > orig): {(df.alt_pnl > df.orig_pnl).sum()}  worsened: {(df.alt_pnl < df.orig_pnl).sum()}"
)
