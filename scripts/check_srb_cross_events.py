"""Quick sanity check: do cross events fire on real 2h BTCUSDT data now that we use
prior-bar SR levels?"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.time_series_model.live.srb_cross_state_machine import (
    CrossConfig,
    update_cross_state,
)
from src.time_series_model.live.srb_regime import swing_sr_levels


def main() -> None:
    fs_root = Path(
        "/home/yin/trading/ml_trading_bot/feature_store/"
        "features_srb_120T_0a4b849ebd/BTCUSDT/120T"
    )
    parts = [fs_root / f"2023-{m:02d}.parquet" for m in (8, 9, 10)]
    parts = [p for p in parts if p.exists()]
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs).sort_index()
    print("columns:", df.columns.tolist()[:30])
    print("rows:", len(df))
    print(
        df[["open", "high", "low", "close"]].head(2)
        if "close" in df.columns
        else df.head(2)
    )

    cfg = CrossConfig()
    cand = None
    last_close = None
    last_sup = None
    last_res = None
    cooldown = 0
    bar_idx = 0
    confirmed = 0
    fake = 0
    expired = 0
    pending_max = 0
    for ts, row in df.iterrows():
        bar_idx += 1
        cur_close = float(row["close"])
        sup, res = swing_sr_levels(df, ts, 20)
        new_cand, dec = update_cross_state(
            candidate=cand,
            bar_index=bar_idx,
            close_prev=float(last_close) if last_close is not None else cur_close,
            close_curr=cur_close,
            support=last_sup,
            resistance=last_res,
            has_position=False,
            cfg=cfg,
            cooldown_until_bar=cooldown,
            open_px=float(row.get("open", cur_close)),
            high_px=float(row.get("high", cur_close)),
            low_px=float(row.get("low", cur_close)),
            volume=float(row.get("volume", 0)) if "volume" in row else None,
            volume_ma=None,
        )
        cand = new_cand
        last_close = cur_close
        last_sup = sup
        last_res = res
        if dec.status == "confirmed":
            confirmed += 1
            cooldown = bar_idx + cfg.cooldown_bars
            print(
                f"  [{ts}] bar={bar_idx} CONFIRMED side={dec.side} level={dec.level:.2f} close={cur_close:.2f}"
            )
        elif dec.status == "fake":
            fake += 1
            cooldown = bar_idx + cfg.cooldown_bars
            print(
                f"  [{ts}] bar={bar_idx} FAKE side={dec.side} level={dec.level:.2f} close={cur_close:.2f}"
            )
        elif dec.status == "expired":
            expired += 1
            cooldown = bar_idx + cfg.cooldown_bars
    print(f"total bars={bar_idx} confirmed={confirmed} fake={fake} expired={expired}")


if __name__ == "__main__":
    main()
