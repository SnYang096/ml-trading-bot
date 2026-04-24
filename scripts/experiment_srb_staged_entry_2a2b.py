#!/usr/bin/env python3
"""
SRB 两段式入场离线实验：1 记录 SR 穿越 → 2a 状态机站稳 → 2b EMA1200 位置+斜率同向。

目的（用户 2026-04-24）：
  - 量化 2b 相对「仅 2a」少发多少信号、多少在 post-2a 等待内永远达不到 2b
  - 量化入场价漂移（相对穿越根 close / 相对 2a 根 close），用 ATR 归一
  - 前向质量：同一批事件上对比 bar0 / 2a / 2b 入场点的前向 MFE·MAE·收益·阶梯命中（ATR 当 1R）
  - 可选：与 rolling sim 母仓 entry 对照，看「若等 2b」会推迟几根 bar

数据：feature_store/features_srb_120T_*/<SYM>/120T/*.parquet（120T OHLC + ema_1200_position）。
SR 窄窗：与 event 一致，用 ``swing_sr_levels(df, ts, lookback=20)`` 的 **上一根** 收盘后的 support/resistance
（与 ``check_srb_cross_events.py`` 因果顺序一致）。

2b 定义（可调 CLI）：
  LONG : ema_1200_position > pos_min 且 (pos - pos.shift(slope_bars)) > slope_min
  SHORT: ema_1200_position < -pos_min 且 (pos - pos.shift(slope_bars)) < -slope_min

用法示例：
  python scripts/experiment_srb_staged_entry_2a2b.py \\
    --feature-store feature_store/features_srb_120T_5643a66b47 \\
    --trades-root results/srb/slow-rolling-sim/_rolling_sim/20260422_212338
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.live.srb_cross_state_machine import (  # noqa: E402
    CrossCandidate,
    CrossConfig,
    CrossDecision,
    update_cross_state,
)
from src.time_series_model.live.srb_regime import swing_sr_levels  # noqa: E402


@dataclass
class Event2a:
    symbol: str
    ts: pd.Timestamp
    bar_idx: int
    bar0_idx: int
    side: str
    level: float
    close_2a: float
    atr: float  # ATR @ 2a 确认根（与实盘 initial_risk 不同，仅作可比归一）


@dataclass
class Event2a2b(Event2a):
    iloc_2a: int = 0
    iloc_2b: int = 0
    close_bar0: float = 0.0
    atr_bar0: float = 0.0
    atr_2b: float = 0.0
    bars_wait_after_2a: int = 0
    close_2b: float = 0.0
    drift_atr_from_bar0: float = 0.0
    drift_atr_from_2a: float = 0.0


@dataclass
class Event2aFailed:
    """2a 已确认但未等到 2b（timeout 或等待中穿回 level）。"""

    symbol: str
    iloc_2a: int
    side: str
    close_2a: float
    atr_2a: float
    close_bar0: float
    atr_bar0: float
    reason: str  # "timeout" | "abort_wrong_side"


@dataclass
class RunStats:
    n_bars: int = 0
    n_2a: int = 0
    n_2a2b_same_bar: int = 0
    n_2a2b_delayed: int = 0
    n_2a_no_2b_timeout: int = 0
    n_2a_abort_wrong_side: int = 0
    events_2a2b: List[Event2a2b] = field(default_factory=list)
    events_2a_failed: List[Event2aFailed] = field(default_factory=list)


def _pick_feature_store(cli: str) -> Path:
    cli = (cli or "").strip()
    if cli:
        p = Path(cli)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_dir():
            return p
    cand = sorted(PROJECT_ROOT.glob("feature_store/features_srb_120T_*"))
    dirs = [x for x in cand if x.is_dir()]
    if not dirs:
        raise SystemExit("no features_srb_120T_* directory found")
    # 选 parquet 最多的目录
    best = max(dirs, key=lambda d: sum(1 for _ in d.glob("*/120T/*.parquet")))
    return best


def _load_symbol_bars(store: Path, symbol: str) -> pd.DataFrame:
    sym_dir = store / symbol / "120T"
    if not sym_dir.is_dir():
        return pd.DataFrame()
    parts = sorted(sym_dir.glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    need = {"open", "high", "low", "close", "ema_1200_position", "atr"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{symbol}: missing columns {miss}")
    df["volume_ma"] = df["volume"].rolling(20, min_periods=1).mean().shift(1)
    return df


def _safe_atr(df: pd.DataFrame, i: int) -> float:
    v = float(df["atr"].iloc[max(0, min(len(df) - 1, i))])
    if not np.isfinite(v) or v <= 0 or v > 1e6:
        return float("nan")
    return max(v, 1e-9)


def forward_profile(
    df: pd.DataFrame,
    iloc0: int,
    side: str,
    entry: float,
    atr_unit: float,
    horizons: Tuple[int, ...],
) -> Dict[int, Dict[str, float]]:
    """前向路径：用 entry 当根收盘价，atr_unit 归一（≈1R=1ATR 位移，非真实 initial_risk）。"""
    out: Dict[int, Dict[str, float]] = {}
    if (
        iloc0 < 0
        or iloc0 >= len(df)
        or not np.isfinite(atr_unit)
        or atr_unit <= 0
        or atr_unit > 1e6
        or not np.isfinite(entry)
    ):
        return out
    is_long = str(side).upper() in ("LONG", "BUY")
    au = float(atr_unit)
    for H in horizons:
        i1 = min(len(df) - 1, iloc0 + H)
        if i1 <= iloc0:
            continue
        chunk = df.iloc[iloc0 + 1 : i1 + 1]
        if chunk.empty:
            continue
        hi = chunk["high"].astype(float)
        lo = chunk["low"].astype(float)
        if is_long:
            mfe_px = float((hi - entry).max())
            mae_px = float((entry - lo).max())
            ret_h = (float(df["close"].iloc[i1]) - entry) / au
        else:
            mfe_px = float((entry - lo).max())
            mae_px = float((hi - entry).max())
            ret_h = (entry - float(df["close"].iloc[i1])) / au
        mfe_r = mfe_px / au
        mae_r = mae_px / au
        out[int(H)] = {
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "ret_r": ret_h,
            "hit05": float(mfe_r >= 0.5),
            "hit10": float(mfe_r >= 1.0),
            "hit15": float(mfe_r >= 1.5),
            "hit30": float(mfe_r >= 3.0),
        }
    return out


def _forward_row(
    symbol: str,
    df: pd.DataFrame,
    iloc0: int,
    side: str,
    entry: float,
    atr_unit: float,
    horizons: Tuple[int, ...],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"symbol": symbol}
    ae = abs(float(entry)) if np.isfinite(entry) else 0.0
    au = float(atr_unit) if np.isfinite(atr_unit) else float("nan")
    # 极小 ATR 会把 MFE_R 放大到无意义（低价币 / 脏行）
    if (not np.isfinite(au)) or au < max(1e-10, 1e-7 * max(ae, 1e-12)):
        row["_skip"] = True
        return row
    fp = forward_profile(df, iloc0, side, entry, au, horizons)
    for H, d in fp.items():
        if abs(d["mfe_r"]) > 500 or abs(d["ret_r"]) > 500:
            row["_skip"] = True
            return row
        row[f"mfe_{H}"] = d["mfe_r"]
        row[f"ret_{H}"] = d["ret_r"]
        row[f"h05_{H}"] = d["hit05"]
        row[f"h15_{H}"] = d["hit15"]
        row[f"h30_{H}"] = d["hit30"]
    return row


def _trim(xs: List[float], lim: float = 80.0) -> np.ndarray:
    a = np.array([x for x in xs if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return a
    return a[np.abs(a) <= lim]


def _cohort_forward_table(
    name: str,
    rows: List[Dict[str, Any]],
    horizons: Tuple[int, ...],
) -> None:
    rows = [r for r in rows if not r.get("_skip")]
    if not rows:
        print(f"  [{name}] n=0")
        return
    print(f"  [{name}] n={len(rows)}")
    for H in horizons:
        mfe_raw = [r.get(f"mfe_{H}") for r in rows if f"mfe_{H}" in r]
        ret_raw = [r.get(f"ret_{H}") for r in rows if f"ret_{H}" in r]
        mfe = _trim(mfe_raw, 80.0)
        ret = _trim(ret_raw, 80.0)
        if mfe.size == 0:
            continue
        h05 = np.mean(
            [
                r.get(f"h05_{H}", float("nan"))
                for r in rows
                if np.isfinite(r.get(f"h05_{H}", float("nan")))
            ]
        )
        h15 = np.mean(
            [
                r.get(f"h15_{H}", float("nan"))
                for r in rows
                if np.isfinite(r.get(f"h15_{H}", float("nan")))
            ]
        )
        h30 = np.mean(
            [
                r.get(f"h30_{H}", float("nan"))
                for r in rows
                if np.isfinite(r.get(f"h30_{H}", float("nan")))
            ]
        )
        n_t = len(mfe_raw)
        print(
            f"    H={H:3d} bars: MFE_R mean/med={np.mean(mfe):+.3f}/{np.median(mfe):+.3f}  "
            f"ret_R mean/med={np.mean(ret):+.3f}/{np.median(ret):+.3f}  "
            f"(|·|≤80 截尾 n={mfe.size}/{n_t})  "
            f"P(hit≥0.5R)={h05*100:.1f}%  P(hit≥1.5R)={h15*100:.1f}%  P(hit≥3R)={h30*100:.1f}%"
        )


def _ema_trend_ok(
    df: pd.DataFrame,
    i: int,
    side: str,
    slope_bars: int,
    pos_min: float,
    slope_min: float,
) -> bool:
    """i = iloc index (0-based) for current bar."""
    if i < slope_bars:
        return False
    pos = float(df["ema_1200_position"].iloc[i])
    prev = float(df["ema_1200_position"].iloc[i - slope_bars])
    if not (pos == pos and prev == prev):
        return False
    d = pos - prev
    su = str(side).upper()
    if su in ("LONG", "BUY"):
        return pos > pos_min and d > slope_min
    if su in ("SHORT", "SELL"):
        return pos < -pos_min and d < -slope_min
    return False


def replay_symbol(
    symbol: str,
    df: pd.DataFrame,
    cfg: CrossConfig,
    slope_bars: int,
    pos_min: float,
    slope_min: float,
    post_2a_max_bars: int,
) -> RunStats:
    st = RunStats()
    st.n_bars = len(df)
    cand: Optional[CrossCandidate] = None
    cooldown_until = 0
    last_close: Optional[float] = None
    last_sup: Optional[float] = None
    last_res: Optional[float] = None
    bar_idx = 0

    post2a: Optional[Dict[str, Any]] = None

    for i, (ts, row) in enumerate(df.iterrows()):
        bar_idx += 1
        cur_close = float(row["close"])
        sup, res = swing_sr_levels(df, ts, 20)

        if post2a is not None:
            lvl = float(post2a["level"])
            side = str(post2a["side"])
            is_up = side in ("LONG", "BUY")
            on_ok = (cur_close > lvl) if is_up else (cur_close < lvl)
            if not on_ok:
                st.n_2a_abort_wrong_side += 1
                if post2a.get("rec_2a"):
                    r2 = post2a["rec_2a"]
                    st.events_2a_failed.append(
                        Event2aFailed(
                            symbol=symbol,
                            iloc_2a=int(r2["iloc_2a"]),
                            side=str(r2["side"]),
                            close_2a=float(r2["close_2a"]),
                            atr_2a=float(r2["atr_2a"]),
                            close_bar0=float(r2["close_bar0"]),
                            atr_bar0=float(r2["atr_bar0"]),
                            reason="abort_wrong_side",
                        )
                    )
                post2a = None
                cooldown_until = bar_idx + cfg.cooldown_bars
                last_close = cur_close
                last_sup, last_res = sup, res
                continue
            if bar_idx > int(post2a["deadline_bar"]):
                st.n_2a_no_2b_timeout += 1
                if post2a.get("rec_2a"):
                    r2 = post2a["rec_2a"]
                    st.events_2a_failed.append(
                        Event2aFailed(
                            symbol=symbol,
                            iloc_2a=int(r2["iloc_2a"]),
                            side=str(r2["side"]),
                            close_2a=float(r2["close_2a"]),
                            atr_2a=float(r2["atr_2a"]),
                            close_bar0=float(r2["close_bar0"]),
                            atr_bar0=float(r2["atr_bar0"]),
                            reason="timeout",
                        )
                    )
                post2a = None
                cooldown_until = bar_idx + cfg.cooldown_bars
                last_close = cur_close
                last_sup, last_res = sup, res
                continue
            if _ema_trend_ok(df, i, side, slope_bars, pos_min, slope_min):
                atr = (
                    float(row["atr"])
                    if pd.notna(row.get("atr"))
                    else float(df["atr"].iloc[max(0, i - 1)])
                )
                atr = max(atr, 1e-9)
                if atr > 1e6 or not np.isfinite(atr):
                    post2a = None
                    cooldown_until = bar_idx + cfg.cooldown_bars
                    last_close = cur_close
                    last_sup, last_res = sup, res
                    continue
                b0 = int(post2a["bar0_idx"])
                c0 = float(post2a["close_bar0"])
                c2a = float(post2a["close_2a"])
                b2a = int(post2a["bar_2a_idx"])
                il2a = int(post2a["iloc_2a"])
                atr_2a = float(post2a["atr_2a"])
                ab0 = max(float(post2a["atr_bar0"]), 1e-9)
                drift0 = (cur_close - c0) / ab0
                drift2a = (cur_close - c2a) / atr_2a
                atr2b = max(atr, 1e-9)
                ev = Event2a2b(
                    symbol=symbol,
                    ts=ts,
                    bar_idx=bar_idx,
                    bar0_idx=b0,
                    side=side,
                    level=lvl,
                    close_2a=c2a,
                    atr=atr_2a,
                    iloc_2a=il2a,
                    iloc_2b=i,
                    close_bar0=c0,
                    atr_bar0=float(post2a["atr_bar0"]),
                    atr_2b=atr2b,
                    bars_wait_after_2a=bar_idx - b2a,
                    close_2b=cur_close,
                    drift_atr_from_bar0=drift0,
                    drift_atr_from_2a=drift2a,
                )
                st.events_2a2b.append(ev)
                # post2a 路径只在「2a 当根未过 2b」之后发生 → 等待至少 1 根 bar
                st.n_2a2b_delayed += 1
                post2a = None
                cand = None
                cooldown_until = bar_idx + cfg.cooldown_bars
            last_close = cur_close
            last_sup, last_res = sup, res
            continue

        old_cand = cand
        new_cand, dec = update_cross_state(
            candidate=cand,
            bar_index=bar_idx,
            close_prev=float(last_close) if last_close is not None else cur_close,
            close_curr=cur_close,
            support=last_sup,
            resistance=last_res,
            has_position=False,
            cfg=cfg,
            cooldown_until_bar=cooldown_until,
            open_px=float(row.get("open", cur_close)),
            high_px=float(row.get("high", cur_close)),
            low_px=float(row.get("low", cur_close)),
            volume=float(row.get("volume", 0.0)),
            volume_ma=(
                float(row["volume_ma"]) if pd.notna(row.get("volume_ma")) else None
            ),
        )
        cand = new_cand
        last_close = cur_close
        last_sup, last_res = sup, res

        if dec.status == "confirmed" and old_cand is not None:
            st.n_2a += 1
            side = str(dec.side)
            c2a = cur_close
            b0 = int(old_cand.bar0)
            ix0 = max(0, min(len(df) - 1, b0 - 1))
            c0 = float(df["close"].iloc[ix0])
            atr = (
                float(row["atr"])
                if pd.notna(row.get("atr"))
                else float(df["atr"].iloc[i])
            )
            atr = max(atr, 1e-9)
            if atr > 1e6 or not np.isfinite(atr):
                cand = None
                cooldown_until = bar_idx + cfg.cooldown_bars
                continue
            if _ema_trend_ok(df, i, side, slope_bars, pos_min, slope_min):
                ab0 = _safe_atr(df, ix0)
                if not np.isfinite(ab0):
                    ab0 = atr
                drift0 = (cur_close - c0) / ab0
                drift2a = 0.0
                atr2b = max(atr, 1e-9)
                ev = Event2a2b(
                    symbol=symbol,
                    ts=ts,
                    bar_idx=bar_idx,
                    bar0_idx=b0,
                    side=side,
                    level=float(dec.level or 0.0),
                    close_2a=c2a,
                    atr=atr,
                    iloc_2a=i,
                    iloc_2b=i,
                    close_bar0=c0,
                    atr_bar0=ab0,
                    atr_2b=atr2b,
                    bars_wait_after_2a=0,
                    close_2b=cur_close,
                    drift_atr_from_bar0=drift0,
                    drift_atr_from_2a=drift2a,
                )
                st.events_2a2b.append(ev)
                st.n_2a2b_same_bar += 1
                cand = None
                cooldown_until = bar_idx + cfg.cooldown_bars
            else:
                ix0w = max(0, min(len(df) - 1, b0 - 1))
                c0w = float(df["close"].iloc[ix0w])
                ab0w = _safe_atr(df, ix0w)
                if not np.isfinite(ab0w):
                    ab0w = atr
                post2a = {
                    "side": side,
                    "level": float(dec.level or 0.0),
                    "bar0_idx": b0,
                    "close_bar0": c0w,
                    "atr_bar0": ab0w,
                    "bar_2a_idx": bar_idx,
                    "close_2a": c2a,
                    "iloc_2a": i,
                    "atr_2a": atr,
                    "deadline_bar": bar_idx + post_2a_max_bars,
                    "rec_2a": {
                        "iloc_2a": i,
                        "side": side,
                        "close_2a": c2a,
                        "atr_2a": atr,
                        "close_bar0": c0w,
                        "atr_bar0": ab0w,
                    },
                }
                cand = None
                cooldown_until = 0
        elif dec.status in ("fake", "expired"):
            cand = None
            cooldown_until = bar_idx + cfg.cooldown_bars

    return st


def _load_mother_trades(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("fast_month_*/srb/event_trades_srb.csv"))
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in paths]
    t = pd.concat(dfs, ignore_index=True)
    t = t[~t.get("is_add_position", False).astype(bool)].copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-store",
        default="",
        help="SRB 120T parquet 根目录；默认可自动选 features_srb_120T_* 中文件最多的",
    )
    ap.add_argument(
        "--trades-root",
        default="",
        help="rolling sim 根目录（含 fast_month_*/srb/event_trades_srb.csv），用于母仓列表",
    )
    ap.add_argument(
        "--symbols",
        default="",
        help="逗号分隔；空则 trades-root 里出现过的全部 symbol",
    )
    ap.add_argument("--confirm-k", type=int, default=3)
    ap.add_argument("--fake-lookahead", type=int, default=10)
    ap.add_argument("--cooldown-bars", type=int, default=10)
    ap.add_argument("--post-2a-max-bars", type=int, default=24)
    ap.add_argument("--ema-slope-bars", type=int, default=2)
    ap.add_argument("--ema-pos-min", type=float, default=0.0)
    ap.add_argument("--ema-slope-min", type=float, default=0.0)
    ap.add_argument(
        "--forward-horizons",
        default="60,120,240,480",
        help="前向统计 horizon（根 120T bar），逗号分隔",
    )
    args = ap.parse_args()

    store = _pick_feature_store(args.feature_store)
    print(f"feature_store: {store}")

    symbols: List[str] = []
    mothers = pd.DataFrame()
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.trades_root:
        mothers = _load_mother_trades(Path(args.trades_root))
        if not mothers.empty:
            symbols = sorted(mothers["symbol"].astype(str).unique().tolist())
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "BNBUSDT"]

    cfg = CrossConfig(
        confirm_k=max(1, args.confirm_k),
        fake_lookahead=max(1, args.fake_lookahead),
        cooldown_bars=max(0, args.cooldown_bars),
    )

    agg = RunStats()
    per_sym: Dict[str, RunStats] = {}
    sym_dfs: Dict[str, pd.DataFrame] = {}
    horizons = tuple(
        int(x.strip()) for x in args.forward_horizons.split(",") if x.strip()
    )

    for sym in symbols:
        df = _load_symbol_bars(store, sym)
        if df.empty:
            print(f"[skip] {sym}: no parquet")
            continue
        sym_dfs[sym] = df
        st = replay_symbol(
            sym,
            df,
            cfg,
            slope_bars=max(1, args.ema_slope_bars),
            pos_min=float(args.ema_pos_min),
            slope_min=float(args.ema_slope_min),
            post_2a_max_bars=max(1, args.post_2a_max_bars),
        )
        per_sym[sym] = st
        agg.n_bars += st.n_bars
        agg.n_2a += st.n_2a
        agg.n_2a2b_same_bar += st.n_2a2b_same_bar
        agg.n_2a2b_delayed += st.n_2a2b_delayed
        agg.n_2a_no_2b_timeout += st.n_2a_no_2b_timeout
        agg.n_2a_abort_wrong_side += st.n_2a_abort_wrong_side
        agg.events_2a2b.extend(st.events_2a2b)
        agg.events_2a_failed.extend(st.events_2a_failed)

    print("\n=== 汇总（全 symbol 事件流） ===")
    print(f"bars scanned: {agg.n_bars}")
    print(f"2a confirmed (SM): {agg.n_2a}")
    n2b = len(agg.events_2a2b)
    print(f"2a+2b opened:      {n2b}")
    if agg.n_2a > 0:
        print(
            f"  same-bar 2b:    {agg.n_2a2b_same_bar} ({100*agg.n_2a2b_same_bar/agg.n_2a:.1f}% of 2a)"
        )
        print(
            f"  delayed 2b:     {agg.n_2a2b_delayed} ({100*agg.n_2a2b_delayed/agg.n_2a:.1f}% of 2a)"
        )
        print(
            f"  2a→timeout无2b: {agg.n_2a_no_2b_timeout} ({100*agg.n_2a_no_2b_timeout/agg.n_2a:.1f}% of 2a)"
        )
        print(
            f"  2a→abort穿回:  {agg.n_2a_abort_wrong_side} ({100*agg.n_2a_abort_wrong_side/agg.n_2a:.1f}% of 2a)"
        )
    if agg.events_2a2b:
        dr0 = np.array(
            [
                e.drift_atr_from_bar0
                for e in agg.events_2a2b
                if np.isfinite(e.drift_atr_from_bar0)
            ]
        )
        dr0 = dr0[np.abs(dr0) < 50]  # 去掉 ATR 病态导致的离群
        dr2 = np.array(
            [
                e.drift_atr_from_2a
                for e in agg.events_2a2b
                if e.bars_wait_after_2a > 0 and np.isfinite(e.drift_atr_from_2a)
            ]
        )
        dr2 = dr2[np.abs(dr2) < 50]
        w8 = [e.bars_wait_after_2a for e in agg.events_2a2b if e.bars_wait_after_2a > 0]
        print("\n=== 入场价漂移（ATR 归一，|drift|<50 截尾） ===")
        if len(dr0):
            print(
                f"drift bar0→2b entry (all opens): mean={np.mean(dr0):+.3f}  "
                f"p50={np.percentile(dr0,50):+.3f}  p90={np.percentile(dr0,90):+.3f} ATR  n={len(dr0)}"
            )
        if len(dr2):
            print(
                f"drift 2a→2b (delayed only):     mean={np.mean(dr2):+.3f}  "
                f"p50={np.percentile(dr2,50):+.3f}  p90={np.percentile(dr2,90):+.3f} ATR  n={len(dr2)}"
            )
            print(
                f"bars wait after 2a (delayed):   mean={np.mean(w8):.2f}  "
                f"p90={np.percentile(w8,90):.1f}"
            )

    print("\n=== per-symbol ===")
    for sym, st in sorted(per_sym.items()):
        n2b = len(st.events_2a2b)
        r = 100 * n2b / st.n_2a if st.n_2a else 0.0
        print(
            f"  {sym}: 2a={st.n_2a}  2a+2b={n2b} ({r:.1f}%)  "
            f"to={st.n_2a_no_2b_timeout} abort={st.n_2a_abort_wrong_side}"
        )

    # ---------- 前向质量（ATR 归一 1R；非真实 initial_risk） ----------
    rows_all_2a: List[Dict[str, Any]] = []
    rows_pass_2a: List[Dict[str, Any]] = []
    rows_fail_2a: List[Dict[str, Any]] = []
    rows_bar0_pass: List[Dict[str, Any]] = []
    rows_2b_pass: List[Dict[str, Any]] = []
    rows_2b_delayed: List[Dict[str, Any]] = []
    rows_2b_same: List[Dict[str, Any]] = []

    for ev in agg.events_2a2b:
        df = sym_dfs.get(ev.symbol)
        if df is None or df.empty:
            continue
        rows_pass_2a.append(
            _forward_row(
                ev.symbol,
                df,
                ev.iloc_2a,
                ev.side,
                float(ev.close_2a),
                float(ev.atr),
                horizons,
            )
        )
        rows_all_2a.append(rows_pass_2a[-1])
        il0 = max(0, min(len(df) - 1, int(ev.bar0_idx) - 1))
        rows_bar0_pass.append(
            _forward_row(
                ev.symbol,
                df,
                il0,
                ev.side,
                float(ev.close_bar0),
                float(ev.atr_bar0),
                horizons,
            )
        )
        rows_2b_pass.append(
            _forward_row(
                ev.symbol,
                df,
                ev.iloc_2b,
                ev.side,
                float(ev.close_2b),
                float(ev.atr_2b),
                horizons,
            )
        )
        if ev.bars_wait_after_2a > 0:
            rows_2b_delayed.append(rows_2b_pass[-1])
        else:
            rows_2b_same.append(rows_2b_pass[-1])

    for fx in agg.events_2a_failed:
        df = sym_dfs.get(fx.symbol)
        if df is None or df.empty:
            continue
        r = _forward_row(
            fx.symbol,
            df,
            fx.iloc_2a,
            fx.side,
            float(fx.close_2a),
            float(fx.atr_2a),
            horizons,
        )
        rows_fail_2a.append(r)
        rows_all_2a.append(r)

    print("\n=== 前向路径质量（cohort × horizon） ===")
    print(
        "说明：MFE_R / ret_R 以「入场根 ATR」为 1R；≈ 浮盈阶梯量级，"
        "不等于实盘 pnl_r（无手续费、无 SL 结构）。"
    )
    _cohort_forward_table(
        "A 全部 2a（含未过 2b 的 timeout/abort）", rows_all_2a, horizons
    )
    _cohort_forward_table("B 仅未过 2b（timeout + 穿回）@2a", rows_fail_2a, horizons)
    _cohort_forward_table(
        "C 最终过 2b 的子集 @2a（若仍 2a 根入场）", rows_pass_2a, horizons
    )
    _cohort_forward_table("D 同上子集 @bar0（穿越根）", rows_bar0_pass, horizons)
    _cohort_forward_table("E 同上子集 @2b（延迟/同根真实入场）", rows_2b_pass, horizons)
    _cohort_forward_table("E1 其中同根 2b", rows_2b_same, horizons)
    _cohort_forward_table("E2 其中延迟 2b", rows_2b_delayed, horizons)

    n_chk = len(agg.events_2a2b) + len(agg.events_2a_failed)
    if n_chk != agg.n_2a:
        print(
            f"\n[warn] 2a 计数核对: n_2a={agg.n_2a} vs 2b+failed={n_chk} "
            f"(差值多为 atr 无效跳过等)"
        )

    # 母仓时间对齐：离每笔母仓 entry 最近的一次 2a+2b 是否在之前 D 根内
    if not mothers.empty and agg.events_2a2b:
        evdf = pd.DataFrame(
            [
                {
                    "symbol": e.symbol,
                    "ts": e.ts,
                    "side": e.side,
                    "bars_wait": e.bars_wait_after_2a,
                    "drift0": e.drift_atr_from_bar0,
                }
                for e in agg.events_2a2b
            ]
        )
        evdf["ts"] = pd.to_datetime(evdf["ts"], utc=True)
        # 120T bar = 2h；窗口内才认为与「该笔母仓」可能同源（SRB 实盘方向多为 MACD，
        # cross 状态机仅为独立重放，全局最近事件会严重错位）
        win_bars = 36  # 72h
        sec_per_bar = 120 * 60
        rows_tight: List[Dict[str, Any]] = []
        for _, tr in mothers.iterrows():
            sym = str(tr["symbol"])
            te = tr["entry_time"]
            sub = evdf[(evdf["symbol"] == sym) & (evdf["side"] == tr["side"])]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["dt_sec"] = (te - sub["ts"]).dt.total_seconds()
            sub = sub[(sub["dt_sec"] >= 0) & (sub["dt_sec"] <= win_bars * sec_per_bar)]
            if sub.empty:
                continue
            j = sub["dt_sec"].idxmin()
            rows_tight.append(
                {
                    "symbol": sym,
                    "pnl_r": float(tr["pnl_r"]),
                    "bars_before_entry": float(sub.loc[j, "dt_sec"] / sec_per_bar),
                    "staged_wait": int(sub.loc[j, "bars_wait"]),
                    "drift0": float(sub.loc[j, "drift0"]),
                }
            )
        if rows_tight:
            jn = pd.DataFrame(rows_tight)
            print(
                f"\n=== rolling 母仓 vs 2a+2b（同 symbol+side，entry 前 ≤{win_bars} 根 120T 内最近事件）==="
            )
            print(f"matched mothers: {len(jn)} / {len(mothers)}")
            print(
                f"  staged_wait>0 占比: {(jn['staged_wait']>0).mean()*100:.1f}%  "
                f"bars_before 分布: mean={jn['bars_before_entry'].mean():.2f}  "
                f"p50={jn['bars_before_entry'].median():.1f}  p90={jn['bars_before_entry'].quantile(0.9):.1f}"
            )
            win = jn[jn["pnl_r"] > 0]
            lose = jn[jn["pnl_r"] <= 0]
            print(
                f"  winners n={len(win)} mean pnl_r={win['pnl_r'].mean():+.3f} | "
                f"losers n={len(lose)} mean pnl_r={lose['pnl_r'].mean():+.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
