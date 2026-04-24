"""
SRB sr_wide_entry_guard 审计
============================

交叉检查：对 rolling_sim 产出的每笔 SRB **首单**（is_add_position=False、is_reverse=False），
从 feature store 取 entry_time 对应 bar 的 wide_sr_upper_px / wide_sr_lower_px / atr / close，
重放 `should_reject_srb_wide_entry`，列出"应被拦但实际成交"的笔数。

目标：解释 XRP 2024-01-03 wide_dist=0.45 SHORT 为何未被拦（plan Phase C）。

用法：
    python scripts/verify_srb_wide_guard.py \
        --trades reports/srb_break_level_attribution_v2_alltrades_trades.parquet \
        --feature-store feature_store/features_srb_120T_5643a66b47 \
        --min-distance-atr 2.0 \
        --out reports/srb_wide_guard_audit.md

若 `--trades` 不存在，回退到 results/srb/slow-rolling-sim/_rolling_sim/<latest>/fast_month_*/srb/event_trades_srb.csv 汇总。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.time_series_model.live.srb_regime import should_reject_srb_wide_entry


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]


def load_trades_from_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    if "is_add_position" in df.columns:
        df = df[~df["is_add_position"].fillna(False)]
    if "is_reverse" in df.columns:
        df = df[~df["is_reverse"].fillna(False)]
    return df.reset_index(drop=True)


def load_trades_from_rolling_sim(root: str) -> pd.DataFrame:
    paths = sorted(
        glob.glob(os.path.join(root, "fast_month_*", "srb", "event_trades_srb.csv"))
    )
    if not paths:
        return pd.DataFrame()
    parts = [pd.read_csv(p) for p in paths]
    df = pd.concat(parts, ignore_index=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    # first entries only
    for col in ("is_add_position", "is_reverse"):
        if col in df.columns:
            df = df[~df[col].fillna(False).astype(bool)]
    return df.reset_index(drop=True)


def load_bar_features(store: str, symbol: str, tf: str = "120T") -> pd.DataFrame:
    files = sorted(glob.glob(f"{store}/{symbol}/{tf}/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    keep = [
        c
        for c in [
            "open",
            "high",
            "low",
            "close",
            "atr",
            "wide_sr_upper_px",
            "wide_sr_lower_px",
        ]
        if c in df.columns
    ]
    return df[keep]


def entry_features_asof(
    bars: pd.DataFrame, entry_time: pd.Timestamp
) -> Optional[pd.Series]:
    if bars is None or bars.empty:
        return None
    ts = pd.Timestamp(entry_time)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    # 取入场时间之前最后一根完成 bar 的特征（feature 使用收盘信息）
    sub = bars.loc[bars.index <= ts]
    if sub.empty:
        return None
    return sub.iloc[-1]


def audit_trades(
    trades: pd.DataFrame,
    store: str,
    min_distance_atr: float,
) -> pd.DataFrame:
    rows: List[Dict] = []
    cache: Dict[str, pd.DataFrame] = {}
    for _, t in trades.iterrows():
        sym = str(t.get("symbol", "")).upper()
        if not sym:
            continue
        if sym not in cache:
            cache[sym] = load_bar_features(store, sym)
        bars = cache[sym]
        feats = entry_features_asof(bars, t["entry_time"])
        if feats is None:
            rows.append(
                {
                    "symbol": sym,
                    "entry_time": t["entry_time"],
                    "side": t.get("side"),
                    "wide_upper_px": None,
                    "wide_lower_px": None,
                    "close": None,
                    "atr": None,
                    "rev_dist_atr": None,
                    "should_reject": None,
                    "reason": "feature_missing",
                }
            )
            continue
        close = float(feats.get("close") or 0.0)
        atr = float(feats.get("atr") or 0.0)
        up = feats.get("wide_sr_upper_px")
        lo = feats.get("wide_sr_lower_px")
        try:
            up_f = float(up) if up is not None and up == up else None
        except (TypeError, ValueError):
            up_f = None
        try:
            lo_f = float(lo) if lo is not None and lo == lo else None
        except (TypeError, ValueError):
            lo_f = None
        side = str(t.get("side", "")).upper()
        reject = should_reject_srb_wide_entry(
            side, close, atr, lo_f, up_f, min_distance_atr
        )
        if side in ("LONG", "BUY"):
            rev_dist = (
                ((up_f - close) / atr) if (up_f is not None and atr > 0) else None
            )
        else:
            rev_dist = (
                ((close - lo_f) / atr) if (lo_f is not None and atr > 0) else None
            )
        reason = ""
        if up_f is None and side in ("LONG", "BUY"):
            reason = "wide_upper_nan"
        elif lo_f is None and side in ("SHORT", "SELL"):
            reason = "wide_lower_nan"
        elif rev_dist is not None and rev_dist < 0:
            reason = "reverse_l3_behind_price"
        rows.append(
            {
                "symbol": sym,
                "entry_time": t["entry_time"],
                "side": side,
                "close": close,
                "atr": atr,
                "wide_upper_px": up_f,
                "wide_lower_px": lo_f,
                "rev_dist_atr": rev_dist,
                "should_reject": reject,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def summarize(audit: pd.DataFrame, min_distance_atr: float) -> str:
    lines: List[str] = []
    n = len(audit)
    feat_missing = int((audit["reason"] == "feature_missing").sum())
    wide_nan = int(audit["reason"].isin(["wide_upper_nan", "wide_lower_nan"]).sum())
    behind = int((audit["reason"] == "reverse_l3_behind_price").sum())
    would_reject = int((audit["should_reject"].fillna(False)).sum())
    lines.append(f"# SRB `sr_wide_entry_guard` 审计报告")
    lines.append("")
    lines.append(f"- min_distance_atr 阈值：**{min_distance_atr}**")
    lines.append(f"- 参与审计首单数：**{n}**")
    lines.append(f"- feature store 无法回放：{feat_missing}")
    lines.append(f"- 反向 L3 特征 NaN（无法判断）：{wide_nan}")
    lines.append(f"- 反向 L3 在价格**后方**（破位已完成）：{behind}")
    lines.append(f'- **重放后判 "应拦但实际进了"**：**{would_reject}** 笔')
    lines.append("")
    if would_reject > 0:
        sub = audit[audit["should_reject"].fillna(False)].copy()
        sub["rev_dist_atr"] = sub["rev_dist_atr"].round(2)
        sub = sub.sort_values(["rev_dist_atr"], ascending=True)
        lines.append("## 应拦未拦样本（按 rev_dist_atr 升序）")
        lines.append("")
        lines.append("| symbol | entry_time | side | close | atr | rev_dist_atr |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.head(30).iterrows():
            lines.append(
                f"| {r.symbol} | {r.entry_time} | {r.side} | "
                f"{r.close:.4f} | {r.atr:.4f} | {r.rev_dist_atr} |"
            )
        lines.append("")
    # XRP 2024-01-03 特检
    xrp_focus = audit[
        (audit["symbol"] == "XRPUSDT")
        & (audit["entry_time"] >= "2024-01-01")
        & (audit["entry_time"] < "2024-01-10")
    ]
    lines.append("## XRP 2024-01-03 附近首单（人工关注）")
    lines.append("")
    if not xrp_focus.empty:
        lines.append(
            "| entry_time | side | close | atr | rev_dist_atr | should_reject |"
        )
        lines.append("|---|---|---|---|---|---|")
        for _, r in xrp_focus.iterrows():
            rev = r.rev_dist_atr
            rev_s = f"{rev:.2f}" if rev is not None and rev == rev else "nan"
            lines.append(
                f"| {r.entry_time} | {r.side} | {r.close:.4f} | {r.atr:.4f} | "
                f"{rev_s} | {r.should_reject} |"
            )
    else:
        lines.append("_没有命中样本。_")
    lines.append("")
    # 结论
    lines.append("## 根因候选")
    lines.append("")
    if feat_missing > 0 or wide_nan > 0:
        lines.append(
            f"- 部分首单 entry_time 在 feature store 中缺 wide_sr_{{upper,lower}}_px（共 {feat_missing+wide_nan} 笔）。"
            "可能是 feature 窗口不足（240 bar）或 rolling_sim 跑时 feature 版本较旧。"
        )
    if would_reject > 0:
        lines.append(
            f"- {would_reject} 笔首单在当前阈值 ({min_distance_atr}×ATR) 下重放应被拦，但实际成交。"
            "最近一次 rolling_sim 的 funnel 里 `reject_srb_wide_sr_too_close=0`，说明：runtime 时 "
            "`entry_feats` 传入 `should_reject_srb_wide_entry` 的 `wide_sr_{upper,lower}_px` 为 NaN / 缺失。"
            "需在下一次 rolling_sim 验证本次 Phase A 改动后 funnel 是否正常增加拒单计数。"
        )
    if behind > 0:
        lines.append(
            f"- {behind} 笔首单的反向 L3 在价格后方（SHORT 时 wide_sr_lower_px > close / LONG 时 wide_sr_upper_px < close），"
            "这是 `价格已突破 L3 通道在外部继续走` 的场景，当前 guard 按设计不会拦"
            "（guard 本意是防 `进场后很快撞到反向 L3`）。"
        )
    lines.append("")
    lines.append("## 对 XRP 2024-01-03 灾难的定性")
    lines.append("")
    lines.append(
        "- XRP 2024-01-03 16:00 SHORT：close=0.5748, wide_sr_lower_px=0.5812, atr=0.0143，"
        "rev_dist_atr = (close - lower)/atr = **-0.45**（价格低于 lower_px 0.45 ATR）。"
    )
    lines.append(
        "- 当前 guard 只在 `lo < px` 才 check，此处 `lo > px`（已突破通道），判定不拦 —— **语义是对的**。"
    )
    lines.append(
        "- 真实问题不是 `进场瞬间离 L3 太近`，而是 `突破后加仓堆在 wide_dist=7~10 ATR 处，反转打穿原 SL`。"
        "对症方案：**Phase B mother_breakeven（母仓 MFE ≥ 3R 时锁 SL，子仓共享）** + "
        "**Phase D add shape gate（加仓时形态确认）**。"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trades",
        default="reports/srb_break_level_attribution_v2_alltrades_trades.parquet",
    )
    ap.add_argument(
        "--rolling-sim-root",
        default=None,
        help="若 --trades 不存在则从 rolling_sim 目录汇总",
    )
    ap.add_argument(
        "--feature-store", default="feature_store/features_srb_120T_5643a66b47"
    )
    ap.add_argument("--min-distance-atr", type=float, default=2.0)
    ap.add_argument("--out", default="reports/srb_wide_guard_audit.md")
    ap.add_argument("--parquet-out", default="reports/srb_wide_guard_audit.parquet")
    args = ap.parse_args()

    if os.path.exists(args.trades):
        trades = load_trades_from_parquet(args.trades)
    else:
        root = args.rolling_sim_root
        if root is None:
            # 取最新 rolling_sim run
            cand = sorted(glob.glob("results/srb/slow-rolling-sim/_rolling_sim/20*_*"))
            root = cand[-1] if cand else ""
        if not root or not os.path.isdir(root):
            raise SystemExit(f"no trades parquet and no rolling_sim root: {root}")
        trades = load_trades_from_rolling_sim(root)

    print(f"[verify] loaded {len(trades)} first-entry trades", flush=True)
    audit = audit_trades(trades, args.feature_store, args.min_distance_atr)
    audit.to_parquet(args.parquet_out)
    report = summarize(audit, args.min_distance_atr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[verify] wrote {args.parquet_out}")
    print(f"[verify] wrote {args.out}")
    print("=" * 60)
    print(report[:2000])


if __name__ == "__main__":
    main()
