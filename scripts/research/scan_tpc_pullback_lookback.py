#!/usr/bin/env python3
"""Scan historical 120T bars: TPC pullback depth vs macro price drawdown by lookback.

Answers: which lookback_breakout makes tpc_pullback_depth align with
「大周期回调」(price drawdown from swing high), before running event_backtest grids.

Usage:
  PYTHONPATH=src:scripts python scripts/research/scan_tpc_pullback_lookback.py
  PYTHONPATH=src:scripts python scripts/research/scan_tpc_pullback_lookback.py \\
    --symbols BTCUSDT,SOLUSDT --out results/tpc/research/pullback_lookback_scan_20260608
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

LOOKBACKS = (20, 60, 120, 240, 480)
MACRO_ANCHOR = 240  # bars on 120T ≈ 20d — reference for 「大回调」
MACRO_LOOKBACK = 240  # default N for tpc_macro_pullback_pct_f
DRAWDOWN_THRESHOLDS = (0.10, 0.15, 0.20)
REBOUND_THRESHOLDS = (0.10, 0.12, 0.15)
DEPTH_THRESHOLDS = (0.35, 0.45, 0.50, 0.55)
MACRO_PCT_TAU_GRID = tuple(round(x, 2) for x in np.arange(0.10, 0.31, 0.02))


def _load_bars(
    *,
    symbols: list[str],
    data_path: str,
    start: str,
    end: str,
    timeframe: str = "120T",
) -> pd.DataFrame:
    from src.data_tools.data_handler import DataHandler

    dh = DataHandler(data_path=data_path)
    parts: list[pd.DataFrame] = []
    for sym in symbols:
        raw = dh.load_ohlcv(symbol=sym, timeframe=timeframe)
        raw = raw.loc[start:end]
        if raw.empty:
            continue
        df = raw.copy()
        df["symbol"] = sym
        parts.append(df)
    if not parts:
        raise RuntimeError("no bars loaded")
    return pd.concat(parts).sort_index()


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev).abs(),
            (low - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _ema_position(close: pd.Series, span: int = 1200) -> pd.Series:
    ema = close.ewm(span=span, adjust=False, min_periods=max(3, span // 10)).mean()
    return (close - ema) / close.replace(0, np.nan)


def _pullback_depth_long(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int
) -> pd.Series:
    rh = high.rolling(lookback, min_periods=1).max().shift(1)
    rl = low.rolling(lookback, min_periods=1).min().shift(1)
    rng = (rh - rl).clip(lower=1e-8)
    return ((rh - close) / rng).clip(0, 1)


def _drawdown_pct_from_high(
    high: pd.Series, close: pd.Series, lookback: int
) -> pd.Series:
    rh = high.rolling(lookback, min_periods=1).max().shift(1)
    return ((rh - close) / rh.replace(0, np.nan)).clip(0, 1)


def _rebound_pct_from_low(low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    rl = low.rolling(lookback, min_periods=1).min().shift(1)
    return ((close - rl) / rl.replace(0, np.nan)).clip(0, 1)


def _macro_pullback_long(
    high: pd.Series, close: pd.Series, lookback: int = MACRO_LOOKBACK
) -> pd.Series:
    return _drawdown_pct_from_high(high, close, lookback)


def _macro_pullback_short(
    low: pd.Series, close: pd.Series, lookback: int = MACRO_LOOKBACK
) -> pd.Series:
    return _rebound_pct_from_low(low, close, lookback)


def _bull_mask(ema_pos: pd.Series) -> pd.Series:
    return ema_pos >= 0.10


def _bear_mask(ema_pos: pd.Series) -> pd.Series:
    return ema_pos <= -0.10


def _segment_mask(index: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC")
    if index.tz is None:
        t0 = t0.tz_localize(None)
        t1 = t1.tz_localize(None)
    return (index >= t0) & (index < t1)


def _pct(s: pd.Series, q: float) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return float("nan")
    return float(v.quantile(q))


def _best_tau_row(
    sub: pd.DataFrame,
    *,
    feat_col: str,
    truth_col: str,
    truth_th: float,
    side: str,
) -> tuple[dict | None, list[dict]]:
    truth = sub[truth_col] >= truth_th
    best: dict | None = None
    rows: list[dict] = []
    for tau in MACRO_PCT_TAU_GRID:
        hit = sub[feat_col] >= tau
        if not hit.any():
            continue
        prec = float(truth[hit].mean())
        recall = float(hit[truth].mean()) if truth.any() else 0.0
        row = {
            "side": side,
            "tau": tau,
            "truth_ge": truth_th,
            "n_hit": int(hit.sum()),
            "precision": prec,
            "recall": recall,
        }
        rows.append(row)
        if best is None or (prec, recall) > (best["precision"], best["recall"]):
            best = row
    return best, rows


def _scan_macro_pct_calibration(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_pos: pd.Series,
) -> dict:
    bull = _bull_mask(ema_pos)
    bear = _bear_mask(ema_pos)
    macro_dd = _drawdown_pct_from_high(high, close, MACRO_ANCHOR)
    macro_rebound = _rebound_pct_from_low(low, close, MACRO_ANCHOR)
    long_pct = _macro_pullback_long(high, close, MACRO_LOOKBACK)
    short_pct = _macro_pullback_short(low, close, MACRO_LOOKBACK)

    bull_sub = pd.DataFrame(
        {
            "long_pct": long_pct[bull],
            "macro_dd": macro_dd[bull],
        }
    ).dropna()
    bear_sub = pd.DataFrame(
        {
            "short_pct": short_pct[bear],
            "macro_rebound": macro_rebound[bear],
        }
    ).dropna()

    bull_best, bull_rows = _best_tau_row(
        bull_sub,
        feat_col="long_pct",
        truth_col="macro_dd",
        truth_th=0.15,
        side="bull",
    )
    bear_best, bear_rows = _best_tau_row(
        bear_sub,
        feat_col="short_pct",
        truth_col="macro_rebound",
        truth_th=0.12,
        side="bear",
    )

    # Prefer τ with precision≥50% and n≥50; else fall back to best F-score proxy
    def _pick(rows: list[dict], side: str) -> dict | None:
        qualified = [r for r in rows if r["precision"] >= 0.5 and r["n_hit"] >= 50]
        pool = qualified or rows
        if not pool:
            return None
        return max(pool, key=lambda r: (r["precision"], r["recall"]))

    bull_pick = _pick(bull_rows, "bull")
    bear_pick = _pick(bear_rows, "bear")

    return {
        "lookback": MACRO_LOOKBACK,
        "bull_calibration": bull_rows,
        "bear_calibration": bear_rows,
        "recommended_tau_bull": bull_pick["tau"] if bull_pick else None,
        "recommended_tau_bear": bear_pick["tau"] if bear_pick else None,
        "tau_bull_detail": bull_pick,
        "tau_bear_detail": bear_pick,
        "n_bull": int(len(bull_sub)),
        "n_bear": int(len(bear_sub)),
    }


def scan_frame(df: pd.DataFrame, *, label: str) -> dict:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    ema_pos = _ema_position(close)
    bull = _bull_mask(ema_pos)
    macro_dd = _drawdown_pct_from_high(high, close, MACRO_ANCHOR)

    rows: list[dict] = []
    align_rows: list[dict] = []

    for lb in LOOKBACKS:
        depth = _pullback_depth_long(high, low, close, lb)
        dd_same = _drawdown_pct_from_high(high, close, lb)
        m = bull & depth.notna() & macro_dd.notna()
        sub = pd.DataFrame(
            {
                "depth": depth[m],
                "macro_dd": macro_dd[m],
                "dd_same": dd_same[m],
            }
        )
        if sub.empty:
            continue
        corr_macro = float(sub["depth"].corr(sub["macro_dd"]))
        corr_same = float(sub["depth"].corr(sub["dd_same"]))

        row = {
            "lookback": lb,
            "calendar_days_120T": round(lb * 2 / 24, 1),
            "n_bull": int(len(sub)),
            "depth_p50": _pct(sub["depth"], 0.50),
            "depth_p90": _pct(sub["depth"], 0.90),
            "depth_p95": _pct(sub["depth"], 0.95),
            "corr_depth_vs_macro_dd240": corr_macro,
            "corr_depth_vs_dd_same_lb": corr_same,
        }
        for dt in DEPTH_THRESHOLDS:
            row[f"pct_depth_ge_{dt:.2f}"] = float((sub["depth"] >= dt).mean())
        rows.append(row)

        for macro_th in DRAWDOWN_THRESHOLDS:
            macro_hit = sub["macro_dd"] >= macro_th
            if not macro_hit.any():
                continue
            hit = sub.loc[macro_hit, "depth"]
            align_rows.append(
                {
                    "lookback": lb,
                    "macro_dd_ge": macro_th,
                    "n_macro_hit": int(macro_hit.sum()),
                    "depth_p50_on_macro": _pct(hit, 0.50),
                    "depth_p75_on_macro": _pct(hit, 0.75),
                    "pct_depth_ge_0.45_on_macro": float((hit >= 0.45).mean()),
                    "pct_depth_ge_0.50_on_macro": float((hit >= 0.50).mean()),
                    "pct_depth_ge_0.55_on_macro": float((hit >= 0.55).mean()),
                    "macro_dd_p50_on_hit": _pct(sub.loc[macro_hit, "macro_dd"], 0.50),
                }
            )

        for dt in DEPTH_THRESHOLDS:
            dhit = sub["depth"] >= dt
            if not dhit.any():
                continue
            align_rows.append(
                {
                    "lookback": lb,
                    "depth_ge": dt,
                    "n_depth_hit": int(dhit.sum()),
                    "macro_dd_p50_when_depth": _pct(sub.loc[dhit, "macro_dd"], 0.50),
                    "macro_dd_p75_when_depth": _pct(sub.loc[dhit, "macro_dd"], 0.75),
                    "pct_macro_dd_ge_0.15": float(
                        (sub.loc[dhit, "macro_dd"] >= 0.15).mean()
                    ),
                    "pct_macro_dd_ge_0.20": float(
                        (sub.loc[dhit, "macro_dd"] >= 0.20).mean()
                    ),
                }
            )

    # Recalibrate depth floor: for each lookback, find τ where P(macro_dd≥15% | depth≥τ) is best
    calib_rows: list[dict] = []
    depth_grid = [round(x, 2) for x in np.arange(0.05, 0.55, 0.05)]
    for lb in LOOKBACKS:
        depth = _pullback_depth_long(high, low, close, lb)
        m = bull & depth.notna() & macro_dd.notna()
        sub = pd.DataFrame({"depth": depth[m], "macro_dd": macro_dd[m]})
        if sub.empty:
            continue
        macro15 = sub["macro_dd"] >= 0.15
        best_tau: dict | None = None
        for tau in depth_grid:
            dhit = sub["depth"] >= tau
            if not dhit.any():
                continue
            prec = float(macro15[dhit].mean()) if dhit.any() else 0.0
            recall = float(dhit[macro15].mean()) if macro15.any() else 0.0
            row = {
                "lookback": lb,
                "depth_tau": tau,
                "n_depth_hit": int(dhit.sum()),
                "precision_macro15": prec,
                "recall_macro15": recall,
            }
            calib_rows.append(row)
            if best_tau is None or (prec, recall) > (
                best_tau["precision_macro15"],
                best_tau["recall_macro15"],
            ):
                best_tau = row

    best_lb = max(
        (r for r in rows if r["lookback"] in LOOKBACKS),
        key=lambda r: r["corr_depth_vs_macro_dd240"],
        default=None,
    )

    macro_pct = _scan_macro_pct_calibration(
        high=high, low=low, close=close, ema_pos=ema_pos
    )

    return {
        "label": label,
        "n_bars": int(len(df)),
        "distribution": rows,
        "alignment": align_rows,
        "calibration": calib_rows,
        "macro_pct_calibration": macro_pct,
        "recommendation": {
            "macro_anchor_bars": MACRO_ANCHOR,
            "best_lookback_by_corr_macro240": best_lb["lookback"] if best_lb else None,
            "best_depth_tau_per_lookback": {
                str(lb): next(
                    (
                        c
                        for c in reversed(calib_rows)
                        if c["lookback"] == lb
                        and c["precision_macro15"] >= 0.5
                        and c["n_depth_hit"] >= 50
                    ),
                    None,
                )
                for lb in LOOKBACKS
            },
            "tau_bull": macro_pct.get("recommended_tau_bull"),
            "tau_bear": macro_pct.get("recommended_tau_bear"),
            "note": (
                "depth 是区间位置分位(0-1)，不是价格回撤%；"
                "macro≥15% 时 depth 很少超过 0.5；应重标定 τ 或改用 macro_pullback_pct"
            ),
        },
    }


def _render_md(payload: dict) -> str:
    lines = [
        "# TPC pullback lookback 历史扫描",
        "",
        f"**窗口**: {payload.get('window')}",
        f"**标的**: {', '.join(payload.get('symbols') or [])}",
        f"**宏观回撤锚定**: roll_high_{MACRO_ANCHOR}（120T ≈ {MACRO_ANCHOR * 2 / 24:.0f} 天）",
        f"**牛市过滤**: ema_1200_position >= 0.10",
        "",
        "## 1. depth 分布（牛市子样本）",
        "",
        "| lookback | 日历天 | depth p50 | p90 | corr(macro_dd) | P(depth≥0.5) |",
        "|----------|--------|-----------|-----|----------------|--------------|",
    ]
    for seg in payload.get("segments") or []:
        lines.append(f"\n### {seg['label']}\n")
        for r in seg.get("distribution") or []:
            lines.append(
                f"| {r['lookback']} | {r['calendar_days_120T']} | "
                f"{r['depth_p50']:.3f} | {r['depth_p90']:.3f} | "
                f"{r['corr_depth_vs_macro_dd240']:.3f} | "
                f"{r.get('pct_depth_ge_0.50', 0)*100:.1f}% |"
            )
        rec = seg.get("recommendation") or {}
        lines.append(
            f"\n**推荐 lookback（与 macro_dd_240 相关性最高）**: "
            f"**{rec.get('best_lookback_by_corr_macro240')}**\n"
        )
        lines.append(f"_{rec.get('note', '')}_\n")
        lines.append("\n#### depth 下界重标定（macro_dd≥15% 时 precision≥50%）\n")
        lines.append("| lookback | 建议 depth≥τ | precision | recall | n_hit |")
        lines.append("|----------|--------------|-----------|--------|-------|")
        for lb in LOOKBACKS:
            c = (rec.get("best_depth_tau_per_lookback") or {}).get(str(lb))
            if not c:
                lines.append(f"| {lb} | — | — | — | — |")
                continue
            lines.append(
                f"| {lb} | {c['depth_tau']:.2f} | "
                f"{c['precision_macro15']*100:.0f}% | "
                f"{c['recall_macro15']*100:.0f}% | {c['n_depth_hit']} |"
            )
        lines.append("#### macro_dd ≥ 15% 时 depth 表现\n")
        lines.append("| lookback | n | depth p50 | P(depth≥0.5) | P(depth≥0.55) |")
        lines.append("|----------|---|-----------|--------------|---------------|")
        for a in seg.get("alignment") or []:
            if a.get("macro_dd_ge") != 0.15:
                continue
            lines.append(
                f"| {a['lookback']} | {a['n_macro_hit']} | "
                f"{a['depth_p50_on_macro']:.3f} | "
                f"{a['pct_depth_ge_0.50_on_macro']*100:.1f}% | "
                f"{a['pct_depth_ge_0.55_on_macro']*100:.1f}% |"
            )
        lines.append("\n#### depth ≥ 0.5 时 macro_dd 表现\n")
        lines.append("| lookback | n | macro_dd p50 | P(macro≥15%) | P(macro≥20%) |")
        lines.append("|----------|---|--------------|--------------|--------------|")
        for a in seg.get("alignment") or []:
            if a.get("depth_ge") != 0.50:
                continue
            lines.append(
                f"| {a['lookback']} | {a['n_depth_hit']} | "
                f"{a['macro_dd_p50_when_depth']:.3f} | "
                f"{a['pct_macro_dd_ge_0.15']*100:.1f}% | "
                f"{a['pct_macro_dd_ge_0.20']*100:.1f}% |"
            )
    lines.append("\n## 2. 解读\n")
    lines.append(
        "- **corr(macro_dd)** 越高，该 lookback 的 depth 越贴近「从 swing high 跌了多少」。\n"
        "- 若 **depth≥0.5 时 P(macro≥15%)** 仍很低，说明应用 **macro_pullback_pct** 或更长 lookback。\n"
        "- 若 **macro≥15% 时 P(depth≥0.5)** 随 lookback 单调升，取拐点 lookback 作为实验候选。\n"
    )
    lines.append("\n## 3. macro_pullback_pct 阈值标定（N=240）\n")
    lines.append("| segment | side | 建议 τ | precision | recall | n_hit | n_sample |")
    lines.append("|---------|------|--------|-----------|--------|-------|----------|")
    for seg in payload.get("segments") or []:
        mc = seg.get("macro_pct_calibration") or {}
        rec = seg.get("recommendation") or {}
        for side, key, detail_key in (
            ("bull", "recommended_tau_bull", "tau_bull_detail"),
            ("bear", "recommended_tau_bear", "tau_bear_detail"),
        ):
            d = mc.get(detail_key)
            if not d:
                lines.append(
                    f"| {seg['label']} | {side} | — | — | — | — | "
                    f"{mc.get('n_bull' if side == 'bull' else 'n_bear', '—')} |"
                )
                continue
            lines.append(
                f"| {seg['label']} | {side} | {d['tau']:.2f} | "
                f"{d['precision']*100:.0f}% | {d['recall']*100:.0f}% | "
                f"{d['n_hit']} | "
                f"{mc.get('n_bull' if side == 'bull' else 'n_bear')} |"
            )
        lines.append(
            f"\n**{seg['label']} 推荐 prefilter**: "
            f"long≥**{rec.get('tau_bull')}**, short≥**{rec.get('tau_bear')}**\n"
        )
    lines.append(
        "\n实验初稿变体：`M240_L15_S12` / `M240_L20_S15` / `M240_L15_lb240`（对照 soft_phase lb=240）。\n"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,SOLUSDT")
    ap.add_argument("--data-path", default="data/parquet_data")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-04-01")
    ap.add_argument(
        "--out",
        default="results/tpc/research/macro_pullback_scan_20260609",
    )
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {symbols} 120T {args.start} → {args.end} ...")
    bars = _load_bars(
        symbols=symbols,
        data_path=args.data_path,
        start=args.start,
        end=args.end,
    )

    segments = [
        ("full", args.start, args.end),
        ("bull_2023_2024", "2023-01-01", "2025-01-01"),
        ("recent_2025", "2025-01-01", args.end),
    ]
    seg_payloads: list[dict] = []
    for name, s0, s1 in segments:
        mask = _segment_mask(bars.index, s0, s1)
        sub = bars.loc[mask]
        if sub.empty:
            continue
        print(f"  scan {name}: {len(sub)} bars ...")
        seg_payloads.append(scan_frame(sub, label=name))

    payload = {
        "symbols": symbols,
        "window": f"{args.start}/{args.end}",
        "lookbacks": list(LOOKBACKS),
        "segments": seg_payloads,
    }
    json_path = out_dir / "pullback_lookback_scan.json"
    md_path = out_dir / "pullback_lookback_scan.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(_render_md(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    for seg in seg_payloads:
        rec = seg.get("recommendation") or {}
        print(
            f"[{seg['label']}] best lookback (corr macro240): {rec.get('best_lookback_by_corr_macro240')}"
        )
        print(f"  macro_pct τ_bull={rec.get('tau_bull')} τ_bear={rec.get('tau_bear')}")
        for lb in LOOKBACKS:
            c = (rec.get("best_depth_tau_per_lookback") or {}).get(str(lb))
            if c:
                print(
                    f"  LB={lb}: depth>={c['depth_tau']:.2f} "
                    f"prec={c['precision_macro15']*100:.0f}% "
                    f"recall={c['recall_macro15']*100:.0f}%"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
