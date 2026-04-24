"""update_pnl.py — 用最新价结算纸交易账户当前持仓的未实现 PnL。

默认不修改 account_state.json；可选 parquet 或 Binance 期货公开 ticker，
支持轮询（poll）多次取价。

期中止损请用: python -m ... paper_engine mid_stop --name ... [--apply]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .price_source import fetch_last_prices


def _paper_dir(name: str) -> Path:
    return Path("reports/cross_section/exp07_paper") / name


def _load_state(name: str) -> dict:
    p = _paper_dir(name) / "account_state.json"
    if not p.exists():
        raise SystemExit(f"account_state.json not found under {p.parent}")
    return json.loads(p.read_text())


def print_status(
    name: str,
    as_of: Optional[str],
    price_dir: Path,
    price_source: Optional[str] = None,
    poll_sec: float = 0.0,
    poll_max: int = 1,
):
    state = _load_state(name)
    as_of_ts = pd.Timestamp(as_of) if as_of else None
    src = price_source or state.get("default_price_source", "parquet")

    equity = state["equity_usd"]
    realized = state["realized_pnl_usd"]
    positions = state.get("positions", {})
    last_rb = state.get("last_rebalance_time")
    hold_bars = state.get("hold_bars", 336)

    lines = [
        f"# Paper trading status — {name}",
        f"Generated: {pd.Timestamp.utcnow().replace(tzinfo=None)}",
        f"As-of: {as_of_ts or 'latest'}",
        f"Price source: {src}  (poll_sec={poll_sec}, poll_max={poll_max})",
        "",
        f"- Account size: ${state['account_size_usd']:,.2f}",
        f"- Equity (last rebalance): ${equity:,.2f}",
        f"- Realized cum PnL: ${realized:+,.2f}",
        f"- Preset: {state['preset']}  use_regime_switch: {state.get('use_regime_switch')}",
        f"- Last rebalance: {last_rb}",
        "",
    ]

    if not positions:
        lines.append("No open positions.")
        text = "\n".join(lines)
        print(text)
        (_paper_dir(name) / "current_status.md").write_text(text)
        return

    syms = list(positions.keys())
    pxmap = fetch_last_prices(
        syms,
        source=src,
        price_dir=price_dir,
        as_of=as_of_ts,
        poll_sec=poll_sec,
        poll_max=poll_max,
    )

    lines.append("## Open positions (marked to latest price)")
    lines.append(
        "| symbol | side | entry_time | entry | last | ret% | unreal PnL | src |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    total_unreal = 0.0
    scheduled_times = []
    for sym, p in positions.items():
        if sym not in pxmap:
            lines.append(
                f"| {sym} | {p['side']} | {p['entry_time'][:16]} | "
                f"{p['entry_price']:.4f} | n/a | n/a | n/a | — |"
            )
            continue
        last_px, detail = pxmap[sym]
        side_sign = 1.0 if p["side"] == "long" else -1.0
        ret = (last_px / p["entry_price"] - 1.0) * side_sign
        unreal = p["notional_usd"] * ret
        total_unreal += unreal
        lines.append(
            f"| {sym} | {p['side']} | {p['entry_time'][:16]} | "
            f"{p['entry_price']:.4f} | {last_px:.4f} | "
            f"{ret*100:+.2f}% | ${unreal:+.2f} | {detail} |"
        )
        if p.get("scheduled_exit_time"):
            scheduled_times.append(pd.Timestamp(p["scheduled_exit_time"]))

    lines.append("")
    lines.append(f"**Unrealized PnL: ${total_unreal:+,.2f}**")
    lines.append(f"**Mark-to-market equity: ${equity + total_unreal:+,.2f}**")

    thr = float(state.get("stop_loss_per_leg", 0.15))
    lines.append(
        f"\nMid-stop threshold: **-{thr*100:.1f}%** per leg (see `paper_engine mid_stop`)"
    )

    if scheduled_times:
        next_exit = min(scheduled_times)
        now = as_of_ts or pd.Timestamp.utcnow().replace(tzinfo=None)
        delta = next_exit - now
        lines.append(f"\nNext scheduled rebalance/exit: {next_exit}  (in {delta})")

    text = "\n".join(lines)
    print(text)
    (_paper_dir(name) / "current_status.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--as-of", help="YYYY-MM-DD[ HH:MM]（parquet 源有效）")
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--price-source", choices=["parquet", "binance_futures"])
    ap.add_argument("--poll-sec", type=float, default=0.0)
    ap.add_argument("--poll-max", type=int, default=1)
    args = ap.parse_args()
    print_status(
        args.name,
        args.as_of,
        Path(args.price_dir),
        price_source=args.price_source,
        poll_sec=args.poll_sec,
        poll_max=args.poll_max,
    )


if __name__ == "__main__":
    main()
