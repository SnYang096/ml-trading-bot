"""exp07 Paper Trading Engine — 离线 walk-forward dry-run。

不连交易所，只做"决策 + 记账"：每次 invocation 读 as-of 时间点的数据、
算 composite score、按 preset 或 regime_weights 决定持仓，和 account_state
对比算 orders，然后把新持仓写回 account_state。上一次持仓的 PnL 用 as-of 时
最新价结算（真正的 walk-forward）。

命令:
    init      初始化一个空账户
    rebalance 按 as-of 时间运行一次决策
    status    打印当前持仓快照（也可单独用 update_pnl.py）

输出 (reports/cross_section/exp07_paper/<name>/):
    account_state.json     当前虚拟持仓 + equity
    trade_log.jsonl        append-only 每次决策记录
    equity_history.parquet 历次 rebalance 后权益
    latest_decision.txt    最近一次人眼可读输出
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from ..exp02_multi_factor.backtester import FactorSpec, build_composite_score
from ..exp02_multi_factor.data_loader import build_panels
from ..exp02_multi_factor.sectors import get_sectors
from ..exp04_small_account.config import (
    ACCOUNT_SIZE_USD,
    FEE_BPS_PER_SIDE,
    HOLD_BARS_DEFAULT,
    LIQUID_POOL,
    MAX_LONGS,
    MAX_SHORTS,
    STOP_LOSS_PER_LEG,
)
from ..exp04_small_account.run import PRESETS
from ..exp05_regime_ic.regimes import compute_regime_labels


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    entry_time: str
    entry_price: float
    notional_usd: float
    target_weight: float
    scheduled_exit_time: Optional[str] = None
    stop_loss_per_leg: float = STOP_LOSS_PER_LEG


@dataclass
class AccountState:
    name: str
    account_size_usd: float
    equity_usd: float
    realized_pnl_usd: float
    preset: str
    use_regime_switch: bool
    weights_yaml_path: Optional[str]
    positions: Dict[str, Dict] = field(
        default_factory=dict
    )  # symbol -> Position as dict
    last_rebalance_time: Optional[str] = None
    hold_bars: int = HOLD_BARS_DEFAULT
    stop_loss_per_leg: float = STOP_LOSS_PER_LEG
    default_price_source: str = "parquet"  # parquet | binance_futures

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "AccountState":
        kwargs: Dict = {}
        for f in dataclasses.fields(cls):
            if f.name in d:
                kwargs[f.name] = d[f.name]
            elif f.default_factory is not dataclasses.MISSING:
                kwargs[f.name] = f.default_factory()
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            else:
                raise KeyError(f.name)
        return cls(**kwargs)


def _paper_dir(name: str) -> Path:
    return Path("reports/cross_section/exp07_paper") / name


def _load_state(name: str) -> AccountState:
    p = _paper_dir(name) / "account_state.json"
    if not p.exists():
        raise SystemExit(f"account state not found: {p}  (先跑 init)")
    return AccountState.from_dict(json.loads(p.read_text()))


def _save_state(state: AccountState) -> None:
    outdir = _paper_dir(state.name)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "account_state.json").write_text(
        json.dumps(state.to_dict(), indent=2, default=str)
    )


def _append_trade_log(name: str, record: Dict) -> None:
    p = _paper_dir(name) / "trade_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _load_data_up_to(
    as_of: pd.Timestamp,
    price_dir: Path,
    funding_dir: Path,
    lookback_bars: int,
    timeframe: str = "1h",
) -> Dict[str, pd.DataFrame]:
    """加载 as_of 前后足够多的历史 + 前瞻数据（用于结算当前持仓）。"""
    start = (as_of - pd.Timedelta(hours=lookback_bars * 2)).strftime("%Y-%m")
    end = as_of.strftime("%Y-%m")
    panels = build_panels(
        LIQUID_POOL, start, end, price_dir, funding_dir, timeframe, 0.3, verbose=False
    )
    for k in ("prices", "returns", "funding"):
        panels[k] = panels[k].loc[:as_of]
    return panels


def _specs_from_preset(preset: str) -> List[FactorSpec]:
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset: {preset}  (choices: {list(PRESETS)})")
    return PRESETS[preset]


def _specs_from_regime(weights_yaml: Dict, regime: str) -> List[FactorSpec]:
    factor_specs = weights_yaml["factor_specs"]
    reg_conf = weights_yaml["regime_weights"].get(
        regime, weights_yaml["regime_weights"].get("ALL")
    )
    out = []
    for name, w in reg_conf["factors"].items():
        spec = factor_specs.get(name)
        if spec is None:
            continue
        out.append(
            FactorSpec(
                name=name,
                kind=spec["kind"],
                lookback=int(spec["lookback"]),
                skip=int(spec.get("skip", 0)),
                weight=float(w),
            )
        )
    return out


def cmd_init(args):
    name = args.name
    outdir = _paper_dir(name)
    if outdir.exists() and (outdir / "account_state.json").exists() and not args.force:
        raise SystemExit(f"{outdir} already exists; pass --force to overwrite")
    state = AccountState(
        name=name,
        account_size_usd=args.account_size,
        equity_usd=args.account_size,
        realized_pnl_usd=0.0,
        preset=args.preset,
        use_regime_switch=args.use_regime_switch,
        weights_yaml_path=args.weights_yaml if args.use_regime_switch else None,
        hold_bars=args.hold_bars,
        stop_loss_per_leg=args.stop_loss_per_leg,
        default_price_source=args.default_price_source,
    )
    _save_state(state)
    print(
        f"[init] {name}: account={args.account_size} preset={args.preset} "
        f"regime_switch={args.use_regime_switch}"
    )
    print(f"       state -> {outdir / 'account_state.json'}")


def _settle_positions(
    state: AccountState, prices: pd.DataFrame, as_of: pd.Timestamp
) -> float:
    """用 as_of 最新价结算当前持仓（模拟收到交易所成交回报），返回已实现 PnL。"""
    if not state.positions:
        return 0.0
    total_pnl = 0.0
    closed = []
    for sym, pos_d in state.positions.items():
        pos = Position(**pos_d)
        if sym not in prices.columns:
            print(f"  [warn] {sym} not in prices, skipping settle")
            continue
        px_ser = prices[sym].dropna()
        if px_ser.empty:
            continue
        last_px = float(px_ser.iloc[-1])
        side_sign = 1.0 if pos.side == "long" else -1.0
        ret = (last_px / pos.entry_price - 1.0) * side_sign
        pnl = pos.notional_usd * ret
        fee = pos.notional_usd * FEE_BPS_PER_SIDE / 1e4 * 2  # in + out
        pnl -= fee
        total_pnl += pnl
        closed.append(
            {
                "symbol": sym,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": last_px,
                "entry_time": pos.entry_time,
                "exit_time": as_of.isoformat(),
                "notional_usd": pos.notional_usd,
                "ret_pct": ret * 100,
                "pnl_usd": pnl,
                "fee_usd": fee,
            }
        )
    for c in closed:
        state.positions.pop(c["symbol"], None)
    state.realized_pnl_usd += total_pnl
    state.equity_usd += total_pnl
    state.__closed_this_run__ = closed  # 非持久化，仅 log 用
    return total_pnl


def cmd_rebalance(args):
    state = _load_state(args.name)
    as_of = (
        pd.Timestamp(args.as_of)
        if args.as_of
        else pd.Timestamp.utcnow().replace(tzinfo=None)
    )

    # 选 factor specs
    if state.use_regime_switch:
        if not state.weights_yaml_path:
            raise SystemExit("use_regime_switch=True but weights_yaml_path not set")
        weights_yaml = yaml.safe_load(Path(state.weights_yaml_path).read_text())
        factor_specs_map = weights_yaml["factor_specs"]
        max_lb = max(
            int(v["lookback"]) + int(v.get("skip", 0))
            for v in factor_specs_map.values()
        )
    else:
        specs = _specs_from_preset(state.preset)
        max_lb = max(s.lookback + s.skip for s in specs)

    print(f"[rebalance] as_of={as_of}  name={state.name}")
    panels = _load_data_up_to(
        as_of,
        Path(args.price_dir),
        Path(args.funding_dir),
        lookback_bars=max_lb + 24 * 30,
        timeframe=args.timeframe,
    )
    prices = panels["prices"]
    returns = panels["returns"].fillna(0.0)
    funding = panels["funding"].fillna(0.0)
    print(f"    data: {returns.shape[0]} bars, {returns.shape[1]} symbols")

    # 1. 结算上一期持仓
    realized = _settle_positions(state, prices, as_of)
    print(
        f"    settled prev positions: realized_pnl=${realized:+.2f}  equity=${state.equity_usd:,.2f}"
    )

    # 2. 识别 regime 并选 specs
    regime_label = None
    if state.use_regime_switch:
        regimes = compute_regime_labels(prices, funding)
        if len(regimes) > 0:
            regime_label = regimes["collapsed"].iloc[-1]
            print(f"    regime={regime_label}")
        else:
            regime_label = "range_normal"
        specs = _specs_from_regime(weights_yaml, regime_label)

    # 3. 计算 composite score
    sectors = get_sectors(list(returns.columns))
    score = build_composite_score(
        returns, funding, specs, sectors, sector_neutral=True, winsorize_pct=0.02
    )
    last_score = score.dropna(how="all").iloc[-1].dropna()
    if len(last_score) < MAX_LONGS + MAX_SHORTS:
        print(
            f"    [abort] only {len(last_score)} scored symbols, need {MAX_LONGS+MAX_SHORTS}"
        )
        return

    ranked = last_score.sort_values(ascending=False)
    longs = ranked.head(MAX_LONGS).index.tolist()
    shorts = ranked.tail(MAX_SHORTS).index.tolist()

    # 4. 建新仓（假设即时执行，用 as_of bar 收盘价）
    per_leg_notional_long = state.equity_usd * 0.5 / max(len(longs), 1)
    per_leg_notional_short = state.equity_usd * 0.5 / max(len(shorts), 1)
    new_positions: Dict[str, Dict] = {}
    for sym in longs:
        px = float(prices[sym].dropna().iloc[-1])
        pos = Position(
            symbol=sym,
            side="long",
            entry_time=as_of.isoformat(),
            entry_price=px,
            notional_usd=per_leg_notional_long,
            target_weight=+0.5 / len(longs),
            scheduled_exit_time=(
                as_of + pd.Timedelta(hours=state.hold_bars)
            ).isoformat(),
            stop_loss_per_leg=state.stop_loss_per_leg,
        )
        new_positions[sym] = dataclasses.asdict(pos)
    for sym in shorts:
        px = float(prices[sym].dropna().iloc[-1])
        pos = Position(
            symbol=sym,
            side="short",
            entry_time=as_of.isoformat(),
            entry_price=px,
            notional_usd=per_leg_notional_short,
            target_weight=-0.5 / len(shorts),
            scheduled_exit_time=(
                as_of + pd.Timedelta(hours=state.hold_bars)
            ).isoformat(),
            stop_loss_per_leg=state.stop_loss_per_leg,
        )
        new_positions[sym] = dataclasses.asdict(pos)
    state.positions = new_positions
    state.last_rebalance_time = as_of.isoformat()

    # 5. 写日志
    closed = getattr(state, "__closed_this_run__", [])
    trade_record = {
        "timestamp": as_of.isoformat(),
        "action": "rebalance",
        "regime": regime_label,
        "preset": (
            state.preset if not state.use_regime_switch else f"regime:{regime_label}"
        ),
        "realized_pnl_usd": realized,
        "equity_usd": state.equity_usd,
        "closed": closed,
        "opened_longs": [
            {
                "symbol": s,
                "price": new_positions[s]["entry_price"],
                "notional": new_positions[s]["notional_usd"],
            }
            for s in longs
        ],
        "opened_shorts": [
            {
                "symbol": s,
                "price": new_positions[s]["entry_price"],
                "notional": new_positions[s]["notional_usd"],
            }
            for s in shorts
        ],
    }
    _append_trade_log(state.name, trade_record)

    # equity history
    hist_path = _paper_dir(state.name) / "equity_history.parquet"
    new_row = pd.DataFrame(
        [
            {
                "time": as_of,
                "equity_usd": state.equity_usd,
                "realized_pnl_usd": state.realized_pnl_usd,
                "regime": regime_label,
            }
        ]
    ).set_index("time")
    if hist_path.exists():
        hist = pd.read_parquet(hist_path)
        hist = pd.concat([hist, new_row])
    else:
        hist = new_row
    hist.to_parquet(hist_path)

    # 人眼可读输出
    lines = [
        f"# Paper trading rebalance  {as_of}",
        f"Account: {state.name}",
        f"Equity: ${state.equity_usd:,.2f}  (realized PnL cum: ${state.realized_pnl_usd:+.2f})",
        f"Regime: {regime_label or 'n/a'}  Preset: {state.preset}",
        "",
    ]
    if closed:
        lines.append("## Closed positions (this rebalance)")
        for c in closed:
            lines.append(
                f"- {c['symbol']:<10s} {c['side']:<5s}  "
                f"entry={c['entry_price']:.4f} -> exit={c['exit_price']:.4f}  "
                f"ret={c['ret_pct']:+.2f}%  PnL=${c['pnl_usd']:+.2f}"
            )
        lines.append("")
    lines.append("## New positions")
    for sym in longs + shorts:
        p = new_positions[sym]
        lines.append(
            f"- {sym:<10s} {p['side']:<5s}  "
            f"entry=${p['entry_price']:.4f}  notional=${p['notional_usd']:.0f}  "
            f"exit_scheduled={p['scheduled_exit_time'][:16]}"
        )
    (_paper_dir(state.name) / "latest_decision.txt").write_text("\n".join(lines))

    if hasattr(state, "__closed_this_run__"):
        delattr(state, "__closed_this_run__")
    _save_state(state)
    print(f"    opened {len(longs)} longs / {len(shorts)} shorts")
    print(f"    log -> {_paper_dir(state.name)}/")


def cmd_status(args):
    from .update_pnl import print_status

    print_status(
        args.name,
        as_of=args.as_of,
        price_dir=Path(args.price_dir),
        price_source=args.price_source,
        poll_sec=args.poll_sec,
        poll_max=args.poll_max,
    )


def cmd_mid_stop(args):
    """期中止损：按最新价检查各腿；--apply 时平仓并记账。"""
    from .price_source import fetch_last_prices

    state = _load_state(args.name)
    if not state.positions:
        print("[mid_stop] no open positions")
        return
    syms = list(state.positions.keys())
    src = args.price_source or state.default_price_source
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else None
    pxmap = fetch_last_prices(
        syms,
        source=src,
        price_dir=Path(args.price_dir),
        as_of=as_of_ts,
        poll_sec=args.poll_sec,
        poll_max=args.poll_max,
    )
    now = pd.Timestamp.utcnow().replace(tzinfo=None)
    triggered: List[Dict] = []
    for sym, pos_d in list(state.positions.items()):
        thr = float(pos_d.get("stop_loss_per_leg", state.stop_loss_per_leg))
        if sym not in pxmap:
            print(f"  [warn] no price for {sym} (source={src})")
            continue
        last_px, detail = pxmap[sym][0], pxmap[sym][1]
        side_sign = 1.0 if pos_d["side"] == "long" else -1.0
        ret = (last_px / pos_d["entry_price"] - 1.0) * side_sign
        if ret <= -thr:
            triggered.append(
                {
                    "symbol": sym,
                    "side": pos_d["side"],
                    "entry_price": pos_d["entry_price"],
                    "exit_price": last_px,
                    "ret_pct": ret * 100,
                    "threshold_pct": -thr * 100,
                    "notional_usd": pos_d["notional_usd"],
                    "price_detail": detail,
                }
            )

    if not triggered:
        print(
            f"[mid_stop] ok  source={src}  checked={len(syms)}  none hit "
            f"(thr={state.stop_loss_per_leg*100:.1f}% per leg)"
        )
        return

    print(f"[mid_stop] {len(triggered)} leg(s) hit stop  source={src}")
    for t in triggered:
        print(
            f"  - {t['symbol']} {t['side']} ret={t['ret_pct']:+.2f}% (thr {t['threshold_pct']:.1f}%)"
        )

    if not args.apply:
        print("  (dry-run; pass --apply to close & update account_state)")
        return

    total_pnl = 0.0
    for t in triggered:
        sym = t["symbol"]
        pos_d = state.positions.pop(sym, None)
        if not pos_d:
            continue
        ret = t["ret_pct"] / 100.0
        pnl = pos_d["notional_usd"] * ret
        fee = pos_d["notional_usd"] * FEE_BPS_PER_SIDE / 1e4 * 2
        pnl -= fee
        total_pnl += pnl
        t["pnl_usd"] = pnl
        t["fee_usd"] = fee
    state.realized_pnl_usd += total_pnl
    state.equity_usd += total_pnl
    _append_trade_log(
        state.name,
        {
            "timestamp": now.isoformat(),
            "action": "mid_stop",
            "price_source": src,
            "realized_pnl_usd": total_pnl,
            "equity_usd": state.equity_usd,
            "closed": triggered,
        },
    )
    _save_state(state)
    print(f"  applied: realized=${total_pnl:+.2f}  equity=${state.equity_usd:,.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_init = sub.add_parser("init")
    ap_init.add_argument("--name", required=True)
    ap_init.add_argument("--preset", default="mom_only", choices=list(PRESETS.keys()))
    ap_init.add_argument("--account-size", type=float, default=ACCOUNT_SIZE_USD)
    ap_init.add_argument("--hold-bars", type=int, default=HOLD_BARS_DEFAULT)
    ap_init.add_argument("--use-regime-switch", action="store_true")
    ap_init.add_argument(
        "--weights-yaml",
        default="reports/cross_section/exp05_regime_ic/regime_ic/regime_weights.yaml",
    )
    ap_init.add_argument(
        "--stop-loss-per-leg",
        type=float,
        default=STOP_LOSS_PER_LEG,
        help="单腿亏损比例触发期中止损（与回测一致默认 0.15）",
    )
    ap_init.add_argument(
        "--default-price-source",
        choices=["parquet", "binance_futures"],
        default="parquet",
    )
    ap_init.add_argument("--force", action="store_true")
    ap_init.set_defaults(func=cmd_init)

    ap_reb = sub.add_parser("rebalance")
    ap_reb.add_argument("--name", required=True)
    ap_reb.add_argument("--as-of", help="YYYY-MM-DD[ HH:MM]，默认当前 UTC")
    ap_reb.add_argument("--price-dir", default="data/parquet_data")
    ap_reb.add_argument("--funding-dir", default="data/funding_rate/parquet")
    ap_reb.add_argument("--timeframe", default="1h")
    ap_reb.set_defaults(func=cmd_rebalance)

    ap_st = sub.add_parser("status")
    ap_st.add_argument("--name", required=True)
    ap_st.add_argument("--as-of")
    ap_st.add_argument("--price-dir", default="data/parquet_data")
    ap_st.add_argument(
        "--price-source",
        choices=["parquet", "binance_futures"],
        help="覆盖 account_state.default_price_source",
    )
    ap_st.add_argument("--poll-sec", type=float, default=0.0)
    ap_st.add_argument("--poll-max", type=int, default=1)
    ap_st.set_defaults(func=cmd_status)

    ap_ms = sub.add_parser("mid_stop", help="期中止损检查（可 cron 每小时）")
    ap_ms.add_argument("--name", required=True)
    ap_ms.add_argument("--as-of", help="parquet 源下用于截取历史 K 线")
    ap_ms.add_argument("--price-dir", default="data/parquet_data")
    ap_ms.add_argument("--price-source", choices=["parquet", "binance_futures"])
    ap_ms.add_argument("--poll-sec", type=float, default=0.0, help="binance 轮询间隔秒")
    ap_ms.add_argument("--poll-max", type=int, default=1, help="轮询次数")
    ap_ms.add_argument(
        "--apply", action="store_true", help="命中则平仓并更新 account_state"
    )
    ap_ms.set_defaults(func=cmd_mid_stop)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
