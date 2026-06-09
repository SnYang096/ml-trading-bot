from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from scripts.event_backtest._bootstrap import logger
from scripts.event_backtest.features.timeline import _get_timeframe
from scripts.event_backtest.backtester import EventBacktester
from scripts.event_backtest.reporting.json_export import (
    save_json,
    save_path_efficiency_sidecar,
)
from scripts.event_backtest.reporting.trading_map import generate_trading_map_html
from scripts.event_backtest.results import BacktestResult
from scripts.event_backtest.variant_grid import run_variant_grid
from scripts.capital_report import write_capital_report_from_trades

try:
    from src.order_management.mock_binance_api import MockBinanceAPI

    OM_AVAILABLE = True
except ImportError:
    OM_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EVENT_BACKTEST_RESULTS = _REPO_ROOT / "results" / "event_backtest"


def _default_trades_csv_path(strategy_tag: str) -> Path:
    """Default trades + capital_report output under ``results/event_backtest/``."""
    out_dir = _DEFAULT_EVENT_BACKTEST_RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"event_trades_{strategy_tag}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="事件驱动回测 — 多策略 PCM 仲裁 + 1min bar 持仓管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        "-s",
        required=False,
        default=None,
        help="策略名, 逗号分隔 (例: bpc / fer)；使用 --variant-grid 时可省略",
    )
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT",
        help="逗号分隔的交易对",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="回测天数 (默认 180, 被 --start-date/--end-date 覆盖)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="回测开始日期 (YYYY-MM-DD), 覆盖 --days",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="回测结束日期 (YYYY-MM-DD), 默认 now()",
    )
    parser.add_argument(
        "--live-root",
        default="live/highcap",
        help="实盘数据根目录 (仅用于 --data-path 未指定时的 fallback)",
    )
    parser.add_argument(
        "--data-path",
        default="data/parquet_data",
        help="研究数据目录 (默认 data/parquet_data, 设为 none 使用实盘数据)",
    )
    parser.add_argument(
        "--strategies-root",
        default=None,
        help="策略配置目录 (默认 config/strategies)",
    )
    parser.add_argument(
        "--constitution-yaml",
        default="config/constitution/constitution.yaml",
        help="宪法配置路径 (默认 config/constitution/constitution.yaml)",
    )
    parser.add_argument(
        "--trades-csv",
        default=None,
        dest="trades_csv",
        metavar="PATH",
        help=(
            "成交明细 CSV 路径（列含入场审计）；省略则写入 "
            "results/event_backtest/event_trades_<策略列表>.csv"
        ),
    )
    parser.add_argument(
        "--export",
        default=None,
        dest="trades_csv",
        metavar="PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--export-add-attempts",
        default=None,
        help="导出每次加仓尝试的特征快照 CSV 路径（用于规则研究）",
    )
    parser.add_argument(
        "--capital-report",
        default=None,
        help="输出 capital_report.json/html 的目录；默认与 trades CSV 输出目录一致",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="capital report 初始资金，默认 10000",
    )
    parser.add_argument(
        "--risk-per-r",
        type=float,
        default=0.01,
        help="event pnl_r 转美元时每 1R 占初始资金比例，默认 1%%",
    )
    parser.add_argument(
        "--no-compound-sizing",
        action="store_true",
        default=False,
        help="冻结 initial-capital×risk_per_slot  sizing（legacy 对照；默认按当前 equity 复利）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="保存 JSON 结果路径",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="订单落库 SQLite 路径 (启用 order_management mock)",
    )
    parser.add_argument(
        "--trading-map",
        default=None,
        help="交易地图 HTML 输出路径 (4H K线 + 交易标记)",
    )
    parser.add_argument(
        "--map-extra-months",
        type=int,
        default=12,
        help=(
            "交易地图: 从 --data-path 向前多加载 1m 月数，仅用于 VWAP / 长窗 EMA warm-up；"
            "K 线横轴仍为回测窗。0=不向前扩展"
        ),
    )
    parser.add_argument(
        "--map-vwap-window-bars",
        type=int,
        default=1200,
        help="交易地图: 滚动典型价 VWAP 窗口 (按地图 K 线根数)",
    )
    parser.add_argument(
        "--map-long-ema-span",
        type=int,
        default=1200,
        help="交易地图: 主图叠画的 EMA(close) 周期（与策略主周期 K 线上 span 语义一致，默认 1200）",
    )
    parser.add_argument(
        "--compare-trades",
        default=None,
        help="对比交易 CSV 路径 (向量回测导出的 trades.csv), 将以蓝圆标记叠加显示",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0004,
        help="单边手续费率 (默认 0.0004 = 0.04%%%% Binance taker, 设 0 关闭)",
    )
    parser.add_argument(
        "--universe-group",
        default=None,
        help=(
            "从 universe_groups yaml 读取 symbols，格式: universe_set/group，"
            "例: starter_a/highcap，默认文件: config/download/crypto_4h_token_universe_groups.yaml"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="持仓更新用策略主周期 bar（非 1min）；R/SL 与 prod 1min 路径可能略有差异，适合 trend 类 R&D 提速",
    )
    parser.add_argument(
        "--resume-state",
        default=None,
        help="可选: 从 JSON 恢复上期未平仓状态",
    )
    parser.add_argument(
        "--dump-end-state",
        default=None,
        help="可选: 导出本期结束时未平仓状态 JSON",
    )
    parser.add_argument(
        "--keep-open-positions",
        action="store_true",
        default=False,
        help="不在回测结束时强平，保留未平仓用于下一期续跑",
    )
    parser.add_argument(
        "--no-kill-switch",
        action="store_true",
        default=False,
        help="禁用 constitution kill switch（用于诊断策略真实表现，不受亏损限额约束）",
    )
    parser.add_argument(
        "--inject-add-ml-scores",
        default=None,
        help=(
            "Parquet with columns: symbol, timestamp (UTC), add_ml_score — "
            "merged into primary features as 'add_ml_score' for add_regime_gate rules"
        ),
    )
    parser.add_argument(
        "--quiet-signal-logs",
        action="store_true",
        default=False,
        help="降低逐信号日志级别（不影响回测逻辑，仅减少 stdout IO）",
    )
    parser.add_argument(
        "--variant-grid",
        default=None,
        metavar="YAML",
        help="YAML grid of variants; runs each then updates EXPERIMENT_INDEX.json",
    )
    args, extra = parser.parse_known_args()

    if args.variant_grid:
        grid_path = Path(args.variant_grid)
        return run_variant_grid(grid_path, extra_argv=extra)
    if not args.strategy:
        parser.error("--strategy is required unless --variant-grid is set")
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.quiet_signal_logs:
        logging.getLogger("src.time_series_model.live.generic_live_strategy").setLevel(
            logging.WARNING
        )
        logging.getLogger("src.time_series_model.portfolio.live_pcm").setLevel(
            logging.WARNING
        )

    strategies = [s.strip() for s in args.strategy.split(",")]
    strategy_keys = {s.strip().lower() for s in strategies if s.strip()}

    spot_cfg: Dict[str, Any] = {}
    try:
        const_obj = yaml.safe_load(
            Path(args.constitution_yaml).read_text(encoding="utf-8")
        )
        if not isinstance(const_obj, dict):
            const_obj = {}
        spot_obj = const_obj.get("spot") or {}
        if isinstance(spot_obj, dict):
            raw_spot_strategies = spot_obj.get("strategies")
            if isinstance(raw_spot_strategies, str):
                spot_strategies = {
                    p.strip().lower()
                    for p in raw_spot_strategies.split(",")
                    if p.strip()
                }
            elif isinstance(raw_spot_strategies, (list, tuple)):
                spot_strategies = {
                    str(x).strip().lower()
                    for x in raw_spot_strategies
                    if str(x).strip()
                }
            else:
                spot_strategies = set()
            if strategy_keys & spot_strategies:
                spot_cfg = dict(spot_obj)
    except Exception:
        spot_cfg = {}

    initial_capital = float(args.initial_capital)
    if "--initial-capital" not in sys.argv and spot_cfg:
        spot_account = spot_cfg.get("account") or {}
        if isinstance(spot_account, dict):
            from src.live_data_stream.constitution_config import (
                spot_account_equity_anchor_usdt,
            )

            bt_anchor = spot_account_equity_anchor_usdt(spot_account, default=0.0)
            if bt_anchor > 0:
                initial_capital = bt_anchor
                print(
                    f"  资金锚点: spot.account.equity_usdt={initial_capital:.2f} "
                    "(未显式传 --initial-capital)"
                )

    # 解析 symbols：--universe-group 优先，其次 --symbols
    if args.universe_group:
        import yaml as _yaml

        _ug_file = (
            Path(__file__).resolve().parents[2]
            / "config"
            / "download"
            / "crypto_4h_token_universe_groups.yaml"
        )
        _ug_data = _yaml.safe_load(_ug_file.read_text(encoding="utf-8"))
        _parts = args.universe_group.split("/")
        if len(_parts) != 2:
            parser.error(
                "--universe-group 格式应为 universe_set/group，例: starter_a/highcap"
            )
        _universe_set, _group = _parts
        _tokens = _ug_data["universe_sets"][_universe_set]["groups"][_group]
        _quote = _ug_data.get("quote", "USDT")
        symbols = [f"{t}{_quote}" for t in _tokens]
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]

    print("=" * 72)
    print("  🔬 事件驱动回测 (多策略 PCM 仲裁)")
    print("=" * 72)
    print(f"  策略:    {', '.join(strategies)}")
    print(f"  Symbols: {symbols}")
    print(f"  天数:    {args.days}")
    fee_pct = args.fee_rate * 100
    print(
        f"  手续费:  {fee_pct:.2f}% 单边 ({fee_pct*2:.2f}% 双边)"
        if args.fee_rate > 0
        else "  手续费:  关闭"
    )
    # --data-path none → 显式使用实盘数据做验证
    if args.data_path and args.data_path.lower() == "none":
        args.data_path = None

    if args.data_path:
        print(f"  数据源:  {args.data_path} (研究数据)")
    else:
        print(f"  数据源:  {args.live_root}/data (实盘数据, 验证模式)")
    if args.db:
        print(f"  订单落库: {args.db}")
    if args.trading_map:
        print(f"  交易地图: {args.trading_map}")
    if args.resume_state:
        print(f"  恢复状态: {args.resume_state}")
    if args.dump_end_state:
        print(f"  导出状态: {args.dump_end_state}")
    print("=" * 72)

    resume_state_obj = None
    if args.resume_state:
        _rp = Path(args.resume_state)
        if _rp.exists():
            resume_state_obj = json.loads(_rp.read_text(encoding="utf-8"))
        else:
            logger.warning("resume state not found: %s", _rp)

    bt = EventBacktester(
        strategies=strategies,
        live_root=args.live_root,
        strategies_root=args.strategies_root,
        constitution_yaml=args.constitution_yaml,
        db_path=args.db,
        data_path=args.data_path,
        fee_rate=args.fee_rate,
    )

    result = bt.run(
        symbols=symbols,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        fast_mode=args.fast,
        resume_state=resume_state_obj,
        force_close_end=not bool(args.keep_open_positions),
        no_kill_switch=args.no_kill_switch,
        inject_add_ml_scores_path=args.inject_add_ml_scores,
        equity_anchor_usdt=initial_capital,
        compound_sizing=not bool(args.no_compound_sizing),
    )

    result.print_report()

    _strat_tag = "_".join(strategies)
    export_path = (
        Path(args.trades_csv).expanduser()
        if args.trades_csv
        else _default_trades_csv_path(_strat_tag)
    )
    export_path = export_path.resolve()
    if not args.trades_csv:
        print(f"\n  默认成交 CSV: {export_path}")

    result.export_trades_csv(str(export_path))
    cap_dir = Path(args.capital_report) if args.capital_report else export_path.parent
    cap = write_capital_report_from_trades(
        trades_path=str(export_path),
        out_dir=cap_dir,
        unit="r_multiple",
        title=f"{','.join(strategies)} Capital Report",
        initial_capital=initial_capital,
        risk_per_r=float(args.risk_per_r),
        start_date=args.start_date or "",
        end_date=args.end_date or "",
        total_r=float(sum(t.pnl_r for t in result.trades)),
        compound_sizing=not bool(args.no_compound_sizing),
    )
    print(
        f"  资金报告: {cap_dir / 'capital_report.html'} "
        f"(final=${cap.get('final_capital', 0.0):,.2f}, CAGR={cap.get('cagr', 0.0):.2%})"
    )

    if args.export_add_attempts:
        add_attempt_path = Path(args.export_add_attempts)
        add_attempt_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.add_attempt_rows or []).to_csv(
            add_attempt_path, index=False
        )
        print(
            f"  加仓尝试特征导出: {len(result.add_attempt_rows or [])} rows → {add_attempt_path}"
        )

    if args.output:
        save_json(result, args.output)

    if args.dump_end_state:
        state_obj = {
            "strategy": ",".join(strategies),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "open_positions_count": len(result.open_positions_end),
            "symbols": {},
        }
        by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in result.open_positions_end:
            sym = str(row.get("symbol", ""))
            if not sym:
                continue
            by_symbol[sym].append(
                {"pid": row.get("pid"), "position": row.get("position", {})}
            )
        for sym, rows in by_symbol.items():
            state_obj["symbols"][sym] = {
                "open_positions_count": len(rows),
                "open_positions": rows,
            }
        _dst = Path(args.dump_end_state)
        _dst.parent.mkdir(parents=True, exist_ok=True)
        _dst.write_text(
            json.dumps(state_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n  ♻️ End state saved → {_dst}")

    if args.trading_map:
        # 根据策略 timeframe 选择 K 线频率
        tf_to_freq = {"15T": "15min", "60T": "1h", "120T": "2h", "240T": "4h"}
        _sr_map = args.strategies_root or "config/strategies"
        primary_tf = _get_timeframe(strategies[0], strategies_root=_sr_map)
        map_freq = tf_to_freq.get(primary_tf, "4h")
        generate_trading_map_html(
            result,
            args.trading_map,
            bar_freq=map_freq,
            compare_trades_csv=getattr(args, "compare_trades", None),
            data_path=args.data_path,
            map_extra_months=int(getattr(args, "map_extra_months", 12)),
            map_vwap_window_bars=int(getattr(args, "map_vwap_window_bars", 1200)),
            map_long_ema_span=int(getattr(args, "map_long_ema_span", 1200)),
        )

    if args.db:
        print(f"\n  💾 订单数据已保存 → {args.db}")

    result.print_path_efficiency_footer()
    _anchor = args.output or str(export_path) or args.trading_map
    save_path_efficiency_sidecar(result, _anchor)

    return 0
