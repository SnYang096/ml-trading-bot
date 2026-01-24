from decimal import Decimal
from pathlib import Path
from nautilus_trader.backtest.config import MarginModelConfig
from nautilus_trader.backtest.models import LeveragedMarginModel
import pandas as pd
import os
import numpy as np
from datetime import datetime, timezone
import yaml

# ============ Bokeh Imports ============
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
from bokeh.layouts import column, row
from bokeh.palettes import RdYlGn
from bokeh.io import output_file, save

# ============ NautilusTrader Imports ============
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.enums import AccountType, AggressorSide, OmsType
from nautilus_trader.config import BacktestEngineConfig, StrategyConfig
from nautilus_trader.model.identifiers import Venue, Symbol, InstrumentId, TraderId
from nautilus_trader.model.instruments import CurrencyPair  # 更新为 CurrencyPair
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.data import TradeTick, Bar
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.enums import AggressorSide
# ============ 导入策略 ============
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.objects import Money
from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.adapters.binance import get_cached_binance_http_client
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.futures.providers import BinanceFuturesInstrumentProvider
from nautilus_trader.test_kit.stubs.execution import TestExecStubs
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from nautilus_trader.persistence.wranglers import TradeTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider

# ============ 导入自定义模块 ============
from yin_bot.common.helper import create_advanced_bokeh_charts
from yin_bot.common.data_loader import load_trade_data
from yin_bot.intraday_sniper.strategy import IntradaySniperStrategy, IntradaySniperConfig


def load_config(config_path: str = "config.yaml"):
    # 如果是相对路径，基于当前工作目录解析
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        # 如果仍然找不到，尝试在脚本目录中查找
        script_dir = Path(__file__).parent
        config_path_obj = script_dir / config_path
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path_obj, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_instrument_from_config(config):
    """Get instrument based on config"""
    instrument_id_str = config.get('instrument_id', 'BTCUSDT-PERP.BINANCE')

    # For Binance futures perpetual contracts
    if 'BTCUSDT-PERP' in instrument_id_str:
        from nautilus_trader.test_kit.providers import TestInstrumentProvider
        return TestInstrumentProvider.btcusdt_perp_binance()
    else:
        # Add other instrument types as needed
        raise ValueError(f"Unsupported instrument: {instrument_id_str}")


def convert_to_tradeticks(df, instrument):
    df = df.copy()
    # ✅ 正确的列映射：注意 quantity -> size
    # df = df.rename(
    #     columns={
    #         "ts": "ts_event",  # 时间戳
    #         "price": "price",  # 价格
    #         "qty": "quantity",  # ✅ 关键修复：quantity → size
    #         "is_buyer_maker": "is_buyer_maker"
    #     })

    # 确保时间转换为 UTC datetime
    df["ts_event"] = pd.to_datetime(df.index, unit='ms', utc=True)

    # ts_init 可与 ts_event 相同
    df["ts_init"] = df["ts_event"]

    # 生成 trade_id（可用 agg_trade_id 更好）
    df["trade_id"] = df["agg_trade_id"].astype(str)  # 推荐使用实际 ID，而非索引

    # ✅ 设置时间索引（必须！）
    df = df.set_index("ts_event")

    # 实例化 wrangler 并处理
    wrangler = TradeTickDataWrangler(instrument)
    ticks = wrangler.process(df)
    return ticks


def get_ticks(data_dir, file_pattern, instrument):
    df = load_trade_data(data_dir, file_pattern)
    ticks = convert_to_tradeticks(df, instrument)
    print(f"✅ Loaded {len(ticks)} trade ticks")
    return ticks, df


# ============ 4. 回测主函数 ============
def run_backtest_with_config(config_path: str = "config.yaml",
                             data_dir: str = ".",
                             file_pattern: str = "trades.csv",
                             timeframe: str = "15min"):
    # Load configuration
    config = load_config(config_path)
    print(f"✅ Loaded configuration: {config.get('strategy_name', 'IntradaySniper')}")

    # Get instrument based on config
    instrument = get_instrument_from_config(config)

    ticks, df = get_ticks(data_dir, file_pattern, instrument)
    print(f"✅ Loaded {len(ticks)} trade ticks")

    # === 配置引擎 ===
    # Configure backtest engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level=config.get("logging_level", "INFO")),
    )

    engine = BacktestEngine(config=engine_config)

    engine.add_venue(
        venue=BINANCE_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[
            Money(config.get("initial_capital", 100000),
                  instrument.quote_currency)
        ],
        default_leverage=Decimal("100.0"),  # 100x leverage
        margin_model=LeveragedMarginModel(),  # Leveraged margin model
    )

    engine.add_instrument(instrument)

    # === 添加策略 ===
    # Create strategy configuration from loaded config
    strategy_config = IntradaySniperConfig(
        order_id_tag=config.get("strategy_name", "INTRADAY_SNIPE"),
        instrument_id=instrument.id,
        bar_type=config.get("bar_type",
                            "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"),
        # All other parameters are passed through the config structure
        indicators=config.get("indicators", {}),
        breakout_quality_scorer=config.get("breakout_quality_scorer", {
            "weights": {
                "price_breakout": 0.2,
                "volume_spike": 0.2,
                "delta_strength": 0.2,
                "cvd_momentum": 0.15,
                "order_absorption_bar": 0.15,
                "high_liquidity_time": 0.1
            }
        }),
        risk_management=config.get("risk_management", {
            "risk_per_trade": 0.01,
            "target_r_ratio": 3.0,
            "initial_capital": 100000.0
        }),
        stop_loss=config.get("stop_loss", {
            "atr_period": 14,
            "trailing_stop_atr_mult": 1.5
        }),
        session=config.get("session", {
            "start": "09:30",
            "end": "15:45"
        }),
        event_filter=config.get("event_filter", {
            "cooloff_after_event": 300
        }),
        logging={
            "level": config.get("logging", {}).get("level", "INFO")
        }
    )

    strategy = IntradaySniperStrategy(config=strategy_config)
    engine.add_strategy(strategy)

    # === 加载数据 ===
    engine.add_data(ticks)

    # === 运行回测 ===
    print("🚦 Starting backtest ...")
    engine.run()
    print("✅ Backtest complete.")

    # 创建输出目录（如果不存在）
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    # 生成并保存账户报告
    account_report = engine.trader.generate_account_report(BINANCE_VENUE)
    account_report.to_csv(os.path.join(output_dir, "account_report.csv"),
                          index=False)
    print("===== Account Report saved to account_report.csv =====")

    # 生成并保存成交报告
    order_fills_report = engine.trader.generate_order_fills_report()
    order_fills_report.to_csv(
        os.path.join(output_dir, "order_fills_report.csv"))
    print("===== Order Fills Report saved to order_fills_report.csv =====")

    # 生成并保存持仓报告
    positions_report = engine.trader.generate_positions_report()
    positions_report.to_csv(os.path.join(output_dir, "positions_report.csv"))
    print("===== Positions Report saved to positions_report.csv =====")

    # ============ 收集压缩区域和突破信号数据 ============
    # 生成K线数据用于分析
    # Convert timeframe to pandas compatible format to avoid FutureWarning
    pandas_timeframe = timeframe
    if timeframe.endswith('m') and not timeframe.endswith('min'):
        pandas_timeframe = timeframe.replace('m', 'min')
    elif timeframe.endswith('h') and not timeframe.endswith('H'):
        pandas_timeframe = timeframe.replace('h', 'H')
    elif timeframe.endswith('d') and not timeframe.endswith('D'):
        pandas_timeframe = timeframe.replace('d', 'D')
    
    price_df = df.resample(pandas_timeframe).agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        quantity=('quantity', 'sum'),
    )
    
    # 创建K线对象用于策略分析
    bars = []
    for idx in price_df.index:
        row = price_df.loc[idx]
        # 确保时间戳是正确的类型
        if isinstance(idx, pd.Timestamp):
            ts_value = int(idx.value)  # pd.Timestamp 的 value 属性已经是纳秒
        else:
            # 如果不是 pd.Timestamp，转换为 pd.Timestamp
            timestamp = pd.Timestamp(idx)
            ts_value = int(timestamp.value)
        
        bar = Bar(
            bar_type=strategy.bar_type,
            open=Price.from_str(str(row['open'])),
            high=Price.from_str(str(row['high'])),
            low=Price.from_str(str(row['low'])),
            close=Price.from_str(str(row['close'])),
            volume=Quantity.from_str(str(row['quantity'])),
            ts_event=ts_value,
            ts_init=ts_value
        )
        bars.append(bar)
    
    # 收集压缩区域和突破信号数据
    compression_data = strategy.get_compression_and_breakout_data(bars)

    # 正确提取权益曲线 (使用索引而不是ts_event列)
    equity_curve = pd.Series(dtype=float)  # 创建空的Series而不是None
    if 'total' in account_report.columns and not account_report.empty:
        equity_curve = account_report['total']
    
    # 2️⃣ 生成 Bokeh 布局
    layout = create_advanced_bokeh_charts(
        price_df=price_df,
        positions_report=positions_report,
        order_fills_report=order_fills_report,
        equity_curve=equity_curve,
        compression_data=compression_data)  # 传递压缩数据

    # 3️⃣ 保存为 HTML (只有当layout不是None时才保存)
    if layout is not None:
        output_file("advanced_report.html")
        save(layout)
        print("✅ 高级 Bokeh 报告已生成: advanced_report.html")
        
        # 自动在浏览器中打开报告
        try:
            import webbrowser
            # 获取当前工作目录下的完整路径
            report_path = os.path.abspath("advanced_report.html")
            # 在浏览器中打开
            webbrowser.open(f"file://{report_path}")
            print("🌐 报告已在浏览器中打开")
        except Exception as e:
            print(f"⚠️ 无法自动打开浏览器: {e}")
    else:
        print("⚠️ 无法生成 Bokeh 报告")

    # For repeated backtest runs make sure to reset the engine
    engine.reset()

    # Good practice to dispose of the object
    engine.dispose()


def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


# ============ 主入口 ============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument(
        '--filename',
        type=str,
        required=True,
        help=
        'Path to trades CSV (timestamp,close,quantity,side/is_buyer_maker optional)'
    )
    parser.add_argument('--timeframe', type=str, default='15min')
    args = parser.parse_args()

    data_dir = os.getenv('DATA_DIR', '.')
    trades_full_path = os.path.join(data_dir, args.filename)
    print(f"Using trades file: {trades_full_path}")

    run_backtest_with_config(config_path=args.config,
                             data_dir=data_dir,
                             file_pattern=args.filename,
                             timeframe=args.timeframe)