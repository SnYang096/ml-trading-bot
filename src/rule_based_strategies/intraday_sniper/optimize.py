"""
Optuna optimization script for IntradaySniperStrategy
"""
import os
import argparse
import subprocess
import time
from datetime import datetime
from decimal import Decimal
import pandas as pd
import glob

import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.enums import OmsType, AccountType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.backtest.models import LeveragedMarginModel
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler

# Import the intraday sniper strategy and config
from yin_bot.intraday_sniper.strategy import IntradaySniperStrategy, IntradaySniperConfig

# Import data loading functions from common
from yin_bot.common.data_loader import load_trade_data

# Define instrument
BTCUSDT_PERP_BINANCE = TestInstrumentProvider.btcusdt_perp_binance()


def convert_to_tradeticks(df, instrument):
    df = df.copy()
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


def get_ticks(data_dir, file_patterns, sample_size=None):
    """Load ticks from multiple files"""
    all_dfs = []
    
    # 如果传入的是字符串，转换为列表
    if isinstance(file_patterns, str):
        file_patterns = [file_patterns]
    
    # 加载所有指定的文件
    for file_pattern in file_patterns:
        # Handle glob patterns
        if '*' in file_pattern or '?' in file_pattern:
            # Use glob to find matching files
            matching_files = glob.glob(os.path.join(data_dir, file_pattern))
            for file_path in matching_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # 如果指定了采样大小，则只取前sample_size行
                    if sample_size:
                        df = df.head(sample_size)
                    all_dfs.append(df)
                    print(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
        else:
            file_path = os.path.join(data_dir, file_pattern)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # 如果指定了采样大小，则只取前sample_size行
                if sample_size:
                    df = df.head(sample_size)
                all_dfs.append(df)
                print(f"Loaded {len(df)} rows from {file_pattern}")
    
    # 合并所有数据
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined {len(all_dfs)} files with total {len(combined_df)} rows")
        
        # 处理数据列名，确保有正确的列
        if 'transact_time' in combined_df.columns:
            combined_df.rename(columns={'transact_time': 'ts'}, inplace=True)
        elif 'timestamp' in combined_df.columns:
            combined_df.rename(columns={'timestamp': 'ts'}, inplace=True)
        
        # 确保必要的列存在
        required_columns = ['ts', 'price', 'quantity', 'is_buyer_maker']
        missing_cols = [col for col in required_columns if col not in combined_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 使用 common.data_loader 中的函数处理数据
        combined_df['ts'] = pd.to_datetime(combined_df['ts'], unit='ms')
        combined_df.set_index('ts', inplace=True)
        combined_df.sort_index(inplace=True)
        
        instrument = BTCUSDT_PERP_BINANCE
        ticks = convert_to_tradeticks(combined_df, instrument)
        print(f"✅ Loaded {len(ticks)} trade ticks from {len(all_dfs)} files")
        return ticks, combined_df
    else:
        raise FileNotFoundError("No data files found")


# ======================
# 🔧 全局设置
# ======================
RESULTS_DIR = "optuna_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global tick data cache
ticks = None


def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


# ======================
# 🎯 目标函数
# ======================
def objective(trial: optuna.Trial):
    global ticks
    
    # 如果tick数据尚未加载，则加载它
    if ticks is None:
        try:
            # 使用项目中的实际数据目录
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'agg_data')
            if not os.path.exists(data_dir):
                # 备用路径
                data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'agg_data')
            
            # 为了测试目的，加载一周的数据
            trade_files = ["BTCUSDT-aggTrades-2025-05-0*.csv"]  # 使用通配符加载一周数据
            log(f"Loading ticks from files matching pattern: {trade_files[0]}")
            
            # 加载tick数据，不采样以使用完整数据集
            ticks, _ = get_ticks(data_dir, trade_files)  # 不使用采样以获得完整数据集
            log(f"✅ Loaded {len(ticks)} trade ticks from all matching files")
        except Exception as e:
            log(f"❌ Failed to load tick data: {e}")
            raise optuna.TrialPruned()

    # 为IntradaySniper策略定义参数搜索空间（使用更合理的范围）
    # Focus on key parameters that are more likely to generate trades
    params = {
        # 风险管理参数
        "risk_per_trade": trial.suggest_float("risk_per_trade", 0.01, 0.1),  # Increased range for better risk management
        "target_r_ratio": trial.suggest_float("target_r_ratio", 1.0, 5.0),
        "min_breakout_score": trial.suggest_float("min_breakout_score", 0.01, 1.0),  # Much lower range for better sensitivity
        
        # 止损参数
        "atr_period": trial.suggest_int("atr_period", 10, 20),
        "trailing_stop_atr_mult": trial.suggest_float("trailing_stop_atr_mult", 1.0, 4.0),
        
        # MDC指标参数 - Focus on the most critical ones for trade generation
        "compression_window": trial.suggest_int("compression_window", 15, 50),
        "cvd_window_size": trial.suggest_int("cvd_window_size", 5, 30),
        "compression_confidence_threshold": trial.suggest_float("compression_confidence_threshold", 0.001, 0.3),
        "expansion_confidence_threshold": trial.suggest_float("expansion_confidence_threshold", 0.001, 0.3),
        "compression_atr_threshold_quantile": trial.suggest_float("compression_atr_threshold_quantile", 0.1, 0.7),
        "compression_volume_threshold_quantile": trial.suggest_float("compression_volume_threshold_quantile", 0.1, 0.7),
        
        # 订单流参数
        "delta_ma_window": trial.suggest_int("delta_ma_window", 3, 10),
        
        # Hierarchical decision parameters
        "structural_compression_body_ratio": trial.suggest_float("structural_compression_body_ratio", 0.3, 0.9),
        "momentum_convergence_price_threshold": trial.suggest_float("momentum_convergence_price_threshold", 1e-5, 1e-2, log=True),
        "direction_ordered_entropy_threshold": trial.suggest_float("direction_ordered_entropy_threshold", 0.3, 0.9),
        "volatility_dense_threshold": trial.suggest_float("volatility_dense_threshold", 0.1, 0.7),
        "volatility_buffer_base_threshold": trial.suggest_float("volatility_buffer_base_threshold", 0.001, 0.1),
        "pre_breakout_silence_atr_threshold_quantile": trial.suggest_float("pre_breakout_silence_atr_threshold_quantile", 0.1, 0.6),
    }

    log(f"Trial {trial.number} start: {params}")

    # === 引擎配置 ===
    config = BacktestEngineConfig(
        trader_id=TraderId(f"INTRADAY_SNIPE-{trial.number}"),
        logging=LoggingConfig(log_level="ERROR"),
    )

    engine = BacktestEngine(config=config)

    try:
        # 添加账户与交易所
        engine.add_venue(
            venue=BINANCE_VENUE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=None,
            starting_balances=[
                Money(10_000, BTCUSDT_PERP_BINANCE.quote_currency)
            ],
            default_leverage=Decimal("100.0"),  # 100x leverage
            margin_model=LeveragedMarginModel(),  # Leveraged margin model
        )

        # 数据加载
        if ticks is None or len(ticks) == 0:
            log(f"Trial {trial.number} ❌ No tick data loaded.")
            raise optuna.TrialPruned()

        engine.add_instrument(BTCUSDT_PERP_BINANCE)
        engine.add_data(ticks)

        # 策略初始化 - 使用优化的参数
        strategy_config = IntradaySniperConfig(
            instrument_id=BTCUSDT_PERP_BINANCE.id,
            order_id_tag=f"OPT{trial.number}",
            bar_type="BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-INTERNAL",
            indicators={
                "bollinger_bands": {
                    "period": 20,
                    "stddev": 2.0
                },
                "adaptive_multi_dim_compression": {
                    "indicator_config": {
                        "compression_window": params["compression_window"],
                        "volatility_window": 10,
                        "volume_window": 10,
                        "entropy_window": 10,
                        "compression_tdigest": 100.0,
                        "minimum_tdigest_warmup": 10,  # Lower warmup for faster initialization
                        "duration_bonus_window": 50,   # Lower window for faster response
                        "pre_breakout_silence_window": 3,  # Lower window for faster response
                        "internal_price_density_window": 5  # Lower window for faster response
                    },
                    "weight_config": {
                        "compression_atr": 0.15,
                        "compression_volume": 0.15,
                        "structural_compression": 0.15,
                        "momentum_convergence": 0.10,
                        "direction_ordered": 0.08,
                        "volatility_dense": 0.08,
                        "duration_bonus": 0.10,
                        "pre_breakout_silence": 0.10,
                        "internal_price_density": 0.09
                    },
                    "threshold_config": {
                        "compression_atr_threshold_quantile": params["compression_atr_threshold_quantile"],  # Optimized parameter
                        "compression_volume_threshold_quantile": params["compression_volume_threshold_quantile"],  # Optimized parameter
                        "structural_compression_body_ratio": params["structural_compression_body_ratio"],  # Optimized parameter
                        "structural_compression_atr_ratio": 1.0,  # Lower threshold for more sensitivity
                        "momentum_convergence_price_threshold": params["momentum_convergence_price_threshold"],  # Optimized parameter
                        "direction_ordered_entropy_threshold": params["direction_ordered_entropy_threshold"],  # Optimized parameter
                        "volatility_dense_threshold": params["volatility_dense_threshold"],  # Optimized parameter
                        "compression_confidence_threshold": params["compression_confidence_threshold"],  # Optimized parameter
                        "expansion_confidence_threshold": params["expansion_confidence_threshold"],  # Optimized parameter
                        "volatility_buffer_base_threshold": params["volatility_buffer_base_threshold"],  # Optimized parameter
                        "pre_breakout_silence_atr_threshold_quantile": params["pre_breakout_silence_atr_threshold_quantile"],  # Optimized parameter
                        "internal_price_density_threshold": 0.5  # Lower threshold for more sensitivity
                    }
                },
                "cvd": {
                    "window_size": params["cvd_window_size"]
                },
                "order_flow": {
                    "delta_ma_window": params["delta_ma_window"]
                }
            },
            breakout_quality_scorer={
                "weights": {
                    "price_breakout": 0.111,
                    "volume_spike": 0.111,
                    "cvd_momentum_strong": 0.081,
                    "cvd_momentum_weak": 0.031,
                    "volume_and_cvd_aligned": 0.051,
                    "order_absorption_bar": 0.008,
                    "order_absorption_tick": 0.008,
                    "high_liquidity_time": 0.051,
                    "compression_bonus": 0.051,
                    "poc_direction_up": 0.081,
                    "poc_direction_down": 0.081,
                    "pre_break_proximity": 0.051,
                    "first_breakout_bar": 0.111,
                    "volume_dominant": 0.051,
                    "pre_breakout_silence": 0.122
                }
            },
            risk_management={
                "risk_per_trade": params["risk_per_trade"],
                "target_r_ratio": params["target_r_ratio"],
                "initial_capital": 10000.0,
                "min_breakout_score": params["min_breakout_score"]
            },
            stop_loss={
                "atr_period": params["atr_period"],
                "trailing_stop_atr_mult": params["trailing_stop_atr_mult"]
            },
            session={
                "start": "00:00",  # 全天交易时间
                "end": "23:59"
            },
            event_filter={
                "cooloff_after_event": 0  # Disabled cooloff for testing
            },
            logging={
                "level": "INFO"  # Increased logging level
            }
        )

        strategy = IntradaySniperStrategy(config=strategy_config)
        engine.add_strategy(strategy)
        
        log(f"Trial {trial.number} - Strategy config: min_breakout_score={params['min_breakout_score']}")

        # 运行回测
        engine.run()

        # 绩效获取（新版API）
        account_report = engine.trader.generate_account_report(BINANCE_VENUE)
        order_fills_report = engine.trader.generate_order_fills_report()
        
        # 计算总交易数
        total_trades = len(order_fills_report) if not order_fills_report.empty else 0

        # 从账户报告中提取绩效指标
        if not account_report.empty and len(account_report) > 1:
            net_profit = float(account_report["total"].iloc[-1] - account_report["total"].iloc[0])
            # 计算收益率
            initial_capital = float(account_report["total"].iloc[0])
            if initial_capital > 0:
                profit_pct = (net_profit / initial_capital) * 100
            else:
                profit_pct = 0.0
        else:
            net_profit = 0.0
            profit_pct = 0.0

        # 如果交易太少，不考虑这个试验
        if total_trades < 1:
            log(f"Trial {trial.number} ⚠️ No trades executed")
            # Instead of pruning, let's return a small negative value to encourage parameter exploration
            return -10.0  # Small penalty for no trades

        # 计算更复杂的评分函数
        # 考虑净收益、交易数量和风险调整收益
        # 奖励更多的交易，但也要考虑收益
        if total_trades == 0:
            # 严厉惩罚没有交易的情况
            score = -100.0
        else:
            # 基于收益和交易数量的综合评分
            # 奖励正收益和更多交易，但惩罚过度交易导致的小额收益
            profit_score = net_profit * 0.1  # 收益权重
            trade_score = min(total_trades, 20) * 5  # 交易数量奖励，但有上限
            efficiency_score = (net_profit / max(total_trades, 1)) * 2 if net_profit > 0 else 0  # 效率奖励
            
            score = profit_score + trade_score + efficiency_score
            
            # 如果有显著的正收益，给予额外奖励
            if net_profit > 50:
                score += 50
            elif net_profit > 20:
                score += 20

        log(f"Trial {trial.number} ✅ Done | Profit={net_profit:.2f} ({profit_pct:.2f}%), Score={score:.2f}, Trades={total_trades}")

        return score

    except Exception as e:
        log(f"Trial {trial.number} ❌ Exception: {e}")
        raise optuna.TrialPruned()
    finally:
        engine.dispose()

# ======================
# ⚙️ 主优化流程
# ======================
def optimize_with_optuna(args):
    global ticks
    log("🚀 Starting Optuna optimization...")
    
    # 尝试加载tick数据
    try:
        # 使用项目中的实际数据目录
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'agg_data')
        if not os.path.exists(data_dir):
            # 备用路径
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'agg_data')
        
        # 检查目录是否存在
        if not os.path.exists(data_dir):
            # 直接使用绝对路径
            data_dir = "/home/yin/trading/rlbot/data/agg_data"
        
        # 加载一周的数据
        trade_files = ["BTCUSDT-aggTrades-2025-05-0*.csv"]  # 使用通配符加载一周数据
        log(f"Loading ticks from files matching pattern: {trade_files[0]} in directory: {data_dir}")
        
        # 检查目录中是否有匹配的文件
        import glob
        matching_files = glob.glob(os.path.join(data_dir, trade_files[0]))
        log(f"Found {len(matching_files)} matching files: {matching_files[:3]}...")  # 只显示前3个文件
        
        # 加载tick数据，不采样以使用完整数据集
        ticks, _ = get_ticks(data_dir, trade_files)  # 不使用采样以获得完整数据集
        log(f"✅ Loaded {len(ticks)} trade ticks from all matching files")
    except Exception as e:
        log(f"⚠️ Failed to preload tick data: {e}")
        import traceback
        traceback.print_exc()

    study_name = f"intraday_sniper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_path = os.path.join(RESULTS_DIR, f"{study_name}.db")
    storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_url,
        load_if_exists=False,
    )

    # 启动 dashboard
    if args.dashboard:
        log("💻 Launching optuna-dashboard...")
        port = args.port or 8080
        try:
            subprocess.Popen(
                ["optuna-dashboard", f"--port={port}", storage_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)
            log(f"✅ Dashboard running at http://127.0.0.1:{port}")
        except Exception as e:
            log(f"⚠️ Failed to start dashboard: {e}")

    # 优化开始
    study.optimize(objective,
                   n_trials=args.trials,
                   n_jobs=1,
                   show_progress_bar=True)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    log(f"Trials total={len(study.trials)}, completed={len(completed)}")

    if not completed:
        log("❌ No successful trials. Please check data or objective logic.")
        return

    best = study.best_trial
    log(f"🏆 Best Trial #{best.number}: value={best.value:.4f}")
    log(f"📊 Params: {best.params}")

    # 保存图表
    try:
        plot_optimization_history(study).write_html(
            os.path.join(RESULTS_DIR, f"{study_name}_history.html"))
        plot_param_importances(study).write_html(
            os.path.join(RESULTS_DIR, f"{study_name}_importance.html"))
        log("📈 Saved HTML charts.")
    except Exception as e:
        log(f"⚠️ Visualization failed: {e}")

    log(f"💾 DB saved: {db_path}")
    log(f"💻 Dashboard cmd: optuna-dashboard {storage_url}")


# ======================
# 🏁 入口
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna optimizer for IntradaySniperStrategy (nautilus 0.67+)")
    parser.add_argument("--dashboard",
                        action="store_true",
                        help="Enable optuna-dashboard for live monitoring")
    parser.add_argument("--port",
                        type=int,
                        default=8080,
                        help="Dashboard port (default 8080)")
    parser.add_argument("--trials",
                        type=int,
                        default=10,
                        help="Number of trials")
    parser.add_argument("--data-dir",
                        type=str,
                        help="Directory containing trade data files")
    args = parser.parse_args()

    optimize_with_optuna(args)