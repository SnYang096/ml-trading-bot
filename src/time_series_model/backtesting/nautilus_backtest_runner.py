"""
Nautilus Trader 回测运行器

提供便捷的方式运行基于 Nautilus Trader 的事件驱动回测。
支持从训练好的模型和策略配置直接运行回测。

用法:
    python -m src.time_series_model.backtesting.nautilus_backtest_runner \
        --strategy sr_reversal_rr_reg_long \
        --symbol BTCUSDT \
        --timeframe 240T \
        --start-date 2024-01-01 \
        --end-date 2024-12-31 \
        --model-path models/sr_reversal/model_artifact.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# 检查 Nautilus 可用性
try:
    from nautilus_trader.backtest.node import BacktestNode
    from nautilus_trader.backtest.config import BacktestRunConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.identifiers import InstrumentId, Venue
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.trading.strategy import Strategy

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    print("⚠️ Nautilus Trader 未安装，使用模拟回测模式")


# 项目内导入
from src.time_series_model.strategy_config import StrategyConfigLoader
from src.time_series_model.model_artifact import ModelArtifact


class SimpleBacktestResult:
    """简单回测结果（无需 Nautilus）"""

    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.metrics: Dict[str, float] = {}

    def add_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
    ):
        self.trades.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )

    def compute_metrics(self):
        if not self.trades:
            self.metrics = {
                "total_trades": 0,
                "sharpe_ratio": np.nan,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }
            return

        pnl_list = [t["pnl_pct"] for t in self.trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        total_return = np.sum(pnl_list)
        sharpe = (
            np.mean(pnl_list) / np.std(pnl_list) * np.sqrt(252)
            if np.std(pnl_list) > 0
            else 0.0
        )

        # 计算最大回撤
        cumulative = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        self.metrics = {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0.0,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_win": np.mean(wins) if wins else 0.0,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "profit_factor": (
                abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else np.inf
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        self.compute_metrics()
        return {
            "metrics": self.metrics,
            "trades": self.trades[:10],  # 只返回前 10 笔交易作为示例
            "total_trades": len(self.trades),
        }


class VectorizedBacktest:
    """
    向量化回测（不依赖 Nautilus）

    使用预计算的特征和信号进行快速回测。
    适用于策略开发和参数优化。
    """

    def __init__(
        self,
        strategy_config_path: str,
        model_path: Optional[str] = None,
        initial_capital: float = 100000.0,
    ):
        self.strategy_config_path = Path(strategy_config_path)
        self.model_path = model_path
        self.initial_capital = initial_capital

        # 加载策略配置
        self.config_loader = StrategyConfigLoader(self.strategy_config_path)
        self.strategy_config = self.config_loader.load()

        # 加载模型（如果提供）
        self.model = None
        self.model_artifact = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载模型"""
        path = Path(model_path)
        if path.suffix == ".pkl":
            import pickle

            with open(path, "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, ModelArtifact):
                    self.model_artifact = obj
                    self.model = obj.model
                else:
                    self.model = obj
        print(f"✅ 已加载模型: {model_path}")

    def run(
        self,
        features_df: pd.DataFrame,
        price_col: str = "close",
        signal_col: Optional[str] = None,
    ) -> SimpleBacktestResult:
        """
        运行向量化回测

        Args:
            features_df: 包含特征和价格的 DataFrame
            price_col: 价格列名
            signal_col: 信号列名（如果没有模型）

        Returns:
            SimpleBacktestResult
        """
        result = SimpleBacktestResult()

        # 生成信号
        if signal_col and signal_col in features_df.columns:
            signals = features_df[signal_col].values
        elif self.model is not None:
            # 使用模型预测
            feature_cols = self._get_feature_columns(features_df)
            X = features_df[feature_cols].values
            signals = self.model.predict(X)
        else:
            print("⚠️ 没有信号列或模型，无法运行回测")
            return result

        # 模拟交易
        prices = features_df[price_col].values
        timestamps = features_df.index

        position = 0  # 0: 无仓位, 1: 多头, -1: 空头
        entry_price = 0.0
        entry_time = None

        for i in range(len(signals)):
            signal = signals[i]
            price = prices[i]
            time = timestamps[i]

            # 处理信号
            if position == 0:
                # 开仓
                if signal > 0:
                    position = 1
                    entry_price = price
                    entry_time = time
                elif signal < 0:
                    position = -1
                    entry_price = price
                    entry_time = time
            else:
                # 检查是否平仓
                close_position = False
                if position == 1 and signal <= 0:
                    close_position = True
                elif position == -1 and signal >= 0:
                    close_position = True

                if close_position:
                    # 计算收益
                    if position == 1:
                        pnl_pct = (price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - price) / entry_price

                    pnl = self.initial_capital * pnl_pct

                    result.add_trade(
                        entry_time=entry_time,
                        exit_time=time,
                        direction="long" if position == 1 else "short",
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )

                    position = 0

                    # 如果有反向信号，立即开新仓
                    if signal > 0:
                        position = 1
                        entry_price = price
                        entry_time = time
                    elif signal < 0:
                        position = -1
                        entry_price = price
                        entry_time = time

        result.compute_metrics()
        return result

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取特征列"""
        if self.model_artifact and hasattr(self.model_artifact, "used_features"):
            return self.model_artifact.used_features
        # 排除常见非特征列
        exclude = {"open", "high", "low", "close", "volume", "timestamp", "symbol"}
        return [c for c in df.columns if c.lower() not in exclude]


def run_backtest_cli():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Nautilus/Vectorized 回测运行器")
    parser.add_argument(
        "--strategy",
        "-s",
        required=True,
        help="策略名称（对应 config/strategies/ 下的目录）",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="交易标的")
    parser.add_argument("--timeframe", "-t", default="240T", help="时间框架")
    parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--model-path", help="模型文件路径")
    parser.add_argument(
        "--data-path", help="数据文件路径（Parquet 格式）", default=None
    )
    parser.add_argument("--output-dir", default="results/backtest", help="输出目录")
    parser.add_argument(
        "--mode",
        choices=["vectorized", "event-driven"],
        default="vectorized",
        help="回测模式",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🚀 运行 {args.strategy} 回测")
    print(f"{'='*60}")
    print(f"   标的: {args.symbol}")
    print(f"   时间框架: {args.timeframe}")
    print(f"   时间范围: {args.start_date} ~ {args.end_date}")
    print(f"   模式: {args.mode}")
    print(f"{'='*60}\n")

    # 构建策略配置路径
    strategy_config_path = f"config/strategies/{args.strategy}"

    if args.mode == "vectorized":
        # 向量化回测
        backtest = VectorizedBacktest(
            strategy_config_path=strategy_config_path,
            model_path=args.model_path,
        )

        # 加载数据
        if args.data_path:
            df = pd.read_parquet(args.data_path)
        else:
            # 尝试从默认位置加载
            default_path = Path(
                f"data/features/{args.symbol}/{args.timeframe}/features.parquet"
            )
            if default_path.exists():
                df = pd.read_parquet(default_path)
            else:
                print(f"❌ 数据文件不存在: {default_path}")
                print("   请使用 --data-path 指定数据文件")
                sys.exit(1)

        # 过滤时间范围
        df = df[(df.index >= args.start_date) & (df.index <= args.end_date)]

        if len(df) == 0:
            print("❌ 指定时间范围内没有数据")
            sys.exit(1)

        print(f"📊 数据量: {len(df)} 行")

        # 运行回测
        result = backtest.run(df)

        # 输出结果
        print(f"\n{'='*60}")
        print("📈 回测结果")
        print(f"{'='*60}")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

        # 保存结果
        output_dir = Path(args.output_dir) / args.strategy
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = (
            output_dir / f"backtest_result_{args.start_date}_{args.end_date}.json"
        )
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"\n✅ 结果已保存: {result_path}")

    elif args.mode == "event-driven":
        if not NAUTILUS_AVAILABLE:
            print("❌ Nautilus Trader 未安装，无法使用事件驱动模式")
            print("   安装: pip install nautilus-trader")
            sys.exit(1)

        # 事件驱动回测
        result = run_event_driven_backtest(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
        )

        # 输出结果
        print(f"\n{'='*60}")
        print("📈 事件驱动回测结果")
        print(f"{'='*60}")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")


def run_event_driven_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    model_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: str = "results/backtest",
) -> Dict[str, Any]:
    """
    运行事件驱动回测

    使用增强版 Nautilus 策略进行回测，支持：
    - ModelArtifact 集成
    - RR 止损止盈
    - Trailing stop
    - 完整的执行逻辑
    """
    from src.time_series_model.live.nautilus_strategy_enhanced import (
        NautilusStrategyEnhanced,
        EnhancedFeatureManager,
    )

    print(f"📊 加载数据...")

    # 加载数据
    if data_path:
        df = pd.read_parquet(data_path)
    else:
        # 尝试从默认位置加载
        default_paths = [
            Path(f"data/features/{symbol}/{timeframe}/features.parquet"),
            Path(f"data/parquet_data/{symbol}/"),
        ]
        df = None
        for p in default_paths:
            if p.exists():
                if p.is_dir():
                    # 合并目录下的所有 parquet 文件
                    parquet_files = list(p.glob("*.parquet"))
                    if parquet_files:
                        dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
                        df = pd.concat(dfs, ignore_index=False)
                        break
                else:
                    df = pd.read_parquet(p)
                    break

        if df is None:
            raise FileNotFoundError(f"找不到数据文件，请使用 --data-path 指定")

    # 确保有 datetime 索引
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 过滤时间范围
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    if len(df) == 0:
        raise ValueError("指定时间范围内没有数据")

    print(f"   数据量: {len(df)} 行")
    print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")

    # 初始化特征管理器
    feature_manager = EnhancedFeatureManager(
        strategy_name=strategy_name,
        config_base_path="config/strategies",
        history_window=500,
    )

    # 加载 ModelArtifact
    model_artifact = None
    if model_path:
        artifact_path = Path(model_path)
    else:
        artifact_path = Path("models") / strategy_name

    if artifact_path.exists():
        try:
            from src.time_series_model.strategies.models.model_artifact import (
                ModelArtifact,
            )

            model_artifact = ModelArtifact.load(artifact_path)
            print(f"✅ 已加载 ModelArtifact: {artifact_path}")
        except Exception as e:
            print(f"⚠️ 加载 ModelArtifact 失败: {e}")

    # 加载 backtest 配置
    import yaml

    backtest_config = {}
    config_path = Path(f"config/strategies/{strategy_name}/backtest.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        backtest_config = config.get("backtest", {}).get("params", {})

    # RR 参数
    use_rr_exit = backtest_config.get("use_rr_exit", True)
    rr = backtest_config.get("rr", {})
    stop_loss_r = rr.get("stop_loss_r", 1.0)
    take_profit_r = rr.get("take_profit_r", 2.0)
    max_holding_bars = rr.get("max_holding_bars", 50)
    use_trailing_stop = rr.get("use_trailing_stop", True)
    trailing_atr_mult = rr.get("trailing_atr_mult", 1.0)
    min_confidence = backtest_config.get("min_confidence", 0.3)

    print(f"\n📋 回测配置:")
    print(f"   use_rr_exit: {use_rr_exit}")
    print(f"   stop_loss_r: {stop_loss_r}")
    print(f"   take_profit_r: {take_profit_r}")
    print(f"   max_holding_bars: {max_holding_bars}")
    print(f"   use_trailing_stop: {use_trailing_stop}")

    # 模拟事件驱动回测
    print(f"\n🚀 开始事件驱动回测...")

    trades = []
    current_position = None

    for i, (timestamp, row) in enumerate(df.iterrows()):
        # 构造 bar DataFrame
        bar_df = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "datetime": timestamp,
                    "open": row.get("open", row.get("Open", 0)),
                    "high": row.get("high", row.get("High", 0)),
                    "low": row.get("low", row.get("Low", 0)),
                    "close": row.get("close", row.get("Close", 0)),
                    "volume": row.get("volume", row.get("Volume", 0)),
                    "symbol": symbol,
                }
            ]
        )

        # 更新特征
        feature_manager.update(bar_df)

        close_price = float(bar_df["close"].iloc[0])
        high_price = float(bar_df["high"].iloc[0])
        low_price = float(bar_df["low"].iloc[0])
        atr = feature_manager.get_atr()

        # 管理现有仓位
        if current_position is not None:
            current_position["bars_held"] += 1

            # 更新 trailing stop 参考价格
            if current_position["direction"] == 1:
                current_position["highest_price"] = max(
                    current_position["highest_price"], high_price
                )
            else:
                current_position["lowest_price"] = min(
                    current_position["lowest_price"], low_price
                )

            exit_reason = None
            exit_price = close_price

            if use_rr_exit:
                # 止损检查
                if current_position["direction"] == 1:
                    if low_price <= current_position["stop_loss"]:
                        exit_reason = "stop_loss"
                        exit_price = current_position["stop_loss"]
                else:
                    if high_price >= current_position["stop_loss"]:
                        exit_reason = "stop_loss"
                        exit_price = current_position["stop_loss"]

                # 止盈检查
                if exit_reason is None:
                    if current_position["direction"] == 1:
                        if high_price >= current_position["take_profit"]:
                            exit_reason = "take_profit"
                            exit_price = current_position["take_profit"]
                    else:
                        if low_price <= current_position["take_profit"]:
                            exit_reason = "take_profit"
                            exit_price = current_position["take_profit"]

                # Trailing stop 检查
                if exit_reason is None and use_trailing_stop:
                    trailing_dist = trailing_atr_mult * current_position["atr_at_entry"]
                    if current_position["direction"] == 1:
                        trailing_stop = (
                            current_position["highest_price"] - trailing_dist
                        )
                        if (
                            close_price <= trailing_stop
                            and trailing_stop > current_position["entry_price"]
                        ):
                            exit_reason = "trailing_stop"
                            exit_price = trailing_stop
                    else:
                        trailing_stop = current_position["lowest_price"] + trailing_dist
                        if (
                            close_price >= trailing_stop
                            and trailing_stop < current_position["entry_price"]
                        ):
                            exit_reason = "trailing_stop"
                            exit_price = trailing_stop

            # 时间退出
            if (
                exit_reason is None
                and current_position["bars_held"] >= max_holding_bars
            ):
                exit_reason = "time_exit"
                exit_price = close_price

            # 执行平仓
            if exit_reason is not None:
                direction = current_position["direction"]
                entry_price = current_position["entry_price"]

                if direction == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                trades.append(
                    {
                        "entry_time": current_position["entry_time"],
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "bars_held": current_position["bars_held"],
                    }
                )
                current_position = None

        # 检查入场信号（仅当无仓位时）
        if current_position is None and model_artifact is not None and i >= 50:
            latest_features = feature_manager.get_latest_features()
            if latest_features is not None and len(latest_features) > 0:
                try:
                    # 确保特征列存在
                    available_cols = [
                        c
                        for c in model_artifact.used_features
                        if c in latest_features.columns
                    ]
                    if len(available_cols) >= len(model_artifact.used_features) * 0.5:
                        pred = model_artifact.predict(latest_features)

                        if len(pred) > 0:
                            signal = pred[0]

                            if signal > 0:  # 做多
                                if atr <= 0:
                                    atr = close_price * 0.01
                                current_position = {
                                    "entry_time": timestamp,
                                    "entry_price": close_price,
                                    "direction": 1,
                                    "stop_loss": close_price - stop_loss_r * atr,
                                    "take_profit": close_price + take_profit_r * atr,
                                    "atr_at_entry": atr,
                                    "bars_held": 0,
                                    "highest_price": close_price,
                                    "lowest_price": close_price,
                                }
                            elif signal < 0:  # 做空
                                if atr <= 0:
                                    atr = close_price * 0.01
                                current_position = {
                                    "entry_time": timestamp,
                                    "entry_price": close_price,
                                    "direction": -1,
                                    "stop_loss": close_price + stop_loss_r * atr,
                                    "take_profit": close_price - take_profit_r * atr,
                                    "atr_at_entry": atr,
                                    "bars_held": 0,
                                    "highest_price": close_price,
                                    "lowest_price": close_price,
                                }
                except Exception as e:
                    pass  # 忽略预测错误

        # 进度显示
        if (i + 1) % 1000 == 0:
            print(f"   处理进度: {i+1}/{len(df)} ({(i+1)/len(df)*100:.1f}%)")

    # 计算结果
    if trades:
        pnl_list = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        total_return = sum(pnl_list)
        sharpe = (
            (np.mean(pnl_list) / np.std(pnl_list) * np.sqrt(252))
            if np.std(pnl_list) > 0
            else 0
        )

        # 最大回撤
        cumulative = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        result = {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "exit_reasons": {
                "stop_loss": sum(1 for t in trades if t["exit_reason"] == "stop_loss"),
                "take_profit": sum(
                    1 for t in trades if t["exit_reason"] == "take_profit"
                ),
                "trailing_stop": sum(
                    1 for t in trades if t["exit_reason"] == "trailing_stop"
                ),
                "time_exit": sum(1 for t in trades if t["exit_reason"] == "time_exit"),
            },
        }
    else:
        result = {
            "total_trades": 0,
            "sharpe_ratio": float("nan"),
            "total_return": 0,
            "max_drawdown": 0,
            "win_rate": 0,
        }

    # 保存结果
    output_path = Path(output_dir) / strategy_name
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / f"event_driven_result_{start_date}_{end_date}.json"
    with open(result_file, "w") as f:
        json.dump(
            {
                "metrics": result,
                "trades": trades[:20],  # 只保存前20笔
                "config": {
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_date": start_date,
                    "end_date": end_date,
                    "use_rr_exit": use_rr_exit,
                    "stop_loss_r": stop_loss_r,
                    "take_profit_r": take_profit_r,
                },
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n✅ 结果已保存: {result_file}")

    return result


if __name__ == "__main__":
    run_backtest_cli()
