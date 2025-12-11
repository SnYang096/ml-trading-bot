"""
分析波动率预测准确性和保本止损触发情况
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategy_config import StrategyConfigLoader
from src.time_series_model.strategies.labels.sr_reversal_label import (
    SRSignalConfig,
    _generate_sr_reversal_signals,
    _ensure_atr,
)
from src.time_series_model.pipeline.training.label_utils import (
    future_volatility_label,
    compute_rr_label_with_details,
)
from src.time_series_model.diagnostics.compute_adaptive_rr_with_predicted_vol import (
    compute_adaptive_rr_label_with_predicted_vol_details,
)

try:
    from src.time_series_model.strategies.models.lightgbm_model import LightGBMTrainer

    LIGHTGBM_TRAINER_AVAILABLE = True
except ImportError:
    LIGHTGBM_TRAINER_AVAILABLE = False
    print("⚠️ LightGBMTrainer not available")


def analyze_future_volatility_label(df: pd.DataFrame) -> None:
    """分析未来波动率标签计算是否正确"""
    print("\n" + "=" * 60)
    print("1️⃣ 分析未来波动率标签计算")
    print("=" * 60)

    # 计算未来波动率标签
    future_vol = future_volatility_label(df["close"], horizon=10)

    print(f"   📊 未来波动率标签统计:")
    print(f"      总样本数: {len(future_vol)}")
    print(f"      非NaN数量: {future_vol.notna().sum()}")
    print(f"      NaN数量: {future_vol.isna().sum()}")

    if future_vol.notna().sum() > 0:
        print(f"      均值: {future_vol.mean():.6f}")
        print(f"      中位数: {future_vol.median():.6f}")
        print(f"      标准差: {future_vol.std():.6f}")
        print(f"      最小值: {future_vol.min():.6f}")
        print(f"      最大值: {future_vol.max():.6f}")
        print(f"      25%分位数: {future_vol.quantile(0.25):.6f}")
        print(f"      75%分位数: {future_vol.quantile(0.75):.6f}")

        # 检查是否有异常值
        if future_vol.mean() == 0.0:
            print(f"   ⚠️  警告：未来波动率均值为0，可能存在问题")
            print(f"      检查前10个值: {future_vol.head(10).tolist()}")

            # 检查计算逻辑
            returns = df["close"].pct_change()
            print(f"      收益率统计:")
            print(f"        非NaN数量: {returns.notna().sum()}")
            print(f"        均值: {returns.mean():.6f}")
            print(f"        标准差: {returns.std():.6f}")

            # 手动计算一个窗口的未来波动率
            if len(returns) > 20:
                test_idx = 100
                if test_idx + 10 < len(returns):
                    test_window = returns.iloc[test_idx + 1 : test_idx + 11]
                    manual_vol = np.sqrt(np.mean(np.square(test_window.dropna())))
                    print(f"      手动计算示例 (idx={test_idx}):")
                    print(f"        未来10期收益率: {test_window.tolist()}")
                    print(f"        手动计算的波动率: {manual_vol:.6f}")
                    print(f"        函数计算的波动率: {future_vol.iloc[test_idx]:.6f}")
    else:
        print(f"   ⚠️  所有未来波动率标签都是NaN")


def analyze_breakeven_trigger(
    df: pd.DataFrame,
    atr_series: pd.Series,
    params: dict,
) -> None:
    """分析保本止损触发情况"""
    print("\n" + "=" * 60)
    print("2️⃣ 分析保本止损触发情况")
    print("=" * 60)

    # 生成信号
    sr_cfg = SRSignalConfig(
        min_sr_strength=params.get("sr_strength_min", 0.5),
        min_support_score=params.get("sqs_min", 0.5),
        min_resistance_score=params.get("sqs_min", 0.5),
        tolerance_mult=params.get("touch_distance_atr", 1.0),
    )

    auto_signals = _generate_sr_reversal_signals(
        df,
        price_col="close",
        high_col="high",
        low_col="low",
        atr_series=atr_series,
        cfg=sr_cfg,
    )
    df["signal"] = auto_signals

    # 使用标准R/R计算带详细信息的标签
    details = compute_rr_label_with_details(
        df.copy(),
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=params.get("max_holding_bars", 50),
        stop_loss_r=params.get("stop_loss_r", 1.0),
        take_profit_r=params.get("take_profit_r", 2.0),
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
        use_breakeven_stop=True,
    )

    # 分析保本止损触发情况
    mask_valid = (auto_signals != 0) & details["label"].notna()
    n_trades = int(mask_valid.sum())

    if n_trades == 0:
        print("   ⚠️  没有有效交易")
        return

    print(f"   📊 保本止损分析:")
    print(f"      总交易数: {n_trades}")

    valid_indices = df.index[mask_valid]
    breakeven_activated = details.loc[valid_indices, "breakeven_activated"].fillna(
        False
    )
    n_breakeven_activated = int(breakeven_activated.sum())

    print(
        f"      保本止损激活数: {n_breakeven_activated} ({100*n_breakeven_activated/n_trades:.1f}%)"
    )

    if n_breakeven_activated > 0:
        # 分析激活后的结果
        activated_mask = breakeven_activated
        activated_results = details.loc[valid_indices[activated_mask], "final_result"]

        n_breakeven_win = int((activated_results == "breakeven_win").sum())
        n_breakeven_loss = int((activated_results == "breakeven_loss").sum())

        print(f"      保本后胜利: {n_breakeven_win}")
        print(f"      保本后失败: {n_breakeven_loss}")

        # 分析为什么其他交易没有激活保本
        not_activated_mask = ~breakeven_activated
        not_activated_trades = details.loc[valid_indices[not_activated_mask]]

        print(f"\n   📊 未激活保本止损的交易分析:")
        print(f"      未激活交易数: {int(not_activated_mask.sum())}")

        # 检查这些交易是否在达到触发点前就止盈/止损了
        not_activated_hit_tp = not_activated_trades["hit_tp"].fillna(False)
        not_activated_hit_sl = not_activated_trades["hit_sl"].fillna(False)

        print(f"      未激活交易中:")
        print(f"        先触达止盈: {int(not_activated_hit_tp.sum())}")
        print(f"        先触达止损: {int(not_activated_hit_sl.sum())}")

        # 分析触发条件
        stop_loss_r = params.get("stop_loss_r", 1.0)
        print(f"\n   📊 保本触发条件分析:")
        print(f"      触发条件: 价格达到 entry_price ± {stop_loss_r}×ATR")
        print(f"      保本止损: 移动到 entry_price (保本)")

        # 检查实际价格移动
        if len(not_activated_trades) > 0:
            # 对于long信号，检查是否达到触发点
            long_signals = auto_signals[valid_indices[not_activated_mask]] > 0
            if long_signals.sum() > 0:
                long_indices = valid_indices[not_activated_mask][long_signals]
                entry_prices = df.loc[long_indices, "open"]
                atr_values = atr_series.loc[long_indices]
                max_highs = (
                    df.loc[long_indices, "high"]
                    .rolling(window=params.get("max_holding_bars", 50))
                    .max()
                )

                trigger_levels = entry_prices + stop_loss_r * atr_values
                reached_trigger = max_highs >= trigger_levels

                print(f"      Long信号未激活交易:")
                print(
                    f"        应该达到触发点: {int(reached_trigger.sum())} / {len(long_indices)}"
                )
                if reached_trigger.sum() > 0:
                    print(
                        f"        ⚠️  有 {int(reached_trigger.sum())} 笔交易达到了触发点但未激活保本"
                    )
    else:
        print(f"   ⚠️  没有交易激活保本止损")
        print(f"      可能原因:")
        print(
            f"        1. 触发条件过于严格 (需要达到 {params.get('stop_loss_r', 1.0)}×ATR)"
        )
        print(f"        2. 交易在达到触发点前就止盈/止损了")
        print(f"        3. 最大持仓期太短")


def analyze_volatility_prediction_accuracy(
    df: pd.DataFrame,
    vol_model: any,
    X: pd.DataFrame,
    atr_series: pd.Series,
) -> None:
    """分析波动率预测准确性"""
    print("\n" + "=" * 60)
    print("3️⃣ 分析波动率预测准确性")
    print("=" * 60)

    # 获取预测波动率
    pred_vol_relative = vol_model.predict(X)
    pred_vol_relative = np.maximum(pred_vol_relative, 0.0)

    # 转换为绝对波动率
    prices = df["close"].values
    pred_vol_absolute = pred_vol_relative * prices

    # 计算实际未来波动率
    future_vol = future_volatility_label(df["close"], horizon=10)

    # 对齐数据
    valid_mask = ~(
        np.isnan(pred_vol_absolute) | np.isnan(future_vol) | np.isnan(atr_series.values)
    )

    if valid_mask.sum() == 0:
        print("   ⚠️  没有有效数据用于分析")
        return

    pred_vol_valid = pred_vol_absolute[valid_mask]
    future_vol_valid = future_vol.values[valid_mask]
    atr_valid = atr_series.values[valid_mask]

    # 转换为相对ATR的比率
    pred_vol_relative_to_atr = pred_vol_valid / (atr_valid + 1e-8)
    future_vol_relative_to_atr = future_vol_valid / (atr_valid + 1e-8)

    # 计算误差指标
    error = pred_vol_relative_to_atr - future_vol_relative_to_atr
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    mape = np.mean(np.abs(error) / (future_vol_relative_to_atr + 1e-8)) * 100

    # 计算相关性
    correlation = np.corrcoef(pred_vol_relative_to_atr, future_vol_relative_to_atr)[
        0, 1
    ]

    print(f"   📊 预测准确性统计 (相对于ATR):")
    print(f"      样本数: {len(pred_vol_relative_to_atr)}")
    print(f"      MAE: {mae:.4f}")
    print(f"      RMSE: {rmse:.4f}")
    print(f"      MAPE: {mape:.2f}%")
    print(f"      相关系数: {correlation:.4f}")
    print(f"      预测均值: {np.mean(pred_vol_relative_to_atr):.3f}")
    print(f"      实际均值: {np.mean(future_vol_relative_to_atr):.3f}")
    print(f"      预测中位数: {np.median(pred_vol_relative_to_atr):.3f}")
    print(f"      实际中位数: {np.median(future_vol_relative_to_atr):.3f}")

    # 分析预测偏差
    bias = np.mean(error)
    print(f"\n   📊 预测偏差分析:")
    print(
        f"      平均偏差: {bias:.4f} ({'高估' if bias > 0 else '低估' if bias < 0 else '无偏差'})"
    )

    # 分位数分析
    print(f"\n   📊 分位数对比:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        pred_q = np.quantile(pred_vol_relative_to_atr, q)
        actual_q = np.quantile(future_vol_relative_to_atr, q)
        print(
            f"      {q*100:.0f}%分位数: 预测={pred_q:.3f}, 实际={actual_q:.3f}, 误差={pred_q-actual_q:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze volatility prediction and breakeven stop-loss"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Symbol to analyze"
    )
    parser.add_argument("--timeframe", type=str, default="240T", help="Timeframe")
    parser.add_argument(
        "--data-path", type=str, default="data/parquet_data", help="Data path"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config/strategies/sr_reversal",
        help="Strategy config directory",
    )

    args = parser.parse_args()

    print("📊 Loading data...")
    df = load_raw_data(
        data_path=Path(args.data_path),
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    print(f"   Loaded {len(df)} bars")

    print("\n🔧 Loading features...")
    feature_loader = StrategyFeatureLoader()
    config_loader = StrategyConfigLoader(Path(args.config_dir))
    strategy_config = config_loader.load_strategy_config()

    df_features = feature_loader.load_features(
        df,
        strategy_config.feature_pipeline,
    )
    print(f"   Loaded {len(df_features.columns)} features")

    # 确保ATR存在
    atr_series = _ensure_atr(df_features, "atr", "close", "high", "low", 14)

    # 分析未来波动率标签
    analyze_future_volatility_label(df_features)

    # 分析保本止损触发情况
    params = {
        "sr_strength_min": 0.5,
        "sqs_min": 0.5,
        "touch_distance_atr": 1.0,
        "max_holding_bars": 50,
        "stop_loss_r": 1.25,
        "take_profit_r": 3.0,
    }
    analyze_breakeven_trigger(df_features, atr_series, params)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
