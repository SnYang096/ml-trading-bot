"""
SR Reversal 模型对比：规则类 vs ML模型 vs ML+波动率模型

功能：
1. 训练ML模型（分类模型）
2. 训练波动率模型
3. 在backtest中使用波动率模型动态调整R/R
4. 对比三种方法的性能
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.strategy_config import StrategyConfigLoader  # noqa: E402
from src.time_series_model.strategies.labels.sr_reversal_label import (  # noqa: E402
    SRSignalConfig,
    _generate_sr_reversal_signals,
    _ensure_atr,
)
from src.time_series_model.pipeline.training.label_utils import (  # noqa: E402
    compute_rr_label,
    future_volatility_label,
    compute_rr_label_with_details,
)

try:
    from src.time_series_model.strategies.models.lightgbm_model import (
        LightGBMTrainer,
    )  # noqa: E402

    LIGHTGBM_TRAINER_AVAILABLE = True
except ImportError:
    LIGHTGBM_TRAINER_AVAILABLE = False
    print("⚠️ LightGBMTrainer not available, will use simple LightGBM")
from scripts import train_strategy_pipeline as strategy_runner  # noqa: E402

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SR Reversal Model Comparison: Rule-based vs ML vs ML+Volatility"
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        required=True,
        help="Path to strategy config directory",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to OHLCV data file",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="4H",
        help="Timeframe (e.g., '4H', '1D')",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test set size (0.0-1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--rule-params",
        type=str,
        default=None,
        help="Path to optimized rule parameters JSON (optional)",
    )
    return parser.parse_args()


def train_ml_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Any, Dict[str, float]]:
    """训练ML分类模型"""
    print("   🤖 Training ML classification model...")

    try:
        model = LightGBMTrainer(model_type="classification", use_gpu=True)
        metrics, _ = model.train(
            X_train,
            y_train,
            n_splits=5,
            use_time_series_cv=True,
            groups=None,
            auto_tune_params=False,
        )
        return model, metrics
    except Exception as e:
        print(f"   ⚠️ LightGBMTrainer failed: {e}")
        print("   Using simple LightGBM instead...")
        import lightgbm as lgb

        # Simple LightGBM training
        train_data = lgb.Dataset(X_train.values, label=y_train.values)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        model = lgb.train(params, train_data, num_boost_round=100)

        # Create a simple wrapper
        class SimpleModel:
            def __init__(self, lgb_model):
                self.lgb_model = lgb_model
                self.is_trained = True

            def predict_proba(self, X):
                preds = self.lgb_model.predict(
                    X.values if isinstance(X, pd.DataFrame) else X
                )
                return np.column_stack([1 - preds, preds])

            def predict(self, X):
                preds = self.lgb_model.predict(
                    X.values if isinstance(X, pd.DataFrame) else X
                )
                return (preds >= 0.5).astype(int)

        wrapped_model = SimpleModel(model)
        metrics = {"train_accuracy": 0.0}  # Placeholder
        return wrapped_model, metrics


def train_volatility_model(
    X_train: pd.DataFrame,
    y_vol_train: pd.Series,
    X_test: pd.DataFrame,
    y_vol_test: pd.Series,
) -> Tuple[Any, Dict[str, float]]:
    """训练波动率模型"""
    print("   📊 Training volatility model...")

    try:
        model = LightGBMTrainer(model_type="regression", use_gpu=True)
        metrics, _ = model.train(
            X_train,
            y_vol_train,
            n_splits=5,
            use_time_series_cv=True,
            groups=None,
            auto_tune_params=False,
        )
        return model, metrics
    except Exception as e:
        print(f"   ⚠️ LightGBMTrainer failed: {e}")
        print("   Using simple LightGBM instead...")
        import lightgbm as lgb

        # Simple LightGBM training
        train_data = lgb.Dataset(X_train.values, label=y_vol_train.values)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        model = lgb.train(params, train_data, num_boost_round=100)

        # Create a simple wrapper
        class SimpleVolModel:
            def __init__(self, lgb_model):
                self.lgb_model = lgb_model
                self.is_trained = True

            def predict(self, X):
                return self.lgb_model.predict(
                    X.values if isinstance(X, pd.DataFrame) else X
                )

        wrapped_model = SimpleVolModel(model)
        metrics = {"train_rmse": 0.0}  # Placeholder
        return wrapped_model, metrics


def evaluate_rule_based(
    df_features: pd.DataFrame,
    atr_series: pd.Series,
    params: Dict[str, Any],
) -> Dict[str, float]:
    """评估规则类策略"""
    # 配置SR信号生成
    sqs_min = params.get("sqs_min", 0.5)
    sr_cfg = SRSignalConfig(
        min_sr_strength=params.get("sr_strength_min", 0.5),
        min_support_score=sqs_min,
        min_resistance_score=sqs_min,
        tolerance_mult=params.get("touch_distance_atr", 1.0),
        use_vpin_filter=params.get("use_vpin_filter", False),
        min_vpin=(
            params.get("min_vpin", 0.4)
            if params.get("use_vpin_filter", False)
            else None
        ),
        max_vpin=(
            params.get("max_vpin", 0.6)
            if params.get("use_vpin_filter", False)
            else None
        ),
    )

    # 生成信号
    auto_signals = _generate_sr_reversal_signals(
        df_features,
        price_col="close",
        high_col="high",
        low_col="low",
        atr_series=atr_series,
        cfg=sr_cfg,
    )
    df_features["signal"] = auto_signals

    # 计算RR标签（标准版本和保本版本）
    labels_standard = compute_rr_label(
        df_features.copy(),
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
        use_breakeven_stop=False,
    )

    # 使用保本止损计算保本率
    details_breakeven = compute_rr_label_with_details(
        df_features.copy(),
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
        use_breakeven_stop=True,  # 启用保本止损
    )

    # 统计指标
    mask_valid = (auto_signals != 0) & labels_standard.notna()
    n_trades = int(mask_valid.sum())

    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "breakeven_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "sharpe_ratio": 0.0,
        }

    df_trades = pd.DataFrame(
        {
            "signal": auto_signals[mask_valid],
            "label": labels_standard[mask_valid],
        }
    )

    n_win = int((df_trades["label"] == 1.0).sum())
    win_rate = n_win / n_trades if n_trades > 0 else 0.0

    # 计算保本率
    # 保本率 = 保本+胜利 / (保本+胜利 + 亏损)
    # 其中亏损包括：保本+亏损 和 直接亏损（loss）
    mask_valid_breakeven = (auto_signals != 0) & details_breakeven["label"].notna()
    if mask_valid_breakeven.sum() > 0:
        details_valid = details_breakeven[mask_valid_breakeven]
        n_breakeven_win = int((details_valid["final_result"] == "breakeven_win").sum())
        n_loss_total = int(
            (details_valid["final_result"] == "breakeven_loss").sum()
            + (details_valid["final_result"] == "loss").sum()
        )
        breakeven_rate = (
            n_breakeven_win / (n_breakeven_win + n_loss_total)
            if (n_breakeven_win + n_loss_total) > 0
            else 0.0
        )
    else:
        breakeven_rate = 0.0

    # 计算R
    stop_loss_r = params.get("stop_loss_r", 1.0)
    take_profit_r = params.get("take_profit_r", 2.0)
    realized_r = np.where(
        df_trades["label"].values == 1.0,
        take_profit_r,
        -stop_loss_r,
    )
    total_r = float(realized_r.sum())  # Total R = 所有交易的R总和（成功+失败）
    avg_r = float(realized_r.mean())

    # 计算Sharpe ratio（基于R序列，简化版）
    # 注意：R不是收益率，这里使用R的均值/标准差作为风险调整后的表现指标
    # 不乘以sqrt(252)，因为R不是收益率，且交易频率不是每天
    if len(realized_r) > 1:
        r_mean = np.mean(realized_r)
        r_std = np.std(realized_r)
        if r_std > 1e-8:
            sharpe_ratio = float(r_mean / r_std)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "breakeven_rate": breakeven_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "sharpe_ratio": sharpe_ratio,
    }


def evaluate_ml_model(
    df_features: pd.DataFrame,
    atr_series: pd.Series,
    ml_model: Any,
    params: Dict[str, Any],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """评估ML模型策略"""
    # 生成信号（使用ML预测）
    feature_cols = [
        col
        for col in df_features.columns
        if col
        not in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "signal",
            "label",
            "atr",
            "_symbol",
            "symbol",
            "timestamp",
            "datetime",
            "date",
        ]
    ]
    # Filter to numeric columns only
    numeric_cols = (
        df_features[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )
    X = df_features[numeric_cols].fillna(0)

    # 获取预测
    preds_proba = (
        ml_model.predict_proba(X)
        if hasattr(ml_model, "predict_proba")
        else ml_model.predict(X)
    )
    if len(preds_proba.shape) > 1:
        preds_proba = preds_proba[
            :, 1
        ]  # Binary classification: get positive class probability

    # 生成SR信号（规则类）
    sqs_min = params.get("sqs_min", 0.5)
    sr_cfg = SRSignalConfig(
        min_sr_strength=params.get("sr_strength_min", 0.5),
        min_support_score=sqs_min,
        min_resistance_score=sqs_min,
        tolerance_mult=params.get("touch_distance_atr", 1.0),
        use_vpin_filter=params.get("use_vpin_filter", False),
    )

    auto_signals = _generate_sr_reversal_signals(
        df_features,
        price_col="close",
        high_col="high",
        low_col="low",
        atr_series=atr_series,
        cfg=sr_cfg,
    )

    # 结合ML预测：只有当ML预测概率 >= threshold 时才交易
    ml_signals = np.where(
        (auto_signals != 0) & (preds_proba >= threshold),
        auto_signals,
        0,
    )
    df_features["signal"] = ml_signals

    # 计算RR标签
    labels = compute_rr_label(
        df_features.copy(),
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
        use_breakeven_stop=False,
    )

    # 统计指标
    mask_valid = (ml_signals != 0) & labels.notna()
    n_trades = int(mask_valid.sum())

    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "breakeven_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "sharpe_ratio": 0.0,
        }

    df_trades = pd.DataFrame(
        {
            "signal": ml_signals[mask_valid],
            "label": labels[mask_valid],
        }
    )

    n_win = int((df_trades["label"] == 1.0).sum())
    win_rate = n_win / n_trades if n_trades > 0 else 0.0

    # 计算R
    stop_loss_r = params.get("stop_loss_r", 1.0)
    take_profit_r = params.get("take_profit_r", 2.0)
    realized_r = np.where(
        df_trades["label"].values == 1.0,
        take_profit_r,
        -stop_loss_r,
    )
    total_r = float(realized_r.sum())  # Total R = 所有交易的R总和（成功+失败）
    avg_r = float(realized_r.mean())

    # 计算Sharpe ratio（基于R序列，简化版）
    if len(realized_r) > 1:
        r_mean = np.mean(realized_r)
        r_std = np.std(realized_r)
        if r_std > 1e-8:
            sharpe_ratio = float(r_mean / r_std)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "breakeven_rate": 0.0,  # ML模型暂不支持保本率
        "total_r": total_r,
        "avg_r": avg_r,
        "sharpe_ratio": sharpe_ratio,
    }


def evaluate_ml_volatility_model(
    df_features: pd.DataFrame,
    atr_series: pd.Series,
    ml_model: Any,
    vol_model: Any,
    params: Dict[str, Any],
    threshold: float = 0.5,
    atr_lower_bound: float = 0.8,
    atr_upper_bound: float = 1.5,
) -> Dict[str, float]:
    """评估ML+波动率模型策略（使用预测波动率动态调整R/R）"""
    # 生成信号（使用ML预测）
    feature_cols = [
        col
        for col in df_features.columns
        if col
        not in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "signal",
            "label",
            "atr",
            "_symbol",
            "symbol",
            "timestamp",
            "datetime",
            "date",
        ]
    ]
    # Filter to numeric columns only
    numeric_cols = (
        df_features[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )
    X = df_features[numeric_cols].fillna(0)

    # 获取ML预测
    preds_proba = (
        ml_model.predict_proba(X)
        if hasattr(ml_model, "predict_proba")
        else ml_model.predict(X)
    )
    if len(preds_proba.shape) > 1:
        preds_proba = preds_proba[:, 1]

    # 获取波动率预测（相对波动率，例如0.007475 = 0.75%）
    pred_vol_relative = vol_model.predict(X)
    pred_vol_relative = np.maximum(pred_vol_relative, 0.0)  # Ensure non-negative

    # 将相对波动率转换为绝对波动率（乘以价格）
    prices = df_features["close"].values
    pred_vol = pred_vol_relative * prices  # 绝对波动率

    # 调试：检查预测波动率的分布
    print(f"   📊 Predicted volatility stats (relative):")
    print(
        f"      Mean: {np.mean(pred_vol_relative):.6f} ({np.mean(pred_vol_relative)*100:.2f}%)"
    )
    print(f"      Std: {np.std(pred_vol_relative):.6f}")
    print(f"      Min: {np.min(pred_vol_relative):.6f}")
    print(f"      Max: {np.max(pred_vol_relative):.6f}")
    print(f"      Median: {np.median(pred_vol_relative):.6f}")

    # 检查ATR的分布
    atr_values = atr_series.values
    print(f"   📊 ATR stats:")
    print(f"      Mean: {np.mean(atr_values):.2f}")
    print(f"      Std: {np.std(atr_values):.2f}")
    print(f"      Min: {np.min(atr_values):.2f}")
    print(f"      Max: {np.max(atr_values):.2f}")
    print(f"      Median: {np.median(atr_values):.2f}")

    # 检查预测波动率（绝对）与ATR的比率
    vol_atr_ratio = pred_vol / (atr_values + 1e-8)
    print(f"   📊 Predicted Vol (absolute) / ATR ratio:")
    print(f"      Mean: {np.mean(vol_atr_ratio):.3f}")
    print(f"      Std: {np.std(vol_atr_ratio):.3f}")
    print(f"      Min: {np.min(vol_atr_ratio):.3f}")
    print(f"      Max: {np.max(vol_atr_ratio):.3f}")
    print(f"      Median: {np.median(vol_atr_ratio):.3f}")

    # 生成SR信号（规则类）
    sqs_min = params.get("sqs_min", 0.5)
    sr_cfg = SRSignalConfig(
        min_sr_strength=params.get("sr_strength_min", 0.5),
        min_support_score=sqs_min,
        min_resistance_score=sqs_min,
        tolerance_mult=params.get("touch_distance_atr", 1.0),
        use_vpin_filter=params.get("use_vpin_filter", False),
    )

    auto_signals = _generate_sr_reversal_signals(
        df_features,
        price_col="close",
        high_col="high",
        low_col="low",
        atr_series=atr_series,
        cfg=sr_cfg,
    )

    # 结合ML预测
    ml_signals = np.where(
        (auto_signals != 0) & (preds_proba >= threshold),
        auto_signals,
        0,
    )

    # 使用自适应R/R（基于预测波动率）
    # 导入自适应R/R计算函数
    from scripts.diagnostics.compute_adaptive_rr_with_predicted_vol import (
        compute_adaptive_rr_label_with_predicted_vol,
    )

    # 将信号赋值到DataFrame（必须在计算标签之前）
    df_temp = df_features.copy()
    df_temp["signal"] = ml_signals

    # 使用预测波动率计算自适应R/R标签
    # 方案：使用Ensemble方法，混合预测波动率和ATR
    # 这样可以避免预测波动率不准确时的问题
    atr_values = atr_series.values

    # 计算预测波动率与ATR的比率
    vol_atr_ratio = pred_vol / (atr_values + 1e-8)

    # Ensemble: 混合预测波动率和ATR
    # 权重：预测波动率30%，ATR 70%（更保守，因为预测波动率偏高）
    final_vol = 0.3 * pred_vol + 0.7 * atr_values

    print(f"   🔧 Using ensemble method: 30% predicted vol + 70% ATR")
    print(f"   📊 Final vol stats:")
    print(f"      Mean: {np.mean(final_vol):.2f}, Std: {np.std(final_vol):.2f}")
    print(
        f"      Final vol / ATR ratio - Mean: {np.mean(final_vol / (atr_values + 1e-8)):.3f}"
    )

    # 仍然需要clip到合理范围（但范围更宽松，因为已经ensemble了）
    effective_atr_lower = max(atr_lower_bound, 0.7)  # 至少是ATR的70%
    effective_atr_upper = min(atr_upper_bound, 1.3)  # 最多是ATR的130%

    labels = compute_adaptive_rr_label_with_predicted_vol(
        df_temp,
        predicted_vol=final_vol,  # 使用ensemble后的波动率
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=params.get("max_holding_bars", 50),
        stop_loss_multiplier=params.get("stop_loss_r", 1.0),
        take_profit_multiplier=params.get("take_profit_r", 2.0),
        atr_lower_bound=effective_atr_lower,
        atr_upper_bound=effective_atr_upper,
        use_breakeven_stop=True,  # 启用保本止损
        entry_price_col="open",
        entry_offset=1,
    )

    # 统计指标
    mask_valid = (ml_signals != 0) & labels.notna()
    n_trades = int(mask_valid.sum())

    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "breakeven_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "sharpe_ratio": 0.0,
        }

    df_trades = pd.DataFrame(
        {
            "signal": ml_signals[mask_valid],
            "label": labels[mask_valid],
        }
    )

    n_win = int((df_trades["label"] == 1.0).sum())
    win_rate = n_win / n_trades if n_trades > 0 else 0.0

    # 计算R（使用自适应R/R，需要从预测波动率计算实际R值）
    # 对于自适应R/R，每笔交易的R值可能不同，需要根据实际止盈止损计算
    # 简化：使用平均的stop_loss_multiplier和take_profit_multiplier
    stop_loss_multiplier = params.get("stop_loss_r", 1.0)
    take_profit_multiplier = params.get("take_profit_r", 2.0)

    # 对于成功的交易，使用take_profit_multiplier作为R值
    # 对于失败的交易，使用-stop_loss_multiplier作为R值
    realized_r = np.where(
        df_trades["label"].values == 1.0,
        take_profit_multiplier,
        -stop_loss_multiplier,
    )
    total_r = float(realized_r.sum())  # Total R = 所有交易的R总和（成功+失败）
    avg_r = float(realized_r.mean())

    # 计算Sharpe ratio（基于R序列，简化版）
    if len(realized_r) > 1:
        r_mean = np.mean(realized_r)
        r_std = np.std(realized_r)
        if r_std > 1e-8:
            sharpe_ratio = float(r_mean / r_std)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # 计算保本率（需要重新计算带details的标签）
    # 注意：自适应R/R的保本率计算比较复杂，这里暂时返回0
    # 未来可以扩展compute_adaptive_rr_label_with_predicted_vol来支持details
    breakeven_rate = 0.0

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "breakeven_rate": breakeven_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "sharpe_ratio": sharpe_ratio,
    }


def generate_comparison_report(
    rule_results: Dict[str, float],
    ml_results: Dict[str, float],
    ml_vol_results: Dict[str, float],
    output_path: Path,
) -> None:
    """生成对比报告"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SR Reversal Model Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .positive {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .negative {{
            color: #f44336;
            font-weight: bold;
        }}
        .best {{
            background-color: #e8f5e9;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 SR Reversal Model Comparison Report</h1>
        
        <h2>🎯 Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Rule-Based</th>
                    <th>ML Model</th>
                    <th>ML + Volatility Model</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Trades</strong></td>
                    <td>{int(rule_results['n_trades'])}</td>
                    <td>{int(ml_results['n_trades'])}</td>
                    <td>{int(ml_vol_results['n_trades'])}</td>
                </tr>
                <tr>
                    <td><strong>Win Rate</strong></td>
                    <td class="{'best' if rule_results['win_rate'] >= max(ml_results['win_rate'], ml_vol_results['win_rate']) else ''}">{rule_results['win_rate']:.2%}</td>
                    <td class="{'best' if ml_results['win_rate'] >= max(rule_results['win_rate'], ml_vol_results['win_rate']) else ''}">{ml_results['win_rate']:.2%}</td>
                    <td class="{'best' if ml_vol_results['win_rate'] >= max(rule_results['win_rate'], ml_results['win_rate']) else ''}">{ml_vol_results['win_rate']:.2%}</td>
                </tr>
                <tr>
                    <td><strong>Breakeven Rate</strong></td>
                    <td>{rule_results['breakeven_rate']:.2%}</td>
                    <td>{ml_results['breakeven_rate']:.2%}</td>
                    <td>{ml_vol_results['breakeven_rate']:.2%}</td>
                </tr>
                <tr>
                    <td><strong>Total R</strong></td>
                    <td class="{'best positive' if rule_results['total_r'] >= max(ml_results['total_r'], ml_vol_results['total_r']) else ('positive' if rule_results['total_r'] > 0 else 'negative')}">{rule_results['total_r']:.2f}</td>
                    <td class="{'best positive' if ml_results['total_r'] >= max(rule_results['total_r'], ml_vol_results['total_r']) else ('positive' if ml_results['total_r'] > 0 else 'negative')}">{ml_results['total_r']:.2f}</td>
                    <td class="{'best positive' if ml_vol_results['total_r'] >= max(rule_results['total_r'], ml_results['total_r']) else ('positive' if ml_vol_results['total_r'] > 0 else 'negative')}">{ml_vol_results['total_r']:.2f}</td>
                </tr>
                <tr>
                    <td><strong>Avg R per Trade</strong></td>
                    <td class="{'best' if rule_results['avg_r'] >= max(ml_results['avg_r'], ml_vol_results['avg_r']) else ''}">{rule_results['avg_r']:.3f}</td>
                    <td class="{'best' if ml_results['avg_r'] >= max(rule_results['avg_r'], ml_vol_results['avg_r']) else ''}">{ml_results['avg_r']:.3f}</td>
                    <td class="{'best' if ml_vol_results['avg_r'] >= max(rule_results['avg_r'], ml_results['avg_r']) else ''}">{ml_vol_results['avg_r']:.3f}</td>
                </tr>
                <tr>
                    <td><strong>Sharpe Ratio</strong></td>
                    <td class="{'best' if rule_results['sharpe_ratio'] >= max(ml_results['sharpe_ratio'], ml_vol_results['sharpe_ratio']) else ''}">{rule_results['sharpe_ratio']:.2f}</td>
                    <td class="{'best' if ml_results['sharpe_ratio'] >= max(rule_results['sharpe_ratio'], ml_vol_results['sharpe_ratio']) else ''}">{ml_results['sharpe_ratio']:.2f}</td>
                    <td class="{'best' if ml_vol_results['sharpe_ratio'] >= max(rule_results['sharpe_ratio'], ml_results['sharpe_ratio']) else ''}">{ml_vol_results['sharpe_ratio']:.2f}</td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"   ✅ Comparison report saved to {output_path}")


def main() -> None:
    args = parse_args()

    # Load data
    print("📊 Loading data...")
    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Load features
    print("🔧 Loading features...")
    cfg_dir = Path(args.strategy_config).resolve()
    strategy_cfg_loader = StrategyConfigLoader(cfg_dir)
    strategy_cfg = strategy_cfg_loader.load()

    feature_loader = StrategyFeatureLoader()
    df_features = strategy_runner.run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )

    # Ensure ATR
    atr_series = _ensure_atr(df_features, "atr", "close", "high", "low", 14)

    # Split train/test
    split_idx = int(len(df_features) * (1 - args.test_size))
    df_train = df_features.iloc[:split_idx].copy()
    df_test = df_features.iloc[split_idx:].copy()

    # Load optimized rule parameters
    if args.rule_params and Path(args.rule_params).exists():
        # Try to load from CSV (optimization results)
        try:
            results_df = pd.read_csv(args.rule_params)
            if len(results_df) > 0:
                best_row = results_df.loc[results_df["total_r"].idxmax()]
                rule_params = {
                    "sr_strength_min": float(best_row.get("sr_strength_min", 0.3)),
                    "sqs_min": float(best_row.get("sqs_min", 0.7)),
                    "touch_distance_atr": float(
                        best_row.get("touch_distance_atr", 1.5)
                    ),
                    "stop_loss_r": float(best_row.get("stop_loss_r", 1.25)),
                    "take_profit_r": float(best_row.get("take_profit_r", 3.0)),
                    "max_holding_bars": int(best_row.get("max_holding_bars", 72)),
                    "use_vpin_filter": bool(best_row.get("use_vpin_filter", False)),
                    "min_vpin": (
                        float(best_row.get("min_vpin", 0.4))
                        if best_row.get("use_vpin_filter", False)
                        else None
                    ),
                    "max_vpin": (
                        float(best_row.get("max_vpin", 0.6))
                        if best_row.get("use_vpin_filter", False)
                        else None
                    ),
                }
            else:
                raise ValueError("Empty results file")
        except Exception as e:
            print(f"   ⚠️ Could not load rule params from {args.rule_params}: {e}")
            print("   Using default optimized parameters...")
            rule_params = {
                "sr_strength_min": 0.3,
                "sqs_min": 0.7,
                "touch_distance_atr": 1.5,
                "stop_loss_r": 1.25,
                "take_profit_r": 3.0,
                "max_holding_bars": 72,
                "use_vpin_filter": False,
            }
    else:
        # Use default optimized parameters from previous run
        rule_params = {
            "sr_strength_min": 0.3,
            "sqs_min": 0.7,
            "touch_distance_atr": 1.5,
            "stop_loss_r": 1.25,
            "take_profit_r": 3.0,
            "max_holding_bars": 72,
            "use_vpin_filter": False,
        }

    print("\n" + "=" * 60)
    print("1️⃣ Evaluating Rule-Based Strategy")
    print("=" * 60)
    rule_results = evaluate_rule_based(
        df_test, atr_series.iloc[split_idx:], rule_params
    )
    print(f"   Trades: {int(rule_results['n_trades'])}")
    print(f"   Win Rate: {rule_results['win_rate']:.2%}")
    print(f"   Breakeven Rate: {rule_results['breakeven_rate']:.2%}")
    print(f"   Total R: {rule_results['total_r']:.2f}")
    print(f"   Sharpe: {rule_results['sharpe_ratio']:.2f}")

    # Prepare labels for ML training
    print("\n" + "=" * 60)
    print("2️⃣ Preparing Labels for ML Training")
    print("=" * 60)

    # Generate signals for training
    sqs_min = rule_params.get("sqs_min", 0.5)
    sr_cfg = SRSignalConfig(
        min_sr_strength=rule_params.get("sr_strength_min", 0.5),
        min_support_score=sqs_min,
        min_resistance_score=sqs_min,
        tolerance_mult=rule_params.get("touch_distance_atr", 1.0),
        use_vpin_filter=rule_params.get("use_vpin_filter", False),
    )

    train_signals = _generate_sr_reversal_signals(
        df_train,
        price_col="close",
        high_col="high",
        low_col="low",
        atr_series=atr_series.iloc[:split_idx],
        cfg=sr_cfg,
    )
    df_train["signal"] = train_signals

    # Compute labels
    train_labels = compute_rr_label(
        df_train.copy(),
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=rule_params.get("max_holding_bars", 50),
        stop_loss_r=rule_params.get("stop_loss_r", 1.0),
        take_profit_r=rule_params.get("take_profit_r", 2.0),
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
        use_breakeven_stop=False,
    )

    # Compute volatility labels
    train_vol_labels = future_volatility_label(
        df_train["close"],
        horizon=10,
    )

    # Prepare features (exclude non-numeric columns)
    feature_cols = [
        col
        for col in df_train.columns
        if col
        not in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "signal",
            "label",
            "atr",
            "_symbol",
            "symbol",
            "timestamp",
            "datetime",
            "date",
        ]
    ]

    # Filter to numeric columns only
    numeric_cols = (
        df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )
    X_train = df_train[numeric_cols].fillna(0)
    y_train = train_labels.fillna(0).astype(int)
    y_vol_train = train_vol_labels.fillna(train_vol_labels.median())

    # Filter valid samples
    valid_mask = (train_signals != 0) & train_labels.notna()
    X_train_valid = X_train[valid_mask]
    y_train_valid = y_train[valid_mask]
    y_vol_train_valid = y_vol_train[valid_mask]

    print(f"   Training samples: {len(X_train_valid)}")
    print(
        f"   Positive labels: {int(y_train_valid.sum())} ({y_train_valid.mean():.2%})"
    )

    # Train ML model
    print("\n" + "=" * 60)
    print("3️⃣ Training ML Model")
    print("=" * 60)
    ml_model, ml_metrics = train_ml_model(
        X_train_valid,
        y_train_valid,
        X_train_valid,  # Use same data for test (simplified)
        y_train_valid,
    )

    # Train volatility model
    print("\n" + "=" * 60)
    print("4️⃣ Training Volatility Model")
    print("=" * 60)
    vol_model, vol_metrics = train_volatility_model(
        X_train_valid,
        y_vol_train_valid,
        X_train_valid,
        y_vol_train_valid,
    )

    # Evaluate ML model
    print("\n" + "=" * 60)
    print("5️⃣ Evaluating ML Model")
    print("=" * 60)
    ml_results = evaluate_ml_model(
        df_test,
        atr_series.iloc[split_idx:],
        ml_model,
        rule_params,
        threshold=0.5,
    )
    print(f"   Trades: {int(ml_results['n_trades'])}")
    print(f"   Win Rate: {ml_results['win_rate']:.2%}")
    print(f"   Total R: {ml_results['total_r']:.2f}")
    print(f"   Sharpe: {ml_results['sharpe_ratio']:.2f}")

    # Evaluate ML + Volatility model
    print("\n" + "=" * 60)
    print("6️⃣ Evaluating ML + Volatility Model")
    print("=" * 60)
    ml_vol_results = evaluate_ml_volatility_model(
        df_test,
        atr_series.iloc[split_idx:],
        ml_model,
        vol_model,
        rule_params,
        threshold=0.5,
    )
    print(f"   Trades: {int(ml_vol_results['n_trades'])}")
    print(f"   Win Rate: {ml_vol_results['win_rate']:.2%}")
    print(f"   Total R: {ml_vol_results['total_r']:.2f}")
    print(f"   Sharpe: {ml_vol_results['sharpe_ratio']:.2f}")

    # Generate comparison report
    print("\n" + "=" * 60)
    print("7️⃣ Generating Comparison Report")
    print("=" * 60)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_comparison_report(
        rule_results,
        ml_results,
        ml_vol_results,
        output_dir / "comparison_report.html",
    )

    # Save results to CSV
    results_df = pd.DataFrame(
        {
            "Method": ["Rule-Based", "ML Model", "ML + Volatility Model"],
            "Trades": [
                rule_results["n_trades"],
                ml_results["n_trades"],
                ml_vol_results["n_trades"],
            ],
            "Win Rate": [
                rule_results["win_rate"],
                ml_results["win_rate"],
                ml_vol_results["win_rate"],
            ],
            "Breakeven Rate": [
                rule_results["breakeven_rate"],
                ml_results["breakeven_rate"],
                ml_vol_results["breakeven_rate"],
            ],
            "Total R": [
                rule_results["total_r"],
                ml_results["total_r"],
                ml_vol_results["total_r"],
            ],
            "Avg R": [
                rule_results["avg_r"],
                ml_results["avg_r"],
                ml_vol_results["avg_r"],
            ],
            "Sharpe Ratio": [
                rule_results["sharpe_ratio"],
                ml_results["sharpe_ratio"],
                ml_vol_results["sharpe_ratio"],
            ],
        }
    )
    results_df.to_csv(output_dir / "comparison_results.csv", index=False)

    print(f"\n✅ Comparison complete!")
    print(f"   Results saved to {output_dir}")


if __name__ == "__main__":
    main()
