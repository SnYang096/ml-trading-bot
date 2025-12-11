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
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402
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
from src.time_series_model.pipeline.training.volatility_model_config import (  # noqa: E402
    load_volatility_model_config,
    get_volatility_model_params,
    get_categorical_features,
    prepare_volatility_model_data,
)
from src.data_tools.tick_loader import build_tick_loader_payload  # noqa: E402

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


def _timeframe_to_minutes(tf: str) -> Optional[int]:
    """Convert timeframe string to minutes."""
    tf = (tf or "").strip().upper()
    if tf.endswith("T"):
        try:
            return int(float(tf[:-1]))
        except ValueError:
            return None
    if tf.endswith("H"):
        try:
            return int(float(tf[:-1]) * 60)
        except ValueError:
            return None
    if tf.endswith("D"):
        try:
            return int(float(tf[:-1]) * 1440)
        except ValueError:
            return None
    if tf.isdigit():
        return int(tf)
    return None


# Removed _should_use_tick_data - always use tick data for VPIN


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
    # Removed --tick-data-mode argument - tick data is always enabled for VPIN
    parser.add_argument(
        "--ticks-dir",
        type=str,
        default="data/parquet_data",
        help="Directory containing tick parquet files for VPIN.",
    )
    parser.add_argument(
        "--ticks-lookback-minutes",
        type=int,
        default=60,
        help="Extra minutes of tick history to load before/after the data window.",
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
    config_path: Optional[Path | str] = None,
    feature_loader: Optional[Any] = None,
) -> Tuple[Any, Dict[str, float]]:
    """
    训练波动率模型（使用配置文件选择特征和参数）

    特征选择优先级（根据配置文件）：
    1. VPIN Volatility 特征（核心特征，参考文档强调的重要性）
    2. VPIN 衍生特征（VPIN volatility ratio, spike等）
    3. GARCH 特征（波动聚集性和杠杆效应）
    4. 扩展波动率特征（历史波动率、滞后特征、趋势特征）
    5. ATR 相关特征
    6. 其他波动率相关特征

    注意：
    - EVT特征不用于波动率预测，而是用于风险管理/仓位控制（离场、不加仓）
    - DTW特征不用于波动率模型，而是用于SR Reversal策略（反转模板匹配）

    Args:
        X_train: 训练特征DataFrame
        y_vol_train: 训练波动率标签
        X_test: 测试特征DataFrame
        y_vol_test: 测试波动率标签
        config_path: 波动率模型配置文件路径，如果为None，使用默认路径

    Returns:
        (模型, 指标字典)
    """
    print("   📊 Training volatility model with config-based feature selection...")

    # 加载配置
    try:
        config = load_volatility_model_config(config_path)
        print("   ✅ Loaded volatility model config")
    except Exception as e:
        print(f"   ⚠️ Failed to load config: {e}, using default feature selection")
        config = None

    if config is not None:
        X_train_prepared, available_features, categorical_features = (
            prepare_volatility_model_data(
                X_train, config, feature_loader=feature_loader
            )
        )
        X_test_prepared, _, _ = prepare_volatility_model_data(
            X_test, config, feature_loader=feature_loader
        )
        if not available_features:
            print("   ⚠️ No volatility-specific features found, using all features")
            available_features = list(X_train_prepared.columns)
        else:
            print(
                f"   ✅ Using {len(available_features)} volatility features from config"
            )
    else:
        # Fallback: 使用原有的特征选择逻辑
        print("   ⚠️ Using fallback feature selection (no config)")
        volatility_relevant_features = []

        # GARCH特征
        garch_features = [col for col in X_train.columns if col.startswith("garch_")]
        volatility_relevant_features.extend(garch_features)

        # 扩展波动率特征
        extended_vol_features = [
            col for col in X_train.columns if col.startswith("vol_")
        ]
        volatility_relevant_features.extend(extended_vol_features)

        # ATR相关特征
        atr_features = [col for col in X_train.columns if "atr" in col.lower()]
        volatility_relevant_features.extend(atr_features)

        # VPIN volatility特征（如果存在）
        vpin_vol_features = [
            col
            for col in X_train.columns
            if col.startswith("vpin_volatility") or col.startswith("vpin_vol")
        ]
        volatility_relevant_features.extend(vpin_vol_features)

        # 其他波动率相关特征
        other_features = [
            col
            for col in X_train.columns
            if any(
                keyword in col.lower()
                for keyword in [
                    "bb_width",
                    "compression",
                    "squeeze",
                    "range",
                    "range_ratio",
                ]
            )
        ]
        volatility_relevant_features.extend(other_features)

        # 排除EVT和DTW特征
        volatility_relevant_features = [
            f
            for f in volatility_relevant_features
            if not f.startswith("evt_") and not f.startswith("dtw_")
        ]

        available_features = list(set(volatility_relevant_features))
        available_features = [f for f in available_features if f in X_train.columns]

        if not available_features:
            available_features = list(X_train.columns)

        X_train_prepared = X_train
        X_test_prepared = X_test
        categorical_features = None

    # 使用选定的特征
    X_train_vol = X_train_prepared[available_features].copy()
    X_test_vol = X_test_prepared[available_features].copy()

    # 获取分类特征
    if config is None and categorical_features is None:
        if "_symbol" in X_train_vol.columns and X_train_vol["_symbol"].nunique() > 1:
            categorical_features = ["_symbol"]

    if categorical_features:
        print(f"   ✅ Using categorical features: {categorical_features}")

    # 获取训练参数
    if config is not None:
        trainer_config = config.get("trainer", {})
        use_gpu = trainer_config.get("use_gpu", True)
        n_splits = trainer_config.get("n_splits", 5)
        auto_tune_params = trainer_config.get("auto_tune_params", False)
        model_params = get_volatility_model_params(config)
    else:
        use_gpu = True
        n_splits = 5
        auto_tune_params = False
        model_params = None

    try:
        model = LightGBMTrainer(model_type="regression", use_gpu=use_gpu)

        # 如果配置了模型参数，设置它们
        if model_params:
            model.params = model_params

        metrics, _ = model.train(
            X_train_vol,
            y_vol_train,
            n_splits=n_splits,
            use_time_series_cv=True,
            groups=None,
            auto_tune_params=auto_tune_params,
            categorical_features=categorical_features,
        )

        # 存储使用的特征列表，供预测时使用
        model._volatility_features = available_features
        if categorical_features:
            model._categorical_features = categorical_features

        return model, metrics
    except Exception as e:
        print(f"   ⚠️ LightGBMTrainer failed: {e}")
        print("   Using simple LightGBM instead...")
        import lightgbm as lgb

        # Simple LightGBM training - 使用选定的特征
        X_train_vol_values = X_train_vol.values
        train_data = lgb.Dataset(X_train_vol_values, label=y_vol_train.values)

        # 使用配置中的参数或默认参数
        if model_params:
            params = model_params.copy()
        else:
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
            def __init__(self, lgb_model, features=None):
                self.lgb_model = lgb_model
                self.is_trained = True
                self._volatility_features = features  # 存储特征列表

            def predict(self, X):
                # 如果指定了特征，只使用这些特征
                if self._volatility_features and isinstance(X, pd.DataFrame):
                    X_used = X[self._volatility_features].copy()
                else:
                    X_used = X

                return self.lgb_model.predict(
                    X_used.values if isinstance(X_used, pd.DataFrame) else X_used
                )

        wrapped_model = SimpleVolModel(model, features=available_features)
        # 存储特征列表供预测使用
        wrapped_model._volatility_features = available_features
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
    from src.time_series_model.diagnostics.compute_adaptive_rr_with_predicted_vol import (
        compute_adaptive_rr_label_with_predicted_vol,
    )

    # 将信号赋值到DataFrame（必须在计算标签之前）
    df_temp = df_features.copy()
    df_temp["signal"] = ml_signals

    # 添加未来波动率标签用于分析（如果还没有）
    # 注意：必须在完整的df_features上计算，然后对齐到df_temp（测试集）
    if "future_volatility" not in df_temp.columns:
        # 问题：如果df_temp是测试集，直接在上面计算会导致最后horizon行无法计算
        # 解决方案：在完整的df_features上计算，然后对齐到df_temp
        if "future_volatility" in df_features.columns:
            # 如果df_features已经有未来波动率标签，直接使用
            df_temp["future_volatility"] = df_features.loc[
                df_temp.index, "future_volatility"
            ]
        else:
            # 在完整的df_features上计算未来波动率标签
            future_vol_full = future_volatility_label(
                df_features["close"],
                horizon=10,
            )
            # 对齐到df_temp（测试集）
            df_temp["future_volatility"] = future_vol_full.loc[df_temp.index]

            # 调试：检查计算是否正确
            if df_temp["future_volatility"].notna().sum() > 0:
                print(f"   🔍 Future volatility label debug:")
                print(f"      Total samples in df_temp: {len(df_temp)}")
                print(
                    f"      Non-NaN future_vol samples: {df_temp['future_volatility'].notna().sum()}"
                )
                print(f"      Mean: {df_temp['future_volatility'].mean():.8f}")
                print(f"      Median: {df_temp['future_volatility'].median():.8f}")
                print(f"      Min: {df_temp['future_volatility'].min():.8f}")
                print(f"      Max: {df_temp['future_volatility'].max():.8f}")
                print(
                    f"      First 5 non-NaN values: {df_temp['future_volatility'].dropna().head(5).tolist()}"
                )

    # 使用预测波动率计算自适应R/R标签
    # 去掉Ensemble方法，直接使用预测波动率
    atr_values = atr_series.values

    # 计算预测波动率与ATR的比率
    vol_atr_ratio = pred_vol / (atr_values + 1e-8)

    print(f"   🔧 Using predicted volatility directly (no ensemble)")
    print(f"   📊 Predicted vol / ATR ratio stats:")
    print(f"      Mean: {np.mean(vol_atr_ratio):.3f}, Std: {np.std(vol_atr_ratio):.3f}")
    print(f"      Min: {np.min(vol_atr_ratio):.3f}, Max: {np.max(vol_atr_ratio):.3f}")
    print(f"      Median: {np.median(vol_atr_ratio):.3f}")

    # 分析波动率预测准确性（如果有未来波动率标签）
    if "future_volatility" in df_temp.columns:
        future_vol = df_temp["future_volatility"].values
        valid_mask = ~(np.isnan(pred_vol) | np.isnan(future_vol) | np.isnan(atr_values))
        if valid_mask.sum() > 0:
            pred_vol_valid = pred_vol[valid_mask]
            future_vol_valid = future_vol[valid_mask]
            atr_valid = atr_values[valid_mask]

            # 检查未来波动率标签是否有问题
            if np.mean(future_vol_valid) == 0.0:
                print(f"   ⚠️  警告：未来波动率标签均值为0，可能存在计算问题")
                print(f"      未来波动率标签统计:")
                print(f"        非NaN数量: {np.sum(~np.isnan(future_vol))}")
                print(f"        均值: {np.mean(future_vol_valid):.6f}")
                print(f"        中位数: {np.median(future_vol_valid):.6f}")
                print(f"        标准差: {np.std(future_vol_valid):.6f}")
                print(f"        最小值: {np.min(future_vol_valid):.6f}")
                print(f"        最大值: {np.max(future_vol_valid):.6f}")

            # 计算预测误差（统一单位：都转换为相对ATR的比率）
            # 注意：future_vol_valid 是相对波动率（RMS of returns，例如0.0066 = 0.66%）
            # 需要先转换为绝对波动率，再除以ATR
            prices_valid = df_temp.loc[df_temp.index[valid_mask], "close"].values
            future_vol_absolute = future_vol_valid * prices_valid  # 转换为绝对波动率
            future_vol_relative = future_vol_absolute / (
                atr_valid + 1e-8
            )  # 转换为相对ATR的比率

            pred_vol_relative = pred_vol_valid / (
                atr_valid + 1e-8
            )  # 预测波动率已经是绝对波动率

            error = pred_vol_relative - future_vol_relative
            mae = np.mean(np.abs(error))
            rmse = np.sqrt(np.mean(error**2))

            # 计算相关性（需要有效数据）
            if (
                len(pred_vol_relative) > 1
                and np.std(pred_vol_relative) > 1e-8
                and np.std(future_vol_relative) > 1e-8
            ):
                correlation = np.corrcoef(pred_vol_relative, future_vol_relative)[0, 1]
            else:
                correlation = np.nan

            print(f"   📊 Volatility Prediction Accuracy:")
            print(f"      Valid samples: {len(pred_vol_relative)}")
            print(f"      MAE (relative to ATR): {mae:.4f}")
            print(f"      RMSE (relative to ATR): {rmse:.4f}")
            if not np.isnan(correlation):
                print(f"      Correlation: {correlation:.4f}")
            print(
                f"      Predicted mean: {np.mean(pred_vol_relative):.3f}, Actual mean: {np.mean(future_vol_relative):.3f}"
            )
            print(
                f"      Predicted median: {np.median(pred_vol_relative):.3f}, Actual median: {np.median(future_vol_relative):.3f}"
            )

    # 直接使用预测波动率，但需要clip到合理范围
    effective_atr_lower = atr_lower_bound
    effective_atr_upper = atr_upper_bound

    # 使用带详细信息的函数来计算标签
    from src.time_series_model.diagnostics.compute_adaptive_rr_with_predicted_vol import (
        compute_adaptive_rr_label_with_predicted_vol_details,
    )

    # 如果函数存在，使用详细信息版本；否则使用普通版本
    breakeven_info = None
    try:
        result_details = compute_adaptive_rr_label_with_predicted_vol_details(
            df_temp,
            predicted_vol=pred_vol,  # 直接使用预测波动率
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
        labels = result_details["label"]
        breakeven_info = result_details
    except (ImportError, AttributeError, NameError) as e:
        # 如果详细信息版本不存在，使用普通版本
        print(f"   ⚠️  Using standard version (details not available: {e})")
        labels = compute_adaptive_rr_label_with_predicted_vol(
            df_temp,
            predicted_vol=pred_vol,  # 直接使用预测波动率
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

    # 如果有详细信息，分析保本止损触发情况和自适应R/R逻辑
    if breakeven_info is not None and isinstance(breakeven_info, pd.DataFrame):
        valid_indices = df_temp.index[mask_valid]
        if len(valid_indices) > 0 and all(
            idx in breakeven_info.index for idx in valid_indices
        ):
            breakeven_activated = breakeven_info.loc[
                valid_indices, "breakeven_activated"
            ].fillna(False)
            n_breakeven_activated = int(breakeven_activated.sum())
            final_results = breakeven_info.loc[valid_indices, "final_result"]
            n_breakeven_win = int(
                (final_results == "breakeven_win").fillna(False).sum()
            )
            n_breakeven_loss = int(
                (final_results == "breakeven_loss").fillna(False).sum()
            )
            n_loss_total = int((labels[mask_valid] == 0.0).sum())

            print(f"   📊 Breakeven Stop-Loss Analysis:")
            print(f"      Total trades: {n_trades}")
            print(
                f"      Breakeven activated: {n_breakeven_activated} ({100*n_breakeven_activated/n_trades:.1f}%)"
            )
            print(f"      Breakeven → Win: {n_breakeven_win}")
            print(f"      Breakeven → Loss: {n_breakeven_loss}")
            print(f"      Total losses: {n_loss_total}")
            if n_breakeven_win + n_loss_total > 0:
                breakeven_rate_calc = n_breakeven_win / (n_breakeven_win + n_loss_total)
                print(f"      Breakeven rate: {100*breakeven_rate_calc:.2f}%")

            # 分析自适应R/R逻辑
            pred_vol_used = breakeven_info.loc[valid_indices, "predicted_vol_used"]
            stop_loss_prices = breakeven_info.loc[valid_indices, "stop_loss_price"]
            take_profit_prices = breakeven_info.loc[valid_indices, "take_profit_price"]
            entry_prices = df_temp.loc[valid_indices, "open"]

            # 计算SL/TP距离
            sl_distances = np.abs(stop_loss_prices - entry_prices)
            tp_distances = np.abs(take_profit_prices - entry_prices)
            atr_valid = atr_series.loc[valid_indices]
            sl_atr_ratios = sl_distances / (atr_valid + 1e-8)
            tp_atr_ratios = tp_distances / (atr_valid + 1e-8)

            print(f"   📊 Adaptive R/R Analysis:")
            print(
                f"      Predicted vol used - Mean: {pred_vol_used.mean():.2f}, Std: {pred_vol_used.std():.2f}"
            )
            print(
                f"      SL distance / ATR - Mean: {sl_atr_ratios.mean():.3f}, Std: {sl_atr_ratios.std():.3f}"
            )
            print(
                f"      TP distance / ATR - Mean: {tp_atr_ratios.mean():.3f}, Std: {tp_atr_ratios.std():.3f}"
            )
            print(
                f"      SL distance / ATR - Min: {sl_atr_ratios.min():.3f}, Max: {sl_atr_ratios.max():.3f}"
            )
            print(
                f"      TP distance / ATR - Min: {tp_atr_ratios.min():.3f}, Max: {tp_atr_ratios.max():.3f}"
            )

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

    # 计算保本率（如果有详细信息）
    breakeven_rate = 0.0
    if breakeven_info is not None and isinstance(breakeven_info, pd.DataFrame):
        valid_indices = df_temp.index[mask_valid]
        if len(valid_indices) > 0 and all(
            idx in breakeven_info.index for idx in valid_indices
        ):
            final_results = breakeven_info.loc[valid_indices, "final_result"]
            breakeven_win = int((final_results == "breakeven_win").fillna(False).sum())
            n_loss_total = int((labels[mask_valid] == 0.0).sum())
            if breakeven_win + n_loss_total > 0:
                breakeven_rate = breakeven_win / (breakeven_win + n_loss_total)

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

    # Always configure tick loader for VPIN (required for all timeframes)
    if df_raw.empty:
        raise ValueError("No bars available for tick-loader configuration.")
    print(f"   📦 Tick data enabled for VPIN (always enabled for all timeframes)")
    print(f"   📁 Using ticks_dir: {args.ticks_dir}")
    tick_loader_json = build_tick_loader_payload(
        symbol=args.symbol.upper(),
        start_ts=df_raw.index.min().isoformat(),
        end_ts=df_raw.index.max().isoformat(),
        ticks_dir=args.ticks_dir,
        lookback_minutes=args.ticks_lookback_minutes,
    )

    feature_loader = StrategyFeatureLoader()
    if tick_loader_json:
        vpin_feature = feature_loader.feature_deps.get("features", {}).get(
            "vpin_features"
        )
        if vpin_feature is not None:
            vpin_feature.setdefault("compute_params", {})[
                "ticks_loader_json"
            ] = tick_loader_json
        else:
            print(
                "   ⚠️  vpin_features missing in feature dependencies; cannot pass tick loader."
            )

    # ⚠️ CRITICAL FIX: Split train/test BEFORE feature fitting to avoid look-ahead bias
    # Split raw data first
    split_idx = int(len(df_raw) * (1 - args.test_size))
    df_raw_train = df_raw.iloc[:split_idx].copy()
    df_raw_test = df_raw.iloc[split_idx:].copy()

    # Fit features on training set only
    print("   🔧 Fitting features on training set only (to avoid look-ahead bias)...")
    df_train = strategy_runner.run_feature_pipeline(
        df_raw_train,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,  # Fit on training set
    )

    # Transform test set using fitted features (fit=False)
    # Note: tick_loader_json is already configured in feature_loader, so it will be used for test set too
    print("   🔧 Transforming test set using fitted features...")

    # Ensure tick_loader_json is still available for test set (needed for VPIN)
    if tick_loader_json:
        vpin_feature = feature_loader.feature_deps.get("features", {}).get(
            "vpin_features"
        )
        if vpin_feature is not None:
            vpin_feature.setdefault("compute_params", {})[
                "ticks_loader_json"
            ] = tick_loader_json
            # Debug: verify ticks_loader_json is set
            actual_value = vpin_feature.get("compute_params", {}).get(
                "ticks_loader_json"
            )
            if actual_value:
                print(f"   ✅ VPIN ticks_loader_json configured for test set")
            else:
                print(
                    f"   ⚠️  WARNING: ticks_loader_json not found in VPIN compute_params!"
                )
        else:
            print(f"   ⚠️  WARNING: vpin_features not found in feature_deps!")

    df_test = strategy_runner.run_feature_pipeline(
        df_raw_test,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=False,  # Don't fit on test set!
    )

    # Combine for ATR calculation (ATR is a simple rolling window, safe to compute on full data)
    # Check for duplicate indices before concat
    train_dup = df_train.index.duplicated().sum()
    test_dup = df_test.index.duplicated().sum()
    if train_dup > 0 or test_dup > 0:
        print(
            f"   ⚠️  Warning: Duplicate indices detected (train: {train_dup}, test: {test_dup})"
        )
        # Remove duplicates (keep last)
        df_train = df_train[~df_train.index.duplicated(keep="last")]
        df_test = df_test[~df_test.index.duplicated(keep="last")]

    # Use index-based split instead of iloc to preserve original indices
    df_features = pd.concat([df_train, df_test]).sort_index()

    # Check for overlapping indices between train and test
    overlap = set(df_train.index) & set(df_test.index)
    if overlap:
        print(
            f"   ⚠️  Warning: {len(overlap)} overlapping indices between train and test"
        )
        # Remove overlapping indices from test set
        df_test = df_test[~df_test.index.isin(overlap)]
        df_features = pd.concat([df_train, df_test]).sort_index()

    # Store original train/test indices before ATR calculation
    train_indices = df_train.index
    test_indices = df_test.index

    # Ensure ATR
    atr_series = _ensure_atr(df_features, "atr", "close", "high", "low", 14)

    # Re-split using original indices (not iloc) to preserve index alignment
    df_train = df_features.loc[train_indices].copy()
    df_test = df_features.loc[test_indices].copy()

    # Split ATR series using original indices
    atr_train = atr_series.loc[train_indices]
    atr_test = atr_series.loc[test_indices]

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
    rule_results = evaluate_rule_based(df_test, atr_test, rule_params)
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
        atr_series=atr_train,
        cfg=sr_cfg,
    )
    df_train["signal"] = train_signals

    # Debug: Check signal generation
    n_signals = int((train_signals != 0).sum())
    print(
        f"   📊 Generated {n_signals} signals in training set (out of {len(df_train)} samples)"
    )

    # Compute labels
    # Debug: Check df_train before label computation
    print(
        f"   🔍 Debug df_train: shape={df_train.shape}, index range=[{df_train.index[0]} to {df_train.index[-1]}]"
    )
    print(
        f"   🔍 Debug df_train: duplicate indices={df_train.index.duplicated().sum()}"
    )
    print(
        f"   🔍 Debug signals: shape={train_signals.shape}, non-zero={int((train_signals != 0).sum())}"
    )
    print(
        f"   🔍 Debug signals: duplicate indices={train_signals.index.duplicated().sum()}"
    )
    print(f"   🔍 Debug df_train columns: {sorted(df_train.columns.tolist())[:10]}...")

    # Check if df_train has required columns for label computation
    required_cols = ["close", "high", "low", "open", "atr", "signal"]
    missing_cols = [col for col in required_cols if col not in df_train.columns]
    if missing_cols:
        print(f"   ⚠️  Missing required columns for label computation: {missing_cols}")

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

    # Debug: Check label computation result
    print(
        f"   🔍 Debug labels: shape={train_labels.shape}, not NaN={int(train_labels.notna().sum())}, NaN={int(train_labels.isna().sum())}"
    )
    if train_labels.notna().sum() == 0 and (train_signals != 0).sum() > 0:
        # Check if signals and labels have matching indices
        signal_indices = train_signals[train_signals != 0].index
        label_indices = train_labels.index
        print(
            f"   🔍 Debug indices: signal indices match={signal_indices.equals(label_indices)}"
        )
        print(
            f"   🔍 Debug: First 5 signal indices with non-zero: {signal_indices[:5].tolist()}"
        )
        print(f"   🔍 Debug: First 5 label indices: {label_indices[:5].tolist()}")

    # Compute volatility labels
    # 注意：future_volatility_label使用未来数据，这是正确的（标签可以使用未来信息）
    # 但为了索引对齐，直接在df_train上计算
    if "future_volatility" not in df_train.columns:
        # 在训练集上计算未来波动率标签
        train_vol_labels = future_volatility_label(
            df_train["close"],
            horizon=10,
        )
    else:
        train_vol_labels = df_train["future_volatility"]

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

    # Debug: Check why valid samples might be 0
    n_signals_nonzero = int((train_signals != 0).sum())
    n_labels_notna = int(train_labels.notna().sum())
    n_valid = int(valid_mask.sum())
    print(
        f"   🔍 Debug: Signals (non-zero): {n_signals_nonzero}, Labels (not NaN): {n_labels_notna}, Valid: {n_valid}"
    )

    if n_valid == 0:
        print(f"   ⚠️  WARNING: No valid training samples!")
        print(f"      This might be due to:")
        print(f"      - No signals generated (signals non-zero: {n_signals_nonzero})")
        print(f"      - No valid labels (labels not NaN: {n_labels_notna})")
        print(f"      - Index mismatch between signals and labels")
        if n_signals_nonzero > 0 and n_labels_notna == 0:
            print(
                f"      ⚠️  Signals exist but labels are all NaN - check label computation"
            )
        if n_signals_nonzero == 0:
            print(f"      ⚠️  No signals generated - check signal generation parameters")

    X_train_valid = X_train[valid_mask]
    y_train_valid = y_train[valid_mask]
    y_vol_train_valid = y_vol_train[valid_mask]

    print(f"   Training samples: {len(X_train_valid)}")
    if len(X_train_valid) > 0:
        print(
            f"   Positive labels: {int(y_train_valid.sum())} ({y_train_valid.mean():.2%})"
        )
    else:
        print(f"   ⚠️  Cannot proceed: No valid training samples")
        return

    # 检查DTW特征是否被加载
    dtw_cols = [col for col in X_train_valid.columns if col.startswith("dtw_")]
    if dtw_cols:
        print(f"   ✅ DTW features loaded: {len(dtw_cols)} features")
        print(f"      Examples: {dtw_cols[:5]}")
    else:
        print(f"   ⚠️  No DTW features found in training data")

    # 检查其他关键特征
    garch_cols = [col for col in X_train_valid.columns if col.startswith("garch_")]
    print(
        f"   📊 Feature summary: GARCH={len(garch_cols)}, DTW={len(dtw_cols)}, Total={len(X_train_valid.columns)}"
    )
    print(
        f"      Note: EVT features excluded from volatility model (used for risk management)"
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

    # Ensure tick_loader_json is configured for volatility model feature computation
    if tick_loader_json and feature_loader:
        # Update VPIN feature config in feature_deps
        features_dict = feature_loader.feature_deps.get("features", {})
        vpin_feature = features_dict.get("vpin_features")
        if vpin_feature is not None:
            # Ensure compute_params exists
            if "compute_params" not in vpin_feature:
                vpin_feature["compute_params"] = {}
            vpin_feature["compute_params"]["ticks_loader_json"] = tick_loader_json
            print(f"   ✅ VPIN ticks_loader_json configured for volatility model")
            # Also update in the computer's feature_deps if it exists
            if hasattr(feature_loader, "computer") and hasattr(
                feature_loader.computer, "feature_deps"
            ):
                computer_features = feature_loader.computer.feature_deps.get(
                    "features", {}
                )
                computer_vpin = computer_features.get("vpin_features")
                if computer_vpin is not None:
                    if "compute_params" not in computer_vpin:
                        computer_vpin["compute_params"] = {}
                    computer_vpin["compute_params"][
                        "ticks_loader_json"
                    ] = tick_loader_json

    vol_model, vol_metrics = train_volatility_model(
        X_train_valid,
        y_vol_train_valid,
        X_train_valid,
        y_vol_train_valid,
        feature_loader=feature_loader,  # 传入feature_loader以计算缺失特征
    )

    # Evaluate ML model
    print("\n" + "=" * 60)
    print("5️⃣ Evaluating ML Model")
    print("=" * 60)
    ml_results = evaluate_ml_model(
        df_test,
        atr_test,
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
        atr_test,
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
