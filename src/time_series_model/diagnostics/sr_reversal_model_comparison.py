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
        description="Strategy Model Comparison: Compare different strategy configurations (labels, backtest, stop-loss/take-profit, features)"
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        required=True,
        help="Comma-separated list of strategy config directories (e.g., 'sr_reversal_long,sr_reversal_long_vol')",
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
        "--seed",
        type=int,
        default=42,
        help="Seed forwarded to train_strategy_pipeline for reproducible runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_comparison",
        help="Output directory for results",
    )
    # NOTE: We intentionally do not expose a pipeline flag here.
    # If multiple strategy configs are provided, we auto-run the main train/backtest pipeline and emit:
    # - strategy_pipeline_metrics.csv
    # - comparison_report.html (embedding the table)
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
    parser.add_argument(
        "--rule-based-entry",
        type=str,
        default=None,
        help="Python module path for rule-based strategy entry point (e.g., 'src.time_series_model.strategies.rules.sr_reversal_rule.evaluate_rule_based'). If not provided, uses default SR reversal rule.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for model outputs. Used to auto-detect volatility model if --volatility-model-path not provided.",
    )
    parser.add_argument(
        "--test-low-threshold",
        action="store_true",
        help="Test with low threshold (0.25) to verify breakeven stop effect",
    )
    return parser.parse_args()


def run_train_pipeline_multi_strategy(args: argparse.Namespace) -> None:
    """
    Run scripts/train_strategy_pipeline.py for each strategy config and export a unified metrics table.

    Output columns match the table we use in discussions:
    strategy | task | train | CV | corr | return% | Sharpe | DD% | trades
    """
    import subprocess
    import os
    import json

    strategies = [
        s.strip() for s in (args.strategy_config or "").split(",") if s.strip()
    ]
    if not strategies:
        raise ValueError("No strategies provided in --strategy-config")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_root = out_dir / "train_pipeline_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if args.start_date:
        env["TRAIN_START_DATE"] = args.start_date
    if args.end_date:
        env["TRAIN_END_DATE"] = args.end_date

    rows: list[dict[str, Any]] = []
    for name in strategies:
        cfg_dir = Path("config/strategies") / name
        cmd = [
            sys.executable,
            "scripts/train_strategy_pipeline.py",
            "--config",
            str(cfg_dir),
            "--symbol",
            args.symbol,
            "--timeframe",
            args.timeframe,
            "--test-size",
            str(args.test_size),
            "--seed",
            str(int(getattr(args, "seed", 42))),
            "--deterministic",
            "--output-root",
            str(out_root),
            "--data-path",
            args.data_path,
        ]
        print(f"\n{'='*80}")
        print(f"🚀 Train pipeline: {name}")
        print(f"{'='*80}")
        subprocess.run(cmd, check=True, env=env)

        results_path = out_root / name / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing results.json for {name}: {results_path}")
        d = json.loads(results_path.read_text(encoding="utf-8"))
        bt = d.get("backtest", {}) or {}
        ev = d.get("evaluation", {}) or {}
        rows.append(
            {
                "strategy": name,
                "task": d.get("task_type"),
                "train": d.get("n_train_samples"),
                "CV": d.get("avg_cv_metric"),
                "corr": ev.get("test_correlation") or ev.get("pearson_correlation"),
                "return%": bt.get("total_return_pct"),
                "Sharpe": bt.get("sharpe"),
                "DD%": bt.get("max_drawdown_pct"),
                "trades": bt.get("total_trades"),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "strategy_pipeline_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved: {csv_path}")
    try:
        print(df.to_string(index=False))
    except Exception:
        pass

    # Write an HTML report that embeds the same table (so users can open comparison_report.html directly).
    html_path = out_dir / "comparison_report.html"
    try:

        def _fmt(x):
            if isinstance(x, (int, float, np.floating, np.integer)):
                if np.isnan(x):
                    return "NaN"
                return f"{float(x):.4f}"
            return str(x)

        # Pick a "best" strategy for this single run (heuristic):
        # - Prefer highest Sharpe
        # - Require some minimum trades so Sharpe isn't meaningless
        min_trades = 10
        best_reason = ""
        best_name = None
        try:
            df2 = df.copy()
            if "trades" in df2.columns:
                df2 = df2[df2["trades"].fillna(0) >= min_trades]
            if not df2.empty and "Sharpe" in df2.columns:
                best_row = df2.sort_values(
                    ["Sharpe", "return%"], ascending=[False, False]
                ).iloc[0]
                best_name = str(best_row.get("strategy", ""))
                best_reason = (
                    f"Chosen by highest Sharpe (min_trades={min_trades}). "
                    f"Sharpe={_fmt(best_row.get('Sharpe'))}, return%={_fmt(best_row.get('return%'))}, "
                    f"DD%={_fmt(best_row.get('DD%'))}, trades={best_row.get('trades')}."
                )
        except Exception:
            pass

        table_html = df.to_html(index=False, formatters={c: _fmt for c in df.columns})
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Strategy Comparison</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; padding: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f6f6f6; text-align: left; }}
    td {{ text-align: right; }}
    td:first-child {{ text-align: left; }}
    code {{ background: #f2f2f2; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h2>Strategy Comparison (train_strategy_pipeline)</h2>
  <p>
    symbol=<code>{args.symbol}</code>,
    timeframe=<code>{args.timeframe}</code>,
    start=<code>{args.start_date or ""}</code>,
    end=<code>{args.end_date or ""}</code>,
    test_size=<code>{args.test_size}</code>
  </p>
  <div style="background:#f8f9fa;border:1px solid #e0e0e0;padding:10px;border-radius:6px;margin:12px 0;">
    <div><b>How to read:</b> This is a <b>single-run</b> comparison. For robust conclusions, use multi-seed / multi-symbol sweep (see <code>docs/strategies/SR_REVERSAL_EXPERIMENT_PROTOCOL.md</code>).</div>
    <div style="margin-top:6px;"><b>Heuristic best (this run):</b> <code>{best_name or "N/A"}</code></div>
    <div><b>Why:</b> {best_reason or "Not enough trades or missing metrics to rank."}</div>
  </div>
  <p>CSV: <code>{csv_path.name}</code></p>
  {table_html}
</body>
</html>"""
        html_path.write_text(html, encoding="utf-8")
        print(f"✅ Saved: {html_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Failed to write comparison_report.html: {exc}")


def train_ml_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Any, Dict[str, float]]:
    """训练ML分类模型"""
    print("   🤖 Training ML classification model...")

    # 计算类别不平衡比例，用于设置 scale_pos_weight
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(
        f"   📊 Class distribution: positive={pos_count} ({pos_count/len(y_train):.2%}), "
        f"negative={neg_count} ({neg_count/len(y_train):.2%})"
    )
    print(f"   📊 scale_pos_weight: {scale_pos_weight:.2f}")

    try:
        # 使用改进的参数：添加 scale_pos_weight 处理类别不平衡
        # ✅ 固定随机种子以确保可重复性
        model = LightGBMTrainer(
            model_type="classification",
            use_gpu=True,
            params={
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_data_in_leaf": 20,
                "lambda_l1": 0.5,
                "lambda_l2": 1.0,
                "scale_pos_weight": scale_pos_weight,  # 处理类别不平衡
                "random_state": 42,  # ✅ 固定随机种子
                "bagging_seed": 42,  # ✅ 固定bagging随机种子
                "feature_fraction_seed": 42,  # ✅ 固定特征采样随机种子
                "data_random_seed": 42,  # ✅ 固定数据随机种子
                "verbose": -1,
            },
        )
        # ✅ 增加CV折数到10以提高稳定性（对于3102样本，10折更稳定）
        metrics, _ = model.train(
            X_train,
            y_train,
            n_splits=10,  # ✅ 从5增加到10
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
            "random_state": 42,  # ✅ 固定随机种子
            "bagging_seed": 42,  # ✅ 固定bagging随机种子
            "feature_fraction_seed": 42,  # ✅ 固定特征采样随机种子
            "data_random_seed": 42,  # ✅ 固定数据随机种子
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
    original_df_train: Optional[pd.DataFrame] = None,
    original_df_test: Optional[pd.DataFrame] = None,
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
                X_train,
                config,
                feature_loader=feature_loader,
                original_df=original_df_train,
            )
        )
        X_test_prepared, _, _ = prepare_volatility_model_data(
            X_test, config, feature_loader=feature_loader, original_df=original_df_test
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

        # ✅ 确保波动率模型也固定随机种子
        if "random_state" not in model.params:
            model.params["random_state"] = 42
        if "bagging_seed" not in model.params:
            model.params["bagging_seed"] = 42
        if "feature_fraction_seed" not in model.params:
            model.params["feature_fraction_seed"] = 42
        if "data_random_seed" not in model.params:
            model.params["data_random_seed"] = 42

        # ✅ 增加CV折数到10以提高稳定性
        effective_n_splits = max(10, n_splits) if n_splits < 10 else n_splits

        metrics, _ = model.train(
            X_train_vol,
            y_vol_train,
            n_splits=effective_n_splits,  # ✅ 使用10折或配置的值
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
                "random_state": 42,  # ✅ 固定随机种子
                "bagging_seed": 42,  # ✅ 固定bagging随机种子
                "feature_fraction_seed": 42,  # ✅ 固定特征采样随机种子
                "data_random_seed": 42,  # ✅ 固定数据随机种子
                "verbose": -1,
            }

        # ✅ 确保所有随机种子都已设置
        if "random_state" not in params:
            params["random_state"] = 42
        if "bagging_seed" not in params:
            params["bagging_seed"] = 42
        if "feature_fraction_seed" not in params:
            params["feature_fraction_seed"] = 42
        if "data_random_seed" not in params:
            params["data_random_seed"] = 42

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
                    # 只选择训练时使用的特征，按训练时的顺序
                    vol_features = [
                        f for f in self._volatility_features if f in X.columns
                    ]
                    if len(vol_features) != len(self._volatility_features):
                        missing = set(self._volatility_features) - set(vol_features)
                        print(
                            f"   ⚠️  Warning: Some volatility features missing in predict. Missing: {missing}"
                        )
                        # 用 0 填充缺失的特征
                        X_used = X[vol_features].copy()
                        for f in missing:
                            X_used[f] = 0.0
                        # 确保列的顺序与训练时一致
                        X_used = X_used[self._volatility_features]
                    else:
                        # 确保列的顺序与训练时一致
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

    # 调试：检查信号生成所需的列
    print(f"   🔍 Debug: Checking required columns for signal generation...")
    print(f"      sr_strength_max exists: {'sr_strength_max' in df_features.columns}")
    print(f"      sqs_hal_high exists: {'sqs_hal_high' in df_features.columns}")
    print(f"      sqs_hal_low exists: {'sqs_hal_low' in df_features.columns}")
    print(f"      poc exists: {'poc' in df_features.columns}")
    print(f"      hal_high exists: {'hal_high' in df_features.columns}")
    print(f"      hal_low exists: {'hal_low' in df_features.columns}")

    if "sr_strength_max" in df_features.columns:
        sr_stats = df_features["sr_strength_max"].describe()
        print(
            f"      sr_strength_max stats: mean={sr_stats['mean']:.3f}, max={sr_stats['max']:.3f}, min={sr_stats['min']:.3f}, non-null={df_features['sr_strength_max'].notna().sum()}/{len(df_features)}"
        )
        print(
            f"      sr_strength_max >= {sr_cfg.min_sr_strength}: {(df_features['sr_strength_max'] >= sr_cfg.min_sr_strength).sum()} samples"
        )
    if "sqs_hal_high" in df_features.columns:
        sqs_high_stats = df_features["sqs_hal_high"].describe()
        print(
            f"      sqs_hal_high stats: mean={sqs_high_stats['mean']:.3f}, max={sqs_high_stats['max']:.3f}, min={sqs_high_stats['min']:.3f}, non-null={df_features['sqs_hal_high'].notna().sum()}/{len(df_features)}"
        )
        print(
            f"      sqs_hal_high >= {sr_cfg.min_resistance_score}: {(df_features['sqs_hal_high'] >= sr_cfg.min_resistance_score).sum()} samples"
        )
    if "sqs_hal_low" in df_features.columns:
        sqs_low_stats = df_features["sqs_hal_low"].describe()
        print(
            f"      sqs_hal_low stats: mean={sqs_low_stats['mean']:.3f}, max={sqs_low_stats['max']:.3f}, min={sqs_low_stats['min']:.3f}, non-null={df_features['sqs_hal_low'].notna().sum()}/{len(df_features)}"
        )
        print(
            f"      sqs_hal_low >= {sr_cfg.min_support_score}: {(df_features['sqs_hal_low'] >= sr_cfg.min_support_score).sum()} samples"
        )

    print(f"   🔍 Signal generation config:")
    print(f"      min_sr_strength: {sr_cfg.min_sr_strength}")
    print(f"      min_support_score: {sr_cfg.min_support_score}")
    print(f"      min_resistance_score: {sr_cfg.min_resistance_score}")
    print(f"      tolerance_mult: {sr_cfg.tolerance_mult}")

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

    n_signals = int((auto_signals != 0).sum())
    print(f"   📊 Generated {n_signals} signals (out of {len(df_features)} samples)")
    if n_signals > 0:
        signal_indices = auto_signals[auto_signals != 0].index[:5]
        print(f"      First 5 signal timestamps: {signal_indices.tolist()}")
        print(
            f"      Signal values: {auto_signals[auto_signals != 0].head(5).tolist()}"
        )

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
        # ✅ 添加调试信息
        if "final_result" in details_valid.columns:
            final_result_counts = details_valid["final_result"].value_counts()
            print(f"   🔍 Debug: final_result distribution:")
            for result, count in final_result_counts.items():
                print(f"      {result}: {count}")
        else:
            print(f"   ⚠️  'final_result' column not found in details_breakeven")
            print(f"   🔍 Available columns: {details_valid.columns.tolist()}")

        n_breakeven_win = int((details_valid["final_result"] == "breakeven_win").sum())
        n_breakeven_loss = int(
            (details_valid["final_result"] == "breakeven_loss").sum()
        )
        n_loss = int((details_valid["final_result"] == "loss").sum())
        n_win = int((details_valid["final_result"] == "win").sum())
        n_loss_total = n_breakeven_loss + n_loss

        print(f"   🔍 Debug: Breakeven breakdown:")
        print(f"      breakeven_win: {n_breakeven_win}")
        print(f"      breakeven_loss: {n_breakeven_loss}")
        print(f"      loss: {n_loss}")
        print(f"      win: {n_win}")
        print(f"      total_loss: {n_loss_total}")

        breakeven_rate = (
            n_breakeven_win / (n_breakeven_win + n_loss_total)
            if (n_breakeven_win + n_loss_total) > 0
            else 0.0
        )
    else:
        breakeven_rate = 0.0
        print(
            f"   ⚠️  No valid breakeven samples (mask_valid_breakeven.sum()={mask_valid_breakeven.sum()})"
        )

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

    # 调试：检查预测分布
    print(f"   🔍 Debug: ML model predictions stats:")
    print(f"      Shape: {preds_proba.shape}")
    print(f"      Mean: {np.mean(preds_proba):.4f}")
    print(f"      Median: {np.median(preds_proba):.4f}")
    print(f"      Min: {np.min(preds_proba):.4f}")
    print(f"      Max: {np.max(preds_proba):.4f}")
    print(f"      Std: {np.std(preds_proba):.4f}")
    print(f"      >= {threshold}: {(preds_proba >= threshold).sum()} samples")
    print(f"      >= 0.3: {(preds_proba >= 0.3).sum()} samples")
    print(f"      >= 0.2: {(preds_proba >= 0.2).sum()} samples")

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

    # 调试：检查ML信号生成
    n_auto_signals = int((auto_signals != 0).sum())
    n_ml_signals = int((ml_signals != 0).sum())
    print(f"   🔍 Debug: Signal filtering:")
    print(f"      Rule-based signals: {n_auto_signals}")
    print(f"      After ML filter (threshold={threshold}): {n_ml_signals}")
    if n_auto_signals > 0 and n_ml_signals == 0:
        print(f"      ⚠️  All rule-based signals filtered out by ML threshold!")
        print(f"      Max prediction: {np.max(preds_proba[auto_signals != 0]):.4f}")
        print(
            f"      Mean prediction (on signals): {np.mean(preds_proba[auto_signals != 0]):.4f}"
        )

    # 计算RR标签（启用保本止损）
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
        use_breakeven_stop=True,  # 启用保本止损
    )

    # 使用保本止损计算保本率（需要详细信息）
    from src.time_series_model.pipeline.training.label_utils import (
        compute_rr_label_with_details,
    )

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

    # 计算保本率
    mask_valid_breakeven = (ml_signals != 0) & details_breakeven["label"].notna()
    breakeven_rate = 0.0
    if mask_valid_breakeven.sum() > 0:
        details_valid = details_breakeven[mask_valid_breakeven]
        # ✅ 添加调试信息
        if "final_result" in details_valid.columns:
            final_result_counts = details_valid["final_result"].value_counts()
            print(f"   🔍 Debug: final_result distribution (ML model):")
            for result, count in final_result_counts.items():
                print(f"      {result}: {count}")

        n_breakeven_win = int((details_valid["final_result"] == "breakeven_win").sum())
        n_breakeven_loss = int(
            (details_valid["final_result"] == "breakeven_loss").sum()
        )
        n_loss = int((details_valid["final_result"] == "loss").sum())
        n_win = int((details_valid["final_result"] == "win").sum())
        n_loss_total = n_breakeven_loss + n_loss

        print(f"   🔍 Debug: Breakeven breakdown (ML model):")
        print(f"      breakeven_win: {n_breakeven_win}")
        print(f"      breakeven_loss: {n_breakeven_loss}")
        print(f"      loss: {n_loss}")
        print(f"      win: {n_win}")
        print(f"      total_loss: {n_loss_total}")

        breakeven_rate = (
            n_breakeven_win / (n_breakeven_win + n_loss_total)
            if (n_breakeven_win + n_loss_total) > 0
            else 0.0
        )
    else:
        print(
            f"   ⚠️  No valid breakeven samples for ML model (mask_valid_breakeven.sum()={mask_valid_breakeven.sum()})"
        )

    n_win = int((df_trades["label"] == 1.0).sum())
    win_rate = n_win / n_trades if n_trades > 0 else 0.0

    # 计算R（考虑保本止损）
    stop_loss_r = params.get("stop_loss_r", 1.0)
    take_profit_r = params.get("take_profit_r", 2.0)

    # 对于保本止损的交易，需要从details中获取实际R值
    if mask_valid_breakeven.sum() > 0:
        details_valid = details_breakeven[mask_valid_breakeven]
        # 使用details中的realized_rr（如果可用）
        if "realized_rr" in details_valid.columns:
            realized_r = details_valid["realized_rr"].values
        else:
            # 回退到简单计算
            realized_r = np.where(
                df_trades["label"].values == 1.0,
                take_profit_r,
                -stop_loss_r,
            )
    else:
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
        "breakeven_rate": breakeven_rate,  # 启用保本止损后计算保本率
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
    # 确保只使用训练时的特征
    if hasattr(vol_model, "_volatility_features") and vol_model._volatility_features:
        # 只使用训练时的特征
        vol_features = [f for f in vol_model._volatility_features if f in X.columns]
        if len(vol_features) != len(vol_model._volatility_features):
            print(
                f"   ⚠️  Warning: Some volatility features missing. Expected {len(vol_model._volatility_features)}, found {len(vol_features)}"
            )
            print(
                f"      Missing: {set(vol_model._volatility_features) - set(vol_features)}"
            )
        X_vol = X[vol_features] if vol_features else X
    else:
        X_vol = X

    pred_vol_relative = vol_model.predict(X_vol)
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

    # 检查ATR的分布（确保索引对齐）
    atr_aligned = atr_series.reindex(df_features.index, fill_value=np.nan)
    atr_values = atr_aligned.values
    # 过滤 NaN 值进行统计
    atr_valid = atr_values[~np.isnan(atr_values)]
    print(f"   📊 ATR stats:")
    if len(atr_valid) > 0:
        print(f"      Mean: {np.mean(atr_valid):.2f}")
        print(f"      Std: {np.std(atr_valid):.2f}")
        print(f"      Min: {np.min(atr_valid):.2f}")
        print(f"      Max: {np.max(atr_valid):.2f}")
        print(f"      Median: {np.median(atr_valid):.2f}")
    else:
        print(f"      ⚠️  All ATR values are NaN!")
        # 尝试从 df_features 中获取 ATR
        if "atr" in df_features.columns:
            atr_values = df_features["atr"].values
            atr_valid = atr_values[~np.isnan(atr_values)]
            if len(atr_valid) > 0:
                print(f"      Using ATR from df_features:")
                print(f"      Mean: {np.mean(atr_valid):.2f}")
                print(f"      Std: {np.std(atr_valid):.2f}")
                print(f"      Min: {np.min(atr_valid):.2f}")
                print(f"      Max: {np.max(atr_valid):.2f}")
                print(f"      Median: {np.median(atr_valid):.2f}")
                atr_values = df_features["atr"].values
            else:
                print(f"      ⚠️  ATR column in df_features also has all NaN values!")

    # 检查预测波动率（绝对）与ATR的比率（确保使用对齐后的 ATR）
    if "atr" in df_features.columns:
        atr_for_ratio = df_features["atr"].values
    else:
        atr_for_ratio = atr_values
    vol_atr_ratio = pred_vol / (atr_for_ratio + 1e-8)
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


def run_single_strategy_comparison(
    strategy_config_path: str,
    df_raw_train: pd.DataFrame,
    df_raw_test: pd.DataFrame,
    feature_loader: StrategyFeatureLoader,
    tick_loader_json: Optional[str],
    args: argparse.Namespace,
    rule_based_entry: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行单个策略配置的完整比较流程

    Returns:
        Dict containing:
        - strategy_name: 策略名称
        - rule_results: 规则类策略结果
        - ml_results: ML模型结果
        - ml_vol_results: ML+波动率模型结果（如果启用）
        - config_info: 配置信息（标签、回测、止损止盈、特征等）
    """
    strategy_name = Path(strategy_config_path).name
    print(f"\n{'='*80}")
    print(f"📊 Processing Strategy: {strategy_name}")
    print(f"{'='*80}")

    # Load strategy config
    cfg_dir = Path(strategy_config_path).resolve()
    if not cfg_dir.exists():
        raise FileNotFoundError(f"Strategy config not found: {cfg_dir}")

    strategy_cfg_loader = StrategyConfigLoader(cfg_dir)
    strategy_cfg = strategy_cfg_loader.load()

    # Load features for this strategy
    print(f"   🔧 Loading features for {strategy_name}...")
    df_train = strategy_runner.run_feature_pipeline(
        df_raw_train,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )
    df_test = strategy_runner.run_feature_pipeline(
        df_raw_test,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=False,
    )

    # Ensure ATR exists
    atr_train = _ensure_atr(
        df_train,
        atr_col="atr",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_window=14,
    )
    atr_test = _ensure_atr(
        df_test,
        atr_col="atr",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_window=14,
    )
    if "atr" not in df_train.columns:
        df_train["atr"] = atr_train
    if "atr" not in df_test.columns:
        df_test["atr"] = atr_test

    # Extract config info
    config_info = {
        "strategy_name": strategy_name,
        "label_config": {
            "target_column": strategy_cfg.labels.target_column,
            "generator": f"{strategy_cfg.labels.generator.module}.{strategy_cfg.labels.generator.function}",
            "params": dict(strategy_cfg.labels.generator.params or {}),
        },
        "backtest_config": {
            "params": dict(strategy_cfg.backtest.params or {}),
        },
        "model_config": {
            "task_type": strategy_cfg.model.trainer.params.get("task_type", "binary"),
            "volatility_model_enabled": (
                strategy_cfg.model.volatility_model.enabled
                if strategy_cfg.model.volatility_model
                else False
            ),
        },
        "features": {
            "count": (
                len(strategy_cfg.features.requested_features)
                if strategy_cfg.features.requested_features
                else 0
            ),
            "list": (
                strategy_cfg.features.requested_features[:10]
                if strategy_cfg.features.requested_features
                else []
            ),  # First 10
        },
    }

    # Evaluate rule-based strategy (if entry point provided)
    rule_results = None
    if rule_based_entry:
        print(f"\n   📋 Evaluating Rule-Based Strategy (using {rule_based_entry})...")
        try:
            from scripts.train_strategy_pipeline import import_callable

            rule_func = import_callable(*rule_based_entry.rsplit(".", 1))
            # Extract rule params from backtest config
            rule_params = dict(strategy_cfg.backtest.params.get("rr", {}))
            rule_params.update(
                {
                    "stop_loss_r": rule_params.get("stop_loss_r", 1.0),
                    "take_profit_r": rule_params.get("take_profit_r", 2.0),
                    "max_holding_bars": rule_params.get("max_holding_bars", 50),
                }
            )
            rule_results = rule_func(df_test.copy(), atr_test, rule_params)
        except Exception as e:
            print(f"   ⚠️  Rule-based evaluation failed: {e}")
            rule_results = {
                "n_trades": 0,
                "win_rate": 0.0,
                "breakeven_rate": 0.0,
                "total_r": 0.0,
                "sharpe_ratio": 0.0,
            }

    # Train ML model
    print(f"\n   🤖 Training ML Model...")
    # Generate labels using strategy config
    from scripts.train_strategy_pipeline import import_callable

    label_func = import_callable(
        strategy_cfg.labels.generator.module,
        strategy_cfg.labels.generator.function,
    )
    label_params = dict(strategy_cfg.labels.generator.params or {})
    label_params.pop("signal_col", None)  # Remove signal_col if present

    train_labels_raw = label_func(df_train.copy(), **label_params)
    test_labels_raw = label_func(df_test.copy(), **label_params)

    # Handle label return type (Series or DataFrame)
    target_col = strategy_cfg.labels.target_column
    if isinstance(train_labels_raw, pd.DataFrame):
        train_labels = (
            train_labels_raw[target_col]
            if target_col in train_labels_raw.columns
            else train_labels_raw.iloc[:, 0]
        )
    else:
        train_labels = train_labels_raw

    if isinstance(test_labels_raw, pd.DataFrame):
        test_labels = (
            test_labels_raw[target_col]
            if target_col in test_labels_raw.columns
            else test_labels_raw.iloc[:, 0]
        )
    else:
        test_labels = test_labels_raw

    # Prepare features
    feature_cols = strategy_runner.determine_feature_columns(
        df_train, strategy_cfg.features
    )
    X_train = df_train[feature_cols].fillna(0)
    X_test = df_test[feature_cols].fillna(0)

    task_type = strategy_cfg.model.trainer.params.get("task_type", "binary")
    if task_type == "binary":
        y_train = train_labels.fillna(0).astype(int)
        y_test = test_labels.fillna(0).astype(int)
    else:
        y_train = train_labels.fillna(
            train_labels.median() if train_labels.notna().any() else 0.0
        )
        y_test = test_labels.fillna(
            test_labels.median() if test_labels.notna().any() else 0.0
        )

    # Train model
    ml_model, ml_metrics = train_ml_model(X_train, y_train, X_test, y_test)

    # Evaluate ML model
    print(f"\n   📊 Evaluating ML Model...")
    # Use strategy config for backtesting
    preds = (
        ml_model.predict_proba(X_test)[:, 1]
        if hasattr(ml_model, "predict_proba")
        else ml_model.predict(X_test)
    )
    ml_results = strategy_runner.run_backtest_with_strategy(
        df_test,
        preds,
        strategy_cfg,
        task_type=strategy_cfg.model.trainer.params.get("task_type", "binary"),
        vol_model=None,  # ML model only, no volatility model
    ) or {
        "n_trades": 0,
        "win_rate": 0.0,
        "breakeven_rate": 0.0,
        "total_r": 0.0,
        "sharpe_ratio": 0.0,
    }

    # Train and evaluate volatility model (if enabled)
    ml_vol_results = None
    if (
        strategy_cfg.model.volatility_model
        and strategy_cfg.model.volatility_model.enabled
    ):
        print(f"\n   📈 Training Volatility Model...")
        # Generate volatility labels
        target_col = strategy_cfg.model.volatility_model.target_column
        if target_col not in df_train.columns:
            from src.time_series_model.pipeline.training.volatility_model_config import (
                load_volatility_model_config,
            )

            vol_config = load_volatility_model_config(
                strategy_cfg.model.volatility_model.config_path
            )
            horizon = vol_config.get("prediction", {}).get("horizon", 10)
            df_train[target_col] = future_volatility_label(
                df_train["close"], horizon=horizon
            )
            df_test[target_col] = future_volatility_label(
                df_test["close"], horizon=horizon
            )

        y_vol_train = df_train[target_col]
        y_vol_test = df_test[target_col]

        vol_model, vol_metrics = train_volatility_model(
            X_train,
            y_vol_train,
            X_test,
            y_vol_test,
            config_path=strategy_cfg.model.volatility_model.config_path,
            feature_loader=feature_loader,
            original_df_train=df_train,
            original_df_test=df_test,
        )
        if vol_model:
            print(f"   📊 Evaluating ML + Volatility Model...")
            ml_vol_results = evaluate_ml_volatility_model(
                df_test,
                atr_test,
                ml_model,
                vol_model,
                dict(strategy_cfg.backtest.params.get("rr", {})),
                threshold=0.5,
            )

    return {
        "strategy_name": strategy_name,
        "config_info": config_info,
        "rule_results": rule_results,
        "ml_results": ml_results,
        "ml_vol_results": ml_vol_results,
        "ml_metrics": ml_metrics,
    }


def main() -> None:
    args = parse_args()

    # Auto behavior:
    # - Multiple strategy configs => run the main train/backtest pipeline and emit unified table + HTML.
    # - Single strategy config => run the legacy diagnostic (rule vs ML vs ML+vol).
    strategy_count = len(
        [s.strip() for s in (args.strategy_config or "").split(",") if s.strip()]
    )
    if strategy_count > 1:
        run_train_pipeline_multi_strategy(args)
        return

    # ✅ 固定所有随机种子以确保可重复性
    import numpy as np
    import random

    np.random.seed(42)
    random.seed(42)
    # 设置pandas的随机种子（如果使用）
    try:
        import pandas as pd

        # pandas 2.0+ 使用 numpy 的随机数生成器，所以已经通过 np.random.seed(42) 固定
    except ImportError:
        pass

    # Parse multiple strategy configs
    strategy_configs_raw = [
        s.strip() for s in args.strategy_config.split(",") if s.strip()
    ]
    if not strategy_configs_raw:
        raise ValueError("No strategy configs provided")

    # Resolve strategy config paths (support both relative and absolute paths)
    strategy_configs = []
    base_config_dir = Path("config/strategies")
    for cfg in strategy_configs_raw:
        # Try as relative path first (from config/strategies/)
        cfg_path = base_config_dir / cfg
        if not cfg_path.exists():
            # Try as absolute path
            cfg_path = Path(cfg)
            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"Strategy config not found: {cfg} (tried {base_config_dir / cfg} and {cfg_path})"
                )
        strategy_configs.append(str(cfg_path.resolve()))

    print(f"📋 Comparing {len(strategy_configs)} strategy configurations:")
    for i, cfg in enumerate(strategy_configs, 1):
        print(f"   {i}. {Path(cfg).name}")

    # Load data (shared across all strategies)
    print("\n📊 Loading data...")
    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # ✅ 确保数据按时间顺序排序（避免随机性）
    if not df_raw.empty:
        df_raw = df_raw.sort_index()
        print(f"   ✅ Data sorted by index (time order)")

    # Split data
    split_idx = int(len(df_raw) * (1 - args.test_size))
    train_end_idx = df_raw.index[split_idx - 1] if split_idx > 0 else df_raw.index[0]
    df_raw_train = df_raw.loc[df_raw.index <= train_end_idx].copy()
    df_raw_test = df_raw.loc[df_raw.index > train_end_idx].copy()
    print(f"   📊 Data split: train={len(df_raw_train)}, test={len(df_raw_test)}")

    # Configure tick loader (shared)
    if df_raw.empty:
        raise ValueError("No bars available for tick-loader configuration.")
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

    # Parse rule-based entry point
    rule_based_entry = args.rule_based_entry

    # Run comparison for each strategy
    all_results = []
    for strategy_config_path in strategy_configs:
        try:
            result = run_single_strategy_comparison(
                strategy_config_path,
                df_raw_train,
                df_raw_test,
                feature_loader,
                tick_loader_json,
                args,
                rule_based_entry=rule_based_entry,
            )
            all_results.append(result)
        except Exception as e:
            print(f"   ❌ Failed to process {strategy_config_path}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Generate comparison report
    print("\n" + "=" * 80)
    print("📊 Generating Comparison Report")
    print("=" * 80)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_multi_strategy_comparison_report(
        all_results, output_dir / "comparison_report.html"
    )

    # Save results to CSV
    save_comparison_results_csv(all_results, output_dir / "comparison_results.csv")

    print(f"\n✅ Comparison complete!")
    print(f"   Results saved to {output_dir}")


def generate_multi_strategy_comparison_report(
    all_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """生成多策略对比报告"""
    # TODO: Implement comprehensive comparison report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Strategy Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Strategy Comparison Report</h1>
    <p>Comparing {len(all_results)} strategy configurations</p>
    <h2>Results Summary</h2>
    <table>
        <tr>
            <th>Strategy</th>
            <th>ML Trades</th>
            <th>ML Win Rate</th>
            <th>ML Total R</th>
            <th>ML Sharpe</th>
            <th>ML+Vol Trades</th>
            <th>ML+Vol Win Rate</th>
            <th>ML+Vol Total R</th>
            <th>ML+Vol Sharpe</th>
        </tr>
"""
    for result in all_results:
        ml = result.get("ml_results", {}) or {}
        ml_vol = result.get(
            "ml_vol_results"
        )  # Can be None if volatility model not enabled
        ml_vol_dict = ml_vol if isinstance(ml_vol, dict) else {}
        html_content += f"""
        <tr>
            <td>{result.get('strategy_name', 'Unknown')}</td>
            <td>{ml.get('n_trades', 0)}</td>
            <td>{ml.get('win_rate', 0.0):.2%}</td>
            <td>{ml.get('total_r', 0.0):.2f}</td>
            <td>{ml.get('sharpe_ratio', 0.0):.2f}</td>
            <td>{ml_vol_dict.get('n_trades', 0) if ml_vol_dict else 0}</td>
            <td>{ml_vol_dict.get('win_rate', 0.0):.2% if ml_vol_dict else 'N/A'}</td>
            <td>{ml_vol_dict.get('total_r', 0.0):.2f if ml_vol_dict else 'N/A'}</td>
            <td>{ml_vol_dict.get('sharpe_ratio', 0.0):.2f if ml_vol_dict else 'N/A'}</td>
        </tr>
"""
    html_content += """
    </table>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   ✅ Comparison report saved to {output_path}")


def save_comparison_results_csv(
    all_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """保存对比结果到CSV（包含配置信息）"""
    rows = []
    for result in all_results:
        ml = result.get("ml_results", {}) or {}
        ml_vol_raw = result.get("ml_vol_results")
        ml_vol = ml_vol_raw if isinstance(ml_vol_raw, dict) else {}
        config_info = result.get("config_info", {})
        label_cfg = config_info.get("label_config", {})
        backtest_cfg = config_info.get("backtest_config", {})
        model_cfg = config_info.get("model_config", {})
        features_cfg = config_info.get("features", {})
        rr_params = backtest_cfg.get("params", {}).get("rr", {})

        rows.append(
            {
                "Strategy": result["strategy_name"],
                "Label_Generator": label_cfg.get("generator", "N/A"),
                "Label_Target": label_cfg.get("target_column", "N/A"),
                "Task_Type": model_cfg.get("task_type", "N/A"),
                "Volatility_Model": (
                    "Enabled"
                    if model_cfg.get("volatility_model_enabled")
                    else "Disabled"
                ),
                "Stop_Loss_R": rr_params.get("stop_loss_r", "N/A"),
                "Take_Profit_R": rr_params.get("take_profit_r", "N/A"),
                "Max_Holding_Bars": rr_params.get("max_holding_bars", "N/A"),
                "Breakeven_Stop": (
                    "Enabled" if rr_params.get("use_breakeven_stop") else "Disabled"
                ),
                "Feature_Count": features_cfg.get("count", 0),
                "ML_Trades": ml.get("n_trades", 0),
                "ML_Win_Rate": ml.get("win_rate", 0.0),
                "ML_Breakeven_Rate": ml.get("breakeven_rate", 0.0),
                "ML_Total_R": ml.get("total_r", 0.0),
                "ML_Sharpe": ml.get("sharpe_ratio", 0.0),
                "ML_Vol_Trades": ml_vol.get("n_trades", 0) if ml_vol else 0,
                "ML_Vol_Win_Rate": ml_vol.get("win_rate", 0.0) if ml_vol else 0.0,
                "ML_Vol_Breakeven_Rate": (
                    ml_vol.get("breakeven_rate", 0.0) if ml_vol else 0.0
                ),
                "ML_Vol_Total_R": ml_vol.get("total_r", 0.0) if ml_vol else 0.0,
                "ML_Vol_Sharpe": ml_vol.get("sharpe_ratio", 0.0) if ml_vol else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"   ✅ Results CSV saved to {output_path}")

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
    # Split raw data first using index-based split to preserve time order
    split_idx = int(len(df_raw) * (1 - args.test_size))

    # Use index-based split to ensure clean separation
    train_end_idx = df_raw.index[split_idx - 1] if split_idx > 0 else df_raw.index[0]
    test_start_idx = (
        df_raw.index[split_idx] if split_idx < len(df_raw) else df_raw.index[-1]
    )

    df_raw_train = df_raw.loc[df_raw.index <= train_end_idx].copy()
    df_raw_test = df_raw.loc[df_raw.index > train_end_idx].copy()

    # Ensure no overlap in raw data
    overlap_raw = set(df_raw_train.index) & set(df_raw_test.index)
    if overlap_raw:
        print(
            f"   ⚠️  Warning: {len(overlap_raw)} overlapping indices in raw data split, removing from test set"
        )
        df_raw_test = df_raw_test[~df_raw_test.index.isin(overlap_raw)]

    print(f"   📊 Raw data split: train={len(df_raw_train)}, test={len(df_raw_test)}")

    # Fit features on training set only
    print("   🔧 Fitting features on training set only (to avoid look-ahead bias)...")
    print(
        f"   🔍 Raw train indices: [{df_raw_train.index.min()} to {df_raw_train.index.max()}] ({len(df_raw_train)} samples)"
    )
    df_train = strategy_runner.run_feature_pipeline(
        df_raw_train,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,  # Fit on training set
    )
    print(
        f"   🔍 Train indices after features: [{df_train.index.min()} to {df_train.index.max()}] ({len(df_train)} samples)"
    )

    # Transform test set using fitted features (fit=False)
    # Note: tick_loader_json is already configured in feature_loader, so it will be used for test set too
    print("   🔧 Transforming test set using fitted features...")
    print(
        f"   🔍 Raw test indices: [{df_raw_test.index.min()} to {df_raw_test.index.max()}] ({len(df_raw_test)} samples)"
    )

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
    print(
        f"   🔍 Test indices after features: [{df_test.index.min()} to {df_test.index.max()}] ({len(df_test)} samples)"
    )

    # Debug: Check sizes after feature computation
    print(
        f"   📊 After feature computation: train={len(df_train)}, test={len(df_test)}"
    )

    # Filter out any indices that were not in the original raw data
    # This prevents feature computation from introducing overlapping indices
    train_new_indices = set(df_train.index) - set(df_raw_train.index)
    test_new_indices = set(df_test.index) - set(df_raw_test.index)

    if train_new_indices:
        print(
            f"   ⚠️  Warning: {len(train_new_indices)} new indices in train set after feature computation"
        )
        print(f"      Examples: {list(train_new_indices)[:5]}")
        print(f"   🔧 Filtering out new indices to preserve original data split...")
        df_train = df_train.loc[df_train.index.isin(df_raw_train.index)]
        print(f"   ✅ Train set after filtering: {len(df_train)} samples")

    if test_new_indices:
        print(
            f"   ⚠️  Warning: {len(test_new_indices)} new indices in test set after feature computation"
        )
        print(f"      Examples: {list(test_new_indices)[:5]}")
        print(f"   🔧 Filtering out new indices to preserve original data split...")
        df_test = df_test.loc[df_test.index.isin(df_raw_test.index)]
        print(f"   ✅ Test set after filtering: {len(df_test)} samples")

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

    # Check for overlapping indices between train and test BEFORE combining
    # This is critical - feature computation might have introduced overlap
    overlap = set(df_train.index) & set(df_test.index)
    if overlap:
        print(
            f"   ⚠️  Warning: {len(overlap)} overlapping indices between train and test after feature computation"
        )
        print(
            f"   🔍 Debug: Train index range: [{df_train.index.min()} to {df_train.index.max()}]"
        )
        print(
            f"   🔍 Debug: Test index range: [{df_test.index.min()} to {df_test.index.max()}]"
        )

        # Store original sizes before removal
        original_train_size = len(df_train)
        original_test_size = len(df_test)

        # Strategy: Always remove from test set to preserve training set integrity
        # This ensures training set remains intact for model training
        df_test = df_test[~df_test.index.isin(overlap)]
        print(
            f"   ℹ️  Removed {len(overlap)} overlapping indices from test set (train: {len(df_train)}, test: {len(df_test)})"
        )

        # Check if test set is too small after removal
        if len(df_test) == 0:
            raise ValueError(
                f"❌ Test set is empty after removing {len(overlap)} overlapping indices. "
                f"Original test size: {original_test_size}, overlap: {len(overlap)}. "
                f"This suggests feature computation introduced overlap. Please check feature pipeline."
            )
        if len(df_test) < 10:
            print(
                f"   ⚠️  Warning: Test set is very small ({len(df_test)} samples) after removing overlap"
            )

        # Check if training set is empty (should not happen if we remove from test)
        if len(df_train) == 0:
            raise ValueError(
                f"❌ Training set is empty. Original train size: {original_train_size}. "
                f"Please check data split logic or reduce test_size."
            )

    # Use index-based split instead of iloc to preserve original indices
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

    # Ensure ATR column exists in df_train and df_test for label computation
    if "atr" not in df_train.columns:
        df_train["atr"] = atr_train
    if "atr" not in df_test.columns:
        df_test["atr"] = atr_test

    # ✅ 从策略配置中读取参数（替代硬编码）
    # 优先使用优化后的参数（如果提供），否则使用配置中的默认值
    rule_params = {}

    # 从 backtest 配置中读取 RR 参数
    if strategy_cfg.backtest and strategy_cfg.backtest.params:
        backtest_params = strategy_cfg.backtest.params
        rr_params = backtest_params.get("rr", {})
        rule_params.update(
            {
                "stop_loss_r": float(rr_params.get("stop_loss_r", 1.0)),
                "take_profit_r": float(rr_params.get("take_profit_r", 2.0)),
                "max_holding_bars": int(rr_params.get("max_holding_bars", 50)),
            }
        )

    # 从 labels 配置中读取参数（如果 backtest 中没有）
    if strategy_cfg.labels and strategy_cfg.labels.generator.params:
        label_params = strategy_cfg.labels.generator.params
        if "stop_loss_r" not in rule_params:
            rule_params["stop_loss_r"] = float(label_params.get("stop_loss_r", 1.0))
        if "take_profit_r" not in rule_params:
            rule_params["take_profit_r"] = float(label_params.get("take_profit_r", 2.0))
        if "max_holding_bars" not in rule_params:
            rule_params["max_holding_bars"] = int(
                label_params.get("max_holding_bars", 50)
            )

    # 设置默认值（如果配置中没有）
    rule_params.setdefault("sr_strength_min", 0.3)
    rule_params.setdefault("sqs_min", 0.7)
    rule_params.setdefault("touch_distance_atr", 1.5)
    rule_params.setdefault("stop_loss_r", 1.0)
    rule_params.setdefault("take_profit_r", 2.0)
    rule_params.setdefault("max_holding_bars", 50)
    rule_params.setdefault("use_vpin_filter", False)

    # 如果提供了优化后的参数文件，优先使用（覆盖配置中的值）
    if args.rule_params and Path(args.rule_params).exists():
        # Try to load from CSV (optimization results)
        try:
            results_df = pd.read_csv(args.rule_params)
            if len(results_df) > 0:
                best_row = results_df.loc[results_df["total_r"].idxmax()]
                rule_params.update(
                    {
                        "sr_strength_min": float(
                            best_row.get(
                                "sr_strength_min",
                                rule_params.get("sr_strength_min", 0.3),
                            )
                        ),
                        "sqs_min": float(
                            best_row.get("sqs_min", rule_params.get("sqs_min", 0.7))
                        ),
                        "touch_distance_atr": float(
                            best_row.get(
                                "touch_distance_atr",
                                rule_params.get("touch_distance_atr", 1.5),
                            )
                        ),
                        "stop_loss_r": float(
                            best_row.get(
                                "stop_loss_r", rule_params.get("stop_loss_r", 1.25)
                            )
                        ),
                        "take_profit_r": float(
                            best_row.get(
                                "take_profit_r", rule_params.get("take_profit_r", 3.0)
                            )
                        ),
                        "max_holding_bars": int(
                            best_row.get(
                                "max_holding_bars",
                                rule_params.get("max_holding_bars", 72),
                            )
                        ),
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
                )
        except Exception as e:
            print(f"   ⚠️ Could not load rule params from {args.rule_params}: {e}")
            print("   Using parameters from strategy config...")

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

    # 调试：检查训练集的特征列
    print(f"   🔍 Debug: Checking required columns for training signal generation...")
    print(f"      sr_strength_max exists: {'sr_strength_max' in df_train.columns}")
    print(f"      sqs_hal_high exists: {'sqs_hal_high' in df_train.columns}")
    print(f"      sqs_hal_low exists: {'sqs_hal_low' in df_train.columns}")

    if "sr_strength_max" in df_train.columns:
        sr_stats = df_train["sr_strength_max"].describe()
        print(
            f"      sr_strength_max stats: mean={sr_stats['mean']:.3f}, max={sr_stats['max']:.3f}, min={sr_stats['min']:.3f}, non-null={df_train['sr_strength_max'].notna().sum()}/{len(df_train)}"
        )
        print(
            f"      sr_strength_max >= {sr_cfg.min_sr_strength}: {(df_train['sr_strength_max'] >= sr_cfg.min_sr_strength).sum()} samples"
        )
    if "sqs_hal_high" in df_train.columns:
        sqs_high_stats = df_train["sqs_hal_high"].describe()
        print(
            f"      sqs_hal_high stats: mean={sqs_high_stats['mean']:.3f}, max={sqs_high_stats['max']:.3f}, min={sqs_high_stats['min']:.3f}, non-null={df_train['sqs_hal_high'].notna().sum()}/{len(df_train)}"
        )
        print(
            f"      sqs_hal_high >= {sr_cfg.min_resistance_score}: {(df_train['sqs_hal_high'] >= sr_cfg.min_resistance_score).sum()} samples"
        )
    if "sqs_hal_low" in df_train.columns:
        sqs_low_stats = df_train["sqs_hal_low"].describe()
        print(
            f"      sqs_hal_low stats: mean={sqs_low_stats['mean']:.3f}, max={sqs_low_stats['max']:.3f}, min={sqs_low_stats['min']:.3f}, non-null={df_train['sqs_hal_low'].notna().sum()}/{len(df_train)}"
        )
        print(
            f"      sqs_hal_low >= {sr_cfg.min_support_score}: {(df_train['sqs_hal_low'] >= sr_cfg.min_support_score).sum()} samples"
        )

    print(f"   🔍 Signal generation config:")
    print(f"      min_sr_strength: {sr_cfg.min_sr_strength}")
    print(f"      min_support_score: {sr_cfg.min_support_score}")
    print(f"      min_resistance_score: {sr_cfg.min_resistance_score}")
    print(f"      tolerance_mult: {sr_cfg.tolerance_mult}")

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
    if n_signals > 0:
        signal_indices = train_signals[train_signals != 0].index[:5]
        print(f"      First 5 signal timestamps: {signal_indices.tolist()}")
        print(
            f"      Signal values: {train_signals[train_signals != 0].head(5).tolist()}"
        )

    # Compute labels
    # Debug: Check df_train before label computation
    if len(df_train) == 0:
        raise ValueError(
            "❌ Training set is empty! Cannot compute labels. Please check data split and feature computation."
        )

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

    # ✅ 使用配置中的标签生成函数（替代硬编码 compute_rr_label）
    from scripts.train_strategy_pipeline import import_callable

    label_func = import_callable(
        strategy_cfg.labels.generator.module,
        strategy_cfg.labels.generator.function,
    )
    target_col = strategy_cfg.labels.target_column

    # 生成标签（使用配置中的参数）
    label_params = dict(strategy_cfg.labels.generator.params or {})
    # 确保必要的参数存在（向后兼容）
    # compute_sr_reversal_label_full_scan 不需要 signal_col，它会自动生成信号
    # 移除可能存在的 signal_col 参数（如果配置中有）
    label_params.pop("signal_col", None)
    if "price_col" not in label_params:
        label_params["price_col"] = "close"
    if "atr_col" not in label_params:
        label_params["atr_col"] = "atr"

    try:
        train_labels = label_func(df_train.copy(), **label_params)
    except TypeError as e:
        # 如果参数不匹配，尝试移除不支持的参数
        print(f"   ⚠️  Label function parameter error: {e}")
        print(f"   🔧 Trying with minimal parameters...")
        # 只保留函数签名中明确支持的参数
        minimal_params = {
            "price_col": label_params.get("price_col", "close"),
            "atr_col": label_params.get("atr_col", "atr"),
        }
        # 添加其他可能需要的参数（如果函数支持）
        for key in [
            "max_holding_bars",
            "stop_loss_r",
            "take_profit_r",
            "combine_mode",
            "high_col",
            "low_col",
            "atr_window",
            "dist_to_sr_col",
            "dist_atr_mult",
            "sr_mask_col",
        ]:
            if key in label_params:
                minimal_params[key] = label_params[key]
        train_labels = label_func(df_train.copy(), **minimal_params)

    # compute_sr_reversal_label_full_scan 返回的是 Series，直接使用
    if not isinstance(train_labels, pd.Series):
        raise ValueError(
            f"Label function should return pd.Series, got {type(train_labels)}"
        )

    # 确保索引对齐
    train_labels = train_labels.reindex(df_train.index)

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
        print(
            f"   ℹ️  No DTW features in volatility model (expected: DTW used for SR Reversal strategy, not volatility prediction)"
        )

    # 检查其他关键特征
    garch_cols = [col for col in X_train_valid.columns if col.startswith("garch_")]
    print(
        f"   📊 Feature summary: GARCH={len(garch_cols)}, DTW={len(dtw_cols)}, Total={len(X_train_valid.columns)}"
    )
    print(f"      Note: EVT and DTW features excluded from volatility model")
    print(f"      - EVT: used for risk management/position sizing")
    print(f"      - DTW: used for SR Reversal strategy (pattern matching)")

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

    # Load or train volatility model
    print("\n" + "=" * 60)
    print("4️⃣ Volatility Model")
    print("=" * 60)

    vol_model = None
    vol_metrics = None

    # ✅ 检查是否应该训练波动率模型
    # 优先级：命令行参数 > 策略配置
    should_train_vol_model = False
    if args.enable_volatility_model is not None:
        # 命令行参数明确指定
        should_train_vol_model = args.enable_volatility_model
        print(
            f"   📋 Volatility model training: {should_train_vol_model} (from command line)"
        )
    elif strategy_cfg.model and strategy_cfg.model.volatility_model:
        # 使用策略配置
        should_train_vol_model = strategy_cfg.model.volatility_model.enabled
        print(
            f"   📋 Volatility model training: {should_train_vol_model} (from strategy config)"
        )
    else:
        # 默认不训练（向后兼容）
        should_train_vol_model = False
        print(f"   📋 Volatility model training: {should_train_vol_model} (default)")

    # Try to load pre-trained model
    vol_model_path = args.volatility_model_path
    if not vol_model_path and args.output_root:
        # Auto-detect: look for volatility_model.pkl in output_root/strategy_name/
        strategy_name = Path(args.strategy_config).name
        potential_path = Path(args.output_root) / strategy_name / "volatility_model.pkl"
        if potential_path.exists():
            vol_model_path = str(potential_path)
            print(f"   📂 Auto-detected volatility model: {vol_model_path}")

    if vol_model_path:
        print(f"   📂 Loading pre-trained volatility model from {vol_model_path}")
        try:
            import joblib

            vol_model = joblib.load(vol_model_path)
            print(f"   ✅ Volatility model loaded successfully")
            # Try to load metrics from results.json if available
            if args.output_root:
                strategy_name = Path(args.strategy_config).name
                results_file = Path(args.output_root) / strategy_name / "results.json"
                if results_file.exists():
                    try:
                        import json

                        with open(results_file, "r") as f:
                            results = json.load(f)
                            if (
                                "volatility_model" in results
                                and "metrics" in results["volatility_model"]
                            ):
                                vol_metrics = results["volatility_model"]["metrics"]
                                print(
                                    f"   ✅ Loaded volatility model metrics from results.json"
                                )
                    except Exception:
                        pass
        except Exception as e:
            print(f"   ⚠️  Failed to load volatility model: {e}")
            print(f"   🔄 Falling back to training...")
            vol_model_path = None

    # Train if not loaded and enabled
    if not vol_model and should_train_vol_model:
        print("   🔄 Training volatility model...")

        # ✅ 确定波动率模型配置路径
        vol_config_path = args.volatility_model_config
        if not vol_config_path:
            # 优先使用策略配置中的路径
            if (
                strategy_cfg.model
                and strategy_cfg.model.volatility_model
                and strategy_cfg.model.volatility_model.config_path
            ):
                vol_config_path = strategy_cfg.model.volatility_model.config_path
                print(
                    f"   📋 Using volatility model config from strategy: {vol_config_path}"
                )
            else:
                # 使用默认配置
                vol_config_path = None
                print(f"   📋 Using default volatility model config")
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

        # Pass original dataframes to ensure base columns are available for feature computation
        vol_model, vol_metrics = train_volatility_model(
            X_train_valid,
            y_vol_train_valid,
            X_train_valid,
            y_vol_train_valid,
            config_path=vol_config_path,  # ✅ 使用配置路径
            feature_loader=feature_loader,  # 传入feature_loader以计算缺失特征
            original_df_train=df_train,  # Pass original dataframe with base columns
            original_df_test=df_test,  # Pass original dataframe with base columns
        )
    elif not should_train_vol_model:
        print(
            "   ⏭️  Skipping volatility model training (disabled in config or command line)"
        )
        vol_model = None
        vol_metrics = None

    # Evaluate ML model
    print("\n" + "=" * 60)
    print("5️⃣ Evaluating ML Model")
    print("=" * 60)
    # 根据训练时的预测分布，动态调整阈值
    # 如果模型预测值都很低，使用更低的阈值（例如使用训练集预测的90分位数）
    suggested_threshold = 0.5  # 默认值
    if len(X_train_valid) > 0:
        try:
            # 尝试使用 predict_proba 方法
            if hasattr(ml_model, "predict_proba"):
                train_preds_sample = ml_model.predict_proba(
                    X_train_valid.head(min(500, len(X_train_valid)))
                )
            # 如果没有 predict_proba，尝试使用 predict 方法（LightGBMTrainer 的 predict 返回概率）
            elif hasattr(ml_model, "predict"):
                train_preds_sample = ml_model.predict(
                    X_train_valid.head(min(500, len(X_train_valid)))
                )
                # 如果是分类任务，predict 返回的是概率（单列），需要转换为 (n_samples, 2) 格式
                if (
                    ml_model.model_type == "classification"
                    and train_preds_sample.ndim == 1
                ):
                    train_preds_sample = np.column_stack(
                        [1 - train_preds_sample, train_preds_sample]
                    )
            else:
                print("   ⚠️  Model does not have predict_proba or predict method")
                train_preds_sample = None

            if train_preds_sample is not None:
                if len(train_preds_sample.shape) > 1:
                    train_preds_sample = train_preds_sample[:, 1]
                # 计算90分位数并设置阈值
                percentile_90 = np.percentile(train_preds_sample, 90)
                suggested_threshold = min(0.5, max(0.15, percentile_90))
                print(
                    f"   💡 Suggested threshold based on training predictions (90th percentile): {suggested_threshold:.3f}"
                )
                print(
                    f"      Training predictions stats: min={np.min(train_preds_sample):.4f}, "
                    f"max={np.max(train_preds_sample):.4f}, mean={np.mean(train_preds_sample):.4f}, "
                    f"median={np.median(train_preds_sample):.4f}"
                )
        except Exception as e:
            print(f"   ⚠️  Failed to compute dynamic threshold: {e}")
            suggested_threshold = 0.5

    # 如果动态阈值仍然太高（>0.3），使用更低的阈值进行测试
    if suggested_threshold > 0.3:
        # 使用训练集预测的中位数或25分位数作为阈值
        try:
            if len(X_train_valid) > 0:
                if hasattr(ml_model, "predict_proba"):
                    train_preds_sample = ml_model.predict_proba(
                        X_train_valid.head(min(500, len(X_train_valid)))
                    )
                elif hasattr(ml_model, "predict"):
                    train_preds_sample = ml_model.predict(
                        X_train_valid.head(min(500, len(X_train_valid)))
                    )
                    if (
                        ml_model.model_type == "classification"
                        and train_preds_sample.ndim == 1
                    ):
                        train_preds_sample = np.column_stack(
                            [1 - train_preds_sample, train_preds_sample]
                        )

                if train_preds_sample is not None:
                    if len(train_preds_sample.shape) > 1:
                        train_preds_sample = train_preds_sample[:, 1]
                    # 使用中位数或75分位数作为阈值（更保守）
                    alternative_threshold = min(
                        0.3, max(0.15, np.percentile(train_preds_sample, 75))
                    )
                    if alternative_threshold < suggested_threshold:
                        print(
                            f"   💡 Using alternative threshold (75th percentile): {alternative_threshold:.3f} "
                            f"(original: {suggested_threshold:.3f})"
                        )
                        suggested_threshold = alternative_threshold
        except Exception:
            pass

    # 如果动态阈值仍然太高，自动使用更低的阈值（0.25 或 0.3）
    # 这样可以确保即使模型预测值很低，也能产生一些交易来验证保本止损效果
    if suggested_threshold > 0.3:
        # 自动降低阈值到 0.25，确保有交易可以测试保本止损
        auto_low_threshold = 0.25
        print(
            f"   🧪 Auto-lowering threshold to {auto_low_threshold:.2f} "
            f"(original: {suggested_threshold:.3f}) to ensure trades for breakeven stop testing"
        )
        suggested_threshold = auto_low_threshold

    ml_results = evaluate_ml_model(
        df_test,
        atr_test,
        ml_model,
        rule_params,
        threshold=suggested_threshold,  # 使用动态阈值
    )
    print(f"   Trades: {int(ml_results['n_trades'])}")
    print(f"   Win Rate: {ml_results['win_rate']:.2%}")
    print(f"   Breakeven Rate: {ml_results.get('breakeven_rate', 0.0):.2%}")
    print(f"   Total R: {ml_results['total_r']:.2f}")
    print(f"   Sharpe: {ml_results['sharpe_ratio']:.2f}")

    # Evaluate ML + Volatility model
    print("\n" + "=" * 60)
    print("6️⃣ Evaluating ML + Volatility Model")
    print("=" * 60)
    if vol_model:
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
        print(f"   Breakeven Rate: {ml_vol_results.get('breakeven_rate', 0.0):.2%}")
        print(f"   Total R: {ml_vol_results['total_r']:.2f}")
        print(f"   Sharpe: {ml_vol_results['sharpe_ratio']:.2f}")
    else:
        print("   ⏭️  Skipped (volatility model not available)")
        ml_vol_results = {
            "n_trades": 0,
            "win_rate": 0.0,
            "breakeven_rate": 0.0,
            "total_r": 0.0,
            "sharpe_ratio": 0.0,
        }

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
