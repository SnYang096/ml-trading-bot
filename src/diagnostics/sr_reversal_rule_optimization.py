"""
规则类参数调优模块：找到规则的参数的平坦高原，对比机器学习模型的效果

功能：
1. 网格搜索/随机搜索规则类策略参数
2. 计算各种指标（胜率、总R、Sharpe等）
3. 识别参数平坦高原（plateau）
4. 对比ML模型效果
5. 输出HTML报告
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from itertools import product
import warnings

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available, falling back to grid/random search")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategies.labels.sr_reversal_label import (  # noqa: E402
    SRSignalConfig,
    _generate_sr_reversal_signals,
)
from src.time_series_model.pipeline.training.label_utils import (  # noqa: E402
    compute_rr_label,
)
from src.time_series_model.strategies.labels.sr_reversal_label import (  # noqa: E402
    _ensure_atr,
)
from src.strategy_config import StrategyConfigLoader  # noqa: E402

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SR Reversal Rule-Based Parameter Optimization"
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
        "--output-dir",
        type=str,
        default="results/rule_optimization",
        help="Output directory for results",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        default="optuna" if OPTUNA_AVAILABLE else "random",
        choices=["grid", "random", "optuna"],
        help="Search type: grid, random, or optuna (if available)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials (for random or optuna search)",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=None,
        help="Optuna timeout in seconds (None = no timeout)",
    )
    parser.add_argument(
        "--optuna-n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Optuna (1 = sequential)",
    )
    return parser.parse_args()


def define_parameter_grid() -> Dict[str, List[Any]]:
    """定义参数网格"""
    return {
        # SR信号生成参数
        "sr_strength_min": [0.3, 0.4, 0.5, 0.6, 0.7],
        "sqs_min": [0.3, 0.4, 0.5, 0.6, 0.7],
        "touch_distance_atr": [0.5, 1.0, 1.5, 2.0],
        # R/R参数
        "stop_loss_r": [0.5, 0.75, 1.0, 1.25, 1.5],
        "take_profit_r": [1.5, 2.0, 2.5, 3.0, 3.5],
        "max_holding_bars": [24, 36, 48, 60, 72],
        # VPIN过滤参数
        "use_vpin_filter": [False, True],
        "min_vpin": [0.3, 0.4, 0.5],
        "max_vpin": [0.5, 0.6, 0.7],
    }


def sample_random_params(
    param_grid: Dict[str, List[Any]], n_trials: int
) -> List[Dict[str, Any]]:
    """随机采样参数"""
    import random

    param_combinations = []
    for _ in range(n_trials):
        params = {}
        for key, values in param_grid.items():
            if key == "use_vpin_filter":
                params[key] = random.choice(values)
            elif key in ["min_vpin", "max_vpin"]:
                if params.get("use_vpin_filter", False):
                    params[key] = random.choice(values)
                else:
                    params[key] = None
            else:
                params[key] = random.choice(values)
        param_combinations.append(params)
    return param_combinations


def evaluate_rule_strategy(
    df_features: pd.DataFrame,
    atr_series: pd.Series,
    params: Dict[str, Any],
) -> Dict[str, float]:
    """
    评估规则类策略

    Returns:
        包含各种指标的字典
    """
    # 配置SR信号生成
    # Note: SRSignalConfig uses different parameter names
    sqs_min = params.get("sqs_min", 0.5)
    sr_cfg = SRSignalConfig(
        min_sr_strength=params.get("sr_strength_min", 0.5),
        min_support_score=sqs_min,  # Use same value for support and resistance
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

    # 保本版本
    from src.time_series_model.pipeline.training.label_utils import (
        compute_rr_label_with_details,
    )  # noqa: E402

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
        use_breakeven_stop=True,
    )
    labels_breakeven = details_breakeven["label"]

    # 统计指标（标准版本）
    mask_valid = (auto_signals != 0) & labels_standard.notna()
    n_signals = int((auto_signals != 0).sum())
    n_trades = int(mask_valid.sum())

    if n_trades == 0:
        return {
            "n_signals": n_signals,
            "n_trades": 0,
            "win_rate": 0.0,
            "breakeven_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    df_trades = pd.DataFrame(
        {
            "signal": auto_signals[mask_valid],
            "label": labels_standard[mask_valid],
        }
    )

    n_win = int((df_trades["label"] == 1.0).sum())
    win_rate = n_win / n_trades if n_trades > 0 else 0.0

    # 计算保本率（保本版本）
    # 保本率 = 保本+胜利 / (保本+胜利 + 亏损)
    # 其中亏损包括：保本+亏损 和 直接亏损（loss）
    mask_valid_breakeven = (auto_signals != 0) & labels_breakeven.notna()
    n_trades_breakeven = int(mask_valid_breakeven.sum())
    if n_trades_breakeven > 0:
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

    # 计算Sharpe ratio（基于R序列，但需要正确年化）
    # 注意：R不是收益率，但我们可以用R的均值/标准差来近似风险调整后的表现
    # 更准确的做法是基于实际收益率序列，但这里我们简化处理
    # 年化因子：4H时间框架，每天6个bar，每年约252*6=1512个bar
    # 但交易频率不是每天，需要根据实际交易频率调整
    if len(realized_r) > 1:
        # 使用R的均值/标准差，但不乘以sqrt(252)，因为R不是收益率
        # 或者我们可以计算一个近似的Sharpe：假设每笔交易的平均R代表风险调整后的表现
        # 更保守的计算：使用R的变异系数（CV）的倒数
        r_mean = np.mean(realized_r)
        r_std = np.std(realized_r)
        if r_std > 1e-8:
            # 简化的Sharpe：R均值 / R标准差（不年化，因为R不是收益率）
            # 这个值表示每单位R风险的R收益
            sharpe_ratio = float(r_mean / r_std)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # 计算Profit Factor
    gross_profit = (
        float(realized_r[realized_r > 0].sum()) if (realized_r > 0).any() else 0.0
    )
    gross_loss = (
        float(abs(realized_r[realized_r < 0].sum())) if (realized_r < 0).any() else 1e-8
    )
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # 计算最大回撤（简化版：使用累计R）
    cumulative_r = np.cumsum(realized_r)
    running_max = np.maximum.accumulate(cumulative_r)
    drawdown = cumulative_r - running_max
    max_drawdown = float(abs(drawdown.min())) if len(drawdown) > 0 else 0.0

    return {
        "n_signals": n_signals,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "breakeven_rate": breakeven_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


def find_plateau_regions(
    results_df: pd.DataFrame,
    metric_col: str = "total_r",
    threshold_percentile: float = 0.8,
    min_neighbors: int = 5,
) -> pd.DataFrame:
    """
    找到参数的平坦高原区域

    Args:
        results_df: 包含参数和指标的结果DataFrame
        metric_col: 用于评估的指标列
        threshold_percentile: 阈值百分位数（例如0.8表示前20%）
        min_neighbors: 最小邻居数量（用于判断是否为高原）

    Returns:
        包含高原区域的DataFrame
    """
    # 计算阈值
    threshold = results_df[metric_col].quantile(threshold_percentile)

    # 找到高于阈值的点
    high_performance = results_df[results_df[metric_col] >= threshold].copy()

    # 对于每个参数，计算其在高性能区域的分布
    param_cols = [
        col
        for col in results_df.columns
        if col
        not in [
            "n_signals",
            "n_trades",
            "win_rate",
            "total_r",
            "avg_r",
            "sharpe_ratio",
            "profit_factor",
            "max_drawdown",
            metric_col,
        ]
    ]

    plateau_info = []
    for param in param_cols:
        if param in high_performance.columns:
            value_counts = high_performance[param].value_counts()
            if len(value_counts) > 0:
                most_common = value_counts.index[0]
                frequency = value_counts.iloc[0] / len(high_performance)
                plateau_info.append(
                    {
                        "parameter": param,
                        "most_common_value": most_common,
                        "frequency_in_high_performance": frequency,
                        "n_occurrences": value_counts.iloc[0],
                    }
                )

    return pd.DataFrame(plateau_info)


def generate_html_report(
    results_df: pd.DataFrame,
    plateau_df: pd.DataFrame,
    output_path: Path,
    ml_model_results: Optional[Dict[str, float]] = None,
) -> None:
    """生成HTML报告"""

    # 计算统计信息
    best_result = results_df.loc[results_df["total_r"].idxmax()]
    top_10_results = results_df.nlargest(10, "total_r")

    # 生成HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SR Reversal Rule-Based Parameter Optimization Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
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
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        .metric-box {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #f9f9f9;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
            min-width: 150px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
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
        .comparison-box {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            padding: 20px;
            background-color: #e8f5e9;
            border-radius: 8px;
        }}
        .comparison-item {{
            text-align: center;
        }}
        .comparison-label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        .comparison-value {{
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 SR Reversal Rule-Based Parameter Optimization Report</h1>
        
        <h2>🎯 Best Configuration</h2>
        <div class="metric-box">
            <div class="metric-label">Total R</div>
            <div class="metric-value">{best_result['total_r']:.2f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{best_result['win_rate']:.2%}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{best_result['sharpe_ratio']:.2f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Trades</div>
            <div class="metric-value">{int(best_result['n_trades'])}</div>
        </div>
        
        <h2>📈 Top 10 Configurations</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Total R</th>
                    <th>Win Rate</th>
                    <th>Sharpe</th>
                    <th>Trades</th>
                    <th>Parameters</th>
                </tr>
            </thead>
            <tbody>
"""

    for idx, (_, row) in enumerate(top_10_results.iterrows(), 1):
        params_str = ", ".join(
            [
                f"{k}={v}"
                for k, v in row.items()
                if k
                not in [
                    "n_signals",
                    "n_trades",
                    "win_rate",
                    "total_r",
                    "avg_r",
                    "sharpe_ratio",
                    "profit_factor",
                    "max_drawdown",
                ]
            ]
        )
        html_content += f"""
                <tr>
                    <td>{idx}</td>
                    <td class="{'positive' if row['total_r'] > 0 else 'negative'}">{row['total_r']:.2f}</td>
                    <td>{row['win_rate']:.2%}</td>
                    <td>{row['sharpe_ratio']:.2f}</td>
                    <td>{int(row['n_trades'])}</td>
                    <td style="font-size: 11px;">{params_str[:100]}...</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
        
        <h2>🏔️ Parameter Plateau Analysis</h2>
        <p>Parameters that appear frequently in high-performance configurations (top 20% by Total R):</p>
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Most Common Value</th>
                    <th>Frequency</th>
                    <th>Occurrences</th>
                </tr>
            </thead>
            <tbody>
"""

    for _, row in plateau_df.iterrows():
        html_content += f"""
                <tr>
                    <td><strong>{row['parameter']}</strong></td>
                    <td>{row['most_common_value']}</td>
                    <td>{row['frequency_in_high_performance']:.2%}</td>
                    <td>{int(row['n_occurrences'])}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
"""

    if ml_model_results:
        html_content += f"""
        <h2>🤖 Rule-Based vs Machine Learning Model Comparison</h2>
        <div class="comparison-box">
            <div class="comparison-item">
                <div class="comparison-label">Rule-Based (Best)</div>
                <div class="comparison-value">Total R: {best_result['total_r']:.2f}</div>
                <div class="comparison-value">Win Rate: {best_result['win_rate']:.2%}</div>
                <div class="comparison-value">Sharpe: {best_result['sharpe_ratio']:.2f}</div>
            </div>
            <div class="comparison-item">
                <div class="comparison-label">ML Model</div>
                <div class="comparison-value">Total R: {ml_model_results.get('total_r', 0):.2f}</div>
                <div class="comparison-value">Win Rate: {ml_model_results.get('win_rate', 0):.2%}</div>
                <div class="comparison-value">Sharpe: {ml_model_results.get('sharpe_ratio', 0):.2f}</div>
            </div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"   ✅ HTML report saved to {output_path}")


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

    # Use the same feature loading approach as baseline script
    from scripts import train_strategy_pipeline as strategy_runner  # noqa: E402

    feature_loader = StrategyFeatureLoader()
    df_features = strategy_runner.run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )

    # Ensure ATR
    atr_series = _ensure_atr(df_features, "atr", "close", "high", "low", 14)

    # Define parameter grid
    param_grid = define_parameter_grid()

    # Optuna optimization
    if args.search_type == "optuna" and OPTUNA_AVAILABLE:
        print(f"   🎯 Using Optuna optimization ({args.n_trials} trials)...")

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function"""
            # Suggest parameters
            params = {
                "sr_strength_min": trial.suggest_float(
                    "sr_strength_min", 0.3, 0.7, step=0.1
                ),
                "sqs_min": trial.suggest_float("sqs_min", 0.3, 0.7, step=0.1),
                "touch_distance_atr": trial.suggest_float(
                    "touch_distance_atr", 0.5, 2.0, step=0.5
                ),
                "stop_loss_r": trial.suggest_float("stop_loss_r", 0.5, 1.5, step=0.25),
                "take_profit_r": trial.suggest_float(
                    "take_profit_r", 1.5, 3.5, step=0.5
                ),
                "max_holding_bars": trial.suggest_int(
                    "max_holding_bars", 24, 72, step=12
                ),
                "use_vpin_filter": trial.suggest_categorical(
                    "use_vpin_filter", [False, True]
                ),
            }

            # VPIN parameters (conditional)
            if params["use_vpin_filter"]:
                params["min_vpin"] = trial.suggest_float("min_vpin", 0.3, 0.5, step=0.1)
                params["max_vpin"] = trial.suggest_float("max_vpin", 0.5, 0.7, step=0.1)
                if params["min_vpin"] >= params["max_vpin"]:
                    params["max_vpin"] = params["min_vpin"] + 0.1
            else:
                params["min_vpin"] = None
                params["max_vpin"] = None

            # Evaluate strategy
            try:
                metrics = evaluate_rule_strategy(df_features, atr_series, params)

                # Use total_r as the objective (can be weighted with other metrics)
                objective_value = metrics["total_r"]

                # Add penalty for too few trades
                if metrics["n_trades"] < 10:
                    objective_value -= 100.0

                # Report intermediate values for pruning
                trial.set_user_attr("win_rate", metrics["win_rate"])
                trial.set_user_attr("n_trades", metrics["n_trades"])
                trial.set_user_attr("sharpe_ratio", metrics["sharpe_ratio"])

                return objective_value
            except Exception as e:
                print(f"   ⚠️ Error in trial: {e}")
                return -1000.0  # Return a very low value for failed trials

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.optuna_timeout,
            n_jobs=args.optuna_n_jobs,
            show_progress_bar=True,
        )

        # Convert study results to DataFrame
        results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                params = trial.params.copy()
                params.update(
                    {
                        "win_rate": trial.user_attrs.get("win_rate", 0.0),
                        "n_trades": trial.user_attrs.get("n_trades", 0),
                        "sharpe_ratio": trial.user_attrs.get("sharpe_ratio", 0.0),
                        "total_r": trial.value,
                        "n_signals": 0,  # Not tracked in Optuna
                        "avg_r": 0.0,  # Not tracked in Optuna
                        "profit_factor": 0.0,  # Not tracked in Optuna
                        "max_drawdown": 0.0,  # Not tracked in Optuna
                    }
                )
                results.append(params)

        results_df = pd.DataFrame(results)

        # Save Optuna study
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        study_path = output_dir / "optuna_study.pkl"
        import pickle

        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        print(f"   💾 Saved Optuna study to {study_path}")

        # Print best trial
        print(f"\n   🏆 Best trial:")
        print(f"      Total R: {study.best_value:.2f}")
        print(f"      Params: {study.best_params}")

    # Grid or random search
    elif args.search_type == "grid":
        # 网格搜索：生成所有组合（但需要处理条件参数）
        param_combinations = []
        for sr_strength in param_grid["sr_strength_min"]:
            for sqs in param_grid["sqs_min"]:
                for touch_dist in param_grid["touch_distance_atr"]:
                    for sl_r in param_grid["stop_loss_r"]:
                        for tp_r in param_grid["take_profit_r"]:
                            for max_bars in param_grid["max_holding_bars"]:
                                for use_vpin in param_grid["use_vpin_filter"]:
                                    params = {
                                        "sr_strength_min": sr_strength,
                                        "sqs_min": sqs,
                                        "touch_distance_atr": touch_dist,
                                        "stop_loss_r": sl_r,
                                        "take_profit_r": tp_r,
                                        "max_holding_bars": max_bars,
                                        "use_vpin_filter": use_vpin,
                                    }
                                    if use_vpin:
                                        for min_vpin in param_grid["min_vpin"]:
                                            for max_vpin in param_grid["max_vpin"]:
                                                if min_vpin < max_vpin:
                                                    params["min_vpin"] = min_vpin
                                                    params["max_vpin"] = max_vpin
                                                    param_combinations.append(
                                                        params.copy()
                                                    )
                                    else:
                                        params["min_vpin"] = None
                                        params["max_vpin"] = None
                                        param_combinations.append(params)

        # 限制组合数量（避免过多）
        if len(param_combinations) > 1000:
            print(
                f"   ⚠️ Too many combinations ({len(param_combinations)}), using random sampling..."
            )
            param_combinations = sample_random_params(param_grid, args.n_trials)
    else:
        # 随机搜索
        param_combinations = sample_random_params(param_grid, args.n_trials)

        print(f"   🔍 Testing {len(param_combinations)} parameter combinations...")

        # Evaluate all combinations
        results = []
        for i, params in enumerate(param_combinations, 1):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(param_combinations)}")

            try:
                metrics = evaluate_rule_strategy(df_features, atr_series, params)
                results.append({**params, **metrics})
            except Exception as e:
                print(f"   ⚠️ Error with params {params}: {e}")
                continue

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

    # Find plateau regions
    plateau_df = find_plateau_regions(results_df, metric_col="total_r")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "optimization_results.csv", index=False)
    plateau_df.to_csv(output_dir / "plateau_analysis.csv", index=False)

    # Generate HTML report
    generate_html_report(
        results_df,
        plateau_df,
        output_path=output_dir / "optimization_report.html",
        ml_model_results=None,  # TODO: Load ML model results
    )

    print(f"\n✅ Optimization complete!")
    print(f"   Results saved to {output_dir}")
    print(f"   Best Total R: {results_df['total_r'].max():.2f}")
    print(f"   Best Win Rate: {results_df['win_rate'].max():.2%}")


if __name__ == "__main__":
    main()
