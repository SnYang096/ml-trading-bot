"""
ET 对冲有效性分析脚本

ET 作为条件式对冲的 KPI 验证：
1. left-tail reduction（最大单日/单事件回撤）
2. ET 激活次数是否集中在极端行情
3. ET 的长期成本是否 < 可接受的"保险费"

注意：ET 的 KPI 不看 Sharpe，只看尾部风险减少和成本。
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json


def compute_left_tail_metrics(
    returns: pd.Series,
    window: int = 1,
) -> Dict[str, float]:
    """
    计算左尾风险指标

    Args:
        returns: 收益率序列
        window: 滚动窗口大小（1=单日，5=5日滚动）

    Returns:
        Dict with keys:
        - max_drawdown: 最大回撤
        - max_daily_loss: 最大单日亏损
        - var_95: 95% VaR
        - cvar_95: 95% CVaR (Conditional VaR)
        - left_tail_ratio: 左尾比例（负收益占比）
    """
    if len(returns) == 0:
        return {
            "max_drawdown": 0.0,
            "max_daily_loss": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "left_tail_ratio": 0.0,
        }

    # 计算滚动窗口的累计收益
    if window > 1:
        rolling_returns = returns.rolling(window=window).sum()
    else:
        rolling_returns = returns

    # 最大回撤
    cumulative = (1 + rolling_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # 最大单日亏损
    max_daily_loss = float(rolling_returns.min())

    # VaR 和 CVaR
    var_95 = float(np.percentile(rolling_returns, 5))
    cvar_95 = float(rolling_returns[rolling_returns <= var_95].mean())

    # 左尾比例
    left_tail_ratio = float((rolling_returns < 0).sum() / len(rolling_returns))

    return {
        "max_drawdown": max_drawdown,
        "max_daily_loss": max_daily_loss,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "left_tail_ratio": left_tail_ratio,
    }


def analyze_et_activation_timing(
    df: pd.DataFrame,
    et_col: str = "gate_archetype",
    risk_cols: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    分析 ET 激活时机是否集中在极端行情

    Args:
        df: 包含 ET 信号和风险指标的数据
        et_col: ET archetype 列名
        risk_cols: 风险指标列名映射，例如 {"ofci_p": "ofci_pct", "shd_p": "shd_pct", "vol_spike_p": "atr_percentile"}

    Returns:
        Dict with activation statistics
    """
    if risk_cols is None:
        risk_cols = {
            "ofci_p": "ofci_pct",
            "shd_p": "shd_pct",
            "vol_spike_p": "atr_percentile",
        }

    # 识别 ET 激活
    et_mask = df[et_col].str.contains("ET", case=False, na=False)
    et_count = int(et_mask.sum())
    total_count = len(df)

    if et_count == 0:
        return {
            "et_activation_count": 0,
            "et_activation_rate": 0.0,
            "avg_risk_when_activated": {},
            "extreme_risk_activation_rate": 0.0,
        }

    # 计算激活时的平均风险指标
    et_df = df[et_mask]
    avg_risk = {}
    for risk_name, col_name in risk_cols.items():
        if col_name in et_df.columns:
            avg_risk[risk_name] = float(et_df[col_name].mean())

    # 计算极端风险时的激活率
    # 极端风险：ofci_p > 0.8 或 shd_p > 0.8 或 vol_spike_p > 0.8
    extreme_mask = pd.Series(False, index=df.index)
    for col_name in risk_cols.values():
        if col_name in df.columns:
            extreme_mask |= df[col_name] > 0.8

    extreme_et_count = int((extreme_mask & et_mask).sum())
    extreme_count = int(extreme_mask.sum())
    extreme_activation_rate = (
        extreme_et_count / extreme_count if extreme_count > 0 else 0.0
    )

    return {
        "et_activation_count": et_count,
        "et_activation_rate": et_count / total_count if total_count > 0 else 0.0,
        "avg_risk_when_activated": avg_risk,
        "extreme_risk_activation_rate": extreme_activation_rate,
        "extreme_risk_et_count": extreme_et_count,
        "extreme_risk_total_count": extreme_count,
    }


def compute_et_cost_analysis(
    df: pd.DataFrame,
    et_col: str = "gate_archetype",
    ret_mean_col: str = "ret_mean",
    ret_trend_col: str = "ret_trend",
) -> Dict[str, Any]:
    """
    计算 ET 的长期成本（保险费）

    Args:
        df: 包含 ET 信号和收益的数据
        et_col: ET archetype 列名
        ret_mean_col: 均值回归收益列
        ret_trend_col: 趋势收益列

    Returns:
        Dict with cost analysis
    """
    # 识别 ET 交易
    et_mask = df[et_col].str.contains("ET", case=False, na=False)
    et_df = df[et_mask]

    if len(et_df) == 0:
        return {
            "et_trade_count": 0,
            "et_total_cost": 0.0,
            "et_avg_cost_per_trade": 0.0,
            "et_cost_rate": 0.0,
        }

    # 计算 ET 收益（使用 ret_mean，因为 ET 是 mean reversion）
    et_returns = et_df[ret_mean_col].fillna(0.0)
    et_total_cost = float(et_returns.sum())
    et_avg_cost = float(et_returns.mean())

    # 计算总收益（用于计算成本率）
    total_returns = df[ret_mean_col].fillna(0.0) + df[ret_trend_col].fillna(0.0)
    total_pnl = float(total_returns.sum())

    # 成本率 = ET 成本 / 总收益（如果总收益为正）
    cost_rate = abs(et_total_cost) / abs(total_pnl) if total_pnl != 0 else 0.0

    return {
        "et_trade_count": len(et_df),
        "et_total_cost": et_total_cost,
        "et_avg_cost_per_trade": et_avg_cost,
        "et_cost_rate": cost_rate,
        "total_pnl": total_pnl,
    }


def compare_with_without_et(
    df_with_et: pd.DataFrame,
    df_without_et: pd.DataFrame,
    ret_mean_col: str = "ret_mean",
    ret_trend_col: str = "ret_trend",
) -> Dict[str, Any]:
    """
    对比有 ET vs 无 ET 的尾部风险

    Args:
        df_with_et: 有 ET 对冲的数据
        df_without_et: 无 ET 对冲的数据
        ret_mean_col: 均值回归收益列
        ret_trend_col: 趋势收益列

    Returns:
        Dict with comparison metrics
    """
    # 计算总收益
    returns_with_et = df_with_et[ret_mean_col].fillna(0.0) + df_with_et[
        ret_trend_col
    ].fillna(0.0)
    returns_without_et = df_without_et[ret_mean_col].fillna(0.0) + df_without_et[
        ret_trend_col
    ].fillna(0.0)

    # 计算左尾指标
    tail_with_et = compute_left_tail_metrics(returns_with_et)
    tail_without_et = compute_left_tail_metrics(returns_without_et)

    # 计算改善
    max_dd_improvement = tail_without_et["max_drawdown"] - tail_with_et["max_drawdown"]
    max_loss_improvement = (
        tail_without_et["max_daily_loss"] - tail_with_et["max_daily_loss"]
    )
    cvar_improvement = tail_without_et["cvar_95"] - tail_with_et["cvar_95"]

    return {
        "with_et": tail_with_et,
        "without_et": tail_without_et,
        "max_drawdown_improvement": max_dd_improvement,
        "max_daily_loss_improvement": max_loss_improvement,
        "cvar_95_improvement": cvar_improvement,
        "left_tail_reduction_pct": (
            (tail_without_et["left_tail_ratio"] - tail_with_et["left_tail_ratio"]) * 100
            if tail_without_et["left_tail_ratio"] > 0
            else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze ET hedge effectiveness")
    parser.add_argument(
        "--logs",
        type=str,
        required=True,
        help="Path to logs file (parquet or csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/et_hedge_analysis.json",
        help="Output path for analysis results",
    )
    parser.add_argument(
        "--et-col",
        type=str,
        default="gate_archetype",
        help="ET archetype column name",
    )
    parser.add_argument(
        "--ret-mean-col",
        type=str,
        default="ret_mean",
        help="Mean reversion returns column",
    )
    parser.add_argument(
        "--ret-trend-col",
        type=str,
        default="ret_trend",
        help="Trend returns column",
    )

    args = parser.parse_args()

    # 读取数据
    logs_path = Path(args.logs)
    if logs_path.suffix == ".parquet":
        df = pd.read_parquet(logs_path)
    else:
        df = pd.read_csv(logs_path)

    # 分析
    results = {
        "left_tail_metrics": compute_left_tail_metrics(
            df[args.ret_mean_col].fillna(0.0) + df[args.ret_trend_col].fillna(0.0)
        ),
        "et_activation_timing": analyze_et_activation_timing(df, et_col=args.et_col),
        "et_cost_analysis": compute_et_cost_analysis(
            df,
            et_col=args.et_col,
            ret_mean_col=args.ret_mean_col,
            ret_trend_col=args.ret_trend_col,
        ),
    }

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"ET hedge analysis saved to {output_path}")
    print("\n=== ET Hedge Effectiveness Summary ===")
    print(
        f"ET Activation Rate: {results['et_activation_timing']['et_activation_rate']:.2%}"
    )
    print(f"ET Total Cost: {results['et_cost_analysis']['et_total_cost']:.4f}")
    print(f"Max Drawdown: {results['left_tail_metrics']['max_drawdown']:.4f}")
    print(f"Max Daily Loss: {results['left_tail_metrics']['max_daily_loss']:.4f}")


if __name__ == "__main__":
    main()
