#!/usr/bin/env python3
"""
实验3：生成最终上线报告

按symbol分析KPI，计算左尾风险指标，判断上线条件，生成综合报告。

使用方法:
    python scripts/experiment_final_report.py \
        --exec-log results/pipeline_<run_id>/execution_log.jsonl \
        --out-dir results/experiments/final_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.diagnostics.execution_log_aggregate import (
    aggregate_stage_logs,
)


def compute_var_cvar(returns: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
    """计算VaR和CVaR"""
    if len(returns) == 0:
        return {"var": 0.0, "cvar": 0.0}

    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return {"var": 0.0, "cvar": 0.0}

    var = float(np.percentile(returns_clean, (1 - confidence) * 100))
    cvar = float(returns_clean[returns_clean <= var].mean())

    return {
        "var": var,
        "cvar": cvar,
    }


def compute_metrics(returns: pd.Series) -> Dict[str, float]:
    """计算性能指标"""
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
        }

    # Sharpe ratio
    mean_ret = returns_clean.mean()
    std_ret = returns_clean.std()
    sharpe = (mean_ret / std_ret * np.sqrt(240)) if std_ret > 0 else 0.0  # 年化

    # Max drawdown
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())

    # Win rate
    win_rate = float((returns_clean > 0).mean())

    # Average R/R (简化计算，使用正收益/负收益的绝对值比)
    positive_returns = returns_clean[returns_clean > 0]
    negative_returns = returns_clean[returns_clean < 0]
    avg_rr = (
        abs(positive_returns.mean() / negative_returns.mean())
        if len(negative_returns) > 0 and negative_returns.mean() != 0
        else 0.0
    )

    # Total return
    total_return = float((1 + returns_clean).prod() - 1)

    # Volatility (年化)
    volatility = float(std_ret * np.sqrt(240))

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "total_return": total_return,
        "volatility": volatility,
    }


def analyze_symbol(
    symbol: str,
    records: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """分析单个symbol"""
    symbol_records = [r for r in records if r.get("symbol") == symbol]

    if len(symbol_records) == 0:
        return None

    # 提取returns
    returns_list = []
    trades = {
        "total": 0,
        "winning": 0,
        "losing": 0,
    }

    for rec in symbol_records:
        execution = rec.get("execution") or {}
        if not execution.get("intent", False):
            continue

        returns_data = rec.get("returns") or {}
        ret_mean = returns_data.get("ret_mean")
        ret_trend = returns_data.get("ret_trend")

        # 根据archetype选择return
        gate = rec.get("gate") or {}
        archetype = gate.get("archetype") or execution.get("archetype", "")

        ret = None
        if archetype and ("TC" in str(archetype) or "TE" in str(archetype)):
            ret = ret_trend
        elif archetype and ("FR" in str(archetype) or "ET" in str(archetype)):
            ret = ret_mean
        else:
            ret = ret_mean if ret_mean is not None else ret_trend

        if ret is not None and not pd.isna(ret):
            returns_list.append(ret)
            trades["total"] += 1
            if ret > 0:
                trades["winning"] += 1
            elif ret < 0:
                trades["losing"] += 1

    if len(returns_list) == 0:
        return {
            "symbol": symbol,
            "period": f"{start_date} to {end_date}",
            "metrics": {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "avg_rr": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
            },
            "trades": trades,
            "recommendation": "REJECTED",
            "reasons": ["No trades executed"],
        }

    returns_series = pd.Series(returns_list)

    # 计算指标
    metrics = compute_metrics(returns_series)
    risk_metrics = compute_var_cvar(returns_series, confidence=0.95)

    metrics.update(
        {
            "var_95": risk_metrics["var"],
            "cvar_95": risk_metrics["cvar"],
        }
    )

    # 判断上线条件
    recommendation = "APPROVED"
    reasons = []

    if metrics["sharpe_ratio"] < 1.0:
        recommendation = "REJECTED"
        reasons.append(f"Sharpe ratio too low: {metrics['sharpe_ratio']:.2f} < 1.0")
    elif metrics["sharpe_ratio"] < 1.5:
        recommendation = "CONDITIONAL"
        reasons.append(f"Sharpe ratio moderate: {metrics['sharpe_ratio']:.2f}")

    if abs(metrics["max_drawdown"]) > 0.2:
        if recommendation == "APPROVED":
            recommendation = "CONDITIONAL"
        reasons.append(
            f"Max drawdown too high: {abs(metrics['max_drawdown']):.2%} > 20%"
        )

    if metrics["win_rate"] < 0.45:
        if recommendation == "APPROVED":
            recommendation = "CONDITIONAL"
        reasons.append(f"Win rate too low: {metrics['win_rate']:.2%} < 45%")

    if trades["total"] < 50:
        if recommendation == "APPROVED":
            recommendation = "CONDITIONAL"
        reasons.append(f"Insufficient trades: {trades['total']} < 50")

    if not reasons:
        reasons.append("All criteria met")

    return {
        "symbol": symbol,
        "period": f"{start_date} to {end_date}",
        "metrics": metrics,
        "trades": trades,
        "recommendation": recommendation,
        "reasons": reasons,
    }


def generate_report(
    symbol_results: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """生成最终报告"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 生成JSON报告
    json_path = out_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(symbol_results, f, indent=2, default=str)

    # 生成Markdown报告
    md_path = out_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 最终上线报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 总体统计\n\n")
        total_trades = sum(r["trades"]["total"] for r in symbol_results)
        approved_count = sum(
            1 for r in symbol_results if r["recommendation"] == "APPROVED"
        )
        conditional_count = sum(
            1 for r in symbol_results if r["recommendation"] == "CONDITIONAL"
        )
        rejected_count = sum(
            1 for r in symbol_results if r["recommendation"] == "REJECTED"
        )

        f.write(f"- 总交易数: {total_trades}\n")
        f.write(f"- 批准上线: {approved_count}\n")
        f.write(f"- 条件批准: {conditional_count}\n")
        f.write(f"- 拒绝上线: {rejected_count}\n\n")

        f.write("## 按Symbol详细分析\n\n")
        for result in symbol_results:
            f.write(f"### {result['symbol']}\n\n")
            f.write(f"**期间**: {result['period']}\n\n")

            f.write("#### 性能指标\n")
            metrics = result["metrics"]
            f.write(f"- Sharpe比率: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"- 最大回撤: {metrics['max_drawdown']:.2%}\n")
            f.write(f"- 胜率: {metrics['win_rate']:.2%}\n")
            f.write(f"- 平均R/R: {metrics['avg_rr']:.2f}\n")
            if "total_return" in metrics:
                f.write(f"- 总收益: {metrics['total_return']:.2%}\n")
            if "volatility" in metrics:
                f.write(f"- 波动率: {metrics['volatility']:.2%}\n")
            f.write(f"- VaR(95%): {metrics['var_95']:.4f}\n")
            f.write(f"- CVaR(95%): {metrics['cvar_95']:.4f}\n\n")

            f.write("#### 交易统计\n")
            trades = result["trades"]
            f.write(f"- 总交易数: {trades['total']}\n")
            f.write(f"- 盈利交易: {trades['winning']}\n")
            f.write(f"- 亏损交易: {trades['losing']}\n\n")

            f.write(f"#### 上线建议: **{result['recommendation']}**\n\n")
            f.write("**原因**:\n")
            for reason in result["reasons"]:
                f.write(f"- {reason}\n")
            f.write("\n")

    print(f"✅ 报告已生成:")
    print(f"   - JSON: {json_path}")
    print(f"   - Markdown: {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成最终上线报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exec-log",
        required=True,
        help="Execution log文件或目录（jsonl文件或stage logs目录）",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="开始日期 YYYY-MM-DD（用于报告）",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束日期 YYYY-MM-DD（用于报告）",
    )

    args = parser.parse_args()

    exec_log_path = Path(args.exec_log)
    if not exec_log_path.exists():
        print(f"❌ Execution log不存在: {exec_log_path}")
        return 1

    print(f"📊 加载execution log: {exec_log_path}")
    if exec_log_path.is_dir():
        records = aggregate_stage_logs(exec_log_path)
    else:
        import json

        records = []
        with open(exec_log_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    print(f"✅ 加载了 {len(records)} 条记录")

    # 提取symbols
    symbols = sorted(set(r.get("symbol") for r in records if r.get("symbol")))
    print(f"📈 找到 {len(symbols)} 个symbols: {', '.join(symbols)}")

    # 分析每个symbol
    symbol_results = []
    for symbol in symbols:
        print(f"\n📊 分析 {symbol}...")
        result = analyze_symbol(
            symbol=symbol,
            records=records,
            start_date=args.start_date or "N/A",
            end_date=args.end_date or "N/A",
        )
        if result:
            symbol_results.append(result)
            print(f"  推荐: {result['recommendation']}")

    # 生成报告
    print("\n📝 生成报告...")
    generate_report(symbol_results, Path(args.out_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
