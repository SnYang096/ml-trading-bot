#!/usr/bin/env python3
"""
单个因子测试工具

功能：
1. 测试单个或多个指定因子的效果
2. 计算因子的 IC、IR、相关性等指标
3. 生成因子分析报告
4. 支持实盘时只计算需要的因子

用法：
    python scripts/factor_management/test_single_factor.py \
        --factors rsi_7 zigzag_normalized \
        --data-path /data/parquet_data \
        --symbol BTCUSDT \
        --start-date 2024-01-01 \
        --end-date 2024-12-31
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
)
from src.data_tools.data_loader import MarketDataLoader
from src.data_tools.rolling_data import create_labels_multi_horizon


def calculate_factor_ic(
    df: pd.DataFrame,
    factor_name: str,
    target_col: str = "future_return_1",
    method: str = "spearman",
) -> Dict:
    """计算单个因子的 IC 值"""
    if factor_name not in df.columns:
        return {"error": f"Factor {factor_name} not found"}

    if target_col not in df.columns:
        return {"error": f"Target {target_col} not found"}

    # 对齐数据
    aligned = pd.DataFrame(
        {"factor": df[factor_name], "target": df[target_col]}
    ).dropna()

    if len(aligned) < 10:
        return {"error": "Insufficient samples", "sample_count": len(aligned)}

    # 计算 IC
    if method == "spearman":
        ic, p_value = spearmanr(
            aligned["factor"].values, aligned["target"].values, nan_policy="omit"
        )
    else:  # pearson
        ic, p_value = pearsonr(aligned["factor"].values, aligned["target"].values)

    # 计算统计信息
    factor_values = aligned["factor"].values
    target_values = aligned["target"].values

    result = {
        "factor_name": factor_name,
        "ic": float(ic) if not np.isnan(ic) else 0.0,
        "ic_abs": float(abs(ic)) if not np.isnan(ic) else 0.0,
        "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
        "sample_count": len(aligned),
        "method": method,
        "factor_stats": {
            "mean": float(np.mean(factor_values)),
            "std": float(np.std(factor_values)),
            "min": float(np.min(factor_values)),
            "max": float(np.max(factor_values)),
            "nan_count": int(df[factor_name].isna().sum()),
        },
        "target_stats": {
            "mean": float(np.mean(target_values)),
            "std": float(np.std(target_values)),
        },
    }

    # 计算 IC 的滚动窗口统计（如果数据足够）
    if len(aligned) > 100:
        window = min(100, len(aligned) // 10)
        rolling_ic = []
        for i in range(0, len(aligned) - window, window):
            window_data = aligned.iloc[i : i + window]
            if method == "spearman":
                ic_window, _ = spearmanr(
                    window_data["factor"].values,
                    window_data["target"].values,
                    nan_policy="omit",
                )
            else:
                ic_window, _ = pearsonr(
                    window_data["factor"].values, window_data["target"].values
                )
            if not np.isnan(ic_window):
                rolling_ic.append(ic_window)

        if rolling_ic:
            result["rolling_ic_stats"] = {
                "mean": float(np.mean(rolling_ic)),
                "std": float(np.std(rolling_ic)),
                "ir": float(np.mean(rolling_ic) / (np.std(rolling_ic) + 1e-8)),  # IC IR
            }

    return result


def analyze_factor_distribution(df: pd.DataFrame, factor_name: str) -> Dict:
    """分析因子的分布特征"""
    if factor_name not in df.columns:
        return {"error": f"Factor {factor_name} not found"}

    factor_values = df[factor_name].dropna()

    if len(factor_values) == 0:
        return {"error": "No valid values"}

    # 计算分位数
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantile_values = {
        f"q{int(q*100)}": float(factor_values.quantile(q)) for q in quantiles
    }

    # 检查异常值
    q1 = factor_values.quantile(0.25)
    q3 = factor_values.quantile(0.75)
    iqr = q3 - q1
    outliers = factor_values[
        (factor_values < q1 - 3 * iqr) | (factor_values > q3 + 3 * iqr)
    ]

    return {
        "quantiles": quantile_values,
        "outlier_count": len(outliers),
        "outlier_ratio": len(outliers) / len(factor_values),
        "is_normalized": abs(factor_values.mean()) < 1.0 and factor_values.std() < 10.0,
    }


def test_factors(
    factors: List[str],
    data_path: str,
    symbol: str,
    start_date: str,
    end_date: str,
    feature_type: str = "comprehensive",
    timeframe: str = "5T",
    horizons: List[int] = [1, 5, 10, 15],
    output_dir: Optional[str] = None,
) -> Dict:
    """测试指定的因子"""

    # 处理逗号分隔的因子列表（如果传入的是单个字符串）
    if len(factors) == 1 and "," in factors[0]:
        factors = [f.strip() for f in factors[0].split(",")]

    print("=" * 80)
    print("单个因子测试工具")
    print("=" * 80)
    print(f"因子列表: {factors}")
    print(f"数据路径: {data_path}")
    print(f"交易对: {symbol}")
    print(f"时间范围: {start_date} 到 {end_date}")
    print(f"特征类型: {feature_type}")
    print(f"时间框架: {timeframe}")
    print("=" * 80)

    # 1. 加载数据（支持多个交易对）
    print("\n1. 加载数据...")
    try:
        # 处理多个交易对（逗号分隔）
        symbols = [s.strip() for s in symbol.split(",")]

        dfs = []
        loader = MarketDataLoader(data_path)

        for sym in symbols:
            print(f"   加载 {sym}...")
            df_single = loader.load_data(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
            )
            if len(df_single) > 0:
                # 添加 symbol 列以便区分
                df_single = df_single.copy()
                df_single["_symbol"] = sym
                dfs.append(df_single)
                print(f"   ✅ {sym}: {len(df_single)} 条数据")
            else:
                print(f"   ⚠️  {sym}: 无数据")

        if not dfs:
            return {"error": f"No data loaded for any symbol in: {symbols}"}

        # 合并所有交易对的数据
        df = pd.concat(dfs, ignore_index=False).sort_index()
        print(f"   ✅ 总共加载了 {len(df)} 条数据（来自 {len(symbols)} 个交易对）")
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    # 2. 特征工程（只计算需要的因子）
    print("\n2. 特征工程（只计算指定因子）...")
    try:
        engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
        factors_set = set(factors)
        df_features = engineer.engineer_all_features(
            df, fit=True, required_features=factors_set
        )
        print(f"   ✅ 特征工程完成，生成了 {len(df_features.columns)} 个特征")
    except Exception as e:
        return {"error": f"Failed to engineer features: {e}"}

    # 3. 创建标签
    print("\n3. 创建标签...")
    try:
        df_features = create_labels_multi_horizon(df_features, horizons=horizons)
        print(f"   ✅ 标签创建完成")
    except Exception as e:
        return {"error": f"Failed to create labels: {e}"}

    # 4. 检查因子是否存在
    print("\n4. 检查因子可用性...")
    available_factors = []
    missing_factors = []

    for factor in factors:
        if factor in df_features.columns:
            available_factors.append(factor)
            print(f"   ✅ {factor}")
        else:
            missing_factors.append(factor)
            print(f"   ❌ {factor} (未找到)")

    if not available_factors:
        return {"error": "No factors available", "missing": missing_factors}

    # 5. 计算每个因子的 IC
    print("\n5. 计算因子 IC 值...")
    results = {}

    for factor in available_factors:
        print(f"\n   分析因子: {factor}")

        # IC 分析
        ic_results = {}
        for horizon in horizons:
            target_col = f"future_return_{horizon}"
            if target_col in df_features.columns:
                ic_result = calculate_factor_ic(
                    df_features, factor, target_col, method="spearman"
                )
                if "error" not in ic_result:
                    ic_results[f"horizon_{horizon}"] = ic_result
                    print(
                        f"      Horizon {horizon}: IC={ic_result['ic']:.6f}, |IC|={ic_result['ic_abs']:.6f}"
                    )

        # 分布分析
        dist_result = analyze_factor_distribution(df_features, factor)

        results[factor] = {
            "ic_analysis": ic_results,
            "distribution": dist_result,
            "available": True,
        }

    # 6. 生成报告
    print("\n6. 生成报告...")

    summary = {
        "factors_tested": factors,
        "factors_available": available_factors,
        "factors_missing": missing_factors,
        "results": results,
        "config": {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "feature_type": feature_type,
            "timeframe": timeframe,
            "horizons": horizons,
        },
    }

    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # JSON 报告
        json_path = output_path / "factor_test_results.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   ✅ JSON 报告保存到: {json_path}")

        # 文本报告
        txt_path = output_path / "factor_test_report.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("因子测试报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"测试因子: {', '.join(factors)}\n")
            f.write(f"可用因子: {', '.join(available_factors)}\n")
            if missing_factors:
                f.write(f"缺失因子: {', '.join(missing_factors)}\n")
            f.write("\n")

            for factor, result in results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"因子: {factor}\n")
                f.write(f"{'='*80}\n")

                if "ic_analysis" in result:
                    f.write("\nIC 分析:\n")
                    for horizon_key, ic_data in result["ic_analysis"].items():
                        horizon = horizon_key.replace("horizon_", "")
                        f.write(f"  Horizon {horizon}:\n")
                        f.write(f"    IC: {ic_data['ic']:.6f}\n")
                        f.write(f"    |IC|: {ic_data['ic_abs']:.6f}\n")
                        f.write(f"    p-value: {ic_data['p_value']:.6f}\n")
                        if "rolling_ic_stats" in ic_data:
                            f.write(
                                f"    IC IR: {ic_data['rolling_ic_stats']['ir']:.6f}\n"
                            )

                if "distribution" in result and "error" not in result["distribution"]:
                    f.write("\n分布分析:\n")
                    dist = result["distribution"]
                    f.write(f"  是否归一化: {dist.get('is_normalized', False)}\n")
                    f.write(f"  异常值比例: {dist.get('outlier_ratio', 0):.2%}\n")

        print(f"   ✅ 文本报告保存到: {txt_path}")

        # HTML 报告（包含可视化）
        html_path = output_path / "factor_test_report.html"
        generate_html_report(summary, html_path)
        print(f"   ✅ HTML 报告保存到: {html_path}")

    return summary


def generate_html_report(summary: Dict, output_path: Path) -> None:
    """生成HTML格式的因子测试报告（包含可视化图表）"""

    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>因子测试报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-left: 10px;
            border-left: 4px solid #3498db;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .info-card strong {{
            color: #2c3e50;
            display: block;
            margin-bottom: 5px;
        }}
        .factor-section {{
            margin-top: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .metric-card {{
            display: inline-block;
            background: white;
            padding: 15px 20px;
            margin: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            min-width: 200px;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-value.positive {{
            color: #27ae60;
        }}
        .metric-value.negative {{
            color: #e74c3c;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge.success {{
            background: #d4edda;
            color: #155724;
        }}
        .badge.warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .badge.danger {{
            background: #f8d7da;
            color: #721c24;
        }}
        .explanation {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
        .explanation h3 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .explanation ul {{
            margin-left: 20px;
        }}
        .explanation li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 因子测试报告</h1>
        
        <div class="info-grid">
            <div class="info-card">
                <strong>测试因子</strong>
                {', '.join(summary['factors_tested'])}
            </div>
            <div class="info-card">
                <strong>交易对</strong>
                {summary['config']['symbol']}
            </div>
            <div class="info-card">
                <strong>时间范围</strong>
                {summary['config']['start_date']} 至 {summary['config']['end_date']}
            </div>
            <div class="info-card">
                <strong>特征类型</strong>
                {summary['config']['feature_type']}
            </div>
            <div class="info-card">
                <strong>时间框架</strong>
                {summary['config']['timeframe']}
            </div>
            <div class="info-card">
                <strong>可用因子</strong>
                {len(summary['factors_available'])} / {len(summary['factors_tested'])}
            </div>
        </div>
"""

    # 添加解释说明
    html_content += """
        <div class="explanation">
            <h3>📖 指标说明</h3>
            <ul>
                <li><strong>IC (Information Coefficient)</strong>: 因子值与未来收益率的相关系数，范围 [-1, 1]。IC > 0.05 表示因子有效，IC > 0.1 表示因子很强。</li>
                <li><strong>IC IR (Information Ratio)</strong>: IC 的均值除以标准差，衡量因子的稳定性。IR > 0.5 表示因子稳定，IR > 1.0 表示因子非常稳定。</li>
                <li><strong>p-value</strong>: 统计显著性，p < 0.05 表示结果显著。</li>
                <li><strong>是否归一化</strong>: 因子值是否已归一化（均值接近0，标准差较小）。归一化的因子更适合机器学习模型。</li>
                <li><strong>异常值比例</strong>: 使用 IQR 方法检测的异常值占比。比例过高可能影响模型性能。</li>
            </ul>
        </div>
"""

    # 为每个因子生成详细报告
    for factor_name, result in summary["results"].items():
        ic_analysis = result.get("ic_analysis", {})
        distribution = result.get("distribution", {})

        html_content += f"""
        <div class="factor-section">
            <h2>因子: {factor_name}</h2>
            
            <h3>IC 分析结果</h3>
            <div class="chart-container">
                <canvas id="icChart_{factor_name.replace('.', '_')}"></canvas>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>预测周期</th>
                        <th>IC</th>
                        <th>|IC|</th>
                        <th>IC IR</th>
                        <th>p-value</th>
                        <th>样本数</th>
                        <th>评价</th>
                    </tr>
                </thead>
                <tbody>
"""

        # IC 表格数据
        horizons = []
        ic_values = []
        ic_abs_values = []
        ic_ir_values = []

        for horizon_key, ic_data in sorted(ic_analysis.items()):
            horizon = horizon_key.replace("horizon_", "")
            horizons.append(f"H{horizon}")
            ic = ic_data.get("ic", 0)
            ic_abs = ic_data.get("ic_abs", 0)
            ic_ir = ic_data.get("rolling_ic_stats", {}).get("ir", 0)
            p_value = ic_data.get("p_value", 1)
            sample_count = ic_data.get("sample_count", 0)

            ic_values.append(ic)
            ic_abs_values.append(ic_abs)
            ic_ir_values.append(ic_ir)

            # 评价
            if ic_abs > 0.1 and ic_ir > 0.5:
                evaluation = '<span class="badge success">优秀</span>'
            elif ic_abs > 0.05 and ic_ir > 0.3:
                evaluation = '<span class="badge warning">良好</span>'
            else:
                evaluation = '<span class="badge danger">一般</span>'

            html_content += f"""
                    <tr>
                        <td>{horizon}</td>
                        <td class="metric-value {'positive' if ic > 0 else 'negative'}">{ic:.6f}</td>
                        <td>{ic_abs:.6f}</td>
                        <td>{ic_ir:.2f}</td>
                        <td>{p_value:.6f}</td>
                        <td>{sample_count:,}</td>
                        <td>{evaluation}</td>
                    </tr>
"""

        html_content += """
                </tbody>
            </table>
            
            <h3>因子分布分析</h3>
            <div class="info-grid">
"""

        # 分布分析
        quantiles = distribution.get("quantiles", {})
        is_normalized = distribution.get("is_normalized", False)
        outlier_ratio = distribution.get("outlier_ratio", 0)

        html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">是否归一化</div>
                    <div class="metric-value">{'是' if is_normalized else '否'}</div>
                    <div style="margin-top: 5px; font-size: 0.85em; color: #7f8c8d;">
                        {'✅ 因子已归一化，适合机器学习模型' if is_normalized else '⚠️ 因子未归一化，建议进行归一化处理'}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">异常值比例</div>
                    <div class="metric-value">{outlier_ratio:.2%}</div>
                    <div style="margin-top: 5px; font-size: 0.85em; color: #7f8c8d;">
                        {'✅ 异常值很少' if outlier_ratio < 0.05 else '⚠️ 异常值较多，可能影响模型性能'}
                    </div>
                </div>
"""

        # 分位数信息
        if quantiles:
            html_content += """
                <div class="metric-card">
                    <div class="metric-label">分位数统计</div>
                    <div style="font-size: 0.9em; margin-top: 5px;">
                        <div>Q25: {:.4f}</div>
                        <div>Q50: {:.4f}</div>
                        <div>Q75: {:.4f}</div>
                    </div>
                </div>
""".format(
                quantiles.get("q25", 0),
                quantiles.get("q50", 0),
                quantiles.get("q75", 0),
            )

        html_content += """
            </div>
        </div>
"""

        # 添加图表脚本
        chart_id = factor_name.replace(".", "_").replace("-", "_").replace(" ", "_")
        horizons_json = json.dumps(horizons)
        ic_values_json = json.dumps(ic_values)
        ic_abs_values_json = json.dumps(ic_abs_values)
        ic_ir_values_json = json.dumps(ic_ir_values)
        factor_name_escaped = factor_name.replace("'", "\\'").replace('"', '\\"')

        html_content += f"""
        <script>
            const ctx_{chart_id} = document.getElementById('icChart_{factor_name.replace('.', '_')}');
            new Chart(ctx_{chart_id}, {{
                type: 'bar',
                data: {{
                    labels: {horizons_json},
                    datasets: [
                        {{
                            label: 'IC',
                            data: {ic_values_json},
                            backgroundColor: 'rgba(52, 152, 219, 0.6)',
                            borderColor: 'rgba(52, 152, 219, 1)',
                            borderWidth: 1
                        }},
                        {{
                            label: '|IC|',
                            data: {ic_abs_values_json},
                            backgroundColor: 'rgba(46, 204, 113, 0.6)',
                            borderColor: 'rgba(46, 204, 113, 1)',
                            borderWidth: 1
                        }},
                        {{
                            label: 'IC IR',
                            data: {ic_ir_values_json},
                            backgroundColor: 'rgba(241, 196, 15, 0.6)',
                            borderColor: 'rgba(241, 196, 15, 1)',
                            borderWidth: 1,
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'IC / |IC|'
                            }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{
                                display: true,
                                text: 'IC IR'
                            }},
                            grid: {{
                                drawOnChartArea: false
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        title: {{
                            display: true,
                            text: '因子 IC 分析 - {factor_name_escaped}'
                        }}
                    }}
                }}
            }});
        </script>
"""

    html_content += """
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="测试单个或多个因子")
    parser.add_argument(
        "--factors",
        nargs="+",
        required=True,
        help="要测试的因子名称列表，支持空格或逗号分隔，例如: rsi_7 zigzag_normalized 或 price_to_zz_high_pct,price_to_poc_pct",
    )
    parser.add_argument(
        "--data-path", type=str, default="/data/parquet_data", help="数据路径"
    )
    parser.add_argument(
        "--symbol", type=str, required=True, help="交易对符号，例如: BTCUSDT"
    )
    parser.add_argument(
        "--start-date", type=str, required=True, help="开始日期，格式: YYYY-MM-DD"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="结束日期，格式: YYYY-MM-DD"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="comprehensive",
        help="特征类型: baseline, default, enhanced, comprehensive",
    )
    parser.add_argument(
        "--timeframe", type=str, default="5T", help="时间框架，例如: 5T, 15T, 1H"
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=[1, 5, 10, 15], help="预测周期列表"
    )
    parser.add_argument("--output-dir", type=str, help="输出目录（可选）")

    args = parser.parse_args()

    # 如果没有指定输出目录，使用默认目录
    if not args.output_dir:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/factor_tests/{args.symbol}_{timestamp}"

    results = test_factors(
        factors=args.factors,
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        feature_type=args.feature_type,
        timeframe=args.timeframe,
        horizons=args.horizons,
        output_dir=args.output_dir,
    )

    if "error" in results:
        print(f"\n❌ 错误: {results['error']}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
