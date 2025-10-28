"""Generate comprehensive backtest report from rolling training results."""

import os
import sys
import pandas as pd
import json
from datetime import datetime


def generate_html_report(results_dir: str, output_path: str):
    """Generate HTML report from rolling training results."""

    # Load results
    csv_path = os.path.join(results_dir, "monthly_results_advanced_2025.csv")
    summary_path = os.path.join(results_dir, "summary_advanced.json")

    if not os.path.exists(csv_path):
        print(f"❌ Results file not found: {csv_path}")
        return

    results_df = pd.read_csv(csv_path)

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Advanced Rolling Training Report 2025</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .summary-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-value.positive {{
            color: #27ae60;
        }}
        .metric-value.negative {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .config {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            text-align: right;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>📊 Advanced Rolling Training Report 2025</h1>
    
    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    
    <h2>Configuration</h2>
    <div class="config">
        <strong>Feature Engineering:</strong> {summary.get('feature_engineering', 'N/A')}<br>
        <strong>Feature Management:</strong> {summary.get('feature_management', 'N/A')}<br>
        <strong>CVD Improvements:</strong> {summary.get('cvd_improvements', 'N/A')}<br>
        <strong>Dimensionality Reduction:</strong> {summary.get('dimensionality_reduction', 'N/A')}<br>
        <strong>Training Method:</strong> {summary.get('training_method', 'N/A')}<br>
        <strong>Initial Train:</strong> {summary.get('initial_train', 'N/A')}<br>
        <strong>Test Period:</strong> {summary.get('test_period', 'N/A')}<br>
        <strong>PCA Features:</strong> {summary.get('num_features', 'N/A')}<br>
        <strong>Original Features:</strong> {summary.get('original_features', 'N/A')}<br>
    </div>
    
    <h2>Overall Performance</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-label">Average Return</div>
            <div class="metric-value {'positive' if summary.get('avg_return', 0) > 0 else 'negative'}">
                {summary.get('avg_return', 0):.2f}%
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">Average Win Rate</div>
            <div class="metric-value">
                {summary.get('avg_win_rate', 0):.1f}%
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">Average Profit Factor</div>
            <div class="metric-value">
                {summary.get('avg_profit_factor', 0):.2f}
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">Average Max Drawdown</div>
            <div class="metric-value negative">
                {summary.get('avg_max_drawdown', 0):.2f}%
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">Average Sharpe Ratio</div>
            <div class="metric-value">
                {summary.get('avg_sharpe_ratio', 0):.3f}
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">Average Quality Score</div>
            <div class="metric-value">
                {summary.get('avg_quality_score', 0):.3f}
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">
                {summary.get('total_trades', 0)}
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">Months Tested</div>
            <div class="metric-value">
                {summary.get('total_months_tested', 0)}
            </div>
        </div>
    </div>
    
    <h2>Monthly Results</h2>
    <table>
        <tr>
            <th>Month</th>
            <th>Trades</th>
            <th>Return (%)</th>
            <th>Win Rate (%)</th>
            <th>Profit Factor</th>
            <th>Max DD (%)</th>
            <th>Sharpe</th>
            <th>Quality</th>
        </tr>
"""

    for _, row in results_df.iterrows():
        return_class = "positive" if row["total_return"] > 0 else "negative"
        html += f"""
        <tr>
            <td>{row['test_month']}</td>
            <td>{row['total_trades']}</td>
            <td style="color: {'green' if row['total_return'] > 0 else 'red'}">{row['total_return']:.2f}</td>
            <td>{row['win_rate']:.1f}</td>
            <td>{row['profit_factor']:.2f}</td>
            <td style="color: red">{row['max_drawdown']:.2f}</td>
            <td>{row.get('sharpe_ratio', 0):.3f}</td>
            <td>{row.get('quality_score', 0):.3f}</td>
        </tr>
"""

    html += """
    </table>
    
    <h2>Feature Management</h2>
    <div class="summary-box">
        <p><strong>Dynamic Feature Selection:</strong> Features are dynamically selected based on importance across rolling windows.</p>
        <p><strong>CVD Improvements:</strong> Multi-timeframe CVD (short/medium/long) with rolling windows instead of cumsum.</p>
        <p><strong>Transformer Features:</strong> Time series features extracted using Transformer encoder (60 bars → 64 dimensions).</p>
        <p><strong>Incremental PCA:</strong> Dimensionality reduction with incremental updates to adapt to new data.</p>
        <p><strong>Warm Start:</strong> Models retain knowledge from previous training rounds for faster convergence.</p>
    </div>
    
</body>
</html>
"""

    # Save report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Report generated: {output_path}")


def main():
    results_dir = "results/monthly_rolling_2025_advanced"
    output_path = os.path.join(results_dir, "backtest_report.html")

    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        print("   Run training first with: make rolling-2025-advanced")
        return

    print("\n📊 Generating backtest report...")
    generate_html_report(results_dir, output_path)
    print(f"\n✅ Report saved to: {output_path}")


if __name__ == "__main__":
    main()
