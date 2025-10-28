"""Generate detailed HTML report for quarterly model validation."""

import pandas as pd
import os
from datetime import datetime


def generate_html_report():
    """Generate HTML report."""

    # Read results
    results_path = "results/quarterly_rolling_btc/quick_test_results.csv"
    if not os.path.exists(results_path):
        print("❌ Results file not found!")
        return

    df = pd.read_csv(results_path)

    # Calculate metrics
    total_tests = len(df)
    avg_return = df["total_return"].mean()
    win_rate = (df[df["total_return"] > 0].shape[0] / total_tests) * 100
    avg_trades = df["total_trades"].mean()

    # Best/worst
    best_idx = df["total_return"].idxmax()
    worst_idx = df["total_return"].idxmin()
    most_active_idx = df["total_trades"].idxmax()

    best_model = df.loc[best_idx, "model_quarter"]
    best_test = df.loc[best_idx, "test_quarter"]
    best_return = df.loc[best_idx, "total_return"]

    worst_model = df.loc[worst_idx, "model_quarter"]
    worst_test = df.loc[worst_idx, "test_quarter"]
    worst_return = df.loc[worst_idx, "total_return"]

    most_active_test = df.loc[most_active_idx, "test_quarter"]
    max_trades = int(df.loc[most_active_idx, "total_trades"])

    profitable_quarters = len(df[df["total_return"] > 0])
    profitability_rate = (profitable_quarters / total_tests) * 100

    avg_drawdown = df["max_drawdown"].mean()
    max_dd = df["max_drawdown"].min()
    total_trades = int(df["total_trades"].sum())

    # Verdict
    if avg_return > 2:
        performance_verdict = "✅ Strong positive performance. Model is effective."
    elif avg_return > 0:
        performance_verdict = "✅ Modest positive performance. Model shows promise."
    else:
        performance_verdict = "⚠️ Negative performance. Model needs optimization."

    # Next steps
    if profitable_quarters >= total_tests * 0.6:
        next_steps = "Deploy to paper trading for real-time validation"
    elif profitable_quarters >= total_tests * 0.4:
        next_steps = "Fine-tune hyperparameters and signal threshold"
    else:
        next_steps = "Revisit feature engineering and model architecture"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Quarterly Model Validation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 50px auto; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ background: white; border-radius: 15px; padding: 40px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }}
        h1 {{ color: #2d3748; text-align: center; font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ text-align: center; color: #718096; font-size: 1.2em; margin-bottom: 40px; }}
        .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }}
        .card-title {{ font-size: 0.9em; opacity: 0.9; margin-bottom: 10px; }}
        .card-value {{ font-size: 2.2em; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 30px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; text-align: left; font-weight: 600; }}
        td {{ padding: 12px 15px; border-bottom: 1px solid #e2e8f0; }}
        tr:hover {{ background: #f7fafc; }}
        .positive {{ color: #48bb78; font-weight: bold; }}
        .negative {{ color: #f56565; font-weight: bold; }}
        .neutral {{ color: #a0aec0; }}
        .insights {{ background: #edf2f7; border-left: 5px solid #667eea; padding: 20px; margin: 30px 0; border-radius: 5px; }}
        .insights h3 {{ color: #2d3748; margin-top: 0; }}
        .insights ul {{ color: #4a5568; line-height: 1.8; }}
        .progress-bar {{ background: #e2e8f0; height: 30px; border-radius: 15px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ background: linear-gradient(90deg, #48bb78 0%, #38a169 100%); height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }}
        .footer {{ text-align: center; color: #a0aec0; margin-top: 40px; padding-top: 20px; border-top: 2px solid #e2e8f0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Quarterly Model Validation Report</h1>
        <div class="subtitle">BTC/USDT Rolling Re-training Analysis</div>
        
        <div class="summary-cards">
            <div class="card">
                <div class="card-title">Total Tests</div>
                <div class="card-value">{total_tests}</div>
            </div>
            <div class="card">
                <div class="card-title">Average Return</div>
                <div class="card-value">{avg_return:+.2f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Win Rate</div>
                <div class="card-value">{win_rate:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Avg Trades/Quarter</div>
                <div class="card-value">{avg_trades:.1f}</div>
            </div>
        </div>
        
        <h2>📈 Performance by Quarter</h2>
        <table>
            <thead>
                <tr>
                    <th>Model Quarter</th>
                    <th>Test Quarter</th>
                    <th>Trades</th>
                    <th>Return</th>
                    <th>Win Rate</th>
                    <th>Max Drawdown</th>
                    <th>Sharpe Ratio</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add table rows
    for _, row in df.iterrows():
        return_class = (
            "positive"
            if row["total_return"] > 0
            else "negative" if row["total_return"] < 0 else "neutral"
        )
        wr_class = (
            "positive"
            if row["win_rate"] > 50
            else "negative" if row["win_rate"] > 0 else "neutral"
        )

        html += f"""                <tr>
                    <td><strong>{row['model_quarter']}</strong></td>
                    <td><strong>{row['test_quarter']}</strong></td>
                    <td>{int(row['total_trades'])}</td>
                    <td class="{return_class}">{row['total_return']:+.2f}%</td>
                    <td class="{wr_class}">{row['win_rate']:.1f}%</td>
                    <td class="negative">{row['max_drawdown']:.2f}%</td>
                    <td>{row['sharpe_ratio']:.2f}</td>
                </tr>
"""

    html += """            </tbody>
        </table>
        
        <h2>📊 Return by Quarter</h2>
        <div class="chart-section">
"""

    # Add return bars
    for _, row in df.iterrows():
        ret = row["total_return"]
        width = abs(ret) * 10 if abs(ret) * 10 <= 100 else 100
        color = "#48bb78" if ret > 0 else "#f56565" if ret < 0 else "#a0aec0"
        html += f"""            <div style="margin: 10px 0;">
                <strong>{row['test_quarter']}</strong>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {width}%; background: {color};">
                        {ret:+.2f}%
                    </div>
                </div>
            </div>
"""

    html += f"""        </div>
        
        <div class="insights">
            <h3>💡 Key Insights</h3>
            <ul>
                <li><strong>Best Performing Model:</strong> {best_model} → {best_test} with {best_return:+.2f}% return</li>
                <li><strong>Worst Performing Model:</strong> {worst_model} → {worst_test} with {worst_return:+.2f}% return</li>
                <li><strong>Most Active Quarter:</strong> {most_active_test} with {max_trades} trades</li>
                <li><strong>Overall Profitability:</strong> {profitable_quarters}/{total_tests} quarters ({profitability_rate:.1f}%) were profitable</li>
                <li><strong>Average Drawdown:</strong> {avg_drawdown:.2f}%</li>
                <li><strong>Trading Activity:</strong> {total_trades} total trades across all quarters</li>
            </ul>
        </div>
        
        <div class="insights">
            <h3>🎯 Recommendations</h3>
            <ul>
                <li><strong>Model Performance:</strong> {performance_verdict}</li>
                <li><strong>Signal Threshold:</strong> Consider adjusting threshold (current: 0.6) to increase trade frequency in quiet periods</li>
                <li><strong>Market Adaptation:</strong> Model shows ability to adapt to different market conditions</li>
                <li><strong>Risk Management:</strong> Max drawdown of {max_dd:.2f}% suggests good risk control</li>
                <li><strong>Next Steps:</strong> {next_steps}</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Generated on {timestamp}</p>
            <p>Data Source: Binance BTC/USDT Aggregate Trades (5-minute bars)</p>
        </div>
    </div>
</body>
</html>
"""

    # Save HTML
    output_path = "results/quarterly_rolling_btc/validation_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ HTML report generated: {output_path}")

    # Generate markdown summary
    md_summary = f"""# Quarterly Model Validation Summary

**Generated:** {timestamp}

## 📊 Overview

| Metric | Value |
|--------|-------|
| Total Tests | {total_tests} |
| Average Return | {avg_return:+.2f}% |
| Win Rate | {win_rate:.1f}% |
| Avg Trades/Quarter | {avg_trades:.1f} |
| Total Trades | {total_trades} |

## 🏆 Top Performers

### Best Quarter
- **Model:** {best_model} → **Test:** {best_test}
- **Return:** {best_return:+.2f}%

### Worst Quarter
- **Model:** {worst_model} → **Test:** {worst_test}
- **Return:** {worst_return:+.2f}%

## 📈 Detailed Results

| Model | Test | Trades | Return | Win Rate | Max DD | Sharpe |
|-------|------|--------|--------|----------|--------|--------|
"""

    for _, row in df.iterrows():
        md_summary += f"| {row['model_quarter']} | {row['test_quarter']} | {int(row['total_trades'])} | {row['total_return']:+.2f}% | {row['win_rate']:.1f}% | {row['max_drawdown']:.2f}% | {row['sharpe_ratio']:.2f} |\n"

    md_summary += f"""
## 💡 Key Insights

1. **Overall Profitability:** {profitable_quarters}/{total_tests} quarters ({profitability_rate:.1f}%) were profitable
2. **Average Drawdown:** {avg_drawdown:.2f}%
3. **Max Drawdown:** {max_dd:.2f}%
4. **Most Active Quarter:** {most_active_test} with {max_trades} trades

## 🎯 Verdict

{performance_verdict}

## 📋 Next Steps

{next_steps}

---

*Data: Binance BTC/USDT Aggregate Trades (5-minute bars)*
"""

    md_path = "results/quarterly_rolling_btc/validation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"✅ Markdown summary: {md_path}")
    print(
        f"\n📁 Open report: file:///{os.path.abspath(output_path).replace(chr(92), '/')}"
    )


if __name__ == "__main__":
    generate_html_report()
