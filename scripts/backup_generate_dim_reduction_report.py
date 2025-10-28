#!/usr/bin/env python3
"""
Generate Comprehensive Dimensionality Reduction Report
Creates a detailed HTML report summarizing all dimensionality reduction results.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime


def generate_dim_reduction_report():
    """
    Generate a comprehensive dimensionality reduction report.
    """
    print("📋 Generating Dimensionality Reduction Summary Report")
    print("=" * 60)

    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    # Collect all results
    results = collect_results()

    # Generate HTML report
    html_content = create_html_report(results)

    # Save report
    report_path = "reports/dimensionality_reduction_summary.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Report generated: {report_path}")
    return report_path


def collect_results():
    """
    Collect all dimensionality reduction results.
    """
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "comparison_results": None,
        "individual_reports": [],
        "visualizations": [],
    }

    # Look for comparison results
    comparison_file = "reports/dimensionality_comparison_results.csv"
    if os.path.exists(comparison_file):
        results["comparison_results"] = pd.read_csv(comparison_file)
        print(
            f"✅ Found comparison results: {len(results['comparison_results'])} methods compared"
        )

    # Look for individual reports
    individual_reports = glob.glob("reports/dimensionality_reduction_report_*.txt")
    for report_file in individual_reports:
        with open(report_file, "r") as f:
            content = f.read()
            results["individual_reports"].append(
                {"file": report_file, "content": content}
            )

    print(f"✅ Found {len(results['individual_reports'])} individual reports")

    # Look for visualizations
    viz_files = glob.glob("reports/*.png")
    results["visualizations"] = [
        f for f in viz_files if "dimensionality" in f or "factor_contributions" in f
    ]

    print(f"✅ Found {len(results['visualizations'])} visualizations")

    return results


def create_html_report(results):
    """
    Create HTML report content.
    """
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dimensionality Reduction Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px dates(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 25px;
            }}
            .summary {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .metric {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                margin: 5px;
                font-weight: bold;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .highlight {{
                background-color: #f39c12;
                color: white;
                padding: 2px 5px;
                border-radius: 3px;
            }}
            .visualization {{
                text-align: center;
                margin: 20px 0;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .method-comparison {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }}
            .method-card {{
                flex: 1;
                min-width: 250px;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Advanced Dimensionality Reduction Report</h1>
            
            <div class="summary">
                <h3>📊 Executive Summary</h3>
                <p><strong>Generated:</strong> {results['timestamp']}</p>
                <p><strong>Methods Compared:</strong> {len(results['comparison_results']) if results['comparison_results'] is not None else 'N/A'}</p>
                <p><strong>Individual Reports:</strong> {len(results['individual_reports'])}</p>
                <p><strong>Visualizations:</strong> {len(results['visualizations'])}</p>
            </div>
            
            <h2>🎯 Methodology Overview</h2>
            <p>This report compares different dimensionality reduction methods for quantitative trading factor compression:</p>
            <ul>
                <li><strong>Autoencoder + SHAP Distillation:</strong> Advanced nonlinear compression with interpretability</li>
                <li><strong>Principal Component Analysis (PCA):</strong> Linear dimensionality reduction</li>
                <li><strong>Feature Selection:</strong> Traditional statistical feature selection</li>
                <li><strong>Mutual Information Selection:</strong> Information-theoretic feature selection</li>
            </ul>
            
            {create_comparison_section(results)}
            {create_individual_reports_section(results)}
            {create_visualizations_section(results)}
            {create_recommendations_section(results)}
            
            <div class="footer">
                <p>Generated by Advanced Dimensionality Reduction Pipeline</p>
                <p>Using Autoencoder + SHAP Distillation Technology</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html


def create_comparison_section(results):
    """
    Create comparison section HTML.
    """
    if results["comparison_results"] is None:
        return "<h2>📊 Method Comparison</h2><p>No comparison results available.</p>"

    df = results["comparison_results"]

    # Sort by R² score
    df_sorted = df.sort_values("R²", ascending=False)

    html = f"""
    <h2>📊 Method Comparison</h2>
    
    <div class="method-comparison">
    """

    for _, row in df_sorted.iterrows():
        html += f"""
        <div class="method-card">
            <h3>{row['Method']}</h3>
            <p><span class="metric">R²: {row['R²']:.4f}</span></p>
            <p><span class="metric">RMSE: {row['RMSE']:.4f}</span></p>
            <p><span class="metric">Time: {row['Training Time (s)']:.2f}s</span></p>
            <p><span class="metric">Compression: {row['Compression Ratio']:.1f}x</span></p>
            <p><strong>Interpretability:</strong> {row['Interpretability']}</p>
        </div>
        """

    html += "</div>"

    # Add detailed table
    html += f"""
    <h3>📋 Detailed Results Table</h3>
    <table>
        <tr>
            <th>Method</th>
            <th>R² Score</th>
            <th>RMSE</th>
            <th>MAE</th>
            <th>Training Time (s)</th>
            <th>Compression Ratio</th>
            <th>Interpretability</th>
        </tr>
    """

    for _, row in df_sorted.iterrows():
        best_r2 = df_sorted.iloc[0]["R²"]
        is_best = row["R²"] == best_r2
        highlight_class = "highlight" if is_best else ""

        html += f"""
        <tr>
            <td class="{highlight_class}">{row['Method']}</td>
            <td class="{highlight_class}">{row['R²']:.4f}</td>
            <td>{row['RMSE']:.4f}</td>
            <td>{row['MAE']:.4f}</td>
            <td>{row['Training Time (s)']:.2f}</td>
            <td>{row['Compression Ratio']:.1f}x</td>
            <td>{row['Interpretability']}</td>
        </tr>
        """

    html += "</table>"

    return html


def create_individual_reports_section(results):
    """
    Create individual reports section HTML.
    """
    if not results["individual_reports"]:
        return "<h2>📄 Individual Reports</h2><p>No individual reports available.</p>"

    html = "<h2>📄 Individual Reports</h2>"

    for report in results["individual_reports"]:
        html += f"""
        <h3>{os.path.basename(report['file'])}</h3>
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace;">
{report['content']}
        </div>
        """

    return html


def create_visualizations_section(results):
    """
    Create visualizations section HTML.
    """
    if not results["visualizations"]:
        return "<h2>📊 Visualizations</h2><p>No visualizations available.</p>"

    html = "<h2>📊 Visualizations</h2>"

    for viz_file in results["visualizations"]:
        html += f"""
        <div class="visualization">
            <h3>{os.path.basename(viz_file)}</h3>
            <img src="{viz_file}" alt="Visualization">
        </div>
        """

    return html


def create_recommendations_section(results):
    """
    Create recommendations section HTML.
    """
    html = """
    <h2>🎯 Recommendations</h2>
    
    <div class="summary">
        <h3>Based on the comparison results:</h3>
        <ul>
            <li><strong>For Maximum Interpretability:</strong> Use Feature Selection methods (SelectKBest, Mutual Information)</li>
            <li><strong>For Balanced Performance:</strong> Use Autoencoder + SHAP Distillation</li>
            <li><strong>For Speed:</strong> Use PCA or Feature Selection methods</li>
            <li><strong>For Nonlinear Relationships:</strong> Use Autoencoder + SHAP Distillation</li>
            <li><strong>For Production Deployment:</strong> Use distilled linear combinations from Autoencoder + SHAP</li>
        </ul>
    </div>
    
    <h3>🚀 Next Steps</h3>
    <ol>
        <li>Review factor contributions and weights</li>
        <li>Integrate top factors into trading strategies</li>
        <li>Monitor factor performance over time</li>
        <li>Retrain models periodically with new data</li>
        <li>Consider ensemble approaches combining multiple methods</li>
    </ol>
    """

    return html


if __name__ == "__main__":
    # Generate report
    report_path = generate_dim_reduction_report()

    print(f"\n🎉 Dimensionality reduction report generated successfully!")
    print(f"📄 Report location: {report_path}")
    print(f"🌐 Open the report in your web browser to view the full analysis.")
