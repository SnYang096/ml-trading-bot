"""Utilities for building comprehensive dimensionality reports."""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _format_float(val, digits: int = 4) -> str:
    """Format float value for HTML report display."""
    try:
        if val is None or (isinstance(val, float) and
                           (np.isnan(val) or np.isinf(val))):
            return "NA"
        return f"{val:.{digits}f}"
    except Exception:
        return str(val)


def _get_oos_period_html(oos_metrics: Dict, oos_months: int) -> str:
    """Generate HTML row for OOS test period."""
    oos_period = oos_metrics.get('oos_period', {})
    start = oos_period.get('start', 'N/A')
    end = oos_period.get('end', 'N/A')
    start_str = start.split('T')[0] if start and start != 'N/A' else 'N/A'
    end_str = end.split('T')[0] if end and end != 'N/A' else 'N/A'
    return f"<tr><th>OOS Test Period</th><td>{start_str} to {end_str} ({oos_months} months)</td></tr>"


def _build_feature_importance_table(info: Dict) -> str:
    """Build feature importance table HTML."""
    feature_importance = info.get('feature_importance', [])
    if not feature_importance:
        return ""

    # Get top 20 features
    top_features = feature_importance[:20]

    rows = []
    for feat in top_features:
        feat_name = feat.get('feature', 'N/A')
        importance_gain = _format_float(feat.get('importance_gain', 0), 2)
        importance_split = feat.get('importance_split', 0)
        rows.append(f"""
            <tr>
                <td>{feat_name}</td>
                <td>{importance_gain}</td>
                <td>{importance_split:,}</td>
            </tr>""")

    return f"""
        <h2>Feature Importance (Top 20)</h2>
        <div class="explanation">
            <h3>Feature Importance Explanation</h3>
            <p>Feature importance measures how much each feature contributes to the model's predictions.</p>
            <ul>
                <li><strong>Importance (Gain):</strong> The average gain (improvement in accuracy) when the feature is used for splitting. Higher is better.</li>
                <li><strong>Importance (Split):</strong> The number of times the feature is used for splitting in the tree. Higher indicates more usage.</li>
            </ul>
        </div>
        <table>
            <tr>
                <th>Feature</th>
                <th>Importance (Gain)</th>
                <th>Importance (Split)</th>
            </tr>
            {"".join(rows)}
        </table>"""


def _build_rolling_feature_importance_section(summary: Dict) -> str:
    """Build aggregated feature-importance section for rolling reports."""
    feature_map = summary.get("feature_importance", {})
    if not feature_map:
        return ""

    label_map = {
        "classification": "Directional Classification",
        "return": "Return Regression",
        "volatility": "Volatility Regression",
    }

    sections: list[str] = []
    for key, label in label_map.items():
        data = feature_map.get(key)
        if not data:
            continue
        rows = []
        for rank, item in enumerate(data, start=1):
            feat = item.get("feature", "N/A")
            importance = _format_float(item.get("importance", 0.0), 6)
            rows.append(
                f"<tr><td>{rank}</td><td>{feat}</td><td>{importance}</td></tr>"
            )
        if rows:
            sections.append(f"""
            <h3>{label}</h3>
            <table>
                <tr><th>Rank</th><th>Feature</th><th>Importance (Gain)</th></tr>
                {''.join(rows)}
            </table>
            """)

    if not sections:
        return ""

    return f"""
    <h2>📊 Rolling Feature Importance (Top 100 per Model)</h2>
    <div class="explanation">
        <p>基于所有滚动窗口累积的 LightGBM gain，展示各模型贡献最大的特征（每类最多 100 个）。</p>
    </div>
    {''.join(sections)}
    """


def _build_oos_table(oos_metrics: Dict, oos_months: int) -> str:
    """Build OOS test results table HTML."""
    if not oos_metrics or oos_months <= 0:
        return ""

    stage1 = oos_metrics.get('stage1', {})
    stage1_acc = _format_float(stage1.get('accuracy'), 4)
    stage1_precision = _format_float(stage1.get('precision'), 4)
    stage1_recall = _format_float(stage1.get('recall'), 4)
    stage1_f1 = _format_float(stage1.get('f1'), 4)
    stage1_auc = _format_float(stage1.get('auc'), 4)
    stage1_pr_auc = _format_float(stage1.get('pr_auc'), 4)
    stage1_samples = stage1.get('samples', 0)

    # Confusion matrix
    cm = stage1.get('confusion_matrix', [])
    cm_html = ""
    if cm and len(cm) == 2 and len(cm[0]) == 2:
        tn, fp = cm[0]
        fn, tp = cm[1]
        cm_html = f"""
            <h3>Confusion Matrix</h3>
            <table style="margin: 10px 0;">
                <tr>
                    <th></th>
                    <th>Predicted: 0</th>
                    <th>Predicted: 1</th>
                </tr>
                <tr>
                    <th>Actual: 0</th>
                    <td>{tn}</td>
                    <td>{fp}</td>
                </tr>
                <tr>
                    <th>Actual: 1</th>
                    <td>{fn}</td>
                    <td>{tp}</td>
                </tr>
            </table>
            <p><strong>TN (True Negative):</strong> {tn}, <strong>FP (False Positive):</strong> {fp}, 
            <strong>FN (False Negative):</strong> {fn}, <strong>TP (True Positive):</strong> {tp}</p>"""

    # Best threshold
    best_threshold = _format_float(stage1.get('best_threshold'), 3)
    best_threshold_f1 = _format_float(stage1.get('best_threshold_f1'), 4)

    # Quality check
    quality_check = stage1.get('quality_check', {})
    quality_check_passed = quality_check.get('passed', True)
    quality_issues = quality_check.get('issues', [])
    quality_check_html = ""
    if quality_issues or not quality_check_passed:
        if quality_check_passed:
            quality_check_html = '<div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 20px 0;"><strong>✅ Model Quality Check: PASSED</strong></div>'
        else:
            quality_check_html = '<div style="background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; margin: 20px 0;"><strong>❌ Model Quality Check: FAILED</strong><ul>'
            for issue in quality_issues:
                quality_check_html += f'<li>{issue}</li>'
            quality_check_html += '</ul></div>'
    elif quality_check_passed:
        quality_check_html = '<div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 20px 0;"><strong>✅ Model Quality Check: PASSED</strong></div>'

    stage2_rows = ""
    if oos_metrics.get('stage2'):
        stage2_rmse = _format_float(
            oos_metrics.get('stage2', {}).get('rmse'), 6)
        stage2_mse = _format_float(oos_metrics.get('stage2', {}).get('mse'), 8)
        stage2_samples = oos_metrics.get('stage2', {}).get('samples', 0)
        stage2_rows = f"""
            <h3>Stage2: Regression Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Samples</th>
                </tr>
                <tr>
                    <td>RMSE</td>
                    <td>{stage2_rmse}</td>
                    <td>{stage2_samples:,}</td>
                </tr>
                <tr>
                    <td>MSE</td>
                    <td>{stage2_mse}</td>
                    <td>{stage2_samples:,}</td>
                </tr>
            </table>"""

    return f"""
        <h2>Out-of-Sample (OOS) Test Results</h2>
        <div class="explanation">
            <h3>OOS Testing Explanation</h3>
            <p>The last {oos_months} months of data were reserved for out-of-sample testing. 
            This provides an unbiased evaluation of model performance on unseen data, 
            simulating real-world deployment scenarios.</p>
        </div>
        {quality_check_html}
        <h3>Stage1: Classification Metrics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Explanation</th>
            </tr>
            <tr>
                <td><strong>Accuracy</strong></td>
                <td>{stage1_acc}</td>
                <td>Overall classification accuracy (0-1, higher is better)</td>
            </tr>
            <tr>
                <td><strong>Precision</strong></td>
                <td>{stage1_precision if stage1_precision != 'NA' else 'N/A'}</td>
                <td>控制误开仓（预测为做多时，真的做多比例）</td>
            </tr>
            <tr>
                <td><strong>Recall</strong></td>
                <td>{stage1_recall if stage1_recall != 'NA' else 'N/A'}</td>
                <td>抓住行情能力（实际该做多时，模型抓到比例）</td>
            </tr>
            <tr>
                <td><strong>F1 Score</strong></td>
                <td>{stage1_f1 if stage1_f1 != 'NA' else 'N/A'}</td>
                <td>综合指标（Precision和Recall的调和平均，推荐阈值：F1 &gt; 0.3）</td>
            </tr>
            <tr>
                <td><strong>AUC-ROC</strong></td>
                <td>{stage1_auc if stage1_auc != 'NA' else 'N/A'}</td>
                <td>区分能力（对阈值不敏感，推荐阈值：AUC &gt; 0.6）</td>
            </tr>
            <tr>
                <td><strong>PR-AUC</strong></td>
                <td>{stage1_pr_auc if stage1_pr_auc != 'NA' else 'N/A'}</td>
                <td>精确率-召回率曲线下面积（更适合不平衡数据）</td>
            </tr>
            <tr>
                <td><strong>Best Threshold (F1)</strong></td>
                <td>{best_threshold if best_threshold != 'NA' else 'N/A'}</td>
                <td>最优分类阈值（最大化F1 Score，当前使用0.5）</td>
            </tr>
            <tr>
                <td><strong>Best F1 (at threshold)</strong></td>
                <td>{best_threshold_f1 if best_threshold_f1 != 'NA' else 'N/A'}</td>
                <td>在最优阈值下的F1 Score</td>
            </tr>
            <tr>
                <td><strong>Samples</strong></td>
                <td>{stage1_samples:,}</td>
                <td>OOS test samples</td>
            </tr>
        </table>
        {cm_html if cm_html else ""}
        {stage2_rows if stage2_rows else ""}
        """


def _format_price(val) -> str:
    """Format price value with thousands separator."""
    try:
        if val is None or (isinstance(val, float) and
                           (np.isnan(val) or np.isinf(val))):
            return "NA"
        return f"{val:,.2f}"
    except Exception:
        return str(val)


def generate_comprehensive_report() -> str:
    print("📋 Generating Comprehensive Dimensionality Training Report")
    print("=" * 60)

    os.makedirs("reports", exist_ok=True)

    results = collect_all_results()
    html_content = create_html_report(results)

    report_path = "reports/dimensionality_comprehensive_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Comprehensive report generated: {report_path}")
    return report_path


def collect_all_results() -> Dict[str, any]:
    results: Dict[str, any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_engineering_results": [],
        "rolling_training_results": [],
        "production_training_results": [],
        "integration_results": [],
        "summary_statistics": {},
    }

    print("📊 Collecting feature engineering results...")
    feature_files = glob.glob("results/feature_engineering_*.json")
    for file in feature_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                results["feature_engineering_results"].append(data)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Error reading {file}: {exc}")

    print("📊 Collecting rolling training results...")
    rolling_dirs = glob.glob("results/rolling_dim_*")
    for dir_path in rolling_dirs:
        summary_file = os.path.join(dir_path, "summary_report.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r") as f:
                    data = json.load(f)
                    results["rolling_training_results"].append(data)
            except Exception as exc:  # noqa: BLE001
                print(f"❌ Error reading {summary_file}: {exc}")

    print("📊 Collecting production training results...")
    production_dirs = glob.glob("results/production_dimensionality_*")
    for dir_path in production_dirs:
        results_file = os.path.join(dir_path, "production_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                    results["production_training_results"].append(data)
            except Exception as exc:  # noqa: BLE001
                print(f"❌ Error reading {results_file}: {exc}")

    print("📊 Collecting integration results...")
    integration_dirs = glob.glob("results/integration_*")
    for dir_path in integration_dirs:
        report_file = os.path.join(dir_path, "integration_report.json")
        if os.path.exists(report_file):
            try:
                with open(report_file, "r") as f:
                    data = json.load(f)
                    results["integration_results"].append(data)
            except Exception as exc:  # noqa: BLE001
                print(f"❌ Error reading {report_file}: {exc}")

    results["summary_statistics"] = calculate_summary_statistics(results)

    print(
        f"✅ Collected {len(results['feature_engineering_results'])} feature engineering results"
    )
    print(
        f"✅ Collected {len(results['rolling_training_results'])} rolling training results"
    )
    print(
        f"✅ Collected {len(results['production_training_results'])} production training results"
    )
    print(
        f"✅ Collected {len(results['integration_results'])} integration results"
    )

    return results


def calculate_summary_statistics(results: Dict[str, any]) -> Dict[str, any]:
    stats = {
        "total_experiments": 0,
        "average_compression_ratio": 0,
        "average_performance_improvement": 0,
        "best_performing_method": "N/A",
        "total_features_processed": 0,
    }

    if results["feature_engineering_results"]:
        total_features = sum(
            r.get("total_features", 0)
            for r in results["feature_engineering_results"])
        filtered_features = sum(
            r.get("filtered_features", 0)
            for r in results["feature_engineering_results"])
        stats["total_features_processed"] = total_features
        if total_features > 0 and filtered_features > 0:
            stats[
                "average_compression_ratio"] = total_features / filtered_features

    if results["production_training_results"]:
        performance_changes = []
        for result in results["production_training_results"]:
            performance = result.get("performance", {})
            if "performance_change" in performance:
                performance_changes.append(performance["performance_change"])

        if performance_changes:
            stats["average_performance_improvement"] = float(
                np.mean(performance_changes))
            stats["best_performing_method"] = "LightGBM"

    stats["total_experiments"] = (len(results["feature_engineering_results"]) +
                                  len(results["rolling_training_results"]) +
                                  len(results["production_training_results"]) +
                                  len(results["integration_results"]))

    return stats


def create_html_report(results: Dict[str, any]) -> str:
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dimensionality Training Comprehensive Report</title>
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
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
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
            .summary {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
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
            .section {{
                margin: 30px 0;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Dimensionality Training Comprehensive Report</h1>
            <div class="summary">
                <h3>📊 Executive Summary</h3>
                <p><strong>Generated:</strong> {results['timestamp']}</p>
                <p><strong>Total Experiments:</strong> {results['summary_statistics']['total_experiments']}</p>
                <p><strong>Average Compression Ratio:</strong> {results['summary_statistics']['average_compression_ratio']:.1f}x</p>
                <p><strong>Average Performance Improvement:</strong> {results['summary_statistics']['average_performance_improvement']:.3f}</p>
                <p><strong>Best Method:</strong> {results['summary_statistics']['best_performing_method']}</p>
            </div>
            {create_feature_engineering_section(results)}
            {create_rolling_training_section(results)}
            {create_production_training_section(results)}
            {create_integration_section(results)}
            {create_recommendations_section(results)}
            <div class="footer" style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
                <p>Generated by Dimensionality Training System</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html


def create_feature_engineering_section(results: Dict[str, any]) -> str:
    if not results["feature_engineering_results"]:
        return "<h2>🔧 Feature Engineering</h2><p>No feature engineering results available.</p>"

    html = "<h2>🔧 Feature Engineering Results</h2>"
    for i, result in enumerate(results["feature_engineering_results"]):
        html += f"""
        <div class="section">
            <h3>Experiment {i+1}</h3>
            <p><strong>Total Features:</strong> {result.get('total_features', 'N/A')}</p>
            <p><strong>Filtered Features:</strong> {result.get('filtered_features', 'N/A')}</p>
            <p><strong>Timestamp:</strong> {result.get('timestamp', 'N/A')}</p>
        </div>
        """

    return html


def create_rolling_training_section(results: Dict[str, any]) -> str:
    if not results["rolling_training_results"]:
        return "<h2>🚀 Rolling Training</h2><p>No rolling training results available.</p>"

    html = "<h2>🚀 Rolling Training Results</h2>"
    for result in results["rolling_training_results"]:
        stats = result.get("summary_statistics", {})
        html += f"""
        <div class="section">
            <h3>Symbol: {result.get('symbol', 'N/A')}</h3>
            <p><strong>Training R²:</strong> {stats.get('training_compressed_r2', 'N/A')}</p>
            <p><strong>Test R²:</strong> {stats.get('average_test_r2', 'N/A')}</p>
            <p><strong>Improvement:</strong> {stats.get('training_improvement', 'N/A')}</p>
            <p><strong>Compression Ratio:</strong> {stats.get('compression_ratio', 'N/A')}</p>
        </div>
        """

    return html


def create_production_training_section(results: Dict[str, any]) -> str:
    if not results["production_training_results"]:
        return "<h2>🏭 Production Training</h2><p>No production training results available.</p>"

    html = "<h2>🏭 Production Training Results</h2>"
    for i, result in enumerate(results["production_training_results"]):
        performance = result.get("performance", {})
        html += f"""
        <div class="section">
            <h3>Production Model {i+1}</h3>
            <p><strong>Compression Ratio:</strong> {result.get('data_info', {}).get('compression_ratio', 'N/A')}</p>
            <p><strong>Original R²:</strong> {performance.get('original_features', {}).get('r2', 'N/A')}</p>
            <p><strong>Compressed R²:</strong> {performance.get('compressed_features', {}).get('r2', 'N/A')}</p>
            <p><strong>Performance Change:</strong> {performance.get('performance_change', 'N/A')}</p>
        </div>
        """

    return html


def create_integration_section(results: Dict[str, any]) -> str:
    if not results["integration_results"]:
        return "<h2>🔗 Integration</h2><p>No integration results available.</p>"

    html = "<h2>🔗 Integration Results</h2>"
    for i, result in enumerate(results["integration_results"]):
        perf = result.get("new_data_performance", {})
        html += f"""
        <div class="section">
            <h3>Integration Test {i+1}</h3>
            <p><strong>New Data R²:</strong> {perf.get('r2', 'N/A')}</p>
            <p><strong>RMSE:</strong> {perf.get('rmse', 'N/A')}</p>
            <p><strong>MAE:</strong> {perf.get('mae', 'N/A')}</p>
        </div>
        """

    return html


def write_html_report(results: Dict, html_path: str) -> None:
    """Write HTML report for a single dimensionality reduction experiment.
    
    This generates a detailed report showing:
    - 4-stage comparison (All Features → IC-Filtered → Representatives → Compressed)
    - Performance metrics (R², RMSE, MAE)
    - Financial metrics (Sharpe Ratio, Total Return, Max Drawdown, etc.)
    - Training diagnostics
    """
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    ts_start = results.get("timestamp_start", results.get("timestamp", "-"))
    ts_end = results.get("timestamp_end", "-")
    # Prefer training date range if available, otherwise show runtime timestamps
    train_start_date = results.get("train_start_date")
    train_end_date = results.get("train_end_date")
    if train_start_date and train_end_date:
        # Format: YYYYMMDD -> YYYY-MM-DD
        date_range_str = (
            f"Training Data: {train_start_date[:4]}-{train_start_date[4:6]}-"
            f"{train_start_date[6:8]} to {train_end_date[:4]}-{train_end_date[4:6]}-"
            f"{train_end_date[6:8]}")
        runtime_str = f"Run Time: {ts_start} to {ts_end}"
    else:
        date_range_str = f"Start: {ts_start}  |  End: {ts_end}"
        runtime_str = ""
    d = results.get("data_info", {})
    p = results.get("performance", {})
    train_info = results.get("training_info", {})
    multi_horizon_results = results.get("multi_horizon_results", {})
    task_type = results.get("task_type", "classification_multiclass")
    selection_metric = results.get(
        "selection_metric",
        results.get("selection", {}).get("metric", "composite"))
    label_threshold = results.get("label_threshold", None)
    artifacts = {
        "top_factors":
        d.get("top_factors_path") or results.get("top_factors_path"),
        "representatives":
        d.get("representatives_path") or results.get("representatives_path"),
        "autoencoder":
        d.get("autoencoder_path") or results.get("autoencoder_path")
        or "results/production_autoencoder.pth",
    }

    # Support both old format (original/compressed) and new 4-stage format
    stage1 = p.get("stage1_all_features", p.get("original_features", {}))
    stage2 = p.get("stage2_ic_filtered", {})
    stage3 = p.get("stage3_representatives", {})
    stage4 = p.get("stage4_compressed", p.get("compressed_features", {}))

    # Legacy support
    orig = p.get("original_features", stage1)
    comp = p.get("compressed_features", stage4)
    orig_val = p.get("original_features_val", {})
    comp_val = p.get("compressed_features_val", {})

    # Get delta comparisons
    stage2_vs_1 = p.get("stage2_vs_stage1", {})
    stage3_vs_2 = p.get("stage3_vs_stage2", {})
    stage4_vs_3 = p.get("stage4_vs_stage3", {})
    delta_r2 = p.get("performance_change", stage4_vs_3.get("delta_r2"))

    has_4_stages = stage2 and stage3

    conclusion = ("Dimensionality reduction appears beneficial." if
                  (delta_r2 is not None and delta_r2 > 0) else
                  "Dimensionality reduction is not beneficial under this run.")

    # Extract financial metrics
    stage1_fin = stage1.get("financial_metrics", {})
    stage2_fin = stage2.get("financial_metrics", {})
    stage3_fin = stage3.get("financial_metrics", {})
    stage4_fin = stage4.get("financial_metrics", {})
    orig_fin = orig.get("financial_metrics", stage1_fin)
    comp_fin = comp.get("financial_metrics", stage4_fin)
    orig_val_fin = orig_val.get("financial_metrics", {})
    comp_val_fin = comp_val.get("financial_metrics", {})

    # Optional grid table
    grid_rows = []
    grid = results.get("grid_search", [])
    if grid:
        for row in grid:
            grid_rows.append(
                f"<tr><td>{row.get('encoding_dim', '-')}</td>"
                f"<td>{_format_float(row.get('r2_stage3_reps') or row.get('r2_original'))}</td>"
                f"<td>{_format_float(row.get('r2_compressed'))}</td>"
                f"<td>{_format_float(row.get('delta_r2'))}</td>"
                f"<td>{_format_float(row.get('rmse_stage3_reps') or row.get('rmse_original'))}</td>"
                f"<td>{_format_float(row.get('rmse_compressed'))}</td>"
                "</tr>")

    # Build HTML content
    html = _build_html_report_content(
        date_range_str, runtime_str, d, stage1, stage2, stage3, stage4,
        has_4_stages, orig, comp, delta_r2, stage1_fin, stage2_fin, stage3_fin,
        stage4_fin, orig_fin, comp_fin, orig_val_fin, comp_val_fin, train_info,
        grid_rows, conclusion, stage2_vs_1, stage3_vs_2, stage4_vs_3,
        multi_horizon_results, task_type, selection_metric, label_threshold,
        artifacts)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📝 HTML report written to: {html_path}")


def _build_html_report_content(
    date_range_str: str,
    runtime_str: str,
    d: Dict,
    stage1: Dict,
    stage2: Dict,
    stage3: Dict,
    stage4: Dict,
    has_4_stages: bool,
    orig: Dict,
    comp: Dict,
    delta_r2,
    stage1_fin: Dict,
    stage2_fin: Dict,
    stage3_fin: Dict,
    stage4_fin: Dict,
    orig_fin: Dict,
    comp_fin: Dict,
    orig_val_fin: Dict,
    comp_val_fin: Dict,
    train_info: Dict,
    grid_rows: list,
    conclusion: str,
    stage2_vs_1: Dict,
    stage3_vs_2: Dict,
    stage4_vs_3: Dict,
    multi_horizon_results: Dict = None,
    task_type: str = "classification_multiclass",
    selection_metric: str | None = None,
    label_threshold: float | None = None,
    artifacts: Dict | None = None,
) -> str:
    """Build HTML content string for the report."""
    # Build conditional 4-stage comparison table
    stage_comparison_table = ""
    if has_4_stages:
        if task_type.startswith("classification"):
            # Show classification-centric table
            stage_comparison_table = (
                f'<h2>4-Stage Comparison (Test Set)</h2><table>'
                f'<tr><th>Stage</th><th>Features</th><th>Directional Win Rate</th><th>Active Ratio</th></tr>'
                f'<tr><td>Stage 1: All Features</td><td>{d.get("stage1_all_features", "-")}</td>'
                f'<td>{_format_float(stage1_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage1_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'<tr><td>Stage 2: IC-Filtered</td><td>{d.get("stage2_ic_filtered", "-")}</td>'
                f'<td>{_format_float(stage2_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage2_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'<tr><td>Stage 3: Representatives</td><td>{d.get("stage3_representatives", "-")}</td>'
                f'<td>{_format_float(stage3_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage3_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'<tr><td>Stage 4: Compressed</td><td>{d.get("compressed_dimensions", "-")}</td>'
                f'<td>{_format_float(stage4_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage4_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'</table>')
        else:
            stage_comparison_table = (
                f'<h2>4-Stage Comparison (Test Set)</h2><table>'
                f'<tr><th>Stage</th><th>Features</th><th>R²</th><th>RMSE</th><th>MAE</th><th>vs Previous (ΔR²)</th></tr>'
                f'<tr><td>Stage 1: All Features</td><td>{d.get("stage1_all_features", "-")}</td>'
                f'<td>{_format_float(stage1.get("r2"))}</td><td>{_format_float(stage1.get("rmse"))}</td>'
                f'<td>{_format_float(stage1.get("mae"))}</td><td>-</td></tr>'
                f'<tr><td>Stage 2: IC-Filtered</td><td>{d.get("stage2_ic_filtered", "-")}</td>'
                f'<td>{_format_float(stage2.get("r2"))}</td><td>{_format_float(stage2.get("rmse"))}</td>'
                f'<td>{_format_float(stage2.get("mae"))}</td><td>{_format_float(stage2_vs_1.get("delta_r2"))}</td></tr>'
                f'<tr><td>Stage 3: Representatives</td><td>{d.get("stage3_representatives", "-")}</td>'
                f'<td>{_format_float(stage3.get("r2"))}</td><td>{_format_float(stage3.get("rmse"))}</td>'
                f'<td>{_format_float(stage3.get("mae"))}</td><td>{_format_float(stage3_vs_2.get("delta_r2"))}</td></tr>'
                f'<tr><td>Stage 4: Compressed</td><td>{d.get("compressed_dimensions", "-")}</td>'
                f'<td>{_format_float(stage4.get("r2"))}</td><td>{_format_float(stage4.get("rmse"))}</td>'
                f'<td>{_format_float(stage4.get("mae"))}</td><td>{_format_float(stage4_vs_3.get("delta_r2"))}</td></tr>'
                f'</table>')

    # Build conditional 4-stage financial metrics tables
    val_4stage_fin_table = ""
    test_4stage_fin_table = ""
    if has_4_stages and stage1_fin and stage2_fin:
        val_4stage_fin_table = (
            f'<h2>Financial Metrics - 4-Stage Comparison (Validation Set)</h2>'
            f'<table><tr><th>Metric</th><th>Stage 1: All</th><th>Stage 2: IC</th>'
            f'<th>Stage 3: Reps</th><th>Stage 4: AE</th></tr>'
            f'<tr><td>Sharpe Ratio</td><td>{_format_float(stage1_fin.get("sharpe_ratio"))}</td>'
            f'<td>{_format_float(stage2_fin.get("sharpe_ratio"))}</td>'
            f'<td>{_format_float(stage3_fin.get("sharpe_ratio"))}</td>'
            f'<td>{_format_float(stage4_fin.get("sharpe_ratio"))}</td></tr>'
            f'<tr><td>Total Return</td><td>{_format_float(stage1_fin.get("total_return"))}</td>'
            f'<td>{_format_float(stage2_fin.get("total_return"))}</td>'
            f'<td>{_format_float(stage3_fin.get("total_return"))}</td>'
            f'<td>{_format_float(stage4_fin.get("total_return"))}</td></tr>'
            f'<tr><td>Max Drawdown</td><td>{_format_float(stage1_fin.get("max_drawdown"))}</td>'
            f'<td>{_format_float(stage2_fin.get("max_drawdown"))}</td>'
            f'<td>{_format_float(stage3_fin.get("max_drawdown"))}</td>'
            f'<td>{_format_float(stage4_fin.get("max_drawdown"))}</td></tr>'
            f'<tr><td>Win Rate</td><td>{_format_float(stage1_fin.get("win_rate") * 100, 2)}%</td>'
            f'<td>{_format_float(stage2_fin.get("win_rate") * 100, 2)}%</td>'
            f'<td>{_format_float(stage3_fin.get("win_rate") * 100, 2)}%</td>'
            f'<td>{_format_float(stage4_fin.get("win_rate") * 100, 2)}%</td></tr></table>'
        )
        test_4stage_fin_table = val_4stage_fin_table.replace(
            "Validation Set", "Test Set")

    # Helper to safely get financial metric
    def safe_fin_metric(metric: str, use_val: bool = False) -> str:
        val_dict = orig_val_fin if use_val else orig_fin
        comp_dict = comp_val_fin if use_val else comp_fin
        if not val_dict:
            val_dict = orig_fin
        if not comp_dict:
            comp_dict = comp_fin
        orig_val = val_dict.get(metric, 0) or 0
        comp_val = comp_dict.get(metric, 0) or 0
        return _format_float(orig_val), _format_float(comp_val), _format_float(
            comp_val - orig_val)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/><title>Dimensionality Reduction Comparison</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;margin-top:16px;width:100%;max-width:900px}}th,td{{border:1px solid #ddd;padding:8px 10px;text-align:left}}th{{background:#f7f7f7}}.bad{{color:#b00020;font-weight:600}}.good{{color:#0a7c2f;font-weight:600}}.warn{{color:#b36b00;font-weight:600}}</style>
</head><body>
<h1>Dimensionality Reduction Comparison</h1>
<div>{date_range_str}</div>
{f'<div style="font-size:0.9em;color:#666;margin-top:4px;">{runtime_str}</div>' if runtime_str else ''}

<h2>Data Summary</h2>
<table>
<tr><th>Stage</th><th>Features</th><th>Description</th></tr>
<tr><td>Stage 1: All Features</td><td>{d.get('stage1_all_features', d.get('original_features_count','-'))}</td><td>All original features after missing/stability filter</td></tr>
{f'<tr><td>Stage 2: IC-Filtered</td><td>{d.get("stage2_ic_filtered", "-")}</td><td>Top features by |IC| (Spearman correlation)</td></tr>' if d.get('stage2_ic_filtered') else ''}
{f'<tr><td>Stage 3: Representatives</td><td>{d.get("stage3_representatives", "-")}</td><td>Correlation-filtered representative features (60-100)</td></tr>' if d.get('stage3_representatives') else ''}
<tr><td>Stage 4: Compressed</td><td>{d.get('compressed_dimensions','-')}</td><td>Compressed feature dimensions</td></tr>
<tr><th colspan="3">Summary</th></tr>
<tr><td>Final Compression Ratio</td><td colspan="2">{_format_float(d.get('compression_ratio'),2)}x ({d.get('original_features_count','-')} → {d.get('compressed_dimensions','-')})</td></tr>
<tr><td>Samples (train/val/test)</td><td colspan="2">{d.get('training_samples','-')} / {d.get('validation_samples','-')} / {d.get('test_samples','-')}</td></tr>
</table>

<div class="section" style="background:#f8f9fa">
<h3>Run Configuration</h3>
<table>
<tr><th>Task Type</th><td>{task_type}</td></tr>
<tr><th>Selection Metric</th><td>{selection_metric or '-'}</td></tr>
{f'<tr><th>Label Threshold</th><td>{_format_float(label_threshold,6)}</td></tr>' if label_threshold is not None else ''}
<tr><th>Artifacts</th><td>
{('Top Factors: ' + (artifacts.get('top_factors') or '-')) if artifacts else ''}<br/>
{('Representatives: ' + (artifacts.get('representatives') or '-')) if artifacts else ''}<br/>

</td></tr>
</table>
</div>

{stage_comparison_table}

{('<h2>Classification Performance (Test Set)</h2>'
  '<table>'
  '<tr><th>Metric</th><th>Original</th><th>Compressed</th><th>Delta</th></tr>'
  f"<tr><td>Directional Win Rate</td><td>{_format_float(orig_fin.get('win_rate',0)*100,2)}%</td><td>{_format_float(comp_fin.get('win_rate',0)*100,2)}%</td><td>{_format_float((comp_fin.get('win_rate',0)-orig_fin.get('win_rate',0))*100,2)}%</td></tr>"
  f"<tr><td>Active Ratio</td><td>{_format_float(orig_fin.get('active_ratio',0)*100,2)}%</td><td>{_format_float(comp_fin.get('active_ratio',0)*100,2)}%</td><td>{_format_float((comp_fin.get('active_ratio',0)-orig_fin.get('active_ratio',0))*100,2)}%</td></tr>"
  '</table>') if task_type.startswith('classification') else (
  '<h2>Performance (Test Set)</h2>'
  '<table>'
  '<tr><th>Metric</th><th>Original</th><th>Compressed</th><th>Delta</th></tr>'
  f"<tr><td>R²</td><td>{_format_float(orig.get('r2'))}</td><td>{_format_float(comp.get('r2'))}</td><td>{_format_float(delta_r2)}</td></tr>"
  f"<tr><td>RMSE</td><td>{_format_float(orig.get('rmse'))}</td><td>{_format_float(comp.get('rmse'))}</td><td>{_format_float((comp.get('rmse') or 0)-(orig.get('rmse') or 0))}</td></tr>"
  f"<tr><td>MAE</td><td>{_format_float(orig.get('mae'))}</td><td>{_format_float(comp.get('mae'))}</td><td>{_format_float((comp.get('mae') or 0)-(orig.get('mae') or 0))}</td></tr>"
  '</table>')}

{val_4stage_fin_table}

<h2>Financial Metrics (Validation Set)</h2>
<table>
<tr><th>Metric</th><th>Original</th><th>Compressed</th><th>Delta</th></tr>
{_build_financial_metrics_table(orig_val_fin or orig_fin, comp_val_fin or comp_fin)}
</table>
{('<div class="section" style="color:#7a6">Note: Financial metrics appear as 0/NA when no trades were triggered (Active Ratio ~ 0). Consider binary labels or lower thresholds to increase signal activity.</div>' if (isinstance(orig_fin, dict) and (orig_fin.get('active_ratio', 0) == 0 and comp_fin.get('active_ratio', 0) == 0)) else '')}

{test_4stage_fin_table}

<h2>Financial Metrics (Test Set)</h2>
<table>
<tr><th>Metric</th><th>Original</th><th>Compressed</th><th>Delta</th></tr>
{_build_financial_metrics_table(orig_fin, comp_fin)}
</table>

{('<h2>Encoding Grid Results</h2>'
  '<table>'
  '<tr><th>ENCODING_DIM</th><th>R² Original</th><th>R² Compressed</th><th>ΔR²</th><th>RMSE Original</th><th>RMSE Compressed</th></tr>'
  f"{''.join(grid_rows)}"
  '</table>') if grid_rows else ''}

<h2>Training Diagnostics</h2>
<ul>

<li>LightGBM iterations (original/compressed): {train_info.get('lightgbm_original_iterations','-')} / {train_info.get('lightgbm_compressed_iterations','-')}</li>
</ul>

<h2>Conclusion</h2>
<p>{conclusion}</p>

{_build_multi_horizon_table(multi_horizon_results) if multi_horizon_results else ''}
</body></html>"""
    return html


def _build_multi_horizon_table(multi_horizon_results: Dict) -> str:
    """Build multi-horizon comparison table."""
    if not multi_horizon_results:
        return ""

    html = """
<h2>📊 Multi-Horizon Comparison</h2>
<div class="section">
    <p>This table compares the performance of all 4 stages across different prediction horizons (bars ahead).</p>
    <table>
        <thead>
            <tr>
                <th>Horizon</th>
                <th>Stage</th>
                <th>R²</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>Sharpe Ratio</th>
                <th>Total Return</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
            </tr>
        </thead>
        <tbody>
"""

    # Sort horizons numerically
    horizon_keys = sorted(
        [k for k in multi_horizon_results.keys() if k.startswith("horizon_")],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)

    for horizon_key in horizon_keys:
        horizon_num = horizon_key.split("_")[1]
        horizon_data = multi_horizon_results[horizon_key]

        stages = [
            ("Stage 1: All Features", "stage1_all_features"),
            ("Stage 2: IC-Filtered", "stage2_ic_filtered"),
            ("Stage 3: Representatives", "stage3_representatives"),
            ("Stage 4: Compressed", "stage4_compressed"),
        ]

        for stage_name, stage_key in stages:
            stage_perf = horizon_data.get(stage_key, {})
            if not stage_perf:
                continue

            fin_metrics = stage_perf.get("financial_metrics", {})

            html += f"""            <tr>
                <td><strong>{horizon_num} bars</strong></td>
                <td>{stage_name}</td>
                <td>{_format_float(stage_perf.get('r2'))}</td>
                <td>{_format_float(stage_perf.get('rmse'))}</td>
                <td>{_format_float(stage_perf.get('mae'))}</td>
                <td>{_format_float(fin_metrics.get('sharpe_ratio'))}</td>
                <td>{_format_float(fin_metrics.get('total_return'))}</td>
                <td>{_format_float(fin_metrics.get('max_drawdown'))}</td>
                <td>{_format_float(fin_metrics.get('win_rate'))}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
</div>
"""

    return html


def _build_financial_metrics_table(orig_fin: Dict, comp_fin: Dict) -> str:
    """Build financial metrics table rows."""

    def safe_get(key):
        o = orig_fin.get(key, 0) or 0
        c = comp_fin.get(key, 0) or 0
        return o, c, c - o

    rows = []
    o, c, d = safe_get('sharpe_ratio')
    rows.append(
        f'<tr><td>Sharpe Ratio</td><td>{_format_float(o)}</td><td>{_format_float(c)}</td><td>{_format_float(d)}</td></tr>'
    )
    o, c, d = safe_get('total_return')
    rows.append(
        f'<tr><td>Total Return</td><td>{_format_float(o)}</td><td>{_format_float(c)}</td><td>{_format_float(d)}</td></tr>'
    )
    o, c, d = safe_get('annualized_return')
    rows.append(
        f'<tr><td>Annualized Return</td><td>{_format_float(o)}</td><td>{_format_float(c)}</td><td>{_format_float(d)}</td></tr>'
    )
    o, c, d = safe_get('max_drawdown')
    rows.append(
        f'<tr><td>Max Drawdown</td><td>{_format_float(o)}</td><td>{_format_float(c)}</td><td>{_format_float(d)}</td></tr>'
    )
    o, c, d = safe_get('max_drawdown_pct')
    rows.append(
        f'<tr><td>Max Drawdown %</td><td>{_format_float(o * 100, 2)}%</td><td>{_format_float(c * 100, 2)}%</td><td>{_format_float(d * 100, 2)}%</td></tr>'
    )
    o, c, d = safe_get('win_rate')
    rows.append(
        f'<tr><td>Directional Win Rate (non-hold)</td><td>{_format_float(o * 100, 2)}%</td><td>{_format_float(c * 100, 2)}%</td><td>{_format_float(d * 100, 2)}%</td></tr>'
    )
    o, c, d = safe_get('long_win_rate')
    rows.append(
        f'<tr><td>Long Win Rate</td><td>{_format_float(o * 100, 2)}%</td><td>{_format_float(c * 100, 2)}%</td><td>{_format_float(d * 100, 2)}%</td></tr>'
    )
    o, c, d = safe_get('short_win_rate')
    rows.append(
        f'<tr><td>Short Win Rate</td><td>{_format_float(o * 100, 2)}%</td><td>{_format_float(c * 100, 2)}%</td><td>{_format_float(d * 100, 2)}%</td></tr>'
    )
    o, c, d = safe_get('active_ratio')
    rows.append(
        f'<tr><td>Active Ratio (non-hold share)</td><td>{_format_float(o * 100, 2)}%</td><td>{_format_float(c * 100, 2)}%</td><td>{_format_float(d * 100, 2)}%</td></tr>'
    )
    o, c, d = safe_get('win_loss_ratio')
    rows.append(
        f'<tr><td>Win/Loss Ratio</td><td>{_format_float(o)}</td><td>{_format_float(c)}</td><td>{_format_float(d)}</td></tr>'
    )
    o, c, d = safe_get('volatility')
    rows.append(
        f'<tr><td>Volatility</td><td>{_format_float(o)}</td><td>{_format_float(c)}</td><td>{_format_float(d)}</td></tr>'
    )
    o, c, d = safe_get('calmar_ratio')
    rows.append(
        f'<tr><td>Calmar Ratio</td><td>{_format_float(o)}</td><td>{_format_float(c)}</td><td>{_format_float(d)}</td></tr>'
    )
    return '\n'.join(rows)


def create_recommendations_section(results: Dict[str, any]) -> str:
    return """
    <h2>🎯 Recommendations</h2>
    <div class="summary">
        <h3>Based on the comprehensive analysis:</h3>
        <ul>
            <li><strong>Feature Engineering:</strong> Continue using IC/IR filtering for high-quality features</li>
            <li><strong>Dimensionality Reduction:</strong> Use representative selection and Top-K filtering</li>
            <li><strong>Rolling Training:</strong> Quarterly retraining maintains model performance</li>
            <li><strong>Production Deployment:</strong> Models are ready for production use</li>
            <li><strong>Monitoring:</strong> Implement continuous performance monitoring</li>
        </ul>
    </div>
    <h3>🚀 Next Steps</h3>
    <ol>
        <li>Deploy production models to trading systems</li>
        <li>Implement real-time feature engineering pipeline</li>
        <li>Set up automated retraining schedules</li>
        <li>Monitor model performance and drift</li>
        <li>Optimize hyperparameters based on new data</li>
    </ol>
    """


def write_training_report(info_path: str, html_path: str | None = None) -> str:
    """Generate HTML report from training model info JSON.
    
    Args:
        info_path: Path to the training model info JSON file
        html_path: Optional path for HTML output. If None, uses info_path with .html extension
    
    Returns:
        Path to the generated HTML report
    """
    import json
    from pathlib import Path

    info_file = Path(info_path)
    if not info_file.exists():
        raise FileNotFoundError(f"Training info file not found: {info_path}")

    # Load JSON
    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    # Determine output path
    if html_path is None:
        html_path = str(info_file.with_suffix(".html"))
    else:
        html_path = str(Path(html_path))

    # Generate HTML
    html = _build_training_report_html(info)

    # Write HTML
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"📝 Training report written to: {html_path}")

    # Auto-open report in browser
    try:
        import webbrowser
        abs_path = os.path.abspath(html_path)
        file_url = f"file://{abs_path}"
        webbrowser.open(file_url)
        print(f"Report opened in browser: {file_url}")
    except Exception as exc:
        print(f"Note: Could not auto-open report in browser: {exc}")

    return html_path


def _build_training_report_html(info: Dict) -> str:
    """Build HTML content for training report."""
    # Extract data
    symbol = info.get("symbol", "N/A")
    training_date = info.get("training_date", "N/A")
    actual_start = info.get("actual_start", "N/A")
    actual_end = info.get("actual_end", "N/A")
    train_start = info.get("train_start", None)
    train_end = info.get("train_end", None)
    total_bars = info.get("total_bars", 0)
    train_bars = info.get("train_bars", None)
    oos_months = info.get("oos_months", 0)
    oos_metrics = info.get("oos_metrics", {})
    timeframes = info.get("timeframes", {})
    price_range = info.get("price_range", [])
    metrics = info.get("metrics", {})
    model_path = info.get("model_path", "N/A")
    scaler_path = info.get("scaler_path", "N/A")
    pr_curve_path = info.get("pr_curve_path", None)
    roc_curve_path = info.get("roc_curve_path", None)

    pr_roc_section = ""
    if pr_curve_path or roc_curve_path:
        items: list[str] = []
        if pr_curve_path:
            items.append(
                f'<div><img src="{pr_curve_path}" alt="PR Curve" style="max-width:520px; border:1px solid #ddd;"><div style="text-align:center; color:#555; margin-top:6px;">Precision-Recall Curve</div></div>'
            )
        if roc_curve_path:
            items.append(
                f'<div><img src="{roc_curve_path}" alt="ROC Curve" style="max-width:520px; border:1px solid #ddd;"><div style="text-align:center; color:#555; margin-top:6px;">ROC Curve</div></div>'
            )
        pr_roc_section = (
            "<h2>PR / ROC Curves</h2>"
            '<div style="display:flex; gap:20px; flex-wrap: wrap;">'
            f'{"".join(items)}'
            "</div>")
    data_files = info.get("data_files", [])

    # Format date range
    if isinstance(actual_start, str) and isinstance(actual_end, str):
        try:
            from datetime import datetime
            start_dt = datetime.fromisoformat(
                actual_start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(actual_end.replace('Z', '+00:00'))
            date_range_str = f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
        except Exception:
            date_range_str = f"{actual_start} to {actual_end}"
    else:
        date_range_str = f"{actual_start} to {actual_end}"

    # Build timeframe metrics table
    stage1_metrics = metrics.get("stage1", {})
    stage2_metrics = metrics.get("stage2", {})

    timeframe_rows = []
    for tf in sorted(timeframes.keys(),
                     key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 0):
        bars = timeframes.get(tf, 0)
        stage1 = stage1_metrics.get(tf, {})
        stage2 = stage2_metrics.get(tf, {})

        stage1_acc = stage1.get("cv_accuracy", None)
        stage1_std = stage1.get("cv_accuracy_std", None)
        stage2_rmse = stage2.get("cv_rmse", None) if stage2_metrics else None
        stage2_mse = stage2.get("cv_mse", None) if stage2_metrics else None

        # Build row with conditional stage2 columns
        if stage2_metrics:
            timeframe_rows.append(f"""
        <tr>
            <td>{tf}</td>
            <td>{bars:,}</td>
            <td>{_format_float(stage1_acc, 4) if stage1_acc is not None else 'N/A'}</td>
            <td>{_format_float(stage1_std, 4) if stage1_std is not None else 'N/A'}</td>
            <td>{_format_float(stage2_rmse, 6) if stage2_rmse is not None else 'N/A'}</td>
            <td>{_format_float(stage2_mse, 8) if stage2_mse is not None else 'N/A'}</td>
        </tr>""")
        else:
            timeframe_rows.append(f"""
        <tr>
            <td>{tf}</td>
            <td>{bars:,}</td>
            <td>{_format_float(stage1_acc, 4) if stage1_acc is not None else 'N/A'}</td>
            <td>{_format_float(stage1_std, 4) if stage1_std is not None else 'N/A'}</td>
        </tr>""")

    # Build fold details table for stage1
    stage1_fold_details = []
    for tf in sorted(stage1_metrics.keys(),
                     key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 0):
        stage1_tf = stage1_metrics.get(tf, {})
        folds = stage1_tf.get("fold_details", [])
        for fold in folds:
            stage1_fold_details.append(f"""
            <tr>
                <td>{tf}</td>
                <td>{fold.get('fold', 'N/A')}</td>
                <td>{_format_float(fold.get('accuracy'), 4)}</td>
            </tr>""")

    # Build fold details table for stage2
    stage2_fold_details = []
    for tf in sorted(stage2_metrics.keys(),
                     key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 0):
        stage2_tf = stage2_metrics.get(tf, {})
        folds = stage2_tf.get("fold_details", [])
        for fold in folds:
            stage2_fold_details.append(f"""
            <tr>
                <td>{tf}</td>
                <td>{fold.get('fold', 'N/A')}</td>
                <td>{_format_float(fold.get('rmse'), 6)}</td>
                <td>{_format_float(fold.get('mse'), 8)}</td>
            </tr>""")

    # Build stage2 explanation and table outside f-string to avoid nesting
    stage2_explanation = ""
    if stage2_metrics:
        stage2_explanation = """
                <li><strong>Stage2 (CV RMSE)</strong>: Cross-validation Root Mean Squared Error for price prediction (regression task). 
                    Lower is better. Units: price difference (e.g., for BTC, RMSE of 0.001 ~ $0.001 price error).</li>
                <li><strong>Stage2 (CV MSE)</strong>: Cross-validation Mean Squared Error. Lower is better. MSE = RMSE^2.</li>
                """

    stage2_table = ""
    if stage2_fold_details:
        stage2_table = """
        <h2>Stage2: Regression Metrics (Per Fold)</h2>
        <table>
            <tr>
                <th>Timeframe</th>
                <th>Fold</th>
                <th>RMSE</th>
                <th>MSE</th>
            </tr>
            """ + "".join(stage2_fold_details) + """
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Training Report: {symbol}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 24px;
            color: #222;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
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
        .info-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .file-list {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9em;
        }}
        .explanation {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Report: {symbol}</h1>
        
        <div class="info-box">
            <h3>Training Information</h3>
            <table>
                <tr><th>Symbol</th><td>{symbol}</td></tr>
                <tr><th>Training Date</th><td>{training_date}</td></tr>
                <tr><th>Data Period</th><td>{date_range_str}</td></tr>
                {f"<tr><th>Training Period</th><td>{train_start.split('T')[0] if train_start else 'N/A'} to {train_end.split('T')[0] if train_end else 'N/A'}</td></tr>" if train_start and train_end else ""}
                {_get_oos_period_html(oos_metrics, oos_months) if oos_metrics and oos_months > 0 else ""}
                <tr><th>Total Bars</th><td>{total_bars:,}</td></tr>
                {f"<tr><th>Training Bars</th><td>{train_bars:,}</td></tr>" if train_bars is not None else ""}
                {f"<tr><th>OOS Test Bars</th><td>{oos_metrics.get('stage1', {}).get('samples', 0):,}</td></tr>" if oos_metrics and oos_metrics.get('stage1', {}).get('samples') else ""}
                <tr><th>Price Range</th><td>${_format_price(price_range[0] if price_range else 0)} - ${_format_price(price_range[1] if len(price_range) > 1 else 0)}</td></tr>
            </table>
        </div>
        
        <div class="explanation">
            <h3>📦 Model Files Explanation</h3>
            <ul>
                <li><strong>Model File (.pkl)</strong>: Contains the trained LightGBM model, strategy, data loader, and feature engineer.</li>
                <li><strong>Scalers File (_scalers.pkl)</strong>: Contains the StandardScaler objects for each timeframe used to normalize features during training. 
                    These are <strong>required</strong> for making predictions - new data must be scaled using the same scalers before feeding to the model.</li>
                <li><strong>Info File (_info.json)</strong>: Contains metadata about the training run, including metrics and data sources.</li>
            </ul>
            <p><strong>Why two model files?</strong> The scalers (feature normalizers) are separate because:</p>
            <ul>
                <li>They contain the mean/std statistics calculated from training data</li>
                <li>They must be applied to new data before prediction</li>
                <li>Keeping them separate makes it easier to version and update scalers independently</li>
            </ul>
        </div>
        
        <h2>📁 Data Files</h2>
        <div class="file-list">
            <ul>
                {"".join([f"<li>{f}</li>" for f in data_files])}
            </ul>
        </div>
        
        <h2>Multi-Timeframe Metrics</h2>
        <table>
            <tr>
                <th>Timeframe</th>
                <th>Bars</th>
                <th>Stage1: CV Accuracy</th>
                <th>Stage1: Std Dev</th>
                {"<th>Stage2: CV RMSE</th><th>Stage2: CV MSE</th>" if stage2_metrics else ""}
            </tr>
            {"".join(timeframe_rows)}
        </table>
        
        <div class="explanation">
            <h3>Metrics Explanation</h3>
            <ul>
                <li><strong>Stage1 (CV Accuracy)</strong>: Cross-validation accuracy for direction prediction (classification task). 
                    Higher is better. Range: 0-1 (0.5 = random, 1.0 = perfect).</li>
                <li><strong>Stage1 (Std Dev)</strong>: Standard deviation of accuracy across CV folds. Lower means more stable.</li>
                {stage2_explanation if stage2_metrics else ""}
            </ul>
        </div>
        
        <h2>Stage1: Classification Metrics (Per Fold)</h2>
        <table>
            <tr>
                <th>Timeframe</th>
                <th>Fold</th>
                <th>Accuracy</th>
            </tr>
            {"".join(stage1_fold_details)}
        </table>
        
        {stage2_table if stage2_fold_details else ""}
        
        {_build_oos_table(oos_metrics, oos_months) if oos_metrics and oos_months > 0 else ""}
        
        {_build_feature_importance_table(info) if info.get('feature_importance') else ""}

        {pr_roc_section}
        
        <h2>Model Artifacts</h2>
        <table>
            <tr><th>Model Path</th><td>{model_path}</td></tr>
            <tr><th>Scalers Path</th><td>{scaler_path}</td></tr>
            {f"<tr><th>Feature Importance</th><td>{info.get('feature_importance_path', 'N/A')}</td></tr>" if info.get('feature_importance_path') else ""}
            {f"<tr><th>PR Curve</th><td>{pr_curve_path}</td></tr>" if pr_curve_path else ""}
            {f"<tr><th>ROC Curve</th><td>{roc_curve_path}</td></tr>" if roc_curve_path else ""}
        </table>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            <p>Generated by ML Trading Bot Training System</p>
        </div>
    </div>
</body>
</html>"""
    return html


def write_rolling_report(
    results_dir: str,
    summary_path: str | None = None,
    results_csv_path: str | None = None,
    report_type: str = "monthly",
) -> str:
    """Generate HTML report for rolling training (monthly or quarterly).
    
    Args:
        results_dir: Directory containing rolling training results
        summary_path: Path to summary.json (if None, auto-detect)
        results_csv_path: Path to results CSV (if None, auto-detect)
        report_type: "monthly" or "quarterly"
    
    Returns:
        Path to the generated HTML report
    """
    from pathlib import Path

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Auto-detect files
    if summary_path is None:
        summary_path = str(results_path / "summary.json")
    if results_csv_path is None:
        if report_type == "monthly":
            results_csv_path = str(results_path / "monthly_results.csv")
        else:
            results_csv_path = str(results_path / "quarterly_results.csv")

    # Load data
    summary = {}
    if Path(summary_path).exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    results_df = pd.DataFrame()
    if Path(results_csv_path).exists():
        results_df = pd.read_csv(results_csv_path)

    # Generate HTML
    html_path = str(results_path / f"{report_type}_rolling_report.html")
    html = _build_rolling_report_html(summary, results_df, report_type)

    # Write HTML
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"📝 Rolling {report_type} report written to: {html_path}")
    return html_path


def _build_rolling_report_html(
    summary: Dict,
    results_df: pd.DataFrame,
    report_type: str,
) -> str:
    """Build HTML content for rolling training report."""
    report_title = f"{report_type.capitalize()} Rolling Training Report"
    period_col = "test_month" if report_type == "monthly" else "quarter"

    # Extract summary info
    config = summary.get("configuration", {})
    symbol = (summary.get("symbol") or config.get("symbol")
              or (", ".join(config.get("symbols", [])) if isinstance(
                  config.get("symbols"), list) else config.get("symbols"))
              or "N/A")
    total_periods = summary.get(f"total_{report_type}s_tested",
                                len(results_df))
    avg_return = summary.get("avg_return", 0)
    avg_win_rate = summary.get("avg_win_rate", 0)
    avg_profit_factor = summary.get("avg_profit_factor", 0)
    avg_max_drawdown = summary.get("avg_max_drawdown", 0)
    total_trades = summary.get("total_trades", 0)
    feature_engineering = summary.get("feature_engineering",
                                      "EnhancedFeatureEngineer")
    # Training time range (prefer training dates over creation time)
    train_start_date = summary.get("train_start_date") or summary.get(
        "configuration", {}).get("start")
    test_end_date = summary.get("test_end_date") or summary.get(
        "configuration", {}).get("end")
    time_range_str = f"{train_start_date} to {test_end_date}" if (
        train_start_date and test_end_date) else "N/A"

    # Build period results table
    period_rows = []
    if not results_df.empty:
        for _, row in results_df.iterrows():
            period = row.get(period_col, "N/A")
            period_rows.append(f"""
            <tr>
                <td>{period}</td>
                <td>{int(row.get('total_trades', 0))}</td>
                <td>{_format_float(row.get('total_return', 0), 2)}%</td>
                <td>{_format_float(row.get('win_rate', 0), 2)}%</td>
                <td>{_format_float(row.get('profit_factor', 0), 2)}</td>
                <td>{_format_float(row.get('max_drawdown', 0), 2)}%</td>
                <td>{int(row.get('train_samples', 0)):,}</td>
                <td>{int(row.get('test_samples', 0)):,}</td>
                <td>{int(row.get('num_features', 0))}</td>
            </tr>""")

    # Build statistics table
    stats_rows = []
    if not results_df.empty:
        for col in [
                'total_trades', 'total_return', 'win_rate', 'profit_factor',
                'max_drawdown'
        ]:
            if col in results_df.columns:
                mean_val = results_df[col].mean()
                std_val = results_df[col].std()
                min_val = results_df[col].min()
                max_val = results_df[col].max()
                stats_rows.append(f"""
                <tr>
                    <td>{col.replace('_', ' ').title()}</td>
                    <td>{_format_float(mean_val, 2)}</td>
                    <td>{_format_float(std_val, 2)}</td>
                    <td>{_format_float(min_val, 2)}</td>
                    <td>{_format_float(max_val, 2)}</td>
                </tr>""")

    long_term_section = ""
    if not results_df.empty:
        thresholds = {
            "cls_accuracy": ("≥", 0.5, False, "Accuracy"),
            "cls_precision": ("≥", 0.5, False, "Precision"),
            "cls_recall": ("≥", 0.5, False, "Recall"),
            "cls_f1": ("≥", 0.5, False, "F1"),
            "cls_auc": ("≥", 0.5, False, "AUC"),
            "cls_pr_auc": ("≥", 0.5, False, "PR-AUC"),
            "cls_ic_spearman": ("≥", 0.05, True, "IC (Spearman)"),
            "cls_ic_pearson": ("≥", 0.05, True, "IC (Pearson)"),
            "test_r2_return": ("≥", 0.0, False, "Return R²"),
        }
        failing_periods = []
        for _, row in results_df.iterrows():
            period = row.get(period_col, "N/A")
            issues = []
            for col, (symbol, thresh, use_abs, label) in thresholds.items():
                val = row.get(col)
                if pd.isna(val):
                    continue
                comp_val = abs(val) if use_abs else val
                if comp_val < thresh:
                    fmt_val = f"{val:.2f}" if not pd.isna(val) else "N/A"
                    issues.append(f"{label} {fmt_val} < {thresh:.2f}")
            if issues:
                failing_periods.append((period, issues))

        thresholds_text = (
            "1) Accuracy/F1/AUC/PR-AUC ≥ 0.50 保证分类器具备基础识别能力；"
            "2) Precision/Recall ≥ 0.50 代表模型既能控制误开仓也能抓住行情；"
            "3) |IC| ≥ 0.05 表示信号与收益相关性显著；"
            "4) Return R² ≥ 0 说明收益回归模型至少不会反向预测（若 R² < 0，回归模型会削弱信号，可视为不可用）。")
        if failing_periods:
            issue_rows = "".join([
                f"<li><strong>{period}</strong>: " + "; ".join(issues) +
                "</li>" for period, issues in failing_periods
            ])
            long_term_section = f"""
        <div class="explanation" style="background-color:#ffebee;border-left-color:#e53935;">
            <h3>📉 长期有效性结论</h3>
            <p>部分测试周期未达到默认阈值。阈值含义如下：{thresholds_text}</p>
            <p><strong>Return R² 未达标</strong> 说明收益回归模型对收益的“方向/幅度”预测反向或噪声较大，会削弱评分结果，应降低该月回归分数权重或重新训练。</p>
            <ul>{issue_rows}</ul>
        </div>
        """
        else:
            long_term_section = f"""
        <div class="explanation" style="background-color:#e8f5e9;border-left-color:#2e7d32;">
            <h3>✅ 长期有效性结论</h3>
            <p>全部测试周期均达到默认阈值，说明分类与回归模型在滚动窗口内表现稳定，可侧重部署。阈值含义：{thresholds_text}</p>
        </div>
        """

    # Optional CV metrics table if present
    cv_section = ""
    if not results_df.empty and "cv_logloss_mean" in results_df.columns:
        cv_mean_overall = _format_float(results_df["cv_logloss_mean"].mean(),
                                        6)
        cv_std_overall = _format_float(results_df["cv_logloss_std"].mean(), 6)
        cv_section = f"""
        <h2>🧪 Cross-Validation (Training Window)</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Mean multi_logloss (across periods)</td><td>{cv_mean_overall}</td></tr>
            <tr><td>Std multi_logloss (across periods)</td><td>{cv_std_overall}</td></tr>
        </table>
        """

    guidance_section = """
    <h2>📘 Guidance: Rolling vs Time-Series CV</h2>
    <div class="explanation">
        <ul>
            <li><strong>Rolling OOS</strong>: 贴近实盘的“训练→上线→下一期”评估，能暴露概念漂移与逐期稳定性，适合作为主评估。</li>
            <li><strong>时序CV</strong>: 在训练窗内估计方差与过拟合风险，用于调参与特征选择；与OOS对照，若偏差大，优先信任滚动OOS并缩短重训周期。</li>
        </ul>
    </div>
    """
    feature_importance_section = _build_rolling_feature_importance_section(
        summary)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{report_title}: {symbol}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 24px;
            color: #222;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
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
        .info-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .explanation {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .good {{
            color: #0a7c2f;
            font-weight: 600;
        }}
        .bad {{
            color: #b00020;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {report_title}: {symbol}</h1>
        
        <div class="info-box">
            <h3>📋 Summary</h3>
            <table>
                <tr><th>Symbol</th><td>{symbol}</td></tr>
                <tr><th>Report Type</th><td>{report_type.capitalize()} Rolling Training</td></tr>
                <tr><th>Training Period</th><td>{time_range_str}</td></tr>
                <tr><th>Total Periods Tested</th><td>{total_periods}</td></tr>
                <tr><th>Total Trades</th><td>{total_trades:,}</td></tr>
                <tr><th>Feature Engineering</th><td>{feature_engineering}</td></tr>
                <tr><th>Avg Direction F1</th><td>{_format_float(summary.get('avg_cls_f1'), 4)}</td></tr>
                <tr><th>Avg Direction AUC</th><td>{_format_float(summary.get('avg_cls_auc'), 4)}</td></tr>
                <tr><th>Avg Return R²</th><td>{_format_float(summary.get('avg_return_r2'), 4)}</td></tr>
                <tr><th>Avg Volatility R²</th><td>{_format_float(summary.get('avg_vol_r2'), 4)}</td></tr>
            </table>
        </div>
        
        <div class="explanation">
            <h3>📊 Rolling Training Explanation</h3>
            <p><strong>{report_type.capitalize()} Rolling Training</strong> uses an expanding window approach:</p>
            <ul>
                <li><strong>Training Window</strong>: Expands each period, accumulating more data over time</li>
                <li><strong>Test Window</strong>: Next period ({'month' if report_type == 'monthly' else 'quarter'}) after training window</li>
                <li><strong>Purpose</strong>: Simulates real-world deployment where model is retrained periodically</li>
            </ul>
            <p><strong>Example</strong>: Train on periods 1-6, test on period 7; then train on periods 1-7, test on period 8, etc.</p>
        </div>
        
        <h2>📈 Performance Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Average</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
            {"".join(stats_rows)}
        </table>
        {long_term_section}

        {feature_importance_section}
        
        {cv_section}
        {guidance_section}

        <div class="explanation">
            <h3>Metrics Explanation</h3>
            <ul>
                <li><strong>Total Return</strong>: Cumulative return percentage for the test period</li>
                <li><strong>Win Rate</strong>: Percentage of profitable trades</li>
                <li><strong>Profit Factor</strong>: Ratio of gross profit to gross loss (>1 = profitable)</li>
                <li><strong>Max Drawdown</strong>: Maximum peak-to-trough decline during the test period</li>
                <li><strong>Total Trades</strong>: Number of trades executed during the test period</li>
            </ul>
        </div>
        
        <h2>📅 Period-by-Period Results</h2>
        <table>
            <tr>
                <th>{report_type.capitalize()}</th>
                <th>Trades</th>
                <th>Return (%)</th>
                <th>Win Rate (%)</th>
                <th>Profit Factor</th>
                <th>Max DD (%)</th>
                <th>Train Samples</th>
                <th>Test Samples</th>
                <th>Features</th>
            </tr>
            {"".join(period_rows)}
        </table>
        
        <h2>⚙️ Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            {f"<tr><td>Data Directory</td><td>{config.get('data_dir', 'N/A')}</td></tr>" if config.get('data_dir') else ""}
            {f"<tr><td>Initial Train Periods</td><td>{config.get('initial_train_months' if report_type == 'monthly' else 'initial_train_quarters', 'N/A')}</td></tr>" if config.get('initial_train_months' if report_type == 'monthly' else 'initial_train_quarters') else ""}
            {f"<tr><td>GPU</td><td>{config.get('gpu', 'N/A')}</td></tr>" if 'gpu' in config else ""}
            {f"<tr><td>Order Flow Features</td><td>{config.get('add_order_flow', 'N/A')}</td></tr>" if 'add_order_flow' in config else ""}
        </table>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            <p>Generated by ML Trading Bot Rolling Training System</p>
        </div>
    </div>
</body>
</html>"""
    return html


def main() -> str:
    print("📋 Dimensionality Training Report Generator")
    print("=" * 50)
    return generate_comprehensive_report()


if __name__ == "__main__":
    main()
