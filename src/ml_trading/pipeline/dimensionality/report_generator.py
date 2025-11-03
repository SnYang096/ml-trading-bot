"""Utilities for building comprehensive dimensionality reports."""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
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
            stats["best_performing_method"] = "Autoencoder + LightGBM"

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
                <p>Using Autoencoder + LightGBM Technology</p>
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
        multi_horizon_results)

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
) -> str:
    """Build HTML content string for the report."""
    # Build conditional 4-stage comparison table
    stage_comparison_table = ""
    if has_4_stages:
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
<tr><td>Stage 4: Compressed</td><td>{d.get('compressed_dimensions','-')}</td><td>Autoencoder compressed dimensions</td></tr>
<tr><th colspan="3">Summary</th></tr>
<tr><td>Final Compression Ratio</td><td colspan="2">{_format_float(d.get('compression_ratio'),2)}x ({d.get('original_features_count','-')} → {d.get('compressed_dimensions','-')})</td></tr>
<tr><td>Samples (train/val/test)</td><td colspan="2">{d.get('training_samples','-')} / {d.get('validation_samples','-')} / {d.get('test_samples','-')}</td></tr>
</table>

{stage_comparison_table}

<h2>Performance (Test Set)</h2>
<table>
<tr><th>Metric</th><th>Original</th><th>Compressed</th><th>Delta</th></tr>
<tr><td>R²</td><td>{_format_float(orig.get('r2'))}</td><td>{_format_float(comp.get('r2'))}</td><td>{_format_float(delta_r2)}</td></tr>
<tr><td>RMSE</td><td>{_format_float(orig.get('rmse'))}</td><td>{_format_float(comp.get('rmse'))}</td><td>{_format_float((comp.get('rmse') or 0)-(orig.get('rmse') or 0))}</td></tr>
<tr><td>MAE</td><td>{_format_float(orig.get('mae'))}</td><td>{_format_float(comp.get('mae'))}</td><td>{_format_float((comp.get('mae') or 0)-(orig.get('mae') or 0))}</td></tr>
</table>

{val_4stage_fin_table}

<h2>Financial Metrics (Validation Set)</h2>
<table>
<tr><th>Metric</th><th>Original</th><th>Compressed</th><th>Delta</th></tr>
{_build_financial_metrics_table(orig_val_fin or orig_fin, comp_val_fin or comp_fin)}
</table>

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
<li>Autoencoder epochs: {train_info.get('autoencoder_epochs','-')}</li>
<li>Autoencoder final loss: {_format_float(train_info.get('autoencoder_final_loss'))}</li>
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
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0
    )
    
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
            <li><strong>Dimensionality Reduction:</strong> Autoencoder shows consistent performance improvements</li>
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
    return html_path


def _build_training_report_html(info: Dict) -> str:
    """Build HTML content for training report."""
    # Extract data
    symbol = info.get("symbol", "N/A")
    training_date = info.get("training_date", "N/A")
    actual_start = info.get("actual_start", "N/A")
    actual_end = info.get("actual_end", "N/A")
    total_bars = info.get("total_bars", 0)
    timeframes = info.get("timeframes", {})
    price_range = info.get("price_range", [])
    metrics = info.get("metrics", {})
    model_path = info.get("model_path", "N/A")
    scaler_path = info.get("scaler_path", "N/A")
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
        stage2_rmse = stage2.get("cv_rmse", None)
        stage2_mse = stage2.get("cv_mse", None)

        timeframe_rows.append(f"""
        <tr>
            <td>{tf}</td>
            <td>{bars:,}</td>
            <td>{_format_float(stage1_acc, 4) if stage1_acc is not None else 'N/A'}</td>
            <td>{_format_float(stage1_std, 4) if stage1_std is not None else 'N/A'}</td>
            <td>{_format_float(stage2_rmse, 6) if stage2_rmse is not None else 'N/A'}</td>
            <td>{_format_float(stage2_mse, 8) if stage2_mse is not None else 'N/A'}</td>
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
        <h1>📊 Training Report: {symbol}</h1>
        
        <div class="info-box">
            <h3>📋 Training Information</h3>
            <table>
                <tr><th>Symbol</th><td>{symbol}</td></tr>
                <tr><th>Training Date</th><td>{training_date}</td></tr>
                <tr><th>Data Period</th><td>{date_range_str}</td></tr>
                <tr><th>Total Bars (5T)</th><td>{total_bars:,}</td></tr>
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
        
        <h2>📈 Multi-Timeframe Metrics</h2>
        <table>
            <tr>
                <th>Timeframe</th>
                <th>Bars</th>
                <th>Stage1: CV Accuracy</th>
                <th>Stage1: Std Dev</th>
                <th>Stage2: CV RMSE</th>
                <th>Stage2: CV MSE</th>
            </tr>
            {"".join(timeframe_rows)}
        </table>
        
        <div class="explanation">
            <h3>📊 Metrics Explanation</h3>
            <ul>
                <li><strong>Stage1 (CV Accuracy)</strong>: Cross-validation accuracy for direction prediction (classification task). 
                    Higher is better. Range: 0-1 (0.5 = random, 1.0 = perfect).</li>
                <li><strong>Stage1 (Std Dev)</strong>: Standard deviation of accuracy across CV folds. Lower means more stable.</li>
                <li><strong>Stage2 (CV RMSE)</strong>: Cross-validation Root Mean Squared Error for price prediction (regression task). 
                    Lower is better. Units: price difference (e.g., for BTC, RMSE of 0.001 ≈ $0.001 price error).</li>
                <li><strong>Stage2 (CV MSE)</strong>: Cross-validation Mean Squared Error. Lower is better. MSE = RMSE².</li>
            </ul>
        </div>
        
        <h2>🔬 Stage1: Classification Metrics (Per Fold)</h2>
        <table>
            <tr>
                <th>Timeframe</th>
                <th>Fold</th>
                <th>Accuracy</th>
            </tr>
            {"".join(stage1_fold_details)}
        </table>
        
        <h2>📉 Stage2: Regression Metrics (Per Fold)</h2>
        <table>
            <tr>
                <th>Timeframe</th>
                <th>Fold</th>
                <th>RMSE</th>
                <th>MSE</th>
            </tr>
            {"".join(stage2_fold_details)}
        </table>
        
        <h2>💾 Model Artifacts</h2>
        <table>
            <tr><th>Model Path</th><td>{model_path}</td></tr>
            <tr><th>Scalers Path</th><td>{scaler_path}</td></tr>
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
    symbol = summary.get("configuration", {}).get("symbol", summary.get("configuration", {}).get("symbols", ["N/A"])[0] if isinstance(summary.get("configuration", {}).get("symbols"), list) else "N/A")
    total_periods = summary.get(f"total_{report_type}s_tested", len(results_df))
    avg_return = summary.get("avg_return", 0)
    avg_win_rate = summary.get("avg_win_rate", 0)
    avg_profit_factor = summary.get("avg_profit_factor", 0)
    avg_max_drawdown = summary.get("avg_max_drawdown", 0)
    total_trades = summary.get("total_trades", 0)
    feature_engineering = summary.get("feature_engineering", "EnhancedFeatureEngineer")
    config = summary.get("configuration", {})
    
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
        for col in ['total_trades', 'total_return', 'win_rate', 'profit_factor', 'max_drawdown']:
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
                <tr><th>Total Periods Tested</th><td>{total_periods}</td></tr>
                <tr><th>Total Trades</th><td>{total_trades:,}</td></tr>
                <tr><th>Feature Engineering</th><td>{feature_engineering}</td></tr>
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
        
        <div class="explanation">
            <h3>📊 Metrics Explanation</h3>
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
