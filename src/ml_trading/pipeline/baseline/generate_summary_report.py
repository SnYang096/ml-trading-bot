"""Generate summary report for baseline training with multiple configurations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd


def collect_baseline_results(results_dir: str = "results/baseline") -> pd.DataFrame:
    """Collect all baseline training results from subdirectories.
    
    Args:
        results_dir: Base directory containing baseline results
        
    Returns:
        DataFrame with all results
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return pd.DataFrame()
    
    all_results = []
    
    # Check main directory first
    main_info = results_path / "baseline_training_info.json"
    if main_info.exists():
        try:
            with open(main_info, "r", encoding="utf-8") as f:
                info = json.load(f)
                info["config_dir"] = "baseline"
                info["timeframe"] = info.get("timeframe", "5T")
                info["forward_bars"] = info.get("forward_bars", 3)
                all_results.append(info)
        except Exception as exc:
            print(f"Warning: Failed to read {main_info}: {exc}")
    
    # Check subdirectories (fb*_tf*)
    for subdir in results_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith("fb"):
            info_file = subdir / "baseline_training_info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        info = json.load(f)
                        info["config_dir"] = subdir.name
                        all_results.append(info)
                except Exception as exc:
                    print(f"Warning: Failed to read {info_file}: {exc}")
    
    if not all_results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    return df


def generate_summary_report(results_dir: str = "results/baseline", 
                           output_path: Optional[str] = None) -> str:
    """Generate HTML summary report for all baseline training configurations.
    
    Args:
        results_dir: Base directory containing baseline results
        output_path: Output HTML file path (default: results_dir/summary_report.html)
        
    Returns:
        Path to generated HTML report
    """
    df = collect_baseline_results(results_dir)
    
    if df.empty:
        print("No baseline results found.")
        return ""
    
    if output_path is None:
        output_path = os.path.join(results_dir, "summary_report.html")
    
    # Sort by timeframe and forward_bars
    df = df.sort_values(["timeframe", "forward_bars"])
    
    # Build summary table rows
    rows = []
    for idx, row in df.iterrows():
        try:
            # Convert Series to dict for easier access
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
            
            timeframe = row_dict.get("timeframe", "N/A")
            forward_bars = row_dict.get("forward_bars", "N/A")
            symbol = row_dict.get("symbol", "N/A")
            config_dir = row_dict.get("config_dir", "N/A")
            
            # Stage1 metrics - safely extract with type checking
            metrics = row_dict.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            
            stage1_metrics = metrics.get("stage1", {})
            if not isinstance(stage1_metrics, dict):
                stage1_metrics = {}
            
            stage1_tf_metrics = stage1_metrics.get(timeframe, {})
            if not isinstance(stage1_tf_metrics, dict):
                stage1_tf_metrics = {}
            
            cv_accuracy = stage1_tf_metrics.get("cv_accuracy", None)
            cv_accuracy_std = stage1_tf_metrics.get("cv_accuracy_std", None)
            
            # Stage2 metrics - safely extract with type checking
            stage2_metrics = metrics.get("stage2", {})
            if not isinstance(stage2_metrics, dict):
                stage2_metrics = {}
            
            stage2_tf_metrics = stage2_metrics.get(timeframe, {})
            if not isinstance(stage2_tf_metrics, dict):
                stage2_tf_metrics = {}
            
            cv_rmse = stage2_tf_metrics.get("cv_rmse", None)
            cv_mse = stage2_tf_metrics.get("cv_mse", None)
            
            # OOS metrics - safely extract with type checking
            oos_metrics = row_dict.get("oos_metrics", {})
            if not isinstance(oos_metrics, dict):
                oos_metrics = {}
            
            oos_stage1 = oos_metrics.get("stage1", {})
            if not isinstance(oos_stage1, dict):
                oos_stage1 = {}
            oos_accuracy = oos_stage1.get("accuracy", None)
            oos_precision = oos_stage1.get("precision", None)
            oos_recall = oos_stage1.get("recall", None)
            oos_f1 = oos_stage1.get("f1", None)
            oos_auc = oos_stage1.get("auc", None)
            oos_pr_auc = oos_stage1.get("pr_auc", None)
            best_threshold = oos_stage1.get("best_threshold", None)
            quality_check = oos_stage1.get("quality_check", {})
            if not isinstance(quality_check, dict):
                quality_check = {}
            quality_pass = quality_check.get("passed", True)
            oos_bars = oos_stage1.get("samples", 0)
            
            oos_stage2 = oos_metrics.get("stage2", {})
            if not isinstance(oos_stage2, dict):
                oos_stage2 = {}
            oos_rmse = oos_stage2.get("rmse", None)
            
            # Feature importance top 5 (by gain)
            fi_list = row_dict.get("feature_importance", [])
            top_feats = []
            if isinstance(fi_list, list) and fi_list:
                try:
                    # Filter out None values and ensure we have valid dicts
                    valid_feats = [f for f in fi_list if isinstance(f, dict) and 'feature' in f]
                    top_feats = sorted(valid_feats, key=lambda x: x.get('importance_gain', 0), reverse=True)[:5]
                except Exception as e:
                    # Debug: print error if needed
                    # print(f"Warning: Failed to extract top features: {e}")
                    top_feats = fi_list[:5] if isinstance(fi_list, list) else []
            top_feats_str = ", ".join([str(it.get('feature', '')) for it in top_feats if isinstance(it, dict)]) if top_feats else ""

            # Curves/paths (make relative if under results_dir)
            pr_curve_path = row_dict.get('pr_curve_path')
            roc_curve_path = row_dict.get('roc_curve_path')
            fi_csv_path = row_dict.get('feature_importance_path')
            def _to_rel(p):
                if not p:
                    return None
                try:
                    abs_p = os.path.abspath(p)
                    base = os.path.abspath(results_dir)
                    return abs_p[len(base)+1:] if abs_p.startswith(base) else p
                except Exception:
                    return p
            pr_rel = _to_rel(pr_curve_path)
            roc_rel = _to_rel(roc_curve_path)
            fi_rel = _to_rel(fi_csv_path)

            # Training info
            train_bars = row_dict.get("train_bars") or row_dict.get("total_bars", 0)

            quality_label = (
                '<span style="color: #155724; background:#d4edda; padding:2px 6px; border-radius:4px;">PASS</span>'
                if quality_pass else
                '<span style="color: #721c24; background:#f8d7da; padding:2px 6px; border-radius:4px;">FAIL</span>'
            )

            pr_thumb = f"<img src=\"{pr_rel}\" style=\"max-width:160px; border:1px solid #ddd;\">" if pr_rel else ""
            roc_thumb = f"<img src=\"{roc_rel}\" style=\"max-width:160px; border:1px solid #ddd;\">" if roc_rel else ""
            fi_link = f"<a href=\"{fi_rel}\">CSV</a>" if fi_rel else ""

            rows.append(f"""
            <tr>
                <td>{symbol}</td>
                <td>{timeframe}</td>
                <td>{forward_bars}</td>
                <td>{train_bars:,}</td>
                <td>{oos_bars:,}</td>
                <td>{_format_float(cv_accuracy, 4)}</td>
                <td>{_format_float(cv_accuracy_std, 4)}</td>
                <td>{_format_float(oos_accuracy, 4)}</td>
                <td>{_format_float(oos_precision, 4)}</td>
                <td>{_format_float(oos_recall, 4)}</td>
                <td>{_format_float(oos_f1, 4)}</td>
                <td>{_format_float(oos_auc, 4)}</td>
                <td>{_format_float(oos_pr_auc, 4)}</td>
                <td>{_format_float(best_threshold, 3)}</td>
                <td>{quality_label}</td>
                <td>{_format_float(cv_rmse, 6)}</td>
                <td>{_format_float(oos_rmse, 6)}</td>
                <td>{config_dir}</td>
            </tr>
            <tr style=\"background:#fafafa;\">
                <td colspan=5><strong>Top Features:</strong> {top_feats_str} {f'({fi_link})' if fi_link else ''}</td>
                <td colspan=6>
                    <div style=\"display:flex; gap:10px; align-items:center;\">{pr_thumb}{roc_thumb}</div>
                </td>
                <td colspan=5></td>
            </tr>""")
        except Exception as exc:
            print(f"Warning: Failed to process row {idx}: {exc}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Baseline Training Summary Report</title>
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
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            position: sticky;
            top: 0;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f4f8;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Baseline Training Summary Report</h1>
        
        <div class="info-box">
            <h3>Summary Information</h3>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Configurations:</strong> {len(df)}</p>
            <p><strong>Unique Timeframes:</strong> {', '.join(sorted(df['timeframe'].unique()))}</p>
            <p><strong>Unique Forward Bars:</strong> {', '.join(map(str, sorted(df['forward_bars'].unique())))}</p>
        </div>
        
        <div class="explanation">
            <h3>Report Explanation</h3>
            <ul>
                <li><strong>CV Accuracy</strong>: Cross-validation accuracy from training (Stage1 classification)</li>
                <li><strong>CV Accuracy Std</strong>: Standard deviation of CV accuracy across folds</li>
                <li><strong>OOS Accuracy</strong>: Out-of-sample test accuracy (unseen data)</li>
                <li><strong>OOS Precision</strong>: Precision on OOS test (控制误开仓)</li>
                <li><strong>OOS Recall</strong>: Recall on OOS test (抓住行情能力)</li>
                <li><strong>OOS F1</strong>: F1 Score on OOS test (综合指标，推荐阈值: F1 &gt; 0.3)</li>
                <li><strong>OOS AUC</strong>: AUC-ROC on OOS test (区分能力，推荐阈值: AUC &gt; 0.6)</li>
                <li><strong>OOS PR-AUC</strong>: PR-AUC on OOS test (更适合不平衡数据)</li>
                <li><strong>Best Threshold</strong>: Optimal classification threshold (maximizing F1)</li>
                <li><strong>Quality</strong>: Model quality check result (PASS/FAIL based on F1&gt;0.3 OR AUC&gt;0.6)</li>
                <li><strong>CV RMSE</strong>: Cross-validation RMSE from training (Stage2 regression)</li>
                <li><strong>OOS RMSE</strong>: Out-of-sample test RMSE (unseen data)</li>
                <li><strong>Training Bars</strong>: Number of bars used for training</li>
                <li><strong>OOS Bars</strong>: Number of bars used for out-of-sample testing</li>
                <li><strong>Top Features</strong>: Top 5 most important features (by gain), with CSV link</li>
                <li><strong>PR/ROC Curves</strong>: Precision-Recall and ROC curve thumbnails (if available)</li>
            </ul>
            <p><strong>Note:</strong> If you see "N/A" for OOS metrics or empty "Top Features", these results were generated with older code. 
            Please re-run training with the latest code to generate complete metrics, feature importance, and curves.</p>
        </div>
        
        <h2>All Configurations Comparison</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Timeframe</th>
                <th>Forward Bars</th>
                <th>Training Bars</th>
                <th>OOS Bars</th>
                <th>CV Accuracy</th>
                <th>CV Accuracy Std</th>
                <th>OOS Accuracy</th>
                <th>OOS Precision</th>
                <th>OOS Recall</th>
                <th>OOS F1</th>
                <th>OOS AUC</th>
                <th>OOS PR-AUC</th>
                <th>Best Threshold</th>
                <th>Quality</th>
                <th>CV RMSE</th>
                <th>OOS RMSE</th>
                <th>Config Directory</th>
            </tr>
            {"".join(rows)}
        </table>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            <p>Generated by ML Trading Bot Baseline Training System</p>
        </div>
    </div>
</body>
</html>"""
    
    # Write HTML
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Summary report generated: {output_path}")
    
    # Auto-open report in browser
    try:
        import webbrowser
        # Convert to absolute path
        abs_path = os.path.abspath(output_path)
        # Use file:// URL for local files
        file_url = f"file://{abs_path}"
        webbrowser.open(file_url)
        print(f"Report opened in browser: {file_url}")
    except Exception as exc:
        print(f"Note: Could not auto-open report in browser: {exc}")
        print(f"Please open manually: {output_path}")
    
    return output_path


def _format_float(val, digits: int = 4) -> str:
    """Format float value for HTML report display."""
    try:
        if val is None:
            return "N/A"
        if isinstance(val, (int, float)):
            if val != val:  # NaN check
                return "N/A"
            return f"{val:.{digits}f}"
        return str(val)
    except Exception:
        return "N/A"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate baseline training summary report")
    parser.add_argument("--results-dir", type=str, default="results/baseline",
                       help="Base directory containing baseline results")
    parser.add_argument("--output", type=str, default=None,
                       help="Output HTML file path (default: results_dir/summary_report.html)")
    args = parser.parse_args()
    
    generate_summary_report(args.results_dir, args.output)

