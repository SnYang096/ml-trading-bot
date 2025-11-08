"""Generate summary report for training results (no 'baseline' naming)."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import numpy as np


def collect_training_results(
        results_dir: str = "results/training") -> pd.DataFrame:
    """Collect training results from timestamped directories.
    
    Supports both new format (timestamped directories) and old format (fb*_tf* subdirectories).
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return pd.DataFrame()

    all_results = []

    # Check if this is a timestamped directory (new format) or legacy format
    # New format: YYYYMMDD_HHMMSS_{SYMBOL}_{FEATURE_TYPE}/
    # Legacy format: results/training/ with fb*_tf* subdirectories

    # First, check if results_path itself is a timestamped directory
    is_timestamped_dir = (results_path.name[0].isdigit() if results_path.name else False) and len(
        results_path.name) >= 15 and "_" in results_path.name[:15]
    
    # First, try to find timestamped directories (new format)
    timestamped_dirs = []
    legacy_found = False

    if is_timestamped_dir:
        # results_path is already a timestamped directory
        timestamped_dirs.append(results_path)
    else:
        # results_path is the parent directory, search for timestamped subdirectories
        for item in results_path.iterdir():
            if item.is_dir():
                # Check if it's a timestamped directory (format: YYYYMMDD_HHMMSS_*)
                if item.name[0].isdigit() and len(
                        item.name) >= 15 and "_" in item.name[:15]:
                    timestamped_dirs.append(item)
                # Check for legacy format (fb*_tf*)
                elif item.name.startswith("fb"):
                    legacy_found = True

    # If timestamped directories exist, use them (new format)
    if timestamped_dirs:
        # Sort by timestamp (newest first)
        timestamped_dirs.sort(key=lambda x: x.name, reverse=True)

        # Collect from the most recent timestamped directory (or all if needed)
        # For now, use the most recent one to avoid mixing different training runs
        most_recent_dir = timestamped_dirs[0]

        # main directory in timestamped folder
        main_info = most_recent_dir / "training_info.json"
        if main_info.exists():
            try:
                with open(main_info, "r", encoding="utf-8") as f:
                    info = json.load(f)
                    info["config_dir"] = "root"
                    info["timeframe"] = info.get("timeframe", "5T")
                    info["forward_bars"] = info.get("forward_bars", 3)
                    all_results.append(info)
            except Exception as exc:
                print(f"Warning: Failed to read {main_info}: {exc}")

        # subdirectories (fb*_tf*) in timestamped folder
        for subdir in most_recent_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("fb"):
                info_file = subdir / "training_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, "r", encoding="utf-8") as f:
                            info = json.load(f)
                            info["config_dir"] = subdir.name
                            all_results.append(info)
                    except Exception as exc:
                        print(f"Warning: Failed to read {info_file}: {exc}")

    # Legacy format: fallback to old structure
    elif legacy_found:
        # main directory
        main_info = results_path / "training_info.json"
        if main_info.exists():
            try:
                with open(main_info, "r", encoding="utf-8") as f:
                    info = json.load(f)
                    info["config_dir"] = "root"
                    info["timeframe"] = info.get("timeframe", "5T")
                    info["forward_bars"] = info.get("forward_bars", 3)
                    all_results.append(info)
            except Exception as exc:
                print(f"Warning: Failed to read {main_info}: {exc}")

        # subdirectories (fb*_tf*)
        for subdir in results_path.iterdir():
            if subdir.is_dir() and subdir.name.startswith("fb"):
                info_file = subdir / "training_info.json"
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

    return pd.DataFrame(all_results)


def generate_summary_report(results_dir: str = "results/training",
                            output_path: Optional[str] = None) -> str:
    df = collect_training_results(results_dir)
    if df.empty:
        print("No training results found.")
        return ""

    # Sort for readability
    if {"timeframe", "forward_bars"}.issubset(df.columns):
        df = df.sort_values(["timeframe", "forward_bars"])  # type: ignore

    # Extract common info from first row for filename and title
    first_row = df.iloc[0].to_dict() if hasattr(
        df.iloc[0], "to_dict") else dict(df.iloc[0])
    symbol_raw = first_row.get("symbol", "UNKNOWN")
    # Format symbol for filename: replace comma with underscore for multi-asset (e.g., "BTCUSDT,ETHUSDT,SOLUSDT" -> "BTCUSDT_ETHUSDT_SOLUSDT")
    symbol = symbol_raw.replace(",", "_") if isinstance(
        symbol_raw, str) else str(symbol_raw)
    feature_type = first_row.get("feature_type", "unknown")

    # Extract time ranges
    train_start = first_row.get("train_start")
    train_end = first_row.get("train_end")
    actual_start = first_row.get("actual_start")
    actual_end = first_row.get("actual_end")
    oos_start = first_row.get("oos_start")
    oos_end = first_row.get("oos_end")

    # Format dates for filename
    def _format_date_for_filename(date_str):
        if not date_str:
            return ""
        try:
            if isinstance(date_str, str):
                # Parse ISO format or other formats
                if "T" in date_str:
                    # Handle ISO format: 2024-01-01T00:00:00 or 2024-01-01T00:00:00.589319
                    date_part = date_str.split("T")[0]
                    dt = datetime.strptime(date_part, "%Y-%m-%d")
                else:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                return dt.strftime("%Y%m%d")
            return ""
        except Exception as e:
            # If parsing fails, try to extract date part manually
            if isinstance(date_str, str) and len(date_str) >= 10:
                try:
                    return date_str[:10].replace("-", "")
                except:
                    return ""
            return ""

    train_start_str = _format_date_for_filename(train_start)
    train_end_str = _format_date_for_filename(train_end)
    actual_start_str = _format_date_for_filename(
        actual_start) if actual_start else train_start_str
    actual_end_str = _format_date_for_filename(
        actual_end) if actual_end else train_end_str

    # Generate filename
    if output_path is None:
        # Generate summary report in the same directory as training results
        # If results_dir is a timestamped directory, use it directly
        # Otherwise, find the most recent timestamped directory from the collected data
        if os.path.basename(results_dir).startswith("20") and "_" in os.path.basename(results_dir)[:15]:
            # results_dir is already a timestamped directory
            filename = "summary_report.html"
            output_path = os.path.join(results_dir, filename)
        else:
            # results_dir is the parent directory, find the most recent timestamped directory
            # by checking where the training_info.json files are located
            results_path = Path(results_dir)
            timestamped_dirs = []
            for item in results_path.iterdir():
                if item.is_dir() and item.name[0].isdigit() and len(item.name) >= 15 and "_" in item.name[:15]:
                    timestamped_dirs.append(item)
            
            if timestamped_dirs:
                # Sort by timestamp (newest first)
                timestamped_dirs.sort(key=lambda x: x.name, reverse=True)
                most_recent_dir = timestamped_dirs[0]
                # Generate summary report in the most recent timestamped directory
                filename = "summary_report.html"
                output_path = os.path.join(str(most_recent_dir), filename)
            else:
                # Fallback: generate filename with symbol and dates in parent directory
                filename_parts = [symbol, feature_type]
                if train_start_str and train_end_str:
                    filename_parts.append(f"{train_start_str}_{train_end_str}")
                if actual_start_str and actual_end_str and (
                        actual_start_str != train_start_str
                        or actual_end_str != train_end_str):
                    filename_parts.append(f"oos_{actual_start_str}_{actual_end_str}")
                filename = "_".join(filename_parts) + "_summary_report.html"
                output_path = os.path.join(results_dir, filename)

    # Collect unique feature types for title
    # Use the feature_type from first row (most recent training run)
    # If multiple feature types exist, it means there are multiple training runs mixed together
    # In that case, we prefer the feature_type from the first row (most recent or most relevant)
    if "feature_type" in df.columns:
        feature_types = df["feature_type"].dropna().unique()
        # Prefer the feature_type from first row (most recent training run)
        # This ensures we show the feature_type that was actually used in the current training
        if feature_type and feature_type != "unknown":
            feature_types_str = str(feature_type)
        elif len(feature_types) == 1:
            feature_types_str = str(feature_types[0])
        elif len(feature_types) > 0:
            # Multiple feature types: use first row's feature_type (most recent training)
            feature_types_str = str(
                feature_type) if feature_type != "unknown" else str(
                    feature_types[0])
        else:
            feature_types_str = "unknown"
    else:
        feature_types_str = feature_type if feature_type != "unknown" else "unknown"

    # Format dates for display (more readable format)
    def _format_date_for_display(date_str):
        if not date_str:
            return ""
        try:
            if isinstance(date_str, str):
                if "T" in date_str:
                    date_part = date_str.split("T")[0]
                    dt = datetime.strptime(date_part, "%Y-%m-%d")
                else:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                return dt.strftime("%Y-%m-%d")
            return ""
        except Exception:
            if isinstance(date_str, str) and len(date_str) >= 10:
                try:
                    return date_str[:10]
                except:
                    return ""
            return ""

    train_start_display = _format_date_for_display(train_start)
    train_end_display = _format_date_for_display(train_end)
    actual_start_display = _format_date_for_display(
        actual_start) if actual_start else train_start_display
    actual_end_display = _format_date_for_display(
        actual_end) if actual_end else train_end_display
    # Use explicit OOS time range if available, otherwise fallback to actual_start/actual_end
    oos_start_display = _format_date_for_display(
        oos_start) if oos_start else None
    oos_end_display = _format_date_for_display(oos_end) if oos_end else None

    # Generate OOS info HTML if needed
    # Prefer explicit oos_start/oos_end, fallback to actual_start/actual_end if different from train period
    if oos_start_display and oos_end_display:
        # Use explicit OOS time range
        has_oos = True
        oos_info_html = f'<li><strong>测试期 (OOS):</strong> {oos_start_display} 至 {oos_end_display}</li>'
    elif actual_start_display and actual_end_display and (
            actual_start_display != train_start_display
            or actual_end_display != train_end_display):
        # Fallback: use actual_start/actual_end if different from train period
        has_oos = True
        oos_info_html = f'<li><strong>测试期 (OOS):</strong> {actual_start_display} 至 {actual_end_display}</li>'
    else:
        has_oos = False
        oos_info_html = ''

    # Build title with symbol and time ranges
    # Use original symbol for display (with commas if multi-asset)
    symbol_display = symbol_raw if 'symbol_raw' in locals(
    ) else symbol.replace("_", ",")
    title_parts = [f"Training Summary Report - {symbol_display}"]
    if feature_types_str != "unknown":
        title_parts.append(f"Features: {feature_types_str}")
    if train_start_str and train_end_str:
        title_parts.append(f"Train: {train_start_str} to {train_end_str}")
    if actual_start_str and actual_end_str and (
            actual_start_str != train_start_str
            or actual_end_str != train_end_str):
        title_parts.append(f"Test: {actual_start_str} to {actual_end_str}")
    report_title = " | ".join(title_parts)

    # Collect unique timeframes and forward_bars for display
    unique_timeframes = set()
    unique_forward_bars = set()
    if "timeframe" in df.columns:
        unique_timeframes = set(df["timeframe"].dropna().unique())
    if "forward_bars" in df.columns:
        unique_forward_bars = set(df["forward_bars"].dropna().unique())
    
    # Collect all issues and feature importance across all configurations
    all_issues = []
    all_feature_importance: dict[str, dict[str, list[float]]] = {}
    
    # Helper functions for generating sections
    def _generate_issues_section(issues_list):
        if not issues_list:
            return ''
        issues_html = '\n'.join([f'<li style="color:red;font-weight:bold;">{issue}</li>' for issue in issues_list])
        return f'''
<h2>⚠️ 异常信号汇总 (Issues Summary)</h2>
<div style="background-color:#fff3cd;border-left:4px solid #ffc107;padding:15px;margin:20px 0;border-radius:4px;">
<p><strong>发现的问题:</strong></p>
<ul style="margin:10px 0;padding-left:20px;">
{issues_html}
</ul>
</div>'''
    
    def _accumulate_feature_importance(category: str,
                                       records: Optional[list]) -> None:
        if not records:
            return
        cat_dict = all_feature_importance.setdefault(category, {})
        for record in records:
            if not isinstance(record, dict):
                continue
            feat_name = record.get("feature")
            if not feat_name:
                continue
            importance = record.get("importance")
            try:
                importance_val = float(importance)
            except (TypeError, ValueError):
                continue
            cat_dict.setdefault(feat_name, []).append(importance_val)

    def _generate_feature_importance_section(
            feat_imp_dict: dict[str, dict[str, list[float]]]) -> str:
        if not feat_imp_dict:
            return ''

        label_map = {
            "classification": "Directional Classification",
            "return": "Return Regression",
            "volatility": "Volatility Regression",
        }

        sections = []
        for category, feats in feat_imp_dict.items():
            if not feats:
                continue
            sorted_features = sorted(
                feats.items(),
                key=lambda x: np.mean(x[1]) if x[1] else 0.0,
                reverse=True)[:20]
            if not sorted_features:
                continue
            rows = []
            for idx, (feat_name, values) in enumerate(sorted_features, start=1):
                avg_val = np.mean(values) if values else 0.0
                std_val = np.std(values) if values else 0.0
                rows.append(
                    f'<tr><td style="padding:8px;">{idx}</td>'
                    f'<td style="padding:8px;"><strong>{feat_name}</strong></td>'
                    f'<td style="padding:8px;">{avg_val:.2f}</td>'
                    f'<td style="padding:8px;">{std_val:.2f}</td></tr>'
                )
            if rows:
                sections.append(f'''
<h3 style="margin-top:20px;">{label_map.get(category, category.title())}</h3>
<table style="width:100%;margin:10px 0;font-size:0.9em;">
<tr><th style="background:#3498db;color:#fff;padding:10px;">排名</th><th style="background:#3498db;color:#fff;padding:10px;">特征名称</th><th style="background:#3498db;color:#fff;padding:10px;">平均重要性</th><th style="background:#3498db;color:#fff;padding:10px;">标准差</th></tr>
{''.join(rows)}
</table>''')

        if not sections:
            return ''

        return f'''
<h2>📊 特征重要性汇总 (Feature Importance Summary)</h2>
<div style="background-color:#e8f4f8;border-left:4px solid #3498db;padding:15px;margin:20px 0;border-radius:4px;">
<p><strong>说明:</strong> 以下为所有配置的特征重要性统计，按类别（方向/收益/波动）展示 Top 20 特征。</p>
{''.join(sections)}
</div>'''
    
    # Build rows
    rows = []
    # Store per-symbol metrics for each configuration
    per_symbol_sections = []
    for _, row in df.iterrows():
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        timeframe = row_dict.get("timeframe", "N/A")
        forward_bars = row_dict.get("forward_bars", "N/A")
        # Use symbol from row (should be same for all rows in same training run)
        # If multi-asset, symbol will be "BTCUSDT,ETHUSDT,SOLUSDT"
        symbol_row = row_dict.get("symbol", "N/A")
        config_dir = row_dict.get("config_dir", "N/A")
        metrics = row_dict.get("metrics", {}) or {}
        model_type = row_dict.get("model_type", "quantile")  # Default to quantile for backward compatibility
        
        # Extract per-symbol metrics from oos_metrics if available
        oos_metrics = row_dict.get("oos_metrics", {}) or {}
        per_symbol_metrics = oos_metrics.get("per_symbol", {}) if isinstance(oos_metrics, dict) else {}
        
        # Extract metrics based on model type
        if model_type == "classification":
            # Classification model: extract metrics from classification, return, and volatility models
            classification_metrics = metrics.get("classification", {}) if isinstance(metrics, dict) else {}
            classification_tf = classification_metrics.get(timeframe, {}) if isinstance(classification_metrics, dict) else {}
            
            return_metrics = metrics.get("return", {}) if isinstance(metrics, dict) else {}
            return_tf = return_metrics.get(timeframe, {}) if isinstance(return_metrics, dict) else {}
            
            volatility_metrics = metrics.get("volatility", {}) if isinstance(metrics, dict) else {}
            vol_tf = volatility_metrics.get(timeframe, {}) if isinstance(volatility_metrics, dict) else {}

            _accumulate_feature_importance(
                "classification", classification_tf.get("feature_importance"))
            _accumulate_feature_importance(
                "return", return_tf.get("feature_importance"))
            _accumulate_feature_importance(
                "volatility", vol_tf.get("feature_importance"))
            
            # Classification model metrics (classification task, not regression)
            classification_cv_accuracy = classification_tf.get("cv_accuracy")
            classification_cv_precision = classification_tf.get("cv_precision")
            classification_cv_recall = classification_tf.get("cv_recall")
            classification_cv_f1 = classification_tf.get("cv_f1")
            classification_cv_auc = classification_tf.get("cv_auc")
            classification_cv_pr_auc = classification_tf.get("cv_pr_auc")
            
            # Return regression model metrics
            return_cv_rmse = return_tf.get("cv_rmse")
            return_cv_mse = return_tf.get("cv_mse")
            return_cv_r2 = return_tf.get("cv_r2")
            
            # Volatility model metrics
            vol_cv_rmse = vol_tf.get("cv_rmse")
            vol_cv_mse = vol_tf.get("cv_mse")
            
            # Use volatility RMSE/MSE for main table (backward compatibility)
            cv_rmse = vol_cv_rmse
            cv_mse = vol_cv_mse
        else:
            # Quantile model: use existing logic
            volatility_metrics = metrics.get("volatility", {}) if isinstance(
                metrics, dict) else {}
            vol_tf = volatility_metrics.get(timeframe, {}) if isinstance(
                volatility_metrics, dict) else {}
            _accumulate_feature_importance(
                "volatility", vol_tf.get("feature_importance"))
            cv_rmse = vol_tf.get("cv_rmse")
            cv_mse = vol_tf.get("cv_mse")
            # Set classification/return metrics to None for quantile model
            classification_cv_accuracy = None
            classification_cv_precision = None
            classification_cv_recall = None
            classification_cv_f1 = None
            classification_cv_auc = None
            classification_cv_pr_auc = None
            return_cv_rmse = None
            return_cv_mse = None
            return_cv_r2 = None
            vol_cv_rmse = cv_rmse
            vol_cv_mse = cv_mse
        # Quantile loss metrics: q10, q50 (stage2), q90
        q10_metrics = metrics.get("q10", {}) if isinstance(metrics,
                                                           dict) else {}
        q10_tf = q10_metrics.get(timeframe, {}) if isinstance(
            q10_metrics, dict) else {}
        cv_quantile_loss_0_1 = q10_tf.get("cv_quantile_loss")
        stage2_metrics = metrics.get("stage2", {}) if isinstance(
            metrics, dict) else {}
        q50_tf = stage2_metrics.get(timeframe, {}) if isinstance(
            stage2_metrics, dict) else {}
        cv_quantile_loss_0_5 = q50_tf.get("cv_quantile_loss")
        q90_metrics = metrics.get("q90", {}) if isinstance(metrics,
                                                           dict) else {}
        q90_tf = q90_metrics.get(timeframe, {}) if isinstance(
            q90_metrics, dict) else {}
        cv_quantile_loss_0_9 = q90_tf.get("cv_quantile_loss")
        # Fallback 1: derive from volatility fold_details if missing
        if (cv_rmse is None or cv_mse is None) and isinstance(
                vol_tf.get("fold_details"), list):
            try:
                rmses = [
                    fd.get("rmse") for fd in vol_tf["fold_details"]
                    if isinstance(fd, dict) and fd.get("rmse") is not None
                ]
                mses = [
                    fd.get("mse") for fd in vol_tf["fold_details"]
                    if isinstance(fd, dict) and fd.get("mse") is not None
                ]
                if cv_rmse is None and rmses:
                    cv_rmse = float(sum(rmses) / len(rmses))
                if cv_mse is None and mses:
                    cv_mse = float(sum(mses) / len(mses))
            except Exception:
                pass
        # Fallback 2: derive from stage2 fold_details if still missing (quantile model may have rmse)
        if (cv_rmse is None or cv_mse is None):
            stage2_metrics = metrics.get("stage2", {}) if isinstance(
                metrics, dict) else {}
            tf_m = stage2_metrics.get(timeframe, {}) if isinstance(
                stage2_metrics, dict) else {}
            if isinstance(tf_m.get("fold_details"), list):
                try:
                    rmses = [
                        fd.get("rmse") for fd in tf_m["fold_details"]
                        if isinstance(fd, dict) and fd.get("rmse") is not None
                    ]
                    mses = [
                        fd.get("mse") for fd in tf_m["fold_details"]
                        if isinstance(fd, dict) and fd.get("mse") is not None
                    ]
                    if cv_rmse is None and rmses:
                        cv_rmse = float(sum(rmses) / len(rmses))
                    if cv_mse is None and mses:
                        cv_mse = float(sum(mses) / len(mses))
                except Exception:
                    pass
        # Extract directional metrics (derived from q50 regression)
        directional_cv_metrics = metrics.get(
            "directional_cv", {}) if isinstance(metrics, dict) else {}
        directional_cv_tf = directional_cv_metrics.get(
            timeframe, {}) if isinstance(directional_cv_metrics, dict) else {}
        # Fallback to directional_train if directional_cv not available
        if not directional_cv_tf:
            directional_train_metrics = metrics.get(
                "directional_train", {}) if isinstance(metrics, dict) else {}
            directional_cv_tf = directional_train_metrics.get(
                timeframe, {}) if isinstance(directional_train_metrics,
                                             dict) else {}

        f1 = directional_cv_tf.get("f1")
        acc = directional_cv_tf.get("accuracy")
        prec = directional_cv_tf.get("precision")
        rec = directional_cv_tf.get("recall")
        auc = directional_cv_tf.get("auc")
        pr_auc = directional_cv_tf.get("pr_auc")

        # Extract feature type and training bars
        feature_type = row_dict.get("feature_type", "N/A")
        train_bars = row_dict.get("train_bars") or row_dict.get(
            "total_bars", 0)

        # Extract model usability information
        model_usability = row_dict.get("model_usability", {}) or {}
        model_usable = model_usability.get(
            "usable", True)  # Default to True if not found
        
        # Extract issues from OOS metrics
        oos_issues = []
        if oos_metrics and isinstance(oos_metrics, dict):
            directional_oos = oos_metrics.get("directional_oos", {})
            if isinstance(directional_oos, dict):
                oos_acc = directional_oos.get("accuracy")
                oos_f1 = directional_oos.get("f1")
                oos_auc = directional_oos.get("auc")
                oos_ic_spearman = directional_oos.get("ic_spearman")
                if oos_acc is not None and oos_acc < 0.5:
                    oos_issues.append(f"{timeframe}/fb{forward_bars}: OOS准确率 {oos_acc*100:.2f}% < 50%")
                if oos_f1 is not None and oos_f1 < 0.5:
                    oos_issues.append(f"{timeframe}/fb{forward_bars}: OOS F1 {oos_f1*100:.2f}% < 50%")
                if oos_auc is not None and oos_auc < 0.5:
                    oos_issues.append(f"{timeframe}/fb{forward_bars}: OOS AUC {oos_auc*100:.2f}% < 50%")
                if oos_ic_spearman is not None and abs(oos_ic_spearman) < 0.05:
                    oos_issues.append(f"{timeframe}/fb{forward_bars}: OOS IC (Spearman) {oos_ic_spearman:.4f} < 0.05")
        
        # Collect issues for summary
        if oos_issues:
            all_issues.extend(oos_issues)
        
        # Extract feature importance
        directional_cv_metrics = metrics.get("directional_cv", {}) if isinstance(metrics, dict) else {}
        directional_cv_tf = directional_cv_metrics.get(timeframe, {}) if isinstance(directional_cv_metrics, dict) else {}
        feature_importance_list = directional_cv_tf.get("feature_importance")
        _accumulate_feature_importance("classification",
                                       feature_importance_list)

        # Helper functions
        def _format_metric(val, fmt=".4f"):
            if val is None:
                return "N/A"
            try:
                return f"{val:{fmt}}"
            except:
                return str(val)

        def _metric_cell(val,
                         *,
                         fmt=".4f",
                         warn_func=None,
                         default_style="padding:4px;"):
            text = _format_metric(val, fmt)
            if val is None or warn_func is None:
                return f'<td style="{default_style}">{text}</td>'
            try:
                warn = warn_func(val)
            except Exception:
                warn = False
            if warn:
                return (f'<td style="{default_style} color:#721c24; '
                        f'font-weight:bold;">{text}</td>')
            return f'<td style="{default_style}">{text}</td>'

        def _quality_color(val, threshold_good, threshold_excellent=None):
            if val is None:
                return ""
            if threshold_good is None:
                return ""
            if threshold_excellent is not None and val >= threshold_excellent:
                return ' style="background-color:#d4edda; color:#155724;"'
            if val >= threshold_good:
                return ' style="background-color:#fff3cd; color:#856404;"'
            return ' style="background-color:#f8d7da; color:#721c24;"'

        # Quality assessment: combine directional metrics AND model usability
        f1_color = _quality_color(f1, 0.3, 0.5)
        auc_color = _quality_color(auc, 0.6, 0.7) if auc is not None else ""
        pr_auc_color = _quality_color(pr_auc, 0.4,
                                      0.6) if pr_auc is not None else ""

        # F1 = 0 或 None 应该标记为不可用（F1 = 0 表示模型完全没有预测能力）
        # 阈值: F1 > 0.3 为良好，F1 > 0.5 为优秀
        # ⚠️ CRITICAL: F1=0 时即使 AUC 很高也应标记为不可用（F1=0 意味着模型无法预测"涨"）
        f1_valid = f1 is not None and f1 > 0.0
        # Quality check: F1>0 且 F1>=0.3 或 AUC>=0.6
        quality_passed = (f1_valid and f1 >= 0.3) or (auc is not None
                                                      and auc >= 0.6)
        # ⚠️ But if F1=0, quality should fail regardless of AUC (F1=0 means no "up" predictions)
        if f1 == 0.0:
            quality_passed = False
        
        # ⚠️ CRITICAL: Check OOS metrics - if OOS performance is poor, mark as unusable
        # OOS metrics are more important than CV metrics for model usability
        oos_quality_passed = True
        if oos_metrics and isinstance(oos_metrics, dict):
            directional_oos = oos_metrics.get("directional_oos", {})
            if isinstance(directional_oos, dict):
                oos_acc = directional_oos.get("accuracy")
                oos_f1 = directional_oos.get("f1")
                oos_auc = directional_oos.get("auc")
                oos_ic_spearman = directional_oos.get("ic_spearman")
                
                # OOS准确率 < 50% → 不可用（模型在样本外表现比随机猜测还差）
                if oos_acc is not None and oos_acc < 0.5:
                    oos_quality_passed = False
                # OOS AUC < 50% → 不可用（模型在样本外无法区分涨跌）
                if oos_auc is not None and oos_auc < 0.5:
                    oos_quality_passed = False
                # OOS F1 < 0.5 → 不可用（模型在样本外预测能力不足）
                if oos_f1 is not None and oos_f1 < 0.5:
                    oos_quality_passed = False
                # OOS IC (Spearman) < 0.05 → 不可用（预测与真实收益相关性太低）
                if oos_ic_spearman is not None and abs(oos_ic_spearman) < 0.05:
                    oos_quality_passed = False
        
        # Quality must pass BOTH:
        # 1. CV directional metrics AND model usability
        # 2. OOS metrics (if available)
        quality_passed = quality_passed and model_usable and oos_quality_passed
        quality_badge = (
            '<span style="background-color:#d4edda; color:#155724; padding:2px 6px; border-radius:4px;">✅ 可用</span>'
            if quality_passed else
            '<span style="background-color:#f8d7da; color:#721c24; padding:2px 6px; border-radius:4px;">❌ 不可用</span>'
        )

        cv_rmse_color = _quality_color(cv_rmse, None, None) if cv_rmse else ""

        # Add row styling for unusable models
        row_style = ' style="background-color:#ffe6e6;"' if not model_usable else ''

        # Extract OOS metrics for classification model
        oos_classification_metrics = None
        oos_return_metrics = None
        oos_volatility_metrics = None
        if model_type == "classification" and oos_metrics:
            oos_classification_metrics = oos_metrics.get("directional_oos", {}) if isinstance(oos_metrics, dict) else {}
            oos_return_metrics = oos_metrics.get("regression_return", {}) if isinstance(oos_metrics, dict) else {}
            oos_volatility_metrics = oos_metrics.get("regression_volatility", {}) if isinstance(oos_metrics, dict) else {}
        
        # Build row based on model type
        if model_type == "classification":
            # Classification model: show separate metrics for three models in sub-tables
            # CV metrics sub-table
            cv_rows = [
                '<tr style="background-color:#e8f4f8;"><th style="padding:4px; text-align:left;">模型</th><th style="padding:4px;">指标</th><th style="padding:4px;">CV值</th></tr>',
                '<tr><td rowspan="5" style="padding:4px; vertical-align:top; font-weight:bold;">分类模型</td>'
                '<td style="padding:4px;">Accuracy</td>'
                f'{_metric_cell(classification_cv_accuracy, warn_func=lambda v: v < 0.5)}</tr>',
                '<tr><td style="padding:4px;">Precision</td>'
                f'{_metric_cell(classification_cv_precision, warn_func=lambda v: v < 0.3)}</tr>',
                '<tr><td style="padding:4px;">Recall</td>'
                f'{_metric_cell(classification_cv_recall, warn_func=lambda v: v < 0.3)}</tr>',
                '<tr><td style="padding:4px;">F1</td>'
                f'{_metric_cell(classification_cv_f1, warn_func=lambda v: v < 0.3)}</tr>',
                '<tr><td style="padding:4px;">AUC</td>'
                f'{_metric_cell(classification_cv_auc, warn_func=lambda v: v < 0.6)}</tr>',
                '<tr><td rowspan="3" style="padding:4px; vertical-align:top; font-weight:bold;">收益回归</td>'
                '<td style="padding:4px;">RMSE</td>'
                f'{_metric_cell(return_cv_rmse, fmt=".6f")}</tr>',
                '<tr><td style="padding:4px;">MSE</td>'
                f'{_metric_cell(return_cv_mse, fmt=".8f")}</tr>',
                '<tr><td style="padding:4px;">R²</td>'
                f'{_metric_cell(return_cv_r2)}</tr>',
                '<tr><td rowspan="2" style="padding:4px; vertical-align:top; font-weight:bold;">波动率模型</td>'
                '<td style="padding:4px;">RMSE</td>'
                f'{_metric_cell(vol_cv_rmse, fmt=".6f")}</tr>',
                '<tr><td style="padding:4px;">MSE</td>'
                f'{_metric_cell(vol_cv_mse, fmt=".8f")}</tr>',
            ]
            cv_subtable = (
                '<table style="width:100%; margin:5px 0; border-collapse:collapse; font-size:0.85em;">'
                f'{"".join(cv_rows)}'
                '</table>'
            )
            
            # OOS metrics sub-table (if available)
            oos_subtable = ""
            if oos_classification_metrics or oos_return_metrics or oos_volatility_metrics:
                oos_rows = []
                if oos_classification_metrics:
                    oos_acc = oos_classification_metrics.get("accuracy")
                    oos_f1 = oos_classification_metrics.get("f1")
                    oos_auc = oos_classification_metrics.get("auc")
                    oos_ic_spearman = oos_classification_metrics.get("ic_spearman")
                    if oos_acc is not None:
                        oos_rows.append(
                            "<tr><td rowspan=\"4\" style=\"padding:4px; vertical-align:top; font-weight:bold;\">分类模型</td>"
                            "<td style=\"padding:4px;\">Accuracy</td>"
                            f"{_metric_cell(oos_acc, warn_func=lambda v: v < 0.5)}</tr>")
                        oos_rows.append(
                            "<tr><td style=\"padding:4px;\">F1</td>"
                            f"{_metric_cell(oos_f1, warn_func=lambda v: v is not None and v < 0.5)}</tr>")
                        oos_rows.append(
                            "<tr><td style=\"padding:4px;\">AUC</td>"
                            f"{_metric_cell(oos_auc, warn_func=lambda v: v < 0.5)}</tr>")
                        oos_rows.append(
                            "<tr><td style=\"padding:4px;\">IC (Spearman)</td>"
                            f"{_metric_cell(oos_ic_spearman, warn_func=lambda v: abs(v) < 0.05)}</tr>")
                if oos_return_metrics:
                    oos_return_rmse = oos_return_metrics.get("rmse")
                    oos_return_mae = oos_return_metrics.get("mae")
                    oos_return_r2 = oos_return_metrics.get("r2")
                    if oos_return_rmse is not None:
                        oos_rows.append(f"<tr><td rowspan=\"3\" style=\"padding:4px; vertical-align:top; font-weight:bold;\">收益回归</td><td style=\"padding:4px;\">RMSE</td><td style=\"padding:4px;\">{_format_metric(oos_return_rmse, '.6f')}</td></tr>")
                        oos_rows.append(f"<tr><td style=\"padding:4px;\">MAE</td><td style=\"padding:4px;\">{_format_metric(oos_return_mae, '.6f')}</td></tr>")
                        oos_rows.append(f"<tr><td style=\"padding:4px;\">R²</td><td style=\"padding:4px;\">{_format_metric(oos_return_r2)}</td></tr>")
                if oos_volatility_metrics:
                    oos_vol_rmse = oos_volatility_metrics.get("rmse")
                    oos_vol_mae = oos_volatility_metrics.get("mae")
                    if oos_vol_rmse is not None:
                        oos_rows.append(f"<tr><td rowspan=\"2\" style=\"padding:4px; vertical-align:top; font-weight:bold;\">波动率模型</td><td style=\"padding:4px;\">RMSE</td><td style=\"padding:4px;\">{_format_metric(oos_vol_rmse, '.6f')}</td></tr>")
                        oos_rows.append(f"<tr><td style=\"padding:4px;\">MAE</td><td style=\"padding:4px;\">{_format_metric(oos_vol_mae, '.6f')}</td></tr>")
                
                if oos_rows:
                    oos_subtable = f"""
                    <details style="margin-top:5px;">
                    <summary style="cursor:pointer; color:#3498db; font-weight:bold;">📊 OOS指标 (点击展开)</summary>
                    <table style="width:100%; margin:5px 0; border-collapse:collapse; font-size:0.85em;">
                    <tr style="background-color:#fff3cd;"><th style="padding:4px; text-align:left;">模型</th><th style="padding:4px;">指标</th><th style="padding:4px;">OOS值</th></tr>
                    {''.join(oos_rows)}
                    </table>
                    </details>"""
            
            model_metrics_html = f"<div style=\"max-width:400px;\">{cv_subtable}{oos_subtable}</div>"
            
            rows.append(
                f"<tr{row_style}><td>{symbol_row}</td><td>{timeframe}</td><td>{forward_bars}</td>"
                f"<td>{train_bars:,}</td>"
                f"<td>{model_metrics_html}</td>"
                f"<td{f1_color}>{_format_metric(f1)}</td>"
                f"<td>{_format_metric(acc)}</td>"
                f"<td>{_format_metric(prec)}</td>"
                f"<td>{_format_metric(rec)}</td>"
                f"<td{auc_color}>{_format_metric(auc)}</td>"
                f"<td{pr_auc_color}>{_format_metric(pr_auc)}</td>"
                f"<td>{feature_type}</td>"
                f"<td>{quality_badge}</td>"
                f"<td>{config_dir}</td></tr>")
        else:
            # Quantile model: use existing format (CV RMSE and CV MSE in separate columns)
            rows.append(
                f"<tr{row_style}><td>{symbol_row}</td><td>{timeframe}</td><td>{forward_bars}</td>"
                f"<td>{train_bars:,}</td>"
                f"<td{cv_rmse_color}>{_format_metric(cv_rmse, '.6f')}</td>"
                f"<td>{_format_metric(cv_mse, '.8f')}</td>"
                f"<td{f1_color}>{_format_metric(f1)}</td>"
                f"<td>{_format_metric(acc)}</td>"
                f"<td>{_format_metric(prec)}</td>"
                f"<td>{_format_metric(rec)}</td>"
                f"<td{auc_color}>{_format_metric(auc)}</td>"
                f"<td{pr_auc_color}>{_format_metric(pr_auc)}</td>"
                f"<td>{feature_type}</td>"
                f"<td>{quality_badge}</td>"
                f"<td>{config_dir}</td></tr>")
        
        # Build per-symbol metrics section if available
        if per_symbol_metrics and isinstance(per_symbol_metrics, dict):
            symbol_rows_html = []
            for symbol_name, symbol_metrics in sorted(per_symbol_metrics.items()):
                if isinstance(symbol_metrics, dict):
                    symbol_rmse = symbol_metrics.get("rmse")
                    symbol_mae = symbol_metrics.get("mae")
                    symbol_acc = symbol_metrics.get("accuracy")
                    symbol_prec = symbol_metrics.get("precision")
                    symbol_rec = symbol_metrics.get("recall")
                    symbol_f1 = symbol_metrics.get("f1")
                    symbol_ic_spearman = symbol_metrics.get("ic_spearman")
                    symbol_ic_pearson = symbol_metrics.get("ic_pearson")
                    symbol_sharpe_like = symbol_metrics.get("sharpe_like")
                    symbol_samples = symbol_metrics.get("samples", 0)
                    
                    # Color coding for per-symbol metrics
                    symbol_f1_color = _quality_color(symbol_f1, 0.3, 0.5) if symbol_f1 is not None else ""
                    symbol_ic_color = _quality_color(symbol_ic_spearman, 0.05, 0.1) if symbol_ic_spearman is not None else ""
                    
                    symbol_rows_html.append(
                        f"<tr>"
                        f"<td><strong>{symbol_name}</strong></td>"
                        f"<td>{_format_metric(symbol_rmse, '.6f')}</td>"
                        f"<td>{_format_metric(symbol_mae, '.6f')}</td>"
                        f"<td{symbol_f1_color}>{_format_metric(symbol_f1)}</td>"
                        f"<td>{_format_metric(symbol_acc)}</td>"
                        f"<td>{_format_metric(symbol_prec)}</td>"
                        f"<td>{_format_metric(symbol_rec)}</td>"
                        f"<td{symbol_ic_color}>{_format_metric(symbol_ic_spearman)}</td>"
                        f"<td>{_format_metric(symbol_ic_pearson)}</td>"
                        f"<td>{_format_metric(symbol_sharpe_like, '.4f')}</td>"
                        f"<td>{symbol_samples:,}</td>"
                        f"</tr>"
                    )
            
            if symbol_rows_html:
                config_key = f"{timeframe}_fb{forward_bars}_{config_dir}"
                per_symbol_sections.append({
                    "config_key": config_key,
                    "timeframe": timeframe,
                    "forward_bars": forward_bars,
                    "config_dir": config_dir,
                    "rows": symbol_rows_html
                })

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>{report_title}</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#222;background-color:#f5f5f5}}
.container{{max-width:1600px;margin:0 auto;background-color:white;padding:30px;border-radius:10px;box-shadow:0 0 20px rgba(0,0,0,0.1)}}
h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
h2{{color:#34495e;border-left:4px solid #3498db;padding-left:15px;margin-top:30px}}
.explanation{{background-color:#fff3cd;border-left:4px solid #ffc107;padding:15px;margin:20px 0}}
table{{border-collapse:collapse;width:100%;margin:20px 0;font-size:0.9em}}
th,td{{border:1px solid #ddd;padding:10px;text-align:left}}
th{{background:#3498db;color:#fff;position:sticky;top:0}}
tr:nth-child(even){{background-color:#f2f2f2}}
tr:hover{{background-color:#e8f4f8}}
.good{{background-color:#d4edda;color:#155724}}
.warn{{background-color:#fff3cd;color:#856404}}
.bad{{background-color:#f8d7da;color:#721c24}}
</style>
</head><body>
<div class="container">
<h1>{report_title}</h1>
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div style="background-color:#e8f4f8;border-left:4px solid #3498db;padding:15px;margin:20px 0;border-radius:4px;">
<h3 style="margin-top:0;color:#2c3e50;">📅 训练时间范围</h3>
<ul style="margin:10px 0;padding-left:20px;">
<li><strong>训练期:</strong> {train_start_display if train_start_display else 'N/A'} 至 {train_end_display if train_end_display else 'N/A'}</li>
{oos_info_html}
<li><strong>特征类型:</strong> {feature_types_str if feature_types_str != "unknown" else 'N/A'}</li>
<li><strong>交易对:</strong> {symbol_display if 'symbol_display' in locals() else symbol.replace("_", ",")}</li>
<li><strong>时间框架 (Timeframe):</strong> {', '.join(sorted([str(tf) for tf in unique_timeframes])) if unique_timeframes else 'N/A'}</li>
<li><strong>前向周期 (Forward Bars):</strong> {', '.join(sorted([str(fb) for fb in unique_forward_bars])) if unique_forward_bars else 'N/A'}</li>
</ul>
</div>
<div class="explanation">
<h3>📊 指标说明与好坏判断</h3>
<ul>
<li><strong>CV RMSE</strong> (交叉验证均方根误差): 收益回归模型的预测误差，越低越好
    <ul>
        <li>✅ 优秀: RMSE 较小，说明预测精度高</li>
        <li>⚠️ 注意: 如果RMSE过大，说明模型预测不准确</li>
    </ul>
</li>
<li><strong>CV MSE</strong> (交叉验证均方误差): 收益回归模型的预测误差平方，越低越好</li>
<li><strong>F1</strong> (F1 Score): 分类模型的方向性预测F1分数，综合精确率和召回率
    <ul>
        <li>阈值: F1 &gt; 0.3 为良好，F1 &gt; 0.5 为优秀</li>
        <li>⚠️ <strong>F1 = 0 或 None</strong>：模型完全无预测能力，自动标记为不可用</li>
        <li>反映模型对方向（涨/跌）的预测能力</li>
    </ul>
</li>
<li><strong>Acc</strong> (Accuracy): 分类模型的方向性预测准确率
    <ul>
        <li>反映模型预测方向的正确率</li>
        <li>Acc &gt; 0.5 表示模型有预测能力</li>
    </ul>
</li>
<li><strong>Prec</strong> (Precision): 分类模型的方向性预测精确率
    <ul>
        <li>反映模型预测为"涨"时，实际确实是"涨"的比例</li>
    </ul>
</li>
<li><strong>Rec</strong> (Recall): 分类模型的方向性预测召回率
    <ul>
        <li>反映模型捕获所有"涨"的情况的能力</li>
    </ul>
</li>
<li><strong>AUC</strong> (ROC AUC): 方向性预测ROC曲线下面积
    <ul>
        <li>阈值: AUC &gt; 0.6 为良好，AUC &gt; 0.7 为优秀</li>
        <li>反映模型区分"涨"和"跌"的能力</li>
        <li>⚠️ <strong>注意</strong>：仅AUC优秀不足以判断模型质量，需要结合其他指标综合评估</li>
    </ul>
</li>
<li><strong>PR-AUC</strong> (Precision-Recall AUC): 方向性预测PR曲线下面积
    <ul>
        <li>阈值: PR-AUC &gt; 0.4 为良好，PR-AUC &gt; 0.6 为优秀</li>
        <li>在不平衡数据集上比ROC AUC更有意义</li>
        <li>⚠️ <strong>注意</strong>：仅PR-AUC优秀不足以判断模型质量，需要结合其他指标综合评估</li>
    </ul>
</li>
<li><strong>Quality</strong>: 综合评估模型质量（CV指标 + OOS指标 + 模型可用性）
    <ul>
        <li>✅ <strong>可用</strong>: CV方向预测指标（F1≥0.3或AUC≥0.6）且模型可用 <strong>且</strong> OOS指标达标（OOS准确率≥50%、OOS AUC≥50%、OOS F1≥0.5、OOS IC≥0.05）</li>
        <li>⭐ <strong>优秀</strong>: 所有指标达到优秀水平（F1≥0.5且AUC≥0.7且PR-AUC≥0.6）时，模型质量才算过关</li>
        <li>❌ <strong>不可用</strong>: CV方向预测指标不达标（F1=0/None或F1&lt;0.3且AUC&lt;0.6）<strong>或</strong> 模型不可用 <strong>或</strong> OOS指标不达标（OOS准确率&lt;50%、OOS AUC&lt;50%、OOS F1&lt;0.5、OOS IC&lt;0.05）</li>
        <li>⚠️ <strong>重要</strong>: F1=0 表示模型完全没有预测能力，必须标记为不可用。OOS指标不达标表示模型在样本外表现差，也必须标记为不可用。不可用的模型不应用于实际预测，需要重新训练或检查数据质量</li>
        <li>💡 <strong>评估建议</strong>: 综合评估CV和OOS的F1、AUC、PR-AUC、IC等多个指标，所有指标达到优秀水平时，模型质量才算过关。单一指标优秀不足以判断模型质量。OOS指标比CV指标更重要，因为OOS指标反映模型在真实场景中的表现。</li>
    </ul>
</li>
<li><strong>模型架构</strong>: 使用3个模型（Classification, Return Regression, Volatility）的组合架构
    <ul>
        <li><strong>Classification模型</strong>: 预测未来涨跌方向的概率（0-1之间，0.5为阈值），用于判断方向（涨/跌）</li>
        <li><strong>Return Regression模型</strong>: 预测未来收益率的幅度，用于估计潜在收益（方向正确时的幅度）</li>
        <li><strong>Volatility模型</strong>: 预测未来波动率，用于风险调整和仓位管理</li>
        <li><strong>决策逻辑</strong>: signal_strength = p_up * expected_return / expected_volatility
            <ul>
                <li>p_up: 分类模型预测的上涨概率</li>
                <li>expected_return: 收益回归模型预测的期望收益</li>
                <li>expected_volatility: 波动率模型预测的期望波动率</li>
                <li>signal_strength: 风险调整后的信号强度，用于仓位分配和交易决策</li>
            </ul>
        </li>
        <li><strong>使用说明</strong>: 
            <ul>
                <li><strong>分类模型评估</strong>: 关注Accuracy、F1、AUC、PR-AUC等指标。F1≥0.5且AUC≥0.7且PR-AUC≥0.6时，模型质量才算过关。</li>
                <li><strong>收益回归模型评估</strong>: 关注RMSE、MSE、R²等指标。RMSE越小，预测精度越高。</li>
                <li><strong>波动率模型评估</strong>: 关注RMSE、MSE等指标。波动率预测准确有助于风险管理和仓位控制。</li>
                <li><strong>综合评估</strong>: 需要同时评估CV指标和OOS指标。OOS指标比CV指标更重要，因为OOS指标反映模型在真实场景中的表现。</li>
                <li><strong>信号生成</strong>: 使用三个模型的输出计算风险调整后的信号强度，可以用于：
                    <ul>
                        <li>仓位分配：根据signal_strength的大小决定仓位大小</li>
                        <li>交易决策：signal_strength > 阈值时开仓，否则不开仓</li>
                        <li>风险管理：结合expected_volatility进行止损和止盈设置</li>
                    </ul>
                </li>
            </ul>
        </li>
    </ul>
</li>
</ul>
<h3>💡 特征类型比较</h3>
<ul>
<li><strong>baseline</strong>: 基础特征（传统技术指标），特征数量较少，计算快，适合快速验证</li>
<li><strong>default</strong>: 默认特征（TA-Lib + base_indicators），比baseline更丰富，推荐用于生产环境</li>
<li><strong>enhanced</strong>: 增强特征（包含更多高级特征），特征更全面但计算更慢</li>
<li><strong>comprehensive</strong>: 综合特征（最完整，包含所有特征类型），特征最丰富但可能过拟合</li>
</ul>
<p><strong>💡 建议:</strong> 比较不同特征类型时，重点关注分类模型的F1、AUC、PR-AUC和收益回归模型的RMSE、R²，以及OOS指标，这些指标更能反映模型的真实性能。</p>
</div>
<table>
<tr><th>Symbol</th><th>Timeframe</th><th>Forward Bars</th><th>Training Bars</th><th style="min-width:400px;">模型指标 (Model Metrics)<br/><em>CV指标 + OOS指标（可展开）</em></th><th>F1</th><th>Acc</th><th>Prec</th><th>Rec</th><th>AUC</th><th>PR-AUC</th><th>Feature Type</th><th>Quality</th><th>Config</th></tr>
{''.join(rows)}
</table>
{chr(10).join([f'''
<h2>📊 按标的 OOS 指标 (Per-Symbol OOS Metrics) - {section["timeframe"]} / Forward Bars: {section["forward_bars"]} / Config: {section["config_dir"]}</h2>
<div style="background-color:#e8f4f8;border-left:4px solid #3498db;padding:15px;margin:20px 0;border-radius:4px;">
<p><strong>说明:</strong> 以下指标为样本外（OOS）测试期间每个标的的独立表现。这些指标可以帮助识别模型在不同资产上的表现差异。</p>
</div>
<table>
<tr><th>Symbol</th><th>RMSE</th><th>MAE</th><th>F1</th><th>Acc</th><th>Prec</th><th>Rec</th><th>IC (Spearman)</th><th>IC (Pearson)</th><th>Sharpe-like</th><th>Samples</th></tr>
{chr(10).join(section["rows"])}
</table>
''' for section in per_symbol_sections]) if per_symbol_sections else ''}
{_generate_issues_section(all_issues)}
{_generate_feature_importance_section(all_feature_importance)}
</div>
</body></html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    # Print with feature types and time range info
    print(f"Summary report generated: {output_path}")
    if feature_types_str != "unknown":
        print(f"  Feature types: {feature_types_str}")
    if train_start_str and train_end_str:
        print(f"  Training period: {train_start_str} to {train_end_str}")
    if actual_start_str and actual_end_str and (
            actual_start_str != train_start_str
            or actual_end_str != train_end_str):
        print(f"  Test period: {actual_start_str} to {actual_end_str}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate training summary report")
    parser.add_argument("--results-dir", type=str, default="results/training")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    generate_summary_report(args.results_dir, args.output)
