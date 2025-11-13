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


def _format_percent(val, digits: int = 2) -> str:
    if val is None:
        return "NA"
    try:
        return f"{float(val) * 100:.{digits}f}%"
    except Exception:
        return "NA"


PERCENT_METRICS = {
    "accuracy",
    "win_rate",
    "long_win_rate",
    "short_win_rate",
    "active_ratio",
    "f1_macro",
    "f1_weighted",
    "f1_active_macro",
    "roc_auc_macro",
    "pr_auc_macro",
    "precision",
    "recall",
}


def _format_metric_for_display(metric: str, value) -> str:
    """Format metric value based on its semantic meaning."""
    if value is None:
        return "NA"
    try:
        if metric in PERCENT_METRICS:
            return _format_percent(float(value), 2)
        return _format_float(float(value))
    except Exception:
        return str(value)


def _format_metric_delta(metric: str, delta) -> str:
    if delta is None:
        return "NA"
    try:
        if metric in PERCENT_METRICS:
            return _format_percent(float(delta), 2)
        return _format_float(float(delta))
    except Exception:
        return str(delta)


def _build_classification_metrics_table(
    stage_baseline: Dict,
    stage_candidate: Dict,
    baseline_label: str,
    candidate_label: str,
) -> str:
    base_fin = stage_baseline.get("financial_metrics", {})
    cand_fin = stage_candidate.get("financial_metrics", {})
    base_cls = stage_baseline.get("classification_metrics", {})
    cand_cls = stage_candidate.get("classification_metrics", {})

    def _row(label, base_val, cand_val, is_percent: bool = False):
        base_fmt = (_format_percent(base_val)
                    if is_percent else _format_float(base_val))
        cand_fmt = (_format_percent(cand_val)
                    if is_percent else _format_float(cand_val))
        if base_val is not None and cand_val is not None:
            delta_val = cand_val - base_val
            delta_fmt = (_format_percent(delta_val)
                         if is_percent else _format_float(delta_val))
        else:
            delta_fmt = "NA"
        return (
            f"<tr><td>{label}</td>"
            f"<td>{base_fmt}</td><td>{cand_fmt}</td><td>{delta_fmt}</td></tr>")

    rows = [
        _row("Directional Win Rate",
             base_fin.get("win_rate"),
             cand_fin.get("win_rate"),
             is_percent=True),
        _row("Active Ratio",
             base_fin.get("active_ratio"),
             cand_fin.get("active_ratio"),
             is_percent=True),
        _row("F1 (Macro)", base_cls.get("f1_macro"), cand_cls.get("f1_macro")),
        _row("F1 (Weighted)", base_cls.get("f1_weighted"),
             cand_cls.get("f1_weighted")),
        _row("Accuracy", base_cls.get("accuracy"), cand_cls.get("accuracy")),
        _row("ROC AUC (Macro)", base_cls.get("roc_auc_macro"),
             cand_cls.get("roc_auc_macro")),
        _row("PR AUC (Macro)", base_cls.get("pr_auc_macro"),
             cand_cls.get("pr_auc_macro")),
    ]

    rows_html = "".join(rows)
    return f"""
    <div class="card">
        <h3>Classification Metrics Comparison</h3>
        <table class="metric-table">
            <tr><th>Metric</th><th>{baseline_label}</th><th>{candidate_label}</th><th>Δ ({candidate_label}-{baseline_label})</th></tr>
            {rows_html}
        </table>
    </div>
    """


def _build_regression_metrics_table(
    stage_baseline: Dict,
    stage_candidate: Dict,
    baseline_label: str,
    candidate_label: str,
) -> str:
    rows = []
    for metric in ("r2", "rmse", "mae"):
        base_val = stage_baseline.get(metric)
        cand_val = stage_candidate.get(metric)
        if base_val is None and cand_val is None:
            continue
        base_fmt = _format_float(base_val)
        cand_fmt = _format_float(cand_val)
        delta_fmt = ("NA" if base_val is None or cand_val is None else
                     _format_float(cand_val - base_val))
        rows.append(
            f"<tr><td>{metric.upper()}</td><td>{base_fmt}</td><td>{cand_fmt}</td><td>{delta_fmt}</td></tr>"
        )

    if not rows:
        return ""

    return f"""
    <div class="card">
        <h3>Regression Metrics Comparison</h3>
        <table class="metric-table">
            <tr><th>Metric</th><th>{baseline_label}</th><th>{candidate_label}</th><th>Δ ({candidate_label}-{baseline_label})</th></tr>
            {''.join(rows)}
        </table>
    </div>
    """


def _build_confusion_matrix_html(class_metrics: Dict,
                                 title: str = "Confusion Matrix") -> str:
    if not class_metrics:
        return ""
    matrix = class_metrics.get("confusion_matrix")
    labels = class_metrics.get("labels")
    if matrix is None or labels is None:
        return ""

    # Convert labels to True/False for binary classification (0/1 -> False/True)
    # For multi-class, keep original labels
    def format_label(lbl):
        if isinstance(lbl, (int, float)):
            if lbl == 0:
                return "False"
            elif lbl == 1:
                return "True"
        # For multi-class or other labels, convert to string
        return str(lbl)

    formatted_labels = [format_label(lbl) for lbl in labels]
    header = "".join(f"<th>Predicted {lbl}</th>" for lbl in formatted_labels)
    body_rows = []
    for lbl, row in zip(formatted_labels, matrix):
        cells = "".join(f"<td>{int(val)}</td>" for val in row)
        body_rows.append(f"<tr><th>Actual {lbl}</th>{cells}</tr>")
    body_html = "".join(body_rows)

    # Calculate metrics for interpretation
    if len(matrix) == 2 and len(matrix[0]) == 2:
        # Binary classification
        tn = int(matrix[0][0])  # True Negative
        fp = int(matrix[0][1])  # False Positive
        fn = int(matrix[1][0])  # False Negative
        tp = int(matrix[1][1])  # True Positive

        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Determine interpretation
        if accuracy >= 0.75:
            accuracy_interpretation = "优秀"
            accuracy_color = "good"
        elif accuracy >= 0.65:
            accuracy_interpretation = "良好"
            accuracy_color = "good"
        else:
            accuracy_interpretation = "需要改进"
            accuracy_color = "bad"

        interpretation = f"""
        <div class="explanation" style="margin-top: 20px;">
            <h4>📊 如何阅读混淆矩阵</h4>
            <p>混淆矩阵展示了模型预测结果与实际标签的对比：</p>
            <ul>
                <li><strong>True Negative (TN)</strong>: {tn} - 正确预测为 False（实际 False，预测 False）</li>
                <li><strong>False Positive (FP)</strong>: {fp} - 错误预测为 True（实际 False，预测 True）- <span style="color: #d62728;">假正例</span></li>
                <li><strong>False Negative (FN)</strong>: {fn} - 错误预测为 False（实际 True，预测 False）- <span style="color: #d62728;">假负例</span></li>
                <li><strong>True Positive (TP)</strong>: {tp} - 正确预测为 True（实际 True，预测 True）</li>
            </ul>
            
            <h4>📈 关键指标</h4>
            <table style="margin: 10px 0;">
                <tr>
                    <th>指标</th>
                    <th>计算公式</th>
                    <th>数值</th>
                    <th>含义</th>
                </tr>
                <tr>
                    <td><strong>准确率 (Accuracy)</strong></td>
                    <td>(TP + TN) / 总数</td>
                    <td class="{accuracy_color}">{accuracy:.2%}</td>
                    <td>所有预测中正确的比例</td>
                </tr>
                <tr>
                    <td><strong>精确率 (Precision)</strong></td>
                    <td>TP / (TP + FP)</td>
                    <td>{precision:.2%}</td>
                    <td>预测为 True 中实际为 True 的比例（减少假正例）</td>
                </tr>
                <tr>
                    <td><strong>召回率 (Recall)</strong></td>
                    <td>TP / (TP + FN)</td>
                    <td>{recall:.2%}</td>
                    <td>实际为 True 中被正确预测的比例（减少假负例）</td>
                </tr>
                <tr>
                    <td><strong>特异性 (Specificity)</strong></td>
                    <td>TN / (TN + FP)</td>
                    <td>{specificity:.2%}</td>
                    <td>实际为 False 中被正确预测的比例</td>
                </tr>
            </table>
            
            <h4>💡 结论</h4>
            <p>
                <strong>整体表现：</strong>模型准确率为 <span class="{accuracy_color}">{accuracy:.2%}</span>，表现<span class="{accuracy_color}">{accuracy_interpretation}</span>。
            </p>
            <ul>
                <li>在 {total} 个样本中，模型正确预测了 {tp + tn} 个（{accuracy:.2%}）</li>
                <li>假正例 (FP): {fp} 个 - 模型错误地将 {fp} 个负样本预测为正样本</li>
                <li>假负例 (FN): {fn} 个 - 模型错误地将 {fn} 个正样本预测为负样本</li>
            </ul>
            <p>
                <strong>建议：</strong>
                {"模型表现优秀，可以用于生产环境。" if accuracy >= 0.75 else 
                 "模型表现良好，可以考虑进一步优化。" if accuracy >= 0.65 else 
                 "模型需要改进，建议检查特征工程或调整模型参数。"}
            </p>
        </div>
        """
    else:
        # Multi-class classification
        total = sum(sum(row) for row in matrix)
        correct = sum(matrix[i][i] for i in range(len(matrix)))
        accuracy = correct / total if total > 0 else 0

        interpretation = f"""
        <div class="explanation" style="margin-top: 20px;">
            <h4>📊 如何阅读混淆矩阵</h4>
            <p>混淆矩阵展示了模型预测结果与实际标签的对比：</p>
            <ul>
                <li>对角线上的数字表示<strong>正确预测</strong>的数量</li>
                <li>非对角线上的数字表示<strong>错误预测</strong>的数量</li>
            </ul>
            
            <h4>📈 关键指标</h4>
            <p><strong>准确率 (Accuracy):</strong> {accuracy:.2%} - 在 {total} 个样本中，模型正确预测了 {correct} 个</p>
        </div>
        """

    return f"""
    <div class="card">
        <h3>{title}</h3>
        <table class="metric-table">
            <tr><th></th>{header}</tr>
            {body_html}
        </table>
        {interpretation}
    </div>
    """


def _load_factor_preview(path: str, key: str, limit: int = 30) -> list[str]:
    """Load a preview list of factor names from JSON artifacts."""
    names: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        factors = data.get(key, [])
        if key == "top_factors":
            for item in factors:
                if isinstance(item, dict):
                    name = item.get("name")
                else:
                    name = str(item)
                if name:
                    names.append(str(name))
        else:
            names = [str(item) for item in factors if item]
    except Exception:
        return []
    return names[:limit]


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
    insights = results.get("insights", {})
    feature_effective = insights.get("effective")
    if feature_effective is True:
        feature_effective_display = "✅ Effective"
    elif feature_effective is False:
        feature_effective_display = "⚠️ Not Effective"
    else:
        feature_effective_display = "Unknown"
    recomm_horizon = insights.get("recommended_horizon")
    recomm_horizon_metric = insights.get("recommended_horizon_metric")
    recomm_horizon_metric_name = insights.get(
        "recommended_horizon_metric_name", "metric")
    recomm_horizon_effective = insights.get("recommended_horizon_effective")
    stage_label_map = {
        "stage1_all_features": "Stage 1: All Features",
        "stage2_ic_filtered": "Stage 2: IC-Filtered",
        "stage3_representatives": "Stage 3: Representatives",
        "stage4_compressed": "Stage 4: Compressed",
    }
    recommended_stage_key = insights.get("recommended_stage")
    recommended_stage_label = (stage_label_map.get(recommended_stage_key,
                                                   recommended_stage_key)
                               if recommended_stage_key else None)
    if recomm_horizon is not None:
        horizon_badge = ("✅ Effective"
                         if recomm_horizon_effective else "ℹ️ Best Candidate")
        horizon_metric_fmt = _format_metric_for_display(
            recomm_horizon_metric_name, recomm_horizon_metric)
        recommended_horizon_row = (
            "<tr><th>Recommended Forward Horizon</th>"
            f"<td>{int(recomm_horizon)} bars "
            f"({recomm_horizon_metric_name}: {horizon_metric_fmt}) "
            f"{horizon_badge}</td></tr>")
    else:
        recommended_horizon_row = ""
    html_dir = os.path.dirname(os.path.abspath(html_path))

    def _rel_path(target: str | None) -> str | None:
        if not target:
            return None
        try:
            return os.path.relpath(target, start=html_dir)
        except Exception:
            return target

    artifacts: Dict[str, any] = {}
    top_factors_path = d.get("top_factors_path") or results.get(
        "top_factors_path")
    representatives_path = d.get("representatives_path") or results.get(
        "representatives_path")
    shap_dir_path = (results.get("explainability", {}).get("stage3_shap_dir")
                     or results.get("selection", {}).get(
                         "explainability", {}).get("stage3_shap_dir"))

    if top_factors_path and os.path.exists(top_factors_path):
        artifacts["top_factors"] = _rel_path(top_factors_path)
        artifacts["top_factors_preview"] = _load_factor_preview(
            top_factors_path, "top_factors")
    else:
        artifacts["top_factors_preview"] = []

    if representatives_path and os.path.exists(representatives_path):
        artifacts["representatives"] = _rel_path(representatives_path)
        artifacts["representatives_preview"] = _load_factor_preview(
            representatives_path, "representative_factors")
    else:
        artifacts["representatives_preview"] = []

    shap_importance_preview: list[Dict] = []
    if shap_dir_path and os.path.exists(shap_dir_path):
        artifacts["shap"] = _rel_path(shap_dir_path)
        shap_importance_path = Path(
            shap_dir_path) / "stage3_representatives_shap_importance.json"
        if shap_importance_path.exists():
            try:
                with open(shap_importance_path, "r", encoding="utf-8") as f:
                    shap_importance_preview = json.load(f)[:15]
            except Exception as exc:
                print(f"⚠️ Failed to load SHAP importance: {exc}")

    # Support both old format (original/compressed) and new 4-stage format
    stage1 = p.get("stage1_all_features", p.get("original_features", {}))
    stage2 = p.get("stage2_ic_filtered", {})
    stage3 = p.get("stage3_representatives", {})
    stage4 = p.get("stage4_compressed", p.get("compressed_features", {}))

    # Legacy support
    orig = p.get("original_features", stage1)
    comp = p.get("compressed_features", stage4) or stage3
    orig_val = p.get("original_features_val", {})
    comp_val = p.get("compressed_features_val", {}) or {}

    # Get delta comparisons
    stage2_vs_1 = p.get("stage2_vs_stage1", {})
    stage3_vs_2 = p.get("stage3_vs_stage2", {})
    stage4_vs_3 = p.get("stage4_vs_stage3", {})
    delta_r2 = p.get("performance_change", stage4_vs_3.get("delta_r2"))

    compressed_dims = d.get("compressed_dimensions")
    has_4_stages = bool(
        stage4
        and (stage4.get("r2") is not None or stage4.get("rmse") is not None)
        and compressed_dims not in (None, 0, d.get("original_features_count")))
    if not has_4_stages:
        compressed_dims = d.get("stage3_representatives")

    # Use feature insights to determine if dimensionality reduction is beneficial
    # This uses the primary metric (win_rate, f1_macro, accuracy, or r2) based on task type
    feature_effective = insights.get("effective")
    feature_delta = insights.get("delta")
    feature_metric_name = insights.get("metric_name", "r2")

    # Determine conclusion based on feature effectiveness
    if feature_effective is True:
        conclusion = f"Dimensionality reduction appears beneficial. {feature_metric_name} improved by {feature_delta:.4f}."
    elif feature_effective is False and feature_delta is not None:
        conclusion = f"Dimensionality reduction shows mixed results. {feature_metric_name} changed by {feature_delta:.4f}."
    else:
        # Fallback to r2 delta if feature insights are not available
        conclusion_delta = delta_r2
        if not has_4_stages:
            if stage3_vs_2:
                conclusion_delta = stage3_vs_2.get("delta_r2",
                                                   conclusion_delta)
            elif stage2_vs_1:
                conclusion_delta = stage2_vs_1.get("delta_r2",
                                                   conclusion_delta)
        if conclusion_delta is not None and conclusion_delta > 0:
            conclusion = f"Dimensionality reduction appears beneficial. R² improved by {conclusion_delta:.4f}."
        else:
            conclusion = "Dimensionality reduction is not beneficial under this run. Consider reviewing feature selection or model parameters."

    # Extract financial metrics
    stage1_fin = stage1.get("financial_metrics", {})
    stage2_fin = stage2.get("financial_metrics", {})
    stage3_fin = stage3.get("financial_metrics", {})
    stage4_fin = stage4.get("financial_metrics", {})

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

    classification_section = ""
    confusion_html = ""
    if task_type.startswith("classification") and stage1 and stage3:
        classification_section = _build_classification_metrics_table(
            stage1,
            stage3,
            "Stage 1: All Features",
            "Stage 3: Representatives",
        )
        confusion_html = _build_confusion_matrix_html(
            stage3.get("classification_metrics"),
            title="Stage 3 Confusion Matrix (Test Set)",
        )

    insight_items: list[str] = []
    if feature_effective is True:
        insight_items.append(
            "✅ Representative features outperformed the baseline.")
    elif feature_effective is False:
        insight_items.append(
            "⚠️ Representative features did not beat the baseline.")
    else:
        insight_items.append(
            "ℹ️ Feature effectiveness could not be conclusively determined.")

    metric_name = insights.get("metric_name")
    metric_display_map = {
        "win_rate": "Directional Win Rate",
        "long_win_rate": "Long Win Rate",
        "short_win_rate": "Short Win Rate",
        "f1_macro": "F1 (Macro)",
        "f1_weighted": "F1 (Weighted)",
        "accuracy": "Accuracy",
        "roc_auc_macro": "ROC AUC",
        "pr_auc_macro": "PR AUC",
        "r2": "R²",
        "rmse": "RMSE",
        "mae": "MAE",
    }
    metric_display = metric_display_map.get(
        metric_name,
        metric_name.replace("_", " ").title()
        if isinstance(metric_name, str) else "Metric",
    )
    baseline_val = insights.get("baseline_value")
    candidate_val = insights.get("candidate_value")
    delta_val = insights.get("delta")
    if baseline_val is not None and candidate_val is not None:
        base_fmt = _format_metric_for_display(metric_name, baseline_val)
        cand_fmt = _format_metric_for_display(metric_name, candidate_val)
        delta_fmt = _format_metric_delta(metric_name, delta_val)
        insight_items.append(
            f"{metric_display}: {base_fmt} → {cand_fmt} (Δ {delta_fmt}).")

    if recommended_stage_label:
        insight_items.append(
            f"Recommended feature stage: {recommended_stage_label}.")

    if recomm_horizon is not None:
        horizon_badge = ("✅ Effective horizon" if recomm_horizon_effective else
                         "ℹ️ Horizon candidate")
        horizon_metric_fmt = _format_metric_for_display(
            recomm_horizon_metric_name, recomm_horizon_metric)
        insight_items.append(
            f"{horizon_badge}: {int(recomm_horizon)} bars ({recomm_horizon_metric_name}: {horizon_metric_fmt})."
        )

    insights_html = ""
    if insight_items:
        insights_html = (
            "<div class=\"card\">"
            "<h3>Insights Summary</h3>"
            f"<ul>{''.join(f'<li>{item}</li>' for item in insight_items)}</ul>"
            "</div>")

    # Build stability validation section
    stability_html = ""
    stability_validation = results.get("stability_validation")
    if stability_validation:
        val_period = stability_validation.get("validation_period", {})
        sel_period = stability_validation.get("selection_period", {})
        stable_factors = stability_validation.get("stable_factors", [])
        unstable_factors = stability_validation.get("unstable_factors", [])
        stability_rate = stability_validation.get("stability_rate", 0)
        ic_comparison = stability_validation.get("ic_comparison", {})

        # Build stable factors table
        stable_rows = ""
        if stable_factors:
            stable_sorted = sorted(
                stable_factors,
                key=lambda x: abs(
                    ic_comparison.get(x, {}).get("ic_selection", 0)),
                reverse=True)[:20]
            for factor in stable_sorted:
                comp = ic_comparison.get(factor, {})
                ic_sel = comp.get("ic_selection", 0)
                ic_val = comp.get("ic_validation", 0)
                ic_change = comp.get("ic_change", 0)
                stable_rows += f"""
                <tr>
                    <td>{factor}</td>
                    <td>{_format_float(ic_sel, 4)}</td>
                    <td>{_format_float(ic_val, 4)}</td>
                    <td class="{'good' if abs(ic_change) < 0.05 else 'warn'}">{_format_float(ic_change, 4)}</td>
                </tr>"""

        # Build unstable factors table
        unstable_rows = ""
        if unstable_factors:
            unstable_sorted = sorted(
                unstable_factors,
                key=lambda x: abs(
                    ic_comparison.get(x, {}).get("ic_change", 0)),
                reverse=True)[:10]
            for factor in unstable_sorted:
                comp = ic_comparison.get(factor, {})
                ic_sel = comp.get("ic_selection", 0)
                ic_val = comp.get("ic_validation", 0)
                ic_change = comp.get("ic_change", 0)
                unstable_rows += f"""
                <tr>
                    <td>{factor}</td>
                    <td>{_format_float(ic_sel, 4)}</td>
                    <td>{_format_float(ic_val, 4)}</td>
                    <td class="bad">{_format_float(ic_change, 4)}</td>
                </tr>"""

        stability_html = f"""
        <div class="card">
            <h3>🔍 Factor Stability Validation</h3>
            <div class="explanation">
                <h4>📊 验证说明</h4>
                <p>使用更长的历史数据验证因子选择的稳定性：</p>
                <ul>
                    <li><strong>因子选择期</strong>：{sel_period.get('start', 'N/A')} → {sel_period.get('end', 'N/A')}（用于选择因子）</li>
                    <li><strong>稳定性验证期</strong>：{val_period.get('start', 'N/A')} → {val_period.get('end', 'N/A')}（用于验证因子稳定性）</li>
                </ul>
                
                <h4>📈 稳定性统计</h4>
                <ul>
                    <li><strong>稳定因子</strong>：{len(stable_factors)} 个（{stability_rate:.1%}）- IC 符号一致且幅度相似</li>
                    <li><strong>不稳定因子</strong>：{len(unstable_factors)} 个（{1 - stability_rate:.1%}）- IC 变化较大</li>
                </ul>
                
                <h4>💡 解读</h4>
                <ul>
                    <li><strong>稳定因子</strong>：在不同时期表现一致，更可靠，建议优先使用</li>
                    <li><strong>不稳定因子</strong>：可能只在特定时期有效，需要谨慎使用或定期重新评估</li>
                    <li><strong>稳定性率 {stability_rate:.1%}</strong>：{"优秀" if stability_rate >= 0.7 else "良好" if stability_rate >= 0.5 else "需要改进"}</li>
                </ul>
            </div>
            
            {f'''
            <h4>✅ 稳定因子 Top 20（IC 在不同时期保持一致）</h4>
            <table class="metric-table">
                <tr>
                    <th>Factor</th>
                    <th>IC (Selection Period)</th>
                    <th>IC (Validation Period)</th>
                    <th>IC Change</th>
                </tr>
                {stable_rows}
            </table>
            ''' if stable_rows else ''}
            
            {f'''
            <h4>⚠️ 不稳定因子 Top 10（IC 变化较大）</h4>
            <table class="metric-table">
                <tr>
                    <th>Factor</th>
                    <th>IC (Selection Period)</th>
                    <th>IC (Validation Period)</th>
                    <th>IC Change</th>
                </tr>
                {unstable_rows}
            </table>
            ''' if unstable_rows else ''}
        </div>
        """

    # Build HTML content
    html = _build_html_report_content(
        date_range_str,
        runtime_str,
        d,
        stage1,
        stage2,
        stage3,
        stage4,
        has_4_stages,
        orig,
        comp,
        delta_r2,
        stage1_fin,
        stage2_fin,
        stage3_fin,
        stage4_fin,
        train_info,
        grid_rows,
        conclusion,
        stage2_vs_1,
        stage3_vs_2,
        stage4_vs_3,
        multi_horizon_results,
        task_type,
        selection_metric,
        label_threshold,
        artifacts,
        shap_importance=shap_importance_preview,
        feature_effective=feature_effective,
        feature_effective_display=feature_effective_display,
        recommended_horizon_row=recommended_horizon_row,
        recommended_stage_label=recommended_stage_label,
        insights_html=insights_html,
        classification_section=classification_section,
        confusion_html=confusion_html,
        stability_html=stability_html,
    )

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
    shap_importance: list | None = None,
    feature_effective: bool | None = None,
    feature_effective_display: str | None = None,
    recommended_horizon_row: str = "",
    recommended_stage_label: str | None = None,
    insights_html: str = "",
    classification_section: str = "",
    confusion_html: str = "",
    stability_html: str = "",
) -> str:
    """Build HTML content string for the report."""
    # Build conditional 4-stage comparison table
    stage_comparison_table = ""
    if has_4_stages:
        if task_type.startswith("classification"):
            stage_comparison_table = (
                f'<div class="card"><h3>Stage Comparison (Test Set)</h3><table class="metric-table">'
                f'<tr><th>Stage</th><th>Features</th><th>Directional Win Rate</th><th>Active Ratio</th></tr>'
                f'<tr><td>Stage 1: All Features</td><td>{d.get("stage1_all_features", "-")}</td>'
                f'<td>{_format_float(stage1_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage1_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'<tr><td>Stage 2: IC-Filtered</td><td>{d.get("stage2_ic_filtered", "-")}</td>'
                f'<td>{_format_float(stage2_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage2_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'<tr><td>Stage 3: Representatives</td><td>{d.get("stage3_representatives", "-")}</td>'
                f'<td>{_format_float(stage3_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage3_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'<tr><td>Stage 4: Compressed</td><td>{d.get("compressed_dimensions", "-")}</td>'
                f'<td>{_format_float(stage4_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage4_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'</table></div>')
        else:
            stage_comparison_table = (
                f'<div class="card"><h3>Stage Comparison (Test Set)</h3><table class="metric-table">'
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
                f'</table></div>')
    else:
        if task_type.startswith("classification"):
            stage_comparison_table = (
                f'<div class="card"><h3>Stage Comparison (Test Set)</h3><table class="metric-table">'
                f'<tr><th>Stage</th><th>Features</th><th>Directional Win Rate</th><th>Active Ratio</th></tr>'
                f'<tr><td>Stage 1: All Features</td><td>{d.get("stage1_all_features", "-")}</td>'
                f'<td>{_format_float(stage1_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage1_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'<tr><td>Stage 2: IC-Filtered</td><td>{d.get("stage2_ic_filtered", "-")}</td>'
                f'<td>{_format_float(stage2_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage2_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'<tr><td>Stage 3: Representatives</td><td>{d.get("stage3_representatives", "-")}</td>'
                f'<td>{_format_float(stage3_fin.get("win_rate",0)*100,2)}%</td><td>{_format_float(stage3_fin.get("active_ratio",0)*100,2)}%</td></tr>'
                f'</table></div>')
        else:
            stage_comparison_table = (
                f'<div class="card"><h3>Stage Comparison (Test Set)</h3><table class="metric-table">'
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
                f'</table></div>')

    top_factor_preview = artifacts.get(
        "top_factors_preview") if artifacts else []
    rep_factor_preview = artifacts.get(
        "representatives_preview") if artifacts else []
    shap_link = artifacts.get("shap") if artifacts else None

    artifact_lines: list[str] = []
    effective_badge_text = None
    if feature_effective is True:
        effective_badge_text = "✅ effective"
    elif feature_effective is False:
        effective_badge_text = "⚠️ not effective"
    if artifacts is not None:
        top_link = artifacts.get("top_factors")
        if top_link:
            line = (
                f'Top Factors: <a href="{top_link}">{os.path.basename(top_link)}</a>'
            )
            if effective_badge_text:
                line = f"{line} ({effective_badge_text})"
            artifact_lines.append(line)
        elif top_factor_preview:
            artifact_lines.append("Top Factors: (inline preview below)")
        else:
            artifact_lines.append("Top Factors: —")

        rep_link = artifacts.get("representatives")
        if rep_link:
            line = (
                f'Representatives: <a href="{rep_link}">{os.path.basename(rep_link)}</a>'
            )
            if effective_badge_text:
                line = f"{line} ({effective_badge_text})"
            artifact_lines.append(line)
        elif rep_factor_preview:
            artifact_lines.append("Representatives: (inline preview below)")
        else:
            artifact_lines.append("Representatives: —")

        if shap_link:
            artifact_lines.append(
                f'SHAP Visualisations: <a href="{shap_link}">Open directory</a>'
            )
        else:
            artifact_lines.append(
                "SHAP Visualisations: Not generated (run with --shap-analysis)"
            )

    artifacts_html = "<br/>".join(artifact_lines) if artifact_lines else "—"

    top_factor_html = ""
    if top_factor_preview:
        top_factor_html = (
            "<div class=\"card\">"
            "<h3>Top Factors (IC Ranking)</h3>"
            "<ul class=\"pill-list\">"
            f"{''.join(f'<li>{name}</li>' for name in top_factor_preview[:30])}"
            "</ul>"
            "</div>")

    rep_factor_html = ""
    if rep_factor_preview:
        rep_factor_html = (
            "<div class=\"card\">"
            "<h3>Representative Feature Set</h3>"
            "<ul class=\"pill-list\">"
            f"{''.join(f'<li>{name}</li>' for name in rep_factor_preview[:30])}"
            "</ul>"
            "</div>")

    factor_section = ""
    if top_factor_html or rep_factor_html:
        factor_section = f'<div class="grid-two">{top_factor_html}{rep_factor_html}</div>'

    shap_html = ""
    if shap_importance:
        shap_rows = "".join(
            f"<tr><td>{item.get('rank')}</td><td>{item.get('feature')}</td><td>{_format_float(item.get('mean_abs_shap'))}</td></tr>"
            for item in shap_importance)
        shap_html = ("<div class=\"card\">"
                     "<h3>SHAP Importance (Top Factors)</h3>"
                     "<table class=\"metric-table\">"
                     "<tr><th>#</th><th>Feature</th><th>Mean |SHAP|</th></tr>"
                     f"{shap_rows}"
                     "</table>"
                     "</div>")

    regression_section = ""
    if not task_type.startswith("classification") and stage1 and stage3:
        regression_section = _build_regression_metrics_table(
            stage1,
            stage3,
            "Stage 1: All Features",
            "Stage 3: Representatives",
        )

    grid_html = ""
    if grid_rows:
        grid_html = (
            "<div class=\"card\">"
            "<h3>Autoencoder Grid Search</h3>"
            "<table class=\"metric-table\">"
            "<tr><th>Encoding Dim</th><th>Stage 3 R²</th><th>Compressed R²</th>"
            "<th>ΔR²</th><th>Stage 3 RMSE</th><th>Compressed RMSE</th></tr>"
            f"{''.join(grid_rows)}"
            "</table>"
            "</div>")

    training_html = ""
    train_rows = []
    iteration_values = []
    diag_map = {
        "lightgbm_original_iterations": "Stage 1 · All Features",
        "lightgbm_stage1_iterations": "Stage 1 · All Features",
        "lightgbm_stage2_iterations": "Stage 2 · IC-Filtered",
        "lightgbm_stage3_iterations": "Stage 3 · Representatives",
        "lightgbm_compressed_iterations": "Stage 3 · Representatives",
        "lightgbm_stage4_iterations": "Stage 4 · Autoencoder",
    }
    for key, label in diag_map.items():
        if train_info.get(key) is not None:
            iter_val = train_info.get(key)
            train_rows.append(f"<tr><td>{label}</td><td>{iter_val}</td></tr>")
            iteration_values.append((label, iter_val))
    if train_rows:

        # Generate interpretation
        interpretation = ""
        if len(iteration_values) > 1:
            iterations = [val for _, val in iteration_values]
            min_iter = min(iterations)
            max_iter = max(iterations)
            avg_iter = sum(iterations) / len(iterations)

            interpretation = f"""
            <div class="explanation" style="margin-top: 20px;">
                <h4>📊 如何解读 Best Iteration</h4>
                <p><strong>Best Iteration</strong> 是 LightGBM 通过早停（Early Stopping）机制找到的最佳迭代次数。</p>
                
                <h4>为什么不同阶段的 Best Iteration 不同？</h4>
                <ul>
                    <li><strong>特征数量不同</strong>：不同阶段使用的特征数量不同（Stage 1: ~470, Stage 2: ~120, Stage 3: 60-100）</li>
                    <li><strong>特征质量不同</strong>：Stage 2/3 经过 IC 筛选和相关性去冗余，特征质量更高</li>
                    <li><strong>模型复杂度不同</strong>：特征越多，模型越复杂，可能需要更多迭代才能收敛</li>
                    <li><strong>过拟合风险不同</strong>：特征多时容易过拟合，早停会更早触发；特征少时模型更简单，可能需要更多迭代</li>
                </ul>
                
                <h4>📈 当前数据解读</h4>
                <ul>
                    <li><strong>迭代次数范围</strong>：{min_iter} - {max_iter} 次</li>
                    <li><strong>平均迭代次数</strong>：{avg_iter:.1f} 次</li>
                    <li><strong>差异</strong>：最大差异 {max_iter - min_iter} 次</li>
                </ul>
                
                <h4>💡 结论</h4>
                <p>
                    {"迭代次数差异较小（< 10），说明不同阶段的模型收敛速度相近，特征选择效果良好。" if (max_iter - min_iter) < 10 else 
                     "迭代次数差异较大，可能因为："}
                </p>
                <ul>
                    {"<li>特征数量差异导致模型复杂度不同</li>" if (max_iter - min_iter) >= 10 else ""}
                    {"<li>特征质量差异影响模型学习速度</li>" if (max_iter - min_iter) >= 10 else ""}
                    <li>这是正常现象，<strong>Best Iteration 本身不是性能指标</strong>，重要的是模型的最终性能（准确率、F1 等）</li>
                    <li>如果某个阶段的 Best Iteration 特别高（> 200），可能表示模型难以学习，需要检查特征质量</li>
                    <li>如果某个阶段的 Best Iteration 特别低（< 50），可能表示模型过早停止，可以尝试增加迭代次数上限</li>
                </ul>
            </div>
            """

        training_html = ("<div class=\"card\">"
                         "<h3>Training Diagnostics</h3>"
                         "<table class=\"metric-table\">"
                         "<tr><th>Model</th><th>Best Iteration</th></tr>"
                         f"{''.join(train_rows)}"
                         "</table>"
                         f"{interpretation}"
                         "</div>")

    multi_horizon_html = _build_multi_horizon_table(multi_horizon_results,
                                                    task_type)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/><title>Dimensionality Reduction Comparison</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#1f2d3d;background:#f5f7fb}}
h1,h2,h3{{color:#24344d}}
table{{border-collapse:collapse;margin-top:16px;width:100%;max-width:960px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 6px 14px rgba(27,39,53,0.08)}}
th,td{{border:1px solid #e6ecf5;padding:10px 14px;text-align:left;font-size:0.95rem}}
th{{background:#eef2f8;font-weight:600;color:#2b3f64}}
.bad{{color:#c53030;font-weight:600}}
.good{{color:#167a3d;font-weight:600}}
.warn{{color:#b36b00;font-weight:600}}
.badge{{display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;font-size:0.75rem;font-weight:600;margin-left:0.45rem;background:#e6edff;color:#2f4cdd}}
.badge.bad{{background:#fde8e8;color:#c53030}}
.badge.good{{background:#e6f4ea;color:#167a3d}}
.badge.warn{{background:#fff3cd;color:#b36b00}}
.card{{background:#fff;border-radius:10px;padding:18px 22px;box-shadow:0 10px 24px rgba(27,39,53,0.1);margin:20px 0}}
.card h3{{margin:0 0 10px 0;color:#1f2d3d}}
.card p{{margin:6px 0;color:#42516d}}
.grid-two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;margin:18px 0}}
.grid-two .card{{margin:0}}
.pill-list{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0 0;padding:0;list-style:none}}
.pill-list li{{background:#eef2f8;color:#2b3f64;border-radius:999px;padding:6px 16px;font-size:0.9rem}}
.metric-table{{margin-top:12px}}
.reason{{margin-top:8px;font-size:0.95rem;color:#384860}}
.reason strong{{color:#1f2d3d}}
</style>
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
{f'<tr><td>Stage 4: Compressed</td><td>{compressed_dims}</td><td>Compressed feature dimensions</td></tr>' if has_4_stages else ''}
<tr><th colspan="3">Summary</th></tr>
<tr><td>Final Compression Ratio</td><td colspan="2">{_format_float(d.get('compression_ratio'),2)}x ({d.get('original_features_count','-')} → {compressed_dims if has_4_stages else d.get('stage3_representatives','-')})</td></tr>
<tr><td>Samples (train/val/test)</td><td colspan="2">{d.get('training_samples','-')} / {d.get('validation_samples','-')} / {d.get('test_samples','-')}</td></tr>
</table>

<div class="card">
<h3>Run Configuration</h3>
<table class="metric-table">
<tr><th>Task Type</th><td>{task_type}</td></tr>
<tr><th>Selection Metric</th><td>{selection_metric or '-'}</td></tr>
{f'<tr><th>Label Threshold</th><td>{_format_float(label_threshold,6)}</td></tr>' if label_threshold is not None else ''}
<tr><th>Feature Effectiveness</th><td>{feature_effective_display or 'Unknown'}</td></tr>
{f'<tr><th>Recommended Stage</th><td>{recommended_stage_label}</td></tr>' if recommended_stage_label else ''}
{recommended_horizon_row}
<tr><th>Artifacts</th><td>{artifacts_html}</td></tr>
</table>
</div>

{insights_html}
{stability_html}

{factor_section}
{shap_html}

{stage_comparison_table}

{classification_section or regression_section}
{confusion_html}

{grid_html}
{training_html}
<div class="card">
<h3>Conclusion</h3>
<p>{conclusion}</p>
</div>

{multi_horizon_html}
</body></html>"""
    return html


def _build_multi_horizon_table(multi_horizon_results: Dict,
                               task_type: str) -> str:
    """Build multi-horizon comparison table."""
    if not multi_horizon_results:
        return ""

    if task_type.startswith("classification"):
        header = (
            "<tr><th>Horizon</th><th>Stage</th><th>Accuracy</th>"
            "<th>F1 (Macro)</th><th>ROC AUC</th><th>Directional Win Rate</th></tr>"
        )
    else:
        header = ("<tr><th>Horizon</th><th>Stage</th><th>R²</th><th>RMSE</th>"
                  "<th>MAE</th></tr>")

    rows = []
    horizon_keys = sorted(
        [k for k in multi_horizon_results.keys() if k.startswith("horizon_")],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)

    stage_map = [
        ("Stage 1: All Features", "stage1_all_features"),
        ("Stage 2: IC-Filtered", "stage2_ic_filtered"),
        ("Stage 3: Representatives", "stage3_representatives"),
        ("Stage 4: Compressed", "stage4_compressed"),
    ]

    for horizon_key in horizon_keys:
        horizon_num = horizon_key.split("_")[1]
        horizon_data = multi_horizon_results[horizon_key]
        for stage_label, stage_key in stage_map:
            stage_perf = horizon_data.get(stage_key)
            if not stage_perf:
                continue
            if task_type.startswith("classification"):
                cls_metrics = stage_perf.get("classification_metrics", {})
                financial = stage_perf.get("financial_metrics", {})
                rows.append(
                    "<tr>"
                    f"<td><strong>{horizon_num} bars</strong></td>"
                    f"<td>{stage_label}</td>"
                    f"<td>{_format_metric_for_display('accuracy', cls_metrics.get('accuracy'))}</td>"
                    f"<td>{_format_metric_for_display('f1_macro', cls_metrics.get('f1_macro'))}</td>"
                    f"<td>{_format_metric_for_display('roc_auc_macro', cls_metrics.get('roc_auc_macro'))}</td>"
                    f"<td>{_format_metric_for_display('win_rate', financial.get('win_rate'))}</td>"
                    "</tr>")
            else:
                rows.append("<tr>"
                            f"<td><strong>{horizon_num} bars</strong></td>"
                            f"<td>{stage_label}</td>"
                            f"<td>{_format_float(stage_perf.get('r2'))}</td>"
                            f"<td>{_format_float(stage_perf.get('rmse'))}</td>"
                            f"<td>{_format_float(stage_perf.get('mae'))}</td>"
                            "</tr>")

    if not rows:
        return ""

    return ("<div class=\"card\">"
            "<h3>📊 Multi-Horizon Comparison</h3>"
            "<table class=\"metric-table\">"
            f"{header}"
            f"{''.join(rows)}"
            "</table>"
            "</div>")


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


# Grid search report generation functions (moved from dimensionality_comparison.py)


def _generate_metric_3d_plot(enhanced_results: list,
                             time_windows: list,
                             factor_counts: list,
                             metric_name: str,
                             metric_label: str,
                             metric_getter,
                             color_thresholds: dict = None) -> str:
    """Generate 3D visualization for any metric across factor counts and time windows.
    
    Args:
        enhanced_results: List of result dictionaries
        time_windows: List of time window strings
        factor_counts: List of factor counts
        metric_name: Name of the metric (e.g., 'icir', 'sharpe', 'robustness')
        metric_label: Display label for the metric
        metric_getter: Function to extract metric value from result dict
        color_thresholds: Dict with 'good', 'warn', 'bad' thresholds
    """
    try:
        import plotly.graph_objects as go
        import numpy as np

        # Default color thresholds
        if color_thresholds is None:
            color_thresholds = {'good': 1.0, 'warn': 0.5, 'bad': 0.0}

        # Prepare data arrays
        x_data = []
        y_data = []
        z_data = []
        colors = []
        text_labels = []

        for i, tw in enumerate(time_windows):
            for fc in sorted(factor_counts,
                             key=lambda x:
                             (x == 'all', x
                              if isinstance(x, int) else 999999)):
                # Find result for this combination
                result = None
                for r in enhanced_results:
                    params = r.get('grid_search_params', {})
                    if params.get('time_window') == tw and params.get(
                            'factor_count') == fc:
                        result = r
                        break

                if result:
                    metric_val = metric_getter(result)
                    # Include 0 values as well (they are valid data points)
                    # Debug: print first extraction
                    if len(x_data) == 0 and metric_val is not None:
                        print(
                            f"[DEBUG 3D] First data point: tw={tw}, fc={fc}, metric_val={metric_val}"
                        )
                    if metric_val is not None:
                        # X: factor count (numeric)
                        if isinstance(fc, int):
                            x_val = fc
                        else:
                            max_fc = max([
                                x for x in factor_counts if isinstance(x, int)
                            ],
                                         default=120)
                            x_val = max_fc * 1.2

                        x_data.append(x_val)
                        y_data.append(i)  # Time window index
                        z_data.append(metric_val)

                        # Color based on metric value
                        if metric_val > color_thresholds['good']:
                            colors.append('#167a3d')  # Green
                        elif metric_val > color_thresholds['warn']:
                            colors.append('#ffc107')  # Yellow
                        else:
                            colors.append('#dc3545')  # Red

                        text_labels.append(
                            f"Time: {tw}<br>Factors: {fc}<br>{metric_label}: {metric_val:.3f}"
                        )

        if not x_data:
            return f"""
            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                <h4>📊 {metric_label} 3D 可视化</h4>
                <p>⚠️ 没有可用的数据来生成3D图形。</p>
            </div>
            """

        # Create 3D scatter plot
        fig = go.Figure(
            data=go.Scatter3d(x=x_data,
                              y=y_data,
                              z=z_data,
                              mode='markers',
                              marker=dict(size=10,
                                          color=colors,
                                          opacity=0.8,
                                          line=dict(width=1, color='black')),
                              text=text_labels,
                              hovertemplate='%{text}<extra></extra>',
                              name=f'{metric_label} Points'))

        # Add surface plot to show trend
        if len(x_data) > 0 and len(set(x_data)) > 1 and len(set(y_data)) > 1:
            # Create grid for surface
            x_unique = sorted(set(x_data))
            y_unique = sorted(set(y_data))

            # Create meshgrid
            X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
            Z_grid = np.full_like(X_grid, np.nan, dtype=float)

            # Fill Z_grid with metric values
            for i, tw_idx in enumerate(y_unique):
                for j, fc_val in enumerate(x_unique):
                    # Find metric for this combination
                    for r in enhanced_results:
                        params = r.get('grid_search_params', {})
                        tw = time_windows[tw_idx]
                        # Find matching factor count
                        fc_match = None
                        for fc in factor_counts:
                            if isinstance(fc, int) and fc == fc_val:
                                fc_match = fc
                                break
                            elif fc == 'all' and abs(
                                    fc_val - max([
                                        x for x in factor_counts
                                        if isinstance(x, int)
                                    ],
                                                 default=120) * 1.2) < 1:
                                fc_match = fc
                                break

                        if params.get('time_window') == tw and params.get(
                                'factor_count') == fc_match:
                            metric_val = metric_getter(r)
                            if metric_val is not None:
                                Z_grid[i, j] = metric_val
                            break

            # Add surface plot
            fig.add_trace(
                go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale='RdYlGn',
                    showscale=True,
                    opacity=0.6,
                    name=f'{metric_label} Surface',
                    hovertemplate=
                    f'Factor Count: %{{x:.0f}}<br>Time Window: %{{y}}<br>{metric_label}: %{{z:.3f}}<extra></extra>'
                ))

        # Get factor count labels
        fc_labels = []
        for fc in sorted(factor_counts,
                         key=lambda x: (x == 'all', x
                                        if isinstance(x, int) else 999999)):
            if isinstance(fc, int):
                fc_labels.append(str(fc))
            else:
                fc_labels.append('all')

        # Update layout
        fig.update_layout(
            title=
            f'{metric_label} 3D 可视化 - Plateau Point 分析 ({metric_label} 3D Visualization - Plateau Point Analysis)',
            scene=dict(xaxis_title='因子数量 (Factor Count)',
                       yaxis_title='时间窗口索引 (Time Window Index)',
                       zaxis_title=f'{metric_label} 值 ({metric_label} Value)',
                       xaxis=dict(
                           tickmode='array',
                           tickvals=x_unique if 'x_unique' in locals() else [],
                           ticktext=fc_labels[:len(x_unique)]
                           if 'x_unique' in locals() else [],
                       ),
                       yaxis=dict(
                           tickmode='array',
                           tickvals=list(range(len(time_windows))),
                           ticktext=[
                               tw.split(' → ')[0] if ' → ' in tw else tw[:15]
                               for tw in time_windows
                           ],
                       ),
                       zaxis=dict(title=metric_label),
                       camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
            width=900,
            height=700,
            font=dict(size=12),
        )

        # Convert to HTML - use full HTML to ensure Plotly.js is included
        plot_html = fig.to_html(include_plotlyjs='cdn',
                                div_id=f'{metric_name}-3d-plot',
                                full_html=False)

        # Extract script and div more robustly
        import re
        # Match all script tags (may be multiple)
        script_matches = re.findall(r'<script[^>]*>.*?</script>', plot_html,
                                    re.DOTALL)
        script_content = '\n'.join(script_matches) if script_matches else ""

        # Match div with the specific ID
        div_pattern = rf'<div[^>]*id="{metric_name}-3d-plot"[^>]*>.*?</div>'
        div_match = re.search(div_pattern, plot_html, re.DOTALL)
        div_content = div_match.group(
            0
        ) if div_match else f'<div id="{metric_name}-3d-plot" class="plotly-graph-div" style="height:700px; width:900px;"></div>'

        # Debug: print if no data
        if not x_data:
            print(
                f"[DEBUG 3D] No data for {metric_label}: x_data length = {len(x_data)}"
            )
            print(
                f"[DEBUG 3D] enhanced_results count: {len(enhanced_results)}")
            if enhanced_results:
                print(
                    f"[DEBUG 3D] First result keys: {list(enhanced_results[0].keys())}"
                )
                print(
                    f"[DEBUG 3D] First result grid_search_params: {enhanced_results[0].get('grid_search_params', {})}"
                )

        return f"""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #17a2b8;">
            <h4>📊 {metric_label} 3D 可视化说明 (3D Visualization Guide):</h4>
            <ul>
                <li><strong>X轴（因子数量）：</strong>显示不同的因子数量。数值越大，使用的因子越多。</li>
                <li><strong>Y轴（时间窗口）：</strong>显示不同的时间窗口索引。每个索引对应一个时间窗口。</li>
                <li><strong>Z轴（{metric_label}值）：</strong>显示{metric_label}值，越高表示表现越好。</li>
                <li><strong>颜色含义：</strong>
                    <ul>
                        <li><span style="color: #167a3d; font-weight: 600;">绿色点</span>：{metric_label} > {color_thresholds['good']}，表现优秀</li>
                        <li><span style="color: #ffc107; font-weight: 600;">黄色点</span>：{color_thresholds['warn']} < {metric_label} ≤ {color_thresholds['good']}，表现一般</li>
                        <li><span style="color: #dc3545; font-weight: 600;">红色点</span>：{metric_label} ≤ {color_thresholds['warn']}，表现较差</li>
                    </ul>
                </li>
                <li><strong>如何识别Plateau Point：</strong>
                    <ul>
                        <li>观察3D表面图，寻找{metric_label}值不再显著上升的"平台"区域</li>
                        <li>Plateau Point通常出现在：{metric_label}值达到较高水平后，即使增加因子数量，{metric_label}也不再明显提升的位置</li>
                        <li>理想情况下，Plateau Point应该在不同时间窗口（Y轴）上都保持相对稳定的高度（Z轴）</li>
                        <li>可以通过旋转3D图形（点击并拖动）从不同角度观察，更容易识别平台区域</li>
                    </ul>
                </li>
                <li><strong>分析建议：</strong>
                    <ul>
                        <li>寻找Z轴（{metric_label}）值高且在不同Y轴（时间窗口）位置都保持稳定的X轴（因子数量）位置</li>
                        <li>如果表面图在某个因子数量后变得平坦，该位置就是Plateau Point</li>
                        <li>选择Plateau Point对应的因子数量，可以在保持高{metric_label}的同时，避免使用过多因子</li>
                    </ul>
                </li>
            </ul>
        </div>
        <div class="heatmap-container">
            {div_content}
        </div>
        {script_content}
        """
    except ImportError:
        return f"""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
            <h4>📊 {metric_label} 3D 可视化</h4>
            <p>⚠️ Plotly 未安装，无法生成3D图形。请安装: pip install plotly</p>
        </div>
        """
    except Exception as e:
        return f"""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
            <h4>📊 {metric_label} 3D 可视化</h4>
            <p>⚠️ 生成3D图形时出错: {str(e)}</p>
        </div>
        """


def _generate_icir_3d_plot(enhanced_results: list, time_windows: list,
                           factor_counts: list) -> str:
    """Generate 3D visualization of ICIR distribution across factor counts and time windows."""

    def get_icir(result):
        return result.get('enhanced_metrics', {}).get('icir')

    return _generate_metric_3d_plot(enhanced_results,
                                    time_windows,
                                    factor_counts,
                                    'icir',
                                    'ICIR',
                                    get_icir,
                                    color_thresholds={
                                        'good': 1.0,
                                        'warn': 0.5,
                                        'bad': 0.0
                                    })


def _generate_icir_heatmap(enhanced_results: list, time_windows: list,
                           factor_counts: list) -> str:
    """Generate ICIR heatmap visualization using Plotly."""
    try:
        import plotly.graph_objects as go

        # Prepare data matrix for heatmap
        heatmap_data = []
        factor_count_labels = []

        # Build data matrix: rows = factor counts, columns = time windows
        for fc in sorted(factor_counts,
                         key=lambda x: (x == 'all', x
                                        if isinstance(x, int) else 999999)):
            row_data = []
            factor_count_labels.append(str(fc))

            for tw in time_windows:
                # Find result for this combination
                result = None
                for r in enhanced_results:
                    params = r.get('grid_search_params', {})
                    if params.get('time_window') == tw and params.get(
                            'factor_count') == fc:
                        result = r
                        break

                if result:
                    icir = result.get('enhanced_metrics', {}).get('icir')
                    # Use ICIR value if available, otherwise try to get from ic_statistics
                    if icir is None:
                        ic_stats = result.get('ic_statistics', {})
                        ic_mean = ic_stats.get('ic_mean')
                        ic_std = ic_stats.get('ic_std')
                        if ic_mean is not None and ic_std is not None and ic_std > 0:
                            icir = abs(ic_mean) / ic_std
                    row_data.append(icir if icir is not None else 0)
                else:
                    row_data.append(0)

            heatmap_data.append(row_data)

        # Set time window labels (shortened for display)
        time_window_labels = [
            tw.split(' → ')[0] if ' → ' in tw else tw[:10]
            for tw in time_windows
        ]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=time_window_labels,
            y=factor_count_labels,
            colorscale='RdYlGn',  # Red-Yellow-Green scale
            colorbar=dict(title="ICIR"),
            text=[[f"{val:.3f}" if val else "-" for val in row]
                  for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate=
            'Time Window: %{x}<br>Factor Count: %{y}<br>ICIR: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title='ICIR 热力图 (ICIR Heatmap)',
            xaxis_title='时间窗口 (Time Window)',
            yaxis_title='因子数量 (Factor Count)',
            width=800,
            height=500,
            font=dict(size=12),
        )

        # Convert to HTML - use full HTML with CDN for plotly.js
        heatmap_html_full = fig.to_html(include_plotlyjs='cdn',
                                        div_id='icir-heatmap',
                                        full_html=False)

        # Extract script and div from the HTML more robustly
        import re
        # Match all script tags (may be multiple)
        script_matches = re.findall(r'<script[^>]*>.*?</script>',
                                    heatmap_html_full, re.DOTALL)
        script_content = '\n'.join(script_matches) if script_matches else ""

        # Match div with the specific ID
        div_pattern = r'<div[^>]*id="icir-heatmap"[^>]*>.*?</div>'
        div_match = re.search(div_pattern, heatmap_html_full, re.DOTALL)
        div_content = div_match.group(
            0
        ) if div_match else '<div id="icir-heatmap" class="plotly-graph-div" style="height:500px; width:800px;"></div>'

        # Debug: print if no data
        if not heatmap_data or all(
                all(val == 0 for val in row) for row in heatmap_data):
            print(
                f"[DEBUG Heatmap] No data or all zeros: heatmap_data = {heatmap_data}"
            )
            print(
                f"[DEBUG Heatmap] enhanced_results count: {len(enhanced_results)}"
            )
            if enhanced_results:
                print(
                    f"[DEBUG Heatmap] First result enhanced_metrics: {enhanced_results[0].get('enhanced_metrics', {})}"
                )

        return f"""
        <div class="card">
            <h3>🔥 ICIR 热力图 (ICIR Heatmap)</h3>
            <p>可视化不同因子数量和时间窗口的 ICIR 分布。颜色越绿表示 ICIR 越高（预测稳定性越好）。</p>
            <p>Visualization of ICIR distribution across different factor counts and time windows. Greener colors indicate higher ICIR (better predictive stability).</p>
            <div style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #fd7e14;">
                <h4>📖 如何阅读热力图 (How to Read the Heatmap):</h4>
                <ul>
                    <li><strong>颜色含义：</strong>
                        <ul>
                            <li><span style="color: #167a3d; font-weight: 600;">深绿色</span>：ICIR很高（> 1.5），表示因子预测能力非常稳定</li>
                            <li><span style="color: #28a745; font-weight: 600;">浅绿色</span>：ICIR较高（1.0 - 1.5），表示因子预测能力稳定</li>
                            <li><span style="color: #ffc107; font-weight: 600;">黄色</span>：ICIR中等（0.5 - 1.0），表示因子预测能力一般</li>
                            <li><span style="color: #dc3545; font-weight: 600;">红色</span>：ICIR较低（< 0.5），表示因子预测能力不稳定</li>
                        </ul>
                    </li>
                    <li><strong>分析要点：</strong>
                        <ul>
                            <li>观察颜色分布模式，找出ICIR高的区域（绿色区域）</li>
                            <li>比较不同因子数量的ICIR分布，识别最优因子数量范围</li>
                            <li>观察不同时间窗口的ICIR一致性，评估因子的时间稳定性</li>
                            <li>寻找颜色均匀的区域，表示该因子数量在不同时间窗口都表现稳定</li>
                        </ul>
                    </li>
                    <li><strong>结论：</strong>热力图提供了ICIR分布的直观可视化。理想的组合应该是在多个时间窗口都显示绿色或浅绿色，且颜色分布相对均匀，这表示该因子数量在不同市场环境下都能保持稳定的预测能力。</li>
                </ul>
            </div>
            <div class="heatmap-container">
                {div_content}
            </div>
            {script_content}
        </div>
        """
    except ImportError:
        # If plotly is not available, return a message
        return """
        <div class="card">
            <h3>🔥 ICIR 热力图 (ICIR Heatmap)</h3>
            <p>⚠️ Plotly 未安装，无法生成热力图。请安装: pip install plotly</p>
        </div>
        """
    except Exception as e:
        return f"""
        <div class="card">
            <h3>🔥 ICIR 热力图 (ICIR Heatmap)</h3>
            <p>⚠️ 生成热力图时出错: {str(e)}</p>
        </div>
        """


def _build_analysis_conclusions(enhanced_results: list, time_windows: list,
                                factor_counts: list,
                                is_classification: bool) -> str:
    """Build textual analysis conclusions for grid search results."""
    if not enhanced_results:
        return "<div class=\"card\"><h3>📊 Analysis Conclusions</h3><p>No results available for analysis.</p></div>"

    # Collect metrics for analysis
    results_by_fc = {}
    for result in enhanced_results:
        params = result.get('grid_search_params', {})
        fc = params.get('factor_count')
        if fc not in results_by_fc:
            results_by_fc[fc] = []
        results_by_fc[fc].append(result)

    # Find best factor count by robustness score
    best_fc = None
    best_robustness = -1
    for fc, results in results_by_fc.items():
        robustness_values = []
        for r in results:
            metrics = r.get('enhanced_metrics', {})
            icir = metrics.get('icir', 0) or 0
            sharpe = metrics.get('sharpe', 0) or 0
            max_dd = abs(metrics.get('max_drawdown', 0)) or 0.01
            robustness = (icir * sharpe) / (
                1 + max_dd) if icir > 0 and sharpe > 0 else 0
            robustness_values.append(robustness)
        avg_robustness = sum(robustness_values) / len(
            robustness_values) if robustness_values else 0
        if avg_robustness > best_robustness:
            best_robustness = avg_robustness
            best_fc = fc

    # Analyze ICIR stability across time windows
    icir_stability = {}
    for fc in factor_counts:
        icir_values = []
        for result in enhanced_results:
            params = result.get('grid_search_params', {})
            if params.get('factor_count') == fc:
                icir = result.get('enhanced_metrics', {}).get('icir')
                if icir is not None:
                    icir_values.append(icir)
        if icir_values:
            mean_icir = sum(icir_values) / len(icir_values)
            std_icir = (sum((x - mean_icir)**2
                            for x in icir_values) / len(icir_values))**0.5
            icir_stability[fc] = {'mean': mean_icir, 'std': std_icir}

    # Find most stable factor count (lowest std with high mean)
    most_stable_fc = None
    best_stability_score = -1
    for fc, stats in icir_stability.items():
        if stats['mean'] > 0.5:  # Only consider factor counts with decent ICIR
            stability_score = stats['mean'] / (
                1 + stats['std'])  # Higher mean, lower std is better
            if stability_score > best_stability_score:
                best_stability_score = stability_score
                most_stable_fc = fc

    # Build conclusions HTML
    conclusions_html = "<div class=\"card\"><h3>📊 分析结论 (Analysis Conclusions)</h3>"

    # Optimal factor count
    conclusions_html += "<h4>🎯 最优因子数量 (Optimal Factor Count)</h4>"
    if best_fc is not None:
        conclusions_html += f"<p>基于稳健性得分（Robustness Score）分析，<strong>{best_fc}个因子</strong>是最优选择。</p>"
        conclusions_html += f"<p>Based on Robustness Score analysis, <strong>{best_fc} factors</strong> is the optimal choice.</p>"
        if best_robustness > 0.5:
            conclusions_html += f"<p>该因子数量的平均稳健性得分为 <strong>{best_robustness:.3f}</strong>，表现优秀（> 0.5）。</p>"
        else:
            conclusions_html += f"<p>该因子数量的平均稳健性得分为 <strong>{best_robustness:.3f}</strong>，表现一般（≤ 0.5）。</p>"
    else:
        conclusions_html += "<p>无法确定最优因子数量，请检查数据质量。</p>"

    # Factor stability across time windows
    conclusions_html += "<h4>📈 因子在不同周期的有效性 (Factor Effectiveness Across Time Windows)</h4>"
    if icir_stability:
        conclusions_html += "<ul>"
        for fc in sorted(factor_counts,
                         key=lambda x: (x == 'all', x
                                        if isinstance(x, int) else 999999)):
            if fc in icir_stability:
                stats = icir_stability[fc]
                conclusions_html += f"<li><strong>{fc}个因子：</strong>"
                conclusions_html += f"平均ICIR = {stats['mean']:.3f}，标准差 = {stats['std']:.3f}。"
                if stats['mean'] > 1.0 and stats['std'] < 0.3:
                    conclusions_html += "✅ 表现优秀且稳定（高ICIR，低波动）。"
                elif stats['mean'] > 0.5:
                    conclusions_html += "⚠️ 表现一般，稳定性有待提升。"
                else:
                    conclusions_html += "❌ 表现较差，不推荐使用。"
                conclusions_html += "</li>"
        conclusions_html += "</ul>"

    if most_stable_fc is not None and most_stable_fc != best_fc:
        conclusions_html += f"<p><strong>💡 稳定性建议：</strong>如果优先考虑因子在不同时间窗口的稳定性，建议选择 <strong>{most_stable_fc}个因子</strong>（ICIR均值高且标准差低）。</p>"

    # Multi-period effectiveness
    conclusions_html += "<h4>🔄 多周期有效性分析 (Multi-Period Effectiveness Analysis)</h4>"
    if len(time_windows) > 1:
        conclusions_html += f"<p>本次测试覆盖了 <strong>{len(time_windows)}</strong> 个不同的时间窗口：</p>"
        conclusions_html += "<ul>"
        for tw in time_windows:
            conclusions_html += f"<li>{tw}</li>"
        conclusions_html += "</ul>"
        conclusions_html += "<p><strong>关键发现：</strong></p>"
        conclusions_html += "<ul>"
        conclusions_html += "<li>如果某个因子数量在所有时间窗口都表现良好（绿色单元格），说明该因子数量具有强的时间稳定性。</li>"
        conclusions_html += "<li>如果某个因子数量只在部分时间窗口表现良好，说明该因子数量可能对特定市场环境敏感。</li>"
        conclusions_html += "<li>建议优先选择在所有或大部分时间窗口都表现稳定的因子数量。</li>"
        conclusions_html += "</ul>"
    else:
        conclusions_html += "<p>本次测试仅使用单一时间窗口，无法评估多周期有效性。建议增加更多时间窗口进行测试。</p>"

    # Final recommendations
    conclusions_html += "<h4>✅ 最终建议 (Final Recommendations)</h4>"
    conclusions_html += "<ol>"
    if best_fc is not None:
        conclusions_html += f"<li><strong>推荐因子数量：{best_fc}个</strong> - 基于稳健性得分分析，这是综合表现最优的选择。</li>"
    if most_stable_fc is not None and most_stable_fc != best_fc:
        conclusions_html += f"<li><strong>备选因子数量：{most_stable_fc}个</strong> - 如果更关注时间稳定性，可以考虑此选项。</li>"
    conclusions_html += "<li><strong>验证建议：</strong>在实际使用前，建议在最新的数据上验证所选因子数量的表现。</li>"
    conclusions_html += "<li><strong>持续监控：</strong>定期重新评估因子有效性，因为市场环境会发生变化。</li>"
    conclusions_html += "</ol>"

    conclusions_html += "</div>"
    return conclusions_html


def generate_grid_search_html_report(report_data: Dict,
                                     html_path: str) -> None:
    """Generate HTML report for grid search results with enhanced metrics and visualizations."""
    import os
    import json

    time_windows = report_data['time_windows']
    factor_counts = report_data['factor_counts']
    grid_search_results = report_data['grid_search_results']
    task_type = report_data['task_type']
    is_classification = task_type.startswith('classification')

    # Calculate ICIR and robustness metrics for each result
    enhanced_results = []
    for result in grid_search_results:
        perf = result.get('performance', {}).get('stage3_representatives', {})
        # financial_metrics is stored inside perf_reps, not as a separate field
        financial = perf.get('financial_metrics', {}) if isinstance(
            perf, dict) else {}
        # Also check the separate financial field as fallback
        if not financial:
            financial = result.get('performance',
                                   {}).get('stage3_representatives_financial',
                                           {})

        # Extract metrics
        if is_classification:
            # win_rate is stored in financial_metrics, not in performance directly
            win_rate = financial.get('win_rate', 0) if financial else 0
            # Also check performance for win_rate as fallback
            if win_rate == 0:
                win_rate = perf.get('win_rate', 0)
            # Also check classification_metrics for accuracy as fallback for win_rate
            if win_rate == 0:
                classification_metrics = perf.get('classification_metrics',
                                                  {}) if isinstance(
                                                      perf, dict) else {}
                if classification_metrics:
                    # Use accuracy as a proxy for win_rate if available
                    accuracy = classification_metrics.get('accuracy', 0)
                    if accuracy > 0:
                        win_rate = accuracy
            sharpe = financial.get('sharpe_ratio', 0) if financial else 0
            max_dd = financial.get('max_drawdown', 0) if financial else 0
        else:
            r2 = perf.get('r2', 0)
            sharpe = financial.get('sharpe_ratio', 0) if financial else 0
            max_dd = financial.get('max_drawdown', 0) if financial else 0

        # Calculate ICIR if IC data is available
        ic_stats = result.get('ic_statistics', {})
        ic_mean = ic_stats.get('ic_mean', None)
        ic_std = ic_stats.get('ic_std', None)
        icir = ic_stats.get('icir', None)
        if icir is None and ic_mean is not None and ic_std is not None and ic_std > 0:
            icir = abs(ic_mean) / ic_std

        enhanced_results.append({
            **result, 'enhanced_metrics': {
                'icir': icir,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
            }
        })

    # Build multiple comparison matrices
    # Matrix 1: Primary metric (Win Rate or R²)
    matrix_html = "<div class=\"card\"><h3>📊 Grid Search Comparison Matrix - Primary Metric</h3>"
    matrix_html += "<p>Comparison of different factor counts and time windows</p>"

    # Determine primary metric
    if is_classification:
        primary_metric = 'win_rate'
        metric_display = 'Directional Win Rate'
    else:
        primary_metric = 'r2'
        metric_display = 'R²'

    # Build table header
    matrix_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
    matrix_html += "<tr><th>Time Window</th>"
    for fc in factor_counts:
        matrix_html += f"<th>Factors: {fc}</th>"
    matrix_html += "</tr>"

    # Build table rows
    for tw in time_windows:
        matrix_html += f"<tr><td><strong>{tw}</strong></td>"
        for fc in factor_counts:
            # Find result for this combination
            result = None
            for r in enhanced_results:
                params = r.get('grid_search_params', {})
                if params.get('time_window') == tw and params.get(
                        'factor_count') == fc:
                    result = r
                    break

            if result:
                perf = result.get('performance',
                                  {}).get('stage3_representatives', {})
                # financial_metrics is stored inside perf_reps
                financial = perf.get('financial_metrics', {}) if isinstance(
                    perf, dict) else {}
                if not financial:
                    financial = result.get('performance', {}).get(
                        'stage3_representatives_financial', {})
                if is_classification:
                    # win_rate is stored in financial_metrics
                    metric_val = financial.get('win_rate',
                                               0) if financial else 0
                    # Fallback to performance if not in financial
                    if metric_val == 0:
                        metric_val = perf.get('win_rate', 0)
                    # Also check classification_metrics for accuracy as fallback
                    if metric_val == 0:
                        classification_metrics = perf.get(
                            'classification_metrics', {}) if isinstance(
                                perf, dict) else {}
                        if classification_metrics:
                            accuracy = classification_metrics.get(
                                'accuracy', 0)
                            if accuracy > 0:
                                metric_val = accuracy
                    cell_content = f"{_format_float(metric_val * 100, 2)}%"
                else:
                    metric_val = perf.get('r2', 0)
                    cell_content = _format_float(metric_val, 4)

                # Add color coding
                color_class = "good" if metric_val > 0.5 else "warn" if metric_val > 0 else "bad"
                matrix_html += f"<td class=\"{color_class}\">{cell_content}</td>"
            else:
                matrix_html += "<td>-</td>"
        matrix_html += "</tr>"

    matrix_html += "</table>"
    matrix_html += """
    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff;">
        <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
        <ul>
            <li><strong>表格结构：</strong>行表示不同的时间窗口，列表示不同的因子数量。每个单元格显示该组合下的{metric_display}值。</li>
            <li><strong>颜色编码：</strong>
                <ul>
                    <li><span style="color: #167a3d; font-weight: 600;">绿色</span>：表现优秀（{metric_display} > 0.5）</li>
                    <li><span style="color: #b36b00; font-weight: 600;">橙色</span>：表现一般（{metric_display} > 0）</li>
                    <li><span style="color: #c53030; font-weight: 600;">红色</span>：表现较差（{metric_display} ≤ 0）</li>
                </ul>
            </li>
            <li><strong>分析要点：</strong>
                <ul>
                    <li>比较同一时间窗口下不同因子数量的表现，找出最优因子数量</li>
                    <li>比较同一因子数量下不同时间窗口的表现，评估因子在不同时期的稳定性</li>
                    <li>关注绿色单元格，这些是表现最好的组合</li>
                </ul>
            </li>
            <li><strong>结论：</strong>此表格帮助识别在特定时间窗口下，使用多少因子能获得最佳{metric_display}。通常，因子数量不是越多越好，需要找到性能与复杂度的平衡点。</li>
        </ul>
    </div>
    </div>""".format(metric_display=metric_display)

    # Matrix 2: ICIR (if available)
    icir_matrix_html = ""
    if any(
            r.get('enhanced_metrics', {}).get('icir') is not None
            for r in enhanced_results):
        icir_matrix_html = "<div class=\"card\"><h3>📈 ICIR (Information Coefficient Information Ratio) Matrix</h3>"
        icir_matrix_html += "<p>ICIR = |Mean IC| / Std(IC) - Higher is better (indicates stable predictive power)</p>"
        icir_matrix_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
        icir_matrix_html += "<tr><th>Time Window</th>"
        for fc in factor_counts:
            icir_matrix_html += f"<th>Factors: {fc}</th>"
        icir_matrix_html += "</tr>"

        for tw in time_windows:
            icir_matrix_html += f"<tr><td><strong>{tw}</strong></td>"
            for fc in factor_counts:
                result = None
                for r in enhanced_results:
                    params = r.get('grid_search_params', {})
                    if params.get('time_window') == tw and params.get(
                            'factor_count') == fc:
                        result = r
                        break

                if result:
                    icir = result.get('enhanced_metrics', {}).get('icir')
                    if icir is not None:
                        cell_content = _format_float(icir, 3)
                        color_class = "good" if icir > 1.0 else "warn" if icir > 0.5 else "bad"
                        icir_matrix_html += f"<td class=\"{color_class}\">{cell_content}</td>"
                    else:
                        icir_matrix_html += "<td>-</td>"
                else:
                    icir_matrix_html += "<td>-</td>"
            icir_matrix_html += "</tr>"

        icir_matrix_html += "</table>"
        icir_matrix_html += """
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #28a745;">
            <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
            <ul>
                <li><strong>ICIR定义：</strong>ICIR = |平均IC| / IC标准差，衡量因子的预测稳定性和有效性。ICIR越高，表示因子在不同时期的表现越稳定。</li>
                <li><strong>表格结构：</strong>行表示时间窗口，列表示因子数量。每个单元格显示该组合的ICIR值。</li>
                <li><strong>颜色编码：</strong>
                    <ul>
                        <li><span style="color: #167a3d; font-weight: 600;">绿色</span>：ICIR > 1.0，表示因子具有稳定的预测能力</li>
                        <li><span style="color: #b36b00; font-weight: 600;">橙色</span>：0.5 < ICIR ≤ 1.0，表示因子预测能力一般</li>
                        <li><span style="color: #c53030; font-weight: 600;">红色</span>：ICIR ≤ 0.5，表示因子预测能力不稳定</li>
                    </ul>
                </li>
                <li><strong>分析要点：</strong>
                    <ul>
                        <li>ICIR > 1.0 是理想状态，表示因子的平均预测能力超过其波动性</li>
                        <li>比较不同因子数量的ICIR，找出在保持高ICIR的前提下，因子数量最少的组合</li>
                        <li>观察同一因子数量在不同时间窗口的ICIR，评估因子的时间稳定性</li>
                    </ul>
                </li>
                <li><strong>结论：</strong>此表格是选择因子的关键指标。高ICIR意味着因子不仅在历史数据上有效，而且在不同市场环境下都能保持稳定的预测能力。优先选择ICIR > 1.0且在不同时间窗口都表现稳定的因子组合。</li>
            </ul>
        </div>
        </div>"""

    # Matrix 3: Sharpe Ratio
    sharpe_matrix_html = "<div class=\"card\"><h3>💰 Sharpe Ratio Matrix</h3>"
    sharpe_matrix_html += "<p>Risk-adjusted return metric - Higher is better</p>"
    sharpe_matrix_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
    sharpe_matrix_html += "<tr><th>Time Window</th>"
    for fc in factor_counts:
        sharpe_matrix_html += f"<th>Factors: {fc}</th>"
    sharpe_matrix_html += "</tr>"

    for tw in time_windows:
        sharpe_matrix_html += f"<tr><td><strong>{tw}</strong></td>"
        for fc in factor_counts:
            result = None
            for r in enhanced_results:
                params = r.get('grid_search_params', {})
                if params.get('time_window') == tw and params.get(
                        'factor_count') == fc:
                    result = r
                    break

            if result:
                sharpe = result.get('enhanced_metrics', {}).get('sharpe', 0)
                cell_content = _format_float(sharpe, 3)
                color_class = "good" if sharpe > 1.0 else "warn" if sharpe > 0 else "bad"
                sharpe_matrix_html += f"<td class=\"{color_class}\">{cell_content}</td>"
            else:
                sharpe_matrix_html += "<td>-</td>"
        sharpe_matrix_html += "</tr>"

    sharpe_matrix_html += "</table>"
    sharpe_matrix_html += """
    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #ffc107;">
        <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
        <ul>
            <li><strong>Sharpe Ratio定义：</strong>夏普比率 = (策略收益率 - 无风险收益率) / 收益率标准差，衡量风险调整后的收益表现。Sharpe Ratio越高，表示在承担相同风险的情况下，获得的超额收益越多。</li>
            <li><strong>表格结构：</strong>行表示时间窗口，列表示因子数量。每个单元格显示该组合的Sharpe Ratio值。</li>
            <li><strong>颜色编码：</strong>
                <ul>
                    <li><span style="color: #167a3d; font-weight: 600;">绿色</span>：Sharpe > 1.0，表示策略表现优秀</li>
                    <li><span style="color: #b36b00; font-weight: 600;">橙色</span>：0 < Sharpe ≤ 1.0，表示策略表现一般</li>
                    <li><span style="color: #c53030; font-weight: 600;">红色</span>：Sharpe ≤ 0，表示策略表现不佳</li>
                </ul>
            </li>
            <li><strong>分析要点：</strong>
                <ul>
                    <li>Sharpe Ratio > 1.0 通常被认为是可接受的策略表现</li>
                    <li>Sharpe Ratio > 2.0 表示策略表现优秀</li>
                    <li>比较不同因子数量和时间窗口的Sharpe Ratio，找出风险调整后收益最高的组合</li>
                    <li>注意：此指标需要真实的回测数据，如果数据不可用，可能显示为0</li>
                </ul>
            </li>
            <li><strong>结论：</strong>此表格帮助评估策略的实际交易表现。高Sharpe Ratio意味着策略不仅能产生收益，而且风险控制得当。结合ICIR和Sharpe Ratio，可以全面评估因子的有效性和策略的实用性。</li>
        </ul>
    </div>
    </div>"""

    # Matrix 4: Robustness Score (ICIR-weighted composite)
    robustness_matrix_html = "<div class=\"card\"><h3>🛡️ Robustness Score Matrix</h3>"
    robustness_matrix_html += "<p>Composite score: ICIR × Sharpe / (1 + |Max Drawdown|) - Higher is better</p>"
    robustness_matrix_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
    robustness_matrix_html += "<tr><th>Time Window</th>"
    for fc in factor_counts:
        robustness_matrix_html += f"<th>Factors: {fc}</th>"
    robustness_matrix_html += "</tr>"

    # Calculate robustness scores
    robustness_scores = {}
    for tw in time_windows:
        robustness_scores[tw] = {}
        for fc in factor_counts:
            result = None
            for r in enhanced_results:
                params = r.get('grid_search_params', {})
                if params.get('time_window') == tw and params.get(
                        'factor_count') == fc:
                    result = r
                    break

            if result:
                metrics = result.get('enhanced_metrics', {})
                icir = metrics.get('icir', 0) or 0
                sharpe = metrics.get('sharpe', 0) or 0
                max_dd = abs(metrics.get('max_drawdown', 0)) or 0.01

                # Robustness score: ICIR × Sharpe / (1 + |Max Drawdown|)
                robustness = (icir * sharpe) / (
                    1 + max_dd) if icir > 0 and sharpe > 0 else 0
                robustness_scores[tw][fc] = robustness
            else:
                robustness_scores[tw][fc] = None

    for tw in time_windows:
        robustness_matrix_html += f"<tr><td><strong>{tw}</strong></td>"
        for fc in factor_counts:
            score = robustness_scores[tw].get(fc)
            if score is not None:
                cell_content = _format_float(score, 3)
                color_class = "good" if score > 0.5 else "warn" if score > 0 else "bad"
                robustness_matrix_html += f"<td class=\"{color_class}\">{cell_content}</td>"
            else:
                robustness_matrix_html += "<td>-</td>"
        robustness_matrix_html += "</tr>"

    robustness_matrix_html += "</table>"
    robustness_matrix_html += """
    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #6f42c1;">
        <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
        <ul>
            <li><strong>Robustness Score定义：</strong>稳健性得分 = ICIR × Sharpe Ratio / (1 + |最大回撤|)，这是一个综合指标，同时考虑了因子的预测稳定性（ICIR）、策略的风险调整收益（Sharpe Ratio）和风险控制（最大回撤）。</li>
            <li><strong>计算公式说明：</strong>
                <ul>
                    <li>ICIR：衡量因子预测的稳定性</li>
                    <li>Sharpe Ratio：衡量策略的风险调整收益</li>
                    <li>最大回撤：衡量策略的最大风险</li>
                    <li>分母 (1 + |最大回撤|)：惩罚高回撤的策略，回撤越大，得分越低</li>
                </ul>
            </li>
            <li><strong>表格结构：</strong>行表示时间窗口，列表示因子数量。每个单元格显示该组合的Robustness Score值。</li>
            <li><strong>颜色编码：</strong>
                <ul>
                    <li><span style="color: #167a3d; font-weight: 600;">绿色</span>：Robustness Score > 0.5，表示综合表现优秀</li>
                    <li><span style="color: #b36b00; font-weight: 600;">橙色</span>：0 < Robustness Score ≤ 0.5，表示综合表现一般</li>
                    <li><span style="color: #c53030; font-weight: 600;">红色</span>：Robustness Score ≤ 0，表示综合表现不佳</li>
                </ul>
            </li>
            <li><strong>分析要点：</strong>
                <ul>
                    <li>这是最全面的评估指标，综合考虑了预测能力、收益和风险</li>
                    <li>优先选择Robustness Score最高的组合，因为它平衡了所有关键因素</li>
                    <li>比较不同因子数量的Robustness Score，找出最优的因子数量</li>
                    <li>观察不同时间窗口的Robustness Score，评估策略的长期稳定性</li>
                </ul>
            </li>
            <li><strong>结论：</strong>此表格是选择最优参数组合的最重要参考。高Robustness Score意味着因子组合不仅在预测上有效，而且在实际交易中能产生稳定的风险调整收益。建议优先选择Robustness Score > 0.5且在不同时间窗口都表现稳定的组合。</li>
        </ul>
    </div>
    </div>"""

    # Build detailed results section with enhanced metrics
    details_html = "<div class=\"card\"><h3>📋 Detailed Results</h3>"
    for i, result in enumerate(enhanced_results, 1):
        params = result.get('grid_search_params', {})
        perf = result.get('performance', {}).get('stage3_representatives', {})
        # financial_metrics is stored inside perf_reps
        financial = perf.get('financial_metrics', {}) if isinstance(
            perf, dict) else {}
        if not financial:
            financial = result.get('performance',
                                   {}).get('stage3_representatives_financial',
                                           {})
        metrics = result.get('enhanced_metrics', {})

        details_html += f"<h4>Combination {i}: {params.get('time_window')} | Factors: {params.get('factor_count')}</h4>"
        details_html += "<table class=\"metric-table\">"

        if is_classification:
            # win_rate is in financial_metrics, not directly in perf
            win_rate = financial.get('win_rate', 0) if financial else 0
            if win_rate == 0:
                win_rate = perf.get('win_rate', 0)
            # f1_macro and accuracy are in classification_metrics
            classification_metrics = perf.get('classification_metrics',
                                              {}) if isinstance(perf,
                                                                dict) else {}
            f1_macro = classification_metrics.get(
                'f1_macro', 0) if classification_metrics else 0
            accuracy = classification_metrics.get(
                'accuracy', 0) if classification_metrics else 0

            details_html += f"<tr><th>Directional Win Rate</th><td>{_format_float(win_rate * 100, 2)}%</td></tr>"
            details_html += f"<tr><th>F1 (Macro)</th><td>{_format_float(f1_macro, 4)}</td></tr>"
            details_html += f"<tr><th>Accuracy</th><td>{_format_float(accuracy * 100, 2)}%</td></tr>"
        else:
            details_html += f"<tr><th>R²</th><td>{_format_float(perf.get('r2', 0), 4)}</td></tr>"
            details_html += f"<tr><th>RMSE</th><td>{_format_float(perf.get('rmse', 0), 4)}</td></tr>"
            details_html += f"<tr><th>MAE</th><td>{_format_float(perf.get('mae', 0), 4)}</td></tr>"

        # Add financial metrics
        if financial:
            details_html += f"<tr><th>Sharpe Ratio</th><td>{_format_float(financial.get('sharpe_ratio', 0), 3)}</td></tr>"
            details_html += f"<tr><th>Max Drawdown</th><td>{_format_float(financial.get('max_drawdown', 0) * 100, 2)}%</td></tr>"
            details_html += f"<tr><th>Total Return</th><td>{_format_float(financial.get('total_return', 0) * 100, 2)}%</td></tr>"

        # Add ICIR if available
        if metrics.get('icir') is not None:
            details_html += f"<tr><th>ICIR</th><td>{_format_float(metrics.get('icir'), 3)}</td></tr>"

        # Add robustness score
        icir = metrics.get('icir', 0) or 0
        sharpe = metrics.get('sharpe', 0) or 0
        max_dd = abs(metrics.get('max_drawdown', 0)) or 0.01
        robustness = (icir *
                      sharpe) / (1 + max_dd) if icir > 0 and sharpe > 0 else 0
        details_html += f"<tr><th>Robustness Score</th><td>{_format_float(robustness, 3)}</td></tr>"

        details_html += "</table>"
        details_html += """
        <div style="margin-top: 15px; padding: 12px; background: #e9ecef; border-radius: 5px; font-size: 0.9em;">
            <strong>📖 指标说明：</strong>
            <ul style="margin: 5px 0;">
                <li><strong>Directional Win Rate / R²：</strong>主要性能指标。分类任务使用胜率，回归任务使用R²。值越高越好。</li>
                <li><strong>F1 (Macro) / RMSE / MAE：</strong>辅助性能指标。F1用于分类，RMSE/MAE用于回归。F1越高越好，RMSE/MAE越低越好。</li>
                <li><strong>Sharpe Ratio：</strong>风险调整收益。> 1.0表示表现良好，> 2.0表示表现优秀。</li>
                <li><strong>Max Drawdown：</strong>最大回撤，衡量策略的最大风险。绝对值越小越好。</li>
                <li><strong>Total Return：</strong>总收益率。正值表示盈利，负值表示亏损。</li>
                <li><strong>ICIR：</strong>因子预测稳定性。> 1.0表示因子具有稳定的预测能力。</li>
                <li><strong>Robustness Score：</strong>综合稳健性得分，综合考虑ICIR、Sharpe和回撤。> 0.5表示综合表现优秀。</li>
            </ul>
        </div>
        <br/>"""

    details_html += """
    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #dc3545;">
        <h4>📖 如何阅读详细结果表格 (How to Read Detailed Results):</h4>
        <ul>
            <li><strong>表格结构：</strong>每个组合（时间窗口 + 因子数量）都有一个独立的详细结果表格，显示该组合的所有关键指标。</li>
            <li><strong>性能指标：</strong>
                <ul>
                    <li>主要关注<strong>Directional Win Rate</strong>（分类）或<strong>R²</strong>（回归），这是评估模型预测能力的主要指标</li>
                    <li>辅助指标（F1、Accuracy、RMSE、MAE）提供更全面的性能评估</li>
                </ul>
            </li>
            <li><strong>金融指标：</strong>
                <ul>
                    <li><strong>Sharpe Ratio</strong>：评估策略的风险调整收益，是实际交易中最重要的指标之一</li>
                    <li><strong>Max Drawdown</strong>：评估策略的最大风险，帮助了解最坏情况下的损失</li>
                    <li><strong>Total Return</strong>：评估策略的总收益表现</li>
                </ul>
            </li>
            <li><strong>因子质量指标：</strong>
                <ul>
                    <li><strong>ICIR</strong>：评估因子的预测稳定性，高ICIR表示因子在不同时期都能保持有效</li>
                    <li><strong>Robustness Score</strong>：综合评估因子组合的稳健性，是最全面的评估指标</li>
                </ul>
            </li>
            <li><strong>分析建议：</strong>
                <ul>
                    <li>优先查看Robustness Score，这是最全面的评估指标</li>
                    <li>结合ICIR和Sharpe Ratio，评估因子的预测能力和实际交易表现</li>
                    <li>注意Max Drawdown，确保风险在可接受范围内</li>
                    <li>比较不同组合的详细结果，找出最优参数配置</li>
                </ul>
            </li>
            <li><strong>结论：</strong>详细结果表格提供了每个参数组合的完整评估。通过对比不同组合的各项指标，可以全面了解每个组合的优势和劣势，从而做出最优的参数选择决策。</li>
        </ul>
    </div>
    </div>"""

    # Build ICIR trend analysis (Factor Count vs ICIR)
    icir_trend_html = ""
    if any(
            r.get('enhanced_metrics', {}).get('icir') is not None
            for r in enhanced_results):
        icir_trend_html = "<div class=\"card\"><h3>📈 ICIR Trend Analysis</h3>"
        icir_trend_html += "<p>ICIR vs Factor Count for each time window - Look for plateau points</p>"
        icir_trend_html += "<table class=\"metric-table\" style=\"width:100%;font-size:0.9em;\">"
        icir_trend_html += "<tr><th>Factor Count</th>"
        for tw in time_windows:
            icir_trend_html += f"<th>{tw}</th>"
        icir_trend_html += "<th>Mean ICIR</th><th>Std(ICIR)</th></tr>"

        # Calculate mean and std ICIR across time windows for each factor count
        for fc in factor_counts:
            icir_values = []
            icir_trend_html += f"<tr><td><strong>{fc}</strong></td>"
            for tw in time_windows:
                result = None
                for r in enhanced_results:
                    params = r.get('grid_search_params', {})
                    if params.get('time_window') == tw and params.get(
                            'factor_count') == fc:
                        result = r
                        break

                if result:
                    icir = result.get('enhanced_metrics', {}).get('icir')
                    if icir is not None:
                        icir_values.append(icir)
                        icir_trend_html += f"<td>{_format_float(icir, 3)}</td>"
                    else:
                        icir_trend_html += "<td>-</td>"
                else:
                    icir_trend_html += "<td>-</td>"

            # Mean and std across time windows
            if icir_values:
                mean_icir = sum(icir_values) / len(icir_values)
                std_icir = (sum((x - mean_icir)**2
                                for x in icir_values) / len(icir_values))**0.5
                icir_trend_html += f"<td>{_format_float(mean_icir, 3)}</td>"
                icir_trend_html += f"<td>{_format_float(std_icir, 3)}</td>"
            else:
                icir_trend_html += "<td>-</td><td>-</td>"
            icir_trend_html += "</tr>"

        icir_trend_html += "</table>"

        # Generate multiple 3D visualizations for Plateau Point analysis
        icir_trend_html += "<h4>🎯 3D Plateau Point 分析 (3D Plateau Point Analysis)</h4>"

        # ICIR 3D plot
        icir_3d_html = _generate_icir_3d_plot(enhanced_results, time_windows,
                                              factor_counts)
        icir_trend_html += icir_3d_html

        # Robustness Score 3D plot
        def get_robustness(result):
            metrics = result.get('enhanced_metrics', {})
            icir = metrics.get('icir', 0) or 0
            sharpe = metrics.get('sharpe', 0) or 0
            max_dd = abs(metrics.get('max_drawdown', 0)) or 0.01
            return (icir * sharpe) / (1 +
                                      max_dd) if icir > 0 and sharpe > 0 else 0

        robustness_3d_html = _generate_metric_3d_plot(enhanced_results,
                                                      time_windows,
                                                      factor_counts,
                                                      'robustness',
                                                      'Robustness Score',
                                                      get_robustness,
                                                      color_thresholds={
                                                          'good': 0.5,
                                                          'warn': 0.2,
                                                          'bad': 0.0
                                                      })
        icir_trend_html += robustness_3d_html

        # Sharpe Ratio 3D plot
        def get_sharpe(result):
            # Try to get from enhanced_metrics first
            sharpe = result.get('enhanced_metrics', {}).get('sharpe')
            if sharpe is None or sharpe == 0:
                # Fallback: extract from performance
                perf = result.get('performance',
                                  {}).get('stage3_representatives', {})
                financial = perf.get('financial_metrics', {}) if isinstance(
                    perf, dict) else {}
                if not financial:
                    financial = result.get('performance', {}).get(
                        'stage3_representatives_financial', {})
                sharpe = financial.get('sharpe_ratio', 0) if financial else 0
            return sharpe

        sharpe_3d_html = _generate_metric_3d_plot(enhanced_results,
                                                  time_windows,
                                                  factor_counts,
                                                  'sharpe',
                                                  'Sharpe Ratio',
                                                  get_sharpe,
                                                  color_thresholds={
                                                      'good': 1.0,
                                                      'warn': 0.0,
                                                      'bad': -1.0
                                                  })
        icir_trend_html += sharpe_3d_html

        # Primary metric 3D plot (Win Rate or R²)
        def get_primary_metric(result):
            perf = result.get('performance', {}).get('stage3_representatives',
                                                     {})
            # financial_metrics is stored inside perf_reps
            financial = perf.get('financial_metrics', {}) if isinstance(
                perf, dict) else {}
            if not financial:
                financial = result.get('performance', {}).get(
                    'stage3_representatives_financial', {})
            if is_classification:
                # win_rate is stored in financial_metrics
                win_rate = financial.get('win_rate') if financial else None
                if win_rate is None or win_rate == 0:
                    win_rate = perf.get('win_rate')
                return win_rate
            else:
                return perf.get('r2')

        primary_metric_label = 'Directional Win Rate' if is_classification else 'R²'
        primary_3d_html = _generate_metric_3d_plot(
            enhanced_results,
            time_windows,
            factor_counts,
            'primary',
            primary_metric_label,
            get_primary_metric,
            color_thresholds={
                'good': 0.5,
                'warn': 0.0,
                'bad': -0.5
            } if is_classification else {
                'good': 0.5,
                'warn': 0.0,
                'bad': -1.0
            })
        icir_trend_html += primary_3d_html

        # Add summary section for multiple 3D visualizations
        icir_trend_html += """
        <div style="margin-top: 30px; padding: 20px; background: #e7f3ff; border-radius: 5px; border-left: 4px solid #0066cc;">
            <h4>🎯 多指标Plateau Point综合分析 (Multi-Metric Plateau Point Analysis)</h4>
            <p><strong>为什么需要多个3D可视化？</strong></p>
            <ul>
                <li><strong>ICIR 3D图：</strong>识别因子预测稳定性的Plateau Point。高ICIR表示因子在不同时期都能保持有效预测。</li>
                <li><strong>Robustness Score 3D图：</strong>识别综合稳健性的Plateau Point。这是最全面的指标，综合考虑了预测能力、收益和风险。</li>
                <li><strong>Sharpe Ratio 3D图：</strong>识别风险调整收益的Plateau Point。高Sharpe Ratio表示策略在控制风险的同时获得良好收益。</li>
                <li><strong>Primary Metric 3D图：</strong>识别主要性能指标的Plateau Point。对于分类任务是胜率，对于回归任务是R²。</li>
            </ul>
            <p><strong>如何综合使用这些3D图？</strong></p>
            <ol>
                <li><strong>第一步：</strong>查看Robustness Score 3D图，找出综合表现最优的因子数量范围（绿色区域且表面平坦的位置）。</li>
                <li><strong>第二步：</strong>验证ICIR 3D图，确保该因子数量在ICIR上也表现稳定（高ICIR且在不同时间窗口都保持稳定）。</li>
                <li><strong>第三步：</strong>检查Sharpe Ratio 3D图，确认该因子数量在实际交易中能产生良好的风险调整收益。</li>
                <li><strong>第四步：</strong>参考Primary Metric 3D图，确保主要性能指标也达到预期水平。</li>
                <li><strong>第五步：</strong>选择在所有或大部分指标上都显示Plateau Point的因子数量，这表示该数量是最优选择。</li>
            </ol>
            <p><strong>💡 关键洞察：</strong>理想的Plateau Point应该在不同指标的不同3D图中都显示为平坦区域，且在不同时间窗口（Y轴）上都保持相对稳定的高度。如果某个因子数量在多个指标的3D图中都显示为Plateau Point，那么它就是最优选择。</p>
        </div>
        """

        icir_trend_html += """
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #17a2b8;">
            <h4>📖 如何阅读此表格 (How to Read This Table):</h4>
            <ul>
                <li><strong>表格结构：</strong>行表示不同的因子数量，列表示不同的时间窗口。最后两列显示每个因子数量在所有时间窗口上的平均ICIR和标准差。</li>
                <li><strong>Mean ICIR列：</strong>计算该因子数量在所有时间窗口上的平均ICIR值。平均值越高，表示该因子数量在不同时期的表现越稳定。</li>
                <li><strong>Std(ICIR)列：</strong>计算该因子数量在所有时间窗口上的ICIR标准差。标准差越小，表示该因子数量在不同时期的表现越一致，稳定性越好。</li>
                <li><strong>分析要点：</strong>
                    <ul>
                        <li><strong>寻找平台点（Plateau Point）：</strong>找出ICIR不再显著下降的最小因子数量。例如，如果120个因子和60个因子的ICIR相近，但30个因子的ICIR明显下降，那么60个因子可能是平台点。</li>
                        <li><strong>评估稳定性：</strong>比较不同因子数量的Std(ICIR)。Std(ICIR)越小，表示该因子数量在不同时间窗口的表现越一致，越稳定。</li>
                        <li><strong>平衡性能与复杂度：</strong>在Mean ICIR高且Std(ICIR)低的前提下，选择因子数量最少的组合，以降低模型复杂度并提高可解释性。</li>
                        <li><strong>结合3D可视化：</strong>使用上方的3D图形可以更直观地识别Plateau Point。在3D图形中，Plateau Point表现为ICIR值达到较高水平后，表面变得平坦的区域。</li>
                    </ul>
                </li>
                <li><strong>结论：</strong>此表格帮助确定最优因子数量。理想的组合是：Mean ICIR高（> 1.0）、Std(ICIR)低（< 0.3），且因子数量尽可能少。这表示该因子数量既能保持高预测能力，又能在不同市场环境下保持稳定，同时避免了过度复杂化。结合3D可视化，可以更准确地识别Plateau Point。</li>
            </ul>
        </div>
        </div>"""

    # Build analysis conclusions
    analysis_conclusions_html = _build_analysis_conclusions(
        enhanced_results, time_windows, factor_counts, is_classification)

    # Generate ICIR heatmap
    heatmap_html = _generate_icir_heatmap(enhanced_results, time_windows,
                                          factor_counts)

    # Build full HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Grid Search Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .card {{ background: #fff; border-radius: 10px; padding: 18px 22px; box-shadow: 0 10px 24px rgba(27,39,53,0.1); margin: 20px 0; }}
        .metric-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        .metric-table th, .metric-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .metric-table th {{ background-color: #f8f9fa; font-weight: 600; }}
        .good {{ color: #167a3d; font-weight: 600; }}
        .warn {{ color: #b36b00; font-weight: 600; }}
        .bad {{ color: #c53030; font-weight: 600; }}
        .heatmap-container {{ margin: 20px 0; text-align: center; }}
    </style>
</head>
<body>
    <h1>🔍 Grid Search Comparison Report</h1>
    <div class="card">
        <h3>Configuration</h3>
        <p><strong>Symbol:</strong> {report_data.get('symbol', 'N/A')}</p>
        <p><strong>Feature Type:</strong> {report_data.get('feature_type', 'N/A')}</p>
        <p><strong>Task Type:</strong> {task_type}</p>
        <p><strong>Time Windows Tested:</strong> {len(time_windows)}</p>
        <p><strong>Factor Counts Tested:</strong> {len(factor_counts)}</p>
        <p><strong>Total Combinations:</strong> {len(grid_search_results)}</p>
    </div>
    
    {matrix_html}
    {icir_matrix_html}
    {sharpe_matrix_html}
    {robustness_matrix_html}
    {icir_trend_html}
    {heatmap_html}
    {details_html}
    
    {analysis_conclusions_html}
    
    <div class="card">
        <p>Generated by ML Trading Bot Rolling Training System</p>
    </div>
</body>
</html>"""

    # Write HTML to file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ Grid search HTML report saved to: {html_path}")


def main() -> str:
    print("📋 Dimensionality Training Report Generator")
    print("=" * 50)
    return generate_comprehensive_report()


if __name__ == "__main__":
    main()
