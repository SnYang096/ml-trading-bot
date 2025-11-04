"""Generate summary report for training results (no 'baseline' naming)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd


def collect_training_results(results_dir: str = "results/training") -> pd.DataFrame:
    results_path = Path(results_dir)
    if not results_path.exists():
        return pd.DataFrame()

    all_results = []

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


def generate_summary_report(results_dir: str = "results/training", output_path: Optional[str] = None) -> str:
    df = collect_training_results(results_dir)
    if df.empty:
        print("No training results found.")
        return ""

    if output_path is None:
        output_path = os.path.join(results_dir, "summary_report.html")

    # Sort for readability
    if {"timeframe", "forward_bars"}.issubset(df.columns):
        df = df.sort_values(["timeframe", "forward_bars"])  # type: ignore

    # Build rows
    rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        timeframe = row_dict.get("timeframe", "N/A")
        forward_bars = row_dict.get("forward_bars", "N/A")
        symbol = row_dict.get("symbol", "N/A")
        config_dir = row_dict.get("config_dir", "N/A")
        metrics = row_dict.get("metrics", {}) or {}
        stage2_metrics = metrics.get("stage2", {}) if isinstance(metrics, dict) else {}
        tf_m = stage2_metrics.get(timeframe, {}) if isinstance(stage2_metrics, dict) else {}
        cv_rmse = tf_m.get("cv_rmse")
        cv_mse = tf_m.get("cv_mse")
        # Fallback: derive from fold_details if missing
        if (cv_rmse is None or cv_mse is None) and isinstance(tf_m.get("fold_details"), list):
            try:
                rmses = [fd.get("rmse") for fd in tf_m["fold_details"] if isinstance(fd, dict) and fd.get("rmse") is not None]
                mses = [fd.get("mse") for fd in tf_m["fold_details"] if isinstance(fd, dict) and fd.get("mse") is not None]
                if cv_rmse is None and rmses:
                    cv_rmse = float(sum(rmses) / len(rmses))
                if cv_mse is None and mses:
                    cv_mse = float(sum(mses) / len(mses))
            except Exception:
                pass
        # classification (train) metrics
        cls_train = metrics.get("classification_train", {}) if isinstance(metrics, dict) else {}
        cls_tf = cls_train.get(timeframe, {}) if isinstance(cls_train, dict) else {}
        f1 = cls_tf.get("f1")
        acc = cls_tf.get("accuracy")
        prec = cls_tf.get("precision")
        rec = cls_tf.get("recall")
        auc = cls_tf.get("auc")
        pr_auc = cls_tf.get("pr_auc")
        feature_type = row_dict.get("feature_type", "N/A")
        train_bars = row_dict.get("train_bars") or row_dict.get("total_bars", 0)

        # Helper functions
        def _format_metric(val, fmt=".4f"):
            if val is None:
                return "N/A"
            try:
                return f"{val:{fmt}}"
            except:
                return str(val)

        def _quality_color(val, threshold_good, threshold_excellent=None):
            if val is None:
                return ""
            if threshold_excellent and val >= threshold_excellent:
                return ' style="background-color:#d4edda; color:#155724;"'
            if val >= threshold_good:
                return ' style="background-color:#fff3cd; color:#856404;"'
            return ' style="background-color:#f8d7da; color:#721c24;"'

        # Quality assessment
        f1_color = _quality_color(f1, 0.3, 0.5)
        auc_color = _quality_color(auc, 0.6, 0.7) if auc is not None else ""
        pr_auc_color = _quality_color(pr_auc, 0.4, 0.6) if pr_auc is not None else ""
        quality_passed = (f1 is not None and f1 >= 0.3) or (auc is not None and auc >= 0.6)
        quality_badge = ('<span style="background-color:#d4edda; color:#155724; padding:2px 6px; border-radius:4px;">✅ PASS</span>' 
                        if quality_passed 
                        else '<span style="background-color:#f8d7da; color:#721c24; padding:2px 6px; border-radius:4px;">❌ FAIL</span>')

        rows.append(
            f"<tr><td>{symbol}</td><td>{timeframe}</td><td>{forward_bars}</td>"
            f"<td>{train_bars:,}</td>"
            f"<td>{_format_metric(cv_rmse, '.6f')}</td>"
            f"<td>{_format_metric(cv_mse, '.8f')}</td>"
            f"<td{f1_color}>{_format_metric(f1)}</td>"
            f"<td>{_format_metric(acc)}</td>"
            f"<td>{_format_metric(prec)}</td>"
            f"<td>{_format_metric(rec)}</td>"
            f"<td{auc_color}>{_format_metric(auc)}</td>"
            f"<td{pr_auc_color}>{_format_metric(pr_auc)}</td>"
            f"<td>{feature_type}</td>"
            f"<td>{quality_badge}</td>"
            f"<td>{config_dir}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Training Summary Report</title>
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
<h1>Training Summary Report</h1>
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="explanation">
<h3>📊 指标说明与好坏判断</h3>
<ul>
<li><strong>F1 Score</strong> (综合指标): 精确率和召回率的调和平均，推荐阈值 F1 &gt; 0.3
    <ul>
        <li>✅ 优秀 (绿色): F1 &gt;= 0.5</li>
        <li>⚠️ 良好 (黄色): F1 &gt;= 0.3</li>
        <li>❌ 较差 (红色): F1 &lt; 0.3</li>
    </ul>
</li>
<li><strong>AUC-ROC</strong> (区分能力): 衡量模型区分正负样本的能力，对阈值不敏感，推荐阈值 AUC &gt; 0.6
    <ul>
        <li>✅ 优秀 (绿色): AUC &gt;= 0.7</li>
        <li>⚠️ 良好 (黄色): AUC &gt;= 0.6</li>
        <li>❌ 较差 (红色): AUC &lt; 0.6</li>
    </ul>
</li>
<li><strong>PR-AUC</strong> (精确率-召回率曲线下面积): 更适合不平衡数据，推荐阈值 PR-AUC &gt; 0.4
    <ul>
        <li>✅ 优秀 (绿色): PR-AUC &gt;= 0.6</li>
        <li>⚠️ 良好 (黄色): PR-AUC &gt;= 0.4</li>
        <li>❌ 较差 (红色): PR-AUC &lt; 0.4</li>
    </ul>
</li>
<li><strong>Precision</strong> (精确率): 控制误开仓，预测为做多时真的做多比例</li>
<li><strong>Recall</strong> (召回率): 抓住行情能力，实际该做多时模型抓到比例</li>
<li><strong>Accuracy</strong> (准确率): 总体分类准确率，在不平衡数据上可能虚高</li>
<li><strong>CV RMSE/MSE</strong>: 交叉验证的回归误差，越低越好</li>
<li><strong>Quality</strong>: 模型质量检查，✅ PASS = F1 &gt;= 0.3 或 AUC &gt;= 0.6，❌ FAIL = 两者都不满足</li>
</ul>
<h3>💡 特征类型比较</h3>
<ul>
<li><strong>baseline</strong>: 基础特征（传统技术指标），特征数量较少，计算快，适合快速验证</li>
<li><strong>default</strong>: 默认特征（TA-Lib + base_indicators），比baseline更丰富，推荐用于生产环境</li>
<li><strong>enhanced</strong>: 增强特征（包含更多高级特征），特征更全面但计算更慢</li>
<li><strong>comprehensive</strong>: 综合特征（最完整，包含所有特征类型），特征最丰富但可能过拟合</li>
</ul>
<p><strong>⚠️ 注意:</strong> 如果Accuracy很高（接近90%+）但F1很低，可能是模型在"背答案"（预测多数类），需要检查数据平衡性。</p>
<p><strong>💡 建议:</strong> 比较不同特征类型（baseline vs default）时，重点关注F1、AUC和PR-AUC，这些指标更能反映模型真实性能。</p>
</div>
<table>
<tr><th>Symbol</th><th>Timeframe</th><th>Forward Bars</th><th>Training Bars</th><th>CV RMSE</th><th>CV MSE</th><th>F1</th><th>Acc</th><th>Prec</th><th>Rec</th><th>AUC</th><th>PR-AUC</th><th>Feature Type</th><th>Quality</th><th>Config</th></tr>
{''.join(rows)}
</table>
</div>
</body></html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Summary report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate training summary report")
    parser.add_argument("--results-dir", type=str, default="results/training")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    generate_summary_report(args.results_dir, args.output)


