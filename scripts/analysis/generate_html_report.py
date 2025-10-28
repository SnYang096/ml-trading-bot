"""生成HTML格式的特征统计报告（带可视化图表）"""

import json
from pathlib import Path
from datetime import datetime


def generate_html_report():
    """生成HTML格式的特征统计报告."""

    # Load JSON data
    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / "reports" / "feature_count_data.json"

    if not json_path.exists():
        print("❌ 未找到 feature_count_data.json，请先运行 count_features.py")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Count Report - ML Trading Project</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .card:hover {{
            transform: translateY(-5px);
        }}
        .card-title {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        .card-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .card-subtitle {{
            font-size: 0.85em;
            opacity: 0.8;
        }}
        .chart-container {{
            margin: 40px 0;
            background: #f8f9fa;
            padding: 30px;
            border-radius: 12px;
        }}
        .chart-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .module-details {{
            margin: 30px 0;
        }}
        .module-card {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }}
        .module-card:hover {{
            border-color: #667eea;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        }}
        .module-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .module-name {{
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
        }}
        .module-count {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .module-file {{
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        .category {{
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .category-name {{
            font-weight: 600;
            color: #495057;
            margin-right: 10px;
        }}
        .category-count {{
            color: #667eea;
            font-weight: bold;
        }}
        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 30px 0;
            border-radius: 8px;
        }}
        .recommendations h3 {{
            color: #856404;
            margin-bottom: 15px;
        }}
        .recommendations ul {{
            list-style: none;
            padding-left: 0;
        }}
        .recommendations li {{
            padding: 8px 0;
            color: #856404;
        }}
        .recommendations li:before {{
            content: "✓ ";
            color: #28a745;
            font-weight: bold;
            margin-right: 8px;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Feature Count Report</h1>
            <p>ML Trading Project - Feature Engineering Analysis</p>
            <p style="font-size: 0.9em; margin-top: 10px;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <div class="summary-cards">
                <div class="card">
                    <div class="card-title">总特征数量</div>
                    <div class="card-value">{data['grand_total']}</div>
                    <div class="card-subtitle">Total Features</div>
                </div>
                <div class="card">
                    <div class="card-title">特征工程模块</div>
                    <div class="card-value">{len(data['modules'])}</div>
                    <div class="card-subtitle">Feature Modules</div>
                </div>
                <div class="card">
                    <div class="card-title">最多特征模块</div>
                    <div class="card-value">{max(data['modules'].values(), key=lambda x: x['total'])['total']}</div>
                    <div class="card-subtitle">Enhanced Module</div>
                </div>
                <div class="card">
                    <div class="card-title">最少特征模块</div>
                    <div class="card-value">{min(data['modules'].values(), key=lambda x: x['total'])['total']}</div>
                    <div class="card-subtitle">Basic Module</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2 class="chart-title">各模块特征数量分布</h2>
                <canvas id="moduleChart" height="80"></canvas>
            </div>
            
            <h2 style="margin: 40px 0 20px 0; color: #333;">📊 详细统计</h2>
            
            <table>
                <thead>
                    <tr>
                        <th>模块名称</th>
                        <th>文件名</th>
                        <th>特征数量</th>
                        <th>占比</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add table rows
    for module_name, module_data in data["modules"].items():
        percentage = (module_data["total"] / data["grand_total"]) * 100
        module_display_name = {
            "basic": "基础特征工程",
            "improved": "改进特征工程",
            "enhanced": "增强特征工程",
            "wavelet": "小波特征工程",
            "deep_learning": "深度学习特征",
        }.get(module_name, module_name)

        html += f"""
                    <tr>
                        <td><strong>{module_display_name}</strong></td>
                        <td><code>{module_data['file']}</code></td>
                        <td><strong>{module_data['total']}</strong></td>
                        <td>{percentage:.1f}%</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
            
            <h2 style="margin: 40px 0 20px 0; color: #333;">🔍 模块详情</h2>
            
            <div class="module-details">
"""

    # Add module details
    module_names = {
        "basic": "基础特征工程",
        "improved": "改进特征工程",
        "enhanced": "增强特征工程",
        "wavelet": "小波特征工程",
        "deep_learning": "深度学习特征",
    }

    for module_id, module_data in data["modules"].items():
        html += f"""
                <div class="module-card">
                    <div class="module-header">
                        <div class="module-name">{module_names.get(module_id, module_id)}</div>
                        <div class="module-count">{module_data['total']} 个特征</div>
                    </div>
                    <div class="module-file">📄 {module_data['file']}</div>
"""

        # Add categories
        for category, features in module_data["features"].items():
            feature_preview = ", ".join(features[:3])
            if len(features) > 3:
                feature_preview += f", ... (共 {len(features)} 个)"

            html += f"""
                    <div class="category">
                        <span class="category-name">{category}:</span>
                        <span class="category-count">{len(features)} 个</span>
                        <div style="color: #6c757d; font-size: 0.85em; margin-top: 5px;">{feature_preview}</div>
                    </div>
"""

        html += """
                </div>
"""

    html += """
            </div>
            
            <div class="recommendations">
                <h3>💡 建议 (Recommendations)</h3>
                <ul>
                    <li>使用特征重要性分析选择 top 100-200 个特征</li>
                    <li>快速原型使用基础版（13个特征）</li>
                    <li>标准训练使用改进版（25个特征）</li>
                    <li>高级研究使用增强版（331个特征）</li>
                    <li>深度学习可添加序列特征（64个特征）</li>
                    <li>注意计算成本：基础版最快，增强版最慢（WPT + Hurst）</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by ML Trading Project Feature Analysis Tool</p>
            <p>© 2025 - Feature Engineering Report</p>
        </div>
    </div>
    
    <script>
        // Chart data
        const moduleData = {
"""

    # Add chart data
    labels = []
    values = []
    for module_id, module_data in data["modules"].items():
        module_name = module_names.get(module_id, module_id)
        labels.append(module_name)
        values.append(module_data["total"])

    html += f"""
            labels: {json.dumps(labels, ensure_ascii=False)},
            values: {json.dumps(values)}
        }};
        
        // Create chart
        const ctx = document.getElementById('moduleChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: moduleData.labels,
                datasets: [{{
                    label: '特征数量',
                    data: moduleData.values,
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(118, 75, 162, 0.8)',
                        'rgba(237, 100, 166, 0.8)',
                        'rgba(255, 154, 158, 0.8)',
                        'rgba(250, 208, 196, 0.8)'
                    ],
                    borderColor: [
                        'rgba(102, 126, 234, 1)',
                        'rgba(118, 75, 162, 1)',
                        'rgba(237, 100, 166, 1)',
                        'rgba(255, 154, 158, 1)',
                        'rgba(250, 208, 196, 1)'
                    ],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    title: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{
                            color: 'rgba(0, 0, 0, 0.05)'
                        }}
                    }},
                    x: {{
                        grid: {{
                            display: false
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    # Save HTML report
    html_path = project_root / "reports" / "feature_count_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ HTML报告已生成: {html_path}")
    return html_path


if __name__ == "__main__":
    print("\n📊 生成HTML可视化报告...\n")
    generate_html_report()
