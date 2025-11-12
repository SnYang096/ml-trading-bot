#!/usr/bin/env python3
"""
特征指标可视化工具
将 hurst, hilbert, wavelet, spectral 等特征转化成指标并绘制在 HTML 上
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from data_tools.data_loader import MarketDataLoader


def extract_feature_columns(df: pd.DataFrame, feature_type: str) -> Dict[str, List[str]]:
    """提取指定类型的特征列名
    
    Args:
        df: 包含特征的 DataFrame
        feature_type: 特征类型 ('hurst', 'hilbert', 'wavelet', 'spectral')
    
    Returns:
        字典，键为信号源名称，值为该信号源的特征列名列表
    """
    feature_columns = {}
    
    # 定义信号源
    signal_sources = ['close', 'open', 'volume', 'cvd', 'taker_buy_ratio']
    
    for source in signal_sources:
        cols = []
        if feature_type == 'hurst':
            # Hurst 特征列
            hurst_cols = [col for col in df.columns if col.startswith(f'{source}_hurst')]
            cols.extend(hurst_cols)
        elif feature_type == 'hilbert':
            # Hilbert 特征列（排除原始幅度和频率，只保留归一化的相位）
            hilbert_cols = [col for col in df.columns 
                          if col.startswith(f'{source}_hilbert') 
                          and not col.endswith('_hilbert_amplitude')
                          and not col.endswith('_hilbert_frequency')]
            cols.extend(hilbert_cols)
        elif feature_type == 'wavelet':
            # Wavelet 特征列
            wavelet_cols = [col for col in df.columns if col.startswith(f'{source}_wpt')]
            cols.extend(wavelet_cols)
        elif feature_type == 'spectral':
            # Spectral 特征列
            spectral_cols = [col for col in df.columns if col.startswith(f'{source}_spectral')]
            cols.extend(spectral_cols)
        
        # 过滤掉不存在的列
        cols = [col for col in cols if col in df.columns]
        
        if cols:
            feature_columns[source] = cols
    
    return feature_columns


def create_feature_chart(
    df: pd.DataFrame,
    price_col: str,
    feature_cols: List[str],
    feature_type: str,
    source_name: str,
    max_points: int = 5000
) -> go.Figure:
    """创建特征指标图表
    
    Args:
        df: 包含价格和特征的 DataFrame
        price_col: 价格列名（如 'close'）
        feature_cols: 特征列名列表
        feature_type: 特征类型名称
        source_name: 信号源名称
        max_points: 最大显示点数（用于性能优化）
    
    Returns:
        Plotly Figure 对象
    """
    # 采样数据以提高性能
    if len(df) > max_points:
        step = len(df) // max_points
        df_plot = df.iloc[::step].copy()
    else:
        df_plot = df.copy()
    
    # 创建子图：价格 + 特征
    n_features = len(feature_cols)
    if n_features == 0:
        # 如果没有特征，只显示价格
        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.1)
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[price_col],
                name=price_col.upper(),
                line=dict(color='#1f77b4', width=1)
            ),
            row=1, col=1
        )
        fig.update_layout(
            title=f"{source_name.upper()} Price",
            height=400
        )
        return fig
    
    # 创建多子图布局
    fig = make_subplots(
        rows=n_features + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[price_col.upper()] + [col.replace(f'{source_name}_', '').replace('_', ' ').title() for col in feature_cols],
        row_heights=[0.4] + [0.6 / n_features] * n_features
    )
    
    # 绘制价格
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot[price_col],
            name=price_col.upper(),
            line=dict(color='#1f77b4', width=1),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 绘制每个特征
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for idx, col in enumerate(feature_cols):
        if col not in df_plot.columns:
            continue
        
        row = idx + 2
        color = colors[idx % len(colors)]
        
        # 清理数据：移除 NaN 和 Inf
        y_data = df_plot[col].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        
        # 根据特征类型选择不同的图表类型
        if feature_type == 'hilbert' and 'phase' in col:
            # 相位特征使用角度图
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=y_data,
                    name=col,
                    line=dict(color=color, width=1),
                    showlegend=False
                ),
                row=row, col=1
            )
        else:
            # 其他特征使用普通线图
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=y_data,
                    name=col,
                    line=dict(color=color, width=1),
                    showlegend=False
                ),
                row=row, col=1
            )
    
    # 更新布局
    fig.update_layout(
        title=f"{source_name.upper()} - {feature_type.upper()} Features",
        height=300 * (n_features + 1),
        hovermode='x unified',
        template='plotly_white'
    )
    
    # 更新 x 轴
    fig.update_xaxes(title_text="Time", row=n_features + 1, col=1)
    
    return fig


def generate_html_report(
    df: pd.DataFrame,
    output_path: str,
    symbol: str,
    feature_types: List[str] = ['hurst', 'hilbert', 'wavelet', 'spectral'],
    timeframe: str = "5T"
):
    """生成包含特征图表的 HTML 报告
    
    Args:
        df: 包含价格和特征的 DataFrame
        output_path: 输出 HTML 文件路径
        symbol: 交易对符号
        feature_types: 要可视化的特征类型列表
        timeframe: 时间框架
    """
    print(f"📊 Generating feature indicator visualization for {symbol}...")
    
    # 提取价格列
    price_col = 'close' if 'close' in df.columns else df.columns[0]
    
    # 生成所有图表
    charts_html = []
    
    for feature_type in feature_types:
        print(f"   Processing {feature_type} features...")
        feature_columns = extract_feature_columns(df, feature_type)
        
        if not feature_columns:
            print(f"   ⚠️  No {feature_type} features found, skipping...")
            continue
        
        # 为每个信号源创建图表
        for source_name, cols in feature_columns.items():
            if not cols:
                continue
            
            print(f"      Creating chart for {source_name} ({len(cols)} features)...")
            fig = create_feature_chart(df, price_col, cols, feature_type, source_name)
            
            # 将图表转换为 HTML
            chart_html = fig.to_html(include_plotlyjs='cdn', div_id=f"{feature_type}_{source_name}")
            charts_html.append({
                'type': feature_type,
                'source': source_name,
                'html': chart_html,
                'feature_count': len(cols)
            })
    
    # 生成完整的 HTML 报告
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Feature Indicators Visualization - {symbol}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #ecf0f1;
            border-left: 4px solid #3498db;
        }}
        .info-box {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .info-box table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .info-box th {{
            text-align: left;
            padding: 8px;
            color: #495057;
            font-weight: 600;
        }}
        .info-box td {{
            padding: 8px;
            color: #6c757d;
        }}
        .chart-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 4px;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }}
        .feature-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }}
        .badge-hurst {{
            background-color: #e3f2fd;
            color: #1976d2;
        }}
        .badge-hilbert {{
            background-color: #f3e5f5;
            color: #7b1fa2;
        }}
        .badge-wavelet {{
            background-color: #e8f5e9;
            color: #388e3c;
        }}
        .badge-spectral {{
            background-color: #fff3e0;
            color: #f57c00;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Feature Indicators Visualization</h1>
        
        <div class="info-box">
            <h3>📋 Report Information</h3>
            <table>
                <tr>
                    <th>Symbol</th>
                    <td>{symbol}</td>
                </tr>
                <tr>
                    <th>Timeframe</th>
                    <td>{timeframe}</td>
                </tr>
                <tr>
                    <th>Data Period</th>
                    <td>{df.index[0]} to {df.index[-1]}</td>
                </tr>
                <tr>
                    <th>Total Data Points</th>
                    <td>{len(df):,}</td>
                </tr>
                <tr>
                    <th>Feature Types</th>
                    <td>{', '.join(feature_types)}</td>
                </tr>
            </table>
        </div>
"""
    
    # 添加图表
    for chart_info in charts_html:
        badge_class = f"badge-{chart_info['type']}"
        html_content += f"""
        <div class="chart-section">
            <div class="chart-title">
                <span class="feature-badge {badge_class}">{chart_info['type'].upper()}</span>
                <span class="feature-badge">{chart_info['source'].upper()}</span>
                <span style="float: right; color: #6c757d; font-size: 14px; font-weight: normal;">
                    {chart_info['feature_count']} features
                </span>
            </div>
            {chart_info['html']}
        </div>
"""
    
    html_content += """
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            <p>Generated by ML Trading Bot Feature Indicator Visualizer</p>
            <p>Generated at: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 保存 HTML 文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hurst, hilbert, wavelet, spectral features as indicators in HTML"
    )
    parser.add_argument(
        "--data-path",
        default="/workspace/data/parquet_data",
        help="Path to parquet data directory"
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Symbol name (e.g., BTCUSDT, ETHUSDT)"
    )
    parser.add_argument(
        "--timeframe",
        default="5T",
        help="Timeframe for data resampling (e.g., 5T, 15T, 60T, 240T)"
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--feature-types",
        default="hurst,hilbert,wavelet,spectral",
        help="Comma-separated list of feature types to visualize"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML file path (default: results/feature_indicators/{symbol}_{timeframe}.html)"
    )
    parser.add_argument(
        "--feature-type",
        default="comprehensive",
        help="Feature engineering type (default: comprehensive)"
    )
    
    args = parser.parse_args()
    
    # 解析特征类型
    feature_types = [ft.strip() for ft in args.feature_types.split(',')]
    
    # 加载数据
    print(f"📊 Loading data for {args.symbol}...")
    loader = MarketDataLoader(args.data_path)
    df = loader.load_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if df is None or df.empty:
        print(f"❌ No data found for {args.symbol}")
        return
    
    # 重采样数据
    if hasattr(loader, 'resample_data'):
        df = loader.resample_data(args.timeframe)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.resample(args.timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    print(f"✅ Loaded {len(df)} data points")
    
    # 特征工程
    print(f"🔧 Engineering features ({args.feature_type})...")
    engineer = ComprehensiveFeatureEngineer(feature_types=args.feature_type)
    df_features = engineer.engineer_all_features(df, fit=True)
    
    print(f"✅ Generated {len(df_features.columns)} features")
    
    # 生成输出路径
    if args.output is None:
        symbol_slug = args.symbol.replace('-', '_').replace('/', '_')
        timeframe_slug = args.timeframe.replace('T', 'min')
        output_path = f"results/feature_indicators/{symbol_slug}_{timeframe_slug}.html"
    else:
        output_path = args.output
    
    # 生成 HTML 报告
    generate_html_report(
        df_features,
        output_path,
        args.symbol,
        feature_types,
        args.timeframe
    )
    
    print(f"🎉 Visualization complete!")


if __name__ == "__main__":
    main()

