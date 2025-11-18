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


def detect_trading_signals(
    df: pd.DataFrame, price_col: str, feature_cols: List[str], feature_type: str
) -> Tuple[pd.Series, pd.Series]:
    """检测买卖信号

    Args:
        df: 包含价格和特征的 DataFrame
        price_col: 价格列名
        feature_cols: 特征列名列表
        feature_type: 特征类型

    Returns:
        (buy_signals, sell_signals): 买卖信号序列，True表示信号点
    """
    buy_signals = pd.Series(False, index=df.index)
    sell_signals = pd.Series(False, index=df.index)

    if len(feature_cols) == 0:
        return buy_signals, sell_signals

    # 根据特征类型使用不同的信号检测策略
    if feature_type == "hurst":
        # Hurst指数：低值(<0.5)可能表示趋势反转，高值(>0.5)表示趋势持续
        # 当Hurst从高转低时可能是买入信号，从低转高时可能是卖出信号
        for col in feature_cols:
            if col not in df.columns:
                continue
            hurst = df[col].replace([np.inf, -np.inf], np.nan).ffill()
            if hurst.isna().all():
                continue

            # 计算Hurst的移动平均
            ma_short = hurst.rolling(window=5, min_periods=1).mean()
            ma_long = hurst.rolling(window=20, min_periods=1).mean()

            # 买入信号：短期Hurst上穿长期，且当前值较低（反转信号）
            buy = (ma_short > ma_long) & (hurst < 0.4) & (hurst.shift(1) <= hurst)
            # 卖出信号：短期Hurst下穿长期，且当前值较高
            sell = (ma_short < ma_long) & (hurst > 0.6) & (hurst.shift(1) >= hurst)

            buy_signals |= buy
            sell_signals |= sell

    elif feature_type == "hilbert":
        # Hilbert相位：相位变化表示趋势转折
        # 相位从负转正可能是买入信号，从正转负可能是卖出信号
        for col in feature_cols:
            if col not in df.columns or "phase" not in col:
                continue
            phase = df[col].replace([np.inf, -np.inf], np.nan).ffill()
            if phase.isna().all():
                continue

            # 相位变化检测
            phase_diff = phase.diff()
            # 买入：相位大幅上升（趋势转折向上）
            buy = (phase_diff > phase_diff.rolling(10).quantile(0.8)) & (phase > 0)
            # 卖出：相位大幅下降（趋势转折向下）
            sell = (phase_diff < phase_diff.rolling(10).quantile(0.2)) & (phase < 0)

            buy_signals |= buy
            sell_signals |= sell

    elif feature_type == "wavelet":
        # Wavelet：不同频段的能量变化
        # 高频能量增加可能是波动增加（卖出），低频能量增加可能是趋势（买入）
        for col in feature_cols:
            if col not in df.columns:
                continue
            wavelet = df[col].replace([np.inf, -np.inf], np.nan).ffill()
            if wavelet.isna().all():
                continue

            # 计算变化率
            change = wavelet.pct_change()
            ma_change = change.rolling(window=10, min_periods=1).mean()

            # 买入：能量增加且为正
            buy = (change > ma_change * 1.5) & (wavelet > 0)
            # 卖出：能量减少或为负
            sell = (change < ma_change * -1.5) | (wavelet < 0)

            buy_signals |= buy
            sell_signals |= sell

    elif feature_type == "spectral":
        # Spectral：频谱特征的变化
        # 主频率变化可能表示趋势变化
        for col in feature_cols:
            if col not in df.columns:
                continue
            spectral = df[col].replace([np.inf, -np.inf], np.nan).ffill()
            if spectral.isna().all():
                continue

            # 计算Z-score
            zscore = (spectral - spectral.rolling(20).mean()) / (
                spectral.rolling(20).std() + 1e-8
            )

            # 买入：Z-score从负转正
            buy = (zscore > 1) & (zscore.shift(1) <= 0)
            # 卖出：Z-score从正转负
            sell = (zscore < -1) & (zscore.shift(1) >= 0)

            buy_signals |= buy
            sell_signals |= sell

    # 去重：避免同一时间点多个信号
    # 优先保留更强的信号（基于价格变化）
    price_change = df[price_col].pct_change().abs()

    # 合并信号，保留价格变化较大的点
    buy_indices = buy_signals[buy_signals].index
    sell_indices = sell_signals[sell_signals].index

    # 如果信号太密集，只保留变化最大的
    if len(buy_indices) > len(df) * 0.1:  # 如果信号超过10%的点
        buy_price_changes = price_change.loc[buy_indices]
        buy_indices = buy_price_changes.nlargest(
            int(len(df) * 0.05)
        ).index  # 只保留前5%
        buy_signals = pd.Series(False, index=df.index)
        buy_signals.loc[buy_indices] = True

    if len(sell_indices) > len(df) * 0.1:
        sell_price_changes = price_change.loc[sell_indices]
        sell_indices = sell_price_changes.nlargest(int(len(df) * 0.05)).index
        sell_signals = pd.Series(False, index=df.index)
        sell_signals.loc[sell_indices] = True

    return buy_signals, sell_signals


def extract_feature_columns(
    df: pd.DataFrame, feature_type: str
) -> Dict[str, List[str]]:
    """提取指定类型的特征列名

    Args:
        df: 包含特征的 DataFrame
        feature_type: 特征类型 ('hurst', 'hilbert', 'wavelet', 'spectral')

    Returns:
        字典，键为信号源名称，值为该信号源的特征列名列表
    """
    feature_columns = {}

    # 定义信号源
    signal_sources = ["close", "open", "volume", "cvd", "taker_buy_ratio"]

    for source in signal_sources:
        cols = []
        if feature_type == "hurst":
            # Hurst 特征列
            hurst_cols = [
                col for col in df.columns if col.startswith(f"{source}_hurst")
            ]
            cols.extend(hurst_cols)
        elif feature_type == "hilbert":
            # Hilbert 特征列（排除原始幅度和频率，只保留归一化的相位）
            hilbert_cols = [
                col
                for col in df.columns
                if col.startswith(f"{source}_hilbert")
                and not col.endswith("_hilbert_amplitude")
                and not col.endswith("_hilbert_frequency")
            ]
            cols.extend(hilbert_cols)
        elif feature_type == "wavelet":
            # Wavelet 特征列
            wavelet_cols = [
                col for col in df.columns if col.startswith(f"{source}_wpt")
            ]
            cols.extend(wavelet_cols)
        elif feature_type == "spectral":
            # Spectral 特征列
            spectral_cols = [
                col for col in df.columns if col.startswith(f"{source}_spectral")
            ]
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
    max_points: int = 5000,
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
    # 先检测买卖信号（在采样之前，使用完整数据）
    buy_signals, sell_signals = detect_trading_signals(
        df, price_col, feature_cols, feature_type
    )

    # 采样数据以提高性能
    if len(df) > max_points:
        step = len(df) // max_points
        df_plot = df.iloc[::step].copy()
        # 采样后的信号：只保留采样点的信号
        buy_signals_plot = buy_signals.loc[df_plot.index]
        sell_signals_plot = sell_signals.loc[df_plot.index]
    else:
        df_plot = df.copy()
        buy_signals_plot = buy_signals
        sell_signals_plot = sell_signals

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
                line=dict(color="#1f77b4", width=1),
            ),
            row=1,
            col=1,
        )
        # 添加买卖信号
        if buy_signals_plot.any():
            buy_points = df_plot.loc[buy_signals_plot]
            fig.add_trace(
                go.Scatter(
                    x=buy_points.index,
                    y=buy_points[price_col],
                    mode="markers",
                    name="买入信号",
                    marker=dict(symbol="triangle-up", size=10, color="#2ecc71"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        if sell_signals_plot.any():
            sell_points = df_plot.loc[sell_signals_plot]
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index,
                    y=sell_points[price_col],
                    mode="markers",
                    name="卖出信号",
                    marker=dict(symbol="triangle-down", size=10, color="#e74c3c"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        fig.update_layout(title=f"{source_name.upper()} Price", height=400)
        return fig

    # 创建多子图布局
    # 计算垂直间距：对于多行图表，使用较小的间距
    # 最大允许间距是 1/(rows-1)，我们使用更小的值以确保安全
    n_rows = n_features + 1
    max_spacing = 1.0 / (n_rows - 1) if n_rows > 1 else 0.05
    # 使用较小的间距，但至少为 0.01
    vertical_spacing = min(0.02, max(0.01, max_spacing * 0.5))

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=[price_col.upper()]
        + [
            col.replace(f"{source_name}_", "").replace("_", " ").title()
            for col in feature_cols
        ],
        row_heights=[0.4] + [0.6 / n_features] * n_features,
    )

    # 绘制价格
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot[price_col],
            name=price_col.upper(),
            line=dict(color="#1f77b4", width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 标记买卖点
    if buy_signals_plot.any():
        buy_points = df_plot.loc[buy_signals_plot]
        fig.add_trace(
            go.Scatter(
                x=buy_points.index,
                y=buy_points[price_col],
                mode="markers",
                name="买入信号",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color="#2ecc71",
                    line=dict(width=2, color="#27ae60"),
                ),
                showlegend=True,
                hovertemplate="<b>买入信号</b><br>时间: %{x}<br>价格: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if sell_signals_plot.any():
        sell_points = df_plot.loc[sell_signals_plot]
        fig.add_trace(
            go.Scatter(
                x=sell_points.index,
                y=sell_points[price_col],
                mode="markers",
                name="卖出信号",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color="#e74c3c",
                    line=dict(width=2, color="#c0392b"),
                ),
                showlegend=True,
                hovertemplate="<b>卖出信号</b><br>时间: %{x}<br>价格: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 绘制每个特征
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, col in enumerate(feature_cols):
        if col not in df_plot.columns:
            continue

        row = idx + 2
        color = colors[idx % len(colors)]

        # 清理数据：移除 NaN 和 Inf
        y_data = df_plot[col].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

        # 根据特征类型选择不同的图表类型
        if feature_type == "hilbert" and "phase" in col:
            # 相位特征使用角度图
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=y_data,
                    name=col,
                    line=dict(color=color, width=1),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        else:
            # 其他特征使用普通线图
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=y_data,
                    name=col,
                    line=dict(color=color, width=1),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

    # 更新布局
    fig.update_layout(
        title=f"{source_name.upper()} - {feature_type.upper()} Features",
        height=300 * (n_features + 1),
        hovermode="x unified",
        template="plotly_white",
    )

    # 更新 x 轴
    fig.update_xaxes(title_text="Time", row=n_features + 1, col=1)

    return fig


def generate_html_report(
    df: pd.DataFrame,
    output_path: str,
    symbol: str,
    feature_types: List[str] = ["hurst", "hilbert", "wavelet", "spectral"],
    timeframe: str = "5T",
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
    price_col = "close" if "close" in df.columns else df.columns[0]

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
            chart_html = fig.to_html(
                include_plotlyjs="cdn", div_id=f"{feature_type}_{source_name}"
            )
            charts_html.append(
                {
                    "type": feature_type,
                    "source": source_name,
                    "html": chart_html,
                    "feature_count": len(cols),
                }
            )

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
        .guide-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .guide-box h2 {{
            color: white;
            margin-top: 0;
            border: none;
            background: none;
            padding: 0;
        }}
        .guide-box ul {{
            margin: 15px 0;
            padding-left: 25px;
        }}
        .guide-box li {{
            margin: 8px 0;
            line-height: 1.6;
        }}
        .signal-legend {{
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .signal-legend h4 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .signal-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        .signal-marker {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            display: inline-block;
        }}
        .signal-buy {{
            background-color: #2ecc71;
            clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
        }}
        .signal-sell {{
            background-color: #e74c3c;
            clip-path: polygon(50% 100%, 0% 0%, 100% 0%);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Feature Indicators Visualization</h1>
        
        <div class="guide-box">
            <h2>📖 使用指南</h2>
            <h3>如何阅读这些图表：</h3>
            <ul>
                <li><strong>价格图表（最上方）</strong>：显示价格走势，绿色▲标记买入信号，红色▼标记卖出信号</li>
                <li><strong>Hurst指数</strong>：衡量时间序列的长期记忆性
                    <ul>
                        <li>Hurst &lt; 0.5：反持续性，可能反转</li>
                        <li>Hurst = 0.5：随机游走</li>
                        <li>Hurst &gt; 0.5：持续性，趋势延续</li>
                    </ul>
                </li>
                <li><strong>Hilbert相位</strong>：表示信号的相位变化，相位转折点可能表示趋势变化</li>
                <li><strong>Wavelet小波</strong>：不同频段的能量分解，高频表示短期波动，低频表示长期趋势</li>
                <li><strong>Spectral频谱</strong>：频域特征，主频率变化可能表示市场状态变化</li>
            </ul>
            <h3>买卖信号说明：</h3>
            <ul>
                <li><strong>买入信号（绿色▲）</strong>：基于特征分析，当多个指标同时显示看涨信号时标记</li>
                <li><strong>卖出信号（红色▼）</strong>：基于特征分析，当多个指标同时显示看跌信号时标记</li>
                <li><strong>注意</strong>：这些信号仅供参考，实际交易需要结合其他因素（风险管理、市场环境等）</li>
            </ul>
            <h3>操作提示：</h3>
            <ul>
                <li>鼠标悬停在图表上可查看详细数值</li>
                <li>使用图表工具栏可以缩放、平移、下载图片</li>
                <li>建议结合多个特征类型和信号源综合判断</li>
            </ul>
        </div>
        
        <div class="signal-legend">
            <h4>📊 信号图例</h4>
            <div class="signal-item">
                <span class="signal-marker signal-buy"></span>
                <span><strong>买入信号</strong>：当特征指标显示看涨信号时，在价格图上标记绿色向上三角形</span>
            </div>
            <div class="signal-item">
                <span class="signal-marker signal-sell"></span>
                <span><strong>卖出信号</strong>：当特征指标显示看跌信号时，在价格图上标记红色向下三角形</span>
            </div>
        </div>
        
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

    html_content += (
        """
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            <p>Generated by ML Trading Bot Feature Indicator Visualizer</p>
            <p>Generated at: """
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + """</p>
        </div>
    </div>
</body>
</html>
"""
    )

    # 保存 HTML 文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hurst, hilbert, wavelet, spectral features as indicators in HTML"
    )
    parser.add_argument(
        "--data-path",
        default="/workspace/data/parquet_data",
        help="Path to parquet data directory",
    )
    parser.add_argument(
        "--symbol", required=True, help="Symbol name (e.g., BTCUSDT, ETHUSDT)"
    )
    parser.add_argument(
        "--timeframe",
        default="5T",
        help="Timeframe for data resampling (e.g., 5T, 15T, 60T, 240T)",
    )
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--feature-types",
        default="hurst,hilbert,wavelet,spectral",
        help="Comma-separated list of feature types to visualize",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML file path (default: results/feature_indicators/{symbol}_{timeframe}.html)",
    )
    parser.add_argument(
        "--feature-type",
        default="comprehensive",
        help="Feature engineering type (default: comprehensive)",
    )

    args = parser.parse_args()

    # 解析特征类型
    feature_types = [ft.strip() for ft in args.feature_types.split(",")]

    # 加载数据
    print(f"📊 Loading data for {args.symbol}...")
    loader = MarketDataLoader(args.data_path)
    df = loader.load_data(
        symbol=args.symbol, start_date=args.start_date, end_date=args.end_date
    )

    if df is None or df.empty:
        print(f"❌ No data found for {args.symbol}")
        return

    # 重采样数据
    if hasattr(loader, "resample_data"):
        df = loader.resample_data(args.timeframe)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = (
            df.resample(args.timeframe)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

    print(f"✅ Loaded {len(df)} data points")

    # 特征工程
    print(f"🔧 Engineering features ({args.feature_type})...")
    engineer = ComprehensiveFeatureEngineer(feature_types=args.feature_type)
    df_features = engineer.engineer_all_features(df, fit=True)

    print(f"✅ Generated {len(df_features.columns)} features")

    # 生成输出路径
    if args.output is None:
        symbol_slug = args.symbol.replace("-", "_").replace("/", "_")
        timeframe_slug = args.timeframe.replace("T", "min")
        output_path = f"results/feature_indicators/{symbol_slug}_{timeframe_slug}.html"
    else:
        output_path = args.output

    # 生成 HTML 报告
    generate_html_report(
        df_features, output_path, args.symbol, feature_types, args.timeframe
    )

    print(f"🎉 Visualization complete!")


if __name__ == "__main__":
    main()
