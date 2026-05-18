#!/usr/bin/env python
"""
从回测交易记录统计各 archetype 的历史 Edge (平均 R-multiple)

用于实盘信号优先级路由：
    AOS = Edge_archetype × Evidence_score

Usage:
    python scripts/compute_archetype_edges.py \
      --trades results/train_final_xxx/bpc/execution_backtest_trades.parquet \
      --lookback-months 3 \
      --output config/strategies/bad-candidates/bpc/archetypes/archetype_edges.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict

import pandas as pd
import yaml


def compute_edges_from_parquet(
    trades_path: str,
    lookback_months: int = 3,
) -> Dict[str, float]:
    """
    从 parquet 交易记录统计 archetype edges

    Args:
        trades_path: 交易记录文件路径
        lookback_months: 回看周期（月）

    Returns:
        各 archetype 的平均 R-multiple
    """
    # 读取交易记录
    df = pd.read_parquet(trades_path)

    # 检查必需列
    required_cols = ["archetype", "r_multiple", "exit_timestamp"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # 尝试备用列名
        if "closed_at" in df.columns:
            df["exit_timestamp"] = df["closed_at"]
        else:
            raise ValueError(f"缺少必需列: {missing}")

    # 时间过滤
    if lookback_months > 0:
        now = datetime.utcnow()
        cutoff = now - timedelta(days=lookback_months * 30)
        df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"])
        df = df[df["exit_timestamp"] >= cutoff]

    # 按 archetype 统计
    archetype_r_multiples = defaultdict(list)
    for _, row in df.iterrows():
        archetype = row.get("archetype")
        r_multiple = row.get("r_multiple")
        if pd.notna(archetype) and pd.notna(r_multiple):
            archetype_r_multiples[archetype].append(float(r_multiple))

    # 计算平均值
    edges = {}
    for archetype, r_multiples in archetype_r_multiples.items():
        if r_multiples:
            edges[archetype] = round(sum(r_multiples) / len(r_multiples), 3)

    return edges


def write_edges_yaml(
    edges: Dict[str, float],
    output_path: str,
    lookback_months: int,
    total_trades: int,
):
    """写入 YAML 配置文件"""

    config = {
        "archetype_edges": edges,
        "router_config": {
            "max_slots": 2,
            "default_edge": 0.50,
        },
        "metadata": {
            "last_update": datetime.utcnow().strftime("%Y-%m-%d"),
            "lookback_months": lookback_months,
            "total_trades_analyzed": total_trades,
            "data_source": "backtest",
            "notes": "从回测交易记录自动生成",
        },
    }

    # 添加注释
    yaml_header = """# Archetype Edge 配置
# 
# 用于 SignalRouter 的 AOS (Archetype Opportunity Score) 计算
# AOS = Edge_archetype × Evidence_score
#
# Edge 定义：该 archetype 最近 N 个月的平均 R-multiple
# 
# 更新频率：建议每月滚动更新
# 更新方式：运行 python scripts/compute_archetype_edges.py

"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_header)
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(
        description="统计 archetype edges 用于实盘信号路由"
    )
    parser.add_argument(
        "--trades",
        required=True,
        help="交易记录文件路径 (parquet)",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=3,
        help="回看周期（月），0 表示全部数据",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 YAML 配置文件路径",
    )

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.trades).exists():
        print(f"❌ 错误：交易记录文件不存在: {args.trades}")
        sys.exit(1)

    print(f"📊 统计 archetype edges...")
    print(f"   输入: {args.trades}")
    print(f"   回看: {args.lookback_months} 个月")

    # 统计
    edges = compute_edges_from_parquet(
        trades_path=args.trades,
        lookback_months=args.lookback_months,
    )

    if not edges:
        print("❌ 错误：未找到有效的 archetype 交易记录")
        sys.exit(1)

    # 读取原始数据获取总交易数
    df = pd.read_parquet(args.trades)
    total_trades = len(df)

    # 输出统计
    print(f"\n✅ 统计完成 ({total_trades} 笔交易):")
    print(f"\n{'Archetype':<20} {'Avg R':<10} {'Count':<10}")
    print("-" * 40)

    # 按 Edge 排序输出
    for arch in sorted(edges.keys(), key=lambda x: edges[x], reverse=True):
        r_count = len([1 for _, row in df.iterrows() if row.get("archetype") == arch])
        print(f"{arch:<20} {edges[arch]:>6.3f}     {r_count:>6}")

    # 写入配置文件
    write_edges_yaml(
        edges=edges,
        output_path=args.output,
        lookback_months=args.lookback_months,
        total_trades=total_trades,
    )

    print(f"\n📝 配置已保存: {args.output}")
    print(f"\n💡 验证配置:")
    print(f'   python -c "')
    print(
        f"from time_series_model.portfolio.signal_router import load_archetype_edges_from_config"
    )
    print(f"edges = load_archetype_edges_from_config('{args.output}')")
    print(f"print(f'✓ Loaded edges: {{edges}}')")
    print(f'   "')


if __name__ == "__main__":
    main()
