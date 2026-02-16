#!/usr/bin/env python3
"""
Execution Layer Grid Search — 执行参数网格搜索

从 execution.yaml 的 optimization 段读取参数范围，
网格搜索止损/移动止损参数组合，输出最优参数 + Sharpe Heatmap。

自动行为:
    - Gate 过滤: 自动检测 gate_decision 列
    - Entry Filter: 自动读取 entry_filters.yaml OR 组合

用法:
    python scripts/optimize_execution_grid.py \\
        --logs results/train_final_xxx/bpc/predictions.parquet \\
        --strategy bpc

    # 跳过 entry filter
    python scripts/optimize_execution_grid.py \\
        --logs results/train_final_xxx/bpc/predictions.parquet \\
        --strategy bpc --no-entry-filter

输出:
    - JSON 结果 (execution_grid_search.json)
    - HTML Sharpe Heatmap (execution_grid_search.html)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.execution.entry_filter import (
    apply_entry_filters_or,
    load_entry_filters_config,
)
from scripts.backtest_execution_layer import (
    compute_sharpe,
    load_execution_config,
    simulate_rr_execution,
    _estimate_span_years,
    _parse_optimization_grid,
    run_grid_search,
    _identify_plateau,
    _generate_grid_search_html,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Execution Layer Grid Search (optimize stop_loss / trailing params)"
    )
    p.add_argument(
        "--logs",
        required=True,
        help="Input logs file (predictions.parquet or logs_gated.parquet)",
    )
    p.add_argument("--strategy", required=True, help="Strategy name (e.g., bpc)")
    p.add_argument("--strategies-root", default="config/strategies")
    p.add_argument("--features-store-root", default="feature_store")
    p.add_argument(
        "--features-store-layer",
        default=None,
        help="FeatureStore layer (auto-detect from strategy if omitted)",
    )
    p.add_argument("--timeframe", default="240T")
    p.add_argument(
        "--no-entry-filter",
        action="store_true",
        help="Disable automatic entry filter (skip entry_filters.yaml)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output path for results (JSON + HTML).",
    )
    args = p.parse_args()

    # Auto-detect feature store layer if not specified
    if not args.features_store_layer:
        from src.feature_store.layer_naming import detect_layer_for_strategy

        detected = detect_layer_for_strategy(
            strategy=args.strategy,
            features_store_root=args.features_store_root,
        )
        if detected:
            args.features_store_layer = detected
            print(f"ℹ️ Auto-detected feature store layer: {detected}")

    print("=" * 80)
    print("🔍 Execution Layer Grid Search")
    print("=" * 80)

    # 加载 execution.yaml 配置
    try:
        exec_config = load_execution_config(args.strategy, args.strategies_root)
        print(f"\n📋 Loaded execution.yaml for '{args.strategy}'")
    except Exception as e:
        print(f"❌ Failed to load execution.yaml: {e}")
        return 1

    # 检查 optimization section
    opt_cfg = exec_config.get("optimization", {})
    if not opt_cfg.get("enabled"):
        print("❌ optimization.enabled=false in execution.yaml")
        return 1

    # 读取 logs 文件
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ Logs file not found: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"\n📂 Loaded logs: {len(df)} rows")

    # 列名兼容
    if "_symbol" in df.columns and "symbol" not in df.columns:
        df["symbol"] = df["_symbol"]

    # 创建 entry_direction 列
    # 按优先级检测方向列：entry_direction > archetype_breakout_direction > bpc_breakout_direction
    if "entry_direction" not in df.columns:
        direction_col = None
        # 检测 archetype 特定的方向列
        archetype_dir_col = f"{args.strategy}_breakout_direction"
        if archetype_dir_col in df.columns:
            direction_col = archetype_dir_col
        elif "bpc_breakout_direction" in df.columns:
            direction_col = "bpc_breakout_direction"

        if direction_col:
            df["entry_direction"] = df[direction_col].astype(float).copy()
            print(f"   📍 Using direction from: {direction_col}")
        else:
            df["entry_direction"] = 0.0
            print("   ⚠️  No direction column found, using 0.0")
    else:
        print(f"   📍 Using existing entry_direction column")

    # Gate 过滤（自动检测）
    if "gate_decision" in df.columns:
        veto_mask = df["gate_decision"] != "allow"
        n_allowed = int((~veto_mask).sum())
        df.loc[veto_mask, "entry_direction"] = 0.0
        print(f"   🚪 Gate filter (auto): {n_allowed} allow / {len(df)} total")
    elif "gate_ok" in df.columns:
        veto_mask = df["gate_ok"] != True
        n_allowed = int((~veto_mask).sum())
        df.loc[veto_mask, "entry_direction"] = 0.0
        print(f"   🚪 Gate filter (auto): {n_allowed} allow / {len(df)} total")

    n_entries = int((df["entry_direction"] != 0).sum())
    if n_entries == 0:
        print("❌ No entry signals")
        return 1
    print(f"   Entry signals: {n_entries} / {len(df)} bars")

    # 检查 OHLC
    symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
    if not symbols:
        print("❌ No symbols found")
        return 1

    has_ohlc = all(c in df.columns for c in ["high", "low", "close", "atr"])
    if has_ohlc:
        merged = df.copy()
        merged = merged.sort_values(["symbol"]).reset_index(drop=True)
        print(f"\n🔄 Using OHLC from logs: {len(merged)} continuous bars")
    else:
        # 从 FeatureStore 获取 OHLC — 与 backtest 相同逻辑
        if not args.features_store_layer:
            print("❌ Logs don't have OHLC and no FeatureStore layer detected.")
            return 1

        from src.feature_store import FeatureStore, FeatureStoreSpec

        print(f"\n📂 Loading continuous OHLC from FeatureStore...")
        store = FeatureStore(args.features_store_root)
        parts = []
        for sym in symbols:
            spec = FeatureStoreSpec(
                layer=args.features_store_layer, symbol=sym, timeframe=args.timeframe
            )
            try:
                df_sym = store.read_range(
                    spec,
                    start=pd.Timestamp("1970-01-01"),
                    end=pd.Timestamp("2100-01-01"),
                )
                if not df_sym.empty:
                    df_sym = df_sym.copy()
                    if "symbol" not in df_sym.columns:
                        df_sym["symbol"] = sym
                    if df_sym.index.name == "timestamp":
                        df_sym = df_sym.reset_index()
                    elif isinstance(df_sym.index, pd.DatetimeIndex):
                        df_sym["timestamp"] = df_sym.index
                        df_sym = df_sym.reset_index(drop=True)
                    parts.append(df_sym)
            except Exception as e:
                print(f"   ⚠️  Failed to read {sym}: {e}")

        if not parts:
            print("❌ No FeatureStore data loaded")
            return 1

        merged = pd.concat(parts, axis=0, ignore_index=True)
        merged["symbol"] = merged["symbol"].astype(str)

        if "bpc_breakout_direction" in merged.columns:
            merged["entry_direction"] = merged["bpc_breakout_direction"].astype(float)
        else:
            print("❌ No bpc_breakout_direction in FeatureStore")
            return 1

        if "timestamp" in merged.columns:
            merged = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        n_entries = int((merged["entry_direction"] != 0).sum())
        print(f"   Loaded FeatureStore: {len(merged)} bars, {n_entries} entries")

    # Entry Filter (自动 OR)
    if not args.no_entry_filter:
        entry_filters_cfg = load_entry_filters_config(
            args.strategy, args.strategies_root
        )
        if entry_filters_cfg:
            n_entries = apply_entry_filters_or(merged, entry_filters_cfg)
            if n_entries == 0:
                print("❌ No entry signals after entry filter")
                return 1
        else:
            print("   ℹ️  entry_filters.yaml not found, skipping entry filter")
    else:
        print("   ℹ️  Entry filter disabled (--no-entry-filter)")

    # ================================================================
    # Grid Search
    # ================================================================
    param_names, param_values = _parse_optimization_grid(opt_cfg)
    total_combos = 1
    for pv in param_values:
        total_combos *= len(pv)

    print(f"\n🔍 Grid Search")
    print(f"   Parameters: {len(param_names)}")
    for i, (n, v) in enumerate(zip(param_names, param_values)):
        print(f"   [{i+1}] {n}: {v}")
    print(f"   Total combinations: {total_combos}")
    print()

    # 运行网格搜索
    span_years = _estimate_span_years(merged)
    print("📈 Running grid search...")
    results = run_grid_search(
        merged,
        exec_config,
        param_names,
        param_values,
        atr_col="atr",
        span_years=span_years,
    )

    # 平坦高原分析
    plateau = _identify_plateau(results)
    best = plateau["best"]

    # 打印摘要
    print("\n" + "=" * 80)
    print("📊 GRID SEARCH RESULTS")
    print("=" * 80)
    print(
        f"\n   🏆 Best Sharpe: {best['sharpe']:.4f}  (annualized: {best['sharpe_ann']:.1f})"
    )
    short_names = [n.split(".")[-1] for n in param_names]
    print(
        f"   Best params: {', '.join(f'{short_names[i]}={best[param_names[i]]:.1f}' for i in range(len(param_names)))}"
    )
    print(
        f"   Trades: {best['trades']}  |  Mean R: {best['mean_r']:.4f}  |  Win Rate: {best['win_rate']:.1%}"
    )
    print(f"\n   Plateau: {'✅ 稳定' if plateau['is_plateau'] else '⚠️ 不稳定'}")
    print(
        f"   Top-{plateau['top_n']} mean={plateau['mean_sharpe']:.4f}  std={plateau['std_sharpe']:.4f}  CV={plateau['cv']:.3f}"
    )

    # 输出文件
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.logs).parent / "execution_grid_search.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON 结果
    json_data = {
        "strategy": args.strategy,
        "param_names": param_names,
        "param_values": [[float(v) for v in pv] for pv in param_values],
        "total_combinations": total_combos,
        "best": {
            k: (
                float(v)
                if isinstance(v, (np.floating, np.integer))
                else bool(v) if isinstance(v, np.bool_) else v
            )
            for k, v in best.items()
        },
        "plateau": {
            "is_plateau": bool(plateau["is_plateau"]),
            "mean_sharpe": float(plateau["mean_sharpe"]),
            "std_sharpe": float(plateau["std_sharpe"]),
            "cv": float(plateau["cv"]),
        },
        "results": [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, np.integer))
                    else bool(v) if isinstance(v, np.bool_) else v
                )
                for k, v in r.items()
            }
            for r in results
        ],
    }
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n   📄 JSON: {json_path}")

    # HTML 报告
    html = _generate_grid_search_html(
        results=results,
        param_names=param_names,
        param_values=param_values,
        plateau=plateau,
        exec_config=exec_config,
        strategy=args.strategy,
        n_trades_total=n_entries,
    )
    html_path = output_path.with_suffix(".html")
    Path(html_path).write_text(html, encoding="utf-8")
    print(f"   📊 HTML: {html_path}")
    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
