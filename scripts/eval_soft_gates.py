#!/usr/bin/env python3
"""
Soft Gate 开仓质量评估脚本

对 gate.yaml 中每条 soft_filter 规则，比较触发 vs 未触发 bar 的开仓质量。
如果触发的 bar 质量明显更差 → 说明 soft gate 有效，值得迁移到 evidence 控制仓位。

用法:
    # 默认使用 deep_pullback_cvd 基线
    python scripts/eval_soft_gates.py --logs results/*/bpc/predictions.parquet

    # 指定 entry filter
    python scripts/eval_soft_gates.py --logs results/*/bpc/predictions.parquet --entry-filter deep_pullback_full

    # 从 FeatureStore 补全缺失特征 (EVT/VPIN 等)
    python scripts/eval_soft_gates.py --logs results/*/bpc/predictions.parquet --features-store-layer features_00a7951c63
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.execution.entry_filter import (
    apply_entry_filter,
    load_entry_filters_config,
)
from scripts.backtest_execution_layer import (
    compute_sharpe,
    load_execution_config,
    load_gate_config,
    simulate_rr_execution,
    _estimate_span_years,
)


# ================================================================
# FeatureStore 补全缺失特征
# ================================================================


def _load_missing_features_from_store(
    df: pd.DataFrame,
    missing_features: List[str],
    features_store_root: str,
    features_store_layer: str,
    timeframe: str = "240T",
) -> pd.DataFrame:
    """从 FeatureStore 加载缺失特征，通过 (symbol, close) 对齐 merge 到 df。"""
    from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec

    store = FeatureStore(features_store_root)
    sym_col = "symbol" if "symbol" in df.columns else "_symbol"
    symbols = df[sym_col].unique().tolist()

    # 按 symbol 加载 FeatureStore 数据
    fs_parts: List[pd.DataFrame] = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=features_store_layer, symbol=sym, timeframe=timeframe
        )
        try:
            df_sym = store.read_range(
                spec,
                start=pd.Timestamp("1970-01-01"),
                end=pd.Timestamp("2100-01-01"),
            )
            if df_sym.empty:
                continue
            df_sym = df_sym.copy()
            if df_sym.index.name == "timestamp":
                df_sym = df_sym.reset_index()
            # 只保留 close + 需要的特征列
            keep_cols = ["close"] + [f for f in missing_features if f in df_sym.columns]
            if len(keep_cols) <= 1:
                continue
            df_sym = df_sym[keep_cols].copy()
            df_sym[sym_col] = sym
            fs_parts.append(df_sym)
        except Exception as e:
            print(f"   ⚠️  FeatureStore read failed for {sym}: {e}")

    if not fs_parts:
        return df

    fs_all = pd.concat(fs_parts, ignore_index=True)
    loaded_feats = [c for c in fs_all.columns if c in missing_features]
    if not loaded_feats:
        return df

    # 对齐: 用 (symbol, close_rounded) 做 merge key
    df["_merge_key"] = df["close"].round(8).astype(str) + "|" + df[sym_col].astype(str)
    fs_all["_merge_key"] = (
        fs_all["close"].round(8).astype(str) + "|" + fs_all[sym_col].astype(str)
    )

    # 去重 (FeatureStore 可能有重复 close)
    fs_dedup = fs_all.drop_duplicates(subset=["_merge_key"], keep="last")
    merge_cols = ["_merge_key"] + loaded_feats
    merged = df.merge(fs_dedup[merge_cols], on="_merge_key", how="left")
    merged.drop(columns=["_merge_key"], inplace=True)

    matched = merged[loaded_feats[0]].notna().sum()
    print(
        f"   📦 FeatureStore merge: {matched}/{len(df)} rows matched, loaded {loaded_feats}"
    )

    return merged


# ================================================================
# Gate YAML 解析
# ================================================================


def _parse_soft_filters(gate_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 gate.yaml 解析 soft_filters 规则列表。

    Returns:
        [{"id": ..., "feature": ..., "op": ">"|"<", "threshold": float, "weight": float}, ...]
    """
    rules: List[Dict[str, Any]] = []
    for sf in gate_cfg.get("soft_filters", []):
        sf_id = sf.get("id", "?")
        weight = sf.get("then", {}).get("weight", 1.0)
        reason = sf.get("reason", "")
        when = sf.get("when", {})

        # when 格式: {feature: {value_gt: x}} 或 {feature: {value_lt: x}}
        for feat_name, cond in when.items():
            if not isinstance(cond, dict):
                continue
            if "value_gt" in cond:
                rules.append(
                    {
                        "id": sf_id,
                        "feature": feat_name,
                        "op": ">",
                        "threshold": float(cond["value_gt"]),
                        "weight": float(weight),
                        "reason": reason,
                    }
                )
            elif "value_lt" in cond:
                rules.append(
                    {
                        "id": sf_id,
                        "feature": feat_name,
                        "op": "<",
                        "threshold": float(cond["value_lt"]),
                        "weight": float(weight),
                        "reason": reason,
                    }
                )
            elif "quantile_gt" in cond:
                rules.append(
                    {
                        "id": sf_id,
                        "feature": feat_name,
                        "op": "quantile_gt",
                        "threshold": float(cond["quantile_gt"]),
                        "weight": float(weight),
                        "reason": reason,
                    }
                )
            elif "quantile_lt" in cond:
                rules.append(
                    {
                        "id": sf_id,
                        "feature": feat_name,
                        "op": "quantile_lt",
                        "threshold": float(cond["quantile_lt"]),
                        "weight": float(weight),
                        "reason": reason,
                    }
                )
    return rules


# ================================================================
# 质量评估核心
# ================================================================


def _split_by_rule(
    df: pd.DataFrame,
    feature: str,
    op: str,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按规则条件分割 DataFrame 为 triggered / not_triggered。"""
    if feature not in df.columns:
        return pd.DataFrame(), df

    col = df[feature].astype(float)
    if op == ">":
        mask = col > threshold
    elif op == "<":
        mask = col < threshold
    elif op == "quantile_gt":
        q_val = col.quantile(threshold)
        mask = col > q_val
    elif op == "quantile_lt":
        q_val = col.quantile(threshold)
        mask = col < q_val
    else:
        return pd.DataFrame(), df

    return df[mask].copy(), df[~mask].copy()


def _quality_stats(rr: pd.Series) -> Dict[str, float]:
    """计算开仓质量指标。"""
    n = len(rr)
    if n == 0:
        return {"n": 0, "mean_r": 0.0, "win_rate": 0.0, "sharpe": 0.0, "median_r": 0.0}
    return {
        "n": n,
        "mean_r": float(rr.mean()),
        "win_rate": float((rr > 0).mean()),
        "sharpe": float(rr.mean() / rr.std()) if rr.std() > 1e-8 else 0.0,
        "median_r": float(rr.median()),
    }


# ================================================================
# 主评估逻辑
# ================================================================


def evaluate_soft_gates(
    df: pd.DataFrame,
    exec_config: Dict[str, Any],
    gate_cfg: Dict[str, Any],
    entry_filter: str = "deep_pullback_cvd",
    strategy: str = "bpc",
    strategies_root: str = "config/strategies",
    min_trades: int = 10,
) -> List[Dict[str, Any]]:
    """评估所有 soft gate 规则的开仓质量。

    Returns:
        每条规则的评估结果列表
    """
    # 准备 entry_direction
    if "entry_direction" not in df.columns:
        if "bpc_breakout_direction" in df.columns:
            df["entry_direction"] = df["bpc_breakout_direction"].astype(float)
        else:
            raise ValueError("No 'entry_direction' or 'bpc_breakout_direction' column")

    # 应用 entry filter
    if entry_filter != "none":
        entry_cfg = load_entry_filters_config(strategy, strategies_root)
        if entry_cfg:
            n_before = int((df["entry_direction"] != 0).sum())
            apply_entry_filter(df, entry_filter, entry_cfg=entry_cfg, silent=True)
            n_after = int((df["entry_direction"] != 0).sum())
            print(f"   Entry filter '{entry_filter}': {n_before} → {n_after} signals")

    # 运行模拟获取 R/R
    df_sorted = df.sort_values(["symbol"]).reset_index(drop=True)
    returns = simulate_rr_execution(df_sorted, exec_config, atr_col="atr", silent=True)
    df_sorted["_rr"] = returns.values
    valid = df_sorted[df_sorted["_rr"].notna()].copy()

    # 基线统计
    baseline = _quality_stats(valid["_rr"])
    span_years = _estimate_span_years(df_sorted)
    baseline_sharpe_ann = compute_sharpe(
        valid["_rr"], annualize=True, span_years=span_years
    )

    print(f"\n{'=' * 90}")
    print(f"  Baseline ({entry_filter}): {baseline['n']} trades")
    print(
        f"  Mean R = {baseline['mean_r']:.2f}  |  Win = {baseline['win_rate']:.1%}  "
        f"|  Sharpe = {baseline['sharpe']:.4f}  |  Annualized = {baseline_sharpe_ann:.2f}"
    )
    print(f"{'=' * 90}")

    # 解析 soft_filters
    rules = _parse_soft_filters(gate_cfg)
    if not rules:
        print("\n  ⚠️  gate.yaml 中没有 soft_filters 规则")
        return []

    # 逐条评估
    results: List[Dict[str, Any]] = []
    for rule in rules:
        triggered, not_triggered = _split_by_rule(
            valid, rule["feature"], rule["op"], rule["threshold"]
        )
        t_stats = (
            _quality_stats(triggered["_rr"])
            if len(triggered) > 0
            else _quality_stats(pd.Series(dtype=float))
        )
        nt_stats = (
            _quality_stats(not_triggered["_rr"])
            if len(not_triggered) > 0
            else _quality_stats(pd.Series(dtype=float))
        )

        enough_data = t_stats["n"] >= min_trades and nt_stats["n"] >= min_trades
        diff_mean = t_stats["mean_r"] - nt_stats["mean_r"] if enough_data else None
        diff_win = t_stats["win_rate"] - nt_stats["win_rate"] if enough_data else None

        # 判断效果
        if not enough_data:
            verdict = "NO_DATA"
        elif diff_mean is not None and diff_mean < -2.0:
            verdict = "EFFECTIVE"
        elif diff_mean is not None and diff_mean < 0:
            verdict = "WEAK"
        else:
            verdict = "INEFFECTIVE"

        results.append(
            {
                "rule": rule,
                "triggered": t_stats,
                "not_triggered": nt_stats,
                "diff_mean_r": diff_mean,
                "diff_win_rate": diff_win,
                "verdict": verdict,
            }
        )

    return results


# ================================================================
# 输出格式化
# ================================================================

VERDICT_EMOJI = {
    "EFFECTIVE": "✅",
    "WEAK": "➖",
    "INEFFECTIVE": "❌",
    "NO_DATA": "⚪",
}

VERDICT_DESC = {
    "EFFECTIVE": "有效 — 触发位置质量明显更差，值得迁移到 evidence",
    "WEAK": "微弱 — 触发位置略差，可能有一定价值",
    "INEFFECTIVE": "无效 — 触发位置质量不比未触发差，无降权必要",
    "NO_DATA": "数据不足 — 无法判断（特征缺失或触发数太少）",
}


def print_results(results: List[Dict[str, Any]]) -> None:
    """打印评估结果表格。"""
    if not results:
        return

    # 表头
    op_label = lambda r: f"{r['feature']} {r['op']} {r['threshold']}"
    max_rule_len = max(len(op_label(r["rule"])) for r in results)
    max_rule_len = max(max_rule_len, 28)

    header = (
        f"  {'Rule':<{max_rule_len}}  {'Wt':>4} │ "
        f"{'触发 N':>7} {'Mean R':>8} {'Win%':>6} {'Sharpe':>7} │ "
        f"{'未触发 N':>8} {'Mean R':>8} {'Win%':>6} {'Sharpe':>7} │ "
        f"{'ΔMean':>7} {'ΔWin':>6}  {'结论':>4}"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for res in results:
        r = res["rule"]
        t = res["triggered"]
        nt = res["not_triggered"]
        verdict = res["verdict"]
        emoji = VERDICT_EMOJI.get(verdict, "?")
        label = op_label(r)

        if verdict == "NO_DATA":
            if t["n"] == 0 and nt["n"] == 0:
                detail = "特征不在 predictions 中"
            elif t["n"] < 10:
                detail = f"触发仅 {t['n']} 笔 (< 10)"
            else:
                detail = f"未触发仅 {nt['n']} 笔 (< 10)"
            print(
                f"  {emoji} {label:<{max_rule_len - 2}}  {r['weight']:>4.1f} │ {detail}"
            )
            continue

        diff_m = res["diff_mean_r"]
        diff_w = res["diff_win_rate"]
        print(
            f"  {emoji} {label:<{max_rule_len - 2}}  {r['weight']:>4.1f} │ "
            f"{t['n']:>7} {t['mean_r']:>8.2f} {t['win_rate']:>5.1%} {t['sharpe']:>7.4f} │ "
            f"{nt['n']:>8} {nt['mean_r']:>8.2f} {nt['win_rate']:>5.1%} {nt['sharpe']:>7.4f} │ "
            f"{diff_m:>+7.2f} {diff_w:>+5.1%}  {verdict}"
        )

    print(sep)

    # 汇总建议
    effective = [r for r in results if r["verdict"] == "EFFECTIVE"]
    weak = [r for r in results if r["verdict"] == "WEAK"]
    ineffective = [r for r in results if r["verdict"] == "INEFFECTIVE"]
    no_data = [r for r in results if r["verdict"] == "NO_DATA"]

    print(f"\n📊 汇总:")
    print(
        f"   ✅ EFFECTIVE ({len(effective)}): 触发位置 Mean R 差 > 2.0，建议迁移到 evidence"
    )
    for r in effective:
        rule = r["rule"]
        print(
            f"      - {rule['feature']} (weight={rule['weight']}, ΔMean R = {r['diff_mean_r']:+.2f})"
        )
    print(f"   ➖ WEAK ({len(weak)}): 触发位置略差，可选择性保留")
    for r in weak:
        rule = r["rule"]
        print(
            f"      - {rule['feature']} (weight={rule['weight']}, ΔMean R = {r['diff_mean_r']:+.2f})"
        )
    print(f"   ❌ INEFFECTIVE ({len(ineffective)}): 无降权必要，建议移除")
    for r in ineffective:
        rule = r["rule"]
        print(f"      - {rule['feature']}")
    print(f"   ⚪ NO_DATA ({len(no_data)}): 无法判断")
    for r in no_data:
        rule = r["rule"]
        print(f"      - {rule['feature']}")

    print()
    print("判定标准:")
    for k, v in VERDICT_DESC.items():
        print(f"   {VERDICT_EMOJI[k]} {k}: {v}")


# ================================================================
# CLI
# ================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="评估 gate.yaml 中 soft_filter 规则的开仓质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 默认基线 (deep_pullback_cvd)
    python scripts/eval_soft_gates.py --logs results/*/bpc/predictions.parquet

    # 使用 deep_pullback_full 基线
    python scripts/eval_soft_gates.py --logs results/*/bpc/predictions.parquet --entry-filter deep_pullback_full

    # 无 entry filter (原始信号)
    python scripts/eval_soft_gates.py --logs results/*/bpc/predictions.parquet --entry-filter none
""",
    )
    ap.add_argument("--logs", required=True, help="predictions.parquet 路径")
    ap.add_argument("--strategy", default="bpc", help="策略名称 (default: bpc)")
    ap.add_argument(
        "--strategies-root",
        default="config/strategies",
        help="策略配置根目录 (default: config/strategies)",
    )
    ap.add_argument(
        "--features-store-root",
        default="feature_store",
        help="FeatureStore 根目录 (default: feature_store)",
    )
    ap.add_argument(
        "--features-store-layer",
        default=None,
        help="FeatureStore layer 用于补全缺失特征 (e.g. features_00a7951c63)",
    )
    ap.add_argument("--timeframe", default="240T", help="Timeframe (default: 240T)")
    ap.add_argument(
        "--entry-filter",
        default="deep_pullback_cvd",
        help="入场过滤器 (default: deep_pullback_cvd)，'none' 表示不过滤",
    )
    ap.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="每组最少交易数，不足视为 NO_DATA (default: 10)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # 加载数据
    logs_path = Path(args.logs)
    if not logs_path.exists():
        # 尝试 glob 展开
        import glob

        candidates = sorted(glob.glob(args.logs))
        if not candidates:
            print(f"❌ 找不到文件: {args.logs}")
            return 1
        logs_path = Path(candidates[0])

    print(f"📊 读取: {logs_path}")
    df = pd.read_parquet(logs_path)
    if "_symbol" in df.columns and "symbol" not in df.columns:
        df["symbol"] = df["_symbol"]
    print(f"   {len(df)} 行, {df['symbol'].nunique()} symbols")

    # 加载配置
    exec_config = load_execution_config(args.strategy, args.strategies_root)
    gate_cfg = load_gate_config(args.strategy, args.strategies_root)

    if not gate_cfg.get("soft_filters"):
        print("⚠️  gate.yaml 中没有 soft_filters 规则")
        return 0

    n_rules = len(gate_cfg["soft_filters"])
    print(f"   gate.yaml: {n_rules} 条 soft_filter 规则")

    # 检查缺失特征，从 FeatureStore 补全
    rules = _parse_soft_filters(gate_cfg)
    needed_features = {r["feature"] for r in rules}
    missing = [f for f in needed_features if f not in df.columns]
    if missing:
        print(f"\n   ⚠️  缺失 {len(missing)} 个特征: {missing}")
        fs_layer = args.features_store_layer
        if not fs_layer:
            # 尝试自动检测
            try:
                from src.feature_store.layer_naming import detect_layer_for_strategy

                fs_layer = detect_layer_for_strategy(
                    args.strategy, args.features_store_root
                )
            except Exception:
                pass
        if fs_layer:
            print(f"   📦 从 FeatureStore 补全 (layer={fs_layer})...")
            df = _load_missing_features_from_store(
                df, missing, args.features_store_root, fs_layer, args.timeframe
            )
            still_missing = [
                f for f in missing if f not in df.columns or df[f].isna().all()
            ]
            if still_missing:
                print(f"   ⚠️  仍缺失: {still_missing}")
        else:
            print(
                "   💡 提示: 使用 --features-store-layer 指定 FeatureStore layer 以补全"
            )

    # 评估
    results = evaluate_soft_gates(
        df,
        exec_config,
        gate_cfg,
        entry_filter=args.entry_filter,
        strategy=args.strategy,
        strategies_root=args.strategies_root,
        min_trades=args.min_trades,
    )

    # 输出
    print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
