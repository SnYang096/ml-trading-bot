#!/usr/bin/env python3
"""
Entry Filter 阈值平坦高原扫描

对每个 entry filter 中的连续阈值条件（>=, >, <=, <），
扫描阈值范围，计算 Sharpe/Trades，找到平坦高原区间。

用法:
    # 扫描所有 filter 的所有连续阈值
    python scripts/optimize_entry_filter_plateau.py \
        --logs results/train_final_*/bpc/predictions.parquet \
        --strategy bpc

    # 只扫描指定 filter
    python scripts/optimize_entry_filter_plateau.py \
        --logs results/train_final_*/bpc/predictions.parquet \
        --strategy bpc \
        --filter deep_pullback_vol

输出:
    - HTML 报告（含 Sharpe vs Threshold 图表）
"""
from __future__ import annotations

import argparse
import copy
import json
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
    get_available_filters,
    load_entry_filters_config,
    _build_mask_from_conditions,
)
from scripts.backtest_execution_layer import (
    compute_sharpe,
    load_execution_config,
    simulate_rr_execution,
    _estimate_span_years,
)

# ================================================================
# 阈值扫描核心逻辑
# ================================================================

# 可扫描的运算符（连续阈值）
_SCANNABLE_OPS = {">=", ">", "<=", "<"}


def _find_scannable_conditions(
    filter_def: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """找出 filter 中可扫描的连续阈值条件。

    排除 == / != 条件（bool 特征），只保留 >=, >, <=, < 条件。
    """
    scannable = []
    for i, cond in enumerate(filter_def.get("conditions", [])):
        if cond.get("operator") in _SCANNABLE_OPS:
            scannable.append({"index": i, **cond})
    return scannable


def _generate_scan_range(
    current_value: float,
    operator: str,
    n_steps: int = 15,
) -> List[float]:
    """根据当前阈值和运算符生成扫描范围。

    对于 >= / > : 从 0.1 到 0.95 扫描（高阈值 = 更严格）
    对于 <= / < : 从 0.05 到 0.9 扫描（低阈值 = 更严格）
    特殊处理 value=0 的情况（如 liquidity_void > 0）
    """
    if operator in (">=", ">"):
        # 高阈值更严格，扫描 [low, high]
        low = max(0.0, current_value - 0.4)
        high = min(1.0, current_value + 0.4)
    else:
        # <=, < : 低阈值更严格
        low = max(0.0, current_value - 0.4)
        high = min(1.0, current_value + 0.4)

    if low == high:
        low = 0.0
        high = 1.0

    step = (high - low) / (n_steps - 1)
    return [round(low + i * step, 3) for i in range(n_steps)]


def _scan_single_threshold(
    merged: pd.DataFrame,
    exec_config: Dict[str, Any],
    filter_def: Dict[str, Any],
    cond_index: int,
    scan_values: List[float],
    entry_filters_cfg: Dict[str, Any],
    span_years: float,
) -> List[Dict[str, Any]]:
    """对单个条件的阈值范围扫描，返回每个阈值的回测结果。"""
    results = []
    original_entry = merged["entry_direction"].copy()
    conditions = filter_def.get("conditions", [])
    original_value = conditions[cond_index]["value"]

    for val in scan_values:
        # 恢复原始信号
        merged["entry_direction"] = original_entry.copy()

        # 临时修改阈值
        conditions[cond_index]["value"] = val

        # 应用 filter（用修改后的 conditions）
        mask = pd.Series(True, index=merged.index)
        for cond in conditions:
            feat = cond["feature"]
            op_str = cond["operator"]
            v = cond["value"]
            if feat not in merged.columns:
                continue
            op_map = {
                ">": lambda s, v: s > v,
                ">=": lambda s, v: s >= v,
                "<": lambda s, v: s < v,
                "<=": lambda s, v: s <= v,
                "==": lambda s, v: s == v,
                "!=": lambda s, v: s != v,
            }
            op_fn = op_map.get(op_str)
            if op_fn:
                mask = mask & op_fn(merged[feat].astype(float), float(v))

        merged.loc[~mask, "entry_direction"] = 0.0
        n_entries = int((merged["entry_direction"] != 0).sum())

        if n_entries < 20:
            results.append(
                {
                    "threshold": val,
                    "trades": n_entries,
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                    "mean_r": 0.0,
                    "too_few": True,
                }
            )
            continue

        exec_returns = simulate_rr_execution(
            merged, exec_config, atr_col="atr", use_tier_params=False
        )
        valid = exec_returns.dropna()
        if len(valid) < 10:
            results.append(
                {
                    "threshold": val,
                    "trades": n_entries,
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                    "mean_r": 0.0,
                    "too_few": True,
                }
            )
            continue

        sh = compute_sharpe(valid, annualize=False)
        results.append(
            {
                "threshold": val,
                "trades": len(valid),
                "sharpe": float(sh),
                "win_rate": float((valid > 0).mean()),
                "mean_r": float(valid.mean()),
                "too_few": False,
            }
        )

    # 恢复原始值
    conditions[cond_index]["value"] = original_value
    merged["entry_direction"] = original_entry
    return results


def _find_plateau(
    results: List[Dict[str, Any]],
    window: int = 5,
    operator: str = ">=",
) -> Dict[str, Any]:
    """分析扫描结果，找到平坦高原区间。

    使用滑动窗口双 CV 判定：
      - Sharpe CV < 0.3（收益稳定性）
      - Trades CV < 0.4（执行节奏稳定性 — Entry Filter 特有）

    recommended 不取中点，取 plateau 偏宽容侧：
      - >= / > 条件：低阈值 = 更宽松 → start + bias * width
      - <= / < 条件：高阈值 = 更宽松 → end - bias * width
      - bias 动态绑定 width: 窄高原(width<0.25) bias=0.2, 宽高原 bias=0.1
        直觉：plateau 很宽 → 本来就不敏感 → 不用偏太多
               plateau 较窄 → 更容易 miss → 偏宽容更重要

    原理：plateau 的存在证明了「严格 ≠ 更好」，
    Entry Filter 的错误成本是「错过」而非「多等一次确认」。
    在 Sharpe 无本质差别的区间内，选更容易触发入场的一侧。

    plateau 排序用 mean_sharpe × plateau_width（鲁棒性最大化），
    而非单纯 mean_sharpe 最大化，让宽而稳的高原优先于窄而高的尖峰。

    输出 plateau_width 作为置信度指标（宽度越大 = 越稳定 = 越可部署）。
    """
    valid = [r for r in results if not r.get("too_few")]
    if len(valid) < window:
        return {
            "is_plateau": False,
            "reason": f"有效点不足 ({len(valid)} < {window})",
        }

    best_plateau = None
    for i in range(len(valid) - window + 1):
        w = valid[i : i + window]
        sharpes = [r["sharpe"] for r in w]
        trades_list = [r["trades"] for r in w]
        mean_sh = np.mean(sharpes)
        std_sh = np.std(sharpes)
        cv_sharpe = std_sh / mean_sh if mean_sh > 1e-8 else 999
        mean_tr = np.mean(trades_list)
        std_tr = np.std(trades_list)
        cv_trades = std_tr / mean_tr if mean_tr > 1e-8 else 999

        if cv_sharpe < 0.3 and cv_trades < 0.4 and mean_sh > 0:
            start_t = w[0]["threshold"]
            end_t = w[-1]["threshold"]
            plateau_width = end_t - start_t
            # 排序: mean_sharpe × plateau_width（鲁棒性最大化）
            robustness = mean_sh * plateau_width
            if best_plateau is None or robustness > best_plateau.get("_robustness", -1):
                # bias 动态绑定 width: 窄高原偏多，宽高原偏少
                bias = 0.2 if plateau_width < 0.25 else 0.1
                if operator in (">=", ">"):
                    rec_val = start_t + bias * plateau_width
                else:
                    rec_val = end_t - bias * plateau_width
                # snap to nearest scanned point
                rec_idx = int(np.argmin([abs(r["threshold"] - rec_val) for r in w]))
                best_plateau = {
                    "is_plateau": True,
                    "_robustness": float(robustness),
                    "start_threshold": start_t,
                    "end_threshold": end_t,
                    "plateau_width": float(plateau_width),
                    "confidence": _width_to_confidence(plateau_width),
                    "mean_sharpe": float(mean_sh),
                    "cv_sharpe": float(cv_sharpe),
                    "cv_trades": float(cv_trades),
                    "mean_trades": float(mean_tr),
                    "recommended": float(w[rec_idx]["threshold"]),
                }

    if best_plateau is None:
        # 尝试只用 Sharpe CV（放宽 Trades CV 约束）
        for i in range(len(valid) - window + 1):
            w = valid[i : i + window]
            sharpes = [r["sharpe"] for r in w]
            mean_sh = np.mean(sharpes)
            cv_sharpe = np.std(sharpes) / mean_sh if mean_sh > 1e-8 else 999
            if cv_sharpe < 0.3 and mean_sh > 0:
                start_t = w[0]["threshold"]
                end_t = w[-1]["threshold"]
                pw = end_t - start_t
                robustness = mean_sh * pw
                if best_plateau is None or robustness > best_plateau.get(
                    "_robustness", -1
                ):
                    trades_list = [r["trades"] for r in w]
                    cv_trades = (
                        float(np.std(trades_list) / np.mean(trades_list))
                        if np.mean(trades_list) > 0
                        else 999
                    )
                    bias = 0.2 if pw < 0.25 else 0.1
                    if operator in (">=", ">"):
                        rec_val = start_t + bias * pw
                    else:
                        rec_val = end_t - bias * pw
                    rec_idx = int(np.argmin([abs(r["threshold"] - rec_val) for r in w]))
                    best_plateau = {
                        "is_plateau": True,
                        "_robustness": float(robustness),
                        "start_threshold": w[0]["threshold"],
                        "end_threshold": w[-1]["threshold"],
                        "plateau_width": float(pw),
                        "confidence": _width_to_confidence(pw),
                        "mean_sharpe": float(mean_sh),
                        "cv_sharpe": float(cv_sharpe),
                        "cv_trades": float(cv_trades),
                        "cv_trades_warning": True,  # Trades CV 超标
                        "mean_trades": float(np.mean(trades_list)),
                        "recommended": float(w[rec_idx]["threshold"]),
                    }

    if best_plateau is None:
        sharpes = [r["sharpe"] for r in valid]
        best_idx = int(np.argmax(sharpes))
        return {
            "is_plateau": False,
            "reason": "无 CV<0.3 的稳定窗口",
            "best_single": {
                "threshold": valid[best_idx]["threshold"],
                "sharpe": valid[best_idx]["sharpe"],
                "trades": valid[best_idx]["trades"],
            },
        }

    return best_plateau


def _width_to_confidence(width: float) -> str:
    """将 plateau 宽度映射为置信度等级。

    宽度越大 = decision boundary 曲率越低 = 越可部署。
    """
    if width >= 0.3:
        return "HIGH"
    elif width >= 0.15:
        return "MEDIUM"
    else:
        return "LOW"


# ================================================================
# HTML 报告生成
# ================================================================


def _generate_html_report(
    all_results: Dict[str, Any],
    strategy: str,
) -> str:
    """生成 HTML 报告，含 Sharpe vs Threshold 图表。"""

    filter_sections = ""
    for filter_name, filter_data in all_results.items():
        conditions_html = ""
        for cond_result in filter_data.get("scanned_conditions", []):
            feature = cond_result["feature"]
            operator = cond_result["operator"]
            current = cond_result["current_value"]
            scan_data = cond_result["scan_results"]
            plateau = cond_result["plateau"]

            # 构造 SVG 图表（Sharpe vs Threshold）
            valid_pts = [r for r in scan_data if not r.get("too_few")]
            if not valid_pts:
                conditions_html += f"<div class='cond'><h4>{feature} {operator} {current} — 无有效数据</h4></div>"
                continue

            thresholds = [r["threshold"] for r in valid_pts]
            sharpes = [r["sharpe"] for r in valid_pts]
            trades = [r["trades"] for r in valid_pts]

            # 简单 SVG 折线图
            w, h = 600, 250
            pad = 50
            min_sh = min(sharpes) * 0.9 if min(sharpes) > 0 else -0.05
            max_sh = max(sharpes) * 1.1 if max(sharpes) > 0 else 0.5
            min_t = min(thresholds)
            max_t = max(thresholds)
            t_range = max_t - min_t if max_t > min_t else 1
            sh_range = max_sh - min_sh if max_sh > min_sh else 1

            def tx(v):
                return pad + (v - min_t) / t_range * (w - 2 * pad)

            def ty(v):
                return h - pad - (v - min_sh) / sh_range * (h - 2 * pad)

            # Sharpe line
            pts = " ".join(
                f"{tx(t):.1f},{ty(s):.1f}" for t, s in zip(thresholds, sharpes)
            )
            # Current value marker
            cx = tx(current)
            # Plateau region
            plateau_rect = ""
            plateau_text = ""
            if plateau.get("is_plateau"):
                px1 = tx(plateau["start_threshold"])
                px2 = tx(plateau["end_threshold"])
                plateau_rect = f'<rect x="{px1:.0f}" y="{pad}" width="{px2-px1:.0f}" height="{h-2*pad}" fill="#4CAF50" opacity="0.15"/>'
                conf_color = {
                    "HIGH": "#2E7D32",
                    "MEDIUM": "#F57F17",
                    "LOW": "#E65100",
                }.get(plateau.get("confidence", ""), "#333")
                warn_html = (
                    ' <span style="color:#E65100">⚠️ Trades CV>0.4</span>'
                    if plateau.get("cv_trades_warning")
                    else ""
                )
                plateau_text = (
                    f"<div class='plateau-ok'>✅ 平坦高原: "
                    f"[{plateau['start_threshold']:.3f}, {plateau['end_threshold']:.3f}] "
                    f"width={plateau['plateau_width']:.3f} "
                    f"<span style='color:{conf_color};font-weight:bold'>conf={plateau['confidence']}</span> "
                    f"mean Sharpe={plateau['mean_sharpe']:.4f} "
                    f"CV(Sharpe={plateau['cv_sharpe']:.3f}, Trades={plateau['cv_trades']:.3f}) "
                    f"推荐阈值={plateau['recommended']:.3f} "
                    f"(当前={current}){warn_html}</div>"
                )
            else:
                reason = plateau.get("reason", "")
                best = plateau.get("best_single", {})
                plateau_text = f"<div class='plateau-no'>❌ 无平坦高原 ({reason})"
                if best:
                    plateau_text += (
                        f" | 最佳单点: threshold={best.get('threshold')}, "
                        f"Sharpe={best.get('sharpe', 0):.4f}, "
                        f"Trades={best.get('trades', 0)}"
                    )
                plateau_text += "</div>"

            # Trades bar chart (secondary axis, right side)
            max_trades = max(trades) if trades else 1
            bars = ""
            bar_w = max(2, (w - 2 * pad) / len(valid_pts) * 0.6)
            for t, tr in zip(thresholds, trades):
                bx = tx(t) - bar_w / 2
                bh = (tr / max_trades) * (h - 2 * pad) * 0.3
                by = h - pad - bh
                bars += f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="#2196F3" opacity="0.25"/>'

            svg = f"""<svg width="{w}" height="{h}" style="border:1px solid #ddd; background:white">
                {plateau_rect}
                {bars}
                <polyline points="{pts}" fill="none" stroke="#FF5722" stroke-width="2"/>
                {''.join(f'<circle cx="{tx(t):.1f}" cy="{ty(s):.1f}" r="3" fill="#FF5722"/>' for t, s in zip(thresholds, sharpes))}
                <line x1="{cx:.0f}" y1="{pad}" x2="{cx:.0f}" y2="{h-pad}" stroke="#333" stroke-width="1.5" stroke-dasharray="5,3"/>
                <text x="{cx:.0f}" y="{pad-5}" text-anchor="middle" font-size="10" fill="#333">current={current}</text>
                <text x="{pad-5}" y="{pad-5}" text-anchor="end" font-size="10" fill="#999">Sharpe</text>
                <text x="{w-pad+5}" y="{h-pad+15}" font-size="10" fill="#999">threshold</text>
                <text x="{pad}" y="{h-5}" font-size="9" fill="#999">{min_t:.2f}</text>
                <text x="{w-pad}" y="{h-5}" font-size="9" fill="#999">{max_t:.2f}</text>
                <text x="5" y="{ty(max_sh):.0f}" font-size="9" fill="#999">{max_sh:.3f}</text>
                <text x="5" y="{ty(min_sh):.0f}" font-size="9" fill="#999">{min_sh:.3f}</text>
            </svg>"""

            # Data table
            rows = ""
            for r in scan_data:
                marker = " ← current" if abs(r["threshold"] - current) < 0.005 else ""
                style = ' style="background:#fff3e0"' if marker else ""
                if r.get("too_few"):
                    rows += f"<tr{style}><td>{r['threshold']:.3f}</td><td>{r['trades']}</td><td colspan='3'>trades < 20{marker}</td></tr>"
                else:
                    rows += f"<tr{style}><td>{r['threshold']:.3f}</td><td>{r['trades']}</td><td>{r['sharpe']:.4f}</td><td>{r['win_rate']*100:.1f}%</td><td>{r['mean_r']:.4f}{marker}</td></tr>"

            conditions_html += f"""
            <div class="cond">
                <h4>{feature} {operator} <strong>{current}</strong></h4>
                {svg}
                {plateau_text}
                <details><summary>详细数据</summary>
                <table class="small">
                    <tr><th>Threshold</th><th>Trades</th><th>Sharpe</th><th>Win%</th><th>Mean R</th></tr>
                    {rows}
                </table></details>
            </div>"""

        desc = filter_data.get("description", "")
        filter_sections += f"""
        <div class="filter-block">
            <h3>🔍 {filter_name}</h3>
            <p class="desc">{desc}</p>
            {conditions_html if conditions_html else '<p class="skip">无可扫描的连续阈值条件</p>'}
        </div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{strategy} Entry Filter Threshold Plateau</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 20px; background: #fafafa; max-width: 900px; margin: 0 auto; padding: 20px; }}
    h1 {{ color: #333; border-bottom: 2px solid #FF5722; padding-bottom: 8px; }}
    h3 {{ color: #FF5722; margin-top: 30px; }}
    h4 {{ color: #666; margin-bottom: 5px; }}
    .filter-block {{ background: white; padding: 15px 20px; margin: 15px 0; border-radius: 8px;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .cond {{ margin: 15px 0; padding: 10px; background: #f9f9f9; border-radius: 5px; }}
    .desc {{ color: #666; font-size: 0.9em; }}
    .skip {{ color: #999; font-style: italic; }}
    .plateau-ok {{ color: #2E7D32; font-weight: bold; margin: 8px 0; padding: 8px;
                   background: #E8F5E9; border-radius: 4px; }}
    .plateau-no {{ color: #E65100; margin: 8px 0; padding: 8px;
                   background: #FFF3E0; border-radius: 4px; }}
    table.small {{ font-size: 0.85em; border-collapse: collapse; margin-top: 8px; }}
    table.small th {{ background: #FF5722; color: white; padding: 5px 10px; }}
    table.small td {{ padding: 4px 10px; border-bottom: 1px solid #eee; }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: #2196F3; }}
</style></head><body>
<h1>🎯 {strategy.upper()} Entry Filter Threshold Plateau</h1>
<p>对每个 entry filter 的连续阈值条件做 plateau 扫描。
红线 = Sharpe，蓝色柱 = Trades，虚线 = 当前阈值，绿色区域 = 平坦高原。</p>
{filter_sections}
</body></html>"""
    return html


# ================================================================
# Main
# ================================================================


def main() -> int:
    p = argparse.ArgumentParser(description="Entry Filter Threshold Plateau Scan")
    p.add_argument("--logs", required=True, help="Input logs parquet")
    p.add_argument("--strategy", required=True)
    p.add_argument("--strategies-root", default="config/strategies")
    p.add_argument(
        "--filter", default=None, help="只扫描指定 filter ID (默认扫描所有 enabled)"
    )
    p.add_argument("--steps", type=int, default=15, help="每个阈值条件的扫描步数")
    p.add_argument("--output", default=None, help="输出 HTML 路径")
    args = p.parse_args()

    # 加载数据
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ {logs_path} not found")
        return 1

    df = pd.read_parquet(logs_path)
    if "_symbol" in df.columns and "symbol" not in df.columns:
        df["symbol"] = df["_symbol"]

    if "bpc_breakout_direction" in df.columns:
        df["entry_direction"] = df["bpc_breakout_direction"].astype(float).copy()
    else:
        # 使用 direction.yaml 确定方向
        from scripts.backtest_execution_layer import (
            load_direction_config,
            apply_direction_rules,
        )

        dir_cfg = load_direction_config(args.strategy, args.strategies_root)
        if dir_cfg:
            applied = apply_direction_rules(df, args.strategy, dir_cfg)
            if applied:
                print(f"   Direction: {applied} (from direction.yaml)")
            else:
                # direction.yaml 规则无一命中
                if "entry_direction" in df.columns:
                    print(f"   Direction: entry_direction (原始列)")
                else:
                    print(f"❌ direction.yaml 规则无一命中，且无 entry_direction 列")
                    return 1
        elif "entry_direction" in df.columns:
            print(f"   Direction: entry_direction (原始列, 无 direction.yaml)")
        else:
            print(
                f"❌ 无法确定方向: 无 bpc_breakout_direction / direction.yaml / entry_direction"
            )
            return 1

    # 直接使用 logs 中的 OHLC（需要 high, low, close, atr）
    has_ohlc = all(c in df.columns for c in ["high", "low", "close", "atr"])
    if not has_ohlc:
        print("❌ Logs 缺少 OHLC 列 (high, low, close, atr)")
        return 1

    merged = df.sort_values(["symbol"]).reset_index(drop=True)
    span_years = _estimate_span_years(merged)

    # 加载配置
    exec_config = load_execution_config(args.strategy, args.strategies_root)
    entry_cfg = load_entry_filters_config(args.strategy, args.strategies_root)
    if not entry_cfg:
        print("❌ entry_filters.yaml not found")
        return 1

    # 确定要扫描的 filter 列表
    filters_to_scan = []
    for f in entry_cfg.get("filters", []):
        if not f.get("enabled", True):
            continue
        if args.filter and f["id"] != args.filter:
            continue
        filters_to_scan.append(f)

    if not filters_to_scan:
        print("❌ 没有匹配的 enabled filter")
        return 1

    print("=" * 70)
    print("🎯 Entry Filter Threshold Plateau Scan")
    print("=" * 70)
    print(f"   Data: {len(merged)} bars, span={span_years:.2f}yr")
    print(f"   Filters to scan: {len(filters_to_scan)}")
    print()

    all_results = {}

    for fdef in filters_to_scan:
        fname = fdef["id"]
        scannable = _find_scannable_conditions(fdef)

        if not scannable:
            print(f"   ⚡ {fname}: 无可扫描的连续阈值条件 (全部是 bool)")
            all_results[fname] = {
                "description": fdef.get("description", ""),
                "scanned_conditions": [],
            }
            continue

        print(f"   🔍 {fname}: {len(scannable)} 个连续阈值条件")

        cond_results = []
        for sc in scannable:
            feature = sc["feature"]
            operator = sc["operator"]
            current = sc["value"]
            idx = sc["index"]

            scan_values = _generate_scan_range(current, operator, n_steps=args.steps)
            print(
                f"      {feature} {operator} {current}  →  扫描 [{scan_values[0]}, {scan_values[-1]}]"
            )

            scan_data = _scan_single_threshold(
                merged,
                exec_config,
                fdef,
                idx,
                scan_values,
                entry_cfg,
                span_years,
            )

            plateau = _find_plateau(scan_data, operator=operator)

            status = "✅ 高原" if plateau.get("is_plateau") else "❌ 无高原"
            if plateau.get("is_plateau"):
                warn = " ⚠️ Trades CV>0.4" if plateau.get("cv_trades_warning") else ""
                print(
                    f"         {status}: [{plateau['start_threshold']:.3f}, {plateau['end_threshold']:.3f}] "
                    f"width={plateau['plateau_width']:.3f} conf={plateau['confidence']} "
                    f"mean Sharpe={plateau['mean_sharpe']:.4f} "
                    f"CV(sh={plateau['cv_sharpe']:.3f}, tr={plateau['cv_trades']:.3f}) "
                    f"推荐={plateau['recommended']:.3f}{warn}"
                )
            else:
                print(f"         {status}: {plateau.get('reason', '')}")

            cond_results.append(
                {
                    "feature": feature,
                    "operator": operator,
                    "current_value": current,
                    "scan_results": scan_data,
                    "plateau": plateau,
                }
            )

        all_results[fname] = {
            "description": fdef.get("description", ""),
            "scanned_conditions": cond_results,
        }

    # 输出 HTML
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = logs_path.parent / "entry_filter_threshold_plateau.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = _generate_html_report(all_results, args.strategy)
    out_path.write_text(html, encoding="utf-8")
    print(f"\n   📊 HTML Report: {out_path}")

    # 摘要
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    for fname, fdata in all_results.items():
        for cr in fdata.get("scanned_conditions", []):
            p_info = cr["plateau"]
            feat = cr["feature"]
            cur = cr["current_value"]
            if p_info.get("is_plateau"):
                rec = p_info["recommended"]
                delta = (
                    f"({rec - cur:+.3f})" if abs(rec - cur) > 0.005 else "(= current)"
                )
                conf = p_info.get("confidence", "?")
                warn = " ⚠️ Trades不稳" if p_info.get("cv_trades_warning") else ""
                print(
                    f"   {fname}.{feat}: 推荐 {rec:.3f} {delta}, "
                    f"高原=[{p_info['start_threshold']:.3f}, {p_info['end_threshold']:.3f}] "
                    f"width={p_info['plateau_width']:.3f} conf={conf}{warn}"
                )
            else:
                print(f"   {fname}.{feat}: 无平坦高原 — {p_info.get('reason', '')}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
