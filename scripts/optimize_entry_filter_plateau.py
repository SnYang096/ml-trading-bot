#!/usr/bin/env python3
"""
Entry Filter 阈值平坦高原扫描

对每个 entry filter 中的连续阈值条件（>=, >, <=, <），
扫描阈值范围，计算 snotio/Trades，找到平坦高原区间。

snotio = mean(R-multiples) = 平均每笔交易的风险调整收益。
Entry Filter 主 KPI。不受 trade count 影响，只有 per-trade 质量提升才会改善。

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
    - HTML 报告（含 snotio vs Threshold 图表）
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

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

    对于 >= / > : 高阈值更严格
    对于 <= / < : 低阈值更严格
    支持负值范围（如 cvd_divergence_score <= -0.776）
    """
    # 动态范围: 当前值 ± 0.4，不截断到 [0, 1]
    margin = 0.4
    low = current_value - margin
    high = current_value + margin

    # 对于归一化特征 (0~1 范围)，适当截断
    if current_value >= 0 and current_value <= 1:
        low = max(0.0, low)
        high = min(1.0, high)

    if abs(high - low) < 1e-8:
        # fallback: 当前值 ± 0.5
        low = current_value - 0.5
        high = current_value + 0.5

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
                    "snotio": 0.0,
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                    "mean_r": 0.0,
                    "too_few": True,
                }
            )
            continue

        exec_returns, _ = simulate_rr_execution(
            merged, exec_config, atr_col="atr", use_tier_params=False
        )
        valid = exec_returns.dropna()
        if len(valid) < 10:
            results.append(
                {
                    "threshold": val,
                    "trades": n_entries,
                    "snotio": 0.0,
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                    "mean_r": 0.0,
                    "too_few": True,
                }
            )
            continue

        sh = compute_sharpe(valid, annualize=False)
        snotio_val = float(valid.mean())  # snotio = mean(R-multiples)
        results.append(
            {
                "threshold": val,
                "trades": len(valid),
                "snotio": snotio_val,
                "sharpe": float(sh),
                "win_rate": float((valid > 0).mean()),
                "mean_r": snotio_val,
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
      - snotio CV < 0.3（收益稳定性）
      - Trades CV < 0.4（执行节奏稳定性 — Entry Filter 特有）

    recommended 不取中点，取 plateau 偏宽容侧：
      - >= / > 条件：低阈值 = 更宽松 → start + bias * width
      - <= / < 条件：高阈值 = 更宽松 → end - bias * width
      - bias 动态绑定 width: 窄高原(width<0.25) bias=0.2, 宽高原 bias=0.1
        直觉：plateau 很宽 → 本来就不敏感 → 不用偏太多
               plateau 较窄 → 更容易 miss → 偏宽容更重要

    原理：plateau 的存在证明了「严格 ≠ 更好」，
    Entry Filter 的错误成本是「错过」而非「多等一次确认」。
    在 snotio 无本质差别的区间内，选更容易触发入场的一侧。

    plateau 排序用 mean_snotio × plateau_width（鲁棒性最大化），
    而非单纯 mean_snotio 最大化，让宽而稳的高原优先于窄而高的尖峰。

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
        snotios = [r["snotio"] for r in w]
        trades_list = [r["trades"] for r in w]
        mean_sn = np.mean(snotios)
        std_sn = np.std(snotios)
        cv_snotio = std_sn / mean_sn if mean_sn > 1e-8 else 999
        mean_tr = np.mean(trades_list)
        std_tr = np.std(trades_list)
        cv_trades = std_tr / mean_tr if mean_tr > 1e-8 else 999

        if cv_snotio < 0.3 and cv_trades < 0.4 and mean_sn > 0:
            start_t = w[0]["threshold"]
            end_t = w[-1]["threshold"]
            plateau_width = abs(end_t - start_t)
            # 排序: mean_snotio × plateau_width（鲁棒性最大化）
            robustness = mean_sn * plateau_width
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
                    "mean_snotio": float(mean_sn),
                    "cv_snotio": float(cv_snotio),
                    "cv_trades": float(cv_trades),
                    "mean_trades": float(mean_tr),
                    "recommended": float(w[rec_idx]["threshold"]),
                }

    if best_plateau is None:
        # 尝试只用 snotio CV（放宽 Trades CV 约束）
        for i in range(len(valid) - window + 1):
            w = valid[i : i + window]
            snotios = [r["snotio"] for r in w]
            mean_sn = np.mean(snotios)
            cv_snotio = np.std(snotios) / mean_sn if mean_sn > 1e-8 else 999
            if cv_snotio < 0.3 and mean_sn > 0:
                start_t = w[0]["threshold"]
                end_t = w[-1]["threshold"]
                pw = abs(end_t - start_t)
                robustness = mean_sn * pw
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
                        "mean_snotio": float(mean_sn),
                        "cv_snotio": float(cv_snotio),
                        "cv_trades": float(cv_trades),
                        "cv_trades_warning": True,  # Trades CV 超标
                        "mean_trades": float(np.mean(trades_list)),
                        "recommended": float(w[rec_idx]["threshold"]),
                    }

    if best_plateau is None:
        snotios = [r["snotio"] for r in valid]
        best_idx = int(np.argmax(snotios))
        return {
            "is_plateau": False,
            "reason": "无 CV<0.3 的稳定窗口",
            "best_single": {
                "threshold": valid[best_idx]["threshold"],
                "snotio": valid[best_idx]["snotio"],
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
    """生成 HTML 报告，含 snotio vs Threshold 图表。"""

    filter_sections = ""
    for filter_name, filter_data in all_results.items():
        conditions_html = ""
        for cond_result in filter_data.get("scanned_conditions", []):
            feature = cond_result["feature"]
            operator = cond_result["operator"]
            current = cond_result["current_value"]
            scan_data = cond_result["scan_results"]
            plateau = cond_result["plateau"]

            # 构造 SVG 图表（snotio vs Threshold）
            valid_pts = [r for r in scan_data if not r.get("too_few")]
            if not valid_pts:
                conditions_html += f"<div class='cond'><h4>{feature} {operator} {current} — 无有效数据</h4></div>"
                continue

            thresholds = [r["threshold"] for r in valid_pts]
            snotios = [r["snotio"] for r in valid_pts]
            trades = [r["trades"] for r in valid_pts]

            # 简单 SVG 折线图
            w, h = 600, 250
            pad = 50
            min_sh = min(snotios) * 0.9 if min(snotios) > 0 else -0.05
            max_sh = max(snotios) * 1.1 if max(snotios) > 0 else 0.5
            min_t = min(thresholds)
            max_t = max(thresholds)
            t_range = max_t - min_t if max_t > min_t else 1
            sh_range = max_sh - min_sh if max_sh > min_sh else 1

            def tx(v):
                return pad + (v - min_t) / t_range * (w - 2 * pad)

            def ty(v):
                return h - pad - (v - min_sh) / sh_range * (h - 2 * pad)

            # snotio line
            pts = " ".join(
                f"{tx(t):.1f},{ty(s):.1f}" for t, s in zip(thresholds, snotios)
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
                    f"mean snotio={plateau['mean_snotio']:.4f} "
                    f"CV(snotio={plateau['cv_snotio']:.3f}, Trades={plateau['cv_trades']:.3f}) "
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
                        f"snotio={best.get('snotio', 0):.4f}, "
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
                {''.join(f'<circle cx="{tx(t):.1f}" cy="{ty(s):.1f}" r="3" fill="#FF5722"/>' for t, s in zip(thresholds, snotios))}
                <line x1="{cx:.0f}" y1="{pad}" x2="{cx:.0f}" y2="{h-pad}" stroke="#333" stroke-width="1.5" stroke-dasharray="5,3"/>
                <text x="{cx:.0f}" y="{pad-5}" text-anchor="middle" font-size="10" fill="#333">current={current}</text>
                <text x="{pad-5}" y="{pad-5}" text-anchor="end" font-size="10" fill="#999">snotio</text>
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
                    rows += f"<tr{style}><td>{r['threshold']:.3f}</td><td>{r['trades']}</td><td>{r['snotio']:.4f}</td><td>{r['win_rate']*100:.1f}%</td><td>{r['mean_r']:.4f}{marker}</td></tr>"

            conditions_html += f"""
            <div class="cond">
                <h4>{feature} {operator} <strong>{current}</strong></h4>
                {svg}
                {plateau_text}
                <details><summary>详细数据</summary>
                <table class="small">
                    <tr><th>Threshold</th><th>Trades</th><th>snotio</th><th>Win%</th><th>Mean R</th></tr>
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
红线 = snotio (mean R-multiples)，蓝色柱 = Trades，虚线 = 当前阈值，绿色区域 = 平坦高原。</p>
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
    p.add_argument(
        "--research",
        action="store_true",
        help="读取研究文件 (config/strategies/{strategy}/entry_filters.yaml) 而非 archetypes 生产文件",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help="自动写入 archetypes/entry_filters.yaml (Tier A: plateau验证 + Tier B: snotio>baseline放过)",
    )
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

    # Gate 过滤: entry filter 阈值必须在 gate 过滤后的分布上优化,
    # 否则会和 backtest/execution 产生 distribution mismatch
    # (backtest 先过 gate_decision 再应用 entry filter)
    if "gate_decision" in merged.columns:
        veto_mask = merged["gate_decision"] != "allow"
        n_allowed = int((~veto_mask).sum())
        merged.loc[veto_mask, "entry_direction"] = 0.0
        print(
            f"   \U0001f6aa Gate filter (auto): {n_allowed} allow / {len(merged)} total"
        )
    elif "gate_ok" in merged.columns:
        veto_mask = merged["gate_ok"] != True  # noqa: E712
        n_allowed = int((~veto_mask).sum())
        merged.loc[veto_mask, "entry_direction"] = 0.0
        print(
            f"   \U0001f6aa Gate filter (auto): {n_allowed} allow / {len(merged)} total"
        )

    n_entries = int((merged["entry_direction"] != 0).sum())
    if n_entries == 0:
        print("\u274c No entry signals after gate filter")
        return 1
    print(
        f"   Entry signals: {n_entries} / {len(merged)} bars ({n_entries/len(merged)*100:.1f}%)"
    )

    span_years = _estimate_span_years(merged)

    # 加载配置
    exec_config = load_execution_config(args.strategy, args.strategies_root)
    entry_cfg = load_entry_filters_config(
        args.strategy, args.strategies_root, research=args.research
    )
    src_label = "研究文件" if args.research else "archetypes"
    print(f"   Config: {src_label}")
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
                    f"mean snotio={plateau['mean_snotio']:.4f} "
                    f"CV(sn={plateau['cv_snotio']:.3f}, tr={plateau['cv_trades']:.3f}) "
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

    # ================================================================
    # 去冗余分析: Jaccard 重叠 + 贪心前向选择
    # ================================================================
    if len(filters_to_scan) >= 2:
        dedup_result = _run_dedup_analysis(
            merged,
            exec_config,
            filters_to_scan,
            all_results,
            span_years,
        )
        if dedup_result:
            print("\n" + "=" * 70)
            print("🧩 DEDUP: 贪心前向选择 (OR 组合 snotio 最大化)")
            print("=" * 70)

            # Jaccard 矩阵
            names = dedup_result["names"]
            jmat = dedup_result["jaccard_matrix"]
            n = len(names)
            max_name_len = max(len(nm) for nm in names)

            print(f"\n   📊 Jaccard 重叠矩阵 (>0.5 = 高度重叠):")
            # header
            hdr = " " * (max_name_len + 4)
            for j in range(n):
                hdr += f" {j:>4d}"
            print(hdr)
            for i in range(n):
                row = f"   {i:>2d}. {names[i]:<{max_name_len}s}"
                for j in range(n):
                    v = jmat[i][j]
                    if i == j:
                        row += "    -"
                    elif v >= 0.5:
                        row += f" \033[91m{v:.2f}\033[0m"  # red
                    elif v >= 0.3:
                        row += f" \033[93m{v:.2f}\033[0m"  # yellow
                    else:
                        row += f" {v:.2f}"
                print(row)

            # 单独 snotio
            singles = dedup_result["single_snotios"]
            print(f"\n   🎯 单独 snotio (用 plateau 推荐阈值):")
            for nm, sn in sorted(singles.items(), key=lambda x: -x[1]):
                print(f"      {nm:<{max_name_len}s}  snotio={sn:.2f}")

            # 贪心选择结果
            selected = dedup_result["selected"]
            print(f"\n   ✅ 推荐组合 ({len(selected)} 个 filter, OR):")
            for step in selected:
                print(
                    f"      Step {step['step']}: +{step['name']:<{max_name_len}s} "
                    f"OR-snotio={step['combined_snotio']:.2f} "
                    f"(±{step['delta_snotio']:+.2f}) "
                    f"trades={step['combined_trades']}"
                )

            bl_snotio = dedup_result.get("baseline_snotio", 0)
            final_snotio = selected[-1]["combined_snotio"] if selected else 0
            print(
                f"\n   Baseline={bl_snotio:.2f} → "
                f"Final OR={final_snotio:.2f} ({(final_snotio/bl_snotio - 1)*100:+.1f}%)"
            )
            print()

    # --promote: 自动写入 archetypes/entry_filters.yaml
    if args.promote:
        _promote_entry_filters_yaml(
            all_results=all_results,
            filters_scanned=filters_to_scan,
            entry_cfg=entry_cfg,
            strategy=args.strategy,
            strategies_root=args.strategies_root,
            merged=merged,
            exec_config=exec_config,
        )

    return 0


def _promote_entry_filters_yaml(
    all_results: Dict[str, Any],
    filters_scanned: List[Dict[str, Any]],
    entry_cfg: Dict[str, Any],
    strategy: str,
    strategies_root: str,
    merged: Optional[pd.DataFrame] = None,
    exec_config: Optional[Dict[str, Any]] = None,
) -> None:
    """将优化结果写入 archetypes/entry_filters.yaml。

    两级准入 (OR 组合下，放入只增加入场，不移除):
      Tier A (PLATEAU): 有稳定高原 → 用推荐阈值
      Tier B (SNOTIO):  无高原但 snotio > baseline → 用原始阈值直接放过

    两级都需要通过 z-test 显著性检验 (p < 0.05):
      H0: filter snotio = baseline snotio (无效)
      不显著 = 可能是噪声，自动排除
    """
    import yaml
    from scipy.stats import norm as _norm_dist

    # --- 计算 baseline snotio (如果有数据) ---
    bl_snotio = 0.0
    bl_std = 1.5
    filter_snotios: Dict[str, Tuple[float, int]] = {}  # {fname: (snotio, trades)}
    can_eval = merged is not None and exec_config is not None

    if can_eval:
        original_dir = merged["entry_direction"].copy()
        # baseline: 全部入场
        bl_ret, _ = simulate_rr_execution(
            merged, exec_config, atr_col="atr", use_tier_params=False
        )
        bl_valid = bl_ret.dropna()
        bl_snotio = float(bl_valid.mean()) if len(bl_valid) >= 10 else 0.0
        # 计算 R-multiple 标准差 (用于显著性检验)
        bl_std = float(bl_valid.std()) if len(bl_valid) >= 10 else 1.5
        print(
            f"   📏 Baseline snotio={bl_snotio:.4f} (trades={len(bl_valid)}, σ={bl_std:.3f})"
        )

        # 逐个 filter 评估 snotio
        for fdef in filters_scanned:
            fname = fdef["id"]
            mask = _build_mask_from_conditions(
                merged, fdef.get("conditions", []), silent=True
            )
            merged["entry_direction"] = original_dir.copy()
            merged.loc[~mask, "entry_direction"] = 0.0
            n_entries = int((merged["entry_direction"] != 0).sum())
            if n_entries < 5:
                filter_snotios[fname] = (0.0, n_entries)
                continue
            exec_ret, _ = simulate_rr_execution(
                merged, exec_config, atr_col="atr", use_tier_params=False
            )
            valid = exec_ret.dropna()
            sn = float(valid.mean()) if len(valid) >= 5 else 0.0
            filter_snotios[fname] = (sn, len(valid))
        merged["entry_direction"] = original_dir  # 恢复

    # --- 显著性检验函数 ---
    def _is_significant(
        sn: float, n: int, sigma: float = 1.5
    ) -> Tuple[bool, float, float]:
        """z-test: H0 为 filter snotio = baseline snotio。
        返回 (is_significant, z_score, p_value)。
        """
        if n < 5 or sigma <= 0:
            return False, 0.0, 1.0
        se = sigma / np.sqrt(n)
        z = (sn - bl_snotio) / se
        p = 1 - _norm_dist.cdf(z)
        return p < 0.05, z, p

    # 用实际数据的 std 替代假设值
    sigma_for_test = bl_std if can_eval and bl_std > 0 else 1.5

    # --- 分级准入 ---
    promoted_filters = []
    tier_a_count = 0
    tier_b_count = 0

    for fdef in filters_scanned:
        fname = fdef["id"]
        fdata = all_results.get(fname, {})

        # 收集 plateau 推荐阈值
        rec_map: Dict[str, float] = {}
        has_any_plateau = False
        for cr in fdata.get("scanned_conditions", []):
            p = cr["plateau"]
            if p.get("is_plateau"):
                rec_map[cr["feature"]] = round(p["recommended"], 4)
                has_any_plateau = True

        # Tier A: 有 plateau + 显著性检验
        if has_any_plateau:
            # 显著性检查
            if can_eval and fname in filter_snotios:
                sn, trades = filter_snotios[fname]
                sig, z, p = _is_significant(sn, trades, sigma_for_test)
                if not sig:
                    print(
                        f"      ❌ {fname}: plateau通过但snotio不显著 (sn={sn:.4f} z={z:.2f} p={p:.4f}), 排除"
                    )
                    continue
                sig_note = f", z={z:.2f} p={p:.4f}"
            else:
                sig_note = ""

            new_conditions = []
            for cond in fdef.get("conditions", []):
                c = dict(cond)
                if c["feature"] in rec_map:
                    c["value"] = rec_map[c["feature"]]
                new_conditions.append(c)

            plateau_notes = []
            for cr in fdata.get("scanned_conditions", []):
                p = cr["plateau"]
                if p.get("is_plateau"):
                    plateau_notes.append(
                        f"{cr['feature']}={p['recommended']:.3f}(conf={p.get('confidence','?')})"
                    )

            pf = {
                "id": fname,
                "enabled": True,
                "description": fdef.get("description", ""),
                "conditions": new_conditions,
                "tier": "A_PLATEAU",
                "notes": f"plateau: {', '.join(plateau_notes)}{sig_note}",
            }
            promoted_filters.append(pf)
            tier_a_count += 1
            continue

        # Tier B: 无 plateau 但 snotio > baseline + 显著性检验
        if can_eval and fname in filter_snotios:
            sn, trades = filter_snotios[fname]
            sig, z, p = _is_significant(sn, trades, sigma_for_test)
            if sn > bl_snotio and trades >= 5 and sig:
                pf = {
                    "id": fname,
                    "enabled": True,
                    "description": fdef.get("description", ""),
                    "conditions": [dict(c) for c in fdef.get("conditions", [])],
                    "tier": "B_SNOTIO",
                    "notes": f"snotio={sn:.4f}(>{bl_snotio:.4f} bl), trades={trades}, z={z:.2f} p={p:.4f}, 无plateau用原始阈值",
                }
                promoted_filters.append(pf)
                tier_b_count += 1

    arch_dir = Path(strategies_root) / strategy / "archetypes"
    arch_dir.mkdir(parents=True, exist_ok=True)
    output_path = arch_dir / "entry_filters.yaml"

    if not promoted_filters:
        # 没有 filter 通过 → 清空 archetypes/entry_filters.yaml
        empty_header = (
            f"# {strategy.upper()} Entry Filter Archetype\n"
            f"# Auto-promoted by optimize_entry_filter_plateau.py\n"
            f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"# Promoted: 0 filters (全部未通过显著性检验)\n"
            f"# Baseline snotio: {bl_snotio:.4f}, \u03c3={sigma_for_test:.3f}\n"
            f"#\n"
            f"# 没有 entry filter 通过准入, 策略将无条件入场\n"
            f"\n"
        )
        empty_cfg = {"filters": [], "combination_mode": "or"}
        yaml_content = yaml.dump(
            empty_cfg,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
        output_path.write_text(empty_header + yaml_content, encoding="utf-8")
        print(f"\n   ⚠️ --promote: 没有 filter 通过准入, 已清空 {output_path}")
        return

    header = (
        f"# {strategy.upper()} Entry Filter Archetype\n"
        f"# Auto-promoted by optimize_entry_filter_plateau.py\n"
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"# Promoted: {len(promoted_filters)} filters ({tier_a_count} plateau + {tier_b_count} snotio-only)\n"
        f"# Baseline snotio: {bl_snotio:.4f}, \u03c3={sigma_for_test:.3f}\n"
        f"# Significance: z-test p<0.05 required for all tiers\n"
        f"#\n"
        f'# 职责: "现在该入场吗?" \u2192 硬二值控制入场时机\n'
        f"# 多个 filter 之间是 OR 关系\n"
        f"# Tier A = plateau + 显著, Tier B = snotio + 显著 (no plateau)\n"
        f"\n"
    )

    out_cfg = {
        "filters": promoted_filters,
        "combination_mode": entry_cfg.get("combination_mode", "or"),
    }

    yaml_content = yaml.dump(
        out_cfg,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )

    output_path.write_text(header + yaml_content, encoding="utf-8")
    print(f"\n   ✅ --promote: 写入 {output_path}")
    print(f"      Tier A (PLATEAU):  {tier_a_count} filters")
    print(f"      Tier B (SNOTIO):   {tier_b_count} filters")
    for pf in promoted_filters:
        tier_label = "🅰" if pf.get("tier") == "A_PLATEAU" else "🅱"
        print(f"      {tier_label} {pf['id']}: {pf['notes']}")


def _run_dedup_analysis(
    merged: pd.DataFrame,
    exec_config: Dict[str, Any],
    filters_to_scan: List[Dict[str, Any]],
    all_results: Dict[str, Any],
    span_years: float,
) -> Optional[Dict[str, Any]]:
    """去冗余分析: Jaccard 重叠矩阵 + 贪心前向选择。

    1. 用 plateau 推荐阈值重建每个 filter 的 pass mask
    2. 计算 pairwise Jaccard 相似度
    3. 单独评估每个 filter 的 execution snotio
    4. 贪心前向选择: 每轮加入 OR 后 snotio 提升最大的 filter
    """
    original_entry = merged["entry_direction"].copy()

    # --- Step 1: 构建 masks ---
    masks: Dict[str, pd.Series] = {}
    plateau_thresholds: Dict[str, Dict[str, float]] = (
        {}
    )  # {filter_id: {feature: rec_val}}

    for fdef in filters_to_scan:
        fname = fdef["id"]
        fdata = all_results.get(fname, {})

        # 收集 plateau 推荐阈值
        rec_map: Dict[str, float] = {}
        for cr in fdata.get("scanned_conditions", []):
            p = cr["plateau"]
            if p.get("is_plateau"):
                rec_map[cr["feature"]] = p["recommended"]
        plateau_thresholds[fname] = rec_map

        # 用推荐阈值重建 conditions
        adjusted_conds = []
        for cond in fdef.get("conditions", []):
            c = dict(cond)  # shallow copy
            feat = c["feature"]
            if feat in rec_map:
                c["value"] = rec_map[feat]
            adjusted_conds.append(c)

        mask = _build_mask_from_conditions(merged, adjusted_conds, silent=True)
        masks[fname] = mask

    if not masks:
        return None

    names = list(masks.keys())
    n = len(names)

    # --- Step 2: Jaccard 矩阵 ---
    jmat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mi = masks[names[i]]
        for j in range(i + 1, n):
            mj = masks[names[j]]
            intersection = (mi & mj).sum()
            union = (mi | mj).sum()
            jac = float(intersection / union) if union > 0 else 0.0
            jmat[i][j] = jac
            jmat[j][i] = jac
        jmat[i][i] = 1.0

    # --- Step 3: 单独评估 snotio ---
    single_snotios: Dict[str, float] = {}
    single_trades: Dict[str, int] = {}
    for fname in names:
        merged["entry_direction"] = original_entry.copy()
        merged.loc[~masks[fname], "entry_direction"] = 0.0
        n_entries = int((merged["entry_direction"] != 0).sum())
        if n_entries < 20:
            single_snotios[fname] = 0.0
            single_trades[fname] = n_entries
            continue
        exec_returns, _ = simulate_rr_execution(
            merged, exec_config, atr_col="atr", use_tier_params=False
        )
        valid = exec_returns.dropna()
        single_snotios[fname] = float(valid.mean()) if len(valid) >= 10 else 0.0
        single_trades[fname] = len(valid)

    # Baseline (no filter)
    merged["entry_direction"] = original_entry.copy()
    bl_exec, _ = simulate_rr_execution(
        merged, exec_config, atr_col="atr", use_tier_params=False
    )
    bl_valid = bl_exec.dropna()
    bl_snotio = float(bl_valid.mean()) if len(bl_valid) >= 10 else 0.0

    # --- Step 4: 贪心前向选择 ---
    remaining = set(names)
    selected_steps = []
    combined_mask = pd.Series(False, index=merged.index)
    prev_snotio = 0.0

    for step_idx in range(1, n + 1):
        best_name = None
        best_snotio = -999.0
        best_trades = 0

        for cand in remaining:
            trial_mask = combined_mask | masks[cand]
            merged["entry_direction"] = original_entry.copy()
            merged.loc[~trial_mask, "entry_direction"] = 0.0
            n_entries = int((merged["entry_direction"] != 0).sum())
            if n_entries < 20:
                continue
            exec_returns, _ = simulate_rr_execution(
                merged, exec_config, atr_col="atr", use_tier_params=False
            )
            valid = exec_returns.dropna()
            if len(valid) < 10:
                continue
            sn = float(valid.mean())
            if sn > best_snotio:
                best_snotio = sn
                best_name = cand
                best_trades = len(valid)

        if best_name is None:
            break

        delta = best_snotio - prev_snotio if step_idx > 1 else best_snotio
        # 停止条件: snotio 不再提升或下降
        if step_idx > 1 and best_snotio <= prev_snotio:
            break

        combined_mask = combined_mask | masks[best_name]
        remaining.discard(best_name)
        prev_snotio = best_snotio

        selected_steps.append(
            {
                "step": step_idx,
                "name": best_name,
                "combined_snotio": best_snotio,
                "delta_snotio": delta,
                "combined_trades": best_trades,
            }
        )

    # 恢复
    merged["entry_direction"] = original_entry

    return {
        "names": names,
        "jaccard_matrix": jmat,
        "single_snotios": single_snotios,
        "single_trades": single_trades,
        "selected": selected_steps,
        "baseline_snotio": bl_snotio,
    }


if __name__ == "__main__":
    raise SystemExit(main())
