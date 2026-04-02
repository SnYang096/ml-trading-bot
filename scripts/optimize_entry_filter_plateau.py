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
from scripts.stat_method_registry import (
    canonicalize_method_name,
    evaluate_rr_split_method,
    get_canonical_methods,
)
from scripts.locked_entry_filter_utils import load_locked_entry_filters

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
    snotio_cv_max: float = 0.3,
    trades_cv_max: float = 0.4,
) -> Dict[str, Any]:
    """分析扫描结果，找到平坦高原区间。

    使用滑动窗口双 CV 判定：
      - snotio CV < snotio_cv_max（收益稳定性）
      - Trades CV < trades_cv_max（执行节奏稳定性 — Entry Filter 特有）

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

        if cv_snotio < snotio_cv_max and cv_trades < trades_cv_max and mean_sn > 0:
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
            if cv_snotio < snotio_cv_max and mean_sn > 0:
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
    # ── Meta-Algorithm 模式 ──
    p.add_argument(
        "--meta-algorithm",
        action="store_true",
        help="启用 SHAP∩Gain Meta-Algorithm 模式: LightGBM → SHAP∩Gain → holdout 统计验证 → plateau → 规则输出. "
        "替代默认的手工 filter 扫描.",
    )
    p.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        metavar="RATIO",
        help="Meta-algorithm 模式: holdout 比例 (默认: 0.2)",
    )
    p.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Only use data before this date for optimization (IS cutoff, avoid OOS lookahead)",
    )
    p.add_argument(
        "--simple-execution",
        action="store_true",
        dest="simple_execution",
        help="使用中性简单执行模式 (SL=1.5R, TP=3R, 50bar timeout)。"
        "用于研究管线评估信号质量，不受 execution.yaml 参数影响。",
    )
    p.add_argument(
        "--features-entry-filter",
        default=None,
        help="features_entry_filter.yaml 路径 (候选特征白名单). "
        "未指定时自动查找 config/strategies/{strategy}/features_entry_filter.yaml",
    )
    p.add_argument(
        "--scoring-method",
        default=None,
        choices=get_canonical_methods(),
        help="评分方法 (canonical names only). 未指定时从 entry_filter_layer.yaml 读取",
    )
    p.add_argument(
        "--significance-p",
        type=float,
        default=0.10,
        metavar="P",
        help="Tier A/B z-test 显著性水平 (单侧 p < P 才 promote; 默认 0.10, 原 0.05 更严)",
    )
    p.add_argument(
        "--significance-min-trades",
        type=int,
        default=4,
        metavar="N",
        help="显著性检验与 filter snotio 估计所需最少成交笔数 (默认 4)",
    )
    p.add_argument(
        "--plateau-window",
        type=int,
        default=4,
        metavar="W",
        help="高原检测滑动窗口最少有效扫描点数 (默认 4, 原为 5)",
    )
    args = p.parse_args()

    # 加载数据
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ {logs_path} not found")
        return 1

    df = pd.read_parquet(logs_path)

    # Apply cutoff date (IS only — avoid OOS lookahead)
    if args.cutoff_date:
        _ts = "timestamp" if "timestamp" in df.columns else None
        if _ts is None and df.index.name == "timestamp":
            df = df.reset_index()
            _ts = "timestamp"
        if _ts:
            df[_ts] = pd.to_datetime(df[_ts])
            _n0 = len(df)
            df = df[df[_ts] < args.cutoff_date]
            print(f"   IS cutoff {args.cutoff_date}: {_n0} → {len(df)} rows")

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
    else:
        print(
            "   \u26a0\ufe0f  无 gate_decision/gate_ok 列 \u2014 Gate 未生效!"
            " 应使用 logs_gated.parquet (见 research_pipeline.yaml data_flow)"
        )
        return 1

    n_entries = int((merged["entry_direction"] != 0).sum())
    if n_entries == 0:
        print("\u274c No entry signals after gate filter")
        return 1
    print(
        f"   Entry signals: {n_entries} / {len(merged)} bars ({n_entries/len(merged)*100:.1f}%)"
    )

    span_years = _estimate_span_years(merged)

    # ── 执行配置: --simple-execution 覆盖 execution.yaml ──
    if getattr(args, "simple_execution", False):
        exec_config = {
            "stop_loss": {"type": "fixed", "initial_r": 1.5},
            "take_profit": {"enabled": True, "target_r": 3.0},
            "holding": {"max_holding_bars": 50, "time_stop_bars": 50},
        }
        print("   Execution: simple (SL=1.5R, TP=3R, 50bar timeout)")
    else:
        exec_config = load_execution_config(args.strategy, args.strategies_root)

    # ── Meta-Algorithm 路由: 加载数据后直接走新流程 ──
    if args.meta_algorithm:
        return _meta_algorithm_entry_filter(args, merged, exec_config)

    # 加载配置
    entry_cfg = load_entry_filters_config(
        args.strategy, args.strategies_root, research=args.research
    )
    src_label = "研究文件" if args.research else "archetypes"
    print(f"   Config: {src_label}")
    if not entry_cfg:
        print("❌ entry_filters.yaml not found")
        return 1

    # 确定要扫描的 filter 列表
    # locked filter 即使 enabled=false 也要参与扫描（支持后续自动回启）。
    filters_to_scan = []
    for f in entry_cfg.get("filters", []):
        if not f.get("enabled", True) and not f.get("locked", False):
            continue
        if args.filter and f["id"] != args.filter:
            continue
        filters_to_scan.append(f)

    if not filters_to_scan:
        print("❌ 没有匹配的 filter（enabled 或 locked）")
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

            plateau = _find_plateau(
                scan_data,
                operator=operator,
                window=max(2, int(getattr(args, "plateau_window", 4) or 4)),
            )

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
            significance_p=float(getattr(args, "significance_p", 0.10) or 0.10),
            significance_min_trades=max(
                2, int(getattr(args, "significance_min_trades", 4) or 4)
            ),
            preserve_unscanned_locked=bool(args.filter),
        )

    return 0


def _meta_algorithm_entry_filter(
    args,
    merged: pd.DataFrame,
    exec_config: Dict[str, Any],
) -> int:
    """Entry Filter: 统计方法扫描 + 多方法 Sharpe 择优.

    从 features_entry_filter.yaml 读取候选特征白名单,
    对所有特征×分位点阈值做统计扫描,
    每种评分方法独立跑完整流程后按 Sharpe 择优.

    评分方法:
    - uplift: effect = mean_rr_pass - mean_rr_reject (ATE)
    - ks: KS 分布分离度
    - t_test: 单尾 Welch t-test p<alpha

    OR 组合: 任一 filter pass → 允许入场.
    """
    import yaml

    strategy = args.strategy
    strategies_root = args.strategies_root

    print(f"\n{'='*80}")
    print(f"🔬 Entry Filter: 统计方法扫描 + Sharpe 择优 ({strategy.upper()})")
    print(f"{'='*80}")

    # ── 0. 加载 KPI Gate 配置 ──
    _kpi_path = Path("config/kpi_gates/entry_filter_layer.yaml")
    if _kpi_path.exists():
        with open(_kpi_path, "r", encoding="utf-8") as _kf:
            _kpi = yaml.safe_load(_kf) or {}
    else:
        _kpi = {}
    _thr = _kpi.get("thresholds", {})
    _scan_cfg = _kpi.get("scan", {})
    _rob_cfg = _kpi.get("robustness", {})

    min_robustness = _thr.get("min_robustness", 0.5)
    corr_threshold = _thr.get("correlation_threshold", 0.80)
    max_rules = _thr.get("max_rules", 5)
    pr_min = _thr.get("pass_rate_min", 0.10)
    pr_max = _thr.get("pass_rate_max", 0.95)
    combined_or_pr_max = _thr.get("combined_or_pass_rate_max", 0.85)
    quantiles = _scan_cfg.get(
        "quantiles", [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95]
    )
    min_samples_cfg = _scan_cfg.get("min_samples", {})
    n_folds = _rob_cfg.get("n_folds", 3)

    # 评分方法: 命令行 > kpi_gates yaml
    scoring_method = canonicalize_method_name(getattr(args, "scoring_method", None))
    if not scoring_method:
        methods_list = _kpi.get(
            "scoring_method_fallbacks",
            [
                "distribution_ks",
                "mean_effect",
                "tail_bad_rate_ratio",
                "upside_positive_rate_ratio",
            ],
        )
        methods_list = [canonicalize_method_name(m) for m in methods_list]
        scoring_method = methods_list[0] if len(methods_list) == 1 else None
    # scoring_method=None 表示外部会多次调用每种 method

    # ── 1. 加载候选特征 ──
    ef_yaml = getattr(args, "features_entry_filter", None)
    if not ef_yaml:
        ef_yaml = str(Path(strategies_root) / strategy / "features_entry_filter.yaml")
    if not Path(ef_yaml).exists():
        print(f"⚠️  {ef_yaml} 不存在, 无候选特征")
        _write_empty_entry_filters(args, strategy, strategies_root)
        return 0

    deps_path = str(Path("config/feature_dependencies.yaml"))
    if not Path(deps_path).exists():
        print(f"❌ config/feature_dependencies.yaml 不存在")
        return 1

    candidate_cols = _resolve_features_for_entry_filter(
        ef_yaml, deps_path, list(merged.columns)
    )
    if not candidate_cols:
        print("⚠️  无候选特征列匹配")
        _write_empty_entry_filters(args, strategy, strategies_root)
        return 0
    print(f"   候选特征: {len(candidate_cols)} 列")

    # ── 2. 检测 outcome 列 ──
    outcome_col = None
    for col in ["forward_rr", "rr", "realized_rr"]:
        if col in merged.columns and merged[col].notna().sum() > 10:
            outcome_col = col
            break
    if outcome_col is None:
        print("❌ 无 outcome 列 (forward_rr/rr)")
        return 1

    # 只用 gate-passed + 有方向的 bar
    trade_mask = merged["entry_direction"] != 0
    df_trades = merged[trade_mask].copy()
    bl_rr = df_trades[outcome_col].dropna()
    bl_mean = float(bl_rr.mean())
    print(f"\n   📏 Baseline: mean({outcome_col})={bl_mean:.4f}, trades={len(bl_rr)}")

    if len(df_trades) < 30:
        print("⚠️  样本不足 (<30), 跳过")
        _write_empty_entry_filters(args, strategy, strategies_root)
        return 0

    # ── 3. 统计扫描: 所有候选特征 × 所有分位点 ──
    method = canonicalize_method_name(scoring_method or "mean_effect")
    _min_s = min_samples_cfg.get(method, {})
    if not _min_s and method == "mean_effect":
        _min_s = min_samples_cfg.get("uplift", {})
    min_pass = _min_s.get("pass", 30) if isinstance(_min_s, dict) else 30
    min_reject = _min_s.get("reject", 30) if isinstance(_min_s, dict) else 30

    print(f"\n{'='*80}")
    print(
        f"📊 统计扫描: method={method}, quantiles={len(quantiles)}, features={len(candidate_cols)}"
    )
    print(f"{'='*80}")

    candidates = []  # List of dicts with rule + score info

    for feat in candidate_cols:
        if feat not in df_trades.columns:
            continue
        vals = df_trades[feat].dropna()
        if len(vals) < min_pass + min_reject:
            continue

        for q in quantiles:
            thr = float(vals.quantile(q))
            # 「选入最好」逻辑 (selective inclusion), 个体 pass_rate ≈ q 或 (1-q):
            # q <= 0.5: pass = feat <= quantile(q)  → 保留最低 q% (低值好特征)
            # q >  0.5: pass = feat >= quantile(q)  → 保留最高 (1-q)% (高值好特征)
            # 个体 pass_rate 在 0.15~0.35, OR 组合后综合 pass_rate 才合理
            if q <= 0.5:
                pass_mask = df_trades[feat] <= thr
                op_str = "<="
            else:
                pass_mask = df_trades[feat] >= thr
                op_str = ">="

            n_pass = int(pass_mask.sum())
            n_reject = int((~pass_mask).sum())
            if n_pass < min_pass or n_reject < min_reject:
                continue

            pass_rate = n_pass / len(df_trades)
            if pass_rate < pr_min or pass_rate > pr_max:
                continue

            rr_pass = df_trades.loc[pass_mask, outcome_col].dropna()
            rr_reject = df_trades.loc[~pass_mask, outcome_col].dropna()
            if len(rr_pass) < min_pass or len(rr_reject) < min_reject:
                continue

            mean_pass = float(rr_pass.mean())
            mean_reject = float(rr_reject.mean())
            effect = mean_pass - mean_reject

            # ── 评分（共享方法注册表） ──
            passed, score, extra = evaluate_rr_split_method(
                method=method,
                rr_pass=rr_pass.values,
                rr_reject=rr_reject.values,
                thresholds=_thr,
            )

            if not passed:
                continue

            candidates.append(
                {
                    "feature": feat,
                    "operator": op_str,
                    "value": round(float(thr), 6),
                    "quantile": q,
                    "pass_rate": round(pass_rate, 4),
                    "mean_rr_pass": round(mean_pass, 4),
                    "mean_rr_reject": round(mean_reject, 4),
                    "effect": round(effect, 4),
                    "score": round(score, 4),
                    "n_pass": n_pass,
                    "n_reject": n_reject,
                    "method": method,
                    **extra,
                }
            )

    print(f"   ✅ 扫描通过: {len(candidates)} 条候选")
    if not candidates:
        print(f"   ⚠️  无候选通过 {method} 检验")
        _write_empty_entry_filters(args, strategy, strategies_root)
        return 0

    # ── 4. Robustness 验证 (3 折时间交叉) ──
    print(f"\n🛡️  Robustness 验证 ({n_folds} folds)...")
    ts_col = "timestamp" if "timestamp" in df_trades.columns else None
    if ts_col is None and df_trades.index.name == "timestamp":
        df_trades = df_trades.reset_index()
        ts_col = "timestamp"

    robust_candidates = []
    if ts_col:
        df_trades[ts_col] = pd.to_datetime(df_trades[ts_col])
        sorted_ts = df_trades[ts_col].sort_values().reset_index(drop=True)
        _n_ts = len(sorted_ts)
        fold_edges = [
            sorted_ts.iloc[min(int(_n_ts * i / n_folds), _n_ts - 1)]
            for i in range(n_folds)
        ]
        fold_edges.append(sorted_ts.iloc[-1] + pd.Timedelta("1s"))  # upper bound

        for c in candidates:
            feat, op_str, thr = c["feature"], c["operator"], c["value"]
            n_agree = 0
            for fi in range(n_folds):
                fold_start = fold_edges[fi]
                fold_end = (
                    fold_edges[fi + 1]
                    if fi + 1 < len(fold_edges)
                    else sorted_ts.max() + pd.Timedelta("1s")
                )
                fold_df = df_trades[
                    (df_trades[ts_col] >= fold_start) & (df_trades[ts_col] < fold_end)
                ]
                if len(fold_df) < 10:
                    continue
                if op_str == ">=":
                    fm = fold_df[feat] >= thr
                else:
                    fm = fold_df[feat] <= thr
                rr_p = fold_df.loc[fm, outcome_col].dropna()
                rr_r = fold_df.loc[~fm, outcome_col].dropna()
                if len(rr_p) >= 5 and len(rr_r) >= 5:
                    if float(rr_p.mean()) > float(rr_r.mean()):
                        n_agree += 1
            rob_score = n_agree / n_folds if n_folds > 0 else 0
            c["robustness"] = round(rob_score, 2)
            if rob_score >= min_robustness:
                robust_candidates.append(c)
    else:
        # 无时间列, 跳过 robustness
        for c in candidates:
            c["robustness"] = 1.0
            robust_candidates.append(c)

    print(f"   Robustness 通过: {len(robust_candidates)} / {len(candidates)}")
    if not robust_candidates:
        print(f"   ⚠️  无规则通过 robustness 验证")
        _write_empty_entry_filters(args, strategy, strategies_root)
        return 0

    # ── 5. 相关性剪枝: 按 score 降序, Pearson > corr_threshold 去重 ──
    robust_candidates.sort(key=lambda x: -x["score"])
    final_rules = []
    used_masks = []  # List of boolean Series

    for c in robust_candidates:
        feat, op_str, thr = c["feature"], c["operator"], c["value"]
        if op_str == ">=":
            mask = (df_trades[feat] >= thr).astype(float)
        else:
            mask = (df_trades[feat] <= thr).astype(float)

        # 检查与已选规则的相关性
        is_redundant = False
        for um in used_masks:
            corr = mask.corr(um)
            if abs(corr) > corr_threshold:
                is_redundant = True
                break
        if is_redundant:
            continue

        final_rules.append(c)
        used_masks.append(mask)
        if len(final_rules) >= max_rules:
            break

    # ── 6. Combined OR pass_rate 守卫: 确保组合后不会 pass 几乎所有 bar ──
    if final_rules and len(df_trades) > 0:
        _or_mask = pd.Series(False, index=df_trades.index)
        for _r in final_rules:
            _f, _op, _v = _r["feature"], _r["operator"], _r["value"]
            if _op == ">=":
                _or_mask = _or_mask | (df_trades[_f] >= _v)
            else:
                _or_mask = _or_mask | (df_trades[_f] <= _v)
        _combined_pr = float(_or_mask.mean())
        print(
            f"\n   📊 OR组合实际 pass_rate={_combined_pr:.1%} (限制={combined_or_pr_max:.0%})"
        )
        if _combined_pr > combined_or_pr_max:
            # 保留 pass_rate 最低的单条规则 (OR 组合中最严格的那条)
            _min_pr_rule = min(final_rules, key=lambda x: x["pass_rate"])
            print(
                f"   ⚠️  OR 组合 pass_rate={_combined_pr:.1%} > {combined_or_pr_max:.0%}, "
                f"剪枝为最严格单条规则: {_min_pr_rule['feature']} {_min_pr_rule['operator']} {_min_pr_rule['value']:.4f} "
                f"(pass={_min_pr_rule['pass_rate']:.1%})"
            )
            final_rules = [_min_pr_rule]

    # ── 7. 结果汇总 ──
    print(f"\n{'='*80}")
    print(f"📋 Entry Filter 结果: {strategy.upper()} (method={method})")
    print(f"{'='*80}")

    if final_rules:
        print(f"\n   ✅ {len(final_rules)} 条规则 (OR 组合)")
        for i, r in enumerate(final_rules, 1):
            _extra_str = ""
            if "ks_stat" in r:
                _extra_str = f"  ks={r['ks_stat']:.3f}"
            elif "t_stat" in r:
                _extra_str = f"  t={r['t_stat']:+.2f} p={r['p_value']:.4f}"
            print(
                f"   {i}. {r['feature']} {r['operator']} {r['value']:.4f}  "
                f"pass={r['pass_rate']:.1%}  effect={r['effect']:+.4f}  "
                f"rob={r['robustness']:.2f}{_extra_str}"
            )
    else:
        print(f"\n   ⚠️  无规则通过, 策略将无条件入场")

    # ── 7. Promote ──
    if args.promote:
        _write_entry_filters(args, strategy, strategies_root, final_rules)

    return 0


def _write_empty_entry_filters(
    args,
    strategy: str,
    strategies_root: str,
):
    """写入空的 entry_filters.yaml."""
    if not args.promote:
        return
    import yaml

    arch_dir = Path(strategies_root) / strategy / "archetypes"
    arch_dir.mkdir(parents=True, exist_ok=True)
    output_path = arch_dir / "entry_filters.yaml"
    src_cfg = load_entry_filters_config(strategy, strategies_root, research=False) or {}
    locked = [
        copy.deepcopy(f)
        for f in (src_cfg.get("filters") or [])
        if isinstance(f, dict) and bool(f.get("locked", False))
    ]
    out_filters: List[Dict[str, Any]] = []
    if locked:
        for lf in locked:
            lf["locked"] = True
            lf["enabled"] = False
            lf["notes"] = (
                str(lf.get("notes", "")).strip()
                + ", auto-disabled(no passing threshold)"
            ).strip(", ")
            out_filters.append(lf)

    header = (
        f"# {strategy.upper()} Entry Filter Archetype\n"
        f"# Auto-promoted by optimize_entry_filter_plateau.py (统计方法)\n"
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"# Promoted: 0 filters"
        + (f", locked kept: {len(out_filters)}" if out_filters else "")
        + "\n"
        f"#\n"
        f"# 没有 entry filter 通过统计验证。"
        f"{' locked 特征池已保留并禁用。' if out_filters else '策略将无条件入场。'}\n"
        f"\n"
    )
    out_cfg = {
        "filters": out_filters,
        "combination_mode": src_cfg.get("combination_mode", "or"),
    }
    yaml_content = yaml.dump(
        out_cfg, allow_unicode=True, default_flow_style=False, sort_keys=False
    )
    output_path.write_text(header + yaml_content, encoding="utf-8")
    print(f"\n   ⚠️  --promote: 写入空 entry_filters → {output_path}")


def _resolve_features_for_entry_filter(
    ef_yaml_path: str,
    deps_path: str,
    available_columns: List[str],
) -> List[str]:
    """从 features_entry_filter.yaml + feature_dependencies.yaml 解析列名.

    features_entry_filter.yaml 格式:
        feature_pipeline:
          requested_features:
            - fer_failure_signals_f
            - cvd_divergence_v2_f
    """
    import yaml

    with open(ef_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fp = cfg.get("feature_pipeline", {})
    requested = fp.get("requested_features", [])
    exclude = set(fp.get("exclude_columns", []))
    if not requested:
        print(
            f"❌ features_entry_filter.yaml 中没有 requested_features: {ef_yaml_path}"
        )
        return []

    with open(deps_path, "r", encoding="utf-8") as f:
        deps_cfg = yaml.safe_load(f)

    all_features_def = deps_cfg.get("features", {})
    available_set = set(available_columns)
    resolved_columns: List[str] = []

    # Entry-filter 候选列强约束:
    # 1) 明确拒绝 price/raw/usd 量纲；
    # 2) 对未声明 output_normalization_map 的列，仅允许明显的 normalized 命名。
    disallow_norm_types = {
        "price_unit",
        "raw",
        "usd",
        "identity",
        "passthrough",
        # 该类通常仍保留绝对量纲，不适合跨标的阈值扫描
        "log1p_robust_rolling",
    }
    normalized_suffixes = ("_pct", "_rank", "_zscore", "_normalized", "_f")
    raw_exact = {
        "cvd",
        "cvd_change_1",
        "cvd_change_5",
        "cvd_change_20",
        "fp_delta_poc",
    }
    # 读取 feature_dependencies.yaml 中维护的 raw_scale_columns 黑名单
    raw_scale_cfg = deps_cfg.get("raw_scale_columns", {}) or {}
    if isinstance(raw_scale_cfg, dict):
        for vals in raw_scale_cfg.values():
            if isinstance(vals, (list, tuple, set)):
                raw_exact.update(str(v) for v in vals if str(v).strip())
    elif isinstance(raw_scale_cfg, (list, tuple, set)):
        raw_exact.update(str(v) for v in raw_scale_cfg if str(v).strip())
    raw_prefixes = ("cvd_change_",)

    def _looks_normalized(col: str) -> bool:
        c = str(col or "").strip()
        if not c:
            return False
        if c in raw_exact:
            return False
        if any(c.startswith(p) for p in raw_prefixes):
            if c.endswith(normalized_suffixes) or "zscore" in c:
                return True
            return False
        if c.endswith(normalized_suffixes):
            return True
        if any(
            k in c for k in ("zscore", "normalized", "_score", "_ratio", "_entropy")
        ):
            return True
        return False

    for feat_f in requested:
        if feat_f not in all_features_def:
            continue
        feat_def = all_features_def[feat_f] or {}
        output_cols = feat_def.get("output_columns", [])
        norm_map = (feat_def.get("compute_params") or {}).get(
            "output_normalization_map"
        ) or {}
        matched: List[str] = []
        dropped: List[str] = []
        for c in output_cols:
            col = str(c)
            if col not in available_set or col in exclude:
                continue
            norm_type = str(norm_map.get(col, "") or "").strip().lower()
            if norm_type:
                if norm_type in disallow_norm_types:
                    dropped.append(f"{col}({norm_type})")
                    continue
                # 有明确 normalization_map 且非禁止类型，允许进入候选
                matched.append(col)
                continue
            # 未声明 normalization_map: 走命名约束，默认只接纳 normalized 样式
            if _looks_normalized(col):
                matched.append(col)
            else:
                dropped.append(col)
        if matched:
            resolved_columns.extend(matched)
        if dropped:
            print(
                f"   ℹ️  EntryFilter drop raw/non-normalized from {feat_f}: "
                + ", ".join(dropped[:10])
                + (" ..." if len(dropped) > 10 else "")
            )

    resolved_columns = list(dict.fromkeys(resolved_columns))  # 去重保序
    return resolved_columns


def _write_entry_filters(
    args,
    strategy: str,
    strategies_root: str,
    rules: List[Dict[str, Any]],
):
    """写入 entry_filters.yaml (统计方法 + OR)."""
    if not args.promote:
        return
    import yaml

    arch_dir = Path(strategies_root) / strategy / "archetypes"
    arch_dir.mkdir(parents=True, exist_ok=True)
    output_path = arch_dir / "entry_filters.yaml"

    if not rules:
        _write_empty_entry_filters(args, strategy, strategies_root)
        return

    promoted = []
    for r in rules:
        method = r.get("method", "t_test")
        # 构建 notes 字符串
        _notes_parts = [
            f"method={method}",
            f"effect={r.get('effect', r.get('delta_rr', 0)):+.4f}",
        ]
        if "ks_stat" in r:
            _notes_parts.append(f"ks={r['ks_stat']:.3f}")
        if "t_stat" in r:
            _notes_parts.append(f"t={r['t_stat']:.2f}, p={r.get('p_value', 0):.4f}")
        if "robustness" in r:
            _notes_parts.append(f"rob={r['robustness']:.2f}")
        _notes_parts.append(f"pass={r['pass_rate']:.1%}")

        promoted.append(
            {
                "id": f"ef_{r['feature']}_{r.get('quantile', 'na')}",
                "enabled": True,
                "description": (f"auto: {r['feature']} {r['operator']} {r['value']}"),
                "conditions": [
                    {
                        "feature": r["feature"],
                        "operator": r["operator"],
                        "value": r["value"],
                    }
                ],
                "notes": ", ".join(_notes_parts),
            }
        )

    # 锁定特征池：允许调阈值/enable，不允许规则被删除
    src_cfg = load_entry_filters_config(strategy, strategies_root, research=False) or {}
    locked = [
        copy.deepcopy(f)
        for f in (src_cfg.get("filters") or [])
        if isinstance(f, dict) and bool(f.get("locked", False))
    ]
    locked_by_id = {str(f.get("id", "")): f for f in locked if f.get("id")}
    merged_promoted: List[Dict[str, Any]] = []
    promoted_ids: set = set()
    for pf in promoted:
        fid = str(pf.get("id", ""))
        if fid in locked_by_id:
            pf["locked"] = True
            if locked_by_id[fid].get("lock_reason") and not pf.get("lock_reason"):
                pf["lock_reason"] = locked_by_id[fid]["lock_reason"]
        merged_promoted.append(pf)
        promoted_ids.add(fid)
    for fid, lf in locked_by_id.items():
        if fid in promoted_ids:
            continue
        nf = copy.deepcopy(lf)
        nf["locked"] = True
        nf["enabled"] = False
        nf["notes"] = (
            str(nf.get("notes", "")).strip() + ", auto-disabled(no passing threshold)"
        ).strip(", ")
        merged_promoted.append(nf)

    header = (
        f"# {strategy.upper()} Entry Filter Archetype\n"
        f"# Auto-promoted by optimize_entry_filter_plateau.py (统计方法)\n"
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"# Promoted: {len(promoted)} filters ({promoted[0].get('notes', '').split(',')[0] if promoted else 'N/A'})\n"
        f"# Locked pool: {len(locked)} filters (missing ones auto-disabled)\n"
        f"# Significance: 统计验证 + robustness \u2265 0.5\n"
        f"#\n"
        f'# 职责: "现在该入场吗?" \u2192 硬二值控制\n'
        f"# 多个 filter 之间是 OR 关系\n"
        f"\n"
    )

    out_cfg = {
        "filters": merged_promoted,
        "combination_mode": src_cfg.get("combination_mode", "or"),
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
    for pf in merged_promoted:
        print(f"      {pf['id']}: {pf['notes']}")


def _promote_entry_filters_yaml(
    all_results: Dict[str, Any],
    filters_scanned: List[Dict[str, Any]],
    entry_cfg: Dict[str, Any],
    strategy: str,
    strategies_root: str,
    merged: Optional[pd.DataFrame] = None,
    exec_config: Optional[Dict[str, Any]] = None,
    *,
    significance_p: float = 0.10,
    significance_min_trades: int = 4,
    preserve_unscanned_locked: bool = False,
) -> None:
    """将优化结果写入 archetypes/entry_filters.yaml。

    两级准入 (OR 组合下，放入只增加入场，不移除):
      Tier A (PLATEAU): 有稳定高原 → 用推荐阈值
      Tier B (SNOTIO):  无高原但 snotio > baseline → 用原始阈值直接放过

    两级都需要通过 z-test 显著性检验 (单侧 p < significance_p):
      H0: filter snotio = baseline snotio (无效)
      不显著 = 可能是噪声，自动排除；locked 未通过则 enabled=false

    promote_never_disable (filter 级 YAML 字段):
      为 true 时永不因 promote 失败而 enabled=false；优先保留磁盘上已有阈值/条件，
      若无历史则退回本次扫描用的模板 (entry_cfg)。显著性未过但已有 plateau 时同样退回历史阈值。
    """
    import yaml
    from scipy.stats import norm as _norm_dist

    arch_dir_early = Path(strategies_root) / strategy / "archetypes"
    arch_dir_early.mkdir(parents=True, exist_ok=True)
    output_path_early = arch_dir_early / "entry_filters.yaml"
    existing_cfg_early: Dict[str, Any] = {}
    if output_path_early.exists():
        try:
            existing_cfg_early = (
                yaml.safe_load(output_path_early.read_text(encoding="utf-8")) or {}
            )
        except Exception:
            existing_cfg_early = {}
    _ex_f = existing_cfg_early.get("filters", [])
    if not isinstance(_ex_f, list):
        _ex_f = []
    existing_by_id: Dict[str, Dict[str, Any]] = {
        str(f.get("id", "")): copy.deepcopy(f)
        for f in _ex_f
        if isinstance(f, dict) and str(f.get("id", "")).strip()
    }
    never_disable_ids = set()
    for _f in entry_cfg.get("filters", []) or []:
        if (
            isinstance(_f, dict)
            and str(_f.get("id", "")).strip()
            and bool(_f.get("promote_never_disable"))
        ):
            never_disable_ids.add(str(_f["id"]))

    def _append_note_once(note_src: str, note: str) -> str:
        src = str(note_src or "").strip()
        if note in src:
            return src
        return (src + ", " + note).strip(", ")

    def _never_disable(
        lid: str, lf: Optional[Dict[str, Any]], fdef: Optional[Dict[str, Any]] = None
    ) -> bool:
        if fdef and bool(fdef.get("promote_never_disable")):
            return True
        if lf and bool(lf.get("promote_never_disable")):
            return True
        return str(lid) in never_disable_ids

    # --- 计算 baseline snotio (如果有数据) ---
    bl_snotio = 0.0
    bl_std = 1.5
    filter_snotios: Dict[str, Tuple[float, int]] = {}  # {fname: (snotio, trades)}
    can_eval = merged is not None and exec_config is not None

    _sig_n = max(2, int(significance_min_trades))
    _sig_p = (
        float(significance_p) if significance_p > 0 and significance_p < 1 else 0.10
    )

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
        print(f"   📐 Significance gate: p<{_sig_p:g} (one-sided), min_trades={_sig_n}")

        # 逐个 filter 评估 snotio
        for fdef in filters_scanned:
            fname = fdef["id"]
            mask = _build_mask_from_conditions(
                merged, fdef.get("conditions", []), silent=True
            )
            merged["entry_direction"] = original_dir.copy()
            merged.loc[~mask, "entry_direction"] = 0.0
            n_entries = int((merged["entry_direction"] != 0).sum())
            if n_entries < _sig_n:
                filter_snotios[fname] = (0.0, n_entries)
                continue
            exec_ret, _ = simulate_rr_execution(
                merged, exec_config, atr_col="atr", use_tier_params=False
            )
            valid = exec_ret.dropna()
            sn = float(valid.mean()) if len(valid) >= _sig_n else 0.0
            filter_snotios[fname] = (sn, len(valid))
        merged["entry_direction"] = original_dir  # 恢复

    # --- 显著性检验函数 ---
    def _is_significant(
        sn: float, n: int, sigma: float = 1.5
    ) -> Tuple[bool, float, float]:
        """z-test: H0 为 filter snotio = baseline snotio。
        返回 (is_significant, z_score, p_value)。
        """
        if n < _sig_n or sigma <= 0:
            return False, 0.0, 1.0
        se = sigma / np.sqrt(n)
        z = (sn - bl_snotio) / se
        p = 1 - _norm_dist.cdf(z)
        return p < _sig_p, z, p

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
                        f"      ❌ {fname}: plateau通过但snotio不显著 (sn={sn:.4f} z={z:.2f} p={p:.4f}, need p<{_sig_p:g}), 排除"
                    )
                    if _never_disable(fname, None, fdef):
                        prev = existing_by_id.get(fname)
                        sticky = copy.deepcopy(prev) if prev else copy.deepcopy(fdef)
                        sticky["locked"] = True
                        sticky["enabled"] = True
                        sticky["promote_never_disable"] = True
                        if sticky.get("lock_reason") is None and isinstance(fdef, dict):
                            sticky["lock_reason"] = fdef.get("lock_reason")
                        sticky["notes"] = _append_note_once(
                            sticky.get("notes", ""),
                            (
                                "promote_never_disable: z-test未过，保留磁盘历史阈值"
                                if prev
                                else "promote_never_disable: z-test未过，无历史则使用模板阈值"
                            ),
                        )
                        sticky.pop("tier", None)
                        promoted_filters.append(sticky)
                        print(
                            f"      🔒 {fname}: promote_never_disable → enabled=true, 未采用本轮 plateau 阈值"
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
            if bool(fdef.get("promote_never_disable")):
                pf["promote_never_disable"] = True
            promoted_filters.append(pf)
            tier_a_count += 1
            continue

        # Tier B: 无 plateau 但 snotio > baseline + 显著性检验
        if can_eval and fname in filter_snotios:
            sn, trades = filter_snotios[fname]
            sig, z, p = _is_significant(sn, trades, sigma_for_test)
            if sn > bl_snotio and trades >= _sig_n and sig:
                pf = {
                    "id": fname,
                    "enabled": True,
                    "description": fdef.get("description", ""),
                    "conditions": [dict(c) for c in fdef.get("conditions", [])],
                    "tier": "B_SNOTIO",
                    "notes": f"snotio={sn:.4f}(>{bl_snotio:.4f} bl), trades={trades}, z={z:.2f} p={p:.4f}, 无plateau用原始阈值",
                }
                if bool(fdef.get("promote_never_disable")):
                    pf["promote_never_disable"] = True
                promoted_filters.append(pf)
                tier_b_count += 1

    arch_dir = arch_dir_early
    output_path = output_path_early

    locked_filters = load_locked_entry_filters(output_path)
    locked_by_id = {
        str(f.get("id", "")): copy.deepcopy(f)
        for f in locked_filters
        if isinstance(f, dict) and f.get("id")
    }
    scanned_ids = {
        str(f.get("id", "")).strip()
        for f in filters_scanned
        if isinstance(f, dict) and str(f.get("id", "")).strip()
    }
    promoted_ids = {str(f.get("id", "")) for f in promoted_filters if f.get("id")}
    preserved_unscanned = 0
    auto_disabled = 0
    sticky_never_disable = 0

    for pf in promoted_filters:
        _id = str(pf.get("id", ""))
        if _id in locked_by_id:
            pf["locked"] = True
            if locked_by_id[_id].get("lock_reason") and not pf.get("lock_reason"):
                pf["lock_reason"] = locked_by_id[_id]["lock_reason"]
            if locked_by_id[_id].get("promote_never_disable"):
                pf["promote_never_disable"] = True
    for lid, lf in locked_by_id.items():
        if lid in promoted_ids:
            continue
        if preserve_unscanned_locked and lid not in scanned_ids:
            prev = existing_by_id.get(lid)
            nf = copy.deepcopy(prev) if prev else copy.deepcopy(lf)
            nf["locked"] = True
            if lf.get("lock_reason") and not nf.get("lock_reason"):
                nf["lock_reason"] = lf["lock_reason"]
            if _never_disable(lid, lf, None):
                nf["enabled"] = True
                nf["promote_never_disable"] = True
                nf["notes"] = _append_note_once(
                    nf.get("notes", ""),
                    "promote_never_disable: 本轮未扫描，保持启用",
                )
            promoted_filters.append(nf)
            preserved_unscanned += 1
            continue
        if _never_disable(lid, lf, None):
            prev = existing_by_id.get(lid)
            nf = copy.deepcopy(prev) if prev else copy.deepcopy(lf)
            nf["locked"] = True
            nf["enabled"] = True
            nf["promote_never_disable"] = True
            if lf.get("lock_reason") and not nf.get("lock_reason"):
                nf["lock_reason"] = lf["lock_reason"]
            nf["notes"] = _append_note_once(
                nf.get("notes", ""),
                (
                    "promote_never_disable: 未过 Tier A/B，保留磁盘历史阈值"
                    if prev
                    else "promote_never_disable: 未过 Tier A/B，使用模板阈值"
                ),
            )
            promoted_filters.append(nf)
            sticky_never_disable += 1
            print(f"      🔒 {lid}: promote_never_disable sticky (no auto-disable)")
            continue
        nf = copy.deepcopy(lf)
        nf["locked"] = True
        nf["enabled"] = False
        nf["notes"] = _append_note_once(
            nf.get("notes", ""), "auto-disabled(no passing threshold)"
        )
        promoted_filters.append(nf)
        auto_disabled += 1

    if not promoted_filters:
        # 仅当无 Tier A/B 且无 locked 池时为空（有 locked 时已在上方追加为 enabled=false）
        empty_header = (
            f"# {strategy.upper()} Entry Filter Archetype\n"
            f"# Auto-promoted by optimize_entry_filter_plateau.py\n"
            f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"# Promoted: 0 filters (全部未通过显著性检验)\n"
            f"# Baseline snotio: {bl_snotio:.4f}, \u03c3={sigma_for_test:.3f}\n"
            f"# Significance: z-test one-sided p<{significance_p:g}, min_trades={significance_min_trades}\n"
            f"#\n"
            f"# 没有 entry filter 通过准入, 策略将无条件入场\n"
            f"\n"
        )
        empty_cfg = {
            "filters": [],
            "combination_mode": entry_cfg.get("combination_mode", "or"),
        }
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
        f"# Locked pool kept: {len(locked_by_id)} (missing -> disabled)\n"
        f"# Baseline snotio: {bl_snotio:.4f}, \u03c3={sigma_for_test:.3f}\n"
        f"# Significance: z-test one-sided p<{significance_p:g}, min_trades={significance_min_trades}\n"
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
    if preserve_unscanned_locked:
        print(
            f"      Preserve mode: scanned={len(scanned_ids)}, preserved_unscanned={preserved_unscanned}, auto_disabled_scanned_fail={auto_disabled}"
        )
    if sticky_never_disable:
        print(
            f"      promote_never_disable: sticky Tier-fallback count={sticky_never_disable}"
        )
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
