#!/usr/bin/env python3
"""
Unified Gate Optimization Script - Production Grade Gate Parameter Optimizer
┌───────────────────────────────────────────────────────────────┐
│                    整体流程                                      │
├───────────────────────────────────────────────────────────────┤
│  1. Threshold Scan    →   扫描所有阈值，计算 Lift               │
│  2. Plateau Detection →   找到「稳定平台」区间                   │
│  3. Robustness Score  →   在平台内选择「最不容易炸」的点         │
└───────────────────────────────────────────────────────────────┘
整合三种优化方法：
1. Lift-based optimization (基于条件选择性)
2. Robustness-based optimization (基于稳定性)
3. Hard-Gate System (按优先级顺序优化)

核心改进：
- plateau 定义从「连续」升级为「稳定」
- fallback 从「最强」升级为「最稳」
- 引入 Robustness Score 作为最终决策指标
- 支持区间门控而非单点门控

输入文件支持（路线 B / ABC 统一研究框架 §5）：
- ``features_labeled.parquet`` —— 推荐用法（`mlbot train final --prepare-only` 即可产生，
  无需完整 pipeline run）。本脚本不需要模型 score，只需要 features + label/forward_rr。
  缺失 ``label_col`` 时会自动从 ``forward_rr`` 派生 ``is_good``。
- ``predictions.parquet`` —— 兼容旧用法，结果完全一致。

详见 docs/strategy/ABC统一研究框架_CN.md §5、docs/strategy/R&D工具矩阵_CN.md §4.1bis。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.archetype import (
    GateRule,
    StrategyArchetype,
    load_strategy_archetype,
)
from src.research.stat_kernels.gate_lift import (
    compute_lift_for_threshold,
    scan_thresholds_for_lift,
)
from src.research.stat_kernels.plateau import find_stable_lift_plateau
from src.research.stat_kernels.robustness import (
    RobustnessScore,
    UnifiedOptimizationConfig,
    compute_robustness_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Gate feature whitelist (features_gate.yaml)
# ─────────────────────────────────────────────────────────────────────────────
# Gate 是"执行风险层"，只允许 deny 执行/订单流/风险尾部/波动稳定性等"可防御"
# 特征。策略结构 / 趋势方向 / 位置类特征一律交给 prefilter。
#
# 历史教训：BPC 2024-04~06 +1086R → -157R 衰退，元凶是
# `ema_1200_position > 0.1 deny` —— binary label 视角下强趋势浅 pullback 多被 SL
# 打，KS 把 ema_1200_position 选成 deny 候选，把 BPC 最肥的肉切掉了。
#
# 对称于 features_prefilter.yaml 的 forbidden_prefilter_meta_columns 机制：
# 每策略维护一份 config/strategies/<s>/features_gate.yaml，正向声明
# allowed_gate_deny_features (支持 fnmatch 通配符). 空 / 缺失 = 允许所有
# (向后兼容).
def _load_allowed_gate_deny_features_for_strategy(
    strategy: Optional[str],
) -> List[str]:
    """读取 config/strategies/<strategy>/features_gate.yaml 的白名单.

    返回:
        fnmatch patterns 列表. 空列表 = 本策略未配置 = 允许所有特征 (向后兼容).
    """
    if not strategy:
        return []
    try:
        import yaml as _yaml
    except Exception:
        return []
    path = PROJECT_ROOT / "config" / "strategies" / strategy / "features_gate.yaml"
    if not path.exists():
        return []
    try:
        raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    fp = raw.get("feature_pipeline", {}) or {}
    allowed = fp.get("allowed_gate_deny_features", []) or []
    patterns: List[str] = []
    for x in allowed:
        if isinstance(x, (str, int)):
            s = str(x).strip()
            if s:
                patterns.append(s)
    return patterns


def _is_feature_allowed_for_gate_deny(feature: str, patterns: List[str]) -> bool:
    """匹配 fnmatch 模式. 空 patterns 视为允许所有 (向后兼容)."""
    if not patterns:
        return True
    if not feature:
        return False
    import fnmatch as _fn

    for pat in patterns:
        if _fn.fnmatchcase(feature, pat):
            return True
    return False


def _parse_gate_when_condition(
    when: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    解析 Gate 规则的 when 条件，支持多种格式

    支持格式:
    1. 直接格式: {feature: {value_lt: 0.5}}
    2. any_of 格式: {any_of: [{feature: {value_lt: 0.5}}, ...]}
    3. all_of 格式: {all_of: [{feature: {value_lt: 0.5}}, ...]}
    4. quantile 格式: {feature: {quantile_gt: 0.85}}

    Returns:
        (feature_col, operator, threshold)
    """
    if not isinstance(when, dict):
        return None, None, None

    # 处理 any_of / all_of 嵌套结构 - 取第一个条件优化
    if "any_of" in when:
        conditions = when["any_of"]
        if conditions and isinstance(conditions, list):
            return _parse_gate_when_condition(conditions[0])

    if "all_of" in when:
        conditions = when["all_of"]
        if conditions and isinstance(conditions, list):
            return _parse_gate_when_condition(conditions[0])

    # 直接格式: {feature_name: {value_lt/value_gt/quantile_lt/quantile_gt: threshold}}
    for feature_col, value_dict in when.items():
        if feature_col in ("any_of", "all_of"):
            continue

        if not isinstance(value_dict, dict):
            continue

        # 解析操作符和阈值
        for op_key, threshold in value_dict.items():
            # value_lt, value_gt, value_lte, value_gte
            if op_key.startswith("value_"):
                op_suffix = op_key[6:]  # 去掉 "value_" 前缀
                operator = op_suffix.replace("lte", "le").replace("gte", "ge")
                return feature_col, operator, float(threshold)

            # quantile_lt, quantile_gt
            # ❗ 问题 6 确认: quantile_* 语义假设
            # 当前实现假设 feature 已经是预计算的 quantile score (0-1)
            # 即 quantile_gt 0.8 意思是 "feature_value > 0.8"
            # 而不是 "feature_value > df[feature].quantile(0.8)"
            # 如果 feature 是原始值，需要在特征工程阶段先转为 quantile
            if op_key.startswith("quantile_"):
                op_suffix = op_key[9:]  # 去掉 "quantile_" 前缀
                operator = op_suffix.replace("lte", "le").replace("gte", "ge")
                return feature_col, operator, float(threshold)

    return None, None, None


def optimize_gate_rule_unified(
    df: pd.DataFrame,
    rule: GateRule,
    label_col: str = "is_good",
    config: Optional[UnifiedOptimizationConfig] = None,
    step: float = 0.05,
    rr_col: Optional[str] = None,
    strategy: Optional[str] = None,
    allowed_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    统一的门控规则优化函数

    Args:
        df: 包含特征和标签的 DataFrame
        rule: GateRule 对象
        label_col: 标签列名
        config: 优化配置
        step: 阈值扫描步长
        rr_col: 连续收益列名 (e.g. forward_rr / bpc_impulse_return_atr).
            仅当 config.require_positive_effect=True 时使用,
            用于剔除 mean_rr(allow) <= mean_rr(deny) 的病态 gate.
        strategy: 策略名; 用于读取 features_gate.yaml 的白名单 (覆盖特征层
            deny 允许清单). 仅当 allowed_features 未显式传入时使用.
        allowed_features: 显式传入的白名单 fnmatch patterns; 空 = 允许所有.

    Returns:
        优化结果
    """
    if config is None:
        config = UnifiedOptimizationConfig()

    # 从规则的 when 条件中提取特征和运算符
    when = rule.when
    feature_col, operator, current_threshold = _parse_gate_when_condition(when)

    if feature_col is None or operator is None:
        return {
            "rule_id": rule.id,
            "status": "skip",
            "reason": f"Could not parse rule when condition: {when}",
        }

    # 检查特征是否存在
    if feature_col not in df.columns:
        return {
            "rule_id": rule.id,
            "feature": feature_col,
            "status": "skip",
            "reason": f"Feature {feature_col} not found in DataFrame",
        }

    # features_gate.yaml 白名单守门:
    # locked / frozen / promote_never_disable 规则是用户显式决策, 永远保留;
    # 其余规则只有在特征命中策略白名单时才允许进入优化循环.
    _is_protected_user_rule = bool(
        getattr(rule, "locked", False)
        or getattr(rule, "frozen", False)
        or getattr(rule, "promote_never_disable", False)
    )
    if not _is_protected_user_rule:
        _whitelist = (
            allowed_features
            if allowed_features is not None
            else _load_allowed_gate_deny_features_for_strategy(strategy)
        )
        if _whitelist and not _is_feature_allowed_for_gate_deny(
            feature_col, _whitelist
        ):
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "status": "skip",
                "reason": (
                    f"Feature {feature_col} not in features_gate.yaml "
                    f"allowed_gate_deny_features (strategy={strategy!r}); "
                    "gate is execution-layer only."
                ),
            }

    # 确定阈值范围
    # 对于 quantile 特征，范围是 [0, 1]
    if "_pct" in feature_col or "quantile" in feature_col:
        threshold_range = (0.05, 0.95)  # 避免边界值
    else:
        # 使用数据分位数确定范围
        q_low = df[feature_col].quantile(0.05)
        q_high = df[feature_col].quantile(0.95)
        threshold_range = (q_low, q_high)
        step = max(step, (q_high - q_low) / 20)  # 自适应步长

    # 扫描阈值
    results = scan_thresholds_for_lift(
        df, feature_col, operator, threshold_range, step, label_col
    )

    # 先进行lift筛选（可行性检查）
    valid_results = [
        r
        for r in results
        if r["lift"] >= config.min_lift
        and config.min_pass_rate <= r["pass_rate_all"] <= config.max_pass_rate
        and r.get("n_good", 0) >= config.min_samples_good
        and r.get("n_bad", 0) >= config.min_samples_bad
    ]

    # ── require_positive_effect: 连续收益口径补护栏 ──
    # binary lift > 0 并不等价于 mean_rr(allow) > mean_rr(deny), 因为 range_deny
    # 形式的规则可能把中腰砍掉而留下两端尾部 (尾部 good 比例高但亏损大). 当配置
    # require_positive_effect 时, 额外要求候选阈值的连续 effect >= -tol.
    if config.require_positive_effect and valid_results:
        if rr_col is None or rr_col not in df.columns:
            # rr_col 不可用时直接 skip 本护栏, 打印一次警告 (用全局 flag 避免刷屏)
            global _REQUIRE_POS_EFFECT_WARNED
            try:
                _REQUIRE_POS_EFFECT_WARNED
            except NameError:
                _REQUIRE_POS_EFFECT_WARNED = False
            if not _REQUIRE_POS_EFFECT_WARNED:
                print(
                    f"    ⚠️ require_positive_effect=True 但 rr_col={rr_col!r} 不可用, 本次 skip 此护栏"
                )
                _REQUIRE_POS_EFFECT_WARNED = True
        else:
            _rr_vals = df[rr_col].to_numpy(dtype=float, copy=False)
            _feat_vals = df[feature_col].to_numpy(dtype=float, copy=False)
            _feat_valid = ~np.isnan(_feat_vals)
            _kept: list = []
            _skipped_neg_effect = 0
            for r in valid_results:
                _th = r.get("threshold")
                if _th is None or not np.isfinite(_th):
                    _kept.append(r)
                    continue
                if operator == "lt":
                    _deny = _feat_valid & (_feat_vals < _th)
                elif operator == "le":
                    _deny = _feat_valid & (_feat_vals <= _th)
                elif operator == "gt":
                    _deny = _feat_valid & (_feat_vals > _th)
                elif operator == "ge":
                    _deny = _feat_valid & (_feat_vals >= _th)
                else:
                    _kept.append(r)
                    continue
                _allow = _feat_valid & ~_deny
                if not _deny.any() or not _allow.any():
                    _kept.append(r)
                    continue
                _rr_allow_mean = float(np.nanmean(_rr_vals[_allow]))
                _rr_deny_mean = float(np.nanmean(_rr_vals[_deny]))
                _effect = _rr_allow_mean - _rr_deny_mean
                r["mean_rr_allow"] = _rr_allow_mean
                r["mean_rr_deny"] = _rr_deny_mean
                r["mean_effect"] = _effect
                if _effect < -float(config.positive_effect_tol):
                    _skipped_neg_effect += 1
                    continue
                _kept.append(r)
            if _skipped_neg_effect > 0:
                print(
                    f"    🛡️  require_positive_effect: 跳过 {_skipped_neg_effect} 条 mean effect<=0 候选 "
                    f"(feat={feature_col}, rr={rr_col})"
                )
            valid_results = _kept

    if not valid_results:
        # =========================================================================
        # Hard Gate NaN Lift 例外通道
        # 语义：Hard Gate 的合法性来源是"结构稳定性 + 执行风险厌恶"，不是 lift
        # 只有同时满足所有严格条件，才允许 lift=NaN 的 Hard Gate 进入 robustness 仲裁
        # =========================================================================
        if config.allow_hard_nan_lift and rule.tag and "gate_" in rule.id:
            # 检查是否有符合 NaN lift 例外条件的阈值
            nan_lift_candidates = []
            for r in results:
                # 条件 1: lift 是 NaN（pass_rate_bad 极低）
                if not np.isfinite(r.get("lift", 0)):
                    # 条件 2: pass_rate_bad 必须极低
                    if r.get("pass_rate_bad", 1.0) > config.nan_lift_max_pass_rate_bad:
                        continue
                    # 条件 3: pass_rate_good 必须足够高
                    if r.get("pass_rate_good", 0) < config.nan_lift_min_pass_rate_good:
                        continue
                    # 条件 4: 覆盖率必须足够
                    n_valid = r.get("n_valid", 0)
                    n_all = len(df)
                    coverage = n_valid / n_all if n_all > 0 else 0
                    if coverage < config.nan_lift_min_coverage:
                        continue
                    nan_lift_candidates.append(r)

            if nan_lift_candidates:
                # 在候选中找 robustness 最高的
                best_nan_result = None
                best_nan_robustness = -1

                for r in nan_lift_candidates:
                    robustness = compute_robustness_score(
                        df, feature_col, operator, r["threshold"], label_col, config
                    )
                    # 条件 5: robustness 必须足够高
                    if robustness.overall_score >= config.nan_lift_min_robustness:
                        if robustness.overall_score > best_nan_robustness:
                            best_nan_robustness = robustness.overall_score
                            best_nan_result = {
                                **r,
                                "robustness_score": robustness.to_dict(),
                                "recommended_threshold": r["threshold"],
                                "recommended_threshold_type": "robust_but_unproven",
                            }

                if best_nan_result:
                    # 条件 6: 检查阈值区间宽度（排除点估计）
                    # 找到所有通过条件的阈值范围
                    valid_thresholds = [
                        r["threshold"]
                        for r in nan_lift_candidates
                        if compute_robustness_score(
                            df, feature_col, operator, r["threshold"], label_col, config
                        ).overall_score
                        >= config.nan_lift_min_robustness
                    ]
                    if len(valid_thresholds) >= 2:
                        interval_width = max(valid_thresholds) - min(valid_thresholds)
                        if interval_width >= config.nan_lift_min_plateau_width:
                            return {
                                "rule_id": rule.id,
                                "feature": feature_col,
                                "operator": operator,
                                "current_threshold": current_threshold,
                                "status": "robust_but_unproven",  # 明确标记语义
                                "eligibility": "deny_only",  # 只能作为 deny-only safety gate
                                "robustness_selection": True,
                                "nan_lift_exception": True,
                                "nan_lift_reason": "Hard Gate with pass_rate_bad < 1%, structural stability validated",
                                "interval_width": interval_width,
                                **best_nan_result,
                                "scan_results": results,
                            }

        return {
            "rule_id": rule.id,
            "feature": feature_col,
            "status": "no_valid_threshold",
            "reason": "No threshold meets basic lift/pass_rate requirements",
            "scan_results": results,
        }

    # 寻找稳定的平台区间
    # ❗ Bug fix: 必须传入 valid_results，不能在不可执行域上建立 plateau
    # ❗ 问题 4 修复: 传入实际步长，保证连续性判断一致
    stable_plateau = find_stable_lift_plateau(valid_results, config, actual_step=step)

    if stable_plateau is None:
        # ❗ 设计决策点：Hard gate 没有 plateau 时是否允许 fallback
        # strict_hard=True: 不允许 fallback，Hard gate 必须有结构支撑
        # strict_hard=False: 允许 fallback 到 robustness 单点（过渡态）
        if config.strict_hard:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "operator": operator,
                "current_threshold": current_threshold,
                "status": "no_stable_plateau_strict",
                "reason": "Hard gate requires stable plateau (strict mode enabled)",
                "scan_results": results,
            }

        # 找不到稳定平台，使用robustness导向的选择策略
        # 在满足基础条件的阈值中选择robustness最高的
        best_result = None
        best_robustness_score = -1

        for r in valid_results:
            robustness = compute_robustness_score(
                df, feature_col, operator, r["threshold"], label_col, config
            )
            if robustness.overall_score > best_robustness_score:
                best_robustness_score = robustness.overall_score
                best_result = {
                    **r,
                    "robustness_score": robustness.to_dict(),
                    "recommended_threshold": r["threshold"],
                    "recommended_threshold_type": "robust_fallback",  # ❗ Bug fix #3
                }

        if best_result:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "operator": operator,
                "current_threshold": current_threshold,
                "status": "no_stable_plateau",
                "robustness_selection": True,
                "fallback_warning": "Hard gate (weak mode): robustness fallback enabled",
                **best_result,
                "scan_results": results,
            }
        else:
            return {
                "rule_id": rule.id,
                "feature": feature_col,
                "status": "no_robust_threshold",
                "reason": "No robust threshold found even with basic requirements met",
                "scan_results": results,
            }

    # 在稳定平台内选择最稳健的阈值
    plateau_results = stable_plateau["interval_details"]
    best_result_in_plateau = None
    best_robustness_in_plateau = -1

    for r in plateau_results:
        robustness = compute_robustness_score(
            df, feature_col, operator, r["threshold"], label_col, config
        )
        if robustness.overall_score > best_robustness_in_plateau:
            best_robustness_in_plateau = robustness.overall_score
            best_result_in_plateau = {
                **r,
                "robustness_score": robustness.to_dict(),
                "recommended_threshold": r["threshold"],
                "recommended_threshold_type": "plateau_best",  # ❗ Bug fix #3
            }

    # 如果平台内的最佳点与中位数不同，提供选择依据
    if best_result_in_plateau["threshold"] != stable_plateau["plateau_mid"]:
        # 比较两者，选择robustness更好的
        # 找到mid附近的结果（使用更宽松的匹配）
        mid_candidates = [
            r
            for r in plateau_results
            if abs(r["threshold"] - stable_plateau["plateau_mid"]) < step * 1.5
        ]
        if mid_candidates:
            mid_result = min(
                mid_candidates,
                key=lambda r: abs(r["threshold"] - stable_plateau["plateau_mid"]),
            )
            mid_robustness = compute_robustness_score(
                df, feature_col, operator, mid_result["threshold"], label_col, config
            )

            if best_robustness_in_plateau > mid_robustness.overall_score:
                recommended_threshold = best_result_in_plateau["threshold"]
                recommended_threshold_type = "plateau_best"  # ❗ Bug fix #3
            else:
                recommended_threshold = stable_plateau["plateau_mid"]
                recommended_threshold_type = "plateau_mid"  # ❗ Bug fix #3
                best_result_in_plateau = {
                    **mid_result,
                    "robustness_score": mid_robustness.to_dict(),
                    "recommended_threshold": stable_plateau["plateau_mid"],
                    "recommended_threshold_type": "plateau_mid",
                }
        else:
            # 如果找不到mid附近的结果，使用best_result_in_plateau
            recommended_threshold = best_result_in_plateau["threshold"]
            recommended_threshold_type = "plateau_best"  # ❗ Bug fix #3
    else:
        recommended_threshold = best_result_in_plateau["threshold"]
        recommended_threshold_type = "plateau_best"  # ❗ Bug fix #3

    return {
        "rule_id": rule.id,
        "feature": feature_col,
        "operator": operator,
        "current_threshold": current_threshold,
        "status": "stable_plateau_found",
        "robustness_selection": True,
        **stable_plateau,
        "recommended_threshold": recommended_threshold,
        "recommended_threshold_type": recommended_threshold_type,  # ❗ Bug fix #3
        "best_result_in_plateau": best_result_in_plateau,
        "scan_results": results,
    }


def _generate_html_report(
    df: pd.DataFrame,
    opt_results: Dict[str, Any],
    output_path: Path,
    label_col: str = "is_good",
) -> None:
    """生成美化的 HTML 报告"""
    from datetime import datetime

    # 计算汇总指标
    n_all = len(df)
    n_good_all = (df[label_col] == 1).sum()
    n_bad_all = n_all - n_good_all
    good_rate_all = n_good_all / n_all if n_all > 0 else 0

    # 检查是否有 gate_decision 列
    if "gate_decision" in df.columns:
        allowed = df[df["gate_decision"] == "allow"]
        n_allowed = len(allowed)
        good_rate_allowed = allowed[label_col].mean() if len(allowed) > 0 else 0
        lift = (good_rate_allowed / good_rate_all - 1) if good_rate_all > 0 else 0
        pass_rate = n_allowed / n_all if n_all > 0 else 0
        veto_df = df[df["gate_decision"] == "veto"]
        bad_rejection_rate = (
            ((veto_df[label_col] == 0).sum() / n_bad_all) if n_bad_all > 0 else 0
        )
        good_retention_rate = (
            (allowed[label_col].sum() / n_good_all) if n_good_all > 0 else 0
        )
    else:
        n_allowed = n_all
        good_rate_allowed = good_rate_all
        lift = 0
        pass_rate = 1.0
        bad_rejection_rate = 0
        good_retention_rate = 1.0
        allowed = df

    # Sharpe (额外参考)
    rr_cols = ["bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"]
    rr_col = next((c for c in rr_cols if c in df.columns), None)
    if rr_col:
        sharpe_all = df[rr_col].mean() / df[rr_col].std() if df[rr_col].std() > 0 else 0
        sharpe_allowed = (
            allowed[rr_col].mean() / allowed[rr_col].std()
            if len(allowed) > 0 and allowed[rr_col].std() > 0
            else 0
        )
    else:
        sharpe_all = 0
        sharpe_allowed = 0

    # 生成规则表格 HTML
    rules_html = ""
    for rule_id, result in opt_results.items():
        status = result.get("status", "N/A")

        if status == "stable_plateau_found":
            status_html = '<span class="status-ok">✅ 稳定平台</span>'
            threshold = result.get("recommended_threshold")
            th_str = (
                f"{threshold:.3f}" if isinstance(threshold, (int, float)) else "N/A"
            )
            lift_val = result.get("lift_at_mid", result.get("lift", 0))
            lift_html = (
                f"{lift_val*100:.2f}%" if isinstance(lift_val, (int, float)) else "N/A"
            )
            pr = result.get("pass_rate_at_mid", result.get("pass_rate_all", 0))
            pr_html = f"{pr*100:.1f}%" if isinstance(pr, (int, float)) else "N/A"
            rob = (
                result.get("best_result_in_plateau", {})
                .get("robustness_score", {})
                .get("overall_score")
            )
            if rob is None:
                rob = result.get("robustness_score", {}).get("overall_score")
            rob_html = f"{rob:.3f}" if isinstance(rob, (int, float)) else "N/A"
        elif status == "no_stable_plateau":
            status_html = '<span class="status-warn">⚠️ 无稳定平台</span>'
            threshold = result.get("recommended_threshold")
            th_str = (
                f"{threshold:.3f}" if isinstance(threshold, (int, float)) else "N/A"
            )
            lift_val = result.get("lift", 0)
            lift_html = (
                f"{lift_val*100:.2f}%" if isinstance(lift_val, (int, float)) else "N/A"
            )
            pr = result.get("pass_rate_all", 0)
            pr_html = f"{pr*100:.1f}%" if isinstance(pr, (int, float)) else "N/A"
            rob = result.get("robustness_score", {}).get("overall_score")
            rob_html = f"{rob:.3f}" if isinstance(rob, (int, float)) else "N/A"
        else:
            status_html = '<span class="status-fail">❌ 无效</span>'
            th_str = "-"
            lift_html = "-"
            pr_html = "-"
            rob_html = "-"

        rules_html += f"""
            <tr>
                <td><code>{rule_id}</code></td>
                <td>{status_html}</td>
                <td><strong>{th_str}</strong></td>
                <td>{lift_html}</td>
                <td>{pr_html}</td>
                <td>{rob_html}</td>
            </tr>"""

    # 完整 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gate 优化报告 - Execution-Robust v2</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; line-height: 1.6; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #1a73e8; margin-bottom: 30px; font-size: 28px; }}
        h2 {{ color: #34495e; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; margin: 30px 0 20px; }}
        .card {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .kpi-item {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .kpi-item.primary {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .kpi-item.warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .kpi-value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
        .kpi-label {{ font-size: 14px; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #f8f9fa; color: #2c3e50; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .status-ok {{ color: #27ae60; font-weight: bold; }}
        .status-warn {{ color: #f39c12; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        .metric-bar {{ height: 8px; background: #ecf0f1; border-radius: 4px; overflow: hidden; margin-top: 5px; }}
        .metric-fill {{ height: 100%; background: linear-gradient(90deg, #11998e, #38ef7d); border-radius: 4px; }}
        .secondary {{ color: #7f8c8d; font-size: 14px; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .timestamp {{ text-align: center; color: #95a5a6; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🎯 Gate 层效果评估报告</h1>
    <p style="text-align:center;color:#7f8c8d;margin-bottom:30px;">Execution-Robust v2</p>

    <h2>📊 核心 KPI</h2>
    <div class="card">
        <div class="kpi-grid">
            <div class="kpi-item primary">
                <div class="kpi-label">LIFT (核心指标)</div>
                <div class="kpi-value">+{lift*100:.2f}%</div>
                <div class="kpi-label">Good Rate: {good_rate_all*100:.1f}% → {good_rate_allowed*100:.1f}%</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Pass Rate</div>
                <div class="kpi-value">{pass_rate*100:.1f}%</div>
                <div class="kpi-label">{n_allowed:,} / {n_all:,} trades</div>
            </div>
            <div class="kpi-item warning">
                <div class="kpi-label">Bad Rejection Rate</div>
                <div class="kpi-value">{bad_rejection_rate*100:.1f}%</div>
                <div class="kpi-label">拒绝坏样本比例</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Good Retention Rate</div>
                <div class="kpi-value">{good_retention_rate*100:.1f}%</div>
                <div class="kpi-label">保留好样本比例</div>
            </div>
        </div>
    </div>

    <h2>🔧 Gate 规则优化结果</h2>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>规则 ID</th>
                    <th>状态</th>
                    <th>推荐阈值</th>
                    <th>Lift</th>
                    <th>Pass Rate</th>
                    <th>Robustness</th>
                </tr>
            </thead>
            <tbody>{rules_html}
            </tbody>
        </table>
    </div>

    <h2>📈 样本分布</h2>
    <div class="card">
        <table>
            <tr><th>类别</th><th>数量</th><th>比例</th><th>分布</th></tr>
            <tr>
                <td>总样本</td><td>{n_all:,}</td><td>100%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:100%"></div></div></td>
            </tr>
            <tr>
                <td>├─ Good 样本</td><td>{n_good_all:,}</td><td>{good_rate_all*100:.1f}%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:{good_rate_all*100}%"></div></div></td>
            </tr>
            <tr>
                <td>├─ Bad 样本</td><td>{n_bad_all:,}</td><td>{(1-good_rate_all)*100:.1f}%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:{(1-good_rate_all)*100}%;background:linear-gradient(90deg,#e74c3c,#c0392b)"></div></div></td>
            </tr>
            <tr>
                <td>Gate Allow</td><td>{n_allowed:,}</td><td>{pass_rate*100:.1f}%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:{pass_rate*100}%"></div></div></td>
            </tr>
            <tr>
                <td>└─ Good in Allow</td><td>{int(allowed[label_col].sum()):,}</td><td>{good_rate_allowed*100:.1f}%</td>
                <td><div class="metric-bar"><div class="metric-fill" style="width:{good_rate_allowed*100}%"></div></div></td>
            </tr>
        </table>
    </div>

    <div class="secondary">
        <h3 style="margin-bottom:10px;">📎 额外参考 (Sharpe Ratio)</h3>
        <p>基准 Sharpe (无 Gate): <strong>{sharpe_all:.4f}</strong></p>
        <p>Allow Sharpe: <strong>{sharpe_allowed:.4f}</strong></p>
        <p style="color:#95a5a6;font-size:12px;margin-top:10px;">注：Sharpe Ratio 仅作为参考指标，Gate 优化以 Lift 为核心目标</p>
    </div>

    <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# Optimization statuses that are considered "validated" for promotion
_VALID_OPT_STATUSES = {
    "stable_plateau_found",
    "no_stable_plateau",  # robustness fallback is still validated
    "nan_lift_hard_gate",  # NaN-lift exception for hard gates
}

# Prefilter operator → gate deny operator (直接映射, operator 本身已是 deny 方向)
# 分析脚本中 operator="<" 表示 deny when col < threshold,
# 因此 gate deny 条件应保持相同方向: value_lt → deny when col < threshold
_PREFILTER_OP_MAP = {
    ">=": "value_ge",  # deny when col >= X
    ">": "value_gt",  # deny when col > X
    "<=": "value_le",  # deny when col <= X
    "<": "value_lt",  # deny when col < X
}


_SYSTEM_SAFETY_KEYWORDS = (
    "evt_",
    "vol_",
    "volatility",
    "leverage",
    "garch",
    "jump_risk",
    "funding_",
    "oi_",
    "liq_",
    "liquidity",
    "tail_",
    "risk",
)


def _iter_features_in_when(when: Dict[str, Any]) -> List[str]:
    """Extract feature names recursively from gate when clauses."""
    feats: List[str] = []
    if not isinstance(when, dict):
        return feats
    if "all_of" in when and isinstance(when["all_of"], list):
        for sub in when["all_of"]:
            feats.extend(_iter_features_in_when(sub))
        return feats
    if "any_of" in when and isinstance(when["any_of"], list):
        for sub in when["any_of"]:
            feats.extend(_iter_features_in_when(sub))
        return feats
    for k, v in when.items():
        if k in ("all_of", "any_of"):
            continue
        if isinstance(v, dict):
            feats.append(str(k))
    return feats


def _is_system_safety_rule(rule: Dict[str, Any]) -> bool:
    """Heuristic phase classifier: risk/tail/volatility-like features => safety."""
    rid = str(rule.get("id", "")).lower()
    if any(kw in rid for kw in _SYSTEM_SAFETY_KEYWORDS):
        return True
    for feat in _iter_features_in_when(rule.get("when", {})):
        f = feat.lower()
        if any(kw in f for kw in _SYSTEM_SAFETY_KEYWORDS):
            return True
    return False


def _collect_features_from_when(when: Dict[str, Any]) -> List[str]:
    """Collect feature names from nested when/all_of/any_of clauses."""
    if not isinstance(when, dict):
        return []
    feats: List[str] = []
    if "all_of" in when and isinstance(when["all_of"], list):
        for sub in when["all_of"]:
            feats.extend(_collect_features_from_when(sub))
    if "any_of" in when and isinstance(when["any_of"], list):
        for sub in when["any_of"]:
            feats.extend(_collect_features_from_when(sub))
    for key, cond in when.items():
        if key in ("all_of", "any_of", "min_matches"):
            continue
        if isinstance(cond, dict):
            feats.append(str(key))
    # keep order and de-duplicate
    uniq: List[str] = []
    seen = set()
    for f in feats:
        if f in seen:
            continue
        seen.add(f)
        uniq.append(f)
    return uniq


def _build_stat_fallback_rules(
    df: pd.DataFrame,
    source_hard_gates: List[Dict[str, Any]],
    max_rules: int = 3,
    min_source_features: int = 2,
) -> List[Dict[str, Any]]:
    """Generate statistical hard-gate fallback rules from source gate features."""
    rr_candidates = [
        "forward_rr",
        "bpc_impulse_return_atr",
        "rr",
        "return_atr",
        "success_no_rr_extreme",
        "ret_mean",
    ]
    rr_col = next((c for c in rr_candidates if c in df.columns), None)
    if rr_col is None:
        print("  ⚠️  统计兜底跳过: 未找到 RR 列")
        return []

    feature_names: List[str] = []
    for rule in source_hard_gates:
        feature_names.extend(_collect_features_from_when(rule.get("when", {})))
    feature_names = [f for f in feature_names if f in df.columns]
    if not feature_names:
        print("  ⚠️  统计兜底跳过: 无可用 gate 候选特征")
        return []

    # De-duplicate while preserving source order
    seen = set()
    feature_names = [f for f in feature_names if not (f in seen or seen.add(f))]
    if len(feature_names) < min_source_features:
        print(
            f"  ⚠️  统计兜底跳过: source feature 数量不足 "
            f"({len(feature_names)} < {min_source_features})"
        )
        return []

    try:
        from scripts.export_lightgbm_rules_to_readme import (
            _generate_gate_rules_statistical,
        )

        stat_rules = _generate_gate_rules_statistical(
            df,
            feature_names=feature_names,
            rr_col_name=rr_col,
            lgbm_model=None,
            max_rules=max_rules,
        )
    except Exception as e:
        print(f"  ⚠️  统计兜底失败: {e}")
        return []

    if not stat_rules:
        return []

    op_to_key = {"<=": "value_le", "<": "value_lt", ">": "value_gt", ">=": "value_ge"}
    converted: List[Dict[str, Any]] = []
    for idx, rule_dict in enumerate(stat_rules):
        conds = rule_dict.get("conditions", [])
        if not conds:
            continue

        if len(conds) == 1:
            feat, op, thr = conds[0]
            when_clause = {feat: {op_to_key.get(op, "value_lt"): round(float(thr), 4)}}
            rid = f"gate_stat_{str(feat).replace('.', '_').lower()}"
        else:
            all_of = []
            feat_names = []
            for feat, op, thr in conds:
                all_of.append(
                    {feat: {op_to_key.get(op, "value_lt"): round(float(thr), 4)}}
                )
                feat_names.append(str(feat).replace(".", "_").lower())
            when_clause = {"all_of": all_of}
            rid = f"gate_stat_{'_'.join(feat_names)}"

        converted.append(
            {
                "id": rid,
                "tag": f"HARD_STAT_{idx+1}",
                "phase": "hard_gate",
                "priority": 20 + idx,
                "reason": "统计法兜底规则（tree gate 为空）",
                "when": when_clause,
                "then": {"action": "deny"},
                "comment": (
                    f"stat_fallback gate_score={float(rule_dict.get('gate_score', 0.0)):.3f}, "
                    f"tail_capture={float(rule_dict.get('tail_capture', 0.0)):.3f}, "
                    f"good_deny={float(rule_dict.get('good_deny_rate', 0.0)):.3f}"
                ),
            }
        )
    return converted


def _load_prefilter_as_frozen_gates(prefilter_path: Path) -> List[Dict]:
    """
    将 prefilter 规则转换为 frozen hard_gates (deny 格式)。

    确保 prefilter 条件在推理时作为 gate 的一部分执行一次，
    保障训练-推理数据分布一致性。
    frozen=true 的规则不可被优化器移除或修改阈值。
    """
    import yaml as _yaml

    if not prefilter_path.exists():
        return []

    raw = _yaml.safe_load(prefilter_path.read_text(encoding="utf-8")) or {}
    rules = raw.get("rules", [])
    if not rules:
        return []

    gate_rules: List[Dict] = []
    for rule in rules:
        if "any_of" in rule:
            # OR 规则: deny when ALL sub-conditions are NOT met
            # allow = (A OR B) → deny = (NOT A AND NOT B)
            all_of_items = []
            features_desc = []
            for sub in rule["any_of"]:
                feat = sub.get("feature", "")
                op = sub.get("operator", "")
                val = sub.get("value")
                gate_op = _PREFILTER_OP_MAP.get(op)
                if gate_op and feat and val is not None:
                    all_of_items.append({feat: {gate_op: val}})
                    features_desc.append(f"{feat}{op}{val}")

            if all_of_items:
                feats_short = "_".join(
                    "_".join(f.split("_")[:2]) for item in all_of_items for f in item
                )
                gate_rules.append(
                    {
                        "id": f"prefilter_{feats_short}",
                        "tag": f"PREFILTER_{feats_short.upper()}",
                        "phase": "hard_gate",
                        "priority": 1,
                        "reason": f"prefilter OR: {' OR '.join(features_desc)}",
                        "when": {"all_of": all_of_items},
                        "then": {"action": "deny"},
                        "frozen": True,
                        "comment": "prefilter条件 (训练-推理一致性, frozen=true)",
                    }
                )
        else:
            # 简单 AND 规则 → 每条转为单独的 frozen hard_gate
            feat = rule.get("feature", "")
            op = rule.get("operator", "")
            val = rule.get("value")
            gate_op = _PREFILTER_OP_MAP.get(op)

            if gate_op and feat and val is not None:
                gate_rules.append(
                    {
                        "id": f"prefilter_{feat}",
                        "tag": f"PREFILTER_{feat.upper()}",
                        "phase": "hard_gate",
                        "priority": 1,
                        "reason": f"prefilter: {feat} {op} {val}",
                        "when": {feat: {gate_op: val}},
                        "then": {"action": "deny"},
                        "frozen": True,
                        "comment": "prefilter条件 (训练-推理一致性, frozen=true)",
                    }
                )

    return gate_rules


def _promote_gate_to_archetypes(
    strategy: str,
    strategies_root: str,
    arch: "StrategyArchetype",
    optimization_results: Dict[str, Any],
    source_gate_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    min_combined_pass_rate: float = 0.05,
    stat_fallback_on_empty: bool = True,
    stat_fallback_max_rules: int = 3,
    stat_fallback_min_source_features: int = 2,
    max_hard_gates: Optional[int] = 2,
    max_system_safety: Optional[int] = 2,
) -> None:
    """
    将优化后的 gate 规则写入 archetypes/gate.yaml。

    读取源 gate YAML (草稿或现有 gate.yaml)，用优化结果更新阈值，
    写入 archetypes/gate.yaml。

    关键行为:
      - 优化失败的规则 (no_valid_threshold/skip) 会被移除
      - locked 且优化失败 → 默认 disabled=true；若规则带 promote_never_disable=true
        则保留启用并沿用 YAML 阈值（与 entry 的 promote_never_disable 语义对齐）
      - 累积 AND pass rate 过低时自动裁剪最弱规则 (防止全部 veto)
      - 在 phase 分裂之前按 phase 做 **Top-N cap**:
          max_hard_gates / max_system_safety 分别限制两个 phase 的规则数。
          locked / frozen / promote_never_disable 规则必保 (不占额度也不受 cap);
          非 locked 规则按 (robustness_score.overall_score, |lift|) 排序取 top-K,
          K = max(0, max_N - len(locked_in_phase))。超出的规则被丢弃并记入 removed。
          传 None 表示不做 phase cap (旧行为)。
    """
    import yaml

    root = Path(strategies_root)
    arch_dir = root / strategy / "archetypes"
    target_path = arch_dir / "gate.yaml"

    # ── 语义锁定通过 gate 规则的 frozen: true 字段实现 ──
    # prefilter.yaml 中 locked: true 的规则会被 _load_prefilter_as_frozen_gates()
    # 转换为 frozen: true 的 hard_gate, 优化器对 frozen 规则跳过阈值优化 (opt=None),
    # 从而在下方 "if not opt: kept_rules.append(rule)" 路径被自动保留, 无需 meta.yaml.

    # 读取源 YAML (草稿或现有 gate.yaml)
    if source_gate_path:
        source = Path(source_gate_path)
    else:
        source = target_path

    if not source.exists():
        print(f"\u26a0\ufe0f  Cannot promote: source gate not found: {source}")
        return

    raw_text = source.read_text(encoding="utf-8")
    config = yaml.safe_load(raw_text) or {}

    # ── Filter hard_gates: only keep rules that passed optimization ──
    # locked 规则永不删除：优化失败 → 默认 disabled: true；
    # promote_never_disable → 不 disabled，保留 YAML 阈值（仍可对非 frozen 规则做阈值优化）
    hard_gates = config.get("hard_gates", [])
    kept_rules = []
    removed_rules = []
    updated_count = 0

    for rule in hard_gates:
        rule_id = rule.get("id", "")
        if bool(rule.get("disabled", False)):
            # Manual disabled rules are governance decisions, not optimization failures.
            # Keep them in the promoted YAML for traceability, but never revive them
            # via threshold optimization or promote_never_disable.
            kept_rules.append(rule)
            continue
        is_locked = bool(rule.get("locked", False))
        never_disable = bool(rule.get("promote_never_disable"))
        opt = optimization_results.get(rule_id)

        if not opt:
            # No optimization result for this rule → keep as-is (e.g. frozen/locked)
            kept_rules.append(rule)
            continue

        status = opt.get("status", "")
        # 优化器对 frozen 规则跳过调参，结果里 status=frozen；必须原样保留 YAML，
        # 不能走下方「locked 优化失败 → disabled」分支（否则会误伤语义锁定规则）。
        if status == "frozen" or bool(rule.get("frozen")):
            kept_rules.append(rule)
            continue

        rec = opt.get("recommended_threshold")

        if status not in _VALID_OPT_STATUSES or rec is None:
            if never_disable:
                rule.pop("disabled", None)
                rule.pop("disabled_reason", None)
                reason_txt = opt.get("reason", "no valid threshold")
                sticky = (
                    f"promote_never_disable: optimizer {status or 'failed'}, "
                    f"kept YAML threshold ({reason_txt})"
                )
                prev = rule.get("comment")
                rule["comment"] = f"{prev}; {sticky}" if prev else sticky
                kept_rules.append(rule)
                print(
                    f"  🔒 Rule {rule_id}: promote_never_disable → "
                    f"kept active, YAML threshold (not disabled)"
                )
            elif is_locked:
                # Locked 规则优化失败 → disabled 但保留
                rule["disabled"] = True
                rule["disabled_reason"] = (
                    f"optimization_{status or 'failed'}: {opt.get('reason', 'no valid threshold')}"
                )
                kept_rules.append(rule)
                print(
                    f"  🔒 Locked rule {rule_id}: optimization failed → disabled=true"
                )
            else:
                removed_rules.append(
                    {
                        "id": rule_id,
                        "status": status,
                        "reason": opt.get("reason", "unknown"),
                    }
                )
            continue

        # Update threshold from optimization result
        when = rule.get("when", {})
        for feature, conditions in when.items():
            if isinstance(conditions, dict):
                for cond_key in list(conditions.keys()):
                    if cond_key.startswith("value_"):
                        conditions[cond_key] = round(rec, 4)
                        updated_count += 1

        # Add optimization metadata to comment
        lift = opt.get("lift_at_mid", opt.get("lift"))
        rule["comment"] = (
            f"optimizer: {status}, " f"threshold={rec:.4f}, " f"lift={lift:.3f}"
            if isinstance(lift, (int, float))
            else f"optimizer: {status}"
        )
        # Locked / promote_never_disable 规则优化成功 → 确保 disabled 被清除
        if is_locked or never_disable:
            rule.pop("disabled", None)
            rule.pop("disabled_reason", None)
        kept_rules.append(rule)

    # ── Locked rules from system_safety: 保留不可删除 ──
    for rule in config.get("system_safety") or []:
        if isinstance(rule, dict) and rule.get("locked"):
            rule_id = rule.get("id", "")
            if rule_id and rule_id not in {r.get("id") for r in kept_rules}:
                kept_rules.append(rule)

    # ── prefilter 不再注入 gate (prefilter 和 gate 职责分离) ──
    # prefilter 只在训练时正向过滤数据, 不 promote 到 gate.yaml
    prefilter_gates = []  # 空列表, 不注入

    # 去重: 同 id 只保留第一条 (防止 gate_draft 中同一特征多次分裂导致重复)
    _seen_ids: set = set()
    deduped_rules: list = []
    for rule in kept_rules:
        rid = rule.get("id", "")
        if rid in _seen_ids:
            removed_rules.append(
                {"id": rid, "status": "duplicate", "reason": f"duplicate of {rid}"}
            )
            continue
        _seen_ids.add(rid)
        deduped_rules.append(rule)
    kept_rules = deduped_rules
    all_rules = prefilter_gates + kept_rules

    # ── 累积 AND pass rate 模拟: 防止多条规则组合后 pass rate 过低 ──
    if df is not None and all_rules and min_combined_pass_rate > 0:
        import operator as op_module

        _GATE_OPS = {
            "value_lt": op_module.lt,
            "value_le": op_module.le,
            "value_gt": op_module.gt,
            "value_ge": op_module.ge,
        }

        def _apply_when_to_mask(when, data, allow_mask, gate_ops):
            """Apply a when clause to allow_mask. Handles simple + all_of."""
            if "all_of" in when:
                # all_of: deny when ALL sub-conditions match (AND)
                compound_deny = pd.Series(True, index=data.index)
                for sub in when["all_of"]:
                    if isinstance(sub, dict):
                        for feat, conds in sub.items():
                            if isinstance(conds, dict):
                                for ck, th in conds.items():
                                    op_func = gate_ops.get(ck)
                                    if op_func and feat in data.columns:
                                        compound_deny &= op_func(data[feat], th)
                allow_mask &= ~compound_deny
            else:
                for feature, conditions in when.items():
                    if isinstance(conditions, dict):
                        for cond_key, threshold in conditions.items():
                            op_func = gate_ops.get(cond_key)
                            if op_func and feature in data.columns:
                                deny_mask = op_func(data[feature], threshold)
                                allow_mask &= ~deny_mask
            return allow_mask

        def _simulate_combined_pass_rate(rules, data):
            """Simulate cumulative AND pass rate for all rules (incl. prefilter)."""
            allow_mask = pd.Series(True, index=data.index)
            for rule in rules:
                when = rule.get("when", {})
                allow_mask = _apply_when_to_mask(when, data, allow_mask, _GATE_OPS)
            n_allow = allow_mask.sum()
            return n_allow / len(data) if len(data) > 0 else 0.0

        combined_rate = _simulate_combined_pass_rate(all_rules, df)
        n_allow = int(combined_rate * len(df))
        n_prefilter = len(prefilter_gates)
        n_opt = len(kept_rules)
        print(
            f"\n  📊 累积 AND pass rate: {combined_rate:.1%} "
            f"({n_allow}/{len(df)}, {n_prefilter} prefilter + {n_opt} optimized)"
        )

        # ── Fallback: pass-rate 裁剪 ──
        if combined_rate < min_combined_pass_rate:
            print(
                f"  ⚠️  累积 pass rate {combined_rate:.1%} < "
                f"下限 {min_combined_pass_rate:.0%}"
            )

            # Build priority for each rule by lift
            print(f"  🔧 按 lift 裁剪...")
            # BUGFIX: 必须把 locked / frozen / promote_never_disable 规则分离出来 —
            # 它们不能被裁剪, 但在 AND 累积模拟中必须始终参与, 最终也必须保留在 kept_rules 里。
            # 旧代码 `kept_rules = remaining_opt` 会把这些不可裁的规则整批丢弃。
            locked_kept: List[Dict[str, Any]] = []
            prunable = []
            for rule in kept_rules:
                if (
                    rule.get("frozen")
                    or rule.get("locked")
                    or rule.get("promote_never_disable")
                ):
                    locked_kept.append(rule)
                    continue
                rule_id = rule.get("id", "")
                opt = optimization_results.get(rule_id, {})
                lift = opt.get("lift_at_mid", opt.get("lift", 0))
                lv = lift if isinstance(lift, (int, float)) else 0

                # Lower = remove first (low lift)
                prune_priority = abs(lv)
                prunable.append((rule, prune_priority))
            prunable.sort(key=lambda x: x[1])

            remaining_opt = [r for r, _ in prunable]
            pruned_ids = []

            while remaining_opt and combined_rate < min_combined_pass_rate:
                weakest = remaining_opt.pop(0)
                wid = weakest.get("id", "unknown")
                wpri = next((pv for r, pv in prunable if r is weakest), 0)
                pruned_ids.append(wid)
                # 模拟时必须包含 locked_kept, 否则会低估被锁定规则的过滤影响,
                # 导致 pass rate 虚高, 进而少裁或误判 "不需要裁剪"。
                test_rules = prefilter_gates + locked_kept + remaining_opt
                combined_rate = _simulate_combined_pass_rate(test_rules, df)
                print(
                    f"    ✂️  移除 {wid} (priority={wpri:.3f}) "
                    f"→ pass rate={combined_rate:.1%}"
                )

            # 关键: locked_kept 永远保留, 追加在 remaining_opt 之前,
            # 让最终 gate.yaml 里 locked 规则排在统计规则之前 (phase=hard_gate, 先评估)。
            kept_rules = locked_kept + remaining_opt

            # ── Phase 2: frozen prefilter gates ──
            if combined_rate < min_combined_pass_rate and len(prefilter_gates) > 1:
                print(
                    f"  🔧 Phase 2: optimized 裁完仍 "
                    f"{combined_rate:.1%}, "
                    f"裁剪 frozen prefilter gates "
                    f"(保留最强 1 条)..."
                )
                pf_prunable = list(reversed(prefilter_gates[1:]))
                for pf_rule in pf_prunable:
                    if combined_rate >= min_combined_pass_rate:
                        break
                    pfid = pf_rule.get("id", "unknown")
                    prefilter_gates = [g for g in prefilter_gates if g is not pf_rule]
                    pruned_ids.append(pfid)
                    test_rules = prefilter_gates + kept_rules
                    combined_rate = _simulate_combined_pass_rate(test_rules, df)
                    print(
                        f"    ✂️  移除 frozen {pfid} "
                        f"→ pass rate={combined_rate:.1%} "
                        f"(剩余 {len(prefilter_gates)} prefilter)"
                    )

            all_rules = prefilter_gates + kept_rules

            if pruned_ids:
                removed_rules.extend(
                    {
                        "id": rid,
                        "status": "cumulative_pruned",
                        "reason": (f"AND pass rate " f"< {min_combined_pass_rate:.0%}"),
                    }
                    for rid in pruned_ids
                )
                print(
                    f"  ✅ 裁剪后: {len(prefilter_gates)} prefilter"
                    f" + {len(kept_rules)} optimized, "
                    f"pass rate={combined_rate:.1%}"
                )
        else:
            print(
                f"  ✅ 累积 pass rate {combined_rate:.1%} >= "
                f"下限 {min_combined_pass_rate:.0%}, 无需裁剪"
            )

    # Report removed rules
    if removed_rules:
        print(
            f"\n  \u26a0\ufe0f  {len(removed_rules)} 条规则优化失败, 已从 gate.yaml 移除:"
        )
        for rm in removed_rules:
            print(f"     - {rm['id']}: {rm['status']} ({rm['reason']})")

    # Count active (non-disabled) rules for stat_fallback decision
    _active_rules = [r for r in kept_rules if not r.get("disabled")]
    if not _active_rules:
        if not stat_fallback_on_empty:
            print("\n  ℹ️  hard_gates 为空且已禁用统计兜底，跳过 fallback")
        else:
            print(f"\n  ⚠️  所有树规则优化失败，尝试统计法兜底生成 gate ...")
            if df is not None:
                stat_fallback_rules = _build_stat_fallback_rules(
                    df=df,
                    source_hard_gates=hard_gates,
                    max_rules=stat_fallback_max_rules,
                    min_source_features=stat_fallback_min_source_features,
                )
                if stat_fallback_rules:
                    kept_rules.extend(stat_fallback_rules)
                    print(
                        f"  ✅ 统计兜底成功: 生成 {len(stat_fallback_rules)} 条 hard_gate"
                    )
                else:
                    print(
                        f"  ❌ 统计兜底未生成规则, gate.yaml 将只含 guardrails/system_safety"
                    )
            else:
                print(
                    f"  ❌ 无 DataFrame，无法执行统计兜底, gate.yaml 将只含 guardrails/system_safety"
                )

    # ── 单文件 phase 化: system_safety vs hard_gate ──
    safety_rules: List[Dict[str, Any]] = []
    archetype_rules: List[Dict[str, Any]] = []
    for rule in kept_rules:
        rule_phase = "system_safety" if _is_system_safety_rule(rule) else "hard_gate"
        rule["phase"] = rule_phase
        if rule_phase == "system_safety":
            safety_rules.append(rule)
        else:
            archetype_rules.append(rule)

    # ── Top-N cap per phase (locked / frozen / promote_never_disable 必保，不占额度) ──
    # 动机: 不加此守门时 slow pipeline 会把 gate_draft 里所有统计发现的规则
    #       (通常 5~9 条) 一起 promote 到 gate.yaml，导致 base config 的 2+2 精简
    #       在一次 pipeline 后就失效。参见 20260413_144115 vs 20260421_174335 对比
    #       (2024-04~06 同期 +1086R → -157R) 的诊断。
    def _cap_rules_with_locked_priority(
        rules: List[Dict[str, Any]],
        limit: Optional[int],
        phase_label: str,
    ) -> List[Dict[str, Any]]:
        if limit is None or len(rules) <= limit:
            return rules

        def _is_protected(r: Dict[str, Any]) -> bool:
            return bool(
                r.get("locked") or r.get("frozen") or r.get("promote_never_disable")
            )

        protected = [r for r in rules if _is_protected(r)]
        candidates = [r for r in rules if not _is_protected(r)]

        if len(protected) >= limit:
            if len(protected) > limit:
                print(
                    f"  ⚠️  {phase_label}: protected(locked/frozen)={len(protected)} "
                    f"> max={limit}；保留全部 protected 并丢弃 {len(candidates)} 条非 locked 候选"
                )
            for dropped in candidates:
                removed_rules.append(
                    {
                        "id": dropped.get("id", "unknown"),
                        "status": "cap_exceeded",
                        "reason": f"{phase_label} cap={limit}; all slots filled by locked rules",
                    }
                )
            return protected

        slots_for_candidates = limit - len(protected)

        def _rank_key(r: Dict[str, Any]) -> float:
            rid = r.get("id", "")
            opt = optimization_results.get(rid, {}) or {}
            rob = opt.get("robustness_score")
            if isinstance(rob, dict):
                rob_score = rob.get("overall_score")
                if isinstance(rob_score, (int, float)):
                    return float(rob_score) * 10.0
            lift = opt.get("lift_at_mid", opt.get("lift", 0))
            if isinstance(lift, (int, float)):
                return float(abs(lift))
            return 0.0

        candidates_sorted = sorted(candidates, key=_rank_key, reverse=True)
        kept_cand = candidates_sorted[:slots_for_candidates]
        dropped_cand = candidates_sorted[slots_for_candidates:]
        for dropped in dropped_cand:
            removed_rules.append(
                {
                    "id": dropped.get("id", "unknown"),
                    "status": "cap_exceeded",
                    "reason": (
                        f"{phase_label} cap={limit}; outranked by top-{slots_for_candidates} "
                        f"non-locked (rank_key={_rank_key(dropped):.3f})"
                    ),
                }
            )
        if dropped_cand:
            ids = [d.get("id", "?") for d in dropped_cand]
            print(
                f"  ✂️  {phase_label} top-N cap: protected={len(protected)} + "
                f"top-{len(kept_cand)} non-locked kept, dropped {len(dropped_cand)}: {ids}"
            )
        return protected + kept_cand

    archetype_rules = _cap_rules_with_locked_priority(
        archetype_rules, max_hard_gates, "hard_gates"
    )
    safety_rules = _cap_rules_with_locked_priority(
        safety_rules, max_system_safety, "system_safety"
    )

    config["system_safety"] = safety_rules
    config["hard_gates"] = archetype_rules

    # 写入 archetypes/gate.yaml
    n_total = len(hard_gates)
    header = (
        f"# {strategy.upper()} Gate (optimized, auto-promoted)\n"
        f"# 来源: {source}\n"
        f"# 优化规则: {updated_count}/{n_total} 条通过优化"
        f"{f', {len(removed_rules)} 条已移除' if removed_rules else ''}\n"
        f"# Phase split: system_safety={len(safety_rules)}, hard_gate={len(archetype_rules)}\n\n"
    )
    yaml_content = yaml.dump(
        config,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )
    target_path.write_text(header + yaml_content, encoding="utf-8")
    print(
        f"\n\U0001f4e6 Promoted to {target_path} ({updated_count} thresholds updated, "
        f"{len(kept_rules)} rules kept, {len(removed_rules)} removed, "
        f"safety={len(safety_rules)}, archetype={len(archetype_rules)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified Gate Optimization - Production Grade Gate Parameter Optimizer"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy name (e.g., bpc)",
    )
    parser.add_argument(
        "--strategies-root",
        default="config/strategies",
        help="Root directory for strategy configs",
    )
    parser.add_argument(
        "--logs",
        required=True,
        help=(
            "Input parquet path. 接受两种文件 (路线 B, ABC 统一研究框架 §5): "
            "(1) features_labeled.parquet (由 `mlbot train final --prepare-only` 产生, 推荐, "
            "无需完整 pipeline run); (2) predictions.parquet (兼容旧用法). "
            "需要列: features + (forward_rr OR is_good)."
        ),
    )
    parser.add_argument(
        "--label-col",
        default="is_good",
        help="Label column name (1=good, 0=bad)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=0.10,
        help="Minimum lift requirement",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.20,
        help="Minimum pass rate",
    )
    parser.add_argument(
        "--max-pass-rate",
        type=float,
        default=0.80,
        help="Maximum pass rate",
    )
    parser.add_argument(
        "--min-plateau-width",
        type=float,
        default=0.05,
        help="Minimum plateau width",
    )
    parser.add_argument(
        "--max-lift-std-ratio",
        type=float,
        default=0.3,
        help="Maximum lift std / lift mean ratio for stability",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Threshold scan step size",
    )
    parser.add_argument(
        "--write-back-intervals",
        action="store_true",
        help="Write back interval thresholds (start, end) instead of single point",
    )
    parser.add_argument(
        "--interval-method",
        choices=["plateau", "robustness"],
        default="plateau",
        help="Method to determine intervals: plateau bounds or robustness-driven",
    )
    parser.add_argument(
        "--gate-path",
        default=None,
        help="Custom gate YAML path (e.g., config/strategies/fer/gate_draft.yaml). "
        "Default: archetypes/gate.yaml",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="After optimization, write updated gate.yaml with optimized thresholds "
        "to archetypes/gate.yaml (promote draft to production)",
    )
    parser.add_argument(
        "--prefilter",
        default=None,
        help="Prefilter YAML path. If provided, filter logs by prefilter rules "
        "before optimization (ensures plateau validation on production distribution)",
    )
    parser.add_argument(
        "--min-combined-pass-rate",
        type=float,
        default=0.05,
        metavar="RATE",
        help="累积 AND pass rate 下限 (0~1). 多条 gate 规则组合后至少要保留这个比例的 bars. "
        "如果低于这个阈值, 会按 lift 从弱到强自动裁剪规则. 默认 0.05 (5%%). "
        "由 research_pipeline.yaml kpi_gates.gate.min_combined_pass_rate 控制.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Only use data before this date for optimization (IS cutoff, avoid OOS lookahead)",
    )
    parser.add_argument(
        "--stat-fallback-on-empty",
        dest="stat_fallback_on_empty",
        action="store_true",
        help="当 hard_gates 为空时启用统计法 fallback（默认开启）",
    )
    parser.add_argument(
        "--no-stat-fallback-on-empty",
        dest="stat_fallback_on_empty",
        action="store_false",
        help="禁用 hard_gates 为空时的统计法 fallback",
    )
    parser.set_defaults(stat_fallback_on_empty=True)
    parser.add_argument(
        "--stat-fallback-max-rules",
        type=int,
        default=3,
        help="统计法 fallback 最多生成的 hard_gate 数",
    )
    parser.add_argument(
        "--stat-fallback-min-source-features",
        type=int,
        default=2,
        help="统计法 fallback 的最小 source gate 特征数",
    )
    parser.add_argument(
        "--max-hard-gates",
        type=int,
        default=None,
        metavar="N",
        help="promote 时 hard_gates phase 保留的最大规则数 (locked/frozen 必保，不占额度). "
        "超出部分按 robustness_score / |lift| 排序丢弃最弱者. "
        "默认不限制 (None) 以保持向后兼容；推荐由 research_pipeline.yaml "
        "kpi_gates.gate.max_hard_gates (如 BPC=2) 显式设置.",
    )
    parser.add_argument(
        "--max-system-safety",
        type=int,
        default=None,
        metavar="N",
        help="promote 时 system_safety phase 保留的最大规则数 (locked/frozen 必保). "
        "默认不限制 (None)；推荐由 research_pipeline.yaml "
        "kpi_gates.gate.max_system_safety 显式设置.",
    )
    parser.add_argument(
        "--require-positive-effect",
        action="store_true",
        default=False,
        help="要求每个候选阈值满足 mean_rr(allow) > mean_rr(deny), "
        "防止 range_deny 砍中腰留两尾 (lift>0 但连续 effect<=0) 的病态 gate 通过.",
    )
    parser.add_argument(
        "--positive-effect-tol",
        type=float,
        default=0.0,
        help="effect >= -tol 视为非负 (默认 0 严格禁止负 effect).",
    )
    args = parser.parse_args()

    # Load logs
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ Logs file not found: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    # ── Route B (ABC 统一研究框架 §5): 检测输入是 features_labeled 还是 predictions ──
    # features_labeled.parquet 由 `mlbot train final --prepare-only` 产生，不含模型 score；
    # 本脚本只需要 features + label/forward_rr，因此两者都可作为输入。
    _has_pred = any(c in df.columns for c in ("score", "y_pred_proba", "prediction"))
    _kind = "predictions.parquet" if _has_pred else "features_labeled.parquet"
    print(f"✅ Loaded {len(df)} rows from {logs_path}  (detected: {_kind})")
    if not _has_pred:
        print(
            "   ℹ️ 输入为 features_labeled.parquet（无模型 score），"
            "gate plateau/lift/robustness 计算无需 score。"
        )

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

    # ── Prefilter: 在生产分布上验证 plateau ──
    if args.prefilter:
        _pf_path = Path(args.prefilter)
        if _pf_path.exists():
            import yaml
            import operator as _op

            _PF_OPS = {
                ">=": _op.ge,
                ">": _op.gt,
                "<=": _op.le,
                "<": _op.lt,
                "==": _op.eq,
                "!=": _op.ne,
            }
            with open(_pf_path, "r", encoding="utf-8") as _f:
                _pf_cfg = yaml.safe_load(_f)
            _pf_rules = _pf_cfg.get("rules", []) if _pf_cfg else []
            if _pf_rules:
                _n_before = len(df)
                for _rule in _pf_rules:
                    if "any_of" in _rule:
                        # OR 组: 满足任一条件即通过
                        _or_mask = pd.Series(False, index=df.index)
                        for _sub in _rule["any_of"]:
                            _sf = _sub["feature"]
                            _sop = _PF_OPS.get(_sub["operator"])
                            if _sop and _sf in df.columns:
                                _or_mask |= _sop(df[_sf], _sub["value"])
                        df = df[_or_mask].copy()
                    else:
                        _feat = _rule.get("feature", "")
                        _op_str = _rule.get("operator", "")
                        _val = _rule.get("value")
                        _op_func = _PF_OPS.get(_op_str)
                        if _op_func and _feat in df.columns:
                            df = df[_op_func(df[_feat], _val)].copy()
                print(
                    f"🛡️  Prefilter applied: {_n_before} → {len(df)} rows "
                    f"({len(df)/_n_before:.1%} retained)"
                )
            else:
                print(f"ℹ️  Prefilter {_pf_path}: rules 为空, 不过滤")
        else:
            print(f"⚠️  Prefilter file not found: {args.prefilter}, 跳过")

    # 自动生成 is_good 列 (如果不存在)
    rr_col = None
    for candidate in ["bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"]:
        if candidate in df.columns:
            rr_col = candidate
            break

    if args.label_col not in df.columns:
        if rr_col is not None:
            # 基于 rr_extreme 标签定义: Good = RR >= -0.8, Bad = RR < -0.8
            df[args.label_col] = (df[rr_col] >= -0.8).astype(int)
            print(
                f"ℹ️ Auto-generated '{args.label_col}' column from '{rr_col}' (threshold: -0.8)"
            )
        else:
            print(f"❌ Cannot auto-generate '{args.label_col}': no RR column found")
            print(f"   Tried: bpc_impulse_return_atr, forward_rr, rr, return_atr")
            return 1

    # Check label column
    if args.label_col not in df.columns:
        print(f"❌ Label column '{args.label_col}' not found in DataFrame")
        print(f"   Available columns: {list(df.columns)[:20]}...")
        return 1

    n_good = (df[args.label_col] == 1).sum()
    n_bad = (df[args.label_col] == 0).sum()
    print(f"   Good samples: {n_good}, Bad samples: {n_bad}")
    print(f"   Good rate: {n_good/(n_good+n_bad):.3f}")

    # Create config
    config = UnifiedOptimizationConfig(
        min_lift=args.min_lift,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
        min_plateau_width=args.min_plateau_width,
        max_lift_std_ratio=args.max_lift_std_ratio,
        threshold_step=args.step,
        require_positive_effect=bool(args.require_positive_effect),
        positive_effect_tol=float(args.positive_effect_tol),
    )
    if config.require_positive_effect:
        print(
            f"🛡️  require_positive_effect enabled (tol={config.positive_effect_tol}, "
            f"rr_col={rr_col!r})"
        )

    # Run optimizations
    all_results = {}

    # Load strategy archetype
    try:
        arch = load_strategy_archetype(
            args.strategy,
            args.strategies_root,
            gate_path=args.gate_path,
        )
        print(f"✅ Loaded strategy: {arch.name}")
        print(f"   Hard gates: {len(arch.gate.hard_gates)}")

        # Process hard gates
        print("\n📋 Optimizing Hard Gates:")
        for rule in arch.gate.hard_gates:
            print(f"  Processing: {rule.id}")

            # 跳过 frozen 规则
            if getattr(rule, "frozen", False):
                print(f"    ⚠️  FROZEN: 禁止优化，保持当前阈值")
                all_results[rule.id] = {
                    "rule_id": rule.id,
                    "status": "frozen",
                    "reason": "Rule marked as frozen, threshold optimization disabled",
                }
                continue

            result = optimize_gate_rule_unified(
                df,
                rule,
                args.label_col,
                config,
                args.step,
                rr_col=rr_col,
                strategy=args.strategy,
            )
            all_results[rule.id] = result

            if result.get("status") in ["stable_plateau_found", "no_stable_plateau"]:
                rec_thresh = result.get(
                    "recommended_threshold", result.get("threshold")
                )
                lift_val = result.get("lift_at_mid", result.get("lift"))
                pass_rate = result.get("pass_rate_at_mid", result.get("pass_rate_all"))
                rob_score = result.get("robustness_score", {}).get("overall_score")

                status_msg = (
                    "✅ Stable plateau"
                    if result.get("status") == "stable_plateau_found"
                    else "⚠️ No stable plateau"
                )
                # 安全格式化，避免 None 或字符串导致错误
                th_str = (
                    f"{rec_thresh:.3f}"
                    if isinstance(rec_thresh, (int, float))
                    else str(rec_thresh)
                )
                lift_str = (
                    f"{lift_val:.3f}"
                    if isinstance(lift_val, (int, float))
                    else str(lift_val)
                )
                pr_str = (
                    f"{pass_rate:.3f}"
                    if isinstance(pass_rate, (int, float))
                    else str(pass_rate)
                )
                rob_str = (
                    f"{rob_score:.3f}"
                    if isinstance(rob_score, (int, float))
                    else str(rob_score)
                )
                print(
                    f"    {status_msg}: Threshold={th_str}, Lift={lift_str}, PassRate={pr_str}, Robustness={rob_str}"
                )
            else:
                print(f"    ⚠️  {result.get('status')}: {result.get('reason', 'N/A')}")
    except Exception as e:
        print(f"❌ Failed to load strategy '{args.strategy}': {e}")
        return 1

    # Prepare results for output
    final_results = {}
    for k, v in all_results.items():
        # Create clean result without large data
        clean_v = {kk: vv for kk, vv in v.items() if kk != "scan_results"}
        # Also remove interval_details to reduce output size
        if "interval_details" in clean_v:
            del clean_v["interval_details"]

        # Add interval information if requested
        if args.write_back_intervals:
            if v.get("status") == "stable_plateau_found":
                if args.interval_method == "plateau":
                    # Use plateau bounds
                    clean_v["threshold_interval"] = {
                        "start": v["plateau_start"],
                        "end": v["plateau_end"],
                        "method": "plateau_bounds",
                    }
                else:  # robustness method
                    # Calculate interval based on robustness considerations
                    center = v["recommended_threshold"]
                    # Use plateau width divided by 2 as buffer
                    half_width = (
                        v["plateau_width"] / 2
                        if v.get("plateau_width", 0) > 0
                        else args.step
                    )
                    clean_v["threshold_interval"] = {
                        "start": max(v["plateau_start"], center - half_width),
                        "end": min(v["plateau_end"], center + half_width),
                        "method": "robustness_centered",
                    }
            elif v.get("status") == "no_stable_plateau":
                # For non-stable cases, create small interval around recommended threshold
                center = v["recommended_threshold"]
                buffer = args.step
                clean_v["threshold_interval"] = {
                    "start": center - buffer,
                    "end": center + buffer,
                    "method": "single_point_with_buffer",
                }

        final_results[k] = clean_v

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ Results saved to: {output_path}")

    # Summary
    n_plateau = sum(
        1 for r in final_results.values() if r.get("status") == "stable_plateau_found"
    )
    n_no_plateau = sum(
        1 for r in final_results.values() if r.get("status") == "no_stable_plateau"
    )
    n_skip = sum(
        1
        for r in final_results.values()
        if r.get("status") in ("skip", "no_valid_threshold", "no_robust_threshold")
    )
    n_with_intervals = sum(
        1 for r in final_results.values() if "threshold_interval" in r
    )

    print(f"\n📊 Summary:")
    print(f"   Stable plateaus found: {n_plateau}")
    print(f"   No stable plateau (robustness selection used): {n_no_plateau}")
    print(f"   Skipped/Failed: {n_skip}")
    if args.write_back_intervals:
        print(f"   Rules with interval thresholds: {n_with_intervals}")

    # ==========================================================================
    # 生成美化的 HTML 报告
    # ==========================================================================
    html_path = output_path.with_suffix(".html")
    _generate_html_report(df, final_results, html_path, args.label_col)
    print(f"✅ HTML report saved to: {html_path}")

    # ==========================================================================
    # --promote: 将优化后的规则写入 archetypes/gate.yaml
    # ==========================================================================
    if args.promote:
        # stat_fallback 生成的规则全属 hard_gate phase (risk/tail 关键词走 system_safety
        # 时不在该路径)，因此兜底上限不得超过 hard_gates cap，避免绕过 Top-N。
        _eff_stat_fallback_max = args.stat_fallback_max_rules
        if args.max_hard_gates is not None:
            _eff_stat_fallback_max = min(
                _eff_stat_fallback_max, max(0, args.max_hard_gates)
            )
        _promote_gate_to_archetypes(
            args.strategy,
            args.strategies_root,
            arch,
            all_results,
            args.gate_path,
            df=df,
            min_combined_pass_rate=args.min_combined_pass_rate,
            stat_fallback_on_empty=args.stat_fallback_on_empty,
            stat_fallback_max_rules=_eff_stat_fallback_max,
            stat_fallback_min_source_features=args.stat_fallback_min_source_features,
            max_hard_gates=args.max_hard_gates,
            max_system_safety=args.max_system_safety,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
