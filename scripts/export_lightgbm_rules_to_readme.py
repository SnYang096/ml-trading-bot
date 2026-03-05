#!/usr/bin/env python3
"""
从固定训练结果目录中的 LightGBM model.pkl 导出树规则，追加到四个策略的 README。
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# LightGBM/sklearn 常把特征存成 Column_0, Column_1；用 used_features.json 顺序映射回真实名
COLUMN_INDEX_RE = re.compile(r"^Column_(\d+)$", re.IGNORECASE)

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = [
    "sr_reversal_rr_reg_long",
    "compression_breakout",
    "sr_breakout",
    "trend_following",
]


def _get_booster(model):
    """从 pipeline 保存的 model（可能为 list of Boosters）取第一个 Booster。"""
    if isinstance(model, list) and len(model) > 0:
        model = model[0]
    if hasattr(model, "booster_"):
        return model.booster_
    if hasattr(model, "raw_model_"):
        return model.raw_model_
    return model


def _collect_splits_from_dataframe(booster, max_splits=40):
    """用 trees_to_dataframe() 收集分裂条件（更稳定）。"""
    try:
        df = booster.trees_to_dataframe()
    except Exception:
        return []
    if df is None or df.empty:
        return []
    # 有 split 的行：split_feature, threshold
    split_df = df.dropna(subset=["split_feature", "threshold"])
    if split_df.empty:
        return []

    # 过滤掉 categorical splits（threshold 包含 '||' 的是分类特征分裂）
    split_df = split_df[
        ~split_df["threshold"].astype(str).str.contains(r"\|\|", regex=True)
    ]
    if split_df.empty:
        return []

    names = split_df["split_feature"].astype(str)
    thrs = split_df["threshold"].astype(float)
    cnt = defaultdict(int)
    for name, thr in zip(names, thrs):
        key = (name, round(float(thr), 6), "<=")
        cnt[key] += 1
    ordered = sorted(cnt.items(), key=lambda x: -x[1])[:max_splits]
    return [(name, thr, op, count) for (name, thr, op), count in ordered]


def _map_column_to_feature_name(name: str, feature_names: list) -> str:
    """将 Column_N 映射为 used_features[N] 的真实特征名。"""
    if not feature_names:
        return name
    m = COLUMN_INDEX_RE.match(str(name).strip())
    if not m:
        return name
    idx = int(m.group(1))
    if 0 <= idx < len(feature_names):
        return feature_names[idx]
    return name


def _collect_splits(booster, feature_names, max_splits=40):
    """从 LightGBM booster 收集 split 条件，返回 (feature, threshold, op, count) 列表。"""
    # 优先用 trees_to_dataframe（列名稳定）
    rules = _collect_splits_from_dataframe(booster, max_splits)
    if rules:
        # 将 Column_0, Column_1... 映射为 used_features 中的真实特征名
        rules = [
            (_map_column_to_feature_name(name, feature_names), thr, op, count)
            for name, thr, op, count in rules
        ]
        return rules
    try:
        dump = booster.dump_model()
    except Exception:
        return []
    tree_info = dump.get("tree_info") or []
    names = list(feature_names) if feature_names else []
    splits = []

    def walk(node, depth=0):
        if depth > 8:
            return
        if not isinstance(node, dict):
            return
        # 内部节点才有 split_feature（叶子只有 leaf_value 等）
        fidx = node.get("split_feature")
        thr = node.get("threshold")
        if thr is not None and fidx is not None:
            name = names[fidx] if fidx < len(names) else f"f{fidx}"
            splits.append((name, float(thr), "<="))
        left = node.get("left_child")
        right = node.get("right_child")
        if left is not None and isinstance(left, dict):
            walk(left, depth + 1)
        if right is not None and isinstance(right, dict):
            walk(right, depth + 1)

    for ti in tree_info:
        tree = ti.get("tree_structure") if isinstance(ti, dict) else ti
        if tree:
            walk(tree)

    cnt = defaultdict(int)
    for name, thr, op in splits:
        key = (name, round(thr, 6), op)
        cnt[key] += 1
    ordered = sorted(cnt.items(), key=lambda x: -x[1])[:max_splits]
    return [(name, thr, op, count) for (name, thr, op), count in ordered]


def _strip_rules_section(content: str) -> str:
    """移除已有「特征使用规则」或「树模型规则导出」小节。"""
    lines = content.split("\n")
    new_lines = []
    in_section = False
    for line in lines:
        # 检测多种可能的标题格式
        if (
            "特征使用规则" in line
            and line.strip().startswith("##")
            or "树模型规则导出" in line
            and line.strip().startswith("##")
        ):
            in_section = True
            continue
        if in_section and line.strip().startswith("##"):
            in_section = False
        if not in_section:
            new_lines.append(line)
    while new_lines and not new_lines[-1].strip():
        new_lines.pop()
    return "\n".join(new_lines)


def _write_standalone_rules(
    output_path: Path, rules: list, strategy: str, model_source: str
):
    """写入独立的规则文件（到模型目录）。"""
    lines = [
        f"# {strategy} 树模型规则导出",
        "",
        "以下为从 LightGBM 模型中提取的**高频分裂条件**（按出现次数排序）。",
        "",
        "| 特征 | 条件 | 出现次数 |",
        "|------|------|----------|",
    ]
    for name, thr, op, count in rules:
        cond = f"{name} {op} {thr:.4g}"
        lines.append(f"| `{name}` | `{cond}` | {count} |")
    lines.append("")
    lines.append(f"**模型来源**：`{model_source}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _compute_shap_gain_features(
    pred_df,
    feature_names: list,
    lgbm_model=None,
    top_n: int = 10,
    compute_interactions: bool = True,
):
    """
    SHAP ∩ Gain 特征发现 (Gate/Evidence 共用).

    Returns:
        (top_features, shap_importance_map, interaction_pairs)
    """
    import numpy as np

    avail = [f for f in feature_names if f in pred_df.columns]
    top_features = avail[:top_n]
    shap_importance_map = {}
    interaction_pairs = []

    if lgbm_model is None:
        return top_features, shap_importance_map, interaction_pairs

    try:
        booster = _get_booster(lgbm_model)
        feat_names_model = booster.feature_name()
        model_to_real = {}
        for i in range(min(len(feat_names_model), len(feature_names))):
            model_to_real[feat_names_model[i]] = feature_names[i]

        # Gain importance
        gain_importances = booster.feature_importance(importance_type="gain")
        gain_rank = {}
        for i in range(len(feat_names_model)):
            real_name = model_to_real.get(feat_names_model[i], feat_names_model[i])
            if real_name in pred_df.columns:
                gain_rank[real_name] = float(gain_importances[i])
        gain_top10 = set(sorted(gain_rank, key=gain_rank.get, reverse=True)[:10])

        # SHAP importance
        shap_top10 = set()
        try:
            import shap

            X_full = np.zeros((len(pred_df), len(feat_names_model)))
            for fi, fn in enumerate(feat_names_model):
                real_fn = model_to_real.get(fn, fn)
                if real_fn in pred_df.columns:
                    X_full[:, fi] = pred_df[real_fn].fillna(0).values.astype(float)
            sample_n = min(2000, len(X_full))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_full), size=sample_n, replace=False)
            X_sample = X_full[idx]

            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = (
                    shap_values[1] if len(shap_values) == 2 else shap_values[0]
                )
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            for fi, fn in enumerate(feat_names_model):
                real_name = model_to_real.get(fn, fn)
                if real_name in pred_df.columns and fi < len(mean_abs_shap):
                    shap_importance_map[real_name] = float(mean_abs_shap[fi])
            shap_top10 = set(
                sorted(shap_importance_map, key=shap_importance_map.get, reverse=True)[
                    :10
                ]
            )

            # SHAP interaction
            if compute_interactions:
                try:
                    interact_n = min(500, sample_n)
                    X_interact = X_sample[:interact_n]
                    interact_vals = explainer.shap_interaction_values(X_interact)
                    if isinstance(interact_vals, list):
                        interact_vals = (
                            interact_vals[1]
                            if len(interact_vals) == 2
                            else interact_vals[0]
                        )
                    mean_interact = np.abs(interact_vals).mean(axis=0)
                    np.fill_diagonal(mean_interact, 0)
                    n_feat = mean_interact.shape[0]
                    pair_scores = []
                    for fi in range(n_feat):
                        for fj in range(fi + 1, n_feat):
                            fn_i = (
                                model_to_real.get(
                                    feat_names_model[fi], feat_names_model[fi]
                                )
                                if fi < len(feat_names_model)
                                else None
                            )
                            fn_j = (
                                model_to_real.get(
                                    feat_names_model[fj], feat_names_model[fj]
                                )
                                if fj < len(feat_names_model)
                                else None
                            )
                            if (
                                fn_i
                                and fn_j
                                and fn_i in pred_df.columns
                                and fn_j in pred_df.columns
                            ):
                                score = float(mean_interact[fi, fj])
                                if score > 0.001:
                                    pair_scores.append((fn_i, fn_j, score))
                    pair_scores.sort(key=lambda x: -x[2])
                    interaction_pairs = pair_scores[:10]
                    if interaction_pairs:
                        top_pair = interaction_pairs[0]
                        print(
                            f"   📊 SHAP interaction top pair: "
                            f"{top_pair[0]} × {top_pair[1]} = {top_pair[2]:.4f}"
                        )
                except Exception as e:
                    print(f"   ⚠️  SHAP interaction 计算跳过: {e}")

        except ImportError:
            print("   ⚠️  shap 未安装，使用 gain importance fallback")
        except Exception as e:
            print(f"   ⚠️  SHAP 计算失败: {e}，使用 gain importance fallback")

        # SHAP ∩ Gain 交集
        if shap_top10:
            intersection = shap_top10 & gain_top10
            if len(intersection) >= 3:
                top_features = sorted(
                    intersection,
                    key=lambda f: shap_importance_map.get(f, 0),
                    reverse=True,
                )[:top_n]
                print(
                    f"   📊 SHAP∩Gain: {len(intersection)} 特征 "
                    f"(SHAP top10={len(shap_top10)}, Gain top10={len(gain_top10)})"
                )
            else:
                top_features = sorted(
                    shap_importance_map,
                    key=shap_importance_map.get,
                    reverse=True,
                )[:top_n]
                top_features = [f for f in top_features if f in pred_df.columns]
                print(
                    f"   📊 SHAP∩Gain 交集不足({len(intersection)})，"
                    f"使用 SHAP top {len(top_features)}"
                )
        else:
            top_features = sorted(gain_rank, key=gain_rank.get, reverse=True)[:top_n]
            top_features = [f for f in top_features if f in pred_df.columns]
            print(f"   📊 top {len(top_features)} features by gain (fallback)")

        if len(top_features) < 3:
            top_features = avail[:top_n]
    except Exception:
        top_features = avail[:top_n]

    return top_features, shap_importance_map, interaction_pairs


def _generate_gate_rules_statistical(
    pred_df,
    feature_names: list,
    rr_col_name: str,
    lgbm_model=None,
    max_rules: int = 5,
) -> "list | None":
    """
    统计验证法生成 Gate 规则 (v4 最终方案)。

    Pipeline:
      SHAP ∩ Gain → 候选特征 (SHAP 找真实贡献, Gain 找高频使用, 交集最稳)
      分位数阈值 → threshold sweep
      lift + effect_size + robustness → 评分
      相关性剪枝 → 去冗余
      SHAP interaction → 复合规则发现
      ≤ max_rules 条规则 → YAML

    返回: [{"conditions": [(feat, op, thr), ...], "coef": lift}, ...] 或 None
    """
    import numpy as np
    import yaml as _yaml
    from pathlib import Path as _Path

    # ── 加载 kpi_gates/gate_layer.yaml ──
    _gate_kpi_path = _Path("config/kpi_gates/gate_layer.yaml")
    if _gate_kpi_path.exists():
        with open(_gate_kpi_path, "r", encoding="utf-8") as _gf:
            _gkpi = _yaml.safe_load(_gf) or {}
    else:
        _gkpi = {}
    _gt = _gkpi.get("thresholds", {})
    _gc = _gkpi.get("compound", {})
    _gv = _gkpi.get("validation", {})
    _gsg = _gkpi.get("shap_gain", {})

    GATE_MIN_LIFT = _gt.get("min_lift", 1.05)
    GATE_MIN_EFFECT = _gt.get("min_effect", 0.10)
    GATE_MIN_ROBUSTNESS = _gt.get("min_robustness", 0.4)
    GATE_MIN_GATE_SCORE = _gt.get("min_gate_score", 0.0)
    GATE_CORR_THRESHOLD = _gt.get("correlation_threshold", 0.80)
    GATE_DENY_RATE_MIN = _gt.get("deny_rate_min", 0.05)
    GATE_DENY_RATE_MAX = _gt.get("deny_rate_max", 0.70)

    avail = [f for f in feature_names if f in pred_df.columns]
    if len(avail) < 2 or not rr_col_name or rr_col_name not in pred_df.columns:
        print("   ⚠️  统计验证: 可用特征或收益列不足，跳过")
        return None

    # ── Step 1: SHAP ∩ Gain 特征选择 (Gate/Evidence 共用) ──
    _gate_top_n = _gsg.get("top_n", 8)
    top_features, shap_importance_map, _interaction_pairs = _compute_shap_gain_features(
        pred_df,
        feature_names,
        lgbm_model,
        top_n=_gate_top_n,
        compute_interactions=_gsg.get("compute_interactions", True),
    )

    # ── Step 2: 构建坏交易标签 ──
    rr_vals = pred_df[rr_col_name].values.astype(float)
    if rr_col_name == "success_no_rr_extreme":
        bad = (pred_df[rr_col_name] < 0.5).astype(int).values
    else:
        q30 = np.nanpercentile(rr_vals[~np.isnan(rr_vals)], 30)
        bad = (rr_vals < q30).astype(int)

    overall_bad_rate = float(np.nanmean(bad))
    if overall_bad_rate < 0.05 or overall_bad_rate > 0.95:
        print(f"   ⚠️  统计验证: bad_rate={overall_bad_rate:.1%} 极端，跳过")
        return None

    n_total = len(pred_df)
    print(
        f"   📊 统计验证: {n_total} samples, bad_rate={overall_bad_rate:.1%}, "
        f"rr_col={rr_col_name}"
    )

    # ── Step 3: Threshold sweep ──
    quantiles = _gv.get("quantiles", [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85])
    n_folds = _gv.get("n_folds", 5)
    fold_size = n_total // n_folds
    candidates = []

    for feat in top_features:
        col = pred_df[feat].values.astype(float)
        valid = ~np.isnan(col)
        if valid.sum() < 100:
            continue

        thresholds = np.unique(np.quantile(col[valid], quantiles))
        for thr in thresholds:
            for direction in ["gt", "lt"]:
                deny_mask = (col > thr) if direction == "gt" else (col < thr)
                deny_mask = deny_mask & valid
                deny_rate = float(deny_mask.mean())

                if deny_rate < GATE_DENY_RATE_MIN or deny_rate > GATE_DENY_RATE_MAX:
                    continue

                # Lift
                bad_in_deny = float(bad[deny_mask].mean()) if deny_mask.any() else 0
                lift = bad_in_deny / overall_bad_rate if overall_bad_rate > 0 else 0
                if lift <= GATE_MIN_LIFT:
                    continue

                # Effect size
                mean_allow = float(np.nanmean(rr_vals[~deny_mask]))
                mean_deny = float(np.nanmean(rr_vals[deny_mask]))
                effect = mean_allow - mean_deny
                if effect < GATE_MIN_EFFECT:
                    continue

                # Robustness (time-ordered folds + cross-sample stability)
                fold_lifts = []
                for fi in range(n_folds):
                    s = fi * fold_size
                    e = (fi + 1) * fold_size if fi < n_folds - 1 else n_total
                    fb = bad[s:e]
                    fd = deny_mask[s:e]
                    fbr = float(fb.mean())
                    if fbr > 0 and fd.any():
                        fl = float(fb[fd].mean()) / fbr
                        fold_lifts.append(fl)

                rob_time = sum(1 for fl in fold_lifts if fl > 1.0) / max(
                    len(fold_lifts), 1
                )
                # Cross-sample stability: min/max lift ratio across folds
                if len(fold_lifts) >= 2:
                    rob_cross = (
                        min(fold_lifts) / max(fold_lifts) if max(fold_lifts) > 0 else 0
                    )
                    robustness = rob_time * 0.6 + rob_cross * 0.4
                else:
                    robustness = rob_time
                if robustness < GATE_MIN_ROBUSTNESS:
                    continue

                op = ">" if direction == "gt" else "<"
                score = lift * 0.4 + robustness * 0.4 + (1 - deny_rate) * 0.2
                candidates.append(
                    {
                        "feature": feat,
                        "op": op,
                        "threshold": float(thr),
                        "lift": lift,
                        "robustness": robustness,
                        "deny_rate": deny_rate,
                        "effect_size": effect,
                        "score": score,
                        "_deny_mask": deny_mask,
                    }
                )

    if not candidates:
        print("   ⚠️  统计验证: 无候选规则通过 lift+robustness 筛选")
        return None

    candidates.sort(key=lambda x: -x["score"])
    print(f"   📊 统计验证: {len(candidates)} 候选规则通过筛选")

    # ── Step 4: 相关性剪枝 ──
    selected = []
    for cand in candidates:
        if len(selected) >= max_rules * 2:
            break
        correlated = False
        for sel in selected:
            m1 = cand["_deny_mask"].astype(float)
            m2 = sel["_deny_mask"].astype(float)
            if m1.std() == 0 or m2.std() == 0:
                continue
            corr = float(np.corrcoef(m1, m2)[0, 1])
            if abs(corr) > GATE_CORR_THRESHOLD:
                correlated = True
                break
        if not correlated:
            selected.append(cand)

    if not selected:
        print("   ⚠️  统计验证: 剪枝后无规则剩余")
        return None

    # ── Step 5: 复合规则发现 (Lift Surface + 单规则组合) ──
    compound_rules = []

    # 5a: Lift Surface — SHAP interaction 引导的 2D 精细网格扫描
    # 核心思想: 把两个特征的组合空间画成 2D 平面, 找 bad trade lift 最高的矩形区域
    # 比粗略 threshold sweep 更好: 能发现 non-linear alpha region
    if _interaction_pairs:
        # 10 个分位数 bin 边界 → 大约 10×10 = 100 cells per pair
        surface_quantiles = np.linspace(0.10, 0.90, _gc.get("surface_bins", 9))
        min_cell_samples = max(20, n_total // _gc.get("min_cell_ratio", 50))

        for feat_a, feat_b, interact_score in _interaction_pairs[:5]:
            if feat_a not in pred_df.columns or feat_b not in pred_df.columns:
                continue
            col_a = pred_df[feat_a].values.astype(float)
            col_b = pred_df[feat_b].values.astype(float)
            valid_ab = ~(np.isnan(col_a) | np.isnan(col_b))
            if valid_ab.sum() < 100:
                continue

            # 构建分位数 bin 边界
            edges_a = np.unique(np.nanquantile(col_a[valid_ab], surface_quantiles))
            edges_b = np.unique(np.nanquantile(col_b[valid_ab], surface_quantiles))
            if len(edges_a) < 3 or len(edges_b) < 3:
                continue

            # ── 构建 Lift Surface: 每个 cell 计算 lift ──
            cell_grid = []  # [(ia, ib, lift, bad_rate, count, mask)]
            for ia in range(len(edges_a) + 1):
                for ib in range(len(edges_b) + 1):
                    # 确定 cell 边界
                    lo_a = edges_a[ia - 1] if ia > 0 else -np.inf
                    hi_a = edges_a[ia] if ia < len(edges_a) else np.inf
                    lo_b = edges_b[ib - 1] if ib > 0 else -np.inf
                    hi_b = edges_b[ib] if ib < len(edges_b) else np.inf

                    cell_mask = (
                        (col_a >= lo_a)
                        & (col_a < hi_a)
                        & (col_b >= lo_b)
                        & (col_b < hi_b)
                        & valid_ab
                    )
                    cnt = int(cell_mask.sum())
                    if cnt < min_cell_samples:
                        continue
                    cell_br = float(bad[cell_mask].mean())
                    cell_lift = (
                        cell_br / overall_bad_rate if overall_bad_rate > 0 else 0
                    )
                    cell_grid.append(
                        (ia, ib, cell_lift, cell_br, cnt, lo_a, hi_a, lo_b, hi_b)
                    )

            if not cell_grid:
                continue

            # ── 找高 lift 区域并合并相邻 cells ──
            HOT_CELL_LIFT = _gc.get("hot_cell_lift", 1.3)
            hot_cells = [c for c in cell_grid if c[2] > HOT_CELL_LIFT]
            if not hot_cells:
                continue

            # 贪心合并: 从最高 lift cell 开始，尝试扩展到相邻 cells
            hot_cells.sort(key=lambda x: -x[2])
            used_cells = set()

            for seed_cell in hot_cells:
                seed_key = (seed_cell[0], seed_cell[1])
                if seed_key in used_cells:
                    continue

                # 收集 seed + 相邻高 lift cells 形成矩形区域
                region_cells = [seed_cell]
                used_cells.add(seed_key)

                # 尝试扩展: 查找同行/同列的高 lift cells
                for other in hot_cells:
                    ok = (other[0], other[1])
                    if ok in used_cells:
                        continue
                    # 相邻: 行差 ≤ 1 或列差 ≤ 1
                    if (
                        abs(other[0] - seed_cell[0]) <= 1
                        and abs(other[1] - seed_cell[1]) <= 1
                    ):
                        region_cells.append(other)
                        used_cells.add(ok)

                # 计算合并区域的边界
                region_lo_a = min(c[5] for c in region_cells)
                region_hi_a = max(c[6] for c in region_cells)
                region_lo_b = min(c[7] for c in region_cells)
                region_hi_b = max(c[8] for c in region_cells)

                # 构建合并 mask
                region_mask = (
                    (col_a >= region_lo_a)
                    & (col_a < region_hi_a)
                    & (col_b >= region_lo_b)
                    & (col_b < region_hi_b)
                    & valid_ab
                )
                jdr = float(region_mask.mean())
                JOINT_DENY_MIN = _gc.get("joint_deny_rate_min", 0.02)
                JOINT_DENY_MAX = _gc.get("joint_deny_rate_max", 0.50)
                if (
                    jdr < JOINT_DENY_MIN
                    or jdr > JOINT_DENY_MAX
                    or not region_mask.any()
                ):
                    continue

                jbr = float(bad[region_mask].mean())
                jl = jbr / overall_bad_rate if overall_bad_rate > 0 else 0
                if jl < HOT_CELL_LIFT:
                    continue
                je = float(np.nanmean(rr_vals[~region_mask])) - float(
                    np.nanmean(rr_vals[region_mask])
                )
                if je < _gc.get("min_joint_effect", 0.15):
                    continue

                # Robustness: time-fold + cross-sample stability
                fold_lifts = []
                for fi in range(n_folds):
                    s = fi * fold_size
                    e = (fi + 1) * fold_size if fi < n_folds - 1 else n_total
                    fb, fd = bad[s:e], region_mask[s:e]
                    fbr = float(fb.mean())
                    if fbr > 0 and fd.any():
                        fold_lifts.append(float(fb[fd].mean()) / fbr)
                rob_time = sum(1 for fl in fold_lifts if fl > 1.0) / max(
                    len(fold_lifts), 1
                )
                # Cross-sample stability: min/max lift ratio across folds
                if len(fold_lifts) >= 2:
                    rob_cross = (
                        min(fold_lifts) / max(fold_lifts) if max(fold_lifts) > 0 else 0
                    )
                    rob = rob_time * 0.6 + rob_cross * 0.4  # 综合 robustness
                else:
                    rob = rob_time
                if rob < _gc.get("min_joint_robustness", 0.35):
                    continue

                # 生成规则条件: 转换区域边界为简洁的 threshold 规则
                conditions = []
                if region_lo_a > -np.inf:
                    conditions.append((feat_a, ">", float(region_lo_a)))
                if region_hi_a < np.inf:
                    conditions.append((feat_a, "<", float(region_hi_a)))
                if region_lo_b > -np.inf:
                    conditions.append((feat_b, ">", float(region_lo_b)))
                if region_hi_b < np.inf:
                    conditions.append((feat_b, "<", float(region_hi_b)))

                # Gate 规则最多 2 个条件 (经验: 1=不够强, 2=最优, 3+=过拟合)
                # 如果区域是 band (lo < x < hi)，算 1 个"逻辑条件"
                # 如果 > 2 个逻辑条件，选 lift 贡献最大的 2 个
                if len(conditions) > 2:
                    # 简化: 保留每个特征最有区分力的一侧
                    conds_a = [c for c in conditions if c[0] == feat_a]
                    conds_b = [c for c in conditions if c[0] == feat_b]
                    # 每个特征保留 1 个条件
                    best_a = conds_a[0] if conds_a else None
                    best_b = conds_b[0] if conds_b else None
                    conditions = [c for c in [best_a, best_b] if c is not None]

                if not conditions:
                    continue

                compound_rules.append(
                    {
                        "conditions": conditions,
                        "lift": jl,
                        "effect_size": je,
                        "deny_rate": jdr,
                        "robustness": rob,
                        "source": "lift_surface",
                        "_n_cells": len(region_cells),
                    }
                )

    # 5b: 单规则组合 (fallback: 从已筛选的 top 单规则中两两组合)
    for i in range(min(len(selected), 5)):
        for j in range(i + 1, min(len(selected), 5)):
            s1, s2 = selected[i], selected[j]
            joint = s1["_deny_mask"] & s2["_deny_mask"]
            jdr = float(joint.mean())
            if jdr < JOINT_DENY_MIN or jdr > JOINT_DENY_MAX:
                continue
            if not joint.any():
                continue
            joint_bad_rate = float(bad[joint].mean())
            joint_lift = joint_bad_rate / overall_bad_rate
            if joint_lift > max(s1["lift"], s2["lift"]) * 1.2:
                joint_effect = float(np.nanmean(rr_vals[~joint])) - float(
                    np.nanmean(rr_vals[joint])
                )
                compound_rules.append(
                    {
                        "conditions": [
                            (s1["feature"], s1["op"], s1["threshold"]),
                            (s2["feature"], s2["op"], s2["threshold"]),
                        ],
                        "lift": joint_lift,
                        "effect_size": joint_effect,
                        "deny_rate": jdr,
                        "robustness": min(s1["robustness"], s2["robustness"]),
                        "source": "single_combo",
                    }
                )

    # 去重: 同特征对只保留 lift 最高的
    seen_pairs = {}
    for cr in compound_rules:
        pair_key = frozenset(c[0] for c in cr["conditions"])
        if pair_key not in seen_pairs or cr["lift"] > seen_pairs[pair_key]["lift"]:
            seen_pairs[pair_key] = cr
    compound_rules = sorted(seen_pairs.values(), key=lambda x: -x["lift"])

    # ── Step 6: 组装最终 Hard Gate 规则 (Gate Score 选择) ──
    # Gate 只输出 Hard Gate (deny)
    # Soft 信息由 Evidence 处理，不单独设 Soft Filter (避免 double counting)
    #
    # Gate Score = tail_capture - good_deny_rate (Youden's J / Informedness)
    #   tail_capture   = P(deny|bad)  = 规则拦截了多少比例的坏交易
    #   good_deny_rate = P(deny|good) = 规则误杀了多少比例的好交易
    #   Score > 0 → 有区分力; Score ≤ 0 → 淘汰
    # 条数不写死，由 gate_score > 0 + max_rules 上限决定

    # 收集所有候选 (复合 + 单规则)
    all_candidates = []
    for cr in compound_rules[:3]:
        all_candidates.append(
            {
                "conditions": cr["conditions"],
                "lift": cr["lift"],
                "deny_rate": cr["deny_rate"],
                "robustness": cr["robustness"],
                "effect_size": cr["effect_size"],
                "source": cr.get("source", "compound"),
            }
        )
    used_features = set()
    for cr in all_candidates:
        for f, _, _ in cr["conditions"]:
            used_features.add(f)
    for sel in selected:
        if sel["feature"] in used_features:
            continue
        all_candidates.append(
            {
                "conditions": [(sel["feature"], sel["op"], sel["threshold"])],
                "lift": sel["lift"],
                "deny_rate": sel["deny_rate"],
                "robustness": sel["robustness"],
                "effect_size": sel["effect_size"],
                "source": "single",
            }
        )
        used_features.add(sel["feature"])

    # 计算每个候选的 gate_score
    total_bad = int(bad.sum())
    total_good = int((~bad.astype(bool)).sum())
    good = ~bad.astype(bool)

    for cand in all_candidates:
        # 构建 deny_mask
        if len(cand["conditions"]) == 1:
            feat, op, thr = cand["conditions"][0]
            if feat in pred_df.columns:
                col = pred_df[feat].values.astype(float)
                deny_mask = col > thr if op == ">" else col < thr
            else:
                deny_mask = np.zeros(n_total, dtype=bool)
        else:
            # 复合规则: AND
            deny_mask = np.ones(n_total, dtype=bool)
            for feat, op, thr in cand["conditions"]:
                if feat in pred_df.columns:
                    col = pred_df[feat].values.astype(float)
                    if op == ">":
                        deny_mask &= col > thr
                    else:
                        deny_mask &= col < thr

        bad_in_deny = int(bad[deny_mask].sum()) if deny_mask.any() else 0
        good_in_deny = int(good[deny_mask].sum()) if deny_mask.any() else 0
        tail_capture = bad_in_deny / max(total_bad, 1)
        good_deny_rate = good_in_deny / max(total_good, 1)
        gate_score = tail_capture - good_deny_rate

        cand["tail_capture"] = tail_capture
        cand["good_deny_rate"] = good_deny_rate
        cand["gate_score"] = gate_score

    # 按 gate_score 降序排序，筛选 gate_score > 0 的候选
    all_candidates.sort(key=lambda x: -x.get("gate_score", 0))

    final_rules = []
    for cand in all_candidates:
        if cand["gate_score"] > 0 and len(final_rules) < max_rules:
            final_rules.append(
                {
                    "conditions": cand["conditions"],
                    "coef": cand["lift"],
                    "gate_score": cand["gate_score"],
                    "tail_capture": cand["tail_capture"],
                    "good_deny_rate": cand["good_deny_rate"],
                }
            )

    if not final_rules:
        return None

    # ── 诊断输出 ──
    n_ls = sum(
        1
        for c in all_candidates
        if c.get("source") == "lift_surface" and c.get("gate_score", 0) > 0
    )
    src_tag = f" ({n_ls} via Lift Surface)" if n_ls else ""
    print(f"   ✅ 统计验证: {len(final_rules)} hard gate{src_tag}")
    print(f"   📊 Gate Score = tail_capture - good_deny_rate (Youden's J)")

    for i, r in enumerate(final_rules):
        conds_str = " & ".join(f"{f} {o} {t:.4g}" for f, o, t in r["conditions"])
        tc = r["tail_capture"]
        gd = r["good_deny_rate"]
        gs = r["gate_score"]
        # 找对应的完整 metrics
        cand = next(
            (
                c
                for c in all_candidates
                if set((f, o, t) for f, o, t in c["conditions"])
                == set((f, o, t) for f, o, t in r["conditions"])
            ),
            None,
        )
        dr = cand["deny_rate"] if cand else 0
        rob = cand["robustness"] if cand else 0
        print(
            f"   🚫 [gate_{i+1}] {conds_str}"
            f" | score={gs:.3f}"
            f" | tail_cap={tc:.0%}"
            f" | good_deny={gd:.0%}"
            f" | deny={dr:.0%}"
            f" | robust={rob:.0%}"
            f" | lift={r['coef']:.2f}"
        )

    return final_rules


def _generate_risk_gate_yaml(
    output_path: Path,
    rules: list,
    strategy: str,
    model_source: str,
    top_n: int = 5,
    predictions_path: "Path | str | None" = None,
    feature_names: "list | None" = None,
    lgbm_model=None,
):
    """
    从树模型分裂规则生成 risk_gate_draft.yaml 草稿。

    ✅ 生成 archetype 兼容格式 (when/then/action)，可直接用于 gate apply-archetype。
    ✅ 自动用 predictions 数据确定语义方向（deny 收益更差的那一侧）。
    同时写入 config/strategies/{strategy}/archetypes/gate_draft.yaml。
    """
    import yaml

    # 加载 gate_layer kpi_gates 配置 (用于 governance 元数据 + tree_split fallback)
    _gkpi = {}
    _gate_kpi_path = Path("config/kpi_gates/gate_layer.yaml")
    if _gate_kpi_path.exists():
        import yaml as _yaml_kpi

        _gkpi = _yaml_kpi.safe_load(_gate_kpi_path.read_text(encoding="utf-8")) or {}

    # 加载 predictions 数据
    pred_df = None
    rr_col_name = None
    if predictions_path is not None:
        try:
            import pandas as pd

            pred_df = pd.read_parquet(predictions_path)
            for _rc in ["forward_rr", "success_no_rr_extreme", "ret_mean"]:
                if _rc in pred_df.columns:
                    rr_col_name = _rc
                    break
            if rr_col_name is None:
                pred_df = None  # 无收益列，退化为旧逻辑
        except Exception:
            pred_df = None

    # ── 优先级: 统计验证法 > tree_split fallback ──
    # imodels 蒸馏已废弃 (已被 v4 统计验证法替代)
    validated_rules = None

    # Path 1: 统计验证法 (v4 最终方案)
    if pred_df is not None and feature_names and rr_col_name:
        validated_rules = _generate_gate_rules_statistical(
            pred_df,
            feature_names,
            rr_col_name=rr_col_name,
            lgbm_model=lgbm_model,
            max_rules=top_n,
        )
        if validated_rules:
            print("   ✅ 使用统计验证法生成规则")

    # 生成 archetype 兼容的 hard_gates (只有 Hard Gate, 无 Soft Filter)
    hard_gates = []

    if validated_rules is not None:
        # ── 统计验证路径：规则为 {"conditions": [...], "coef": float} ──
        _op_to_key = {
            "<=": "value_le",
            "<": "value_lt",
            ">": "value_gt",
            ">=": "value_ge",
        }
        hard_idx = 0
        for rule_dict in validated_rules:
            conditions = rule_dict["conditions"]
            coef = rule_dict["coef"]

            if len(conditions) == 1:
                # 单条件规则
                feat, op, thr = conditions[0]
                condition_key = _op_to_key.get(op, "value_lt")
                rule_id = f"gate_{feat.replace('.', '_').lower()}"
                tag = f"HARD_{feat.upper().replace('.', '_')}"
                when_clause = {feat: {condition_key: round(thr, 4)}}
                comment_str = f"statistical(lift={coef:.2f}): {feat} {op} {thr:.4g}"
            else:
                # 复合规则 → all_of
                all_of_items = []
                feat_names_list = []
                rule_parts = []
                for feat, op, thr in conditions:
                    condition_key = _op_to_key.get(op, "value_lt")
                    all_of_items.append({feat: {condition_key: round(thr, 4)}})
                    feat_names_list.append(feat)
                    rule_parts.append(f"{feat} {op} {thr:.4g}")
                rule_id = f"gate_{'_'.join(f.replace('.', '_').lower() for f in feat_names_list)}"
                tag = f"HARD_{'_'.join(f.upper().replace('.', '_') for f in feat_names_list)}"
                when_clause = {"all_of": all_of_items}
                comment_str = f"statistical(lift={coef:.2f}): {' & '.join(rule_parts)}"

            rule = {
                "id": rule_id,
                "tag": tag,
                "phase": "hard_gate",
                "priority": 10 + hard_idx,
                "reason": f"统计验证规则 (lift={coef:.2f})",
                "when": when_clause,
                "then": {"action": "deny"},
                "comment": comment_str,
            }
            hard_gates.append(rule)
            hard_idx += 1
    else:
        # ── Fallback：树分裂法（改进版：去重 + 最小效果量过滤）──
        # 每个特征只保留分裂次数最高的那个阈值，避免同特征多阈值产生矛盾区间
        seen_features = {}  # feat_name → (thr, op, count)
        for name, thr, op, count in rules[: top_n * 2]:
            if name not in seen_features or count > seen_features[name][2]:
                seen_features[name] = (thr, op, count)
        deduped_rules = [
            (name, thr, op, count) for name, (thr, op, count) in seen_features.items()
        ]
        # 按分裂次数降序排列
        deduped_rules.sort(key=lambda x: -x[3])

        rule_idx = 0
        for name, thr, op, count in deduped_rules[:top_n]:
            if pred_df is not None and name in pred_df.columns and rr_col_name:
                # 数据驱动: 比较阈值两侧的平均收益，deny 差的那边
                feat_vals = pred_df[name].dropna()
                rr_vals = pred_df.loc[feat_vals.index, rr_col_name]
                low_mask = feat_vals < thr
                high_mask = feat_vals >= thr
                mean_low = float(rr_vals[low_mask].mean()) if low_mask.any() else 0.0
                mean_high = float(rr_vals[high_mask].mean()) if high_mask.any() else 0.0

                # 最小效果量过滤: 两侧差距 < 0.15 的规则不可靠，跳过
                effect_size = abs(mean_low - mean_high)
                _ts_min_effect = _gkpi.get("tree_split", {}).get(
                    "min_effect_size", 0.15
                )
                if effect_size < _ts_min_effect:
                    print(
                        f"   ⚠️  {name}: 效果量不足 ({effect_size:.3f}<0.15)，"
                        f"low={mean_low:.3f} vs high={mean_high:.3f}，跳过"
                    )
                    continue

                if mean_low < mean_high:
                    condition_key = "value_lt"
                else:
                    condition_key = "value_gt"
                direction_source = (
                    f"data_verified (low={mean_low:.3f}, high={mean_high:.3f})"
                )
            else:
                print(f"   ⚠️  {name}: 无 predictions 数据无法确定方向，跳过")
                continue

            rule_id = f"gate_{name.replace('.', '_').lower()}"
            tag = f"HARD_{name.upper().replace('.', '_')}"
            rule = {
                "id": rule_id,
                "tag": tag,
                "phase": "hard_gate",
                "priority": 10 + rule_idx,
                "reason": f"{name} 触发树分裂条件 (分裂 {count} 次)",
                "when": {
                    name: {condition_key: round(thr, 4)},
                },
                "then": {"action": "deny"},
                "comment": f"自动生成(tree_split fallback): 分裂 {count} 次 | 阈值 {thr:.4g} | 方向: {direction_source}",
            }
            hard_gates.append(rule)
            rule_idx += 1

    # 构建完整 archetype gate 配置 (只有 Hard Gate, 无 Soft Filter)
    config = {
        "schema": {
            "phases": ["system_safety", "hard_gate", "guardrail"],
            "evaluation_order": "system_safety -> hard_gate -> guardrail",
            "governance": {
                "selection_method": "gate_score (Youden's J = tail_capture - good_deny_rate)",
                "min_gate_score": _gkpi.get("thresholds", {}).get(
                    "min_gate_score", 0.0
                ),
                "alert_threshold": 0.25,
            },
        },
        "hard_gates": hard_gates,
        "guardrails": [],  # guardrails 应基于实际特征数据验证后手动添加
    }

    # 写入 results 目录 (备份)
    yaml_content = yaml.dump(
        config,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )
    output_path.write_text(yaml_content, encoding="utf-8")

    # 同时写入策略配置目录 (config/strategies/{strategy}/gate_draft.yaml)
    # 与 prefilter.yaml / direction.yaml 同级，审核后复制到 archetypes/gate.yaml
    project_root = Path(__file__).resolve().parents[1]
    strategy_dir = project_root / "config" / "strategies" / strategy
    if strategy_dir.is_dir():
        draft_path = strategy_dir / "gate_draft.yaml"
        direction_note = (
            "# ✅ 方向已由 predictions 数据自动确定 (deny 收益更差的一侧)\n"
            if pred_df is not None
            else "# ⚠️ 无 predictions 数据，无法自动确定方向，规则可能为空\n"
        )
        header = (
            f"# {strategy.upper()} Gate Draft (auto-generated)\n"
            f"# 来源: {model_source}\n"
            f"{direction_note}"
            f"# 已是 archetype 兼容格式，可直接用于:\n"
            f"#   mlbot gate apply-archetype --strategy {strategy}\n"
            f"#   python scripts/optimize_gate_unified.py --strategy {strategy} --promote\n\n"
        )
        draft_path.write_text(header + yaml_content, encoding="utf-8")
        print(f"   \U0001f4c1 Gate draft (archetype format) → {draft_path}")

    return output_path


def _generate_evidence_candidates_yaml(
    output_path: Path,
    rules: list,
    strategy: str,
    model_source: str,
    top_n: int = 15,
    pred_df=None,
    feature_names: "list | None" = None,
    lgbm_model=None,
    rr_col_name: "str | None" = None,
):
    """
    生成 evidence_candidates.yaml (v2: SHAP∩Gain + bad_suppression + 2D interaction).

    如果提供 lgbm_model + pred_df:
      - SHAP∩Gain 替代 tree split count 作为特征排名
      - 每个候选计算 preliminary bad_suppression 验证区分力
      - SHAP interaction 发现交互特征对
    否则 fallback 到 tree split count。
    输出格式与 optimize_evidence_plateau.py 兼容。
    """
    import yaml
    import numpy as np
    from collections import defaultdict

    # ── SHAP∩Gain 特征发现 (复用 Gate 同款方法) ──
    shap_features, shap_importance_map, interaction_pairs = [], {}, []
    if lgbm_model is not None and feature_names and pred_df is not None:
        shap_features, shap_importance_map, interaction_pairs = (
            _compute_shap_gain_features(
                pred_df,
                feature_names,
                lgbm_model,
                top_n=top_n,
                compute_interactions=True,
            )
        )
        print(f"   📊 Evidence SHAP∩Gain: {len(shap_features)} 候选特征")

    # ── bad 标签 (preliminary bad_suppression 用) ──
    bad, good = None, None
    if pred_df is not None and rr_col_name and rr_col_name in pred_df.columns:
        rr_vals = pred_df[rr_col_name].values.astype(float)
        if rr_col_name == "success_no_rr_extreme":
            bad = (pred_df[rr_col_name] < 0.5).values
        else:
            q30 = np.nanpercentile(rr_vals[~np.isnan(rr_vals)], 30)
            bad = rr_vals < q30
        good = ~bad

    # ── 聚合 tree split 信息 (无论哪条路径都提供 threshold_examples) ──
    tree_info = defaultdict(lambda: {"thresholds": [], "count": 0})
    for name, thr, op, count in rules:
        tree_info[name]["thresholds"].append(round(thr, 4))
        tree_info[name]["count"] += count

    candidates = []
    compound_info = []

    if shap_features and pred_df is not None:
        # ── v2 路径: SHAP∩Gain + bad_suppression 验证 ──
        for rank, feature in enumerate(shap_features, 1):
            if feature not in pred_df.columns:
                continue
            shap_imp = shap_importance_map.get(feature, 0.0)

            # Preliminary bad_suppression: 用 gate-conditioned rank 快速评估
            prelim_bad_supp = 0.0
            direction_hint = "higher_is_better"
            if bad is not None:
                feat_vals = pred_df[feature].values.astype(float)
                valid = ~np.isnan(feat_vals)
                ref_vals = (
                    feat_vals[valid & good] if good is not None else feat_vals[valid]
                )
                if len(ref_vals) > 50:
                    q_thr = [np.quantile(ref_vals, q) for q in [0.2, 0.4, 0.6, 0.8]]
                    pct = np.full(len(feat_vals), 0.5)
                    for i in range(len(feat_vals)):
                        v = feat_vals[i]
                        if np.isnan(v):
                            continue
                        if v <= q_thr[0]:
                            pct[i] = 0.1
                        elif v <= q_thr[1]:
                            pct[i] = 0.3
                        elif v <= q_thr[2]:
                            pct[i] = 0.5
                        elif v <= q_thr[3]:
                            pct[i] = 0.7
                        else:
                            pct[i] = 0.9
                    # 两个方向都试，取 bad_suppression 更大的
                    p_low_bad = (pct[bad] <= 0.3).mean() if bad.any() else 0
                    p_low_good = (pct[good] <= 0.3).mean() if good.any() else 0
                    bs_higher = p_low_bad - p_low_good
                    p_hi_bad = (pct[bad] >= 0.7).mean() if bad.any() else 0
                    p_hi_good = (pct[good] >= 0.7).mean() if good.any() else 0
                    bs_lower = p_hi_bad - p_hi_good
                    if bs_higher >= bs_lower:
                        prelim_bad_supp, direction_hint = (
                            float(bs_higher),
                            "higher_is_better",
                        )
                    else:
                        prelim_bad_supp, direction_hint = (
                            float(bs_lower),
                            "lower_is_better",
                        )

            # tree split 补充信息
            ti = tree_info.get(feature, {"thresholds": [], "count": 0})
            thresholds = sorted(ti["thresholds"])
            if thresholds and len(thresholds) >= 2:
                dist_hint = (
                    f"密集分布在 {min(thresholds):.3g}–{max(thresholds):.3g} 区间"
                )
            elif thresholds:
                dist_hint = f"单一阈值 {thresholds[0]:.4g}"
            else:
                dist_hint = "SHAP∩Gain 发现，无 tree split"

            # 交互伙伴
            partners = []
            for fa, fb, sc in interaction_pairs:
                if fa == feature:
                    partners.append({"feature": fb, "interaction_score": round(sc, 4)})
                elif fb == feature:
                    partners.append({"feature": fa, "interaction_score": round(sc, 4)})

            cand = {
                "rank": rank,
                "feature": feature,
                "discovery_method": "shap_gain",
                "shap_importance": round(shap_imp, 4),
                "split_count_total": ti["count"],
                "prelim_bad_suppression": round(prelim_bad_supp, 4),
                "direction_hint": direction_hint,
                "threshold_examples": thresholds[:5],
                "distribution_hint": dist_hint,
                "quantile_mapping": {
                    "_comment": "bins 由 optimize_evidence_plateau.py 优化",
                    "bins": [0.2, 0.4, 0.6, 0.8],
                    "labels": ["suppress", "downweight", "neutral", "favor", "amplify"],
                },
                "usage_hint": "TODO: 填写用途",
                "affects": {
                    "candidates": [
                        "tp_range",
                        "trailing_speed",
                        "scale_in_allowed",
                        "position_size",
                    ]
                },
            }
            if partners:
                cand["interaction_partners"] = partners[:3]
            candidates.append(cand)

        # 交互对元数据
        for fa, fb, sc in interaction_pairs[:5]:
            if fa in pred_df.columns and fb in pred_df.columns:
                compound_info.append(
                    {"features": [fa, fb], "interaction_score": round(sc, 4)}
                )

        n_valid = sum(1 for c in candidates if c["prelim_bad_suppression"] > 0)
        print(f"   📊 Evidence 候选: {len(candidates)} 特征 (bad_supp>0: {n_valid})")
    else:
        # ── Fallback: tree split count (原有逻辑) ──
        sorted_features = sorted(
            tree_info.items(), key=lambda x: x[1]["count"], reverse=True
        )[:top_n]
        for rank, (feature, data) in enumerate(sorted_features, 1):
            thresholds = sorted(data["thresholds"])
            if len(thresholds) >= 2:
                dist_hint = (
                    f"密集分布在 {min(thresholds):.3g}–{max(thresholds):.3g} 区间"
                )
            elif thresholds:
                dist_hint = f"单一阈值 {thresholds[0]:.4g}"
            else:
                dist_hint = ""
            candidates.append(
                {
                    "rank": rank,
                    "feature": feature,
                    "discovery_method": "tree_split",
                    "split_count_total": data["count"],
                    "threshold_examples": thresholds[:5],
                    "distribution_hint": dist_hint,
                    "quantile_mapping": {
                        "_comment": "根据 threshold_examples 定义 bins",
                        "bins": [0.2, 0.4, 0.6, 0.8],
                        "labels": [
                            "suppress",
                            "downweight",
                            "neutral",
                            "favor",
                            "amplify",
                        ],
                    },
                    "usage_hint": "TODO: 填写用途",
                    "affects": {
                        "candidates": [
                            "tp_range",
                            "trailing_speed",
                            "scale_in_allowed",
                            "position_size",
                        ]
                    },
                }
            )

    config = {
        "_meta": {
            "generated_from": str(model_source),
            "strategy": strategy,
            "purpose": "Evidence 轴候选列表",
            "discovery_method": (
                "SHAP∩Gain + bad_suppression" if shap_features else "tree_split"
            ),
            "note": "此为自动生成的草稿，需人工审核后使用",
        },
        "label_semantics": {
            "suppress": "强烈不利 - 拒绝或极度限制",
            "downweight": "不利 - 降低信心/仓位",
            "neutral": "中性 - 标准执行",
            "favor": "有利 - 提高信心/仓位",
            "amplify": "强烈有利 - 最大化执行",
        },
        "evidence_candidates": candidates,
    }
    if compound_info:
        config["interaction_pairs"] = compound_info
    config["usage_guide"] = {
        "step_1": "审核 feature 是否有语义意义 + prelim_bad_suppression",
        "step_2": "运行 optimize_evidence_plateau.py 优化 bins",
        "step_3": "填写 usage_hint 和 affects",
        "step_4": "--promote 自动写入 archetypes/evidence.yaml",
    }

    yaml_content = yaml.dump(
        config,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )
    output_path.write_text(yaml_content, encoding="utf-8")
    return output_path


def _append_rules_section(
    readme_path: Path, rules: list, strategy: str, model_source: str
):
    content = readme_path.read_text(encoding="utf-8")
    base = _strip_rules_section(content)
    new_lines = [base, "", "---", "", "## 📜 树模型规则导出（固定训练 LightGBM）", ""]
    new_lines.append(
        "以下为从固定训练产出的 LightGBM 模型中提取的**高频分裂条件**（按出现次数排序），用于可归因性与规则维护。"
    )
    new_lines.append("")
    new_lines.append("| 特征 | 条件 | 出现次数 |")
    new_lines.append("|------|------|----------|")
    for name, thr, op, count in rules:
        cond = f"{name} {op} {thr:.4g}"
        new_lines.append(f"| `{name}` | `{cond}` | {count} |")
    new_lines.append("")
    new_lines.append(f"**模型来源**：`{model_source}`")
    new_lines.append("")
    readme_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _append_placeholder_section(readme_path: Path, reason: str):
    """无 model.pkl 或无可提取规则时，写入占位说明。"""
    content = readme_path.read_text(encoding="utf-8")
    base = _strip_rules_section(content)
    new_lines = [
        base,
        "",
        "---",
        "",
        "## 📜 树模型规则导出（固定训练 LightGBM）",
        "",
        "当前未导出规则。",
        "",
        f"**说明**：{reason}",
        "",
    ]
    readme_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"  ✅ {readme_path}（占位：{reason[:50]}…）")


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Export LightGBM tree rules to strategy README"
    )
    ap.add_argument(
        "--model-dir",
        help="Directory containing model.pkl (e.g., results/fixed_long/<strategy>/<strategy>)",
    )
    ap.add_argument("--strategy", help="Strategy name (e.g., sr_reversal_rr_reg_long)")
    ap.add_argument(
        "--max-splits", type=int, default=30, help="Maximum number of split conditions"
    )
    ap.add_argument(
        "--generate-risk-gate",
        action="store_true",
        help="Generate risk_gate_draft.yaml from tree splits (requires manual review)",
    )
    ap.add_argument(
        "--risk-gate-output",
        default=None,
        help="Output path for risk_gate_draft.yaml (default: <model-dir>/risk_gate_draft.yaml)",
    )
    ap.add_argument(
        "base", nargs="?", help="Base directory (legacy: results/fixed_long)"
    )

    args = ap.parse_args()

    try:
        import joblib
        import lightgbm as lgb
    except ImportError as e:
        print("Need joblib and lightgbm:", e)
        return 1

    # 新接口：--model-dir + --strategy
    if args.model_dir and args.strategy:
        artifact_dir = Path(args.model_dir).resolve()
        model_path = artifact_dir / "model.pkl"
        features_path = artifact_dir / "used_features.json"

        # 输出到模型目录下，文件名包含策略名
        output_path = artifact_dir / f"{args.strategy}_tree_rules.md"

        # 同时还可以更新策略目录下的 README（如果存在）
        readme_path = ROOT / "config" / "strategies" / args.strategy / "README.md"

        if not readme_path.exists():
            print(f"  ℹ️  {args.strategy}: 策略 README 不存在，将只输出到模型目录")
            readme_path = None  # 不更新 README

        if not model_path.exists():
            msg = "未找到 model.pkl。请先运行固定训练（如 mlbot train fixed）并确保 ModelArtifact 保存成功。"
            if readme_path:
                _append_placeholder_section(readme_path, msg)
            print(f"  ❌ {args.strategy}: {msg}")
            return 1

        model = joblib.load(model_path)
        if isinstance(model, dict):
            model = (
                model.get("regression") or model.get("model") or list(model.values())[0]
            )
        booster = _get_booster(model)

        feature_names = []
        if features_path.exists():
            with open(features_path, encoding="utf-8") as f:
                feature_names = json.load(f)
        if not feature_names and hasattr(booster, "feature_name"):
            feature_names = booster.feature_name() or []

        rules = _collect_splits(booster, feature_names, max_splits=args.max_splits)
        if not rules:
            msg = "模型可加载但未提取到分裂条件（可能为叶节点或格式变化）。可检查 model.pkl 与 LightGBM 版本。"
            if readme_path:
                _append_placeholder_section(readme_path, msg)
            print(f"  ⚠️ {args.strategy}: {msg}")
            return 0

        # 输出到模型目录下的独立文件
        _write_standalone_rules(output_path, rules, args.strategy, str(artifact_dir))
        print(f"  ✅ {output_path}")

        # 生成 risk_gate_draft.yaml（如果指定）
        if args.generate_risk_gate:
            risk_gate_path = (
                Path(args.risk_gate_output)
                if args.risk_gate_output
                else artifact_dir / "risk_gate_draft.yaml"
            )
            # 尝试加载 predictions 数据自动确定方向
            _pred_path = artifact_dir / "predictions.parquet"
            _generate_risk_gate_yaml(
                risk_gate_path,
                rules,
                args.strategy,
                str(artifact_dir),
                predictions_path=_pred_path if _pred_path.exists() else None,
                feature_names=feature_names if feature_names else None,
                lgbm_model=model,
            )
            _dir_msg = "方向已自动确定" if _pred_path.exists() else "需人工审核方向"
            print(f"  ✅ {risk_gate_path} ({_dir_msg})")

        return 0

    # 旧接口：扫描所有策略
    base = ROOT / "results" / "fixed_long"
    if args.base:
        base = Path(args.base).resolve()

    for strategy in STRATEGIES:
        artifact_dir = base / strategy / strategy
        model_path = artifact_dir / "model.pkl"
        features_path = artifact_dir / "used_features.json"
        readme_path = ROOT / "config" / "strategies" / strategy / "README.md"

        if not readme_path.exists():
            print(f"  ⚠️  {strategy}: no README at {readme_path}")
            continue

        if not model_path.exists():
            _append_placeholder_section(
                readme_path,
                "未找到 model.pkl。请先运行固定训练（如 mlbot train fixed）并确保 ModelArtifact 保存成功。",
            )
            continue

        model = joblib.load(model_path)
        if isinstance(model, dict):
            model = (
                model.get("regression") or model.get("model") or list(model.values())[0]
            )
        booster = _get_booster(model)

        feature_names = []
        if features_path.exists():
            with open(features_path, encoding="utf-8") as f:
                feature_names = json.load(f)
        if not feature_names and hasattr(booster, "feature_name"):
            feature_names = booster.feature_name() or []

        rules = _collect_splits(booster, feature_names, max_splits=args.max_splits)
        if not rules:
            _append_placeholder_section(
                readme_path,
                "模型可加载但未提取到分裂条件（可能为叶节点或格式变化）。可检查 model.pkl 与 LightGBM 版本。",
            )
            continue

        _append_rules_section(readme_path, rules, strategy, str(artifact_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
