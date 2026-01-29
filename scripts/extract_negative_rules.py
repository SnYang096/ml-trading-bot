#!/usr/bin/env python3
"""
负规则提取脚本

从训练好的浅树模型中提取负规则，用于生成 veto gate。

使用方法：
    python scripts/extract_negative_rules.py \
        --model-path models/outcome_audit/model.pkl \
        --data-path data/parquet_data \
        --output-path reports/audit/negative_rules.yaml

输出格式：
    - negative_rules.yaml: 负规则列表
    - veto_gates.yaml: 可用于 gate 的规则
    - audit_report.html: 可视化报告
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path):
    """加载训练好的模型"""
    import pickle

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Loaded model from {model_path}")
    return model


def extract_tree_rules(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    从 LightGBM 模型中提取树规则。

    Returns:
        List of rule dicts with structure:
        {
            'leaf_id': int,
            'path': [(feature, threshold, direction), ...],
            'rule_text': str,
        }
    """
    rules = []

    try:
        # LightGBM
        booster = model.booster_
        tree_info = booster.dump_model()

        for tree_idx, tree in enumerate(tree_info.get("tree_info", [])):
            tree_structure = tree.get("tree_structure", {})
            _extract_rules_from_node(
                tree_structure, feature_names, path=[], rules=rules, tree_idx=tree_idx
            )
    except AttributeError:
        logger.warning("Model does not support rule extraction")

    return rules


def _extract_rules_from_node(
    node: Dict,
    feature_names: List[str],
    path: List[tuple],
    rules: List[Dict],
    tree_idx: int,
):
    """递归提取节点规则"""
    if "leaf_value" in node:
        # 叶节点
        rules.append(
            {
                "tree_idx": tree_idx,
                "leaf_value": node["leaf_value"],
                "n_samples": node.get("leaf_count", 0),
                "path": path.copy(),
                "rule_text": _path_to_text(path, feature_names),
            }
        )
        return

    # 内部节点
    feature_idx = node.get("split_feature")
    threshold = node.get("threshold")

    if feature_idx is not None:
        feature_name = (
            feature_names[feature_idx]
            if feature_idx < len(feature_names)
            else f"feature_{feature_idx}"
        )

        # 左子树 (<=)
        left_path = path + [(feature_name, threshold, "<=")]
        if "left_child" in node:
            _extract_rules_from_node(
                node["left_child"], feature_names, left_path, rules, tree_idx
            )

        # 右子树 (>)
        right_path = path + [(feature_name, threshold, ">")]
        if "right_child" in node:
            _extract_rules_from_node(
                node["right_child"], feature_names, right_path, rules, tree_idx
            )


def _path_to_text(path: List[tuple], feature_names: List[str]) -> str:
    """将路径转换为可读文本"""
    if not path:
        return "ROOT"

    conditions = []
    for feature, threshold, direction in path:
        conditions.append(f"{feature} {direction} {threshold:.4f}")

    return " AND ".join(conditions)


def extract_negative_rules(
    model,
    X: pd.DataFrame,
    rr_series: pd.Series,
    feature_names: List[str],
    delta_rr_threshold: float = -0.2,
    min_coverage: float = 0.02,
    min_samples: int = 100,
) -> List[Dict[str, Any]]:
    """
    提取负规则。

    筛选条件：
    1. mean_rr < 0（叶节点平均收益为负）
    2. delta_rr < threshold（显著差于全局）
    3. coverage > min_coverage（有足够样本）

    Returns:
        List of negative rules with metrics
    """
    # 获取每个样本落入的叶节点
    try:
        leaf_ids = model.predict(X, pred_leaf=True)
    except TypeError:
        # XGBoost 或其他
        leaf_ids = model.apply(X)

    # 处理多树情况
    if len(leaf_ids.shape) > 1:
        # 使用第一棵树或组合
        leaf_ids = leaf_ids[:, 0] if leaf_ids.shape[1] > 0 else leaf_ids.flatten()

    unique_leaves = np.unique(leaf_ids)
    n_total = len(X)
    global_mean = rr_series.mean()

    logger.info(f"Global mean RR: {global_mean:.4f}")
    logger.info(f"Total samples: {n_total}")
    logger.info(f"Unique leaves: {len(unique_leaves)}")

    # 提取树规则
    tree_rules = extract_tree_rules(model, feature_names)
    rule_map = {r.get("leaf_id", i): r for i, r in enumerate(tree_rules)}

    negative_rules = []

    for leaf_id in unique_leaves:
        leaf_mask = leaf_ids == leaf_id
        n_leaf = leaf_mask.sum()
        coverage = n_leaf / n_total

        if n_leaf < min_samples or coverage < min_coverage:
            continue

        leaf_rr = rr_series[leaf_mask]
        mean_rr = leaf_rr.mean()
        std_rr = leaf_rr.std()
        delta_rr = mean_rr - global_mean

        if mean_rr < 0 and delta_rr < delta_rr_threshold:
            rule_info = rule_map.get(leaf_id, {})

            negative_rules.append(
                {
                    "leaf_id": int(leaf_id),
                    "mean_rr": float(mean_rr),
                    "std_rr": float(std_rr),
                    "delta_rr": float(delta_rr),
                    "coverage": float(coverage),
                    "n_samples": int(n_leaf),
                    "rule_text": rule_info.get("rule_text", "N/A"),
                    "path": rule_info.get("path", []),
                }
            )

    # 按 delta_rr 排序（最差的在前）
    negative_rules.sort(key=lambda x: x["delta_rr"])

    logger.info(f"Found {len(negative_rules)} negative rules")

    return negative_rules


def validate_rule_stability(
    rule: Dict[str, Any],
    X: pd.DataFrame,
    rr_series: pd.Series,
    leaf_mask: pd.Series,
    time_col: Optional[str] = None,
    n_time_splits: int = 5,
    perturbation_pct: float = 0.05,
    n_perturbations: int = 10,
    delta_rr_threshold: float = -0.2,
) -> Dict[str, Any]:
    """
    验证单条规则的稳定性。

    Returns:
        dict with time_stable, perturb_stable, veto_level
    """
    global_mean = rr_series.mean()

    # 1. 时间切片稳定性
    time_stable = False
    time_details = []

    if time_col and time_col in X.columns:
        time_values = X[time_col]
        try:
            time_splits = pd.qcut(
                time_values, n_time_splits, labels=False, duplicates="drop"
            )
            actual_splits = time_splits.nunique()

            time_stable_count = 0
            for split_id in range(actual_splits):
                split_mask = (time_splits == split_id) & leaf_mask
                if split_mask.sum() < 10:
                    continue

                split_mean = rr_series[split_mask].mean()
                split_delta = split_mean - global_mean
                time_details.append(
                    {
                        "split": int(split_id),
                        "mean_rr": float(split_mean),
                        "delta_rr": float(split_delta),
                        "is_negative": split_delta < delta_rr_threshold,
                    }
                )
                if split_delta < delta_rr_threshold:
                    time_stable_count += 1

            time_stable = time_stable_count >= actual_splits * 0.6
        except ValueError:
            logger.warning("Could not create time splits")
    else:
        # 无时间列，使用索引位置
        n_samples = len(X)
        split_size = n_samples // n_time_splits

        time_stable_count = 0
        for i in range(n_time_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_time_splits - 1 else n_samples

            idx_mask = pd.Series(False, index=X.index)
            idx_mask.iloc[start_idx:end_idx] = True
            split_mask = idx_mask & leaf_mask

            if split_mask.sum() < 10:
                continue

            split_mean = rr_series[split_mask].mean()
            split_delta = split_mean - global_mean
            time_details.append(
                {
                    "split": i,
                    "mean_rr": float(split_mean),
                    "delta_rr": float(split_delta),
                    "is_negative": split_delta < delta_rr_threshold,
                }
            )
            if split_delta < delta_rr_threshold:
                time_stable_count += 1

        time_stable = time_stable_count >= n_time_splits * 0.6

    # 2. 阈值扰动稳定性
    perturb_stable_count = 0
    leaf_rr = rr_series[leaf_mask]

    for _ in range(n_perturbations):
        # RR 值加噪声后检查是否仍为负
        noise = np.random.normal(0, perturbation_pct * leaf_rr.std(), len(leaf_rr))
        perturbed_rr = leaf_rr + noise
        perturbed_mean = perturbed_rr.mean()
        perturbed_delta = perturbed_mean - global_mean

        if perturbed_delta < delta_rr_threshold:
            perturb_stable_count += 1

    perturb_stable = perturb_stable_count >= n_perturbations * 0.8

    # 3. 确定 veto level
    if time_stable and perturb_stable:
        veto_level = "veto_hard"
    elif time_stable:
        veto_level = "veto_soft"
    else:
        veto_level = "discard"

    return {
        "time_stable": time_stable,
        "perturb_stable": perturb_stable,
        "veto_level": veto_level,
        "time_details": time_details,
        "perturb_pass_rate": perturb_stable_count / n_perturbations,
    }


def generate_veto_gates(
    negative_rules: List[Dict[str, Any]],
    stability_results: Dict[int, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    生成 veto gate 配置。

    Returns:
        dict with 'veto_hard' and 'veto_soft' lists
    """
    veto_gates = {
        "veto_hard": [],
        "veto_soft": [],
    }

    for rule in negative_rules:
        leaf_id = rule["leaf_id"]
        stability = stability_results.get(leaf_id, {})
        veto_level = stability.get("veto_level", "discard")

        if veto_level == "discard":
            continue

        gate_config = {
            "rule_id": f"leaf_{leaf_id}",
            "conditions": rule.get("path", []),
            "rule_text": rule.get("rule_text", ""),
            "metrics": {
                "mean_rr": rule["mean_rr"],
                "delta_rr": rule["delta_rr"],
                "coverage": rule["coverage"],
                "n_samples": rule["n_samples"],
            },
            "stability": {
                "time_stable": stability.get("time_stable", False),
                "perturb_stable": stability.get("perturb_stable", False),
                "perturb_pass_rate": stability.get("perturb_pass_rate", 0),
            },
        }

        veto_gates[veto_level].append(gate_config)

    return veto_gates


def save_results(
    negative_rules: List[Dict[str, Any]],
    veto_gates: Dict[str, List[Dict]],
    output_dir: Path,
):
    """保存结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存负规则
    rules_path = output_dir / "negative_rules.yaml"
    with open(rules_path, "w") as f:
        yaml.dump(
            {"negative_rules": negative_rules},
            f,
            default_flow_style=False,
            allow_unicode=True,
        )
    logger.info(f"Saved negative rules to {rules_path}")

    # 保存 veto gates
    gates_path = output_dir / "veto_gates.yaml"
    with open(gates_path, "w") as f:
        yaml.dump(veto_gates, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Saved veto gates to {gates_path}")

    # 保存摘要
    summary = {
        "total_negative_rules": len(negative_rules),
        "veto_hard_count": len(veto_gates["veto_hard"]),
        "veto_soft_count": len(veto_gates["veto_soft"]),
        "discarded_count": len(negative_rules)
        - len(veto_gates["veto_hard"])
        - len(veto_gates["veto_soft"]),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract negative rules from trained model"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports/audit", help="Output directory"
    )
    parser.add_argument(
        "--delta-rr-threshold", type=float, default=-0.2, help="Delta RR threshold"
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.02, help="Minimum coverage"
    )
    parser.add_argument(
        "--min-samples", type=int, default=100, help="Minimum samples per leaf"
    )

    args = parser.parse_args()

    # 加载模型
    model = load_model(Path(args.model_path))

    logger.info(
        "To run full extraction, load your data and call extract_negative_rules()"
    )
    logger.info(f"Output will be saved to {args.output_dir}")

    # 示例输出结构
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建示例配置
    example_config = {
        "extraction_params": {
            "delta_rr_threshold": args.delta_rr_threshold,
            "min_coverage": args.min_coverage,
            "min_samples": args.min_samples,
        },
        "stability_params": {
            "n_time_splits": 5,
            "perturbation_pct": 0.05,
            "n_perturbations": 10,
        },
    }

    config_path = output_dir / "extraction_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(example_config, f, default_flow_style=False)

    logger.info(f"Saved extraction config to {config_path}")


if __name__ == "__main__":
    main()
