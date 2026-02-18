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


def _generate_risk_gate_yaml(
    output_path: Path, rules: list, strategy: str, model_source: str, top_n: int = 10
):
    """
    从树模型分裂规则生成 risk_gate_draft.yaml 草稿。

    生成的 YAML 包含语义注释，需人工审核修改后使用。
    """
    import yaml

    # 所有规则统一为 hard gate
    hard_gates = []

    for name, thr, op, count in rules[:top_n]:
        # 生成规则 ID
        rule_id = f"gate_{name.replace('.', '_').lower()}"

        # 构造规则
        rule = {
            "id": rule_id,
            "key": name,
            "operator": "<" if op == "<=" else op.replace("<=", "<"),
            "quantile_value": round(thr, 4),
            "tag": f"HARD_{name.upper().replace('.', '_')}",
            "_comment": f"树模型分裂 {count} 次 | 阈值 {thr:.4g} | 需人工确认语义",
        }
        hard_gates.append(rule)

    # 构建完整配置
    config = {
        "_meta": {
            "generated_from": str(model_source),
            "strategy": strategy,
            "note": "此为自动生成的草稿，需人工审核修改后使用",
        },
        "system_safety": {
            "pre_checks": [
                {
                    "id": "check_feature_freshness",
                    "description": "检查特征数据时效性",
                },
            ],
        },
        "hard_gates": {
            "_description": "硬规则：任一触发则拒绝交易",
            "rules": hard_gates,
        },
        "governance": {
            "failure_budget": {
                "max_hard_deny_rate": 0.50,
                "_comment": "硬规则拒绝率上限，超过需审查规则",
            },
        },
    }

    # 使用自定义 dumper 保留注释字段
    yaml_content = yaml.dump(
        config,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )

    output_path.write_text(yaml_content, encoding="utf-8")
    return output_path


def _generate_evidence_candidates_yaml(
    output_path: Path, rules: list, strategy: str, model_source: str, top_n: int = 15
):
    """
    从 Return Tree 分裂规则生成 evidence_candidates.yaml。

    这是 Evidence 轴候选列表，不是 Gate 规则。
    用于后续的 quantile_mapping 配置。

    特性：
    - 聚合同一 feature 的多个分裂阈值
    - 使用语义化标签 (suppress/downweight/neutral/favor/amplify)
    - 添加 usage_hint 和 affects 字段
    """
    import yaml
    from collections import defaultdict

    # 第一步：聚合同一 feature 的多个分裂
    feature_splits = defaultdict(lambda: {"thresholds": [], "count": 0})
    for name, thr, op, count in rules:
        feature_splits[name]["thresholds"].append(round(thr, 4))
        feature_splits[name]["count"] += count

    # 第二步：按总分裂次数排序，取 top_n
    sorted_features = sorted(
        feature_splits.items(), key=lambda x: x[1]["count"], reverse=True
    )[:top_n]

    # 第三步：生成聚合后的候选列表
    candidates = []
    for rank, (feature, data) in enumerate(sorted_features, 1):
        thresholds = sorted(data["thresholds"])
        total_count = data["count"]

        # 生成分布提示
        if len(thresholds) >= 2:
            min_thr, max_thr = min(thresholds), max(thresholds)
            distribution_hint = f"密集分布在 {min_thr:.3g}–{max_thr:.3g} 区间"
        else:
            distribution_hint = f"单一阈值 {thresholds[0]:.4g}"

        candidate = {
            "rank": rank,
            "feature": feature,
            "split_count_total": total_count,
            "threshold_examples": thresholds[:5],  # 最多显示 5 个
            "distribution_hint": distribution_hint,
            "quantile_mapping": {
                "_comment": "树告诉你'区间'，不是点。根据 threshold_examples 定义 bins",
                "bins": [0.2, 0.4, 0.6, 0.8],
                "labels": ["suppress", "downweight", "neutral", "favor", "amplify"],
            },
            "usage_hint": "TODO: 填写用途，如'决定 TP 拉伸与 trail 速度'",
            "affects": {
                "_comment": "标注该 Evidence 影响哪些 execution 参数",
                "candidates": [
                    "tp_range",
                    "trailing_speed",
                    "scale_in_allowed",
                    "position_size",
                ],
            },
        }
        candidates.append(candidate)

    config = {
        "_meta": {
            "generated_from": str(model_source),
            "strategy": strategy,
            "purpose": "Evidence 轴候选列表（从 Return Tree 发现）",
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
        "usage_guide": {
            "step_1": "审核 feature 是否有语义意义",
            "step_2": "根据 threshold_examples 和 distribution_hint 定义 bins",
            "step_3": "填写 usage_hint 和 affects",
            "step_4": "将确认的 Evidence 轴复制到 execution_archetype.yaml",
        },
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
            _generate_risk_gate_yaml(
                risk_gate_path, rules, args.strategy, str(artifact_dir)
            )
            print(f"  ✅ {risk_gate_path} (草稿，需人工审核)")

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
