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
        if (
            "## 📜 特征使用规则" in line
            or "## 特征使用规则" in line
            or "## 树模型规则导出" in line
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
    print(f"  ✅ {readme_path}（已写入 {len(rules)} 条规则）")


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
        readme_path = ROOT / "config" / "strategies" / args.strategy / "README.md"

        if not readme_path.exists():
            print(f"  ⚠️  {args.strategy}: no README at {readme_path}")
            return 1

        if not model_path.exists():
            _append_placeholder_section(
                readme_path,
                "未找到 model.pkl。请先运行固定训练（如 mlbot train fixed）并确保 ModelArtifact 保存成功。",
            )
            return 0

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
            return 0

        _append_rules_section(readme_path, rules, args.strategy, str(artifact_dir))
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
