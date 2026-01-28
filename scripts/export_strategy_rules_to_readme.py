#!/usr/bin/env python3
"""
Export tree model rules to strategy README.md files.

This script:
1. Optionally calls export_tree_rules_imodels.py to generate rules
2. Parses the generated rules.md file
3. Appends rules section to config/strategies/<strategy>/README.md

Usage:
    # From existing rules file
    python scripts/export_strategy_rules_to_readme.py \
        --strategy-config config/strategies/sr_reversal_rr_reg_long \
        --rules-md results/rules_export/.../rules_regression.md

    # Generate rules first, then append to README
    python scripts/export_strategy_rules_to_readme.py \
        --strategy-config config/strategies/sr_reversal_rr_reg_long \
        --features-yaml config/strategies/sr_reversal_rr_reg_long/features_suggested_20260128.yaml \
        --symbol BTCUSDT --timeframe 240T \
        --start-date 2023-01-01 --end-date 2025-12-31 \
        --generate-rules
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Regex to parse rules from markdown table
RULE_LINE_RE = re.compile(
    r"^\|\s*(?P<rank>\d+)\s*\|\s*(?P<coef>[-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*\|\s*(?P<support>[\d.]*(?:[eE][-+]?\d+)?)\s*\|\s*`(?P<rule>.*)`\s*\|\s*$"
)


def _parse_rules_md(md_path: Path) -> List[Tuple[float, float, str]]:
    """Parse rules from markdown file."""
    if not md_path.exists():
        return []
    rules: List[Tuple[float, float, str]] = []
    for line in md_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RULE_LINE_RE.match(line.strip())
        if not m:
            continue
        try:
            coef = float(m.group("coef"))
        except Exception:
            continue
        sup_raw = m.group("support")
        try:
            support = float(sup_raw) if sup_raw else float("nan")
        except Exception:
            support = float("nan")
        rule = (m.group("rule") or "").strip()
        rules.append((coef, support, rule))
    # Sort by absolute coefficient descending
    rules.sort(
        key=lambda t: (abs(t[0]), (t[1] if not math.isnan(t[1]) else -1.0)),
        reverse=True,
    )
    return rules


def _format_rule_for_readme(rule: str) -> str:
    """Format rule string for better readability in README."""
    rule_formatted = rule.replace(" and ", " **AND** ")
    rule_formatted = rule_formatted.replace(" <= ", " ≤ ")
    rule_formatted = rule_formatted.replace(" >= ", " ≥ ")
    rule_formatted = rule_formatted.replace(" < ", " < ")
    rule_formatted = rule_formatted.replace(" > ", " > ")
    return rule_formatted


def _append_rules_to_readme(
    readme_path: Path,
    rules: List[Tuple[float, float, str]],
    strategy: str,
    model_source: str = "",
) -> None:
    """Append rules section to existing README.md."""
    if not readme_path.exists():
        print(f"⚠️  README not found: {readme_path}, creating new one")
        content = f"# {strategy} 策略特征说明\n\n"
    else:
        content = readme_path.read_text(encoding="utf-8")

    # Remove existing rules section if present
    lines = content.split("\n")
    new_lines = []
    in_rules_section = False
    for line in lines:
        if line.strip().startswith("## 📜 特征使用规则") or line.strip().startswith(
            "## 特征使用规则"
        ):
            in_rules_section = True
            continue
        if in_rules_section and line.strip().startswith("##"):
            in_rules_section = False
        if not in_rules_section:
            new_lines.append(line)

    # Remove trailing empty lines
    while new_lines and not new_lines[-1].strip():
        new_lines.pop()

    # Append rules section
    new_lines.append("")
    new_lines.append("---")
    new_lines.append("")
    new_lines.append("## 📜 特征使用规则")
    new_lines.append("")
    new_lines.append(
        "以下规则从训练好的树模型中提取（使用 RuleFit），展示了特征如何组合形成交易信号："
    )
    new_lines.append("")
    new_lines.append("| 规则条件 | 系数 | 支持度 | 说明 |")
    new_lines.append("|---------|------|--------|------|")

    if len(rules) == 0:
        new_lines.append("| (无规则) | - | - | 模型未提供可解释规则 |")
    else:
        for i, (coef, support, rule) in enumerate(rules, 1):
            # Determine rule direction/meaning
            coef_val = float(coef) if not math.isnan(float(coef)) else 0.0
            if coef_val > 0:
                direction = "**正向信号**（系数越大，信号越强）"
            elif coef_val < 0:
                direction = "**负向信号**（系数绝对值越大，抑制越强）"
            else:
                direction = "**中性**"

            rule_formatted = _format_rule_for_readme(rule)
            coef_str = f"{coef_val:.4f}" if not math.isnan(coef_val) else "N/A"
            sup_str = (
                f"{float(support):.2%}"
                if support is not None and not math.isnan(float(support))
                else "N/A"
            )

            new_lines.append(
                f"| `{rule_formatted}` | {coef_str} | {sup_str} | {direction} |"
            )

    new_lines.append("")
    new_lines.append("**说明**：")
    new_lines.append("- **系数**：规则对预测的贡献度，绝对值越大影响越大")
    new_lines.append("- **支持度**：规则在训练数据中的覆盖比例（满足条件的样本占比）")
    new_lines.append("- **规则条件**：多个特征条件的组合，满足所有条件时触发")
    new_lines.append("")
    if model_source:
        new_lines.append(f"**模型来源**：`{model_source}`")
        new_lines.append("")
    new_lines.append(
        "> 💡 **提示**：这些规则是从树模型中提取的简化版本，实际模型可能包含更复杂的非线性组合。"
    )

    readme_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"✅ Rules appended to {readme_path}")


def _generate_rules(
    strategy_config: Path,
    features_yaml: Path,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    test_size: float,
    max_rules: int,
    min_support: float,
    max_conditions: int,
    max_rule_len: int,
    random_state: int,
) -> Path:
    """Call export_tree_rules_imodels.py to generate rules."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "rules_export"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "export_tree_rules_imodels.py"),
            "--strategy-config",
            str(strategy_config),
            "--features-yaml",
            str(features_yaml),
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--test-size",
            str(test_size),
            "--output-dir",
            str(output_dir),
            "--max-rules",
            str(max_rules),
            "--min-support",
            str(min_support),
            "--max-conditions",
            str(max_conditions),
            "--max-rule-len",
            str(max_rule_len),
            "--random-state",
            str(random_state),
        ]
        print(f"▶️ Generating rules: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=str(ROOT), check=True, capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Find generated rules.md file
        strategy_name = strategy_config.name
        rules_dir = output_dir / f"{strategy_name}__imodels_rules"
        md_files = list(rules_dir.glob("rules_*.md"))
        if not md_files:
            raise FileNotFoundError(f"No rules.md found in {rules_dir}")
        return md_files[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export tree model rules to strategy README.md"
    )
    ap.add_argument(
        "--strategy-config",
        required=True,
        help="config/strategies/<strategy> directory",
    )
    ap.add_argument(
        "--rules-md",
        default=None,
        help="Path to existing rules.md file (if not provided and --generate-rules, will generate)",
    )
    ap.add_argument(
        "--generate-rules",
        action="store_true",
        help="Generate rules first using export_tree_rules_imodels.py",
    )
    # Arguments for rule generation
    ap.add_argument(
        "--features-yaml",
        default=None,
        help="Features YAML path (required if --generate-rules)",
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--max-rules", type=int, default=20, help="Maximum rules to export")
    ap.add_argument(
        "--min-support", type=float, default=0.01, help="Minimum support threshold"
    )
    ap.add_argument(
        "--max-conditions", type=int, default=3, help="Maximum conditions per rule"
    )
    ap.add_argument(
        "--max-rule-len", type=int, default=120, help="Maximum rule string length"
    )
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    cfg_dir = Path(args.strategy_config).resolve()
    if not cfg_dir.exists():
        raise FileNotFoundError(f"Strategy config not found: {cfg_dir}")

    # Determine rules.md path
    if args.rules_md:
        rules_md_path = Path(args.rules_md).resolve()
        if not rules_md_path.exists():
            raise FileNotFoundError(f"Rules MD not found: {rules_md_path}")
        model_source = str(args.rules_md)
    elif args.generate_rules:
        if not args.features_yaml:
            raise ValueError("--features-yaml required when --generate-rules")
        if not args.start_date or not args.end_date:
            raise ValueError(
                "--start-date and --end-date required when --generate-rules"
            )
        feat_yaml_path = Path(args.features_yaml).resolve()
        if not feat_yaml_path.exists():
            raise FileNotFoundError(f"Features YAML not found: {feat_yaml_path}")
        rules_md_path = _generate_rules(
            strategy_config=cfg_dir,
            features_yaml=feat_yaml_path,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            test_size=args.test_size,
            max_rules=args.max_rules,
            min_support=args.min_support,
            max_conditions=args.max_conditions,
            max_rule_len=args.max_rule_len,
            random_state=args.random_state,
        )
        model_source = f"Generated from {args.features_yaml}"
    else:
        raise ValueError("Either --rules-md or --generate-rules must be provided")

    # Parse rules
    print(f"📊 Parsing rules from {rules_md_path}")
    rules = _parse_rules_md(rules_md_path)
    print(f"✅ Parsed {len(rules)} rules")

    if len(rules) == 0:
        print("⚠️  No rules found in rules.md file")
        return

    # Append to README
    readme_path = cfg_dir / "README.md"
    _append_rules_to_readme(
        readme_path,
        rules,
        strategy=cfg_dir.name,
        model_source=model_source,
    )

    print(f"✅ Done! Rules exported to {readme_path}")


if __name__ == "__main__":
    main()
