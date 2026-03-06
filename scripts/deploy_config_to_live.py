#!/usr/bin/env python3
"""
DEPLOY 脚本 — 将研究确认的 config/strategies/ 部署到 live/highcap/config/strategies/

功能:
  1. 对比 config/strategies/ vs live/highcap/config/strategies/ 差异
  2. 显示 YAML 级别 diff (逐 key 对比)
  3. 确认后自动复制
  4. 可选: 自动 git commit live/ 目录变更
  5. 回滚: git revert 即可恢复

用法:
    # 查看差异 (不复制)
    python scripts/deploy_config_to_live.py --diff

    # 部署所有策略
    python scripts/deploy_config_to_live.py --deploy

    # 部署指定策略
    python scripts/deploy_config_to_live.py --deploy --strategy bpc

    # 部署 + 自动 git commit
    python scripts/deploy_config_to_live.py --deploy --git-commit

    # 非交互模式 (CI/自动化)
    python scripts/deploy_config_to_live.py --deploy --yes

    # 回滚上一次部署
    python scripts/deploy_config_to_live.py --rollback
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# ====================================================================
# Paths
# ====================================================================

RESEARCH_CONFIG = PROJECT_ROOT / "config"
RESEARCH_STRATEGIES = RESEARCH_CONFIG / "strategies"
LIVE_CONFIG = PROJECT_ROOT / "live" / "highcap" / "config"
LIVE_STRATEGIES = LIVE_CONFIG / "strategies"
LIVE_ROOT = PROJECT_ROOT / "live"

# 部署的策略列表 (不含 LV, 暂缓)
DEFAULT_STRATEGIES = ["bpc", "me", "fer"]

# 需要同步的文件 (archetypes 目录 + 顶层配置)
ARCHETYPE_FILES = [
    "gate.yaml",
    "evidence.yaml",
    "entry_filters.yaml",
    "execution.yaml",
    "direction.yaml",
    "prefilter.yaml",
    "holding.yaml",
]

TOP_LEVEL_CONFIGS = [
    "meta.yaml",
    "model.yaml",
    "labels.yaml",
    "labels_return_tree.yaml",
    "labels_rr_extreme.yaml",
    "backtest.yaml",
    "features.yaml",
    "features_gate.yaml",
    "features_evidence.yaml",
    "training_baseline.json",  # P5: OOD baseline (feature distributions q05/q95)
]

# 全局配置: config/ 下的非策略配置 → live/highcap/config/
# (相对路径, 相对于 config/ 根目录)
GLOBAL_CONFIGS = [
    "constitution/constitution.yaml",
    "pcm_regime.yaml",
]


# ====================================================================
# YAML Diff
# ====================================================================


def yaml_diff(d1: dict, d2: dict, prefix: str = "  ") -> List[str]:
    """递归对比两个 dict, 返回差异行."""
    lines = []
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        if d1 != d2:
            lines.append(f"{prefix}~ {d1} → {d2}")
        return lines

    all_keys = sorted(set(list(d1.keys()) + list(d2.keys())))
    for k in all_keys:
        v1, v2 = d1.get(k), d2.get(k)
        if v1 == v2:
            continue
        if k not in d1:
            lines.append(f"{prefix}+ {k}: {v2}")
        elif k not in d2:
            lines.append(f"{prefix}- {k}: {v1}")
        elif isinstance(v1, dict) and isinstance(v2, dict):
            lines.append(f"{prefix}{k}:")
            lines.extend(yaml_diff(v1, v2, prefix + "  "))
        elif isinstance(v1, list) and isinstance(v2, list):
            if v1 != v2:
                lines.append(
                    f"{prefix}~ {k}: (list changed, {len(v1)} → {len(v2)} items)"
                )
        else:
            lines.append(f"{prefix}~ {k}: {v1} → {v2}")
    return lines


def compare_file(src: Path, dst: Path) -> Tuple[str, List[str]]:
    """对比两个文件, 返回 (status, diff_lines).

    status: 'identical' | 'modified' | 'new' | 'missing_src'
    """
    if not src.exists():
        return "missing_src", [f"  (源文件不存在: {src})"]
    if not dst.exists():
        return "new", [f"  (新文件, live 中不存在)"]

    src_text = src.read_text(encoding="utf-8")
    dst_text = dst.read_text(encoding="utf-8")

    if src_text == dst_text:
        return "identical", []

    # YAML-level diff
    try:
        y_src = yaml.safe_load(src_text) or {}
        y_dst = yaml.safe_load(dst_text) or {}
        diff_lines = yaml_diff(y_dst, y_src)
        return "modified", diff_lines
    except Exception:
        return "modified", ["  (YAML 解析失败, 文件内容不同)"]


# ====================================================================
# Diff command
# ====================================================================


def cmd_diff(strategies: List[str], include_global: bool = True) -> Dict[str, dict]:
    """对比 config/ vs live/ 差异, 返回摘要."""
    summary = {}

    for strat in strategies:
        src_dir = RESEARCH_STRATEGIES / strat
        dst_dir = LIVE_STRATEGIES / strat

        if not src_dir.exists():
            print(f"\n⚠️  {strat.upper()}: 研究 config 不存在 ({src_dir})")
            continue

        print(f"\n{'─'*70}")
        print(
            f"📋 {strat.upper()}: config/strategies/{strat} vs live/highcap/config/strategies/{strat}"
        )
        print(f"{'─'*70}")

        strat_summary = {"new": 0, "modified": 0, "identical": 0, "files": {}}

        # archetypes
        src_arch = src_dir / "archetypes"
        dst_arch = dst_dir / "archetypes"

        if src_arch.exists():
            for fname in ARCHETYPE_FILES:
                src_file = src_arch / fname
                dst_file = dst_arch / fname
                if not src_file.exists():
                    continue

                status, diff_lines = compare_file(src_file, dst_file)
                strat_summary[status] = strat_summary.get(status, 0) + 1
                strat_summary["files"][f"archetypes/{fname}"] = status

                if status == "identical":
                    print(f"  ✅ archetypes/{fname}: 无变化")
                elif status == "new":
                    print(f"  🆕 archetypes/{fname}: 新文件")
                elif status == "modified":
                    print(f"  ⚡ archetypes/{fname}: 有差异")
                    for line in diff_lines[:10]:
                        print(f"    {line}")
                    if len(diff_lines) > 10:
                        print(f"    ... 还有 {len(diff_lines) - 10} 行差异")

        # top-level configs
        for fname in TOP_LEVEL_CONFIGS:
            src_file = src_dir / fname
            dst_file = dst_dir / fname
            if not src_file.exists():
                continue

            status, diff_lines = compare_file(src_file, dst_file)
            strat_summary[status] = strat_summary.get(status, 0) + 1
            strat_summary["files"][fname] = status

            if status == "identical":
                print(f"  ✅ {fname}: 无变化")
            elif status == "new":
                print(f"  🆕 {fname}: 新文件")
            elif status == "modified":
                print(f"  ⚡ {fname}: 有差异")
                for line in diff_lines[:5]:
                    print(f"    {line}")

        # Summary for this strategy
        total_changes = strat_summary["new"] + strat_summary["modified"]
        if total_changes == 0:
            print(f"\n  ➡️  {strat.upper()}: 完全同步, 无需部署")
        else:
            print(
                f"\n  ➡️  {strat.upper()}: {total_changes} 个文件需要更新 ({strat_summary['new']} 新 + {strat_summary['modified']} 修改)"
            )

        summary[strat] = strat_summary

    # ── 全局配置 diff ──
    if include_global:
        print(f"\n{'─'*70}")
        print(f"📋 GLOBAL: config/ vs live/highcap/config/ (宪法/PCM/gate 等)")
        print(f"{'─'*70}")

        global_summary = {"new": 0, "modified": 0, "identical": 0, "files": {}}
        for rel_path in GLOBAL_CONFIGS:
            src_file = RESEARCH_CONFIG / rel_path
            dst_file = LIVE_CONFIG / rel_path
            if not src_file.exists():
                continue

            status, diff_lines = compare_file(src_file, dst_file)
            global_summary[status] = global_summary.get(status, 0) + 1
            global_summary["files"][rel_path] = status

            if status == "identical":
                print(f"  ✅ {rel_path}: 无变化")
            elif status == "new":
                print(f"  🆕 {rel_path}: 新文件 (live 中不存在)")
            elif status == "modified":
                print(f"  ⚡ {rel_path}: 有差异")
                for line in diff_lines[:8]:
                    print(f"    {line}")
                if len(diff_lines) > 8:
                    print(f"    ... 还有 {len(diff_lines) - 8} 行差异")

        total_global = global_summary["new"] + global_summary["modified"]
        if total_global == 0:
            print(f"\n  ➡️  GLOBAL: 完全同步, 无需部署")
        else:
            print(f"\n  ➡️  GLOBAL: {total_global} 个文件需要更新")

        summary["__global__"] = global_summary

    return summary


# ====================================================================
# Deploy command
# ====================================================================


def deploy_strategy(strat: str) -> int:
    """部署单个策略: config/strategies/{strat}/ → live/highcap/config/strategies/{strat}/"""
    src_dir = RESEARCH_STRATEGIES / strat
    dst_dir = LIVE_STRATEGIES / strat

    if not src_dir.exists():
        print(f"  ❌ 研究 config 不存在: {src_dir}")
        return 0

    copied = 0

    # archetypes
    src_arch = src_dir / "archetypes"
    dst_arch = dst_dir / "archetypes"
    if src_arch.exists():
        dst_arch.mkdir(parents=True, exist_ok=True)
        for fname in ARCHETYPE_FILES:
            src_file = src_arch / fname
            if src_file.exists():
                shutil.copy2(src_file, dst_arch / fname)
                copied += 1

    # top-level configs
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in TOP_LEVEL_CONFIGS:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)
            copied += 1

    return copied


def deploy_global_configs() -> int:
    """部署全局配置: config/{path} → live/highcap/config/{path}"""
    copied = 0
    for rel_path in GLOBAL_CONFIGS:
        src_file = RESEARCH_CONFIG / rel_path
        dst_file = LIVE_CONFIG / rel_path
        if not src_file.exists():
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1
    return copied


def cmd_deploy(strategies: List[str], auto_yes: bool = False, git_commit: bool = False):
    """执行部署."""
    # 先显示 diff
    summary = cmd_diff(strategies)

    # 检查是否有变更
    total_changes = sum(
        s.get("new", 0) + s.get("modified", 0) for s in summary.values()
    )

    if total_changes == 0:
        print(f"\n✅ 所有策略已同步, 无需部署")
        return

    # 确认
    if not auto_yes:
        print(f"\n{'='*70}")
        response = (
            input(f"🚀 确认部署 {total_changes} 个文件到 live/? (y/N): ")
            .strip()
            .lower()
        )
        if response not in ("y", "yes"):
            print("❌ 已取消")
            return

    # 执行部署
    print(f"\n{'='*70}")
    print("🚀 执行部署...")
    print(f"{'='*70}")

    total_copied = 0
    for strat in strategies:
        if strat not in summary:
            continue
        s = summary[strat]
        if s.get("new", 0) + s.get("modified", 0) == 0:
            print(f"  ⏭️  {strat.upper()}: 已同步, 跳过")
            continue

        n = deploy_strategy(strat)
        total_copied += n
        print(f"  ✅ {strat.upper()}: 已部署 {n} 个文件")

    # 全局配置
    gs = summary.get("__global__", {})
    if gs.get("new", 0) + gs.get("modified", 0) > 0:
        n = deploy_global_configs()
        total_copied += n
        print(f"  ✅ GLOBAL: 已部署 {n} 个全局配置 (constitution/pcm_regime/...)")
    else:
        print(f"  ⏭️  GLOBAL: 已同步, 跳过")

    print(f"\n✅ 部署完成: {total_copied} 个文件已更新")

    # Git commit
    if git_commit:
        _git_commit_live(strategies)


def _git_commit_live(strategies: List[str]):
    """自动 git commit live/ 目录变更."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strats = "+".join(s.upper() for s in strategies)
    msg = f"deploy: {strats} config → live ({timestamp})"

    print(f"\n📝 Git commit: {msg}")
    try:
        subprocess.run(
            ["git", "add", "live/highcap/config/"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            print("  ⏭️  无 staged 变更, 跳过 commit")
            return

        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        print(f"  ✅ Git commit 成功: {msg}")
        print(f"  💡 回滚: git revert HEAD")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Git commit 失败: {e}")


# ====================================================================
# Rollback command
# ====================================================================


def cmd_rollback():
    """显示回滚指引."""
    print("\n🔄 回滚指引:")
    print(f"{'─'*70}")
    print("  # 查看最近的部署 commit")
    print("  git log --oneline -10 -- live/highcap/config/")
    print()
    print("  # 回滚上一次部署")
    print("  git revert HEAD")
    print()
    print("  # 回滚指定 commit")
    print("  git revert <commit-hash>")
    print()
    print("  # 强制恢复到某个版本 (谨慎)")
    print("  git checkout <commit-hash> -- live/highcap/config/strategies/")

    # 显示最近的 deploy commits
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-5", "--", "live/highcap/config/"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(f"\n📋 最近的 live/config 变更:")
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
    except Exception:
        pass


# ====================================================================
# Main
# ====================================================================


def main():
    p = argparse.ArgumentParser(
        description="DEPLOY: config/strategies/ → live/highcap/config/strategies/"
    )
    p.add_argument("--diff", action="store_true", help="只查看差异, 不部署")
    p.add_argument("--deploy", action="store_true", help="执行部署 (对比 + 复制)")
    p.add_argument("--rollback", action="store_true", help="显示回滚指引")
    p.add_argument(
        "--strategy",
        "-s",
        nargs="+",
        help=f"指定策略 (默认: {' '.join(DEFAULT_STRATEGIES)})",
    )
    p.add_argument("--yes", "-y", action="store_true", help="非交互模式, 跳过确认")
    p.add_argument(
        "--git-commit", action="store_true", help="部署后自动 git commit live/ 变更"
    )
    args = p.parse_args()

    strategies = args.strategy or DEFAULT_STRATEGIES

    print("=" * 70)
    print("🚀 DEPLOY: 研究 config → 实盘 config")
    print("=" * 70)
    print(f"   源:   config/strategies/")
    print(f"   目标: live/highcap/config/strategies/")
    print(f"   策略: {', '.join(s.upper() for s in strategies)}")

    if args.rollback:
        cmd_rollback()
    elif args.deploy:
        cmd_deploy(strategies, auto_yes=args.yes, git_commit=args.git_commit)
    elif args.diff:
        cmd_diff(strategies)
    else:
        # 默认显示 diff
        cmd_diff(strategies)
        print(f"\n💡 要执行部署, 请添加 --deploy 参数")
        print(f"   python scripts/deploy_config_to_live.py --deploy")


if __name__ == "__main__":
    main()
