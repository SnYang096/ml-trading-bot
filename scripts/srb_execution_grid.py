#!/usr/bin/env python3
"""
Phase 1 — 读取 config/strategies/bad-candidates/srb/experiment_matrices.yaml，列出执行参数扫描组合。

用法:
  python scripts/srb_execution_grid.py

流程（手工 / CI）:
  1. 复制整份策略目录: cp -a config/strategies/bad-candidates/srb config/strategies/srb_exp_<label>
  2. 编辑副本内 archetypes/execution.yaml 中对应键
  3. 将 rolling / event_backtest 的 --strategies-root 指向该副本
  4. python scripts/srb_experiment_report.py --run-dir <rolling 输出根>
"""
from __future__ import annotations

from pathlib import Path

import yaml

SRB_STRATEGY_DIR = Path("config/strategies/bad-candidates/srb")

_YAML_PATH = {
    "initial_r": "stop_loss.initial_r",
    "activation_r": "stop_loss.trailing.activation_r",
    "trail_r": "stop_loss.trailing.trail_r",
}


def main() -> None:
    p = SRB_STRATEGY_DIR / "experiment_matrices.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    print("# SRB execution grid (edit copies under config/strategies/srb_exp_*)")
    for block_name, block in sorted(data.items()):
        if not isinstance(block, dict):
            continue
        sweep = block.get("sweep")
        if not isinstance(sweep, dict) or len(sweep) != 1:
            continue
        key, values = next(iter(sweep.items()))
        desc = block.get("description", "")
        print(f"\n## {block_name}")
        if desc:
            print(f"   # {desc}")
        ypath = _YAML_PATH.get(str(key), f"stop_loss.{key}")
        src = SRB_STRATEGY_DIR.as_posix()
        for v in values:
            label = f"{block_name}_{key}_{v}".replace(".", "p")
            print(
                f"  - label: {label}\n"
                f"    copy: cp -a {src} config/strategies/srb_exp_{label}\n"
                f"    edit: config/strategies/srb_exp_{label}/archetypes/execution.yaml → {ypath} = {v}"
            )


if __name__ == "__main__":
    main()
