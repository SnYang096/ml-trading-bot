#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_history_dir(config_path: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    rel = (cfg.get("output", {}) or {}).get("history_dir", "results/research_history")
    return PROJECT_ROOT / str(rel)


def main() -> int:
    p = argparse.ArgumentParser(description="Report rolling_sim side state.")
    p.add_argument("--run-id", required=True, help="rolling_sim run id (timestamp)")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "research_pipeline.yaml"),
        help="pipeline config path",
    )
    args = p.parse_args()

    history_dir = _load_history_dir(args.config)
    run_root = history_dir / "_rolling_sim" / str(args.run_id)
    if not run_root.exists():
        print(f"❌ run_id 不存在: {run_root}")
        return 1

    side_files = sorted(run_root.glob("**/symbol_side_state.json"))
    if not side_files:
        print(f"⚠️ 未找到 symbol_side_state.json: {run_root}")
        return 2

    print(f"📁 Run Root: {run_root}")
    for sf in side_files:
        try:
            obj = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            print(f"⚠️ 读取失败: {sf}")
            continue
        month = obj.get("month", "N/A")
        states = obj.get("states", {}) or {}
        counts = {"active": 0, "carry_forward": 0, "disabled": 0}
        for _, v in states.items():
            st = str((v or {}).get("state", "disabled"))
            counts[st] = counts.get(st, 0) + 1
        print(
            f"- {month} | active={counts.get('active',0)} "
            f"carry_forward={counts.get('carry_forward',0)} "
            f"disabled={counts.get('disabled',0)} | {sf}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
