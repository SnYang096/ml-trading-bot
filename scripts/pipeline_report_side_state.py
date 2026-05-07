#!/usr/bin/env python3
"""rolling_sim / fast_month 排障：读 pcm_candidates_*.json（不再读 symbol_side_state）。

逐月列出 trend PCM 候选与 multi-leg 独立候选，二者为不同账户池。"""
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
    p = argparse.ArgumentParser(
        description=(
            "rolling_sim 排障：从 pcm_candidates_*.json 汇总 "
            "trend/multi-leg 两个独立候选池。"
        )
    )
    p.add_argument("--run-id", required=True, help="rolling_sim run id (timestamp)")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "pipelines" / "pcm_orchestrate_2h.yaml"),
        help="pipeline config path",
    )
    args = p.parse_args()

    history_dir = _load_history_dir(args.config)
    run_root = history_dir / "_rolling_sim" / str(args.run_id)
    if not run_root.exists():
        print(f"❌ run_id 不存在: {run_root}")
        return 1

    candidate_files = sorted(run_root.glob("**/pcm_candidates_*.json"))
    if not candidate_files:
        print(f"⚠️ 未找到 pcm_candidates_*.json: {run_root}")
        return 2

    print(f"📁 Run Root: {run_root}")
    for rf in candidate_files:
        try:
            obj = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            print(f"⚠️ 读取失败: {rf}")
            continue
        month = obj.get("month", "N/A")
        rows = obj.get("candidates", []) or []
        trend_sel = [
            str((r or {}).get("strategy", ""))
            for r in rows
            if (r or {}).get("trend_pcm_candidate")
        ]
        multi_leg_sel = [
            str((r or {}).get("strategy", ""))
            for r in rows
            if (r or {}).get("multi_leg_pcm_candidate")
        ]
        print(
            f"- {month} | trend_pcm({len(trend_sel)})={trend_sel} "
            f"| multi_leg_pcm({len(multi_leg_sel)})={multi_leg_sel} | {rf}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
