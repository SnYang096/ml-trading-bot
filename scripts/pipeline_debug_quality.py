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
    p = argparse.ArgumentParser(description="Debug monthly quality ranking.")
    p.add_argument("--run-id", required=True, help="rolling_sim run id (timestamp)")
    p.add_argument("--month", required=True, help="month token YYYY-MM")
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

    qf = run_root / f"fast_month_{args.month}" / f"quality_ranking_{args.month}.json"
    if not qf.exists():
        # rolling_sim layout nested under _rolling_sim/<run_id>/fast_month_<month>/...
        cand = list(run_root.glob(f"**/quality_ranking_{args.month}.json"))
        if cand:
            qf = cand[0]
        else:
            print(f"❌ 未找到 quality 文件: {args.month}")
            return 2

    obj = json.loads(qf.read_text(encoding="utf-8"))
    rows = list(obj.get("rankings", []) or [])
    print(f"📄 {qf}")
    print(f"📅 month={obj.get('month')}, rows={len(rows)}")
    print("rank | strategy | quality | sharpe | trades | mean_r")
    print("-----|----------|---------|--------|--------|-------")
    for i, r in enumerate(rows, 1):
        m = r.get("metrics", {}) or {}
        print(
            f"{i:>4d} | {r.get('strategy','')} | {float(r.get('quality_score',0.0)):>7.4f} "
            f"| {float(m.get('sharpe_r',0.0)):>6.3f} | {int(m.get('n_trades',0)):>6d} "
            f"| {float(m.get('mean_r',0.0)):>6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
