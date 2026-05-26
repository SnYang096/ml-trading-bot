"""Run multiple event_backtest variants from a YAML grid; update EXPERIMENT_INDEX."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_grid(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("variant grid must be a mapping")
    return data


def _run_one(
    *,
    run: Dict[str, Any],
    grid: Dict[str, Any],
    extra_argv: List[str],
) -> int:
    strategy = str(run.get("strategy") or grid.get("strategy") or "tpc")
    variant = str(run["variant"])
    start = str(run["start_date"])
    end = str(run["end_date"])
    strategies_root = str(
        run.get("strategies_root") or grid.get("strategies_root") or "config/strategies"
    )
    out_dir = run.get("output_dir") or (f"results/{strategy}/experiments/{variant}")
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    trades_csv = out_path / f"event_trades_{strategy}.csv"
    symbols = run.get("symbols") or grid.get("symbols")
    sym_arg = (
        ",".join(symbols)
        if isinstance(symbols, list)
        else "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )

    cmd = [
        sys.executable,
        "-m",
        "scripts.event_backtest",
        "--strategy",
        strategy,
        "--symbols",
        sym_arg,
        "--start-date",
        start,
        "--end-date",
        end,
        "--strategies-root",
        strategies_root,
        "--trades-csv",
        str(trades_csv),
        "--capital-report",
        str(out_path),
        "--no-kill-switch",
    ]
    data_path = run.get("data_path") or grid.get("data_path")
    if data_path:
        cmd += ["--data-path", str(data_path)]
    cmd += extra_argv

    print(f"\n=== variant_grid: {variant} ({start} → {end}) ===")
    print(" ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(_REPO_ROOT)).returncode
    return int(rc)


def _write_index(
    *,
    grid_path: Path,
    grid: Dict[str, Any],
    runs_meta: List[Dict[str, Any]],
) -> Path:
    strategy = str(grid.get("strategy") or "tpc")
    index_dir = _REPO_ROOT / "results" / strategy / "experiments"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "EXPERIMENT_INDEX.json"
    payload: Dict[str, Any] = {
        "experiment_id": grid.get("experiment_id")
        or f"variant_grid_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "decision_doc": grid.get("decision_doc"),
        "promoted_variant": grid.get("promoted_variant"),
        "symbols": grid.get("symbols"),
        "grid_yaml": str(grid_path),
        "runs": runs_meta,
    }
    if index_path.exists():
        try:
            prev = json.loads(index_path.read_text(encoding="utf-8"))
            prev_runs = prev.get("runs") or []
            if isinstance(prev_runs, list):
                seen = {r.get("variant") for r in runs_meta if isinstance(r, dict)}
                for pr in prev_runs:
                    if isinstance(pr, dict) and pr.get("variant") not in seen:
                        payload["runs"].append(pr)
        except Exception:
            pass
    index_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return index_path


def run_variant_grid(
    grid_path: Path,
    *,
    extra_argv: List[str] | None = None,
) -> int:
    if not grid_path.is_absolute():
        grid_path = (_REPO_ROOT / grid_path).resolve()
    grid = _load_grid(grid_path)
    runs = grid.get("runs") or []
    if not runs:
        print("ERROR: variant grid has no runs", file=sys.stderr)
        return 3

    extra = list(extra_argv or [])
    runs_meta: List[Dict[str, Any]] = []
    worst_rc = 0
    for run in runs:
        if not isinstance(run, dict):
            continue
        rc = _run_one(run=run, grid=grid, extra_argv=extra)
        worst_rc = max(worst_rc, rc)
        strategy = str(run.get("strategy") or grid.get("strategy") or "tpc")
        variant = str(run["variant"])
        out_dir = run.get("output_dir") or f"results/{strategy}/experiments/{variant}"
        runs_meta.append(
            {
                "variant": variant,
                "period": f"{run['start_date']}/{run['end_date']}",
                "strategies_root": run.get("strategies_root")
                or grid.get("strategies_root"),
                "dir": str(out_dir),
                "exit_code": rc,
            }
        )

    index_path = _write_index(grid_path=grid_path, grid=grid, runs_meta=runs_meta)
    print(f"\nWrote {index_path}")
    return worst_rc
