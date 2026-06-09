"""Run multiple event_backtest variants from a YAML grid; update EXPERIMENT_INDEX.

Engine dispatcher
-----------------

Each ``run:`` entry (or the grid-level fields) may specify ``engine``:

- ``event_backtest`` (default): drives ``scripts.event_backtest`` — used for
  B-system strategies (BPC/TPC/ME/SRB).
- ``chop_grid``: drives ``scripts.chop_grid_backtest`` — used for C-system
  semantic-proxy R&D (each variant = a different ``--config`` chop_grid YAML
  pointing at an alternative ``entry_feature`` or ``max_semantic_chop_*`` band).

Both engines write a per-run output directory and a ``capital_report.json``
(``_new_decision_doc.py`` reads from there). For ``chop_grid`` runs the
``capital_report.json`` is produced by ``chop_grid_backtest`` already
(``write_capital_report_from_trades``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from scripts.event_backtest.market_segment import (
    expand_segment_matrix,
    load_market_segments,
    resolve_segment_run,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ENGINE_EVENT = "event_backtest"
_ENGINE_CHOP_GRID = "chop_grid"
_VALID_ENGINES = (_ENGINE_EVENT, _ENGINE_CHOP_GRID)


def _load_grid(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("variant grid must be a mapping")
    return data


def _resolve_engine(run: Dict[str, Any], grid: Dict[str, Any]) -> str:
    eng = str(run.get("engine") or grid.get("engine") or _ENGINE_EVENT)
    if eng not in _VALID_ENGINES:
        raise ValueError(f"unknown engine {eng!r}; choose from {_VALID_ENGINES}")
    return eng


def _build_event_backtest_cmd(
    *,
    run: Dict[str, Any],
    grid: Dict[str, Any],
    out_path: Path,
    extra_argv: List[str],
) -> List[str]:
    strategy = str(run.get("strategy") or grid.get("strategy") or "tpc")
    strategies_root = str(
        run.get("strategies_root") or grid.get("strategies_root") or "config/strategies"
    )
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
        str(run["start_date"]),
        "--end-date",
        str(run["end_date"]),
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
    if run.get("trading_map") or grid.get("trading_map"):
        map_out = out_path / f"trading_map_{strategy}_event.html"
        cmd += ["--trading-map", str(map_out)]
    if run.get("fast") or grid.get("fast"):
        cmd += ["--fast"]
    constitution_yaml = run.get("constitution_yaml") or grid.get("constitution_yaml")
    if constitution_yaml:
        const_path = Path(str(constitution_yaml))
        if not const_path.is_absolute():
            const_path = (_REPO_ROOT / const_path).resolve()
        cmd += ["--constitution-yaml", str(const_path)]
    inject_scores = run.get("inject_scores") or grid.get("inject_scores")
    if inject_scores:
        inj_path = Path(str(inject_scores))
        if not inj_path.is_absolute():
            inj_path = (_REPO_ROOT / inj_path).resolve()
        cmd += ["--inject-add-ml-scores", str(inj_path)]
    cmd += extra_argv
    return cmd


def _build_chop_grid_cmd(
    *,
    run: Dict[str, Any],
    grid: Dict[str, Any],
    out_path: Path,
    extra_argv: List[str],
) -> List[str]:
    config = run.get("config") or grid.get("config")
    if not config:
        raise ValueError(
            "chop_grid engine requires 'config' (path to grid_backtest YAML)"
        )
    symbols = run.get("symbols") or grid.get("symbols")
    sym_arg = (
        ",".join(symbols)
        if isinstance(symbols, list)
        else "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    cmd = [
        sys.executable,
        "-m",
        "scripts.chop_grid_backtest",
        "--config",
        str(config),
        "--symbols",
        sym_arg,
        "--start",
        str(run["start_date"]),
        "--end",
        str(run["end_date"]),
        "--out-dir",
        str(out_path),
        "--no-maps",
    ]
    timeframe = run.get("timeframe") or grid.get("timeframe")
    if timeframe:
        cmd += ["--timeframe", str(timeframe)]
    cmd += extra_argv
    return cmd


def _run_one(
    *,
    run: Dict[str, Any],
    grid: Dict[str, Any],
    extra_argv: List[str],
) -> int:
    engine = _resolve_engine(run, grid)
    strategy = str(run.get("strategy") or grid.get("strategy") or "tpc")
    variant = str(run["variant"])
    out_dir = run.get("output_dir") or (f"results/{strategy}/experiments/{variant}")
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if engine == _ENGINE_EVENT:
        cmd = _build_event_backtest_cmd(
            run=run, grid=grid, out_path=out_path, extra_argv=extra_argv
        )
    else:
        cmd = _build_chop_grid_cmd(
            run=run, grid=grid, out_path=out_path, extra_argv=extra_argv
        )

    seg = run.get("segment")
    seg_note = f" segment={seg!r}" if seg else ""
    print(
        f"\n=== variant_grid[{engine}]: {variant}{seg_note} "
        f"({run.get('start_date')} → {run.get('end_date')}) ==="
    )
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


def _normalize_runs(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = (
        expand_segment_matrix(grid)
        if grid.get("segment_matrix")
        else (grid.get("runs") or [])
    )
    if not raw:
        return []
    seg_path = grid.get("market_segment_path")
    segments = (
        load_market_segments(seg_path)
        if seg_path or any(r.get("segment") for r in raw)
        else None
    )
    if segments is None and any(isinstance(r, dict) and r.get("segment") for r in raw):
        segments = load_market_segments()
    out: List[Dict[str, Any]] = []
    for run in raw:
        if not isinstance(run, dict):
            continue
        out.append(resolve_segment_run(run, grid=grid, segments=segments))
    return out


def run_variant_grid(
    grid_path: Path,
    *,
    extra_argv: List[str] | None = None,
) -> int:
    if not grid_path.is_absolute():
        grid_path = (_REPO_ROOT / grid_path).resolve()
    grid = _load_grid(grid_path)
    runs = _normalize_runs(grid)
    if not runs:
        print("ERROR: variant grid has no runs", file=sys.stderr)
        return 3

    extra = list(extra_argv or [])
    runs_meta: List[Dict[str, Any]] = []
    worst_rc = 0
    for run in runs:
        rc = _run_one(run=run, grid=grid, extra_argv=extra)
        worst_rc = max(worst_rc, rc)
        strategy = str(run.get("strategy") or grid.get("strategy") or "tpc")
        variant = str(run["variant"])
        out_dir = run.get("output_dir") or f"results/{strategy}/experiments/{variant}"
        runs_meta.append(
            {
                "variant": variant,
                "segment": run.get("segment"),
                "segment_label": run.get("segment_label"),
                "engine": _resolve_engine(run, grid),
                "period": f"{run['start_date']}/{run['end_date']}",
                "strategies_root": run.get("strategies_root")
                or grid.get("strategies_root"),
                "config": run.get("config") or grid.get("config"),
                "dir": str(out_dir),
                "exit_code": rc,
            }
        )

    index_path = _write_index(grid_path=grid_path, grid=grid, runs_meta=runs_meta)
    print(f"\nWrote {index_path}")
    return worst_rc
