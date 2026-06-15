"""Run multiple event_backtest variants from a YAML grid; update EXPERIMENT_INDEX.

Engine dispatcher
-----------------

Each ``run:`` entry (or the grid-level fields) may specify ``engine``:

- ``event_backtest`` (default): drives ``scripts.event_backtest`` — used for
  B-system strategies (BPC/TPC/ME/SRB).
- ``chop_grid``: drives ``scripts.chop_grid_backtest`` — used for C-system
  chop grid R&D.
- ``trend_scalp``: drives ``scripts.diagnose_dual_add_trend`` — used for C-system
  trend scalp R&D.
- ``multileg_joint``: runs chop_grid + trend_scalp per segment, then replays
  both on a shared account timeline via ``sim_multileg_account.py``.

All engines write per-run output under the configured ``output_dir``.
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
_ENGINE_TREND_SCALP = "trend_scalp"
_ENGINE_MULTILEG_JOINT = "multileg_joint"
_VALID_ENGINES = (
    _ENGINE_EVENT,
    _ENGINE_CHOP_GRID,
    _ENGINE_TREND_SCALP,
    _ENGINE_MULTILEG_JOINT,
)


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
    etf = run.get("execution_timeframe") or grid.get("execution_timeframe")
    if etf:
        cmd += ["--execution-timeframe", str(etf)]
    ic = run.get("initial_capital") or grid.get("initial_capital")
    if ic:
        cmd += ["--initial-capital", str(ic)]
    run_extra = run.get("extra_argv") or []
    cmd += [str(x) for x in run_extra]
    cmd += extra_argv
    return cmd


def _build_trend_scalp_cmd(
    *,
    run: Dict[str, Any],
    grid: Dict[str, Any],
    out_path: Path,
    extra_argv: List[str],
) -> List[str]:
    config = run.get("config") or grid.get("config")
    if not config:
        raise ValueError("trend_scalp engine requires 'config'")
    symbols = run.get("symbols") or grid.get("symbols")
    sym_arg = ",".join(symbols) if isinstance(symbols, list) else str(symbols)
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "diagnose_dual_add_trend.py"),
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
    ]
    tf = run.get("timeframe") or grid.get("timeframe") or "2h"
    cmd += ["--timeframe", str(tf)]
    etf = run.get("execution_timeframe") or grid.get("execution_timeframe")
    if etf:
        cmd += ["--execution-timeframe", str(etf)]
    ic = run.get("initial_capital") or grid.get("initial_capital")
    if ic:
        cmd += ["--initial-capital", str(ic)]
    cmd += extra_argv
    return cmd


def _run_multileg_joint_pipeline(
    grid: Dict[str, Any],
    extra_argv: List[str],
) -> int:
    segments = load_market_segments(
        grid.get("market_segment_path", "config/market_segment.yaml")
    )
    seg_ids = grid.get("segments") or list(segments)
    if not seg_ids:
        print("ERROR: multileg_joint requires segments list", file=sys.stderr)
        return 3

    backend_argv = []
    skip = False
    for a in extra_argv:
        if skip:
            skip = False
            continue
        if a in ("--segments",):
            skip = True
            continue
        backend_argv.append(a)

    out_root = Path(str(grid.get("output_dir", "results/multileg_joint")))
    if not out_root.is_absolute():
        out_root = (_REPO_ROOT / out_root).resolve()

    chop_cfg = grid.get("chop_config") or grid.get("config")
    trend_cfg = grid.get("trend_config") or grid.get("config")
    symbols = grid.get(
        "symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    )
    constitution_yaml = grid.get("constitution_yaml")
    trend_extra = list(grid.get("trend_extra_argv") or [])

    worst_rc = 0
    for seg_id in seg_ids:
        seg = segments.get(seg_id)
        if not seg:
            continue

        chop_out = out_root / "chop_grid" / seg_id
        chop_out.mkdir(parents=True, exist_ok=True)
        chop_cmd = _build_chop_grid_cmd(
            run={
                "start_date": seg["start_date"],
                "end_date": seg["end_date"],
                "config": chop_cfg,
                "symbols": symbols,
            },
            grid=grid,
            out_path=chop_out,
            extra_argv=backend_argv,
        )
        print(f"\n=== multileg_joint[chop_grid]: {seg_id} ===")
        print(" ".join(chop_cmd))
        rc = subprocess.run(chop_cmd, cwd=str(_REPO_ROOT)).returncode
        worst_rc = max(worst_rc, rc)

        trend_out = out_root / "trend_scalp" / seg_id
        trend_out.mkdir(parents=True, exist_ok=True)
        trend_cmd = _build_trend_scalp_cmd(
            run={
                "start_date": seg["start_date"],
                "end_date": seg["end_date"],
                "config": trend_cfg,
                "symbols": symbols,
            },
            grid=grid,
            out_path=trend_out,
            extra_argv=trend_extra + backend_argv,
        )
        print(f"\n=== multileg_joint[trend_scalp]: {seg_id} ===")
        print(" ".join(trend_cmd))
        rc = subprocess.run(trend_cmd, cwd=str(_REPO_ROOT)).returncode
        worst_rc = max(worst_rc, rc)

    joint_out = out_root / "joint"
    joint_out.mkdir(parents=True, exist_ok=True)
    joint_cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "sim_multileg_account.py"),
        "--chop-root",
        str(out_root / "chop_grid"),
        "--trend-root",
        str(out_root / "trend_scalp"),
        "--segments",
        *seg_ids,
    ]
    if constitution_yaml:
        joint_cmd += ["--constitution-yaml", str(constitution_yaml)]
        joint_cmd += ["--with-constitution"]
    if grid.get("max_concurrent_multi_leg_symbols"):
        joint_cmd += [
            "--max-concurrent-multi-leg-symbols",
            str(grid["max_concurrent_multi_leg_symbols"]),
        ]
    if grid.get("equity"):
        joint_cmd += ["--equity", str(grid["equity"])]
    print(f"\n=== multileg_joint: account simulation ===")
    print(" ".join(joint_cmd))
    rc = subprocess.run(joint_cmd, cwd=str(_REPO_ROOT)).returncode
    worst_rc = max(worst_rc, rc)
    print(f"\nJoint results: {joint_out}")
    return worst_rc


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
    elif engine == _ENGINE_TREND_SCALP:
        cmd = _build_trend_scalp_cmd(
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

    engine = _resolve_engine(grid, grid)
    if engine == _ENGINE_MULTILEG_JOINT:
        return _run_multileg_joint_pipeline(grid, list(extra_argv or []))

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
