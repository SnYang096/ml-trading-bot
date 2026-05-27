"""Entry filter threshold plateau scan (auto-loop entry_filters.yaml → research snotio)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.research.execution_kernel.entry_rr_scan import (
    load_strategy_exec_config,
    prepare_entry_rr_frame,
)
from src.research.stat_kernels.snotio_calc import snotio_plateau_payload
from src.time_series_model.execution.entry_filter import load_entry_filters_config

SCANNABLE_OPS = {">=", ">", "<=", "<"}


def find_scannable_conditions(filter_def: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Continuous threshold conditions inside one entry filter."""
    scannable: List[Dict[str, Any]] = []
    for i, cond in enumerate(filter_def.get("conditions", [])):
        if cond.get("operator") in SCANNABLE_OPS:
            scannable.append({"index": i, **cond})
    return scannable


def generate_scan_range(
    current_value: float,
    operator: str,
    n_steps: int = 15,
) -> List[float]:
    """Build threshold grid around the locked/current value."""
    del operator  # range is symmetric; strictness direction handled by scan
    margin = 0.4
    low = current_value - margin
    high = current_value + margin
    if 0.0 <= current_value <= 1.0:
        low = max(0.0, low)
        high = min(1.0, high)
    if abs(high - low) < 1e-8:
        low = current_value - 0.5
        high = current_value + 0.5
    step = (high - low) / max(n_steps - 1, 1)
    return [round(low + i * step, 3) for i in range(n_steps)]


def list_entry_plateau_jobs(
    strategy: str,
    *,
    filter_id: Optional[str] = None,
    strategies_root: str = "config/strategies",
    research: bool = False,
) -> List[Dict[str, Any]]:
    """Return enabled/locked entry filters with scannable conditions."""
    entry_cfg = load_entry_filters_config(strategy, strategies_root, research=research)
    if not entry_cfg:
        return []
    jobs: List[Dict[str, Any]] = []
    for fdef in entry_cfg.get("filters", []):
        if bool(fdef.get("skip_plateau", False)):
            continue
        if not fdef.get("enabled", True) and not fdef.get("locked", False):
            continue
        fid = str(fdef.get("id", ""))
        if filter_id and fid != filter_id:
            continue
        scannable = find_scannable_conditions(fdef)
        if not scannable:
            continue
        jobs.append(
            {
                "filter_id": fid,
                "filter_def": fdef,
                "scannable": scannable,
                "description": fdef.get("description", ""),
            }
        )
    return jobs


def _build_sibling_mask(
    prepared: pd.DataFrame,
    filter_def: Dict[str, Any],
    *,
    skip_index: int,
) -> pd.Series:
    """Mask rows passing all other conditions in the same filter."""
    mask = pd.Series(True, index=prepared.index)
    op_map = {
        ">": lambda s, v: s > v,
        ">=": lambda s, v: s >= v,
        "<": lambda s, v: s < v,
        "<=": lambda s, v: s <= v,
    }
    for i, cond in enumerate(filter_def.get("conditions", [])):
        if i == skip_index:
            continue
        feat = cond.get("feature")
        op_str = cond.get("operator")
        val = cond.get("value")
        if feat not in prepared.columns or op_str not in op_map:
            continue
        s = pd.to_numeric(prepared[feat], errors="coerce")
        mask = mask & op_map[op_str](s, float(val)).fillna(False)
    active = prepared["entry_direction"].astype(float) != 0.0
    return mask & active


def _payload_to_legacy_scan_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    legacy: List[Dict[str, Any]] = []
    for row in rows:
        legacy.append(
            {
                "threshold": row["threshold"],
                "trades": row["trades"],
                "snotio": row.get("snotio", 0.0),
                "sharpe": 0.0,
                "win_rate": 0.0,
                "mean_r": row.get("snotio", 0.0),
                "too_few": bool(row.get("too_few")),
            }
        )
    return legacy


def _plateau_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "is_plateau",
        "reason",
        "start_threshold",
        "end_threshold",
        "plateau_width",
        "confidence",
        "mean_snotio",
        "cv_snotio",
        "cv_trades",
        "cv_trades_warning",
        "mean_trades",
        "recommended",
        "best_single",
    )
    return {k: payload[k] for k in keys if k in payload}


def scan_entry_condition(
    prepared: pd.DataFrame,
    strategy: str,
    filter_def: Dict[str, Any],
    condition: Dict[str, Any],
    *,
    snotio_mode: str = "entry_rr",
    steps: int = 15,
    min_trades: int = 20,
    plateau_window: int = 4,
    strategies_root: str = "config/strategies",
    simple_execution: bool = False,
    exec_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Scan one continuous entry condition; return payload + legacy fields."""
    feature = str(condition["feature"])
    operator = str(condition["operator"])
    current = float(condition["value"])
    cond_index = int(condition["index"])
    grid = generate_scan_range(current, operator, n_steps=steps)
    base_mask = _build_sibling_mask(prepared, filter_def, skip_index=cond_index)
    cfg = exec_config or load_strategy_exec_config(
        strategy, strategies_root=strategies_root, simple=simple_execution
    )
    payload = snotio_plateau_payload(
        prepared,
        feature,
        operator,
        grid,
        base_mask,
        min_trades=min_trades,
        window=max(2, plateau_window),
        snotio_mode=snotio_mode,
        strategy=strategy if snotio_mode == "entry_rr" else None,
        exec_config=cfg if snotio_mode == "entry_rr" else None,
    )
    return {
        "feature": feature,
        "operator": operator,
        "current_value": current,
        "cond_index": cond_index,
        "grid": grid,
        "payload": payload,
        "scan_results": _payload_to_legacy_scan_results(payload.get("rows", [])),
        "plateau": _plateau_from_payload(payload),
    }


def _legacy_all_results(batch_filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in batch_filters:
        fid = f["filter_id"]
        out[fid] = {
            "description": f.get("description", ""),
            "scanned_conditions": [
                {
                    "feature": c["feature"],
                    "operator": c["operator"],
                    "current_value": c["current_value"],
                    "scan_results": c["scan_results"],
                    "plateau": c["plateau"],
                }
                for c in f.get("conditions", [])
            ],
        }
    return out


def run_entry_plateau_batch(
    logs_path: Path | str | None = None,
    strategy: str = "",
    *,
    df: pd.DataFrame | None = None,
    filter_id: Optional[str] = None,
    snotio_mode: str = "entry_rr",
    steps: int = 15,
    min_trades: int = 20,
    plateau_window: int = 4,
    strategies_root: str = "config/strategies",
    research: bool = False,
    simple_execution: bool = False,
    require_gate: bool = False,
    out_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """Auto-loop entry filters; optional per-condition plateau.json under out_dir."""
    if df is None:
        if logs_path is None:
            raise ValueError("logs_path or df required")
        df = pd.read_parquet(logs_path)
    if not strategy:
        raise ValueError("strategy required")

    if (
        require_gate
        and "gate_decision" not in df.columns
        and "gate_ok" not in df.columns
    ):
        raise ValueError("logs missing gate_decision/gate_ok — use logs_gated.parquet")
    if (
        not require_gate
        and "gate_decision" not in df.columns
        and "gate_ok" not in df.columns
    ):
        warnings.warn(
            "no gate_decision/gate_ok — entry RR scan runs without gate veto",
            stacklevel=2,
        )

    prepared = prepare_entry_rr_frame(
        df,
        strategy,
        strategies_root=strategies_root,
        apply_gate=True,
    )
    exec_config = load_strategy_exec_config(
        strategy, strategies_root=strategies_root, simple=simple_execution
    )

    jobs = list_entry_plateau_jobs(
        strategy,
        filter_id=filter_id,
        strategies_root=strategies_root,
        research=research,
    )
    if not jobs:
        raise ValueError(f"no scannable entry filters for strategy={strategy!r}")

    out_path = Path(out_dir) if out_dir else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    batch_filters: List[Dict[str, Any]] = []
    for job in jobs:
        fdef = job["filter_def"]
        cond_results: List[Dict[str, Any]] = []
        for sc in job["scannable"]:
            result = scan_entry_condition(
                prepared,
                strategy,
                fdef,
                sc,
                snotio_mode=snotio_mode,
                steps=steps,
                min_trades=min_trades,
                plateau_window=plateau_window,
                strategies_root=strategies_root,
                simple_execution=simple_execution,
                exec_config=exec_config,
            )
            plateau_path: Optional[str] = None
            if out_path is not None:
                fname = f"{job['filter_id']}_{result['feature']}.json"
                p = out_path / fname
                p.write_text(
                    json.dumps(result["payload"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                plateau_path = str(p)
            cond_results.append({**result, "plateau_path": plateau_path})
        batch_filters.append(
            {
                "filter_id": job["filter_id"],
                "description": job["description"],
                "conditions": cond_results,
            }
        )

    summary: Dict[str, Any] = {
        "strategy": strategy,
        "snotio_mode": snotio_mode,
        "kpi": "snotio",
        "sim": snotio_mode if snotio_mode == "entry_rr" else "proxy",
        "filters": batch_filters,
        "all_results": _legacy_all_results(batch_filters),
    }
    if out_path is not None:
        summary_path = out_path / "entry_plateau_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)
    return summary
