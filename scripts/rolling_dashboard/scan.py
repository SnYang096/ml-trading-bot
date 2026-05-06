"""Scan ``results/`` for rolling ledgers and flat runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _is_ledger_ts(name: str) -> bool:
    if not name.startswith(("202", "19")) or "_" not in name:
        return False
    parts = name.split("_")
    if len(parts) < 2 or len(parts[0]) != 8 or len(parts[1]) != 6:
        return False
    return True


def _ledger_summary(ledger_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ledger_ts": ledger_dir.name,
        "bytes_total": 0,
        "has_continuous": False,
        "has_stitched": False,
        "has_summary_json": False,
        "has_pipeline_log": False,
        "has_report_json": False,
        "count_months": None,
        "stitched_total_r": None,
        "stitched_total_trades": None,
        "mode": None,
        "metrics_source": None,
    }
    # 浅层体积估计（只扫批次根 + 一级子目录下文件，避免大目录 rglob 卡死）
    try:
        total_b = 0
        for p in ledger_dir.iterdir():
            if p.is_file():
                total_b += p.stat().st_size
            elif p.is_dir():
                try:
                    for sub in p.iterdir():
                        if sub.is_file():
                            total_b += sub.stat().st_size
                except OSError:
                    pass
        row["bytes_total"] = total_b
    except OSError:
        pass
    row["has_continuous"] = (ledger_dir / "trading_map_continuous.html").is_file()
    row["has_stitched"] = (ledger_dir / "trading_map_stitched.html").is_file()
    row["has_summary_json"] = (ledger_dir / "stitched_summary.json").is_file()
    row["has_pipeline_log"] = (ledger_dir / "pipeline.log").is_file()
    sj = ledger_dir / "stitched_summary.json"
    if sj.is_file():
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
            row["count_months"] = data.get("count_months")
            row["stitched_total_r"] = data.get("stitched_total_r")
            row["stitched_total_trades"] = data.get("stitched_total_trades")
            row["mode"] = data.get("mode")
            row["run_id"] = data.get("run_id")
            row["metrics_source"] = "stitched_summary"
        except (json.JSONDecodeError, OSError):
            pass
    row.setdefault("run_id", None)
    rp = ledger_dir / "report.json"
    if rp.is_file():
        row["has_report_json"] = True
        try:
            rep = json.loads(rp.read_text(encoding="utf-8"))
            bm = (
                rep.get("backtest_metrics")
                if isinstance(rep.get("backtest_metrics"), dict)
                else {}
            )
            if row.get("stitched_total_r") is None and bm.get("mean_r") is not None:
                row["stitched_total_r"] = bm.get("mean_r")
                row["metrics_source"] = row["metrics_source"] or "report_json"
            if (
                row.get("stitched_total_trades") is None
                and bm.get("total_trades") is not None
            ):
                row["stitched_total_trades"] = bm.get("total_trades")
                row["metrics_source"] = row["metrics_source"] or "report_json"
            if row.get("mode") is None and rep.get("rolling") is not None:
                rm = rep.get("rolling")
                if isinstance(rm, dict) and rm.get("mode") is not None:
                    row["mode"] = rm.get("mode")
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    # fast_month_* / slow_snapshot_* 子目录数量（粗略）
    try:
        row["sub_stage_dirs"] = sum(
            1
            for p in ledger_dir.iterdir()
            if p.is_dir()
            and (
                p.name.startswith("fast_month_")
                or p.name.startswith("slow_snapshot_")
                or p.name.startswith("rolling_")
            )
        )
    except OSError:
        row["sub_stage_dirs"] = 0
    row["needs_stitch_cleanup"] = _ledger_needs_stitch_cleanup(ledger_dir)
    return row


def _ledger_needs_stitch_cleanup(ledger_dir: Path) -> bool:
    """尚无可用 Stitch 汇总：无 stitched_summary.json、空文件或非合法 JSON。"""
    sj = ledger_dir / "stitched_summary.json"
    if not sj.is_file():
        return True
    try:
        if sj.stat().st_size == 0:
            return True
        raw = sj.read_text(encoding="utf-8")
        if not raw.strip():
            return True
        data = json.loads(raw)
        if not isinstance(data, dict):
            return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    return False


def scan_rolling_ledgers(
    results_root: Path,
    *,
    strategy_filter: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """返回每个 ledger 目录一条记录，rel_path 为相对 results_root 的 posix 路径。"""
    results_root = results_root.resolve()
    rows: List[Dict[str, Any]] = []
    for roll_parent in results_root.rglob("_rolling_sim"):
        if not roll_parent.is_dir():
            continue
        try:
            children = sorted(
                [
                    p
                    for p in roll_parent.iterdir()
                    if p.is_dir() and _is_ledger_ts(p.name)
                ],
                key=lambda p: p.name,
                reverse=True,
            )
        except OSError:
            continue
        for ledger_dir in children:
            try:
                rel = ledger_dir.relative_to(results_root).as_posix()
            except ValueError:
                continue
            parts = rel.split("/")
            strategy_guess = parts[0] if parts else ""
            if strategy_filter and strategy_guess != strategy_filter:
                continue
            if q and q.lower() not in rel.lower():
                continue
            summary = _ledger_summary(ledger_dir)
            try:
                mtime = ledger_dir.stat().st_mtime
            except OSError:
                mtime = 0.0
            rows.append(
                {
                    "rel_path": rel,
                    "strategy": strategy_guess,
                    "pipeline_dir": parts[1] if len(parts) > 1 else "",
                    "mtime": mtime,
                    **summary,
                }
            )
    rows.sort(key=lambda r: r.get("mtime") or 0.0, reverse=True)
    return rows


def scan_flat_runs(
    results_root: Path,
    *,
    strategy_filter: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """单次实验：``output.history_dir`` 下 ``…/<策略>/<YYYYMMDD_HHMMSS>/``（路径中无 ``_rolling_sim``）。

    与 ``research_pipeline.yaml`` 一致：``results/research_history/<策略>/<时间戳>/``；
    亦匹配 ``…/turbo-rolling-sim/<策略>/<时间戳>/``、``…/prod_train_history/<策略>/<ts>/`` 等。
    """
    results_root = results_root.resolve()
    rows: List[Dict[str, Any]] = []
    try:
        for d in results_root.rglob("*"):
            if not d.is_dir():
                continue
            if "_rolling_sim" in d.parts:
                continue
            if not _is_ledger_ts(d.name):
                continue
            try:
                rel = d.relative_to(results_root).as_posix()
                rel_parts = rel.split("/")
            except ValueError:
                continue
            if len(rel_parts) < 3:
                continue
            if len(rel_parts) > 14:
                continue
            strategy_guess = rel_parts[-2]
            pipeline_seg = "/".join(rel_parts[:-2])
            if strategy_filter and strategy_guess != strategy_filter:
                continue
            if q and q.lower() not in rel.lower():
                continue
            summary = _ledger_summary(d)
            try:
                mtime = d.stat().st_mtime
            except OSError:
                mtime = 0.0
            rows.append(
                {
                    "rel_path": rel,
                    "strategy": strategy_guess,
                    "pipeline_dir": pipeline_seg,
                    "mtime": mtime,
                    "run_kind": "flat",
                    **summary,
                }
            )
    except OSError:
        pass
    return rows


def scan_dashboard_runs(
    results_root: Path,
    *,
    strategy_filter: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rolling = scan_rolling_ledgers(results_root, strategy_filter=strategy_filter, q=q)
    flat = scan_flat_runs(results_root, strategy_filter=strategy_filter, q=q)
    for r in rolling:
        r["run_kind"] = "rolling"
    merged = rolling + flat
    merged.sort(key=lambda r: r.get("mtime") or 0.0, reverse=True)
    return merged


def list_incomplete_rolling_paths(
    results_root: Path,
    *,
    strategy_filter: Optional[str] = None,
    q: Optional[str] = None,
) -> List[str]:
    """待清理 rolling：无有效 ``stitched_summary.json``（缺失、空文件或损坏 JSON）。"""
    rows = scan_rolling_ledgers(results_root, strategy_filter=strategy_filter, q=q)
    return [r["rel_path"] for r in rows if r.get("needs_stitch_cleanup")]


def list_flat_run_paths(
    results_root: Path,
    *,
    strategy_filter: Optional[str] = None,
    q: Optional[str] = None,
) -> List[str]:
    """当前筛选下的 History（单次）批次路径，供一键批量删除。"""
    rows = scan_flat_runs(results_root, strategy_filter=strategy_filter, q=q)
    return [r["rel_path"] for r in rows]


def scan_rolling_run_index(
    results_root: Path,
    *,
    strategy_filter: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """仅枚举批次路径与 mtime，不读 stitched_summary（用于分页卡片 HTML）。"""
    results_root = results_root.resolve()
    rows: List[Dict[str, Any]] = []
    for roll_parent in results_root.rglob("_rolling_sim"):
        if not roll_parent.is_dir():
            continue
        try:
            children = sorted(
                [
                    p
                    for p in roll_parent.iterdir()
                    if p.is_dir() and _is_ledger_ts(p.name)
                ],
                key=lambda p: p.name,
                reverse=True,
            )
        except OSError:
            continue
        for ledger_dir in children:
            try:
                rel = ledger_dir.relative_to(results_root).as_posix()
            except ValueError:
                continue
            parts = rel.split("/")
            strategy_guess = parts[0] if parts else ""
            if strategy_filter and strategy_guess != strategy_filter:
                continue
            if q and q.lower() not in rel.lower():
                continue
            try:
                mtime = ledger_dir.stat().st_mtime
            except OSError:
                mtime = 0.0
            rows.append(
                {
                    "rel_path": rel,
                    "strategy": strategy_guess,
                    "pipeline_dir": parts[1] if len(parts) > 1 else "",
                    "mtime": mtime,
                    "run_kind": "rolling",
                }
            )
    rows.sort(key=lambda r: r.get("mtime") or 0.0, reverse=True)
    return rows


def scan_flat_run_index(
    results_root: Path,
    *,
    strategy_filter: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """单次实验索引：路径 + mtime，不读摘要文件。"""
    results_root = results_root.resolve()
    rows: List[Dict[str, Any]] = []
    try:
        for d in results_root.rglob("*"):
            if not d.is_dir():
                continue
            if "_rolling_sim" in d.parts:
                continue
            if not _is_ledger_ts(d.name):
                continue
            try:
                rel = d.relative_to(results_root).as_posix()
                rel_parts = rel.split("/")
            except ValueError:
                continue
            if len(rel_parts) < 3:
                continue
            if len(rel_parts) > 14:
                continue
            strategy_guess = rel_parts[-2]
            pipeline_seg = "/".join(rel_parts[:-2])
            if strategy_filter and strategy_guess != strategy_filter:
                continue
            if q and q.lower() not in rel.lower():
                continue
            try:
                mtime = d.stat().st_mtime
            except OSError:
                mtime = 0.0
            rows.append(
                {
                    "rel_path": rel,
                    "strategy": strategy_guess,
                    "pipeline_dir": pipeline_seg,
                    "mtime": mtime,
                    "run_kind": "flat",
                }
            )
    except OSError:
        pass
    return rows


def filter_run_index_by_strategy(
    rows: List[Dict[str, Any]], strategy: Optional[str]
) -> List[Dict[str, Any]]:
    """策略 Tab：``None`` / ``__all__`` 表示不过滤。"""
    if not strategy or strategy == "__all__":
        return rows
    return [r for r in rows if (r.get("strategy") or "") == strategy]


def enrich_dashboard_rows(
    results_root: Path, lights: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """对索引行批量附加 `_ledger_summary`（仅当前分页）。"""
    from .paths import resolve_dashboard_run_dir

    out: List[Dict[str, Any]] = []
    for light in lights:
        d = resolve_dashboard_run_dir(results_root, light["rel_path"])
        if d is None:
            continue
        summary = _ledger_summary(d)
        merged = {**light, **summary}
        merged["run_kind"] = light.get("run_kind") or merged.get("run_kind")
        out.append(merged)
    return out
