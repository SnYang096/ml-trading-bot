"""Resolve experiment directories under ``results/``, bulk delete, ledger detail JSON."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .scan import _is_ledger_ts


def bulk_delete_paths(results_root: Path, paths: List[str]) -> Dict[str, Any]:
    """删除若干实验目录；路径须通过 `resolve_dashboard_run_dir`。"""
    deleted: List[str] = []
    errors: List[Dict[str, str]] = []
    for rel in paths:
        rel = str(rel).strip().strip("/").replace("\\", "/")
        target = resolve_dashboard_run_dir(results_root, rel)
        if target is None:
            errors.append({"path": rel, "error": "invalid_path"})
            continue
        try:
            shutil.rmtree(target)
            deleted.append(rel)
        except OSError as exc:
            errors.append({"path": rel, "error": str(exc)})
    return {
        "ok": len(errors) == 0,
        "deleted": deleted,
        "errors": errors,
    }


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MiB"
    return f"{n / 1024**3:.2f} GiB"


def resolve_dashboard_run_dir(results_root: Path, ledger_rel: str) -> Optional[Path]:
    """解析可展示/可删除的实验根目录：``…/_rolling_sim/<批次>`` 或单次 ``…/<时间戳叶子>``。"""
    results_root = results_root.resolve()
    ledger_rel = ledger_rel.strip().strip("/").replace("\\", "/")
    if ".." in ledger_rel.split("/"):
        return None
    candidate = (results_root / ledger_rel).resolve()
    try:
        candidate.relative_to(results_root)
    except ValueError:
        return None
    if not candidate.is_dir():
        return None
    if "_rolling_sim" in candidate.parts:
        return candidate
    if _is_ledger_ts(candidate.name):
        return candidate
    return None


def resolve_ledger_dir(results_root: Path, ledger_rel: str) -> Optional[Path]:
    """安全解析 ledger 目录（必须在 results_root 下且路径含 _rolling_sim）。"""
    results_root = results_root.resolve()
    ledger_rel = ledger_rel.strip().strip("/").replace("\\", "/")
    if ".." in ledger_rel.split("/"):
        return None
    candidate = (results_root / ledger_rel).resolve()
    try:
        candidate.relative_to(results_root)
    except ValueError:
        return None
    if "_rolling_sim" not in candidate.parts:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def _batch_root_quick_links(
    batch_dir: Path, results_root: Path
) -> List[Dict[str, str]]:
    """批次根目录快捷链接，顺序与卡片 ``quick-links`` 一致；另附根目录其余 ``trading_map*.html``。"""
    results_root = results_root.resolve()
    out: List[Dict[str, str]] = []
    seen_h: set[str] = set()

    def _push(path: Path, label: str) -> None:
        if not path.is_file():
            return
        try:
            href = "/" + quote(path.relative_to(results_root).as_posix())
        except ValueError:
            return
        if href in seen_h:
            return
        seen_h.add(href)
        out.append({"label": label, "href": href})

    _push(batch_dir / "trading_map_continuous.html", "continuous")
    _push(batch_dir / "trading_map_stitched.html", "stitched")
    _push(batch_dir / "stitched_summary.json", "stitched_summary")
    _push(batch_dir / "report.json", "report.json")
    _push(batch_dir / "pipeline.log", "pipeline.log")

    fixed_names = {
        "trading_map_continuous.html",
        "trading_map_stitched.html",
    }
    extra_cap = 12
    n_extra = 0
    try:
        for p in sorted(batch_dir.iterdir()):
            if not p.is_file():
                continue
            if p.name in fixed_names:
                continue
            if p.name.startswith("trading_map") and p.suffix.lower() == ".html":
                _push(p, p.name)
                n_extra += 1
                if n_extra >= extra_cap:
                    break
    except OSError:
        pass

    return out


def build_ledger_detail_json(results_root: Path, ledger_rel: str) -> Dict[str, Any]:
    """批次根目录快捷链接（不读摘要 JSON、不扫月度子目录），展开即返回。"""
    ld = resolve_dashboard_run_dir(results_root, ledger_rel)
    results_root = results_root.resolve()
    if ld is None:
        return {"error": "invalid_or_missing_ledger", "ledger": ledger_rel}

    quick = _batch_root_quick_links(ld, results_root)
    return {
        "ledger_rel": ledger_rel,
        "quick_links": quick,
        "note": "与卡片「快捷链接」相同；各月阶段产出请在本页路径下自行进入子目录打开。",
    }


def _encode_rel_path_url(rel: str) -> str:
    """URL 路径段（用于静态文件或 ``/browse/…``）。"""
    rel = rel.strip().strip("/").replace("\\", "/")
    if not rel:
        return ""
    return "/".join(quote(seg, safe="") for seg in rel.split("/"))


def _safe_browse_target(results_root: Path, rel_under: str) -> Optional[Path]:
    """解析 ``/browse`` 子路径；禁止跳出 ``results_root``。"""
    results_root = results_root.resolve()
    rel_under = rel_under.strip().strip("/").replace("\\", "/")
    if ".." in rel_under.split("/"):
        return None
    cand = (results_root / rel_under).resolve() if rel_under else results_root
    try:
        cand.relative_to(results_root)
    except ValueError:
        return None
    return cand
