"""Read-only discovery over ``config/experiments/`` (canonical R&D experiment cards)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .constants import PROJECT_ROOT, experiments_root_path

_DIR_RE = re.compile(r"^(\d{8})(?:_(\d{4}))?_(.+)$")
_SPECIAL_PREFIXES = ("_",)
_STRATEGY_HINTS = (
    "fast_scalp_alts",
    "fast_scalp_majors",
    "short_term_swing",
    "fast_scalp",
    "chop_grid",
    "trend_scalp",
    "tpc",
    "bpc",
    "me",
    "srb",
)
_RESULTS_PATH_RE = re.compile(
    r"`(results/[^`\s]+)`|(?:^|\s)(results/[A-Za-z0-9_./-]+)",
    re.MULTILINE,
)
_VERDICT_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("promote", re.compile(r"\bpromote\b|已写入|Promote prod", re.I)),
    ("reject", re.compile(r"\breject\b|不 promote|reject live|不采纳", re.I)),
    ("park", re.compile(r"\bpark\b|暂缓|待 Phase", re.I)),
    ("needs-more", re.compile(r"needs-more|待建|TODO|结论 TODO", re.I)),
)


def _safe_read_text(path: Path, *, limit: int = 200_000) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _parse_dir_name(name: str) -> Dict[str, Optional[str]]:
    m = _DIR_RE.match(name)
    if not m:
        return {"date": None, "time": None, "topic": name}
    date_s, time_s, topic = m.group(1), m.group(2), m.group(3)
    return {
        "date": f"{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}",
        "time": time_s,
        "topic": topic,
    }


def _infer_strategy(topic: str, rd_loop_yaml: Optional[Path]) -> Optional[str]:
    topic_l = topic.lower()
    for hint in _STRATEGY_HINTS:
        if (
            hint in topic_l
            or topic_l.startswith(hint + "_")
            or topic_l.endswith("_" + hint)
        ):
            return hint
    if rd_loop_yaml and rd_loop_yaml.is_file():
        try:
            data = yaml.safe_load(rd_loop_yaml.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict) and data.get("strategy"):
                return str(data["strategy"])
        except (OSError, yaml.YAMLError):
            pass
    parts = topic.split("_")
    if parts:
        return parts[0]
    return None


def _find_decision_file(exp_dir: Path) -> Optional[Path]:
    for name in ("DECISION.md",):
        p = exp_dir / name
        if p.is_file():
            return p
    for p in sorted(exp_dir.glob("DECISION*.md")):
        if p.is_file():
            return p
    for p in sorted(exp_dir.glob("*_experiment_*.md")):
        if p.is_file():
            return p
    return None


def _extract_hypothesis(readme: str) -> str:
    if not readme:
        return ""
    lines = readme.splitlines()
    in_hypothesis = False
    collected: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("## 假设"):
            in_hypothesis = True
            continue
        if in_hypothesis:
            if stripped.startswith("## "):
                break
            if stripped and not stripped.startswith("|"):
                collected.append(stripped)
                if len(collected) >= 3:
                    break
    if collected:
        return " ".join(collected)[:400]
    for line in lines:
        s = line.strip()
        if s.startswith("**") and "目的" in s:
            return re.sub(r"\*\*", "", s)[:400]
        if (
            s
            and not s.startswith("#")
            and not s.startswith("|")
            and not s.startswith("```")
        ):
            if len(s) > 20:
                return s[:400]
    return ""


def _extract_results_links(text: str) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for m in _RESULTS_PATH_RE.finditer(text):
        path = (m.group(1) or m.group(2) or "").strip().rstrip(".,;)")
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _infer_verdict(decision_text: str) -> Optional[str]:
    if not decision_text.strip():
        return None
    for name, pat in _VERDICT_PATTERNS:
        if pat.search(decision_text):
            return name
    return None


def _decision_title(decision_text: str) -> str:
    for line in decision_text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return ""


def _scan_experiment_dir(exp_dir: Path, *, repo_root: Path) -> Optional[Dict[str, Any]]:
    if not exp_dir.is_dir():
        return None
    name = exp_dir.name
    if name.startswith("."):
        return None

    parsed = _parse_dir_name(name)
    readme_path = exp_dir / "README.md"
    decision_path = _find_decision_file(exp_dir)
    rd_loops = sorted(exp_dir.glob("rd_loop_*.yaml"))
    grids = sorted(exp_dir.glob("*_grid.yaml"))
    readme_text = _safe_read_text(readme_path) if readme_path.is_file() else ""
    decision_text = _safe_read_text(decision_path) if decision_path else ""

    rd_loop_rel = str(rd_loops[0].relative_to(repo_root)) if rd_loops else None
    strategy = _infer_strategy(
        parsed.get("topic") or name, rd_loops[0] if rd_loops else None
    )

    rel_dir = str(exp_dir.relative_to(repo_root))
    results_links = _extract_results_links(readme_text + "\n" + decision_text)

    for yf in rd_loops[:1]:
        try:
            ydata = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
            if isinstance(ydata, dict):
                out_dir = ydata.get("output_dir")
                if out_dir:
                    od = str(out_dir).strip()
                    if od and od not in results_links:
                        results_links.insert(0, od)
        except (OSError, yaml.YAMLError):
            pass

    category = "special" if name.startswith(_SPECIAL_PREFIXES) else "experiment"

    return {
        "id": name,
        "category": category,
        "date": parsed.get("date"),
        "time": parsed.get("time"),
        "topic": parsed.get("topic"),
        "strategy": strategy,
        "hypothesis": _extract_hypothesis(readme_text),
        "has_readme": readme_path.is_file(),
        "has_decision": decision_path is not None,
        "verdict": _infer_verdict(decision_text),
        "decision_title": _decision_title(decision_text) if decision_text else "",
        "dir": rel_dir,
        "readme_path": (
            str(readme_path.relative_to(repo_root)) if readme_path.is_file() else None
        ),
        "decision_path": (
            str(decision_path.relative_to(repo_root)) if decision_path else None
        ),
        "rd_loop_yaml": rd_loop_rel,
        "rd_loop_yamls": [str(p.relative_to(repo_root)) for p in rd_loops],
        "grid_yamls": [str(p.relative_to(repo_root)) for p in grids],
        "results_links": results_links,
    }


def _iter_experiment_dirs(experiments_root: Path) -> List[Path]:
    if not experiments_root.is_dir():
        return []
    dirs = [
        p
        for p in experiments_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    return sorted(dirs, key=lambda p: p.name)


@lru_cache(maxsize=4)
def _cached_scan(
    experiments_root_str: str, repo_root_str: str
) -> Tuple[Dict[str, Any], ...]:
    experiments_root = Path(experiments_root_str)
    repo_root = Path(repo_root_str)
    rows: List[Dict[str, Any]] = []
    for exp_dir in _iter_experiment_dirs(experiments_root):
        row = _scan_experiment_dir(exp_dir, repo_root=repo_root)
        if row:
            rows.append(row)
    return tuple(rows)


def clear_experiments_cache() -> None:
    _cached_scan.cache_clear()


def list_experiments(
    *,
    strategy: Optional[str] = None,
    q: Optional[str] = None,
    since: Optional[str] = None,
    category: Optional[str] = None,
    experiments_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List experiment cards from ``config/experiments/``."""
    root = experiments_root or experiments_root_path()
    repo = repo_root or PROJECT_ROOT
    rows = list(_cached_scan(str(root.resolve()), str(repo.resolve())))

    strat_f = (strategy or "").strip().lower()
    q_f = (q or "").strip().lower()
    since_f = (since or "").strip()
    cat_f = (category or "").strip().lower()

    out: List[Dict[str, Any]] = []
    for row in rows:
        if cat_f and row.get("category", "") != cat_f:
            continue
        if strat_f and (row.get("strategy") or "").lower() != strat_f:
            continue
        if since_f and (row.get("date") or "") < since_f:
            continue
        if q_f:
            hay = " ".join(
                str(row.get(k) or "")
                for k in (
                    "id",
                    "topic",
                    "strategy",
                    "hypothesis",
                    "decision_title",
                    "dir",
                )
            ).lower()
            if q_f not in hay:
                continue
        out.append(row)
    return out


def list_strategies(
    experiments_root: Optional[Path] = None, repo_root: Optional[Path] = None
) -> List[str]:
    rows = list_experiments(experiments_root=experiments_root, repo_root=repo_root)
    strategies = sorted({str(r["strategy"]) for r in rows if r.get("strategy")})
    return strategies


def get_experiment(
    experiment_id: str,
    *,
    experiments_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Full experiment card + markdown bodies."""
    root = experiments_root or experiments_root_path()
    repo = repo_root or PROJECT_ROOT
    exp_dir = (root / experiment_id).resolve()
    try:
        exp_dir.relative_to(root.resolve())
    except ValueError:
        return None
    if not exp_dir.is_dir():
        return None

    base = _scan_experiment_dir(exp_dir, repo_root=repo)
    if not base:
        return None

    readme_path = exp_dir / "README.md"
    decision_path = _find_decision_file(exp_dir)
    readme_text = _safe_read_text(readme_path) if readme_path.is_file() else ""
    decision_text = _safe_read_text(decision_path) if decision_path else ""

    yaml_snippets: Dict[str, str] = {}
    for rel in base.get("rd_loop_yamls") or []:
        p = repo / rel
        if p.is_file():
            yaml_snippets[rel] = _safe_read_text(p, limit=50_000)
    for rel in base.get("grid_yamls") or []:
        p = repo / rel
        if p.is_file():
            yaml_snippets[rel] = _safe_read_text(p, limit=50_000)

    return {
        **base,
        "readme_text": readme_text,
        "decision_text": decision_text,
        "yaml_snippets": yaml_snippets,
    }


def get_experiment_raw_file(
    experiment_id: str,
    filename: str,
    *,
    experiments_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Dict[str, str]]:
    """Serve whitelisted markdown from an experiment directory."""
    allowed_suffixes = (".md",)
    if not filename or filename != Path(filename).name:
        return None
    if not filename.endswith(allowed_suffixes):
        return None
    if filename.startswith("."):
        return None

    root = experiments_root or experiments_root_path()
    repo = repo_root or PROJECT_ROOT
    exp_dir = (root / experiment_id).resolve()
    try:
        exp_dir.relative_to(root.resolve())
    except ValueError:
        return None

    target = (exp_dir / filename).resolve()
    try:
        target.relative_to(exp_dir)
    except ValueError:
        return None
    if not target.is_file():
        return None

    return {
        "filename": filename,
        "path": str(target.relative_to(repo)),
        "content": _safe_read_text(target),
    }
