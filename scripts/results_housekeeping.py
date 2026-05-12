#!/usr/bin/env python3
"""Results disk housekeeping helpers.

Use cases:
1) Pre-run disk guard (fail fast when free space is low)
2) Show top result directories by size
3) Prune old run directories while keeping latest N
4) Feature-store cache sanity summary
5) Remove features_labeled.parquet / predictions.parquet / logs_gated.parquet under results/
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

LABEL_PARQUET_NAMES: Tuple[str, ...] = (
    "features_labeled.parquet",
    "logs_gated.parquet",
    "predictions.parquet",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DiskHealth:
    path: Path
    total_gb: float
    used_gb: float
    free_gb: float
    used_pct: float


def disk_health(path: Path) -> DiskHealth:
    usage = shutil.disk_usage(path)
    total_gb = usage.total / (1024**3)
    used_gb = usage.used / (1024**3)
    free_gb = usage.free / (1024**3)
    used_pct = (usage.used / max(usage.total, 1)) * 100.0
    return DiskHealth(
        path=path,
        total_gb=total_gb,
        used_gb=used_gb,
        free_gb=free_gb,
        used_pct=used_pct,
    )


def is_timestamp_dir_name(name: str) -> bool:
    if len(name) < 15:
        return False
    # YYYYMMDD_HHMMSS optionally with suffix.
    core = name[:15]
    return (
        core[8] == "_"
        and core[:8].isdigit()
        and core[9:15].isdigit()
        and core[:2] in {"19", "20"}
    )


def list_timestamp_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    out = [p for p in root.iterdir() if p.is_dir() and is_timestamp_dir_name(p.name)]
    return sorted(out, key=lambda p: p.name)


def list_train_final_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    out = [
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("train_final_")
    ]
    return sorted(out, key=lambda p: p.name)


def collect_default_prune_roots(results_root: Path) -> List[Path]:
    roots: List[Path] = [results_root / "train_final"]
    for strat in ("bpc", "me", "tpc"):
        roots.extend(
            [
                results_root / strat / "validate_static.constrained" / strat,
                results_root / strat / "validate_static.full_study" / strat,
            ]
        )
    return roots


def prune_candidates(root: Path, keep: int) -> List[Path]:
    if root.name == "train_final":
        dirs = list_train_final_dirs(root)
    else:
        dirs = list_timestamp_dirs(root)
    if keep < 0:
        keep = 0
    if len(dirs) <= keep:
        return []
    return dirs[:-keep] if keep > 0 else dirs


def dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def iter_existing(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.exists():
            yield p


def cmd_preflight(args: argparse.Namespace) -> int:
    target = (
        (PROJECT_ROOT / args.path).resolve()
        if not Path(args.path).is_absolute()
        else Path(args.path)
    )
    h = disk_health(target)
    print(
        f"[preflight] path={h.path} used={h.used_pct:.1f}% "
        f"free={h.free_gb:.1f}GiB total={h.total_gb:.1f}GiB"
    )
    if h.free_gb < args.min_free_gb:
        print(
            f"FAIL: free space {h.free_gb:.1f}GiB < min_free_gb {args.min_free_gb:.1f}GiB"
        )
        return 2
    if h.used_pct > args.max_used_pct:
        print(
            f"FAIL: used_pct {h.used_pct:.1f}% > max_used_pct {args.max_used_pct:.1f}%"
        )
        return 2
    print("OK: disk headroom passed")
    return 0


def cmd_top(args: argparse.Namespace) -> int:
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    roots = list(iter_existing(collect_default_prune_roots(results_root)))
    ranking = []
    for root in roots:
        ranking.append((dir_size_bytes(root), root))
    ranking.sort(key=lambda x: x[0], reverse=True)
    print("[top] results roots by size")
    for size, root in ranking[: args.limit]:
        print(f"  {format_gb(size):>10}  {root}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    roots = list(iter_existing(collect_default_prune_roots(results_root)))
    total_deleted = 0
    total_bytes = 0
    for root in roots:
        keep = (
            args.keep_train_final
            if root.name == "train_final"
            else args.keep_per_strategy
        )
        candidates = prune_candidates(root, keep)
        if not candidates:
            continue
        print(f"[prune] {root} (keep={keep})")
        for d in candidates:
            size = dir_size_bytes(d)
            total_bytes += size
            print(f"  - {d} ({format_gb(size)})")
            if not args.dry_run:
                shutil.rmtree(d, ignore_errors=False)
                total_deleted += 1
    mode = "dry-run" if args.dry_run else "apply"
    print(
        f"[prune:{mode}] reclaimable={format_gb(total_bytes)} "
        f"deleted_dirs={total_deleted}"
    )
    return 0


def iter_labeled_parquet_files(results_root: Path) -> List[Path]:
    if not results_root.is_dir():
        return []
    out: List[Path] = []
    names = set(LABEL_PARQUET_NAMES)
    for name in names:
        out.extend(results_root.rglob(name))
    return sorted({p.resolve() for p in out})


def cmd_delete_labeled_parquets(args: argparse.Namespace) -> int:
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    paths = iter_labeled_parquet_files(results_root)
    total_bytes = 0
    for p in paths:
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass
    print(
        f"[delete-labeled-parquets] root={results_root} "
        f"files={len(paths)} size≈{format_gb(total_bytes)}"
    )
    for p in paths:
        print(f"  - {p.relative_to(PROJECT_ROOT)}")
    if args.dry_run:
        print("[delete-labeled-parquets:dry-run] no files removed")
        return 0
    removed = 0
    for p in paths:
        try:
            p.unlink()
            removed += 1
        except OSError as e:
            print(f"  FAIL: {p} ({e})")
            return 1
    print(f"[delete-labeled-parquets] removed={removed}")
    return 0


def cmd_feature_store(args: argparse.Namespace) -> int:
    root = (PROJECT_ROOT / args.root).resolve()
    meta_files = sorted(root.glob("*.meta.json"))
    print(f"[feature-store] root={root} meta_files={len(meta_files)}")
    for mf in meta_files[: args.limit]:
        print(f"  - {mf.name}")
    if len(meta_files) > args.limit:
        print(f"  ... {len(meta_files) - args.limit} more")
    print(
        "Hint: do not delete .meta.json blindly; missing meta can trigger larger rebuild work."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preflight", help="Check disk headroom before heavy runs")
    p_pre.add_argument(
        "--path", default=".", help="Path to check (default: project root)"
    )
    p_pre.add_argument("--min-free-gb", type=float, default=120.0)
    p_pre.add_argument("--max-used-pct", type=float, default=90.0)
    p_pre.set_defaults(func=cmd_preflight)

    p_top = sub.add_parser("top", help="Show biggest result roots")
    p_top.add_argument("--results-root", default="results")
    p_top.add_argument("--limit", type=int, default=12)
    p_top.set_defaults(func=cmd_top)

    p_prune = sub.add_parser("prune", help="Prune old runs from default roots")
    p_prune.add_argument("--results-root", default="results")
    p_prune.add_argument("--keep-train-final", type=int, default=8)
    p_prune.add_argument("--keep-per-strategy", type=int, default=6)
    p_prune.add_argument("--dry-run", action="store_true")
    p_prune.set_defaults(func=cmd_prune)

    p_fs = sub.add_parser("feature-store", help="Feature-store cache summary")
    p_fs.add_argument("--root", default="feature_store")
    p_fs.add_argument("--limit", type=int, default=20)
    p_fs.set_defaults(func=cmd_feature_store)

    p_del = sub.add_parser(
        "delete-labeled-parquets",
        help="Delete features_labeled.parquet, predictions.parquet, logs_gated.parquet under results/",
    )
    p_del.add_argument("--results-root", default="results")
    p_del.add_argument("--dry-run", action="store_true")
    p_del.set_defaults(func=cmd_delete_labeled_parquets)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
