#!/usr/bin/env python3
"""
将分散的 train_final_* 训练产物目录收纳到统一前缀，便于批量删除或归档。

默认只处理「新布局」：
  results/<策略>/train_final_<时间戳>_<标签>/
→ results/train_final/<策略>/train_final_<时间戳>_<标签>/

示例（收纳 me 策略下所有 train_final）::
    python scripts/collect_train_final_dirs.py --strategy me --execute

预览（不移动）::
    python scripts/collect_train_final_dirs.py --strategy me
    python scripts/collect_train_final_dirs.py --all-strategies
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# 不作为「策略根目录」扫描的 results 直子目录
_SKIP_RESULT_CHILDREN = frozenset(
    {
        "train_final",
        "logs",
        "research_history",
        "browse",
    }
)


def iter_train_final_sources(
    results_root: Path, *, strategies: set[str] | None
) -> list[tuple[str, Path]]:
    """Return (strategy_slug, src_path) sorted by strategy then name."""
    out: list[tuple[str, Path]] = []
    results_root = results_root.resolve()
    if not results_root.is_dir():
        return out

    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("."):
            continue
        if name in _SKIP_RESULT_CHILDREN:
            continue
        if strategies is not None and name not in strategies:
            continue

        for tf in sorted(child.glob("train_final_*")):
            if tf.is_dir():
                out.append((name, tf))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="收纳 results/<策略>/train_final_* → results/train_final/<策略>/",
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="results 目录（默认仓库根下 results/）",
    )
    ap.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        default=[],
        metavar="SLUG",
        help="只处理该策略（可重复）",
    )
    ap.add_argument(
        "--all-strategies",
        action="store_true",
        help="扫描 results 下所有含 train_final_* 的策略子目录",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="实际移动（默认仅打印计划）",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="若目标已存在则先删除再移动（危险）",
    )
    args = ap.parse_args()

    proj = _project_root()
    rr = (args.results_root or (proj / "results")).resolve()

    strat_filter: set[str] | None
    if args.strategies:
        strat_filter = {s.strip() for s in args.strategies if str(s).strip()}
        if not strat_filter:
            strat_filter = None
    elif args.all_strategies:
        strat_filter = None
    else:
        print("请指定 --strategy SLUG 或 --all-strategies", file=sys.stderr)
        return 2

    pairs = iter_train_final_sources(rr, strategies=strat_filter)
    if not pairs:
        print(f"未发现 train_final_* 源目录（under {rr}）。")
        return 0

    dest_root = rr / "train_final"
    moves: list[tuple[Path, Path]] = []
    for slug, src in pairs:
        dest = dest_root / slug / src.name
        moves.append((src, dest))

    for src, dest in moves:
        rel_src = src.relative_to(rr)
        rel_dest = dest.relative_to(rr)
        print(f"  {rel_src}  →  {rel_dest}")

    if not args.execute:
        print(f"\n共 {len(moves)} 项。加 --execute 执行移动。")
        return 0

    dest_root.mkdir(parents=True, exist_ok=True)
    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if args.force:
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            else:
                print(f"跳过（已存在）: {dest}", file=sys.stderr)
                continue
        shutil.move(str(src), str(dest))
        # 若源策略目录已空，可提示手动删父目录（避免误删其它内容）
        try:
            parent = src.parent
            if parent.is_dir() and parent != rr and not any(parent.iterdir()):
                print(f"  （空目录可删）{parent.relative_to(rr)}")
        except OSError:
            pass

    print(f"\n已移动 {len(moves)} 项到 {dest_root.relative_to(proj)}/<策略>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
