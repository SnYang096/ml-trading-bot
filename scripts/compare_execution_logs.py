#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _summary(rows: list[dict]) -> dict:
    modes = Counter()
    intents = 0
    submits = 0
    total = len(rows)
    for r in rows:
        router = r.get("router") or {}
        mode = router.get("mode")
        if mode:
            modes[str(mode).upper()] += 1
        execution = r.get("execution") or {}
        if execution.get("intent"):
            intents += 1
        if execution.get("submit_order"):
            submits += 1
    return {
        "total": total,
        "modes": dict(modes),
        "execution_intent": intents,
        "submit_order": submits,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare canonical execution logs.")
    ap.add_argument("--a", required=True, help="jsonl file A")
    ap.add_argument("--b", required=True, help="jsonl file B")
    ap.add_argument("--out", default=None, help="optional json output")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows_a = _read_jsonl(Path(args.a))
    rows_b = _read_jsonl(Path(args.b))
    summary = {"a": _summary(rows_a), "b": _summary(rows_b)}
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
