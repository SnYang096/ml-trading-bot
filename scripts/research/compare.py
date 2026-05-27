"""mlbot research compare — compare scan/ic/plateau json artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from scripts.research._common import PROJECT_ROOT

_PLATEAU_KEYS = (
    "recommended",
    "recommended_threshold",
    "plateau_mid",
    "is_plateau",
    "feature",
    "operator",
    "confidence",
    "mean_snotio",
    "lift_at_mid",
)


def _summarize_artifact(path: Path, blob: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {"path": str(path)}
    if isinstance(blob, list):
        if not blob:
            return {**base, "type": "list", "n_rows": 0}
        sample = blob[0]
        if isinstance(sample, dict) and "rank_ic" in sample:
            by_feat: Dict[str, List[float]] = {}
            for row in blob:
                feat = str(row.get("feature", "?"))
                by_feat.setdefault(feat, []).append(float(row["rank_ic"]))
            return {
                **base,
                "type": "ic_decay",
                "n_rows": len(blob),
                "features": sorted(by_feat),
                "mean_rank_ic": {
                    f: round(float(sum(v) / len(v)), 6) for f, v in by_feat.items()
                },
            }
        return {
            **base,
            "type": "list",
            "n_rows": len(blob),
            "sample_keys": list(sample.keys())[:8],
        }

    if isinstance(blob, dict):
        if any(k in blob for k in _PLATEAU_KEYS):
            return {
                **base,
                "type": "plateau",
                **{k: blob[k] for k in _PLATEAU_KEYS if k in blob},
            }
        if "fold_z_scores" in blob or "mean_z" in blob:
            return {
                **base,
                "type": "robustness",
                "mean_z": blob.get("mean_z"),
                "std_z": blob.get("std_z"),
                "n_folds": len(blob.get("fold_z_scores", [])),
            }
        return {**base, "type": "dict", "keys": list(blob.keys())[:12]}
    return {**base, "type": type(blob).__name__}


def _diff_summaries(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    diffs: List[Dict[str, Any]] = []
    plateau_rows = [s for s in summaries if s.get("type") == "plateau"]
    if len(plateau_rows) >= 2:
        ref = plateau_rows[0]
        for other in plateau_rows[1:]:
            delta: Dict[str, Any] = {"vs": other["path"]}
            for key in ("recommended", "recommended_threshold", "plateau_mid"):
                if key in ref or key in other:
                    a, b = ref.get(key), other.get(key)
                    if a is not None and b is not None and a != b:
                        delta[key] = {"a": a, "b": b, "delta": float(b) - float(a)}
            if len(delta) > 1:
                diffs.append({"compare": ref["path"], **delta})
    ic_rows = [s for s in summaries if s.get("type") == "ic_decay"]
    if len(ic_rows) >= 2:
        feats_a = set(ic_rows[0].get("features", []))
        feats_b = set(ic_rows[1].get("features", []))
        diffs.append(
            {
                "compare": "ic_decay",
                "only_in_first": sorted(feats_a - feats_b),
                "only_in_second": sorted(feats_b - feats_a),
            }
        )
    return diffs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research compare (json artifacts side-by-side)"
    )
    p.add_argument("paths", nargs="+", help="JSON result files to compare")
    args = p.parse_args(argv)

    summaries: List[Dict[str, Any]] = []
    for raw in args.paths:
        pth = Path(raw)
        if not pth.is_absolute():
            pth = PROJECT_ROOT / pth
        blob = json.loads(pth.read_text(encoding="utf-8"))
        summaries.append(_summarize_artifact(pth, blob))

    report = {"artifacts": summaries, "diffs": _diff_summaries(summaries)}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
