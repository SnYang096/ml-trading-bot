from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EvidenceRule:
    name: str
    kind: str
    any_key_contains: List[str]


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def compute_execution_evidence(
    *,
    features: Dict[str, Any],
    rules: List[Dict[str, Any]] | None,
) -> Dict[str, bool]:
    """
    Build boolean evidence flags from a feature dict.

    Supported rule kinds (v1):
    - any_key_contains: evidence true if any feature key contains any substring.
    - key_exists: evidence true if key exists in features
    - value_gt/value_gte/value_lt/value_lte: numeric comparisons on a key
    - abs_gt: abs(value(key)) > threshold

    Missing key policy:
    - on_missing: "false" (default) | "true" | "error"
    """
    feats = features or {}
    keys = [str(k) for k in feats.keys()]

    out: Dict[str, bool] = {}
    for r in rules or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        kind = str(r.get("kind") or "any_key_contains").strip()
        on_missing = str(r.get("on_missing") or "false").strip().lower()

        def _missing() -> bool:
            if on_missing == "true":
                return True
            if on_missing == "error":
                raise KeyError(f"Missing feature key for evidence '{name}'")
            return False

        if kind == "any_key_contains":
            subs = r.get("any_key_contains") or []
            subs = [str(x) for x in subs if str(x)]
            ok = False
            for s in subs:
                if any(s in k for k in keys):
                    ok = True
                    break
            out[name] = bool(ok)
            continue

        if kind == "key_exists":
            key = str(r.get("key") or "")
            out[name] = bool(key and (key in feats))
            continue

        if kind in ("value_gt", "value_gte", "value_lt", "value_lte", "abs_gt"):
            key = str(r.get("key") or "")
            if not key or key not in feats:
                out[name] = _missing()
                continue
            v = _to_float(feats.get(key))
            if v is None:
                out[name] = _missing()
                continue
            thr = _to_float(r.get("threshold"))
            if thr is None:
                # No threshold means invalid rule => false (strict)
                out[name] = False
                continue
            if kind == "value_gt":
                out[name] = bool(v > thr)
            elif kind == "value_gte":
                out[name] = bool(v >= thr)
            elif kind == "value_lt":
                out[name] = bool(v < thr)
            elif kind == "value_lte":
                out[name] = bool(v <= thr)
            else:  # abs_gt
                out[name] = bool(abs(v) > thr)
            continue

        # Unknown kinds default to false to keep the contract strict/extendable.
        out[name] = False

    return out
