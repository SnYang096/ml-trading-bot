from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class LiveFeatureContractV1:
    required_keys_any: List[str]
    required_pred_keys: List[str]
    on_violation: str = "NO_TRADE"


def load_live_feature_contract_v1(path: str) -> LiveFeatureContractV1:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    req = obj.get("required") or {}
    nn = obj.get("nn_inference") or {}
    pol = obj.get("policy") or {}
    return LiveFeatureContractV1(
        required_keys_any=list(req.get("keys_any") or []),
        required_pred_keys=list(nn.get("required_pred_keys") or []),
        on_violation=str(pol.get("on_violation") or "NO_TRADE").upper(),
    )


def validate_live_features_v1(
    *,
    contract: LiveFeatureContractV1,
    features: Dict[str, Any],
    nn_inference_enabled: bool,
) -> Tuple[bool, List[str]]:
    """
    Returns (ok, reasons).
    """
    reasons: List[str] = []
    feats = features or {}

    missing_any = [k for k in contract.required_keys_any if k not in feats]
    if missing_any:
        reasons.append(f"missing_required_keys_any={missing_any}")

    if nn_inference_enabled:
        missing_pred = [k for k in contract.required_pred_keys if k not in feats]
        if missing_pred:
            reasons.append(f"missing_required_pred_keys={missing_pred}")

    return (len(reasons) == 0), reasons
