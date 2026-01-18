from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os

import yaml


def _load_yaml_dict(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return obj if isinstance(obj, dict) else {}


def resolve_live_runtime_paths(
    *,
    runtime_config_path: Optional[str | Path] = None,
    defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Resolve live runtime paths with precedence:
      1) Explicit runtime_config_path (or env MLBOT_LIVE_RUNTIME_PATHS_YAML)
      2) Config values
      3) Env overrides per key (highest)
      4) Provided defaults
    """
    cfg_path = (
        runtime_config_path
        if runtime_config_path is not None
        else os.getenv(
            "MLBOT_LIVE_RUNTIME_PATHS_YAML",
            "config/nnmultihead/live_runtime_paths.yaml",
        )
    )
    cfg = _load_yaml_dict(cfg_path)
    defaults = defaults or {}

    def pick(env_key: str, cfg_key: str, default_value: str) -> str:
        return str(os.getenv(env_key, cfg.get(cfg_key, default_value) or default_value))

    return {
        "constitution_yaml": pick(
            "MLBOT_CONSTITUTION_YAML",
            "constitution_yaml",
            defaults.get("constitution_yaml", "config/constitution/constitution.yaml"),
        ),
        "live_feature_contract_yaml": pick(
            "MLBOT_LIVE_FEATURE_CONTRACT_YAML",
            "live_feature_contract_yaml",
            defaults.get(
                "live_feature_contract_yaml",
                "config/live/live_feature_contract.yaml",
            ),
        ),
        "execution_rules_yaml": pick(
            "MLBOT_EXECUTION_RULES_YAML",
            "execution_rules_yaml",
            defaults.get("execution_rules_yaml", ""),
        ),
    }
