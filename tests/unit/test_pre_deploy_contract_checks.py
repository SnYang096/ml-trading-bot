"""pre_deploy contract_checks helper."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.pre_deploy_contract_checks import run_pre_deploy_contract_checks


def test_regime_yaml_required_blocks_missing(tmp_path: Path) -> None:
    cfg = {
        "contract_checks": {
            "regime_yaml": {"required": True},
            "locked_features": {"enabled": False},
        }
    }
    root = tmp_path / "strategies"
    (root / "tpc" / "archetypes").mkdir(parents=True)
    summary = run_pre_deploy_contract_checks(
        cfg=cfg,
        strategies=["tpc"],
        strategies_root=root,
        project_root=tmp_path,
    )
    assert summary["status"] == "BLOCKED"
    assert "regime.yaml" in summary["blocked"][0]


def test_regime_yaml_required_passes(tmp_path: Path) -> None:
    cfg = {"contract_checks": {"regime_yaml": {"required": True}}}
    arch = tmp_path / "strategies" / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    (arch / "regime.yaml").write_text(
        yaml.dump(
            {
                "rules": [
                    {"feature": "tpc_semantic_chop", "operator": "<=", "value": 0.4}
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = run_pre_deploy_contract_checks(
        cfg=cfg,
        strategies=["tpc"],
        strategies_root=tmp_path / "strategies",
        project_root=tmp_path,
    )
    assert summary["status"] == "PASS"
