"""pre_deploy contract_checks helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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


def test_plateau_stability_alerts_when_value_outside_range(tmp_path: Path) -> None:
    arch = tmp_path / "strategies" / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    (arch / "regime.yaml").write_text(
        yaml.dump(
            {
                "rules": [
                    {
                        "feature": "ema_1200_position",
                        "operator": ">=",
                        "value": 0.5,
                        "locked": True,
                    }
                ],
                "last_calibration": {
                    "plateaus": [
                        {
                            "feature": "ema_1200_position",
                            "operator": ">=",
                            "plateau": {"start": 0.08, "end": 0.15, "mid": 0.10},
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "contract_checks": {
            "plateau_stability": {
                "enabled": True,
                "on_drift_outside_plateau": "ALERT",
            }
        }
    }
    summary = run_pre_deploy_contract_checks(
        cfg=cfg,
        strategies=["tpc"],
        strategies_root=tmp_path / "strategies",
        project_root=tmp_path,
    )
    assert summary["status"] == "ALERT"
    check = summary["strategies"]["tpc"]["checks"]["plateau_stability"]
    assert check["ok"] is False
    assert any(i.get("status") == "DRIFT" for i in check["yaml_items"])


def test_plateau_stability_gate_robustness_on_parquet(tmp_path: Path) -> None:
    arch = tmp_path / "strategies" / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    (arch / "gate.yaml").write_text(
        yaml.dump(
            {
                "rules": [
                    {
                        "feature": "pulse_z",
                        "operator": "<=",
                        "value": 0.0,
                        "locked": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pq = tmp_path / "features.parquet"
    import numpy as np

    rng = np.random.default_rng(0)
    feat = rng.normal(size=400)
    df = pd.DataFrame(
        {
            "pulse_z": feat,
            "success_no_rr_extreme": (feat > 0).astype(int),
        }
    )
    df.to_parquet(pq)
    cfg = {
        "contract_checks": {
            "plateau_stability": {
                "enabled": True,
                "on_drift_outside_plateau": "ALERT",
                "min_robustness_score": 0.0,
            }
        }
    }
    summary = run_pre_deploy_contract_checks(
        cfg=cfg,
        strategies=["tpc"],
        strategies_root=tmp_path / "strategies",
        project_root=tmp_path,
        features_parquet_by_strategy={"tpc": pq},
    )
    check = summary["strategies"]["tpc"]["checks"]["plateau_stability"]
    assert "robustness_items" in check
    assert len(check["robustness_items"]) >= 1
