"""Regression: constitution safety/slots SQLite must align with OrderManager default binding."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
    canonical_order_management_db_path,
)


def _minimal_constitution_yaml(
    *,
    tmp_path,
    relative_subdir: str,
    persist_to_line: str,
) -> Path:
    root = tmp_path / relative_subdir if relative_subdir else tmp_path
    cfg_dir = root / "config" / "constitution"
    cfg_dir.mkdir(parents=True)
    cy = cfg_dir / "constitution.yaml"
    cy.write_text(
        f"""
version: 1
name: "PATH_TEST"
kill_switch:
  enabled: false
safety_state:
  persist_to: {persist_to_line!r}
slots:
  enabled: true
  slot_count: 1
  risk_per_slot: 0.01
  slot_state_tracking:
    persist_to: {persist_to_line!r}
""",
        encoding="utf-8",
    )
    return cy


@pytest.mark.unit
def test_canonical_live_highcap_peers_into_app_data_partition(tmp_path, monkeypatch):
    monkeypatch.delenv("MLBOT_ORDER_MANAGEMENT_DB_PATH", raising=False)
    monkeypatch.delenv("MLBOT_LIVE_BASE_DIR", raising=False)

    cy = _minimal_constitution_yaml(
        tmp_path=tmp_path,
        relative_subdir="live/highcap",
        persist_to_line="../data/order_management.db",
    )
    ex = ConstitutionExecutor(constitution_yaml=cy)
    want = (tmp_path / "live" / "data" / "order_management.db").resolve()
    assert ex.resolve_safety_db_path() == want
    assert ex._paths.slots_db_path == want


@pytest.mark.unit
def test_canonical_repo_constitution_under_project_data(tmp_path, monkeypatch):
    monkeypatch.delenv("MLBOT_ORDER_MANAGEMENT_DB_PATH", raising=False)
    monkeypatch.delenv("MLBOT_LIVE_BASE_DIR", raising=False)

    cy = _minimal_constitution_yaml(
        tmp_path=tmp_path,
        relative_subdir="",
        persist_to_line="data/order_management.db",
    )
    ex = ConstitutionExecutor(constitution_yaml=cy)
    want = (tmp_path / "data" / "order_management.db").resolve()
    assert ex.resolve_safety_db_path() == want


@pytest.mark.unit
def test_mlb_env_overrides_yaml_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLBOT_ORDER_MANAGEMENT_DB_PATH", "from-env/om.db")

    nested_root = tmp_path / "nested"
    cy = _minimal_constitution_yaml(
        tmp_path=nested_root,
        relative_subdir="live/highcap",
        persist_to_line="../data/order_management.db",
    )
    ex = ConstitutionExecutor(constitution_yaml=cy)
    assert ex.resolve_safety_db_path() == (tmp_path / "from-env" / "om.db").resolve()


@pytest.mark.unit
def test_canonical_helper_direct_call(tmp_path):
    cy = canonical_order_management_db_path(
        constitution_base_dir=(tmp_path / "live" / "highcap").resolve(),
        raw_obj={
            "safety_state": {"persist_to": "../data/order_management.db"},
        },
    )
    assert cy == (tmp_path / "live" / "data" / "order_management.db").resolve()
