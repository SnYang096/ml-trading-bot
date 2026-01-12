import pytest

from src.time_series_model.core.constitution.execution_whitelist import (
    enforce_execution_whitelist,
    load_execution_whitelist_config,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation


@pytest.mark.unit
def test_execution_whitelist_allows_only_known_strategy(tmp_path):
    p = tmp_path / "wh.yaml"
    p.write_text(
        """
version: 1
name: "w"
regimes:
  TREND:
    allowed_strategies: ["A"]
    forbidden_keywords: ["RSI"]
  MEAN:
    allowed_strategies: ["B"]
    forbidden_keywords: []
  NO_TRADE:
    allowed_strategies: []
    forbidden_keywords: []
""",
        encoding="utf-8",
    )
    cfg = load_execution_whitelist_config(p)
    enforce_execution_whitelist(
        cfg=cfg, regime="TREND", strategy_id="A", tags=["ok"], evidence={"e": True}
    )
    with pytest.raises(ConstitutionViolation):
        enforce_execution_whitelist(cfg=cfg, regime="TREND", strategy_id="B", tags=None)
    with pytest.raises(ConstitutionViolation):
        enforce_execution_whitelist(
            cfg=cfg, regime="TREND", strategy_id="A", tags=["RSI"]
        )


@pytest.mark.unit
def test_execution_whitelist_required_evidence_enforced(tmp_path):
    p = tmp_path / "wh.yaml"
    p.write_text(
        """
version: 1
name: "w"
regimes:
  TREND:
    allowed_strategies: ["A"]
    forbidden_keywords: []
    strategy_requirements:
      A:
        required_evidence: ["has_orderflow"]
  MEAN:
    allowed_strategies: ["B"]
    forbidden_keywords: []
  NO_TRADE:
    allowed_strategies: []
    forbidden_keywords: []
""",
        encoding="utf-8",
    )
    cfg = load_execution_whitelist_config(p)
    with pytest.raises(ConstitutionViolation):
        enforce_execution_whitelist(
            cfg=cfg, regime="TREND", strategy_id="A", evidence={}
        )
    enforce_execution_whitelist(
        cfg=cfg, regime="TREND", strategy_id="A", evidence={"has_orderflow": True}
    )
