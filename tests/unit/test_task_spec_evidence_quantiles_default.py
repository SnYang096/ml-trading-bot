import yaml


def test_task_spec_default_emits_evidence_quantiles():
    cfg = yaml.safe_load(open("config/tasks/task_spec.yaml", "r", encoding="utf-8"))
    training = (cfg.get("model_plan") or {}).get("training") or {}
    assert training.get("emit_evidence_quantiles") is True
    assert "evidence_quantiles" in training
    assert "evidence_quantiles_keys" in training
