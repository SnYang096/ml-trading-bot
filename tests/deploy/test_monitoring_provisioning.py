"""Static checks for deploy/monitoring Grafana / Loki / alerting provisioning."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MON = REPO / "deploy/monitoring"
DASH = MON / "grafana-provisioning/dashboards"
ALERTING = MON / "grafana-provisioning/alerting"
COMPOSE = MON / "docker-compose.monitoring.yml"

QUANT_JOBS = {
    "quant-feature-bus",
    "quant-trend-fattail",
    "quant-hedge-multileg",
    "quant-spot-accum",
}


def _load_dashboard(name: str) -> dict:
    return json.loads((DASH / name).read_text(encoding="utf-8"))


def _panel_ids(panels: list) -> list[int]:
    return [p["id"] for p in panels if "id" in p]


def _collect_exprs(obj: object, out: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "expr" and isinstance(v, str):
                out.append(v)
            else:
                _collect_exprs(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_exprs(item, out)


def test_compose_default_home_is_system_health():
    text = COMPOSE.read_text(encoding="utf-8")
    assert "quant_system.json" in text
    assert "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH" in text
    assert (
        "quant_home.json"
        not in text.split("GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH")[1].split("\n")[0]
    )


def test_compose_includes_loki_promtail_grafana_alerting_mount():
    text = COMPOSE.read_text(encoding="utf-8")
    assert "mlbot-loki" in text
    assert "mlbot-promtail" in text
    assert "grafana-provisioning/alerting" in text
    assert "GF_UNIFIED_ALERTING_ENABLED=true" in text
    assert "GRAFANA_ALERT_TELEGRAM_BOT_TOKEN" in text


def test_no_alerting_rules_subdirectory():
    """Grafana ignores rules/ subfolder — files must be flat under alerting/."""
    rules_sub = ALERTING / "rules"
    yaml_files = (
        list(rules_sub.glob("*.yml")) + list(rules_sub.glob("*.yaml"))
        if rules_sub.exists()
        else []
    )
    assert yaml_files == [], f"Move rules to alerting/ root: {yaml_files}"
    assert (ALERTING / "quant_ops.yaml").is_file()


def test_quant_ops_alert_rules_yaml_valid_promql():
    doc = yaml.safe_load((ALERTING / "quant_ops.yaml").read_text(encoding="utf-8"))
    assert doc["apiVersion"] == 1
    groups = doc["groups"]
    assert len(groups) == 1
    rules = groups[0]["rules"]
    assert len(rules) == 7
    for rule in rules:
        assert rule["condition"] == "C"
        expr = rule["data"][0]["model"]["expr"]
        assert "{{" not in expr, f"invalid PromQL braces in {rule['uid']}: {expr}"
        assert "}}" not in expr
    exprs = [r["data"][0]["model"]["expr"] for r in rules]
    assert any("up{job=~" in e for e in exprs)
    assert any("mlbot_reconciliation_ok" in e for e in exprs)
    assert any("mlbot_pipeline_data_fresh" in e for e in exprs)


def test_contact_point_default_empty_telegram_in_template():
    cp = yaml.safe_load((ALERTING / "contact-points.yml").read_text(encoding="utf-8"))
    assert cp.get("contactPoints") == []
    tpl = ALERTING / "contact-points.telegram.yml.template"
    raw = tpl.read_text(encoding="utf-8")
    assert "AAH" not in raw  # bot token fragment must not be committed
    cp_tg = yaml.safe_load(raw)
    recv = cp_tg["contactPoints"][0]["receivers"][0]
    assert recv["type"] == "telegram"
    assert recv["settings"]["chatid"] == "-1002004555233"
    assert "bottoken" in recv.get("secureSettings", {})


def test_quant_home_is_ops_hub_without_account_metrics():
    dash = _load_dashboard("quant_home.json")
    assert dash["uid"] == "quant-home"
    assert dash["title"] == "Ops Hub · 导航"
    assert len(dash["panels"]) == 1
    assert dash["panels"][0]["type"] == "text"
    content = dash["panels"][0]["options"]["content"]
    assert "System Health" in content
    assert "CMS" in content
    exprs: list[str] = []
    _collect_exprs(dash, exprs)
    assert exprs == []


def test_quant_system_overview_and_no_account_refresh():
    dash = _load_dashboard("quant_system.json")
    assert dash["uid"] == "quant-system"
    assert "System Health" in dash["title"]
    titles = [p.get("title", "") for p in dash["panels"]]
    assert any("一眼总览" in t for t in titles)
    assert not any("账户刷新" in t for t in titles)
    ids = _panel_ids(dash["panels"])
    assert len(ids) == len(set(ids)), f"duplicate panel ids: {ids}"
    overview_ids = {
        p["id"]
        for p in dash["panels"]
        if p.get("id", 0) >= 1100 and p.get("id", 0) < 1200
    }
    assert overview_ids >= {
        1100,
        1101,
        1102,
        1103,
        1104,
        1105,
        1106,
        1107,
        1108,
        1109,
        1110,
        1111,
    }
    exprs: list[str] = []
    _collect_exprs(dash, exprs)
    assert not any("mlbot_account_update" in e for e in exprs)


def test_strategy_maps_no_cms_position_panels():
    for name in (
        "quant_strategy_map_trend.json",
        "quant_strategy_map_hedge.json",
    ):
        dash = _load_dashboard(name)
        titles = [p.get("title", "") for p in dash["panels"]]
        assert not any("交易所持仓" in t for t in titles), name
    spot = _load_dashboard("quant_strategy_map_spot.json")
    exprs: list[str] = []
    _collect_exprs(spot, exprs)
    assert not any("mlbot_account_balance" in e for e in exprs)


def test_quant_logs_dashboard_loki_datasource():
    dash = _load_dashboard("quant_logs.json")
    assert dash["uid"] == "quant-logs"
    for panel in dash["panels"]:
        if panel.get("type") == "logs":
            assert panel["datasource"]["uid"] == "loki-monitoring"


def test_loki_datasource_provisioned():
    ds = yaml.safe_load(
        (MON / "grafana-provisioning/datasources/loki.yml").read_text(encoding="utf-8")
    )
    assert any(d["uid"] == "loki-monitoring" for d in ds["datasources"])


def test_promtail_journal_and_audit_paths():
    cfg = yaml.safe_load(
        (MON / "promtail/promtail-config.yml").read_text(encoding="utf-8")
    )
    jobs = {s["job_name"] for s in cfg["scrape_configs"]}
    assert "systemd_quant" in jobs
    assert "audit_feature_bus" in jobs
    assert "audit_trend" in jobs
    assert "audit_multi_leg" in jobs
    journal = next(s for s in cfg["scrape_configs"] if s["job_name"] == "systemd_quant")
    keep = [r for r in journal["relabel_configs"] if r.get("action") == "keep"][0]
    assert "quant-" in keep["regex"]
    for job in ("audit_feature_bus", "audit_trend", "audit_multi_leg"):
        sc = next(s for s in cfg["scrape_configs"] if s["job_name"] == job)
        labels = sc["static_configs"][0]["labels"]
        assert "__path__" in labels
        assert "/logs/" in labels["__path__"]


def test_prometheus_scrape_quant_jobs():
    prom = yaml.safe_load((MON / "prometheus.yml").read_text(encoding="utf-8"))
    jobs = {s["job_name"] for s in prom["scrape_configs"]}
    assert QUANT_JOBS <= jobs


def test_env_example_documents_telegram_without_token():
    example = (MON / ".env.example").read_text(encoding="utf-8")
    assert "GRAFANA_ALERT_TELEGRAM_BOT_TOKEN" in example
    assert "GRAFANA_ALERT_TELEGRAM_CHAT_ID=-1002004555233" in example
    assert re.search(r"\d{8,}:[A-Za-z0-9_-]{20,}", example) is None


def test_patch_scripts_idempotent_guard():
    """Patch script source must strip legacy overview ids on re-run."""
    src = (REPO / "scripts/patch_grafana_ops_monitoring.py").read_text(encoding="utf-8")
    assert "legacy_overview" in src
    assert "1100" in src


def test_strategy_map_trend_dashboard_includes_open_reconcile_updated_panel():
    dash = _load_dashboard("quant_strategy_map_trend.json")
    exprs: list[str] = []
    _collect_exprs(dash, exprs)
    assert any(
        'issue="open_reconcile_updated"' in e and 'scope="trend"' in e for e in exprs
    )


def test_strategy_map_hedge_dashboard_includes_segment_lifecycle_events():
    dash = _load_dashboard("quant_strategy_map_hedge.json")
    exprs: list[str] = []
    _collect_exprs(dash, exprs)
    assert any('event=~"segment_.*"' in e and 'scope="hedge"' in e for e in exprs)
