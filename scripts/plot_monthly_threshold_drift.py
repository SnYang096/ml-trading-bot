#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import yaml


def _iter_month_dirs(run_root: Path) -> List[Path]:
    out = []
    for p in sorted(run_root.glob("fast_month_*")):
        if p.is_dir():
            out.append(p)
    return out


def _month_token(month_dir: Path) -> str:
    name = month_dir.name
    if name.startswith("fast_month_"):
        return name.replace("fast_month_", "", 1)
    return name


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _append_row(rows: List[Dict[str, Any]], **kwargs: Any) -> None:
    rows.append(
        {
            "month": str(kwargs.get("month", "")),
            "strategy": str(kwargs.get("strategy", "")),
            "layer": str(kwargs.get("layer", "")),
            "key": str(kwargs.get("key", "")),
            "feature": str(kwargs.get("feature", "")),
            "operator": str(kwargs.get("operator", "")),
            "value": kwargs.get("value", None),
            "rule_id": str(kwargs.get("rule_id", "")),
            "enabled": bool(kwargs.get("enabled", True)),
            "locked": bool(kwargs.get("locked", False)),
            "source_file": str(kwargs.get("source_file", "")),
        }
    )


def _extract_prefilter(
    month: str, path: Path, rows: List[Dict[str, Any]], strategy: str
) -> None:
    obj = _safe_load_yaml(path)
    for i, rule in enumerate(obj.get("rules") or []):
        if not isinstance(rule, dict):
            continue
        rid = str(rule.get("id", f"rule_{i}"))
        enabled = bool(rule.get("enabled", True))
        locked = bool(rule.get("locked", False))
        if "any_of" in rule and isinstance(rule.get("any_of"), list):
            for j, sub in enumerate(rule.get("any_of") or []):
                if not isinstance(sub, dict):
                    continue
                feat = str(sub.get("feature", ""))
                op = str(sub.get("operator", ""))
                val = sub.get("value", None)
                _append_row(
                    rows,
                    month=month,
                    layer="prefilter",
                    key=f"prefilter:{feat}:{op}:any_of_{i}_{j}",
                    feature=feat,
                    operator=op,
                    value=val,
                    rule_id=rid,
                    enabled=enabled,
                    locked=locked,
                    source_file=str(path),
                    strategy=strategy,
                )
        else:
            feat = str(rule.get("feature", ""))
            op = str(rule.get("operator", ""))
            val = rule.get("value", None)
            _append_row(
                rows,
                month=month,
                layer="prefilter",
                key=f"prefilter:{feat}:{op}",
                feature=feat,
                operator=op,
                value=val,
                rule_id=rid,
                enabled=enabled,
                locked=locked,
                source_file=str(path),
                strategy=strategy,
            )


def _iter_gate_when_atoms(when_block: Any) -> Iterator[Tuple[str, str, Any]]:
    """Flatten gate `when` including nested all_of / any_of lists."""
    if not isinstance(when_block, dict):
        return
    if "all_of" in when_block or "any_of" in when_block:
        key = "all_of" if "all_of" in when_block else "any_of"
        seq = when_block.get(key) or []
        if isinstance(seq, list):
            for item in seq:
                yield from _iter_gate_when_atoms(item)
        return
    for feat, cond in when_block.items():
        if not isinstance(cond, dict):
            continue
        for op, val in cond.items():
            if not str(op).startswith("value_"):
                continue
            yield (str(feat), str(op), val)


# Gate YAML may list rules under hard_gates only (legacy) or split phases
# (e.g. hard_gates: [] + system_safety: [...] after calibration).
_GATE_RULE_SECTIONS: Tuple[str, ...] = ("hard_gates", "system_safety", "guardrails")


def _extract_gate(
    month: str, path: Path, rows: List[Dict[str, Any]], strategy: str
) -> None:
    obj = _safe_load_yaml(path)
    for section in _GATE_RULE_SECTIONS:
        for gate in obj.get(section) or []:
            if not isinstance(gate, dict):
                continue
            rid = str(gate.get("id", ""))
            when = gate.get("when") or {}
            if not isinstance(when, dict):
                continue
            g_en = bool(gate.get("enabled", True)) and not bool(
                gate.get("disabled", False)
            )
            g_locked = bool(gate.get("locked", False))
            for feat, op, val in _iter_gate_when_atoms(when):
                _append_row(
                    rows,
                    month=month,
                    layer="gate",
                    key=f"gate:{rid}:{feat}:{op}",
                    feature=feat,
                    operator=op,
                    value=val,
                    rule_id=rid,
                    enabled=g_en,
                    locked=g_locked,
                    source_file=str(path),
                    strategy=strategy,
                )


def _extract_entry_filters(
    month: str, path: Path, rows: List[Dict[str, Any]], strategy: str
) -> None:
    obj = _safe_load_yaml(path)
    for f in obj.get("filters") or []:
        if not isinstance(f, dict):
            continue
        rid = str(f.get("id", ""))
        enabled = bool(f.get("enabled", True))
        locked = bool(f.get("locked", False))
        for c in f.get("conditions") or []:
            if not isinstance(c, dict):
                continue
            feat = str(c.get("feature", ""))
            op = str(c.get("operator", ""))
            val = c.get("value", None)
            _append_row(
                rows,
                month=month,
                layer="entry_filter",
                key=f"entry_filter:{rid}:{feat}:{op}",
                feature=feat,
                operator=op,
                value=val,
                rule_id=rid,
                enabled=enabled,
                locked=locked,
                source_file=str(path),
                strategy=strategy,
            )


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _load_month_status(run_root: Path, strategy: str) -> Dict[str, Dict[str, Any]]:
    ledger_path = run_root / "monthly_ledger.jsonl"
    if not ledger_path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for ln in ledger_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except Exception:
            continue
        month = str(row.get("month", "") or "").strip()
        if not month:
            continue
        st_map = row.get("slow_guard_by_strategy") or {}
        status: Dict[str, Any] = {}
        if isinstance(st_map, dict) and isinstance(st_map.get(strategy), dict):
            status = dict(st_map.get(strategy) or {})
        elif isinstance(row.get("slow_guard_status"), dict):
            status = dict(row.get("slow_guard_status") or {})
        out[month] = {
            "guard_level": str(status.get("guard_level", "none") or "none"),
            "adoption_status": str(
                status.get("adoption_status", "unchanged") or "unchanged"
            ),
            "fallback_used": bool(status.get("fallback_used", False)),
            "reason": str(status.get("reason", "") or ""),
        }
    return out


def _compute_drift_rows(
    rows: List[Dict[str, Any]],
    month_status: Dict[str, Dict[str, Any]],
    *,
    yellow_rel: float,
    red_rel: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("layer", "")), str(r.get("key", "")))
        grouped.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    eps = 1e-9
    for (_layer, _key), items in grouped.items():
        seq = sorted(items, key=lambda x: str(x.get("month", "")))
        prev: Dict[str, Any] | None = None
        for cur in seq:
            if prev is None:
                prev = cur
                continue
            old_v = _to_float(prev.get("value"))
            new_v = _to_float(cur.get("value"))
            if old_v is None or new_v is None:
                prev = cur
                continue
            abs_change = abs(new_v - old_v)
            rel_change = abs_change / max(abs(old_v), eps)
            drift_level = "green"
            if rel_change >= red_rel:
                drift_level = "red"
            elif rel_change >= yellow_rel:
                drift_level = "yellow"
            st = month_status.get(str(cur.get("month", "")), {})
            out.append(
                {
                    "month": str(cur.get("month", "")),
                    "strategy": str(cur.get("strategy", "")),
                    "layer": str(cur.get("layer", "")),
                    "feature": str(cur.get("feature", "")),
                    "operator": str(cur.get("operator", "")),
                    "old_value": old_v,
                    "new_value": new_v,
                    "abs_change": abs_change,
                    "relative_change": rel_change,
                    "rule_id": str(cur.get("rule_id", "")),
                    "locked": bool(cur.get("locked", False)),
                    "enabled": bool(cur.get("enabled", True)),
                    "drift_level": drift_level,
                    "adoption_status": str(st.get("adoption_status", "unchanged")),
                    "fallback_used": bool(st.get("fallback_used", False)),
                    "reason": str(st.get("reason", "")),
                }
            )
            prev = cur
    return sorted(
        out, key=lambda x: (x["month"], x["layer"], x["feature"], x["operator"])
    )


def _write_json(obj: Any, out_path: Path) -> None:
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_html(
    rows: List[Dict[str, Any]],
    title: str,
    month_status: Dict[str, Dict[str, Any]],
    drift_rows: List[Dict[str, Any]],
) -> str:
    layers = ["prefilter", "gate", "entry_filter"]
    traces_by_layer: Dict[str, List[Dict[str, Any]]] = {k: [] for k in layers}

    # Group rows by layer+key
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {k: {} for k in layers}
    for r in rows:
        layer = str(r.get("layer", ""))
        if layer not in grouped:
            continue
        key = str(r.get("key", ""))
        grouped[layer].setdefault(key, []).append(r)

    for layer in layers:
        for key, items in grouped[layer].items():
            items = sorted(items, key=lambda x: str(x.get("month", "")))
            xs = [str(x.get("month", "")) for x in items]
            ys = [_to_float(x.get("value")) for x in items]
            if not any(y is not None for y in ys):
                continue
            custom = [
                [
                    str(x.get("rule_id", "")),
                    bool(x.get("locked", False)),
                    bool(x.get("enabled", True)),
                    str(
                        month_status.get(str(x.get("month", "")), {}).get(
                            "guard_level", "none"
                        )
                    ),
                    str(
                        month_status.get(str(x.get("month", "")), {}).get(
                            "adoption_status", "unchanged"
                        )
                    ),
                ]
                for x in items
            ]
            traces_by_layer[layer].append(
                {
                    "x": xs,
                    "y": ys,
                    "mode": "lines+markers",
                    "name": key,
                    "connectgaps": False,
                    "customdata": custom,
                    "hovertemplate": (
                        "month=%{x}<br>value=%{y}"
                        "<br>rule_id=%{customdata[0]}"
                        "<br>locked=%{customdata[1]}"
                        "<br>enabled=%{customdata[2]}"
                        "<br>guard=%{customdata[3]}"
                        "<br>adoption=%{customdata[4]}"
                        "<extra></extra>"
                    ),
                }
            )

    payload = {
        "prefilter": traces_by_layer["prefilter"],
        "gate": traces_by_layer["gate"],
        "entry_filter": traces_by_layer["entry_filter"],
    }
    month_rows = sorted(month_status.items(), key=lambda x: x[0])
    month_rows_html = "\n".join(
        [
            "<tr>"
            f"<td>{m}</td>"
            f"<td>{str(st.get('guard_level', 'none'))}</td>"
            f"<td>{str(st.get('adoption_status', 'unchanged'))}</td>"
            f"<td>{'yes' if bool(st.get('fallback_used', False)) else 'no'}</td>"
            f"<td>{str(st.get('reason', ''))}</td>"
            "</tr>"
            for m, st in month_rows
        ]
    )
    drift_rows_html = "\n".join(
        [
            "<tr>"
            f"<td>{str(r.get('month', ''))}</td>"
            f"<td>{str(r.get('layer', ''))}</td>"
            f"<td>{str(r.get('feature', ''))}</td>"
            f"<td>{str(r.get('operator', ''))}</td>"
            f"<td>{str(r.get('old_value', ''))}</td>"
            f"<td>{str(r.get('new_value', ''))}</td>"
            f"<td>{float(r.get('relative_change', 0.0) or 0.0):.4f}</td>"
            f"<td>{str(r.get('drift_level', 'green'))}</td>"
            "</tr>"
            for r in drift_rows
            if str(r.get("drift_level", "green")) in {"yellow", "red"}
        ]
    )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 16px; }}
    #ctrl {{ margin-bottom: 12px; }}
    #chart {{ width: 100%; height: 760px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <h3>Monthly Guard Status</h3>
  <table border="1" cellspacing="0" cellpadding="4">
    <tr><th>month</th><th>guard</th><th>adoption</th><th>fallback</th><th>reason</th></tr>
    {month_rows_html}
  </table>
  <div id="ctrl">
    Layer:
    <select id="layerSel">
      <option value="prefilter">prefilter</option>
      <option value="gate">gate</option>
      <option value="entry_filter">entry_filter</option>
    </select>
  </div>
  <div id="chart"></div>
  <h3>Yellow/Red Drift Details</h3>
  <table border="1" cellspacing="0" cellpadding="4">
    <tr><th>month</th><th>layer</th><th>feature</th><th>operator</th><th>old</th><th>new</th><th>relative_change</th><th>drift_level</th></tr>
    {drift_rows_html}
  </table>
  <script>
    const tracesByLayer = {json.dumps(payload, ensure_ascii=False)};
    const layoutBase = {{
      margin: {{l: 60, r: 20, t: 20, b: 50}},
      xaxis: {{title: "month"}},
      yaxis: {{title: "threshold value"}},
      hovermode: "x unified",
      showlegend: true
    }};
    function render(layer) {{
      const traces = tracesByLayer[layer] || [];
      const layout = Object.assign({{}}, layoutBase, {{
        title: `Layer: ${{layer}} (series=${{traces.length}})`
      }});
      Plotly.newPlot("chart", traces, layout, {{responsive: true}});
    }}
    const sel = document.getElementById("layerSel");
    sel.addEventListener("change", () => render(sel.value));
    render(sel.value);
  </script>
</body>
</html>
"""


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    headers = [
        "month",
        "strategy",
        "layer",
        "key",
        "feature",
        "operator",
        "value",
        "rule_id",
        "enabled",
        "locked",
        "source_file",
    ]
    lines = [",".join(headers)]
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            s = str(v if v is not None else "")
            if "," in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        lines.append(",".join(vals))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv_with_headers(
    rows: List[Dict[str, Any]], out_csv: Path, headers: List[str]
) -> None:
    lines = [",".join(headers)]
    for r in rows:
        vals: List[str] = []
        for h in headers:
            s = str(r.get(h, "") if r.get(h, "") is not None else "")
            if "," in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        lines.append(",".join(vals))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract monthly threshold drift from rolling_sim outputs"
    )
    p.add_argument("--run-root", required=True, help="rolling_sim run root directory")
    p.add_argument(
        "--strategy", required=True, help="strategy name, e.g. bpc-short-120T"
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="optional output directory (default: <run-root>/threshold_tracking/<strategy>)",
    )
    p.add_argument("--yellow-rel", type=float, default=0.20)
    p.add_argument("--red-rel", type=float, default=0.35)
    args = p.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        raise SystemExit(f"run root not found: {run_root}")
    strategy = str(args.strategy).strip()
    out_dir = (
        Path(args.output_dir)
        if str(args.output_dir).strip()
        else (run_root / "threshold_tracking" / strategy)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    month_status = _load_month_status(run_root, strategy)

    rows: List[Dict[str, Any]] = []
    for mdir in _iter_month_dirs(run_root):
        m = _month_token(mdir)
        arch = mdir / "strategies_calibrated" / strategy / "archetypes"
        _extract_prefilter(m, arch / "prefilter.yaml", rows, strategy)
        _extract_gate(m, arch / "gate.yaml", rows, strategy)
        _extract_entry_filters(m, arch / "entry_filters.yaml", rows, strategy)

    rows = sorted(rows, key=lambda x: (str(x["layer"]), str(x["key"]), str(x["month"])))
    drift_rows = _compute_drift_rows(
        rows,
        month_status,
        yellow_rel=float(args.yellow_rel),
        red_rel=float(args.red_rel),
    )
    csv_path = out_dir / "threshold_timeseries.csv"
    html_path = out_dir / "threshold_timeseries.html"
    drift_csv_path = out_dir / "threshold_drift_report.csv"
    drift_json_path = out_dir / "threshold_drift_report.json"
    _write_csv(rows, csv_path)
    html_path.write_text(
        _build_html(rows, f"Threshold Drift · {strategy}", month_status, drift_rows),
        encoding="utf-8",
    )
    _write_csv_with_headers(
        drift_rows,
        drift_csv_path,
        [
            "month",
            "strategy",
            "layer",
            "feature",
            "operator",
            "old_value",
            "new_value",
            "abs_change",
            "relative_change",
            "rule_id",
            "locked",
            "enabled",
            "drift_level",
            "adoption_status",
            "fallback_used",
            "reason",
        ],
    )
    _write_json(
        {
            "strategy": strategy,
            "run_root": str(run_root),
            "month_status": month_status,
            "drift_rows": drift_rows,
        },
        drift_json_path,
    )

    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"html={html_path}")
    print(f"drift_csv={drift_csv_path}")
    print(f"drift_json={drift_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
