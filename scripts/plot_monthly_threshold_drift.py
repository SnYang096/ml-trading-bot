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
            "layer": str(kwargs.get("layer", "")),
            "key": str(kwargs.get("key", "")),
            "feature": str(kwargs.get("feature", "")),
            "operator": str(kwargs.get("operator", "")),
            "value": kwargs.get("value", None),
            "rule_id": str(kwargs.get("rule_id", "")),
            "enabled": bool(kwargs.get("enabled", True)),
            "source_file": str(kwargs.get("source_file", "")),
        }
    )


def _extract_prefilter(month: str, path: Path, rows: List[Dict[str, Any]]) -> None:
    obj = _safe_load_yaml(path)
    for i, rule in enumerate(obj.get("rules") or []):
        if not isinstance(rule, dict):
            continue
        rid = str(rule.get("id", f"rule_{i}"))
        enabled = bool(rule.get("enabled", True))
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
                    source_file=str(path),
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
                source_file=str(path),
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


def _extract_gate(month: str, path: Path, rows: List[Dict[str, Any]]) -> None:
    obj = _safe_load_yaml(path)
    for gate in obj.get("hard_gates") or []:
        if not isinstance(gate, dict):
            continue
        rid = str(gate.get("id", ""))
        when = gate.get("when") or {}
        if not isinstance(when, dict):
            continue
        g_en = bool(gate.get("enabled", True)) and not bool(gate.get("disabled", False))
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
                source_file=str(path),
            )


def _extract_entry_filters(month: str, path: Path, rows: List[Dict[str, Any]]) -> None:
    obj = _safe_load_yaml(path)
    for f in obj.get("filters") or []:
        if not isinstance(f, dict):
            continue
        rid = str(f.get("id", ""))
        enabled = bool(f.get("enabled", True))
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
                source_file=str(path),
            )


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _build_html(rows: List[Dict[str, Any]], title: str) -> str:
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
            traces_by_layer[layer].append(
                {
                    "x": xs,
                    "y": ys,
                    "mode": "lines+markers",
                    "name": key,
                    "connectgaps": False,
                }
            )

    payload = {
        "prefilter": traces_by_layer["prefilter"],
        "gate": traces_by_layer["gate"],
        "entry_filter": traces_by_layer["entry_filter"],
    }

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
  <div id="ctrl">
    Layer:
    <select id="layerSel">
      <option value="prefilter">prefilter</option>
      <option value="gate">gate</option>
      <option value="entry_filter">entry_filter</option>
    </select>
  </div>
  <div id="chart"></div>
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
        "layer",
        "key",
        "feature",
        "operator",
        "value",
        "rule_id",
        "enabled",
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

    rows: List[Dict[str, Any]] = []
    for mdir in _iter_month_dirs(run_root):
        m = _month_token(mdir)
        arch = mdir / "strategies_calibrated" / strategy / "archetypes"
        _extract_prefilter(m, arch / "prefilter.yaml", rows)
        _extract_gate(m, arch / "gate.yaml", rows)
        _extract_entry_filters(m, arch / "entry_filters.yaml", rows)

    rows = sorted(rows, key=lambda x: (str(x["layer"]), str(x["key"]), str(x["month"])))
    csv_path = out_dir / "threshold_timeseries.csv"
    html_path = out_dir / "threshold_timeseries.html"
    _write_csv(rows, csv_path)
    html_path.write_text(
        _build_html(rows, f"Threshold Drift · {strategy}"), encoding="utf-8"
    )

    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"html={html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
