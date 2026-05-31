#!/usr/bin/env python3
"""Build HTML report from rd_loop quick_scan artifacts.

Section order comes from config (rd_loop yaml), not a hardcoded file list.
Charts are inferred from each artifact's table columns / JSON shape:

- markdown tables with ``succ_hit`` → success % lines + base ref
- markdown tables with ``mean_hit`` → continuous target mean lines
- markdown tables with ``rank_ic`` + ``feature`` → IC decay multi-line
- JSON with ``kpi: snotio`` → snotio + trades lines (+ plateau band if present)
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_REPORT_HTML_NAME = "report.html"
_MANIFEST_NAME = "report_manifest.json"

_CHART_COLORS = (
    "#3d8bfd",
    "#5dd39e",
    "#f0b429",
    "#e85d75",
    "#a78bfa",
    "#56cfe1",
)


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _parse_md_table(
    lines: List[str], start: int
) -> Tuple[Optional[List[List[str]]], int]:
    if start >= len(lines) or not lines[start].strip().startswith("|"):
        return None, start
    rows: List[List[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        parts = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not all(re.match(r"^[-:]+$", p.replace(" ", "")) for p in parts):
            rows.append(parts)
        i += 1
    return (rows if rows else None), i


def _cell_float(cell: str) -> Optional[float]:
    s = cell.strip().replace("—", "").replace(",", "")
    if not s or s.lower() == "nan":
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _svg_line_chart(
    *,
    x_labels: List[str],
    series: Mapping[str, List[Optional[float]]],
    width: int = 680,
    height: int = 240,
    y_label: str = "",
    ref_line: Optional[float] = None,
    plateau_x: Optional[Tuple[float, float]] = None,
    y_format: str = "num",
) -> str:
    """Inline SVG multi-series line chart (no external JS)."""
    if not x_labels or not series:
        return ""
    pad_l, pad_r, pad_t, pad_b = 52, 16, 18, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_vals: List[float] = []
    for vals in series.values():
        all_vals.extend(v for v in vals if v is not None)
    if ref_line is not None:
        all_vals.append(ref_line)
    if not all_vals:
        return ""
    y_min = min(all_vals)
    y_max = max(all_vals)
    if y_max - y_min < 1e-9:
        y_min -= 0.5
        y_max += 0.5
    else:
        margin = (y_max - y_min) * 0.08
        y_min -= margin
        y_max += margin

    def x_px(i: int) -> float:
        if len(x_labels) <= 1:
            return pad_l + plot_w / 2
        return pad_l + plot_w * i / (len(x_labels) - 1)

    def y_px(v: float) -> float:
        return pad_t + plot_h * (1 - (v - y_min) / (y_max - y_min))

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_esc(y_label or "chart")}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#121820" rx="6"/>',
    ]
    for tick in range(5):
        frac = tick / 4
        yv = y_min + (y_max - y_min) * frac
        yp = y_px(yv)
        parts.append(
            f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{width - pad_r}" y2="{yp:.1f}" '
            f'stroke="#2a3548" stroke-width="1"/>'
        )
        label = f"{yv:.2f}" if y_format == "num" else f"{yv:.1f}%"
        parts.append(
            f'<text x="{pad_l - 6}" y="{yp + 4:.1f}" fill="#8b9cb3" '
            f'font-size="10" text-anchor="end">{label}</text>'
        )
    if plateau_x is not None:
        lo, hi = plateau_x
        xs = [float(x) for x in x_labels]
        i_lo = next((i for i, xv in enumerate(xs) if xv >= lo - 1e-9), 0)
        i_hi = next((i for i, xv in enumerate(reversed(xs)) if xv <= hi + 1e-9), 0)
        i_hi = len(xs) - 1 - i_hi
        x1, x2 = x_px(i_lo), x_px(i_hi)
        parts.append(
            f'<rect x="{min(x1,x2):.1f}" y="{pad_t}" width="{abs(x2-x1):.1f}" '
            f'height="{plot_h}" fill="#1e3a2f" opacity="0.55"/>'
        )
    if ref_line is not None:
        ry = y_px(ref_line)
        parts.append(
            f'<line x1="{pad_l}" y1="{ry:.1f}" x2="{width - pad_r}" y2="{ry:.1f}" '
            f'stroke="#8b9cb3" stroke-width="1" stroke-dasharray="4,3"/>'
        )
    for si, (name, vals) in enumerate(series.items()):
        color = _CHART_COLORS[si % len(_CHART_COLORS)]
        pts: List[str] = []
        for i, v in enumerate(vals):
            if v is None:
                continue
            pts.append(f"{x_px(i):.1f},{y_px(v):.1f}")
        if len(pts) >= 2:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                f'points="{" ".join(pts)}"/>'
            )
        for i, v in enumerate(vals):
            if v is None:
                continue
            parts.append(
                f'<circle cx="{x_px(i):.1f}" cy="{y_px(v):.1f}" r="3.5" fill="{color}"/>'
            )
    for i, xl in enumerate(x_labels):
        parts.append(
            f'<text x="{x_px(i):.1f}" y="{height - 10}" fill="#8b9cb3" '
            f'font-size="10" text-anchor="middle">{_esc(xl)}</text>'
        )
    lx = pad_l
    for si, name in enumerate(series.keys()):
        color = _CHART_COLORS[si % len(_CHART_COLORS)]
        parts.append(
            f'<rect x="{lx}" y="6" width="10" height="10" fill="{color}"/>'
            f'<text x="{lx + 14}" y="15" fill="#e7ecf3" font-size="11">{_esc(name)}</text>'
        )
        lx += 14 + len(name) * 6 + 20
    parts.append("</svg>")
    return "".join(parts)


def _chart_for_table(rows: List[List[str]], title: str) -> str:
    if len(rows) < 2:
        return ""
    header, *body = rows
    h_lower = [h.lower() for h in header]
    if "rank_ic" in h_lower and h_lower and h_lower[0] == "feature":
        return _chart_ic_decay(header, body)

    idx = {header[i].lower(): i for i in range(len(header))}
    if "condition" in h_lower and "succ_in" in idx:
        return _chart_condition_set(header, body, title)

    if not body or "threshold" not in h_lower[0]:
        return ""
    x_labels: List[str] = []
    series: Dict[str, List[Optional[float]]] = {}
    ref_line: Optional[float] = None
    y_format = "num"
    y_label = ""

    if "succ_hit" in idx:
        series = {"succ_hit": [], "succ_other": []}
        if any("|z|" in h for h in header):
            z_i = next(i for i, h in enumerate(header) if "|z|" in h)
            series["|z|"] = []
        for row in body:
            x_labels.append(row[0])
            series["succ_hit"].append(_cell_float(row[idx["succ_hit"]]))
            if "succ_other" in idx:
                series["succ_other"].append(_cell_float(row[idx["succ_other"]]))
            if "|z|" in series:
                z_i = next(i for i, h in enumerate(header) if "|z|" in h)
                series["|z|"].append(_cell_float(row[z_i]))
        m = re.search(r"base_success\s*=\s*([\d.]+)%", title)
        if m:
            ref_line = float(m.group(1))
        y_format = "pct"
        y_label = "success %"
    elif "mean_hit" in idx:
        series = {"mean_hit": [], "mean_other": []}
        for row in body:
            x_labels.append(row[0])
            series["mean_hit"].append(_cell_float(row[idx["mean_hit"]]))
            if "mean_other" in idx:
                series["mean_other"].append(_cell_float(row[idx["mean_other"]]))
        m = re.search(r"base_mean_\w+\s*=\s*([-\d.]+)", title)
        ref_line = float(m.group(1)) if m else None
        y_label = "mean forward_rr"
    else:
        return ""

    if not x_labels:
        return ""
    chart = _svg_line_chart(
        x_labels=x_labels,
        series=series,
        y_label=y_label,
        ref_line=ref_line,
        y_format=y_format,
    )
    return f'<div class="chart-wrap">{chart}</div>' if chart else ""


def _chart_condition_set(header: List[str], body: List[List[str]], title: str) -> str:
    """Bar-style compare for Pass 2 ``condition_set`` tables (condition × succ_in, |z|)."""
    h_lower = [h.lower() for h in header]
    try:
        ci = h_lower.index("condition")
        ni = h_lower.index("n")
        si = h_lower.index("succ_in")
    except ValueError:
        return ""
    z_i = next((i for i, h in enumerate(header) if "|z|" in h), None)
    x_labels: List[str] = []
    succ: List[Optional[float]] = []
    z_vals: List[Optional[float]] = []
    for row in body:
        if len(row) <= max(ci, ni, si):
            continue
        name = row[ci].strip()
        n = row[ni].strip()
        x_labels.append(f"{name} (n={n})")
        succ.append(_cell_float(row[si]))
        if z_i is not None and z_i < len(row):
            z_vals.append(_cell_float(row[z_i]))
        else:
            z_vals.append(None)
    if not x_labels:
        return ""
    series: Dict[str, List[Optional[float]]] = {"succ_in": succ}
    if any(z is not None for z in z_vals):
        series["|z|"] = z_vals
    m = re.search(r"base_success\s*=\s*([\d.]+)%", title)
    ref_line = float(m.group(1)) if m else None
    chart = _svg_line_chart(
        x_labels=x_labels,
        series=series,
        y_label="success %",
        ref_line=ref_line,
        y_format="pct",
    )
    return f'<div class="chart-wrap">{chart}</div>' if chart else ""


def _chart_ic_decay(header: List[str], body: List[List[str]]) -> str:
    h_lower = [h.lower() for h in header]
    try:
        fi = h_lower.index("feature")
        hi = h_lower.index("horizon")
        ii = h_lower.index("rank_ic")
    except ValueError:
        return ""
    by_feat: Dict[str, Dict[str, Optional[float]]] = {}
    horizons: List[str] = []
    for row in body:
        feat = row[fi]
        hor = row[hi]
        ic = _cell_float(row[ii])
        if hor not in horizons:
            horizons.append(hor)
        by_feat.setdefault(feat, {})[hor] = ic
    series = {f: [by_feat[f].get(h) for h in horizons] for f in sorted(by_feat.keys())}
    chart = _svg_line_chart(
        x_labels=horizons,
        series=series,
        y_label="rank IC",
        ref_line=0.0,
        y_format="num",
    )
    return f'<div class="chart-wrap">{chart}</div>' if chart else ""


def _md_to_html(text: str) -> str:
    lines = text.splitlines()
    title_blob = "\n".join(lines[:8])
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("# "):
            out.append(f"<h2>{_esc(stripped[2:])}</h2>")
            i += 1
            continue
        if stripped.startswith("## "):
            out.append(f"<h3>{_esc(stripped[3:])}</h3>")
            i += 1
            continue
        if stripped.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].strip().startswith("- "):
                body = lines[i].strip()[2:]
                body = re.sub(
                    r"`([^`]+)`",
                    lambda m: f"<code>{_esc(m.group(1))}</code>",
                    body,
                )
                out.append(f"<li>{body}</li>")
                i += 1
            out.append("</ul>")
            continue
        if stripped.startswith("|"):
            table, i = _parse_md_table(lines, i)
            if table:
                chart = _chart_for_table(table, title_blob)
                if chart:
                    out.append(chart)
                out.append(_table_html(table, z_col=True))
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            out.append(f"<p class='note'><strong>{_esc(stripped[2:-2])}</strong></p>")
            i += 1
            continue
        out.append(f"<p>{_esc(stripped)}</p>")
        i += 1
    return "\n".join(out)


def _table_html(rows: List[List[str]], *, z_col: bool = False) -> str:
    if not rows:
        return ""
    header, *body = rows
    z_idx = None
    if z_col:
        for j, h in enumerate(header):
            if "z" in h.lower().replace(" ", ""):
                z_idx = j
                break
    parts = ["<table><thead><tr>"]
    for h in header:
        parts.append(f"<th>{_esc(h)}</th>")
    parts.append("</tr></thead><tbody>")
    n_idx = None
    for j, h in enumerate(header):
        if h.strip().lower() == "n":
            n_idx = j
            break
    for row in body:
        cls = ""
        if z_idx is not None and z_idx < len(row):
            try:
                z = float(row[z_idx].replace("%", "").strip())
                if abs(z) >= 2.0:
                    cls = ' class="highlight"'
            except ValueError:
                pass
        if not cls and n_idx is not None and n_idx < len(row):
            try:
                if int(float(row[n_idx].strip())) < 50:
                    cls = ' class="muted"'
            except ValueError:
                pass
        parts.append(f"<tr{cls}>")
        for cell in row:
            parts.append(f"<td>{_esc(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _snotio_json_html(payload: Dict[str, Any]) -> str:
    feature = payload.get("feature", "?")
    op = payload.get("operator", ">=")
    mode = payload.get("snotio_mode", "proxy")
    rows = payload.get("rows") or []
    start = payload.get("start_threshold")
    end = payload.get("end_threshold")
    rec = payload.get("recommended") or payload.get("recommended_threshold")
    is_plateau = payload.get("is_plateau")
    conf = payload.get("confidence", "")
    reason = payload.get("reason", "")

    title = f"snotio_plateau · {feature} {op} ? ({mode})"
    parts = [f"<h2>{_esc(title)}</h2>"]
    meta = [f"<li>KPI: <code>snotio</code> / <code>{_esc(mode)}</code></li>"]
    if is_plateau:
        meta.append(
            f"<li>Plateau: <strong>[{start}, {end}]</strong> "
            f"recommended=<strong>{rec}</strong> conf={_esc(conf)}</li>"
        )
    else:
        meta.append(f"<li>No plateau: {_esc(reason or 'n/a')}</li>")
    parts.append("<ul>" + "".join(meta) + "</ul>")

    x_labels = [f"{float(r['threshold']):.3g}" for r in rows]
    snotios = [float(r.get("snotio") or 0) for r in rows]
    trades = [int(r.get("trades", 0)) for r in rows]
    plateau_x = (
        (float(start), float(end))
        if is_plateau and start is not None and end is not None
        else None
    )
    parts.append(
        _svg_line_chart(
            x_labels=x_labels,
            series={"snotio": snotios},
            y_label="snotio (entry_rr)",
            plateau_x=plateau_x,
        )
    )
    parts.append(
        _svg_line_chart(
            x_labels=x_labels,
            series={"trades": [float(t) for t in trades]},
            y_label="trade count",
            y_format="num",
        )
    )

    max_sn = max((abs(v) for v in snotios), default=1.0) or 1.0
    parts.append(
        "<table><thead><tr><th>threshold</th><th>trades</th><th>snotio</th><th></th></tr></thead><tbody>"
    )
    for r in rows:
        thr = float(r.get("threshold", 0))
        n_tr = int(r.get("trades", 0))
        sn = float(r.get("snotio", 0))
        too_few = bool(r.get("too_few"))
        in_plateau = (
            is_plateau and start is not None and end is not None and start <= thr <= end
        )
        cls = "highlight" if in_plateau else ("muted" if too_few else "")
        bar_w = min(100, int(100 * abs(sn) / max_sn)) if max_sn else 0
        parts.append(f"<tr class='{cls}'>")
        parts.append(f"<td>{thr:.3g}</td><td>{n_tr}</td><td>{sn:.4f}</td>")
        parts.append(
            f"<td><div class='bar'><span style='width:{bar_w}%'></span></div></td>"
        )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _load_section(path: Path) -> str:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kpi") == "snotio" or (
            "rows" in payload and "snotio_mode" in payload
        ):
            return _snotio_json_html(payload)
        return f"<h2>{_esc(path.name)}</h2><pre>{_esc(json.dumps(payload, indent=2))}</pre>"
    return _md_to_html(path.read_text(encoding="utf-8"))


def _page_css() -> str:
    return """
:root { --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#8b9cb3;
  --accent:#3d8bfd; --hi:#1e3a2f; --border:#2a3548; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  background: var(--bg); color: var(--text); margin: 0; padding: 1.5rem 2rem 3rem; line-height: 1.5; }
header { margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }
h1 { font-size: 1.5rem; margin: 0 0 0.5rem; }
.sub { color: var(--muted); font-size: 0.9rem; }
nav { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
nav a { color: var(--accent); text-decoration: none; font-size: 0.85rem;
  padding: 0.25rem 0.6rem; border: 1px solid var(--border); border-radius: 4px; }
nav a:hover { background: var(--card); }
section { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; }
section h2 { margin-top: 0; font-size: 1.1rem; color: var(--accent); }
.chart-wrap { margin: 0.75rem 0 0.25rem; overflow-x: auto; }
svg.chart { display: block; width: 100%; max-width: 720px; height: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 0.75rem 0; }
th, td { padding: 0.45rem 0.6rem; text-align: right; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
tr.highlight { background: var(--hi); }
tr.muted { opacity: 0.55; }
.bar { background: var(--border); height: 8px; border-radius: 4px; min-width: 80px; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 4px; }
ul { margin: 0.5rem 0; padding-left: 1.25rem; }
code { background: #0b1018; padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.9em; }
.note { color: var(--muted); font-size: 0.9rem; }
.subset { font-size: 0.88rem; color: var(--muted); margin: 0 0 0.75rem; padding: 0.35rem 0.6rem;
  background: #0b1018; border-left: 3px solid var(--accent); border-radius: 4px; }
.subset strong { color: #c5d4e8; }
"""


def section_order_from_scans(scans: Sequence[Dict[str, Any]]) -> List[str]:
    """Derive report section order from ``quick_layer_scans[].out`` paths."""
    names: List[str] = []
    for scan in scans:
        if not isinstance(scan, dict):
            continue
        out = scan.get("out")
        if not out:
            continue
        name = Path(str(out)).name
        if name and name not in names:
            names.append(name)
    return names


def subset_label_from_filters(filters: Sequence[str]) -> str:
    """Human-readable subset tag from rd_loop ``filter`` clauses."""
    text = " ".join(str(f) for f in filters)
    if "ema_1200_position>=" in text:
        return "bull · long-bias subset"
    if "ema_1200_position<=" in text:
        return "bear · short-bias subset"
    return "unfiltered subset"


def manifest_sections_from_scans(
    scans: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build per-artifact metadata from hypothesis ``quick_layer_scans``."""
    sections: List[Dict[str, Any]] = []
    for scan in scans:
        if not isinstance(scan, dict):
            continue
        out = scan.get("out")
        if not out:
            continue
        filt = scan.get("filter")
        if isinstance(filt, str):
            filt_list = [filt]
        elif isinstance(filt, list):
            filt_list = [str(x) for x in filt]
        else:
            filt_list = []
        sections.append(
            {
                "file": Path(str(out)).name,
                "mode": scan.get("mode"),
                "feature": scan.get("feature"),
                "filter": filt_list,
                "subset": subset_label_from_filters(filt_list),
            }
        )
    return sections


def load_manifest(scan_dir: Path) -> Optional[Dict[str, Any]]:
    """Optional ``report_manifest.json`` written by rd_loop."""
    path = scan_dir / _MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(blob, dict):
        return None
    return blob


def manifest_section_order(blob: Dict[str, Any]) -> List[str]:
    order = blob.get("section_order") or blob.get("artifacts")
    if isinstance(order, list):
        return [str(x) for x in order if x]
    sections = blob.get("sections")
    if isinstance(sections, list):
        names: List[str] = []
        for sec in sections:
            if isinstance(sec, dict) and sec.get("file"):
                names.append(str(sec["file"]))
        return names
    return []


def manifest_section_meta(blob: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    sections = blob.get("sections")
    if not isinstance(sections, list):
        return meta
    for sec in sections:
        if isinstance(sec, dict) and sec.get("file"):
            meta[str(sec["file"])] = sec
    return meta


def write_manifest(
    scan_dir: Path,
    *,
    section_order: Sequence[str],
    scans: Optional[Sequence[Dict[str, Any]]] = None,
    hypothesis_topic: Optional[str] = None,
) -> Path:
    """Persist section order + filter/subset labels for HTML regen."""
    scan_dir.mkdir(parents=True, exist_ok=True)
    path = scan_dir / _MANIFEST_NAME
    sections = manifest_sections_from_scans(scans or []) if scans else []
    by_file = {s["file"]: s for s in sections}
    ordered_sections: List[Dict[str, Any]] = []
    for name in section_order:
        if name in by_file:
            ordered_sections.append(by_file[name])
        else:
            ordered_sections.append({"file": name})
    payload: Dict[str, Any] = {
        "section_order": list(section_order),
        "sections": ordered_sections,
        "hypothesis_topic": hypothesis_topic,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def _section_badge_html(meta: Dict[str, Any]) -> str:
    subset = meta.get("subset")
    filt = meta.get("filter") or []
    if not subset and not filt:
        return ""
    parts = ['<p class="subset">']
    if subset:
        parts.append(f"<strong>{_esc(subset)}</strong>")
    if filt:
        parts.append(" · filter: ")
        parts.append(_esc(" AND ".join(str(f) for f in filt)))
    parts.append("</p>")
    return "".join(parts)


@dataclass
class ReportConfig:
    title: str = "Quick Scan Report"
    section_order: Optional[List[str]] = None
    include_extensions: Tuple[str, ...] = (".md", ".json")


def resolve_report_config(
    *,
    hypothesis: Optional[Dict[str, Any]] = None,
    html_block: Optional[Dict[str, Any]] = None,
    scan_dir: Optional[Path] = None,
) -> ReportConfig:
    """Merge ``quick_scan_html`` yaml block + hypothesis scans + manifest."""
    hyp = hypothesis or {}
    block = html_block or {}
    if block.get("title"):
        title = str(block["title"])
    elif hyp.get("topic"):
        title = f"{str(hyp['topic']).replace('_', ' ')} · Quick Scan Report"
    else:
        title = "Quick Scan Report"

    order: Optional[List[str]] = None
    raw_order = block.get("section_order")
    if raw_order == "scan_out" or raw_order is None:
        scans = hyp.get("quick_layer_scans") or hyp.get("research_scans") or []
        order = section_order_from_scans(scans) if scans else None
    elif isinstance(raw_order, list):
        order = [Path(str(x)).name for x in raw_order]

    if scan_dir and (not order or raw_order == "manifest"):
        blob = load_manifest(scan_dir)
        if blob:
            order = manifest_section_order(blob)

    return ReportConfig(title=title, section_order=order or None)


def _order_artifact_paths(
    paths: List[Path], section_order: Optional[Sequence[str]]
) -> List[Path]:
    by_name = {p.name: p for p in paths}
    ordered: List[Path] = []
    seen: set[str] = set()
    for name in section_order or []:
        p = by_name.get(name)
        if p is not None and name not in seen:
            ordered.append(p)
            seen.add(name)
    for p in sorted(paths, key=lambda x: x.name):
        if p.name not in seen:
            ordered.append(p)
            seen.add(p.name)
    return ordered


def build_report(
    scan_dir: Path,
    *,
    title: str = "Quick Scan Report",
    section_order: Optional[Sequence[str]] = None,
    section_meta: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> str:
    all_files = [
        p
        for p in scan_dir.iterdir()
        if p.suffix in (".md", ".json")
        and p.is_file()
        and p.name not in (_REPORT_HTML_NAME, _MANIFEST_NAME)
    ]
    ordered = _order_artifact_paths(all_files, section_order)
    meta = dict(section_meta or {})
    if not meta:
        blob = load_manifest(scan_dir)
        if blob:
            meta = manifest_section_meta(blob)

    toc: List[str] = []
    sections: List[str] = []
    for p in ordered:
        sid = re.sub(r"[^a-zA-Z0-9_-]", "_", p.stem)
        m = meta.get(p.name, {})
        label = p.name
        if m.get("subset"):
            label = f"{p.name} ({m['subset']})"
        toc.append(f'<a href="#{sid}">{_esc(label)}</a>')
        body = _load_section(p)
        badge = _section_badge_html(m)
        sections.append(f'<section id="{sid}">{badge}{body}</section>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)}</title>
<style>{_page_css()}</style>
</head>
<body>
<header>
  <h1>{_esc(title)}</h1>
  <p class="sub">Source: <code>{_esc(str(scan_dir))}</code> · {len(ordered)} artifacts · charts above tables</p>
  <nav>{"".join(toc)}</nav>
</header>
{"".join(sections)}
</body>
</html>
"""


def build_report_from_config(
    scan_dir: Path,
    config: ReportConfig,
    *,
    section_meta: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> str:
    meta = section_meta
    if meta is None:
        blob = load_manifest(scan_dir)
        meta = manifest_section_meta(blob) if blob else None
    return build_report(
        scan_dir,
        title=config.title,
        section_order=config.section_order,
        section_meta=meta,
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="HTML report for rd_loop quick_scan outputs"
    )
    p.add_argument(
        "scan_dir",
        help="Directory with .md / .json scan outputs",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output HTML path (default: <scan_dir>/report.html)",
    )
    p.add_argument("--title", default=None, help="Report title (overrides yaml)")
    p.add_argument(
        "--hypothesis-yaml",
        default=None,
        help="rd_loop hypothesis yaml: section order from quick_layer_scans[].out",
    )
    p.add_argument(
        "--section-order",
        default=None,
        help="Comma-separated artifact basenames (overrides yaml/manifest)",
    )
    args = p.parse_args(argv)

    scan_dir = Path(args.scan_dir)
    if not scan_dir.is_absolute():
        scan_dir = (PROJECT_ROOT / scan_dir).resolve()
    if not scan_dir.is_dir():
        print(f"ERROR: not a directory: {scan_dir}", file=sys.stderr)
        return 3

    hypothesis: Optional[Dict[str, Any]] = None
    html_block: Optional[Dict[str, Any]] = None
    if args.hypothesis_yaml:
        hyp_path = Path(args.hypothesis_yaml)
        if not hyp_path.is_absolute():
            hyp_path = (PROJECT_ROOT / hyp_path).resolve()
        blob = yaml.safe_load(hyp_path.read_text(encoding="utf-8")) or {}
        if isinstance(blob, dict):
            hypothesis = blob
            raw_html = blob.get("quick_scan_html")
            if isinstance(raw_html, dict):
                html_block = raw_html

    config = resolve_report_config(
        hypothesis=hypothesis, html_block=html_block, scan_dir=scan_dir
    )
    if args.title:
        config.title = args.title
    if args.section_order:
        config.section_order = [
            x.strip() for x in args.section_order.split(",") if x.strip()
        ]
    section_meta = None
    if hypothesis:
        scans = (
            hypothesis.get("quick_layer_scans")
            or hypothesis.get("research_scans")
            or []
        )
        section_meta = {
            s["file"]: s for s in manifest_sections_from_scans(scans) if s.get("file")
        }

    out = Path(args.out) if args.out else scan_dir / _REPORT_HTML_NAME
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_report_from_config(scan_dir, config, section_meta=section_meta),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
