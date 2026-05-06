"""HTML fragments for dashboard ledger cards."""

from __future__ import annotations

import html
import re
import time
from typing import Any, Dict, List
from urllib.parse import quote

from .paths import _fmt_bytes

_NEW_TAB = ' target="_blank" rel="noopener noreferrer"'


def _fmt_mtime(ts: float) -> str:
    if not ts:
        return "—"
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    except (OverflowError, OSError, ValueError):
        return "—"


def _strategy_tab_slug(strategy: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", strategy or "unknown")


def _ledger_cards_fragment(
    rows: List[Dict[str, Any]], *, show_adopt_buttons: bool = False
) -> str:
    """生成 ledger / 单次实验 卡片列表 HTML。"""
    if not rows:
        return '<p class="empty">暂无批次</p>'
    parts: List[str] = []
    for r in rows:
        base = "/" + quote(r["rel_path"])
        links: List[str] = []
        if r["has_continuous"]:
            links.append(
                f'<a href="{base}/trading_map_continuous.html"{_NEW_TAB}>continuous</a>'
            )
        if r["has_stitched"]:
            links.append(
                f'<a href="{base}/trading_map_stitched.html"{_NEW_TAB}>stitched</a>'
            )
        if r["has_summary_json"]:
            links.append(
                f'<a href="{base}/stitched_summary.json"{_NEW_TAB}>stitched_summary</a>'
            )
        if r.get("has_report_json"):
            links.append(f'<a href="{base}/report.json"{_NEW_TAB}>report.json</a>')
        if r["has_pipeline_log"]:
            links.append(f'<a href="{base}/pipeline.log"{_NEW_TAB}>pipeline.log</a>')
        link_cell = " · ".join(links) if links else "—"
        r_fmt = (
            f"{r['stitched_total_r']:.4f}"
            if isinstance(r.get("stitched_total_r"), (int, float))
            else "—"
        )
        tr_fmt = (
            str(r["stitched_total_trades"])
            if r.get("stitched_total_trades") is not None
            else "—"
        )
        mo = str(r.get("count_months")) if r.get("count_months") is not None else "—"
        disp_rid = html.escape(str(r.get("run_id") or r["ledger_ts"]))
        rid_note = ""
        if not r.get("run_id"):
            rid_note = (
                f' <span class="dim">（stitched_summary 未写 run_id 时，目录名 '
                f'<code>{html.escape(r["ledger_ts"])}</code> 即批次 ID）</span>'
            )
        mode_pill = html.escape(str(r.get("mode") or "—"))
        rel_path = r["rel_path"]
        rel_esc = html.escape(rel_path)
        strat = html.escape(str(r.get("strategy") or ""), quote=True)
        rk = str(r.get("run_kind") or "")
        rk_attr = html.escape(rk, quote=True)
        rk_disp = "rolling" if rk == "rolling" else "单次"
        ms = r.get("metrics_source")
        if r.get("needs_stitch_cleanup"):
            src_hint = (
                '<span class="src-flag warn" title="无有效 stitched_summary.json（缺失、空文件或损坏），'
                '或 Stitch 未完成">未汇总</span>'
            )
        elif ms == "report_json":
            src_hint = '<span class="src-flag" title="R / Trades 来自 report.json（无 stitched 或未写完）">report</span>'
        elif ms == "stitched_summary":
            src_hint = '<span class="src-flag ok" title="指标来自 stitched_summary.json">stitched</span>'
        else:
            src_hint = '<span class="src-flag dim" title="摘要字段为空">—</span>'
        mt_disp = _fmt_mtime(float(r.get("mtime") or 0))
        adopt_btns = ""
        if show_adopt_buttons and rk == "flat":
            adopt_btns = (
                '<button type="button" class="btn-adopt-cmd">复制 adopt 命令</button>'
                '<button type="button" class="btn-deploy-hint">deploy 说明…</button>'
            )
        parts.append(
            f"""
<article class="ledger-card" data-strategy="{strat}" data-run-kind="{rk_attr}" data-ledger-rel="{html.escape(rel_path, quote=True)}">
  <div class="ledger-top">
    <div class="ledger-ids">
      <span class="ledger-ts">{html.escape(r["ledger_ts"])}</span>
      <span class="pill rk">{html.escape(rk_disp)}</span>
      <span class="pill pipeline">{html.escape(r.get("pipeline_dir") or "")}</span>
      <span class="pill mode">{mode_pill}</span>
    </div>
    <div class="ledger-metrics">
      <span title="数据来源"><strong>R</strong> {r_fmt} {src_hint}</span>
      <span><strong>Trades</strong> {tr_fmt}</span>
      <span><strong>Months</strong> {mo}</span>
      <span><strong>阶段目录</strong> {r.get("sub_stage_dirs", 0)}</span>
      <span><strong>约体积*</strong> {_fmt_bytes(int(r.get("bytes_total") or 0))}</span>
    </div>
  </div>
  <div class="ledger-path"><code>{rel_esc}</code></div>
  <div class="ledger-meta muted">
    <span><strong>run_id</strong> <code>{disp_rid}</code>{rid_note}</span>
    <span class="mtime-meta"><strong>mtime</strong> {html.escape(mt_disp)}</span>
  </div>
  <div class="quick-links">{link_cell}</div>
  <div class="card-actions">
    <button type="button" class="btn-del">删除此目录…</button>
    {adopt_btns}
  </div>
</article>"""
        )
    return "\n".join(parts)
