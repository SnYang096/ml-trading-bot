"""Jinja2-rendered dashboard shell (cards loaded via分页 API)."""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path
from typing import List, Literal, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .cards_html import _strategy_tab_slug
from .constants import (
    DASHBOARD_ASSET_PREFIX,
    DASHBOARD_CARD_PAGE_DEFAULT,
    PACKAGE_DIR,
    dashboard_visibility,
)
from .scan import scan_flat_run_index, scan_rolling_run_index

PageKind = Literal["all", "research", "prod"]


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PACKAGE_DIR / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _tab_buttons_for_counter(
    counter: Counter,
    *,
    initial_tab: str,
) -> str:
    rows_n = sum(counter.values())
    strategies_sorted = sorted(k for k in counter.keys() if k)
    tab_buttons: List[str] = [
        f'<button type="button" class="tab-btn" data-tab="__all__" '
        f'aria-selected="{"true" if initial_tab == "__all__" else "false"}">'
        f'全部 <span class="badge">{rows_n}</span></button>'
    ]
    for s in strategies_sorted:
        slug = _strategy_tab_slug(s)
        n = counter[s]
        sel = 'aria-selected="true"' if s == initial_tab else 'aria-selected="false"'
        tab_buttons.append(
            f'<button type="button" class="tab-btn" data-tab="{html.escape(s, quote=True)}" '
            f'data-slug="{html.escape(slug, quote=True)}" {sel}>'
            f'{html.escape(s)} <span class="badge">{n}</span></button>'
        )
    return "\n".join(tab_buttons)


def render_dashboard(
    results_root: Path,
    *,
    strategy_filter: Optional[str],
    q: Optional[str],
    page: PageKind = "all",
) -> str:
    """Shell HTML：策略 Tab + 卡片容器（``page`` 决定 Rolling / History / 合并）。"""
    rr = results_root.resolve()
    roll_idx = scan_rolling_run_index(rr, strategy_filter=None, q=q)
    flat_idx = scan_flat_run_index(rr, strategy_filter=None, q=q)

    if page == "research":
        rows_meta = list(roll_idx)
        title = "研究管线 · Rolling（_rolling_sim）"
        form_action = "/dashboard/research"
    elif page == "prod":
        rows_meta = list(flat_idx)
        title = "上线管线 · History / 单次实验"
        form_action = "/dashboard/prod"
    else:
        rows_meta = roll_idx + flat_idx
        title = "results 实验看板（合并视图）"
        # 仅单元测试等内部调用；浏览器入口已取消 /dashboard/all。
        form_action = "/dashboard/research"

    counter = Counter((r.get("strategy") or "") for r in rows_meta)

    filter_hint = ""
    if strategy_filter:
        filter_hint += f" · URL strategy=<code>{html.escape(strategy_filter)}</code>（默认策略 tab）"
    if q:
        filter_hint += f" · q=<code>{html.escape(q)}</code>"

    initial_tab = "__all__"
    if strategy_filter and strategy_filter in counter:
        initial_tab = strategy_filter

    tab_buttons_html = _tab_buttons_for_counter(counter, initial_tab=initial_tab)

    q_esc = html.escape(q or "")
    hist_count = len(flat_idx)
    history_banner = ""
    if page in ("all", "prod") and hist_count == 0:
        history_banner = """<p class="hist-zero-banner" role="status">当前 <strong>0</strong> 条单次实验：磁盘上尚无符合规则的 History 目录，或全部被上方搜索/策略筛选掉。
          规则路径示例：<code>research_history/&lt;策略&gt;/&lt;时间戳&gt;/</code>（相对 results/）。点「仅 History」时若此处为空，页面只会剩下本说明与标题。</p>"""

    vis = dashboard_visibility()
    tpl = _jinja_env().get_template("dashboard.html")
    return tpl.render(
        title=title,
        results_root=html.escape(str(rr)),
        rolling_n=len(roll_idx),
        flat_n=len(flat_idx),
        total_n=len(rows_meta),
        filter_hint=filter_hint,
        initial_tab_esc=html.escape(initial_tab, quote=True),
        initial_tab_raw=initial_tab,
        q_esc=q_esc,
        tab_buttons_html=tab_buttons_html,
        history_banner=history_banner,
        card_page=DASHBOARD_CARD_PAGE_DEFAULT,
        dash_assets=DASHBOARD_ASSET_PREFIX,
        form_action=form_action,
        page_scope=page,
        show_sec_rolling=page in ("all", "research"),
        show_sec_history=page in ("all", "prod"),
        show_partition_tabs=page == "all",
        show_btn_stale=page in ("all", "research"),
        show_btn_flat_del=page in ("all", "prod"),
        show_prod_ops_panel=page == "prod",
        adopt_buttons_js=page == "prod",
        nav_active=page,
        show_scope_nav=vis["research"] or vis["prod"],
        vis_research=vis["research"],
        vis_prod=vis["prod"],
    )


def render_dashboard_hub(results_root: Path) -> str:
    """入口页：研究管线 / 上线管线 二选一。"""
    rr = results_root.resolve()
    roll_idx = scan_rolling_run_index(rr, strategy_filter=None, q=None)
    flat_idx = scan_flat_run_index(rr, strategy_filter=None, q=None)
    vis = dashboard_visibility()
    tpl = _jinja_env().get_template("dashboard_hub.html")
    return tpl.render(
        results_root=html.escape(str(rr)),
        rolling_n=len(roll_idx),
        flat_n=len(flat_idx),
        dash_assets=DASHBOARD_ASSET_PREFIX,
        vis_research=vis["research"],
        vis_prod=vis["prod"],
    )
