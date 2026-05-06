"""HTML directory listing for ``/browse``."""

from __future__ import annotations

import html
from pathlib import Path
from typing import List

from .paths import _encode_rel_path_url, _fmt_bytes, _safe_browse_target


def _shallow_dir_bytes(d: Path) -> int:
    """与实验批次卡片类似的浅层体积：本目录文件 + 一级子目录内文件（避免整树 walk 卡死）。"""
    total = 0
    try:
        for child in d.iterdir():
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    pass
            elif child.is_dir():
                try:
                    for sub in child.iterdir():
                        if sub.is_file():
                            try:
                                total += sub.stat().st_size
                            except OSError:
                                pass
                except OSError:
                    pass
    except OSError:
        pass
    return total


def render_browse_page(results_root: Path, listed_dir: Path, rel_under: str) -> str:
    """简易目录列表页。"""
    results_root = results_root.resolve()
    rel_under = rel_under.strip().strip("/").replace("\\", "/")
    rows_html: List[str] = []
    try:
        entries = sorted(
            listed_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except OSError:
        entries = []

    for p in entries:
        name = p.name
        if name.startswith("."):
            continue
        try:
            rel_item = p.relative_to(results_root).as_posix()
        except ValueError:
            continue
        enc = _encode_rel_path_url(rel_item)
        if p.is_dir():
            href = f"/browse/{enc}"
            kind = "目录"
        else:
            href = f"/{enc}"
            kind = "文件"
        try:
            if p.is_file():
                sz_b = p.stat().st_size
                sz_s = _fmt_bytes(sz_b)
            else:
                sz_b = _shallow_dir_bytes(p)
                sz_s = _fmt_bytes(sz_b) if sz_b > 0 else "0 B"
        except OSError:
            sz_s = "—"
        # 打开静态文件新标签页；目录仍在当前页导航。
        blank = ' target="_blank" rel="noopener noreferrer"' if p.is_file() else ""
        rows_html.append(
            f"<tr><td>{html.escape(kind)}</td>"
            f'<td><a href="{html.escape(href, quote=True)}"{blank}>{html.escape(name)}</a></td>'
            f'<td class="muted">{html.escape(sz_s)}</td></tr>'
        )

    crumbs: List[str] = ['<a href="/browse">results 根目录</a>']
    if rel_under:
        acc: List[str] = []
        for part in rel_under.split("/"):
            if not part:
                continue
            acc.append(part)
            sub = "/".join(acc)
            href_c = "/browse/" + _encode_rel_path_url(sub)
            crumbs.append(
                f'<a href="{html.escape(href_c, quote=True)}">{html.escape(part)}</a>'
            )

    crumb_html = " / ".join(crumbs)

    parent_row = ""
    if rel_under:
        parts = rel_under.split("/")
        parent_rel = "/".join(parts[:-1])
        parent_href = (
            "/browse"
            if not parent_rel
            else f"/browse/{_encode_rel_path_url(parent_rel)}"
        )
        parent_row = f'<p class="nav"><a href="{html.escape(parent_href, quote=True)}">↑ 上级目录</a></p>'

    table_body = (
        "\n".join(rows_html)
        if rows_html
        else '<tr><td colspan="3" class="muted">（空目录）</td></tr>'
    )

    title_disp = rel_under or "(根)"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>浏览 {html.escape(title_disp)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #0d1117; color: #e6edf3;
      padding: 1rem 1.25rem 2rem; }}
    a {{ color: #58a6ff; }}
    .muted {{ color: #8b949e; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 720px; }}
    th, td {{ border: 1px solid #30363d; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #161b22; }}
    .back {{ margin-bottom: 1rem; }}
    code {{ font-size: 0.85rem; }}
  </style>
</head>
<body>
  <p class="back"><a href="/dashboard">← 返回实验看板</a></p>
  <h1 style="font-size:1.1rem;">results 目录浏览</h1>
  <p class="muted" style="font-size:0.85rem;">当前：<code>{html.escape(rel_under or ".")}</code></p>
  <p style="font-size:0.9rem;">{crumb_html}</p>
  {parent_row}
  <p class="muted" style="font-size:0.78rem;margin:0.35rem 0 0.5rem;">目录「大小」为近似值：本目录下文件 + 一级子目录内文件（不做整树统计）。</p>
  <table>
    <thead><tr><th>类型</th><th>名称</th><th>大小</th></tr></thead>
    <tbody>
    {table_body}
    </tbody>
  </table>
</body>
</html>"""
