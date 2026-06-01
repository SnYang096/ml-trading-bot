"""HTML shell for local R&D experiment explorer."""

from __future__ import annotations

import html

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .constants import DASHBOARD_ASSET_PREFIX, PACKAGE_DIR, experiments_root_path


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PACKAGE_DIR / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_rd_page() -> str:
    tpl = _jinja_env().get_template("rd.html")
    return tpl.render(
        dash_assets=DASHBOARD_ASSET_PREFIX,
        experiments_root=html.escape(str(experiments_root_path())),
    )
