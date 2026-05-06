"""Paged HTML fragments for dashboard ledger lists."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .cards_html import _ledger_cards_fragment
from .scan import (
    enrich_dashboard_rows,
    filter_run_index_by_strategy,
    scan_flat_run_index,
    scan_rolling_run_index,
)


def build_dashboard_cards_slice_html(
    results_root: Path,
    *,
    kind: str,
    offset: int,
    limit: int,
    strategy_tab: Optional[str],
    q: Optional[str],
    show_adopt_buttons: bool = False,
) -> tuple[str, int, int]:
    """Build HTML for one page of cards.

    Returns ``(html_fragment, total_after_strategy_filter, next_offset)``.
    """
    rr = results_root.resolve()
    if kind == "rolling":
        idx = scan_rolling_run_index(rr, strategy_filter=None, q=q)
    elif kind == "flat":
        idx = scan_flat_run_index(rr, strategy_filter=None, q=q)
    else:
        raise ValueError(f"invalid kind: {kind!r}")
    idx = filter_run_index_by_strategy(idx, strategy_tab)
    total = len(idx)
    off = max(0, int(offset))
    lim = max(1, min(int(limit), 500))
    chunk = idx[off : off + lim]
    enriched = enrich_dashboard_rows(rr, chunk)
    show_adopt = show_adopt_buttons and kind == "flat"
    fragment = _ledger_cards_fragment(enriched, show_adopt_buttons=show_adopt)
    next_off = off + len(chunk)
    return fragment, total, next_off
