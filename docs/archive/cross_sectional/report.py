from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .model import FactorBacktestResult


@dataclass
class ReportContext:
    title: str
    max_lag: int
    periods_per_year: int
    preprocessing: str
    symbols: Optional[str] = None
    horizon: Optional[int] = None
    observations: Optional[int] = None
    timestamps: Optional[int] = None
    assets_per_timestamp: Optional[float] = None


def generate_markdown_report(
    result: FactorBacktestResult,
    context: ReportContext,
) -> str:
    """
    Render a Markdown report combining Fama-MacBeth (Newey-West) and IC diagnostics.
    """
    combined = result.combined_metrics(
        max_lag=context.max_lag,
        periods_per_year=context.periods_per_year,
    )
    factor_summary = result.factor_summary(context.periods_per_year)
    ic_summary = result.ic_summary(context.periods_per_year)
    avg_r2 = float(result.r2.mean()) if not result.r2.empty else float("nan")

    header_lines = [
        f"# {context.title}",
        "",
        f"- Generated at: {datetime.utcnow().isoformat(timespec='seconds')}Z",
        f"- Symbols: {context.symbols or 'N/A'}",
        f"- Forward horizon: {context.horizon or 'N/A'} bars",
        f"- Cross-sectional preprocessing: {context.preprocessing}",
        f"- Valid timestamps: {context.timestamps or 'N/A'}",
        f"- Mean assets per timestamp: {context.assets_per_timestamp or 'N/A'}",
        f"- Total observations: {context.observations or 'N/A'}",
        f"- Average cross-sectional R²: {avg_r2:.4f}",
        "",
    ]

    combined_table = _df_to_markdown(
        combined.reset_index(),
        float_cols=[
            "mean",
            "nw_se",
            "nw_t",
            "ann_mean",
            "ann_vol",
            "ir",
            "ic_mean",
            "ic_std",
            "ic_t",
            "ic_ir",
            "ic_ir_annual",
        ],
    )

    factor_table = _df_to_markdown(
        factor_summary.reset_index(),
        float_cols=["mean", "std", "t_stat", "ann_mean", "ann_vol", "ir"],
    )

    ic_table = _df_to_markdown(
        ic_summary.reset_index(),
        float_cols=["ic_mean", "ic_std", "ic_t", "ic_ir", "ic_ir_annual"],
    )

    sections = [
        "## Combined Factor Diagnostics (Fama-MacBeth + IC)",
        "",
        combined_table,
        "",
        "## Factor Return Statistics (Fama-MacBeth)",
        "",
        factor_table,
        "",
        "## Information Coefficient Summary",
        "",
        ic_table,
    ]

    return "\n".join(header_lines + sections) + "\n"


def _df_to_markdown(df: pd.DataFrame, float_cols: Optional[list[str]] = None) -> str:
    if df.empty:
        return "_No data available_"
    fmt_df = df.copy()
    if float_cols:
        for col in float_cols:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"
                )
    return fmt_df.to_markdown(index=False)


def write_report(path: str | Path, markdown: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
