from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd


def load_top_factors_list(top_factors_path: str) -> List[str]:
    with open(top_factors_path, "r") as f:
        data = json.load(f)
    factors = data.get("top_factors", [])
    top_list: List[str] = []
    for item in factors:
        if isinstance(item, dict) and "name" in item:
            top_list.append(str(item["name"]))
        else:
            top_list.append(str(item))
    return top_list


def filter_engineered_by_topk(
    engineered_data: Dict[str, pd.DataFrame],
    top_list: List[str],
    keep_essentials: List[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    if not top_list:
        return engineered_data
    if keep_essentials is None:
        keep_essentials = ["close", "volume", "taker_buy_ratio", "cvd"]

    filtered: Dict[str, pd.DataFrame] = {}
    for tf, df in engineered_data.items():
        keep_cols = [c for c in df.columns if c in top_list]
        for essential in keep_essentials:
            if essential in df.columns and essential not in keep_cols:
                keep_cols.append(essential)
        if not keep_cols:
            filtered[tf] = df
        else:
            filtered[tf] = df[keep_cols].copy()
    return filtered


