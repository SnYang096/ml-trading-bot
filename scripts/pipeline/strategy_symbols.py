from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import yaml


@dataclass(frozen=True)
class StrategySymbolSelection:
    strategy: str
    base_symbols: List[str]
    resolved_symbols: List[str]
    include_symbols: List[str]
    exclude_symbols: List[str]
    meta_path: Optional[Path]


def parse_symbol_csv(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").split(",") if s.strip()]


def format_symbol_csv(symbols: Iterable[str]) -> str:
    return ",".join(str(s).strip().upper() for s in symbols if str(s).strip())


def resolve_strategy_symbols(
    *,
    strategy: str,
    base_symbols: Iterable[str],
    strategy_config_dir: Optional[Path],
) -> StrategySymbolSelection:
    base = [str(s).strip().upper() for s in base_symbols if str(s).strip()]
    include: List[str] = []
    exclude: List[str] = []
    meta_path: Optional[Path] = None
    if strategy_config_dir:
        meta_path = strategy_config_dir / "meta.yaml"
        include, exclude = _load_strategy_symbol_policies(meta_path)

    resolved = list(base)
    if include:
        include_set = set(include)
        resolved = [s for s in resolved if s in include_set]
    if exclude:
        exclude_set = set(exclude)
        resolved = [s for s in resolved if s not in exclude_set]
    return StrategySymbolSelection(
        strategy=str(strategy),
        base_symbols=base,
        resolved_symbols=resolved,
        include_symbols=include,
        exclude_symbols=exclude,
        meta_path=meta_path if meta_path and meta_path.is_file() else None,
    )


def _load_strategy_symbol_policies(meta_path: Path) -> tuple[List[str], List[str]]:
    if not meta_path.is_file():
        return [], []
    try:
        raw = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return [], []
    if not isinstance(raw, dict):
        return [], []
    block = raw.get("strategy")
    if not isinstance(block, dict):
        block = raw
    include = _normalize_symbol_list(block.get("symbol_include"))
    exclude = _normalize_symbol_list(block.get("symbol_exclude"))
    return include, exclude


def _normalize_symbol_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        vals = [x.strip().upper() for x in raw.split(",") if x.strip()]
        return list(dict.fromkeys(vals))
    if isinstance(raw, list):
        vals = [str(x).strip().upper() for x in raw if str(x).strip()]
        return list(dict.fromkeys(vals))
    return []
