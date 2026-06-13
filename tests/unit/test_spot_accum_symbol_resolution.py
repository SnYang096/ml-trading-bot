from __future__ import annotations

from pathlib import Path

import yaml

from scripts.pipeline.strategy_symbols import resolve_strategy_symbols


def test_spot_meta_empty_include_uses_universe_base(tmp_path: Path):
    pkg = tmp_path / "spot_accum_simple"
    pkg.mkdir(parents=True)
    (pkg / "meta.yaml").write_text(
        yaml.safe_dump({"strategy": {"symbol_include": [], "symbol_exclude": []}}),
        encoding="utf-8",
    )
    sel = resolve_strategy_symbols(
        strategy="spot_accum_simple",
        base_symbols=["BTCUSDT", "ETHUSDT", "HYPEUSDT"],
        strategy_config_dir=pkg,
    )
    assert sel.resolved_symbols == ["BTCUSDT", "ETHUSDT", "HYPEUSDT"]


def test_spot_meta_explicit_include_filters_universe(tmp_path: Path):
    pkg = tmp_path / "spot_accum_simple"
    pkg.mkdir(parents=True)
    (pkg / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "strategy": {
                    "symbol_include": ["BTCUSDT", "ETHUSDT"],
                    "symbol_exclude": [],
                }
            }
        ),
        encoding="utf-8",
    )
    sel = resolve_strategy_symbols(
        strategy="spot_accum_simple",
        base_symbols=["BTCUSDT", "ETHUSDT", "HYPEUSDT"],
        strategy_config_dir=pkg,
    )
    assert sel.resolved_symbols == ["BTCUSDT", "ETHUSDT"]
