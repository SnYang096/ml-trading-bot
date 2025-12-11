from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import yaml

CONFIG_DIR = Path("/home/yin/trading/ml_trading_bot/config/smart_money_triggered")


def _load_list_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    items = data.get("lists") or []
    return [str(x).strip() for x in items if str(x).strip()]


def _load_key(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    val = data.get("key") or ""
    return os.path.expandvars(str(val))


@dataclass
class SmartMoneySettings:
    token: str
    stock_symbols: List[str] = field(default_factory=list)
    crypto_symbols: List[str] = field(default_factory=list)
    storage_dir: Path = Path("/home/yin/trading/ml_trading_bot/data/smart_money_triggered")

    def ensure_storage(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def unique_symbols(self) -> List[str]:
        seen = set()
        result = []
        for sym in list(self.stock_symbols) + list(self.crypto_symbols):
            if sym not in seen:
                seen.add(sym)
                result.append(sym)
        return result


def load_settings(config_dir: Path | str = CONFIG_DIR) -> SmartMoneySettings:
    base = Path(config_dir)
    token = _load_key(base / "key.yaml")
    stock_symbols = _load_list_file(base / "china_stocks.yaml")
    crypto_symbols = _load_list_file(base / "cryptos.yaml")
    settings = SmartMoneySettings(
        token=token,
        stock_symbols=stock_symbols,
        crypto_symbols=crypto_symbols,
    )
    settings.ensure_storage()
    return settings

