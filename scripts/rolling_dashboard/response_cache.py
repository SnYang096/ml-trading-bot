"""进程内短时 TTL 缓存（减轻重复磁盘扫描）；删除成功后由 handler 整包清空。"""

from __future__ import annotations

import threading
import time
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """线程安全的简单 TTL 缓存。"""

    def __init__(self, ttl_s: float, *, max_entries: int = 512) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._data: dict[str, tuple[float, T]] = {}

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def get(self, key: str) -> Optional[T]:
        if self.ttl_s <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            exp, val = item
            if now > exp:
                del self._data[key]
                return None
            return val

    def set(self, key: str, value: T) -> None:
        if self.ttl_s <= 0:
            return
        with self._lock:
            if len(self._data) >= self.max_entries and key not in self._data:
                self._data.clear()
            self._data[key] = (time.monotonic() + self.ttl_s, value)


class DashboardAPICaches:
    """看板 API 用的两组缓存：JSON 字节流 + 卡片 HTML 片段。"""

    def __init__(self, ttl_s: float) -> None:
        self.ttl_s = ttl_s
        self.json_bytes = TTLCache[bytes](ttl_s)
        self.cards = TTLCache[tuple[bytes, int, int]](ttl_s)

    def invalidate_all(self) -> None:
        self.json_bytes.clear()
        self.cards.clear()
