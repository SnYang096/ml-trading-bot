"""Uniform JSON envelopes for API responses."""

from __future__ import annotations

from typing import Any, Dict, Optional


def ok(
    data: Any,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {"ok": True, "data": data, "meta": meta or {}}


def err(message: str, *, code: str = "error", detail: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": False, "error": {"code": code, "message": message}}
    if detail is not None:
        payload["error"]["detail"] = detail
    return payload
