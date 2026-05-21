"""Optional HTTP Basic Auth for business console (P3)."""

from __future__ import annotations

import base64
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, user: str, password: str) -> None:
        super().__init__(app)
        self._user = user
        self._password = password

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/api/health":
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return _challenge()
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return _challenge()
        if ":" not in decoded:
            return _challenge()
        u, p = decoded.split(":", 1)
        if not (
            secrets.compare_digest(u, self._user)
            and secrets.compare_digest(p, self._password)
        ):
            return _challenge()
        return await call_next(request)


def _challenge() -> Response:
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="mlbot-business-console"'},
        content="Unauthorized",
    )
