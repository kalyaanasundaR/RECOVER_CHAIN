"""Cross-cutting HTTP security: startup config checks, a dependency-free
in-process rate limiter, and an upload-size guard.

Kept deliberately small and stdlib-only so the app still runs offline with zero
extra services. For multi-process / multi-node deployments swap the limiter for
a Redis token bucket (see ``infrastructure.redis_client``).
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from threading import Lock

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# --- configuration --------------------------------------------------------------

_DEV_API_KEY = "test-api-key"

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1") != "0"
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))  # requests / minute / client
RATE_LIMIT_WINDOW_S = 60
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))  # 512 MiB


def check_startup_config() -> None:
    """Emit a loud warning when the app boots with insecure defaults.

    Set ``ALLOW_INSECURE_DEFAULTS=1`` to silence it in throwaway environments.
    """
    if os.getenv("ALLOW_INSECURE_DEFAULTS") == "1":
        return
    problems: list[str] = []
    if os.getenv("API_KEY", _DEV_API_KEY) == _DEV_API_KEY:
        problems.append("API_KEY is the built-in dev key — set a strong API_KEY")
    if os.getenv("CORS_ORIGINS", "*") == "*":
        problems.append("CORS_ORIGINS is '*' — set an explicit allow-list")
    if not os.getenv("OPERATOR_API_KEY"):
        problems.append(
            "OPERATOR_API_KEY is unset — API_KEY holders can read traces/audit/execution "
            "history; set OPERATOR_API_KEY to separate operator (read) from user access"
        )
    for p in problems:
        logger.warning("INSECURE DEFAULT: %s", p)


# --- rate limiting ------------------------------------------------------------


class _SlidingWindow:
    """Per-client sliding-window counter. O(1) amortised per request."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, limit: int, window_s: int) -> bool:
        now = time.monotonic()
        cutoff = now - window_s
        with self._lock:
            dq = self._hits.setdefault(key, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= limit:
                return False
            dq.append(now)
            # opportunistic cleanup so idle clients don't leak memory
            if len(self._hits) > 4096:
                for k in [k for k, v in self._hits.items() if not v or v[-1] < cutoff]:
                    self._hits.pop(k, None)
            return True


_window = _SlidingWindow()

# Health / observability endpoints must never be rate limited.
_EXEMPT_PREFIXES = ("/health", "/system/health", "/docs", "/openapi.json", "/redoc")


async def rate_limit_middleware(request: Request, call_next):
    if RATE_LIMIT_ENABLED and not request.url.path.startswith(_EXEMPT_PREFIXES):
        client = request.headers.get("X-API-Key") or (
            request.client.host if request.client else "anon"
        )
        if not _window.allow(client, RATE_LIMIT_RPM, RATE_LIMIT_WINDOW_S):
            logger.warning("rate_limited client=%s path=%s", client[:12], request.url.path)
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_S)},
            )
    return await call_next(request)


# --- upload guard -----------------------------------------------------------


def enforce_upload_size(content_length: int | None, nbytes: int | None = None) -> None:
    """Raise 413 when an upload exceeds ``MAX_UPLOAD_BYTES``.

    Checks the declared Content-Length first (cheap, pre-read) and the actual
    byte count second (authoritative, post-read).
    """
    for n in (content_length, nbytes):
        if n is not None and n > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit",
            )
