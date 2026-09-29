"""Access control: authentication (who) and authorization (may they), on the existing
`X-API-Key` header. Principals and roles live in api/principals.py.

  READ vs ACTION. ACTION endpoints use `verify_api_key` (api/auth.py: any active
  principal) or a stricter role dependency below. Read-only observability uses
  `require_operator`. Nothing behind a read dependency may mutate state (asserted by tests).

  Dependencies (all: 401 = no/unknown/inactive credential, 403 = valid but lacks the role):
    require_principal  any active principal (authenticated read)
    require_operator   OPERATOR / REVIEWER / ADMIN  (observability reads)
    require_reviewer   REVIEWER / ADMIN             (approve / reject escalations)
    require_admin      ADMIN                        (writes that establish evidence: outcomes,
                                                     evidence sources)

Authorization is enforced HERE, server-side, on every request - never by hiding a button.
Limitation: no per-user accounts or tenant ownership; see PHASE_17 report.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, Request, Security
from fastapi.responses import JSONResponse

from api.auth import API_KEY_HEADER
from api.principals import Principal, lookup, resolve_principal


def resolve_role(api_key: str | None) -> str | None:
    """Phase 16 compatibility: 'operator' (can observe), 'user' (can act only) or None."""
    p = resolve_principal(api_key)
    if p is None:
        return None
    return "operator" if p.can_observe else "user"


def _authenticate(api_key: str | None) -> Principal:
    principal, _reason = lookup(api_key)
    if principal is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return principal


def require_principal(api_key: str | None = Security(API_KEY_HEADER)) -> Principal:
    return _authenticate(api_key)


def require_operator(api_key: str | None = Security(API_KEY_HEADER)) -> Principal:
    p = _authenticate(api_key)
    if not p.can_observe:
        raise HTTPException(status_code=403, detail="Operator access required")
    return p


def require_reviewer(api_key: str | None = Security(API_KEY_HEADER)) -> Principal:
    p = _authenticate(api_key)
    if not p.can_approve:
        raise HTTPException(status_code=403, detail="Reviewer access required")
    return p


def require_admin(api_key: str | None = Security(API_KEY_HEADER)) -> Principal:
    p = _authenticate(api_key)
    if not p.is_admin:
        raise HTTPException(status_code=403, detail="Administrator access required")
    return p


# ---------------------------------------------------------------- production read switch

_OPEN_PATHS = ("/health", "/system/health", "/docs", "/openapi.json", "/redoc")


async def protect_reads_middleware(request: Request, call_next):
    """With RECOVERCHAIN_PROTECT_READS=1 every GET except health/docs needs an active
    principal. Off by default so the local demo (and the browser bundle) keep working;
    a deployment turns it on. Per-route role dependencies still apply on top."""
    if (
        os.getenv("RECOVERCHAIN_PROTECT_READS") == "1"
        and request.method == "GET"
        and not request.url.path.startswith(_OPEN_PATHS)
        and resolve_principal(request.headers.get("x-api-key")) is None
    ):
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return await call_next(request)
