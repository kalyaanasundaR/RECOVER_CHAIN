from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from api.principals import resolve_principal

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Guard for mutating / side-effectful endpoints: any ACTIVE principal (Phase 17:
    api/principals.py - the legacy keys or a registry entry). Endpoints that need a
    specific role use require_reviewer / require_admin in api/access.py.

    Dev default key is 'test-api-key' (override with env API_KEY). Tests bypass
    this via a FastAPI dependency override; see tests/conftest.py.
    """
    principal = resolve_principal(api_key)
    if principal is None or not principal.can_act:  # OPERATOR credentials are read-only
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key
