"""Phase 17 - the minimal identity abstraction.

RecoverChain is a single-tenant, controlled development/demo application with no user
accounts. What Phase 17 adds is the smallest thing approval evidence needs: a
*principal* (who acted), a *role* (what they may do), how they authenticated, and
whether they are active.

Two ways a credential becomes a principal (both use the existing `X-API-Key` header):

1. A registry in the `RECOVERCHAIN_PRINCIPALS` env var (JSON):
       {"<key>": {"principal_id": "alice", "role": "REVIEWER", "active": true}, ...}
   The principal id is whatever the deployer configured - RecoverChain does not verify
   that a human is behind it.
2. The two legacy keys. Their principal id is a *credential fingerprint*
   (`key:<first 10 hex of sha256(key)>`), i.e. "the holder of that key" - it is
   deliberately NOT presented as a person. Roles:
       OPERATOR_API_KEY            -> OPERATOR  (read observability)
       API_KEY, OPERATOR key set   -> USER      (run the workflow; no approval, no outcomes)
       API_KEY, OPERATOR key unset -> ADMIN     (dev default: the one shared key does everything)

Roles (bounded, each justified by an existing product capability):
    USER      run the workflow (ingest / advance / execute / stop / verify ...)
    OPERATOR  read observability (traces, audit, executions, model + dataset metadata)
    REVIEWER  approve / reject escalated actions; also reads observability
    ADMIN     everything, including writes that establish evidence (outcomes, evidence sources)

Historical records have no principal; they stay `UNKNOWN` (nothing is back-filled).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ROLES = ("USER", "OPERATOR", "REVIEWER", "ADMIN")
_OBSERVE = frozenset({"OPERATOR", "REVIEWER", "ADMIN"})
_ACT = frozenset({"USER", "REVIEWER", "ADMIN"})
_APPROVE = frozenset({"REVIEWER", "ADMIN"})
_ADMIN = frozenset({"ADMIN"})
UNKNOWN_PRINCIPAL = {
    "principal_id": "UNKNOWN",
    "role": "UNKNOWN",
    "authentication_method": "UNKNOWN",
}


@dataclass(frozen=True)
class Principal:
    principal_id: str
    role: str
    authentication_method: str = "api_key"
    active: bool = True

    @property
    def can_act(self) -> bool:
        """May run workflow actions (ingest/advance/execute/...). OPERATOR is read-only."""
        return self.role in _ACT

    @property
    def can_observe(self) -> bool:
        return self.role in _OBSERVE

    @property
    def can_approve(self) -> bool:
        return self.role in _APPROVE

    @property
    def is_admin(self) -> bool:
        return self.role in _ADMIN

    def as_evidence(self) -> dict:
        """What is written into audit evidence - never the credential itself."""
        return {
            "principal_id": self.principal_id,
            "role": self.role,
            "authentication_method": self.authentication_method,
        }


def fingerprint(key: str) -> str:
    return "key:" + hashlib.sha256(key.encode()).hexdigest()[:10]


def _eq(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and hmac.compare_digest(a.encode(), b.encode())


def _registry() -> dict:
    raw = os.getenv("RECOVERCHAIN_PRINCIPALS")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except ValueError:
        logger.warning("RECOVERCHAIN_PRINCIPALS is not valid JSON - ignored (failing closed)")
        return {}


def lookup(api_key: str | None) -> tuple[Principal | None, str]:
    """(principal, reason). reason is 'ok' | 'missing' | 'unknown' | 'inactive'."""
    if not api_key:
        return None, "missing"
    for key, spec in _registry().items():
        if _eq(api_key, key) and isinstance(spec, dict):
            role = str(spec.get("role", "")).upper()
            if role not in ROLES:
                return None, "unknown"  # a misconfigured role grants nothing
            p = Principal(
                principal_id=str(spec.get("principal_id") or fingerprint(key)),
                role=role,
                active=bool(spec.get("active", True)),
            )
            return (p, "ok") if p.active else (None, "inactive")
    operator_key = os.getenv("OPERATOR_API_KEY")
    user_key = os.getenv("API_KEY", "test-api-key")
    if operator_key and _eq(api_key, operator_key):
        return Principal(fingerprint(api_key), "OPERATOR"), "ok"
    if _eq(api_key, user_key):
        return Principal(fingerprint(api_key), "USER" if operator_key else "ADMIN"), "ok"
    return None, "unknown"


def resolve_principal(api_key: str | None) -> Principal | None:
    return lookup(api_key)[0]
