"""Phase 9 — the external evidence source registry (Step 2/3).

Registration is deliberately NOT trust. A source starts NOT_CONFIGURED and stays
there until `check_integration_status()` finds real credential material for it -
which, in this environment, it never will, because no real payment/invoicing/
subscription/checkout provider credentials exist here (see the module-level
constant CREDENTIAL_ENV_VARS below: every one of these environment variables is
genuinely unset in this deployment). This module does not fabricate a credential
check to make a source appear VERIFIED - it reports NOT_CONFIGURED honestly.

    register_source()   -> a source_id, integration_status=NOT_CONFIGURED, enabled=False
    verify_source()     -> re-runs the real check; only sets VERIFIED if it actually
                            finds credential material; never on request alone
    enable_source()     -> can only enable a VERIFIED source (never a NOT_CONFIGURED one)

Only an ENABLED, VERIFIED source of the matching type may back a per-integration
outcome provenance (payment_provider, invoice_system, subscription_system,
checkout_system) or `production` itself - see application/outcome_events.py's
provenance gate, which queries this registry rather than trusting a bare claim.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from infrastructure.orm import EvidenceSourceModel


class IntegrationStatus:
    NOT_CONFIGURED = "NOT_CONFIGURED"
    VERIFIED = "VERIFIED"


# One real (not simulated) environment-variable name per source type - the actual
# credential this deployment would need to genuinely authenticate that kind of
# provider. Every one of these is unset here; that is the honest, current state of
# this environment, checked live, not asserted.
CREDENTIAL_ENV_VARS: dict[str, str] = {
    "payment_provider": "RECOVERCHAIN_PAYMENT_PROVIDER_API_KEY",
    "invoice_system": "RECOVERCHAIN_INVOICE_SYSTEM_API_KEY",
    "subscription_system": "RECOVERCHAIN_SUBSCRIPTION_SYSTEM_API_KEY",
    "checkout_system": "RECOVERCHAIN_CHECKOUT_SYSTEM_API_KEY",
    "production": "RECOVERCHAIN_PRODUCTION_INTEGRATION_TOKEN",
}


def check_integration_status(source_type: str) -> str:
    """The ONLY function that may return VERIFIED. Looks for real credential
    material via the environment - never a database flag a caller can set, never a
    request parameter, never "trust the caller." An unset/empty env var (the actual
    state of every one of these in this deployment) means NOT_CONFIGURED - a real,
    honestly-reported answer, not a placeholder."""
    env_var = CREDENTIAL_ENV_VARS.get(source_type)
    if env_var and os.getenv(env_var, "").strip():
        return IntegrationStatus.VERIFIED
    return IntegrationStatus.NOT_CONFIGURED


@dataclass
class SourceRegistrationError(ValueError):
    field: str
    detail: str

    def __str__(self) -> str:
        return f"{self.field}: {self.detail}"


def register_source(
    db: Session,
    *,
    source_type: str,
    provider: str,
    domain: str,
    schema_version: str,
    registered_by: str | None = None,
) -> EvidenceSourceModel:
    if source_type not in CREDENTIAL_ENV_VARS:
        raise SourceRegistrationError(
            "source_type", f"{source_type!r} is not a recognised source type"
        )
    if not provider or not provider.strip():
        raise SourceRegistrationError("provider", "is required")
    row = EvidenceSourceModel(
        source_id=f"src_{uuid.uuid4().hex[:12]}",
        source_type=source_type,
        provider=provider.strip(),
        domain=domain,
        integration_status=check_integration_status(source_type),
        schema_version=schema_version,
        enabled=False,  # registration never auto-enables - see enable_source()
        created_at=datetime.now(UTC).replace(tzinfo=None),
        last_verified_at=None,
        registered_by=registered_by,
    )
    db.add(row)
    return row


def verify_source(db: Session, source_id: str) -> EvidenceSourceModel:
    """Re-runs the real credential check. Updates last_verified_at regardless of
    outcome (a verification attempt happened, even if it found nothing) but only
    ever sets integration_status to what check_integration_status() actually says."""
    row = db.query(EvidenceSourceModel).filter(EvidenceSourceModel.source_id == source_id).first()
    if row is None:
        raise SourceRegistrationError("source_id", "not found")
    row.integration_status = check_integration_status(row.source_type)
    row.last_verified_at = datetime.now(UTC).replace(tzinfo=None)
    if row.integration_status != IntegrationStatus.VERIFIED:
        row.enabled = False  # a source that stops being verified can't stay enabled
    return row


def set_enabled(db: Session, source_id: str, enabled: bool) -> EvidenceSourceModel:
    row = db.query(EvidenceSourceModel).filter(EvidenceSourceModel.source_id == source_id).first()
    if row is None:
        raise SourceRegistrationError("source_id", "not found")
    if enabled and row.integration_status != IntegrationStatus.VERIFIED:
        raise SourceRegistrationError(
            "enabled",
            f"cannot enable a source that is not VERIFIED (current status: "
            f"{row.integration_status})",
        )
    row.enabled = enabled
    return row


def is_usable(db: Session, source_id: str | None, *, source_type: str, domain: str) -> bool:
    """Whether this source_id may legitimately back an outcome event of this type and
    domain right now: it must exist, be enabled, be VERIFIED, and match both fields.
    None (no source_id given) is never usable for a source-gated provenance."""
    if not source_id:
        return False
    row = db.query(EvidenceSourceModel).filter(EvidenceSourceModel.source_id == source_id).first()
    if row is None:
        return False
    return (
        row.enabled
        and row.integration_status == IntegrationStatus.VERIFIED
        and row.source_type == source_type
        and row.domain == domain
    )


def source_dict(row: EvidenceSourceModel) -> dict:
    return {
        "source_id": row.source_id,
        "source_type": row.source_type,
        "provider": row.provider,
        "domain": row.domain,
        "integration_status": row.integration_status,
        "schema_version": row.schema_version,
        "enabled": row.enabled,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "last_verified_at": row.last_verified_at.isoformat() if row.last_verified_at else None,
        "registered_by": row.registered_by,
    }
