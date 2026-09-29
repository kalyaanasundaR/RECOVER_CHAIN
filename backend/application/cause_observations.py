"""Phase 8A/8B — recording an independently observed root-cause fact.

Distinct from RootCauseDiagnosis (application/diagnosis_engine.py), which is the
deterministic rule's OUTPUT, not an observation. See infrastructure/orm.py::
CauseObservationModel, docs/history/PHASE_8A_DATA_READINESS_REPORT.md Step 8, and
docs/history/PHASE_8B_INDEPENDENT_EVIDENCE_REPORT.md Step 6.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from application.ingestion import ValidationFailure, normalize_identifier
from application.outcome_events import _looks_like_recoverchain_id
from domain.models import RootCauseCategory
from infrastructure.orm import CauseObservationModel

# The full closed vocabulary a stored observation's `source` may hold - genuinely
# independent of the deterministic diagnosis rule, so a caller can never claim "rule"
# or "diagnosis_engine" as if it were an observation.
ALL_SOURCES = frozenset(
    {
        "human_confirmed",
        "payment_gateway_response",
        "checkout_system_event",
        "subscription_billing_event",
        "invoice_payment_history",
        "external_source_simulated",  # Phase 8B: clearly-labelled simulated evidence
    }
)

# Phase 8B: what this endpoint can actually accept a CALLER'S CLAIM of, mirroring
# application/outcome_provenance.ACCEPTED_VIA_SHARED_API_KEY. The per-integration
# sources (payment_gateway_response, checkout_system_event, subscription_billing_event,
# invoice_payment_history) are real vocabulary values, reserved for when an actual
# connector with its own authentication exists - not reachable through the shared API
# key today, exactly like OutcomeProvenance.PAYMENT_PROVIDER etc.
ACCEPTED_VIA_SHARED_API_KEY = frozenset({"human_confirmed", "external_source_simulated"})

# Backward-compatible alias (Phase 8A name). Kept because Phase 8A's tests and any
# external caller may still reference it; now equal to the accepted set, not the full
# vocabulary, since it always meant "what may actually be recorded."
INDEPENDENT_SOURCES = ACCEPTED_VIA_SHARED_API_KEY


def record_cause_observation(
    db,
    *,
    case_id: str,
    observed_cause_category: str,
    source: str,
    note: str | None = None,
    recorded_by: str | None = None,
    evidence_reference: str | None = None,
    confidence: float | None = None,
) -> CauseObservationModel:
    case_id = normalize_identifier(case_id, "case_id")
    try:
        category = RootCauseCategory(observed_cause_category)
    except ValueError:
        raise ValidationFailure(
            "invalid_value",
            "observed_cause_category",
            f"{observed_cause_category!r} is not a known root-cause category",
        ) from None
    source = (source or "").strip().lower()
    if source not in ALL_SOURCES:
        raise ValidationFailure(
            "invalid_value",
            "source",
            f"{source!r} is not a recognised independent evidence source (must be "
            f"one of {sorted(ALL_SOURCES)})",
        )
    if source not in ACCEPTED_VIA_SHARED_API_KEY:
        raise ValidationFailure(
            "invalid_value",
            "source",
            f"{source!r} requires a stronger authentication mechanism (a registered "
            "source or dedicated integration credential) that does not exist in this "
            "deployment - it cannot be accepted on a bare claim via the shared API key",
        )
    evidence_reference = (evidence_reference or "").strip() or None
    if source == "human_confirmed":
        if not evidence_reference:
            raise ValidationFailure(
                "invalid_value",
                "evidence_reference",
                "human_confirmed requires evidence_reference - a pointer to something "
                "outside RecoverChain (a ticket id, a call-log reference, an email "
                "thread) - so a RecoverChain-generated outcome can never become "
                "'confirmed' evidence merely by being relabelled",
            )
        if _looks_like_recoverchain_id(evidence_reference):
            raise ValidationFailure(
                "invalid_value",
                "evidence_reference",
                f"{evidence_reference!r} looks like a RecoverChain-generated id, not "
                "external evidence - human confirmation must point outside the system "
                "it is confirming",
            )
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        raise ValidationFailure("invalid_value", "confidence", "must be between 0.0 and 1.0")
    row = CauseObservationModel(
        observation_id=f"cobs_{uuid.uuid4().hex[:12]}",
        case_id=case_id,
        observed_cause_category=category.value,
        source=source,
        note=(note or "").strip() or None,
        recorded_by=recorded_by,
        evidence_reference=evidence_reference,
        confidence=confidence,
        recorded_at=datetime.now(UTC).replace(tzinfo=None),
    )
    db.add(row)
    return row


def observation_dict(row: CauseObservationModel) -> dict:
    return {
        "observation_id": row.observation_id,
        "case_id": row.case_id,
        "observed_cause_category": row.observed_cause_category,
        "source": row.source,
        "note": row.note,
        "recorded_by": row.recorded_by,
        "evidence_reference": row.evidence_reference,
        "confidence": row.confidence,
        "recorded_at": row.recorded_at.isoformat() if row.recorded_at else None,
    }
