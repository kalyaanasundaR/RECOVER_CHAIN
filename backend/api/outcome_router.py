"""Authoritative outcome events: ingestion, obligation lookup, and per-case timelines.

Every route here is additive to the Phase 1-4 API surface. Nothing here fabricates a
financial result: an outcome event is either a validated, matched, reconciled fact
(application/outcome_events.py) or it is stored UNMATCHED and visible, never guessed.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.access import require_admin, require_operator
from application.cause_observations import observation_dict, record_cause_observation
from application.ingestion import ValidationFailure
from application.outcome_events import (
    OutcomeEventInput,
    ingest_outcome_event,
    rematch_unmatched_outcomes,
)
from infrastructure.db import get_db
from infrastructure.orm import CaseModel, CauseObservationModel, OutcomeEventModel, PromiseModel
from infrastructure.repositories import SqlAlchemyAuditRecorder, SqlAlchemyCaseRepository

router = APIRouter(tags=["outcomes"])
# Phase 17: writes that establish evidence (outcomes, evidence sources) need ADMIN.
_AUTH = [Depends(require_admin)]


class OutcomeEventRequest(BaseModel):
    external_system: str
    external_event_id: str
    domain: str
    outcome_type: str
    customer_id: str
    reference_id: str | None = None
    customer_system: str | None = None
    amount: float | str | None = None
    currency: str | None = None
    occurred_at: str | datetime
    promised_date: str | datetime | None = None
    raw_payload: dict = {}
    # Phase 8A: where this outcome actually came from (application.outcome_provenance
    # .OutcomeProvenance). Omitted -> TEST, never assumed PRODUCTION.
    provenance: str | None = None
    # Phase 9: required for provenance=human_confirmed (must point outside
    # RecoverChain's own generated ids); required, and checked against the
    # evidence-source registry, for the per-integration provenances.
    evidence_reference: str | None = None
    source_id: str | None = None


class OutcomeBatchRequest(BaseModel):
    events: list[OutcomeEventRequest]


def _to_input(
    item: OutcomeEventRequest, source_ref: str | None = None, ingestion_run_id: str | None = None
) -> OutcomeEventInput:
    return OutcomeEventInput(
        external_system=item.external_system,
        external_event_id=item.external_event_id,
        domain=item.domain,
        outcome_type=item.outcome_type,
        customer_id=item.customer_id,
        reference_id=item.reference_id,
        customer_system=item.customer_system,
        amount=item.amount,
        currency=item.currency,
        occurred_at=item.occurred_at,
        promised_date=item.promised_date,
        raw_payload=item.raw_payload,
        source_type="API",
        source_ref=source_ref,
        ingestion_run_id=ingestion_run_id,
        provenance=item.provenance,
        evidence_reference=item.evidence_reference,
        source_id=item.source_id,
    )


@router.post("/outcomes", dependencies=_AUTH)
def post_outcome(req: OutcomeEventRequest, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)
    try:
        return ingest_outcome_event(db, repo, audit, _to_input(req))
    except ValidationFailure as e:
        raise HTTPException(
            status_code=422, detail={"reason": e.reason, "field": e.field_name, "detail": e.detail}
        ) from e


@router.post("/outcomes/batch", dependencies=_AUTH)
def post_outcomes_batch(req: OutcomeBatchRequest, db: Session = Depends(get_db)):
    """Each item is validated and applied independently - one malformed item is
    rejected on its own, exactly like POST /events/batch."""
    import uuid

    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)
    run_id = f"ocr_{uuid.uuid4().hex[:16]}"
    results = []
    accepted = rejected = 0
    for position, item in enumerate(req.events):
        try:
            result = ingest_outcome_event(db, repo, audit, _to_input(item, ingestion_run_id=run_id))
            accepted += 1
            results.append({"row": position, **result})
        except ValidationFailure as e:
            rejected += 1
            results.append(
                {
                    "row": position,
                    "status": "rejected",
                    "reason": e.reason,
                    "field": e.field_name,
                    "detail": e.detail,
                }
            )
    return {
        "ingestion_run_id": run_id,
        "submitted": len(req.events),
        "accepted": accepted,
        "rejected": rejected,
        "results": results,
    }


def _outcome_row_dict(row: OutcomeEventModel) -> dict:
    return {
        "outcome_event_id": row.outcome_event_id,
        "external_system": row.external_system,
        "external_event_id": row.external_event_id,
        "domain": row.domain,
        "kind": row.kind,
        "outcome_type": row.outcome_type,
        "customer_id": row.customer_id,
        "customer_system": row.customer_system,
        "reference_id": row.reference_id,
        "amount": str(row.amount) if row.amount is not None else None,
        "currency": row.currency,
        "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
        "ingested_at": row.ingested_at.isoformat() if row.ingested_at else None,
        "match_status": row.match_status,
        "match_reason": row.match_reason,
        "case_id": row.case_id,
        "applied_amount": str(row.applied_amount) if row.applied_amount is not None else None,
        "excess_amount": str(row.excess_amount) if row.excess_amount is not None else None,
        "effect": row.effect,
        "provenance": row.provenance,
        "evidence_reference": row.evidence_reference,
        "source_id": row.source_id,
    }


@router.get("/outcomes", dependencies=[Depends(require_operator)])
def list_outcomes(
    match_status: str | None = Query(None),
    domain: str | None = Query(None),
    customer_id: str | None = Query(None),
    case_id: str | None = Query(None),
    provenance: str | None = Query(None),
    limit: int = Query(100),
    db: Session = Depends(get_db),
):
    limit = min(max(1, limit), 500)
    q = db.query(OutcomeEventModel)
    if match_status:
        q = q.filter(OutcomeEventModel.match_status == match_status.upper())
    if domain:
        q = q.filter(OutcomeEventModel.domain == domain)
    if customer_id:
        q = q.filter(OutcomeEventModel.customer_id == customer_id)
    if case_id:
        q = q.filter(OutcomeEventModel.case_id == case_id)
    if provenance:
        q = q.filter(OutcomeEventModel.provenance == provenance.lower())
    rows = q.order_by(OutcomeEventModel.ingested_at.desc()).limit(limit).all()
    return {"total_count": q.count(), "items": [_outcome_row_dict(r) for r in rows]}


@router.post("/outcomes/rematch-unmatched", dependencies=_AUTH)
def rematch_unmatched(domain: str | None = Query(None), db: Session = Depends(get_db)):
    """Re-run matching for stored UNMATCHED outcome events (an obligation that did not
    exist yet, or a customer-identity link that has since been recorded). Explicit and
    auditable, never automatic - see application/outcome_events.rematch_unmatched_outcomes."""
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)
    return rematch_unmatched_outcomes(db, repo, audit, domain=domain)


@router.get("/obligations")
def get_obligation(
    domain: str = Query(...),
    customer_id: str = Query(...),
    reference_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """Every case ever opened for this obligation (domain + customer + reference), each
    with its own financial position, plus the outcome events and promises matched to
    any of them - a single obligation can have several case episodes over time (a new
    failure after an earlier one settled opens a new case for the same identity)."""
    repo = SqlAlchemyCaseRepository(db)
    cases = repo.get_cases_for_obligation(customer_id, domain, reference_id)
    if not cases:
        raise HTTPException(status_code=404, detail="No case found for this obligation")
    case_ids = [c.case_id for c in cases]
    outcomes = (
        db.query(OutcomeEventModel)
        .filter(OutcomeEventModel.case_id.in_(case_ids))
        .order_by(OutcomeEventModel.ingested_at)
        .all()
    )
    promises = (
        db.query(PromiseModel)
        .filter(PromiseModel.case_id.in_(case_ids))
        .order_by(PromiseModel.created_at)
        .all()
    )
    # Stored promise timestamps are naive UTC (SQLite drops the offset); compare like
    # with like.
    now = datetime.now(UTC).replace(tzinfo=None)
    case_by_id = {c.case_id: c for c in cases}

    def promise_dict(p: PromiseModel) -> dict:
        from application.outcome_events import derive_promise_status

        c = case_by_id.get(p.case_id)
        received = c.recovered_amount.amount if c and c.recovered_amount else 0
        return {
            "promise_id": p.promise_id,
            "case_id": p.case_id,
            "promised_amount": str(p.promised_amount),
            "currency": p.currency,
            "promised_date": p.promised_date.isoformat() if p.promised_date else None,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "broken_reported_at": p.broken_reported_at.isoformat()
            if p.broken_reported_at
            else None,
            "status": derive_promise_status(p, received, now),
        }

    return {
        "domain": domain,
        "customer_id": customer_id,
        "reference_id": reference_id,
        "cases": [
            {
                "case_id": c.case_id,
                "current_state": c.current_state.value
                if hasattr(c.current_state, "value")
                else str(c.current_state),
                "amount_at_risk": str(c.amount_at_risk.amount),
                "currency": c.amount_at_risk.currency,
                "recovered_amount": str(c.recovered_amount.amount) if c.recovered_amount else "0",
                "outstanding_amount": str(c.outstanding_amount.amount)
                if c.outstanding_amount is not None
                else str(c.amount_at_risk.amount),
                "cancelled_amount": str(c.cancelled_amount.amount) if c.cancelled_amount else "0",
            }
            for c in cases
        ],
        "outcome_events": [_outcome_row_dict(r) for r in outcomes],
        "promises": [promise_dict(p) for p in promises],
    }


@router.get("/cases/{case_id}/timeline")
def get_case_timeline(case_id: str, db: Session = Depends(get_db)):
    """Every recorded event for this case, chronological, from three append-only
    sources: the audit trail (every stage transition), the execution ledger (every
    dispatched action), and outcome events (every authoritative settlement /
    cancellation / promise matched to it). Nothing here is derived or fabricated - each
    entry is a row already written by the module responsible for it."""
    from infrastructure.orm import AuditModel, ExecutionAttemptModel

    if not db.query(CaseModel.case_id).filter(CaseModel.case_id == case_id).first():
        raise HTTPException(status_code=404, detail="Case not found")

    entries = []
    for a in db.query(AuditModel).filter(AuditModel.case_id == case_id).all():
        entries.append(
            {
                "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                "source": "audit",
                "kind": (a.evidence or {}).get("action", "transition"),
                "from_state": a.from_state,
                "to_state": a.to_state,
                "detail": a.evidence,
            }
        )
    for e in db.query(ExecutionAttemptModel).filter(ExecutionAttemptModel.case_id == case_id).all():
        entries.append(
            {
                "timestamp": e.created_at.isoformat() if e.created_at else None,
                "source": "execution_attempt",
                "kind": "action_execution",
                "detail": {
                    "action_type": e.action_type,
                    "channel": e.channel,
                    "status": e.status,
                    "execution_id": e.execution_id,
                    "contact_permission": e.contact_permission,
                },
            }
        )
    for o in db.query(OutcomeEventModel).filter(OutcomeEventModel.case_id == case_id).all():
        entries.append(
            {
                "timestamp": o.occurred_at.isoformat() if o.occurred_at else None,
                "source": "outcome_event",
                "kind": o.kind,
                "detail": _outcome_row_dict(o),
            }
        )
    entries.sort(key=lambda e: e["timestamp"] or "")
    return {"case_id": case_id, "entries": entries}


# ---------------------------------------------------------------------------------
# Phase 8A — independently observed root-cause evidence, distinct from the
# deterministic diagnosis engine's rule-derived cause_category (see application/
# cause_observations.py and application/data_readiness.py).
# ---------------------------------------------------------------------------------


class CauseObservationRequest(BaseModel):
    observed_cause_category: str
    source: str
    note: str | None = None
    recorded_by: str | None = None
    evidence_reference: str | None = None
    confidence: float | None = None


@router.post("/cases/{case_id}/observed-cause", dependencies=_AUTH)
def post_observed_cause(case_id: str, req: CauseObservationRequest, db: Session = Depends(get_db)):
    case = db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    try:
        row = record_cause_observation(
            db,
            case_id=case_id,
            observed_cause_category=req.observed_cause_category,
            source=req.source,
            note=req.note,
            recorded_by=req.recorded_by,
            evidence_reference=req.evidence_reference,
            confidence=req.confidence,
        )
    except ValidationFailure as e:
        raise HTTPException(
            status_code=422, detail={"reason": e.reason, "field": e.field_name, "detail": e.detail}
        ) from e
    db.commit()
    return observation_dict(row)


@router.get("/cases/{case_id}/observed-cause")
def get_observed_causes(case_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(CauseObservationModel)
        .filter(CauseObservationModel.case_id == case_id)
        .order_by(CauseObservationModel.recorded_at)
        .all()
    )
    return {"case_id": case_id, "observations": [observation_dict(r) for r in rows]}


# ---------------------------------------------------------------------------------
# Phase 9 — external evidence source registry (application/evidence_sources.py).
# Registration alone never grants trust; see that module for the real (never
# decorative) verification check.
# ---------------------------------------------------------------------------------


class SourceRegistrationRequest(BaseModel):
    source_type: str
    provider: str
    domain: str
    schema_version: str
    registered_by: str | None = None


@router.post("/system/evidence-sources", dependencies=_AUTH)
def post_evidence_source(req: SourceRegistrationRequest, db: Session = Depends(get_db)):
    from application.evidence_sources import SourceRegistrationError, register_source, source_dict

    try:
        row = register_source(
            db,
            source_type=req.source_type,
            provider=req.provider,
            domain=req.domain,
            schema_version=req.schema_version,
            registered_by=req.registered_by,
        )
    except SourceRegistrationError as e:
        raise HTTPException(status_code=422, detail={"field": e.field, "detail": e.detail}) from e
    db.commit()
    return source_dict(row)


@router.get("/system/evidence-sources", dependencies=[Depends(require_operator)])
def list_evidence_sources(source_type: str | None = Query(None), db: Session = Depends(get_db)):
    from application.evidence_sources import source_dict
    from infrastructure.orm import EvidenceSourceModel

    q = db.query(EvidenceSourceModel)
    if source_type:
        q = q.filter(EvidenceSourceModel.source_type == source_type)
    rows = q.order_by(EvidenceSourceModel.created_at).all()
    return {"total_count": len(rows), "items": [source_dict(r) for r in rows]}


@router.post("/system/evidence-sources/{source_id}/verify", dependencies=_AUTH)
def post_verify_evidence_source(source_id: str, db: Session = Depends(get_db)):
    from application.evidence_sources import SourceRegistrationError, source_dict, verify_source

    try:
        row = verify_source(db, source_id)
    except SourceRegistrationError as e:
        raise HTTPException(status_code=404, detail={"field": e.field, "detail": e.detail}) from e
    db.commit()
    return source_dict(row)


class SourceEnableRequest(BaseModel):
    enabled: bool


@router.post("/system/evidence-sources/{source_id}/enabled", dependencies=_AUTH)
def post_set_evidence_source_enabled(
    source_id: str, req: SourceEnableRequest, db: Session = Depends(get_db)
):
    from application.evidence_sources import SourceRegistrationError, set_enabled, source_dict

    try:
        row = set_enabled(db, source_id, req.enabled)
    except SourceRegistrationError as e:
        raise HTTPException(status_code=422, detail={"field": e.field, "detail": e.detail}) from e
    db.commit()
    return source_dict(row)
