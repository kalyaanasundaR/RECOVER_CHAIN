"""Customer contact permission, contact history, and event identity reconciliation.

Every state change here is authenticated and audited. Nothing a browser sends is
trusted for policy: a permission takes effect only once it is stored, and only the
Policy Engine reads it (through SqlAlchemyCaseRepository.get_policy_context).
"""

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.access import require_operator
from api.auth import verify_api_key
from infrastructure.db import get_db

router = APIRouter(tags=["domain"])
_AUTH = [Depends(verify_api_key)]


class ContactPermissionRequest(BaseModel):
    channel: str  # communication | payment | all
    status: str  # ALLOWED | OPTED_OUT | SUPPRESSED
    source: str  # who or what recorded it, e.g. "customer_request", "compliance_hold"
    reason: str | None = None
    effective_at: datetime | None = None  # default: now


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _permission_dict(row) -> dict:
    return {
        "id": row.id,
        "customer_id": row.customer_id,
        "channel": row.channel,
        "status": row.status,
        "source": row.source,
        "reason": row.reason,
        "effective_at": _aware(row.effective_at).isoformat(),
        "recorded_at": _aware(row.recorded_at).isoformat() if row.recorded_at else None,
        "source_record_id": row.source_record_id,
        "ingestion_run_id": row.ingestion_run_id,
    }


def _permission_state(db: Session, customer_id: str) -> dict:
    from application.contact_permissions import resolve_all
    from application.policy_engine import MerchantPolicy
    from infrastructure.repositories import SqlAlchemyContactPermissionRepository

    rows = SqlAlchemyContactPermissionRepository(db).list_for_customer(customer_id)
    policy = MerchantPolicy()
    return {
        "customer_id": customer_id,
        "effective": {ch: r.to_dict() for ch, r in resolve_all(rows).items()},
        # What the Policy Engine does when a channel resolves to UNKNOWN (no record).
        "unknown_consent_behaviour": {
            "communication": policy.unknown_consent_communication,
            "payment": policy.unknown_consent_payment,
        },
        "records": [_permission_dict(r) for r in rows],
    }


@router.post("/customers/{customer_id}/contact-permissions", dependencies=_AUTH)
def record_contact_permission(
    customer_id: str, req: ContactPermissionRequest, db: Session = Depends(get_db)
):
    """Append a contact-permission record. Records are never edited or deleted; a
    change of mind is a new record."""
    from infrastructure.repositories import (
        SqlAlchemyAuditRecorder,
        SqlAlchemyContactPermissionRepository,
    )

    customer_id = customer_id.strip()
    if not customer_id:
        raise HTTPException(status_code=422, detail="customer_id is required")
    if not req.source or not req.source.strip():
        raise HTTPException(status_code=422, detail="source is required")
    try:
        row, is_new = SqlAlchemyContactPermissionRepository(db).record(
            customer_id=customer_id,
            channel=req.channel.strip().lower(),
            status=req.status.strip().upper(),
            source=req.source.strip(),
            reason=(req.reason or "").strip() or None,
            effective_at=req.effective_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if is_new:
        SqlAlchemyAuditRecorder(db).log_transition(
            case_id="SYSTEM",
            from_state="N/A",
            to_state="N/A",
            evidence={"action": "contact_permission_recorded", **_permission_dict(row)},
        )
    db.commit()
    return {
        "record": _permission_dict(row),
        "duplicate": not is_new,
        **_permission_state(db, customer_id),
    }


class ConsentImportItem(BaseModel):
    customer_id: str
    channel: str
    status: str
    source: str
    reason: str | None = None
    effective_at: datetime | None = None
    source_record_id: str | None = None


class ConsentImportRequest(BaseModel):
    records: list[ConsentImportItem]


@router.post("/contact-permissions/import", dependencies=_AUTH)
def import_contact_permissions(req: ConsentImportRequest, db: Session = Depends(get_db)):
    """Bulk consent import. Reuses the exact validation the single-record route applies
    (application.contact_permissions.validate_permission plus the same non-blank
    checks); an invalid item is REJECTED WITH A REASON, the rest of the batch still
    imports. Missing consent is never assumed ALLOWED: an item with no `status` at all
    is rejected, not defaulted (see application/contact_permissions.py for what a
    resolved UNKNOWN then means to the Policy Engine)."""
    from application.ingestion import IngestionRunReport, RejectReason, SourceType
    from infrastructure.repositories import (
        SqlAlchemyAuditRecorder,
        SqlAlchemyContactPermissionRepository,
    )

    repo = SqlAlchemyContactPermissionRepository(db)
    audit = SqlAlchemyAuditRecorder(db)
    report = IngestionRunReport(source_type=SourceType.API_BATCH, source_name="consent_import")
    imported = duplicates = 0
    results = []
    for position, item in enumerate(req.records):
        report.rows_received += 1
        customer_id = (item.customer_id or "").strip()
        source = (item.source or "").strip()
        try:
            if not customer_id:
                raise ValueError("customer_id is required")
            if not source:
                raise ValueError("source is required")
            row, is_new = repo.record(
                customer_id=customer_id,
                channel=(item.channel or "").strip().lower(),
                status=(item.status or "").strip().upper(),
                source=source,
                reason=(item.reason or "").strip() or None,
                effective_at=item.effective_at,
                source_record_id=item.source_record_id,
                ingestion_run_id=report.run_id,
            )
        except ValueError as e:
            report.reject(RejectReason.INVALID_VALUE, row=position, detail=str(e))
            results.append({"row": position, "status": "rejected", "detail": str(e)})
            continue
        if is_new:
            imported += 1
            report.rows_accepted += 1
            audit.log_transition(
                case_id="SYSTEM",
                from_state="N/A",
                to_state="N/A",
                evidence={"action": "contact_permission_recorded", **_permission_dict(row)},
            )
        else:
            duplicates += 1
            report.rows_duplicate += 1
        results.append(
            {"row": position, "status": "imported" if is_new else "duplicate", "record_id": row.id}
        )
    db.add(report.to_model("COMPLETED"))
    db.commit()
    return {
        "ingestion_run_id": report.run_id,
        "submitted": len(req.records),
        "imported": imported,
        "duplicates": duplicates,
        "rejected": report.rows_rejected,
        "results": results,
    }


@router.get(
    "/customers/{customer_id}/contact-permissions", dependencies=[Depends(require_operator)]
)
def get_contact_permissions(customer_id: str, db: Session = Depends(get_db)):
    return _permission_state(db, customer_id)


@router.get("/customers/{customer_id}/contact-history", dependencies=[Depends(require_operator)])
def get_contact_history(
    customer_id: str, window_hours: int = 24, limit: int = 200, db: Session = Depends(get_db)
):
    """Contact history, derived from the execution_attempts ledger (the source of truth;
    nothing is duplicated). For each attempt: which channel, when, what action for which
    obligation, under which policy decision and why, the permission the Policy Engine
    evaluated at authorisation, and the permission in force at that moment
    reconstructed from the append-only permission records."""
    from application.contact_permissions import resolve_permission
    from application.policy_engine import MerchantPolicy
    from infrastructure.orm import AuditModel, CaseModel, ExecutionAttemptModel
    from infrastructure.repositories import SqlAlchemyContactPermissionRepository

    window_hours = min(max(1, window_hours), 24 * 90)
    limit = min(max(1, limit), 1000)
    since = datetime.now(UTC) - timedelta(hours=window_hours)

    rows = (
        db.query(ExecutionAttemptModel, CaseModel.risk_category, CaseModel.reference_id)
        .join(CaseModel, CaseModel.case_id == ExecutionAttemptModel.case_id)
        .filter(CaseModel.customer_id == customer_id)
        .order_by(ExecutionAttemptModel.created_at.desc())
        .all()
    )
    in_window = [r for r, _, _ in rows if r.created_at and _aware(r.created_at) >= since]
    shown = rows[:limit]

    reasons: dict[str, str] = {}
    case_ids = list({r.case_id for r, _, _ in shown})
    if case_ids:
        for a in db.query(AuditModel).filter(AuditModel.case_id.in_(case_ids)).all():
            ev = a.evidence or {}
            if ev.get("action") == "policy_evaluation" and ev.get("decision_id"):
                reasons[ev["decision_id"]] = ev.get("reason")

    permissions = SqlAlchemyContactPermissionRepository(db).list_for_customer(customer_id)
    attempts = []
    for r, category, reference in shown:
        channel = r.channel or "communication"
        applicable = channel in ("communication", "payment") and r.created_at
        # Two deliberately different views, both truthful:
        #  - permission_known_at_attempt: bitemporal - only records that had actually
        #    been recorded BY the time of this attempt count. A later-recorded
        #    (backdated) record can never change this: history is not rewritten.
        #  - permission_in_force_at_attempt: what today's full record set says applied
        #    at that moment, including anything recorded since - useful for "what do we
        #    now believe was true then", but NOT what the Policy Engine could have known
        #    at the time (permission_at_authorisation, from the ledger itself, is that).
        in_force = (
            resolve_permission(permissions, channel, at=_aware(r.created_at)).status
            if applicable
            else None
        )
        known_at_attempt = (
            resolve_permission(
                permissions, channel, at=_aware(r.created_at), known_at=_aware(r.created_at)
            ).status
            if applicable
            else None
        )
        attempts.append(
            {
                "attempt_id": r.id,
                "case_id": r.case_id,
                "risk_category": category.value if hasattr(category, "value") else category,
                "reference_id": reference,
                "channel": channel,
                "action_type": r.action_type,
                "status": r.status,
                "created_at": _aware(r.created_at).isoformat() if r.created_at else None,
                "execution_id": r.execution_id,
                "agent_type": r.agent_type,
                "policy_decision_id": r.policy_decision_id,
                "policy_reason": reasons.get(r.policy_decision_id),
                "permission_at_authorisation": r.contact_permission,
                "permission_in_force_at_attempt": in_force,
                "permission_known_at_attempt": known_at_attempt,
            }
        )

    return {
        "customer_id": customer_id,
        "source": "execution_attempts",
        "window_hours": window_hours,
        "communication_in_window": sum(1 for r in in_window if r.channel == "communication"),
        "payment_attempts_in_window": sum(1 for r in in_window if r.channel == "payment"),
        "communication_limit_24h": MerchantPolicy().communication_max_messages_24h,
        "total_attempts": len(rows),
        "attempts": attempts,
    }


@router.post("/events/identity-reconciliation", dependencies=_AUTH)
def run_identity_reconciliation(
    apply: bool = False,
    dataset_id: list[str] | None = Query(None),
    db: Session = Depends(get_db),
):
    """Re-identify events stored under an older identity scheme (dry run by default;
    ``dataset_id`` may be repeated to limit the run). Adds identity aliases only; never
    rewrites, merges or deletes. See application/identity_reconciliation.py for the
    method and its limitations."""
    from application.identity_reconciliation import reconcile_identities

    return reconcile_identities(db, apply=apply, dataset_ids=dataset_id)


class CustomerIdentityLinkRequest(BaseModel):
    system: str
    external_customer_id: str
    customer_id: str
    source: str


@router.post("/customers/identity-links", dependencies=_AUTH)
def create_customer_identity_link(req: CustomerIdentityLinkRequest, db: Session = Depends(get_db)):
    """A deterministic (system, external id) -> customer_id mapping, used only to
    resolve `customer_system` on an outcome or consent-import record
    (application/outcome_events.py). Written only here, never inferred - no name or
    email matching. A link is immutable: to correct a mistake, delete it first
    (DELETE /customers/identity-links/{system}/{external_customer_id})."""
    from infrastructure.orm import CustomerIdentityLinkModel
    from infrastructure.repositories import SqlAlchemyAuditRecorder

    system = req.system.strip()
    external_customer_id = req.external_customer_id.strip()
    customer_id = req.customer_id.strip()
    source = req.source.strip()
    if not (system and external_customer_id and customer_id and source):
        raise HTTPException(
            status_code=422,
            detail="system, external_customer_id, customer_id and source are all required",
        )
    existing = (
        db.query(CustomerIdentityLinkModel)
        .filter(
            CustomerIdentityLinkModel.system == system,
            CustomerIdentityLinkModel.external_customer_id == external_customer_id,
        )
        .first()
    )
    if existing is not None:
        if existing.customer_id != customer_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{system}:{external_customer_id} is already linked to "
                    f"{existing.customer_id}. Delete that link first to relink it - a link "
                    "is never silently overwritten."
                ),
            )
        return _identity_link_dict(existing)
    row = CustomerIdentityLinkModel(
        id=f"cid_{uuid.uuid4().hex[:12]}",
        system=system,
        external_customer_id=external_customer_id,
        customer_id=customer_id,
        source=source,
    )
    db.add(row)
    SqlAlchemyAuditRecorder(db).log_transition(
        case_id="SYSTEM",
        from_state="N/A",
        to_state="N/A",
        evidence={"action": "customer_identity_linked", **_identity_link_dict(row)},
    )
    db.commit()
    return _identity_link_dict(row)


@router.get("/customers/identity-links", dependencies=[Depends(require_operator)])
def list_customer_identity_links(
    customer_id: str | None = None, system: str | None = None, db: Session = Depends(get_db)
):
    from infrastructure.orm import CustomerIdentityLinkModel

    q = db.query(CustomerIdentityLinkModel)
    if customer_id:
        q = q.filter(CustomerIdentityLinkModel.customer_id == customer_id)
    if system:
        q = q.filter(CustomerIdentityLinkModel.system == system)
    return {
        "links": [
            _identity_link_dict(r) for r in q.order_by(CustomerIdentityLinkModel.created_at).all()
        ]
    }


@router.delete("/customers/identity-links/{system}/{external_customer_id}", dependencies=_AUTH)
def delete_customer_identity_link(
    system: str, external_customer_id: str, db: Session = Depends(get_db)
):
    from infrastructure.orm import CustomerIdentityLinkModel
    from infrastructure.repositories import SqlAlchemyAuditRecorder

    row = (
        db.query(CustomerIdentityLinkModel)
        .filter(
            CustomerIdentityLinkModel.system == system,
            CustomerIdentityLinkModel.external_customer_id == external_customer_id,
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="No such identity link")
    info = _identity_link_dict(row)
    db.delete(row)
    SqlAlchemyAuditRecorder(db).log_transition(
        case_id="SYSTEM",
        from_state="N/A",
        to_state="N/A",
        evidence={"action": "customer_identity_link_deleted", **info},
    )
    db.commit()
    return {"status": "deleted", **info}


def _identity_link_dict(row) -> dict:
    return {
        "id": row.id,
        "system": row.system,
        "external_customer_id": row.external_customer_id,
        "customer_id": row.customer_id,
        "source": row.source,
        "created_at": _aware(row.created_at).isoformat() if row.created_at else None,
    }
