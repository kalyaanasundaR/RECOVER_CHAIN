import json
import os
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_, text
from sqlalchemy.orm import Session

from api.access import require_operator, require_principal
from api.auth import verify_api_key
from infrastructure.dataset_orm import DatasetMetadataModel
from infrastructure.db import get_db
from infrastructure.orm import (
    AuditModel,
    CaseModel,
    EventModel,
    ExecutionAttemptModel,
    IdempotencyRecord,
)

router = APIRouter(prefix="/system", tags=["System Observability"])

# ----------------------------------------------------------------------
# Response Schemas (Read-Only)
# ----------------------------------------------------------------------


class HealthStatusResponse(BaseModel):
    status: str
    timestamp: str
    database: dict[str, Any]
    ml_subsystem: dict[str, Any]
    policy_engine: dict[str, Any]
    execution_engine: dict[str, Any]


class SystemSummaryResponse(BaseModel):
    datasets_count: int
    cases_count: int
    revenue_events_count: int
    models_count: int
    policy_decisions_count: int
    executions_count: int
    audit_records_count: int
    idempotency_records_count: int
    # Legacy figure, kept for backward compatibility: counts every case whose
    # `outcome` JSON blob is set, which includes pre-Phase-5 cases carrying a
    # SIMULATED (never authoritative) outcome. Do not read this as "cases with
    # real recovered money" - see authoritative_outcomes_count for that.
    outcomes_count: int
    # Phase 6/7 carry-forward fix: distinct cases with at least one real,
    # matched outcome_events row (application/outcome_events.py) - i.e. an
    # actual settlement/cancellation/promise event, never a simulated one.
    authoritative_outcomes_count: int
    availability: dict[str, str]


class DatasetItemResponse(BaseModel):
    dataset_id: str
    name: str
    filename: str
    file_type: str
    file_size_bytes: int
    status: str
    row_count: int | None = None
    column_count: int | None = None
    upload_timestamp: str | None = None


class DatasetListResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    items: list[DatasetItemResponse]


class DatasetDetailResponse(BaseModel):
    dataset_id: str
    name: str
    filename: str
    file_type: str
    file_size_bytes: int
    status: str
    row_count: int | None = None
    column_count: int | None = None
    upload_timestamp: str | None = None
    error_message: str | None = None
    # The profiler stores one dict PER COLUMN, i.e. a list. This was typed as a single
    # dict, so the endpoint returned HTTP 500 for every dataset that had been analyzed.
    profile_summary: list[dict[str, Any]] | dict[str, Any] | None = None
    semantic_mappings: list[dict[str, Any]] | None = None
    data_quality_report: dict[str, Any] | None = None
    training_suitability: dict[str, Any] | None = None
    leakage_detection: list[dict[str, Any]] | None = None
    associated_models_count: int = 0
    associated_cases_count: int = 0
    content_sha256: str | None = None  # identifies a byte-identical re-upload


class CaseSummaryItem(BaseModel):
    case_id: str
    customer_id: str
    risk_category: str
    reference_id: str | None = None
    amount_at_risk: float
    currency: str
    current_state: str
    risk_level: str | None = None
    cause_category: str | None = None
    recovery_probability: float | None = None
    prediction_status: str | None = None
    policy_decision: str | None = None
    execution_status: str | None = None
    outcome_status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class CaseListResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    items: list[CaseSummaryItem]


class CaseDetailSnapshotResponse(BaseModel):
    case_id: str
    customer_id: str
    risk_category: str
    reference_id: str | None = None
    amount_at_risk: float
    expected_recoverable_value: float | None = None
    currency: str
    current_state: str
    created_at: str | None = None
    updated_at: str | None = None
    data_events: list[dict[str, Any]]
    risk_assessment: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    ml_shadow_prediction: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None
    policy_decision: dict[str, Any] | None = None
    execution_record: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    audit_history: list[dict[str, Any]]
    # Domain semantics (Phase 4): the canonical domain, why the case's events carry it,
    # what identifies the obligation and what verification must confirm.
    domain: dict[str, Any] | None = None
    # The customer's contact permission per channel, resolved now from durable records.
    contact_permission: dict[str, Any] | None = None


class ModelRegistryItem(BaseModel):
    model_id: str
    dataset_id: str | None = None
    task: str
    algorithm: str
    model_version: str
    created_at: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    status: str


class ModelListResponse(BaseModel):
    total_count: int
    items: list[ModelRegistryItem]


class RevenueEventItem(BaseModel):
    event_id: str
    case_id: str
    customer_id: str
    external_system: str
    external_event_id: str
    reference_id: str | None = None
    risk_category: str
    amount: float
    currency: str
    timestamp: str


class EventListResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    items: list[RevenueEventItem]


class PolicyRecordItem(BaseModel):
    case_id: str
    customer_id: str
    decision_status: str
    policy_version: str | None = None
    reason: str | None = None
    candidate_action: str | None = None
    rules_evaluated_count: int = 0
    failed_rules_count: int = 0
    timestamp: str | None = None


class PolicyListResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    items: list[PolicyRecordItem]


class ExecutionRecordItem(BaseModel):
    case_id: str
    execution_id: str | None = None
    action_type: str | None = None
    agent_type: str | None = None
    execution_mode: str = "SIMULATED/SANDBOX"
    status: str
    adapter_used: str | None = None
    timestamp: str | None = None


class ExecutionListResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    items: list[ExecutionRecordItem]


class AuditRecordItem(BaseModel):
    audit_id: str
    case_id: str | None = None
    from_state: str | None = None
    to_state: str | None = None
    evidence: dict[str, Any] | None = None
    timestamp: str | None = None


class AuditListResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    items: list[AuditRecordItem]


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------


def _get_model_registry_dir() -> str:
    from infrastructure.storage import MODEL_REGISTRY_DIR_ENV

    configured = os.getenv(MODEL_REGISTRY_DIR_ENV)
    if configured:
        return os.path.abspath(configured)
    candidates = [
        os.path.join(os.getcwd(), "ml", "models", "registry"),
        os.path.join(os.getcwd(), "backend", "ml", "models", "registry"),
        os.path.join(os.path.dirname(__file__), "..", "ml", "models", "registry"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return candidates[0]


def _read_model_metadata_files() -> list[dict[str, Any]]:
    reg_dir = _get_model_registry_dir()
    if not os.path.exists(reg_dir):
        return []

    models = []
    for fname in os.listdir(reg_dir):
        if fname.endswith("_metadata.json"):
            try:
                fpath = os.path.join(reg_dir, fname)
                with open(fpath, encoding="utf-8") as f:
                    meta = json.load(f)
                    models.append(meta)
            except Exception:
                continue
    models.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return models


# ----------------------------------------------------------------------
# Endpoints (Read-Only)
# ----------------------------------------------------------------------


@router.get("/health", response_model=HealthStatusResponse)
def get_system_health(db: Session = Depends(get_db)):
    """Safe read-only system health and subsystem telemetry."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return HealthStatusResponse(
        status="HEALTHY" if db_ok else "DEGRADED",
        timestamp=datetime.now(UTC).isoformat(),
        database={
            "connected": db_ok,
            "dialect": db.bind.dialect.name if hasattr(db, "bind") and db.bind else "sqlite",
        },
        ml_subsystem={
            "mode": "SHADOW_ONLY",
            "status": "ACTIVE",
            "authority": "ADVISORY_TELEMETRY_ONLY",
        },
        policy_engine={
            "authority": "DETERMINISTIC_AUTHORITY",
            "status": "ACTIVE",
            "execution_gate": "ENFORCED",
        },
        execution_engine={
            "mode": "SIMULATED_SANDBOX",
            "adapter": "MockExecutionAdapter",
            "live_gateways_connected": False,
        },
    )


@router.get("/summary", response_model=SystemSummaryResponse)
def get_system_summary(db: Session = Depends(get_db)):
    """Aggregate counts of persisted domain entities."""
    datasets_cnt = db.query(func.count(DatasetMetadataModel.dataset_id)).scalar() or 0
    cases_cnt = db.query(func.count(CaseModel.case_id)).scalar() or 0
    events_cnt = db.query(func.count(EventModel.event_id)).scalar() or 0
    audits_cnt = db.query(func.count(AuditModel.id)).scalar() or 0
    idemp_cnt = db.query(func.count(IdempotencyRecord.idempotency_key)).scalar() or 0

    # Counts of JSON-persisted lifecycle milestones
    policy_cnt = (
        db.query(func.count(CaseModel.case_id))
        .filter(CaseModel.policy_decision.isnot(None))
        .scalar()
        or 0
    )
    exec_cnt = (
        db.query(func.count(CaseModel.case_id))
        .filter(CaseModel.execution_record.isnot(None))
        .scalar()
        or 0
    )
    outcome_cnt = (
        db.query(func.count(CaseModel.case_id)).filter(CaseModel.outcome.isnot(None)).scalar() or 0
    )
    from infrastructure.orm import OutcomeEventModel

    authoritative_outcome_cnt = (
        db.query(func.count(func.distinct(OutcomeEventModel.case_id)))
        .filter(OutcomeEventModel.case_id.isnot(None))
        .scalar()
        or 0
    )

    # Model registry count from filesystem
    models = _read_model_metadata_files()

    return SystemSummaryResponse(
        datasets_count=datasets_cnt,
        cases_count=cases_cnt,
        revenue_events_count=events_cnt,
        models_count=len(models),
        policy_decisions_count=policy_cnt,
        executions_count=exec_cnt,
        audit_records_count=audits_cnt,
        idempotency_records_count=idemp_cnt,
        outcomes_count=outcome_cnt,
        authoritative_outcomes_count=authoritative_outcome_cnt,
        availability={
            "datasets": "AVAILABLE",
            "cases": "AVAILABLE",
            "revenue_events": "AVAILABLE",
            "models": "AVAILABLE",
            "policy_decisions": "AVAILABLE",
            "executions": "AVAILABLE",
            "audit_records": "AVAILABLE",
            "idempotency_records": "AVAILABLE",
        },
    )


@router.get(
    "/datasets", response_model=DatasetListResponse, dependencies=[Depends(require_operator)]
)
def list_system_datasets(
    limit: int = Query(50, ge=1, le=100, description="Max items to return (1-100)"),
    offset: int = Query(0, ge=0, description="Items to skip"),
    db: Session = Depends(get_db),
):
    """Bounded listing of persisted datasets with safe metadata."""
    total = db.query(func.count(DatasetMetadataModel.dataset_id)).scalar() or 0
    ds_records = (
        db.query(DatasetMetadataModel)
        .order_by(DatasetMetadataModel.upload_timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = []
    for d in ds_records:
        items.append(
            DatasetItemResponse(
                dataset_id=d.dataset_id,
                name=d.name,
                filename=d.filename,
                file_type=d.file_type,
                file_size_bytes=d.file_size_bytes,
                status=d.status.value if hasattr(d.status, "value") else str(d.status),
                row_count=d.row_count,
                column_count=d.column_count,
                upload_timestamp=d.upload_timestamp.isoformat() if d.upload_timestamp else None,
            )
        )

    return DatasetListResponse(total_count=total, limit=limit, offset=offset, items=items)


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetDetailResponse,
    dependencies=[Depends(require_operator)],
)
def get_system_dataset_detail(dataset_id: str, db: Session = Depends(get_db)):
    """Safe detailed inspection of a single dataset without exposing raw files."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", dataset_id):
        raise HTTPException(status_code=400, detail="Invalid dataset ID format")

    ds = (
        db.query(DatasetMetadataModel).filter(DatasetMetadataModel.dataset_id == dataset_id).first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Associated models count
    models = _read_model_metadata_files()
    assoc_models = [m for m in models if m.get("dataset_id") == dataset_id]

    # Associated cases count (via linked revenue event external system matching)
    assoc_cases_cnt = (
        db.query(func.count(EventModel.case_id.distinct()))
        .filter(
            or_(
                and_(EventModel.source_type == "FILE_UPLOAD", EventModel.source_ref == dataset_id),
                EventModel.external_system == f"Dataset-{dataset_id}",  # pre-Phase-3 events
            )
        )
        .scalar()
        or 0
    )

    return DatasetDetailResponse(
        dataset_id=ds.dataset_id,
        name=ds.name,
        filename=ds.filename,
        file_type=ds.file_type,
        file_size_bytes=ds.file_size_bytes,
        status=ds.status.value if hasattr(ds.status, "value") else str(ds.status),
        row_count=ds.row_count,
        column_count=ds.column_count,
        upload_timestamp=ds.upload_timestamp.isoformat() if ds.upload_timestamp else None,
        error_message=ds.error_message,
        profile_summary=ds.columns_profile,
        semantic_mappings=ds.recoverchain_signals,
        data_quality_report=ds.data_quality_report,
        training_suitability=ds.training_suitability,
        leakage_detection=ds.leakage_detection,
        associated_models_count=len(assoc_models),
        associated_cases_count=assoc_cases_cnt,
        content_sha256=ds.content_sha256,
    )


@router.get("/cases", response_model=CaseListResponse, dependencies=[Depends(require_principal)])
def list_system_cases(
    limit: int = Query(50, ge=1, le=100, description="Max items to return (1-100)"),
    offset: int = Query(0, ge=0, description="Items to skip"),
    risk_category: str | None = Query(None, description="Optional category filter"),
    db: Session = Depends(get_db),
):
    """Bounded listing of persisted recovery cases with stored lifecycle fields."""
    query = db.query(CaseModel)
    if risk_category:
        query = query.filter(CaseModel.risk_category == risk_category)

    total = query.count()
    cases = query.order_by(CaseModel.created_at.desc()).offset(offset).limit(limit).all()

    items = []
    for c in cases:
        pol_status = None
        if c.policy_decision:
            pol_status = c.policy_decision.get("status")

        exec_status = None
        if c.execution_record:
            exec_status = c.execution_record.get("status")

        out_status = None
        if c.outcome:
            out_status = c.outcome.get("status")

        pred_prob = None
        pred_status = None
        if c.prediction:
            pred_prob = c.prediction.get("recovery_probability")
            pred_status = c.prediction.get("prediction_status", "SHADOW_ONLY")

        risk_lvl = None
        if c.risk_assessment:
            risk_lvl = c.risk_assessment.get("risk_level")

        diag_cat = None
        if c.diagnosis:
            diag_cat = c.diagnosis.get("cause_category")

        items.append(
            CaseSummaryItem(
                case_id=c.case_id,
                customer_id=c.customer_id,
                risk_category=c.risk_category.value
                if hasattr(c.risk_category, "value")
                else str(c.risk_category),
                reference_id=c.reference_id,
                amount_at_risk=float(c.amount_at_risk) if c.amount_at_risk is not None else 0.0,
                currency=c.currency,
                current_state=c.current_state.value
                if hasattr(c.current_state, "value")
                else str(c.current_state),
                risk_level=risk_lvl,
                cause_category=diag_cat,
                recovery_probability=pred_prob,
                prediction_status=pred_status,
                policy_decision=pol_status,
                execution_status=exec_status,
                outcome_status=out_status,
                created_at=c.created_at.isoformat() if c.created_at else None,
                updated_at=c.updated_at.isoformat() if c.updated_at else None,
            )
        )

    return CaseListResponse(total_count=total, limit=limit, offset=offset, items=items)


@router.get("/cases.csv", dependencies=[Depends(require_operator)])
def export_cases_csv(
    risk_category: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """M2 — download every recovery case as a flat CSV (one row per case)."""
    import csv
    import io

    from fastapi.responses import StreamingResponse

    q = db.query(CaseModel)
    if risk_category:
        q = q.filter(CaseModel.risk_category == risk_category)

    cols = [
        "case_id",
        "customer_id",
        "risk_category",
        "reference_id",
        "amount_at_risk",
        "currency",
        "current_state",
        "risk_level",
        "cause_category",
        "recovery_probability",
        "prediction_status",
        "recommended_action",
        "expected_recoverable_value",
        "policy_status",
        "execution_status",
        "outcome_status",
        "actual_amount_recovered",
        "created_at",
    ]

    def _g(blob, *path):
        for p in path:
            blob = (blob or {}).get(p) if isinstance(blob, dict) else None
        return blob

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for c in q.order_by(CaseModel.created_at.desc()).all():
        rec_amt = _g(c.outcome, "actual_amount_recovered")
        if isinstance(rec_amt, dict):
            rec_amt = rec_amt.get("amount")
        w.writerow(
            [
                c.case_id,
                c.customer_id,
                c.risk_category.value if hasattr(c.risk_category, "value") else c.risk_category,
                c.reference_id,
                float(c.amount_at_risk) if c.amount_at_risk is not None else "",
                c.currency,
                c.current_state.value if hasattr(c.current_state, "value") else c.current_state,
                _g(c.risk_assessment, "risk_level"),
                _g(c.diagnosis, "cause_category"),
                _g(c.prediction, "recovery_probability"),
                _g(c.prediction, "prediction_status"),
                _g(c.recommendation, "top_candidate", "action_type"),
                float(c.expected_recoverable_value)
                if c.expected_recoverable_value is not None
                else "",
                _g(c.policy_decision, "status"),
                _g(c.execution_record, "status"),
                _g(c.outcome, "status"),
                rec_amt if rec_amt is not None else "",
                c.created_at.isoformat() if c.created_at else "",
            ]
        )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="recoverchain_cases.csv"'},
    )


@router.get(
    "/cases/{case_id}",
    response_model=CaseDetailSnapshotResponse,
    dependencies=[Depends(require_principal)],
)
def get_system_case_detail(case_id: str, db: Session = Depends(get_db)):
    """Complete safe 7-stage lifecycle snapshot for a single case."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", case_id):
        raise HTTPException(status_code=400, detail="Invalid case ID format")

    case = db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Safe events
    events_data = []
    for e in case.events:
        events_data.append(
            {
                "event_id": e.event_id,
                "external_system": e.external_system,
                "external_event_id": e.external_event_id,
                "reference_id": e.reference_id,
                "amount": float(e.amount),
                "currency": e.currency,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "source_type": e.source_type,
                "source_ref": e.source_ref,
                "ingestion_run_id": e.ingestion_run_id,
                "ingested_at": e.ingested_at.isoformat() if e.ingested_at else None,
                "domain_basis": e.domain_basis,
            }
        )

    from application.contact_permissions import resolve_all
    from domain.revenue_domains import DomainBasis, domain_spec
    from infrastructure.orm import ContactPermissionModel

    spec = domain_spec(case.risk_category)
    domain = {
        "category": spec.category.value,
        "canonical_name": spec.canonical_name,
        "label": spec.label,
        "bases": sorted({e.domain_basis or DomainBasis.LEGACY_UNRECORDED for e in case.events}),
        "obligation_reference": spec.obligation_reference,
        "obligation_identified": bool(case.reference_id),
        "verification_basis": spec.verification_basis,
        "recovered_means": spec.recovered_means,
    }
    permission_rows = (
        db.query(ContactPermissionModel)
        .filter(ContactPermissionModel.customer_id == case.customer_id)
        .all()
    )
    contact_permission = {ch: r.to_dict() for ch, r in resolve_all(permission_rows).items()}

    # Safe audits
    audits_data = []
    for a in case.audits:
        audits_data.append(
            {
                "audit_id": a.id,
                "from_state": a.from_state,
                "to_state": a.to_state,
                "evidence": a.evidence,
                "timestamp": a.timestamp.isoformat() if a.timestamp else None,
            }
        )

    return CaseDetailSnapshotResponse(
        case_id=case.case_id,
        customer_id=case.customer_id,
        risk_category=case.risk_category.value
        if hasattr(case.risk_category, "value")
        else str(case.risk_category),
        reference_id=case.reference_id,
        amount_at_risk=float(case.amount_at_risk) if case.amount_at_risk is not None else 0.0,
        expected_recoverable_value=float(case.expected_recoverable_value)
        if case.expected_recoverable_value is not None
        else None,
        currency=case.currency,
        current_state=case.current_state.value
        if hasattr(case.current_state, "value")
        else str(case.current_state),
        created_at=case.created_at.isoformat() if case.created_at else None,
        updated_at=case.updated_at.isoformat() if case.updated_at else None,
        data_events=events_data,
        risk_assessment=case.risk_assessment,
        diagnosis=case.diagnosis,
        ml_shadow_prediction=case.prediction,
        recommendation=case.recommendation,
        policy_decision=case.policy_decision,
        execution_record=case.execution_record,
        outcome=case.outcome,
        audit_history=audits_data,
        domain=domain,
        contact_permission=contact_permission,
    )


@router.get("/models", response_model=ModelListResponse, dependencies=[Depends(require_operator)])
def list_system_models(
    dataset_id: str | None = Query(None, description="Optional dataset ID filter"),
):
    """Safe listing of trained model registry metadata without exposing binaries."""
    raw_models = _read_model_metadata_files()
    if dataset_id:
        raw_models = [m for m in raw_models if m.get("dataset_id") == dataset_id]

    items = []
    for m in raw_models:
        metrics = m.get("final_test_metrics") or m.get("test_metrics") or {}
        items.append(
            ModelRegistryItem(
                model_id=m.get("model_id") or m.get("model_version", "unknown"),
                dataset_id=m.get("dataset_id"),
                task=m.get("task", "payment-failure-risk"),
                algorithm=m.get("algorithm", "XGBoostClassifier"),
                model_version=m.get("model_version", "1.0.0"),
                created_at=m.get("created_at"),
                feature_columns=m.get("feature_columns", []),
                target_column=m.get("target_column"),
                metrics=metrics,
                status=m.get("status", "TRAINED"),
            )
        )

    return ModelListResponse(total_count=len(items), items=items)


@router.get("/events", response_model=EventListResponse, dependencies=[Depends(require_operator)])
def list_system_events(
    limit: int = Query(50, ge=1, le=100, description="Max items to return (1-100)"),
    offset: int = Query(0, ge=0, description="Items to skip"),
    case_id: str | None = Query(None, description="Optional case ID filter"),
    db: Session = Depends(get_db),
):
    """Bounded listing of persisted revenue events."""
    query = db.query(EventModel)
    if case_id:
        query = query.filter(EventModel.case_id == case_id)

    total = query.count()
    events = query.order_by(EventModel.timestamp.desc()).offset(offset).limit(limit).all()

    items = []
    for e in events:
        items.append(
            RevenueEventItem(
                event_id=e.event_id,
                case_id=e.case_id,
                customer_id=e.customer_id,
                external_system=e.external_system,
                external_event_id=e.external_event_id,
                reference_id=e.reference_id,
                risk_category=e.risk_category.value
                if hasattr(e.risk_category, "value")
                else str(e.risk_category),
                amount=float(e.amount),
                currency=e.currency,
                timestamp=e.timestamp.isoformat() if e.timestamp else "",
            )
        )

    return EventListResponse(total_count=total, limit=limit, offset=offset, items=items)


@router.get("/policy", response_model=PolicyListResponse, dependencies=[Depends(require_operator)])
def list_system_policy_decisions(
    limit: int = Query(50, ge=1, le=100, description="Max items to return (1-100)"),
    offset: int = Query(0, ge=0, description="Items to skip"),
    db: Session = Depends(get_db),
):
    """Bounded listing of persisted policy decision records."""
    query = db.query(CaseModel).filter(CaseModel.policy_decision.isnot(None))
    total = query.count()
    cases = query.order_by(CaseModel.created_at.desc()).offset(offset).limit(limit).all()

    items = []
    for c in cases:
        pol = c.policy_decision or {}
        items.append(
            PolicyRecordItem(
                case_id=c.case_id,
                customer_id=c.customer_id,
                decision_status=pol.get("status", "UNKNOWN"),
                policy_version=pol.get("policy_version"),
                reason=pol.get("reason"),
                candidate_action=((c.recommendation or {}).get("top_candidate") or {}).get(
                    "action_type"
                ),
                rules_evaluated_count=len(pol.get("rules_evaluated", [])),
                failed_rules_count=len(pol.get("failed_rules", [])),
                timestamp=pol.get("timestamp"),
            )
        )

    return PolicyListResponse(total_count=total, limit=limit, offset=offset, items=items)


@router.get(
    "/executions", response_model=ExecutionListResponse, dependencies=[Depends(require_operator)]
)
def list_system_executions(
    limit: int = Query(50, ge=1, le=100, description="Max items to return (1-100)"),
    offset: int = Query(0, ge=0, description="Items to skip"),
    db: Session = Depends(get_db),
):
    """Bounded listing of every simulated execution ATTEMPT.

    Reads the append-only execution_attempts ledger rather than
    CaseModel.execution_record. That column holds only the most recent execution, so
    a case that really ran several actions previously reported just the last one -
    e.g. three completed sends followed by one rejection showed up as a single
    REJECTED row, hiding the three actions that actually happened.
    """
    query = db.query(ExecutionAttemptModel)
    total = query.count()
    attempts = (
        query.order_by(ExecutionAttemptModel.created_at.desc()).offset(offset).limit(limit).all()
    )

    items = []
    for a in attempts:
        items.append(
            ExecutionRecordItem(
                case_id=a.case_id,
                execution_id=a.execution_id,
                action_type=a.action_type,
                agent_type=a.agent_type,
                execution_mode="SIMULATED/SANDBOX",
                status=a.status or "UNKNOWN",
                adapter_used="MockExecutionAdapter",
                timestamp=a.created_at.isoformat() if a.created_at else None,
            )
        )

    return ExecutionListResponse(total_count=total, limit=limit, offset=offset, items=items)


@router.get("/audit", response_model=AuditListResponse, dependencies=[Depends(require_operator)])
def list_system_audit_records(
    limit: int = Query(50, ge=1, le=100, description="Max items to return (1-100)"),
    offset: int = Query(0, ge=0, description="Items to skip"),
    case_id: str | None = Query(None, description="Optional case ID filter"),
    db: Session = Depends(get_db),
):
    """Bounded listing of audit trail records."""
    query = db.query(AuditModel)
    if case_id:
        query = query.filter(AuditModel.case_id == case_id)

    total = query.count()
    audits = query.order_by(AuditModel.timestamp.desc()).offset(offset).limit(limit).all()

    items = []
    for a in audits:
        items.append(
            AuditRecordItem(
                audit_id=a.id,
                case_id=a.case_id,
                from_state=a.from_state,
                to_state=a.to_state,
                evidence=a.evidence,
                timestamp=a.timestamp.isoformat() if a.timestamp else None,
            )
        )

    return AuditListResponse(total_count=total, limit=limit, offset=offset, items=items)


# ---------------------------------------------------------------------------------
# Phase 8A — data-quality checks, the data-readiness report, temporal coverage, and
# the model-readiness gate. Every figure below is a real query, nothing estimated.
# See docs/history/PHASE_8A_DATA_READINESS_REPORT.md and application/data_readiness.py.
# ---------------------------------------------------------------------------------


@router.get("/data-readiness", dependencies=[Depends(require_operator)])
def data_readiness(db: Session = Depends(get_db)):
    from application.data_readiness import (
        build_data_readiness_report,
        build_root_cause_label_status,
        build_temporal_coverage,
        evaluate_model_readiness,
        run_quality_checks,
    )

    report = build_data_readiness_report(db)
    quality_issues = run_quality_checks(db)
    temporal_all = build_temporal_coverage(db)
    temporal_production = build_temporal_coverage(db, provenance_filter="production")
    temporal_test = build_temporal_coverage(db, provenance_filter="test")
    temporal_external_sim = build_temporal_coverage(
        db, provenance_filter="external_source_simulated"
    )
    root_cause = build_root_cause_label_status(db)
    gates = evaluate_model_readiness(db)

    return {
        "report": report,
        "data_quality_issues": [
            {"outcome_event_id": i.outcome_event_id, "issue": i.issue, "detail": i.detail}
            for i in quality_issues
        ],
        "temporal_coverage": {
            "all_provenance": temporal_all,
            "production_only": temporal_production,
            "test_only": temporal_test,
            "external_source_simulated_only": temporal_external_sim,
        },
        "root_cause_label_status": root_cause,
        "model_readiness_gates": [
            {"task": g.task, "status": g.status, "reasons": g.reasons} for g in gates
        ],
    }


# ---------------------------------------------------------------------------------
# Phase 14 — continuous learning / model monitoring. Read-only: nothing here
# authorizes, promotes, retrains, or executes anything. See docs/history/
# PHASE_14_CONTINUOUS_LEARNING_REPORT.md.
# ---------------------------------------------------------------------------------


@router.get("/cases/{case_id}/agent-trace", dependencies=[Depends(require_operator)])
def case_agent_trace(case_id: str, db: Session = Depends(get_db)):
    """Phase 15/16 - the end-to-end trace for a case, reconstructed read-only from the
    case row, audit trail, execution ledger and outcome events. Operator access."""
    from application.agent_trace import reconstruct_trace

    trace = reconstruct_trace(db, case_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return trace


def _case_id_for_trace(db: Session, trace_id: str) -> str:
    from application.trace_ids import case_id_candidates_for_trace

    candidates = case_id_candidates_for_trace(trace_id)
    if not candidates:
        raise HTTPException(status_code=400, detail="Malformed trace id")
    for cid in candidates:
        if db.query(CaseModel.case_id).filter(CaseModel.case_id == cid).first():
            return cid
    raise HTTPException(status_code=404, detail="Trace not found")


@router.get("/traces/{trace_id}", dependencies=[Depends(require_operator)])
def get_trace(trace_id: str, db: Session = Depends(get_db)):
    from application.agent_trace import reconstruct_trace

    return reconstruct_trace(db, _case_id_for_trace(db, trace_id))


@router.get("/traces/{trace_id}/validation", dependencies=[Depends(require_operator)])
def get_trace_validation(trace_id: str, db: Session = Depends(get_db)):
    """Consistency report for one trace. Reports issues; never repairs history."""
    from application.trace_validation import validate_trace

    return validate_trace(db, _case_id_for_trace(db, trace_id))


@router.get("/trace-consistency", dependencies=[Depends(require_operator)])
def trace_consistency(limit: int = Query(200, ge=1, le=1000), db: Session = Depends(get_db)):
    from application.trace_validation import validate_all

    return validate_all(db, limit)


@router.get("/agent-benchmark", dependencies=[Depends(require_operator)])
def agent_benchmark():
    """Deterministic SYNTHETIC scenario benchmark (never production performance)."""
    from application.agent_evaluation import run_agent_benchmark

    return run_agent_benchmark()


@router.get("/model-monitoring", dependencies=[Depends(require_operator)])
def model_monitoring_summary(db: Session = Depends(get_db)):
    from application.model_monitoring import (
        anomaly_monitoring,
        calibration_bins,
        compute_metrics,
        domain_monitoring,
    )
    from application.model_promotion import (
        data_sufficiency_gate,
        production_validation_status,
        promotion_gate,
    )
    from application.nba_dataset_contract import nba_model_status
    from application.prediction_outcome_linkage import build_prediction_outcome_records

    records = build_prediction_outcome_records(db)
    overall_metrics = compute_metrics(records)

    raw_models = _read_model_metadata_files()
    champion = None
    for m in sorted(raw_models, key=lambda m: m.get("created_at", ""), reverse=True):
        if m.get("status") == "SELECTED":
            champion = m
            break

    champion_summary = None
    if champion is not None:
        champion_summary = {
            "model_id": champion.get("model_id"),
            "model_version": champion.get("model_version"),
            "task": champion.get("task"),
            "dataset_id": champion.get("dataset_id"),
            "dataset_snapshot_checksum": champion.get("dataset_snapshot_checksum"),
            "lifecycle_status": champion.get("lifecycle_status"),
            "provenance_status": champion.get("provenance_status"),
            "calibrated": champion.get("calibrated"),
            "created_at": champion.get("created_at"),
            "quality_gate": champion.get("quality_gate"),
            "last_known_train_test_drift": (
                champion.get("train_test_distribution_drift") or {}
            ).get("any_drifted"),
            "promotion_gate": promotion_gate(champion),
        }

    return {
        "champion": champion_summary,
        "prediction_vs_reality": {
            "status": overall_metrics.status,
            "n": overall_metrics.n,
            "n_recovered": overall_metrics.n_recovered,
            "n_not_recovered": overall_metrics.n_not_recovered,
            "roc_auc": overall_metrics.roc_auc,
            "pr_auc": overall_metrics.pr_auc,
            "brier_score": overall_metrics.brier_score,
            "recovery_rate": overall_metrics.recovery_rate,
            "mean_absolute_erv_error": overall_metrics.mean_absolute_erv_error,
            "provenance_breakdown": overall_metrics.provenance_breakdown,
            "reason": overall_metrics.reason,
        },
        "calibration_bins": calibration_bins(records),
        "domain_breakdown": {
            domain: {"status": block.status, "n": block.n, "reason": block.reason}
            for domain, block in domain_monitoring(records).items()
        },
        "anomaly_monitoring": anomaly_monitoring(records),
        "production_validation_status": production_validation_status(records),
        "data_sufficiency_for_retraining": data_sufficiency_gate(records),
        "nba_model_status": nba_model_status(db),
    }


# ---------------------------------------------------------------------------------
# Phase 8B — training-data eligibility (application/eligibility.py) and immutable
# dataset snapshots (application/dataset_snapshot.py). See docs/history/
# PHASE_8B_INDEPENDENT_EVIDENCE_REPORT.md Step 8/9.
# ---------------------------------------------------------------------------------


@router.get("/eligibility", dependencies=[Depends(require_operator)])
def eligibility_report(db: Session = Depends(get_db)):
    from application.eligibility import evaluate_all

    results = evaluate_all(db)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    return {
        "total_evaluated": len(results),
        "by_status": by_status,
        "results": [
            {"case_id": r.case_id, "status": r.status, "reasons": r.reasons} for r in results
        ],
    }


class DatasetSnapshotRequest(BaseModel):
    dataset_name: str
    created_by: str | None = None


@router.post("/dataset-snapshots", dependencies=[Depends(verify_api_key)])
def create_dataset_snapshot(req: DatasetSnapshotRequest, db: Session = Depends(get_db)):
    from application.dataset_snapshot import build_snapshot

    result = build_snapshot(db, dataset_name=req.dataset_name, created_by=req.created_by)
    db.commit()
    return result


@router.get("/dataset-snapshots", dependencies=[Depends(require_operator)])
def list_dataset_snapshots(dataset_name: str | None = Query(None), db: Session = Depends(get_db)):
    from application.dataset_snapshot import snapshot_dict
    from infrastructure.orm import MLDatasetSnapshotModel

    q = db.query(MLDatasetSnapshotModel)
    if dataset_name:
        q = q.filter(MLDatasetSnapshotModel.dataset_name == dataset_name)
    rows = q.order_by(MLDatasetSnapshotModel.dataset_name, MLDatasetSnapshotModel.version).all()
    return {"total_count": len(rows), "items": [snapshot_dict(r) for r in rows]}


@router.get("/dataset-snapshots/{snapshot_id}", dependencies=[Depends(require_operator)])
def get_dataset_snapshot(snapshot_id: str, db: Session = Depends(get_db)):
    from application.dataset_snapshot import snapshot_dict
    from infrastructure.orm import MLDatasetSnapshotModel

    row = (
        db.query(MLDatasetSnapshotModel)
        .filter(MLDatasetSnapshotModel.snapshot_id == snapshot_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return snapshot_dict(row)
