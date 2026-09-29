import logging
import os
import uuid
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, text
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from api.access import protect_reads_middleware, require_operator, require_reviewer
from api.auth import verify_api_key
from api.schemas import (
    ActionRecommendationResponse,
    AdvanceCaseResponse,
    AuditResponse,
    BatchIngestItemResult,
    BatchIngestRequest,
    BatchIngestResponse,
    CanonicalRecoveryResponse,
    CaseResponse,
    DashboardMetricsResponse,
    DiagnosisResponse,
    ExecutionRecordResponse,
    IngestEventRequest,
    IngestEventResponse,
    PolicyDecisionResponse,
    RecoveryOutcomeResponse,
    RecoveryPredictionResponse,
    RiskAssessmentResponse,
    ShadowERVResponse,
    VerifyOutcomeRequest,
)
from application.case_engine import CaseEngine, DuplicateEventException
from application.trace_ids import trace_id_for_case
from domain.lifecycle import CaseClosedError, CaseLifecycleManager
from domain.models import CaseState, Money, RevenueEvent
from infrastructure.db import Base, engine, get_db

# ... (skipping to line 75) ...
# I will use replace_file_content separately for the APIs and CaseResponse mapping.
from infrastructure.orm import AuditModel, CaseModel, EventModel
from infrastructure.redis_client import get_redis_client
from infrastructure.repositories import SqlAlchemyAuditRecorder, SqlAlchemyCaseRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auto-create tables only for local SQLite dev/test. For Postgres (and any other
# engine) the schema is owned by Alembic migrations: run `alembic upgrade head`.
# Override with AUTO_CREATE_TABLES=1 if you really need create_all elsewhere.
if engine.dialect.name == "sqlite" or os.getenv("AUTO_CREATE_TABLES") == "1":
    Base.metadata.create_all(bind=engine)

from api.security import check_startup_config, rate_limit_middleware

check_startup_config()

app = FastAPI(title="RecoverChain AI Core API", version="0.2.0")

# A wildcard origin and credentialed requests are mutually exclusive per the CORS
# spec (browsers reject the combination). Only enable credentials once an
# explicit allow-list is configured.
_cors_origins = os.getenv("CORS_ORIGINS", "*")
_cors_wildcard = _cors_origins.strip() == "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    if _cors_wildcard
    else [o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(rate_limit_middleware)
app.middleware("http")(protect_reads_middleware)


@app.exception_handler(StaleDataError)
async def _stale_case_handler(request, exc):
    """A concurrent writer changed the case between our read and our write.

    CaseModel carries an optimistic version, so SQLAlchemy refuses the stale UPDATE
    rather than letting the last committer silently win (which could, for example,
    put a settled case back into POLICY_EVALUATED). Surfaced as a retryable conflict
    instead of a 500, because the caller's remedy is simply to re-read and try again.
    """
    from starlette.responses import JSONResponse

    logger.warning("stale_case_write path=%s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=409,
        content={"detail": "Case was modified concurrently. Re-read the case and retry."},
    )


@app.middleware("http")
async def request_context(request, call_next):
    import time as _t

    rid = uuid.uuid4().hex[:12]
    start = _t.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_error rid=%s %s %s", rid, request.method, request.url.path)
        raise
    dur_ms = round((_t.perf_counter() - start) * 1000, 1)
    logger.info(
        "request rid=%s %s %s -> %s %sms",
        rid,
        request.method,
        request.url.path,
        response.status_code,
        dur_ms,
    )
    response.headers["X-Request-ID"] = rid
    return response


from api.dataset_router import router as dataset_router
from api.domain_router import router as domain_router
from api.outcome_router import router as outcome_router
from api.system_router import router as system_router

app.include_router(dataset_router)
app.include_router(system_router)
app.include_router(domain_router)
app.include_router(outcome_router)


from application.ingestion import IngestionRunReport, RejectReason, SourceType  # noqa: E402
from domain.revenue_domains import DomainBasis  # noqa: E402


def get_case_engine(db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)
    return CaseEngine(repository=repo, audit_recorder=audit)


@app.post("/events", response_model=IngestEventResponse)
def ingest_event(
    request: IngestEventRequest,
    engine: CaseEngine = Depends(get_case_engine),
    api_key: str = Depends(verify_api_key),
):
    # Normalize event
    event = RevenueEvent(
        event_id=f"evt_{uuid.uuid4().hex[:8]}",
        customer_id=request.customer_id,
        risk_category=request.risk_category,
        external_system=request.external_system,
        external_event_id=request.external_event_id,
        reference_id=request.reference_id,
        amount=Money(amount=request.amount, currency=request.currency),
        timestamp=datetime.now(UTC),
        raw_payload=request.raw_payload,
        source_type=SourceType.API,
        source_ref=request.external_system,
        domain_basis=DomainBasis.DECLARED_BY_SOURCE,  # risk_category is a required field
    )

    try:
        case, is_new = engine.ingest_normalized_event(event)
        return IngestEventResponse(
            case_id=case.case_id,
            is_new_case=is_new,
            status="success",
            message="Event ingested successfully",
        )
    except DuplicateEventException as e:
        # Idempotent return or 409 Conflict. A 200 with duplicate message is fine for idempotency
        return IngestEventResponse(
            case_id="N/A", is_new_case=False, status="ignored", message=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to ingest event: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during ingestion") from e


@app.post("/events/batch", response_model=BatchIngestResponse)
def ingest_events_batch(
    request: BatchIngestRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Ingest many events in one call; optionally run the deterministic pipeline
    (assess -> diagnose -> predict -> recommend -> policy) on each new/affected case."""
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)
    engine = CaseEngine(repository=repo, audit_recorder=audit)
    from application.case_pipeline import CasePipelineService

    pipeline = CasePipelineService(repo, audit)

    from pydantic import ValidationError

    # One ingestion run per batch call, persisted, so what happened to each item
    # outlives this response.
    report = IngestionRunReport(source_type=SourceType.API_BATCH)
    results: list[BatchIngestItemResult] = []
    ingested = duplicates = failed = 0

    for position, raw_item in enumerate(request.events):
        report.rows_received += 1
        raw_ext_id = (
            str(raw_item.get("external_event_id") or "") if isinstance(raw_item, dict) else ""
        )

        # The same contract as POST /events - but a bad item is REJECTED on its own
        # instead of failing the whole batch with one 422.
        try:
            item = IngestEventRequest.model_validate(raw_item)
        except ValidationError as ve:
            reasons, messages = [], []
            for err in ve.errors():
                failure = (err.get("ctx") or {}).get("error")
                reason = getattr(failure, "reason", None) or (
                    RejectReason.MISSING_FIELD
                    if err.get("type") == "missing"
                    else RejectReason.INVALID_VALUE
                )
                field_name = ".".join(str(p) for p in err.get("loc", ())) or "item"
                reasons.append(reason)
                messages.append(f"{field_name}: {err.get('msg')}")
            report.reject(
                reasons[0], row=position, detail="; ".join(messages)[:300], also=reasons[1:]
            )
            results.append(
                BatchIngestItemResult(
                    external_event_id=raw_ext_id,
                    status="rejected",
                    message="; ".join(messages),
                    reasons=reasons,
                )
            )
            continue

        event = RevenueEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            customer_id=item.customer_id,
            risk_category=item.risk_category,
            external_system=item.external_system,
            external_event_id=item.external_event_id,
            reference_id=item.reference_id,
            amount=Money(amount=item.amount, currency=item.currency),
            timestamp=datetime.now(UTC),
            raw_payload=item.raw_payload,
            source_type=SourceType.API_BATCH,
            source_ref=item.external_system,
            ingestion_run_id=report.run_id,
            domain_basis=DomainBasis.DECLARED_BY_SOURCE,
        )
        try:
            case, is_new = engine.ingest_normalized_event(event)
        except DuplicateEventException as e:
            duplicates += 1
            report.rows_duplicate += 1
            results.append(
                BatchIngestItemResult(
                    external_event_id=item.external_event_id, status="ignored", message=str(e)
                )
            )
            continue
        except Exception as e:
            failed += 1
            db.rollback()
            report.reject(RejectReason.PROCESSING_ERROR, row=position, detail=str(e)[:300])
            logger.error(f"batch ingest failed for {item.external_event_id}: {e}")
            results.append(
                BatchIngestItemResult(
                    external_event_id=item.external_event_id, status="error", message=str(e)
                )
            )
            continue

        ingested += 1
        report.rows_accepted += 1
        if is_new:
            report.cases_created += 1
        else:
            report.cases_updated += 1
        pol_status = None
        if request.auto_advance:
            try:
                case = pipeline.advance(case)
                pol_status = case.policy_decision.status.value if case.policy_decision else None
            except Exception as e:
                logger.error(f"pipeline advance failed for {case.case_id}: {e}")
        results.append(
            BatchIngestItemResult(
                external_event_id=item.external_event_id,
                case_id=case.case_id,
                is_new_case=is_new,
                status="success",
                current_state=case.current_state.value
                if hasattr(case.current_state, "value")
                else str(case.current_state),
                policy_status=pol_status,
                message="ingested" + (" + advanced" if request.auto_advance else ""),
            )
        )

    db.add(report.to_model("COMPLETED"))
    db.commit()

    return BatchIngestResponse(
        submitted=len(request.events),
        ingested=ingested,
        duplicates=duplicates,
        failed=failed,
        rejected=report.rows_rejected - failed,
        ingestion_run_id=report.run_id,
        results=results,
    )


@app.post("/cases/{case_id}/advance", response_model=AdvanceCaseResponse)
def advance_case(
    case_id: str,
    dataset_id: str | None = None,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Run the full deterministic pipeline on an existing case in one call."""
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)
    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from application.case_pipeline import CasePipelineService

    try:
        case = CasePipelineService(repo, audit).advance(case, dataset_id=dataset_id)
    except CaseClosedError as e:
        # A finished (or already-acted) case is a client-state conflict, not a server
        # fault. This previously escaped as an unhandled 500 for PARTIALLY_RECOVERED
        # and DENIED cases while FULLY_RECOVERED returned a polite no-op.
        raise HTTPException(status_code=409, detail=str(e)) from e

    return AdvanceCaseResponse(
        case_id=case.case_id,
        current_state=case.current_state.value
        if hasattr(case.current_state, "value")
        else str(case.current_state),
        risk_level=case.risk_assessment.risk_level if case.risk_assessment else None,
        cause_category=case.diagnosis.cause_category if case.diagnosis else None,
        recovery_probability=case.prediction.recovery_probability if case.prediction else None,
        prediction_status=case.prediction.prediction_status if case.prediction else None,
        recommended_action=(
            case.recommendation.top_candidate.action_type
            if case.recommendation and case.recommendation.top_candidate
            else None
        ),
        expected_recoverable_value=(
            case.expected_recoverable_value.amount if case.expected_recoverable_value else None
        ),
        policy_status=case.policy_decision.status.value if case.policy_decision else None,
        policy_reason=case.policy_decision.reason if case.policy_decision else None,
    )


@app.post("/cases/{case_id}/stop")
def stop_case(
    case_id: str,
    reason: str = "manual_stop",
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Halt all automated recovery on a case (customer opt-out, dispute, manual hold)."""
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)
    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    old = (
        case.current_state.value
        if hasattr(case.current_state, "value")
        else str(case.current_state)
    )
    try:
        CaseLifecycleManager.move_to_stopped(case)
    except CaseClosedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    audit.log_transition(
        case_id=case.case_id,
        from_state=old,
        to_state=CaseState.STOPPED.value,
        evidence={"action": "stop", "reason": reason},
    )
    repo.save(case)
    return {"status": "success", "case_id": case_id, "current_state": "STOPPED", "reason": reason}


@app.get("/cases/{case_id}", response_model=CaseResponse)
def get_case(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return CaseResponse(
        case_id=case.case_id,
        customer_id=case.customer_id,
        risk_category=case.risk_category,
        amount_at_risk=case.amount_at_risk.amount,
        currency=case.amount_at_risk.currency,
        current_state=case.current_state,
        event_count=len(case.linked_events),
        risk_level=case.risk_assessment.risk_level if case.risk_assessment else None,
        cause_category=case.diagnosis.cause_category if case.diagnosis else None,
        recovery_probability=case.prediction.recovery_probability if case.prediction else None,
        expected_recoverable_value=case.expected_recoverable_value.amount
        if hasattr(case, "expected_recoverable_value") and case.expected_recoverable_value
        else None,
        recommended_action=case.recommendation.top_candidate.action_type
        if case.recommendation and case.recommendation.top_candidate
        else None,
        policy_status=case.policy_decision.status.value if case.policy_decision else None,
        execution_status=case.execution_record.status.value
        if hasattr(case, "execution_record") and case.execution_record
        else None,
        outcome_status=case.outcome.status.value
        if hasattr(case, "outcome") and case.outcome
        else None,
        actual_amount_recovered=case.outcome.actual_amount_recovered.amount
        if hasattr(case, "outcome") and case.outcome
        else None,
    )


@app.get("/cases/{case_id}/events")
def get_case_events(case_id: str, db: Session = Depends(get_db)):
    events = db.query(EventModel).filter(EventModel.case_id == case_id).all()
    return [
        {"event_id": e.event_id, "amount": e.amount, "external_id": e.external_event_id}
        for e in events
    ]


@app.get(
    "/cases/{case_id}/audit",
    response_model=list[AuditResponse],
    dependencies=[Depends(require_operator)],
)
def get_case_audit(case_id: str, db: Session = Depends(get_db)):
    audits = (
        db.query(AuditModel)
        .filter(AuditModel.case_id == case_id)
        .order_by(AuditModel.timestamp)
        .all()
    )
    return [
        AuditResponse(
            id=a.id,
            case_id=a.case_id,
            from_state=a.from_state,
            to_state=a.to_state,
            evidence=a.evidence,
            timestamp=a.timestamp.isoformat() if a.timestamp else "",
        )
        for a in audits
    ]


@app.post(
    "/cases/{case_id}/assess-risk",
    dependencies=[Depends(verify_api_key)],
    response_model=RiskAssessmentResponse,
)
def assess_risk(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)

    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from application.risk_detector import DeterministicRiskDetector
    from domain.models import CaseState

    detector = DeterministicRiskDetector()
    assessment = detector.assess_risk(case)

    old_state = (
        case.current_state.value
        if hasattr(case.current_state, "value")
        else str(case.current_state)
    )
    try:
        CaseLifecycleManager.record_risk_assessment(case)
    except CaseClosedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    case.risk_assessment = assessment

    audit.log_transition(
        case_id=case.case_id,
        from_state=old_state,
        to_state=CaseState.ASSESSED.value,
        evidence={
            "action": "risk_assessment",
            "score": assessment.score,
            "level": assessment.risk_level,
        },
    )

    repo.save(case)

    return RiskAssessmentResponse(
        score=assessment.score,
        risk_level=assessment.risk_level,
        detection_status=assessment.detection_status,
        primary_risk_signals=assessment.primary_risk_signals,
        contributing_evidence_references=assessment.contributing_evidence_references,
        assessment_timestamp=assessment.assessment_timestamp.isoformat(),
        detector_version=assessment.detector_version,
    )


@app.get("/cases/{case_id}/risk", response_model=RiskAssessmentResponse)
def get_risk(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case or not case.risk_assessment:
        raise HTTPException(status_code=404, detail="Risk assessment not found")

    assessment = case.risk_assessment
    return RiskAssessmentResponse(
        score=assessment.score,
        risk_level=assessment.risk_level,
        detection_status=assessment.detection_status,
        primary_risk_signals=assessment.primary_risk_signals,
        contributing_evidence_references=assessment.contributing_evidence_references,
        assessment_timestamp=assessment.assessment_timestamp.isoformat(),
        detector_version=assessment.detector_version,
    )


@app.post(
    "/cases/{case_id}/diagnose",
    dependencies=[Depends(verify_api_key)],
    response_model=DiagnosisResponse,
)
def diagnose_case(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)

    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from application.diagnosis_engine import DeterministicDiagnosisEngine
    from domain.models import CaseState

    engine = DeterministicDiagnosisEngine()
    diagnosis = engine.diagnose(case)

    old_state = (
        case.current_state.value
        if hasattr(case.current_state, "value")
        else str(case.current_state)
    )
    try:
        CaseLifecycleManager.record_diagnosis_result(case)
    except CaseClosedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    case.diagnosis = diagnosis

    audit.log_transition(
        case_id=case.case_id,
        from_state=old_state,
        to_state=CaseState.DIAGNOSING.value,
        evidence={
            "action": "diagnosis",
            "cause": diagnosis.cause_category,
            "confidence": diagnosis.confidence,
        },
    )

    repo.save(case)

    return DiagnosisResponse(
        diagnosis_id=diagnosis.diagnosis_id,
        cause_category=diagnosis.cause_category,
        confidence=diagnosis.confidence,
        status=diagnosis.status,
        supporting_signals=diagnosis.supporting_signals,
        evidence_references=diagnosis.evidence_references,
        diagnostic_method=diagnosis.diagnostic_method,
        timestamp=diagnosis.timestamp.isoformat(),
    )


@app.get("/cases/{case_id}/diagnosis", response_model=DiagnosisResponse)
def get_diagnosis(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case or not case.diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    d = case.diagnosis
    return DiagnosisResponse(
        diagnosis_id=d.diagnosis_id,
        cause_category=d.cause_category,
        confidence=d.confidence,
        status=d.status,
        supporting_signals=d.supporting_signals,
        evidence_references=d.evidence_references,
        diagnostic_method=d.diagnostic_method,
        timestamp=d.timestamp.isoformat(),
    )


@app.post(
    "/cases/{case_id}/predict-recovery",
    dependencies=[Depends(verify_api_key)],
    response_model=RecoveryPredictionResponse,
)
def predict_recovery(case_id: str, dataset_id: str | None = None, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)

    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from application.case_pipeline import predict_recovery_for_case

    # Prediction deliberately does not change state, which is why Phase 1 left it
    # unguarded - but it still overwrites case.prediction, so without this an
    # advisory recovery estimate could be recomputed next to an already-settled
    # financial outcome.
    try:
        CaseLifecycleManager.assert_reprocessable(case, "predict recovery")
    except CaseClosedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Shadow ML if a dataset-isolated model exists, else deterministic baseline
    # (never a meaningless 0.0).
    prediction = predict_recovery_for_case(case, dataset_id=dataset_id)
    case.prediction = prediction

    audit.log_transition(
        case_id=case.case_id,
        from_state=case.current_state.value,
        to_state=case.current_state.value,  # State doesn't change here just for prediction
        evidence={
            "action": "recovery_prediction",
            "probability": prediction.recovery_probability,
            "model_version": prediction.model_version,
        },
    )

    repo.save(case)

    return RecoveryPredictionResponse(
        prediction_id=prediction.prediction_id,
        recovery_probability=prediction.recovery_probability,
        confidence=prediction.confidence,
        model_version=prediction.model_version,
        feature_version=prediction.feature_version,
        prediction_timestamp=prediction.prediction_timestamp.isoformat(),
        contributing_features=prediction.contributing_features,
        prediction_status=prediction.prediction_status,
        explanation=prediction.explanation,
        confidence_tier=prediction.confidence_tier,
        anomaly=prediction.anomaly,
    )


@app.get(
    "/cases/{case_id}/recovery-probability",
    dependencies=[Depends(verify_api_key)],
    response_model=CanonicalRecoveryResponse,
)
def get_canonical_recovery_probability(case_id: str, db: Session = Depends(get_db)):
    """Lineage B — the canonical recovery-probability model
    (application/recovery_probability_predictor.py), deliberately separate
    from POST /cases/{id}/predict-recovery (Lineage A, payment-failure-risk
    inverted). Read-only: never writes case.prediction, never affects the
    live policy/ERV path used by the recommendation engine today (see
    docs/ai_reconstruction/MODEL_LINEAGE_RECONCILIATION.md for exactly why
    a full live swap is not made in this pass — most of this model's
    features have no real source in a raw merchant upload yet)."""
    from application.recovery_probability_predictor import RecoveryProbabilityPredictor
    from ml.inference.recovery_features import derive_recovery_features

    repo = SqlAlchemyCaseRepository(db)
    if not repo.get_by_id(case_id):
        raise HTTPException(status_code=404, detail="Case not found")
    return _canonical_recovery(case_id, db, derive_recovery_features, RecoveryProbabilityPredictor)


CANONICAL_APPROXIMATIONS = [
    "amount_vs_customer_average: live = amount / mean(customer's earlier case amounts); "
    "training = amount / a fixed latent per-customer typical amount. NOT mathematically identical; "
    "left missing at cold start (never missing in training).",
    "prior_recovery_rate / number_of_previous_*: live counts only SETTLED prior cases "
    "(recovered, or CLOSED_NOT_RECOVERED); training counts every earlier obligation resolved "
    "within the 30-day window.",
]
CALIBRATION_BASIS = (
    "SYNTHETIC_VALIDATION_ONLY: a sigmoid calibrator was fitted on the synthetic validation split; "
    "no outcome-based live calibration exists (LIVE_OUTCOME_EVALUATION = DATA_BLOCKED)."
)


def _canonical_recovery(case_id, db, derive, predictor_cls):
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    predictor = predictor_cls()
    if predictor.model is None:
        lf = predictor.load_failure or {"reason_code": "UNKNOWN", "detail": ""}
        integrity = lf["reason_code"] not in ("NO_CANONICAL_ARTIFACT", "REGISTRY_MISSING")
        return CanonicalRecoveryResponse(
            status="MODEL_INTEGRITY_FAILED" if integrity else "NO_MODEL",
            integrity_status="REFUSED",
            integrity_failure=lf,
            note=f"Canonical recovery model refused: {lf['reason_code']} ({lf['detail']}). "
            "No fallback model is used.",
        )
    fv = derive(db, case_id, _dt.now(_UTC))
    result = predictor.predict(fv.values)
    base = dict(
        model_id=predictor.model_id,
        model_version=predictor.model_version,
        calibrated=predictor.calibrated,
        calibration_basis=CALIBRATION_BASIS,
        integrity_status="VERIFIED",
        artifact_sha256=predictor.artifact_sha256,
        approximations=CANONICAL_APPROXIMATIONS,
        feature_coverage=round(fv.coverage, 3),
        feature_coverage_label=fv.coverage_label(),
        feature_vector={k: v for k, v in fv.values.items()},
        feature_provenance={k: p.as_dict() for k, p in fv.provenance.items()},
        features_used={k: v for k, v in fv.values.items() if v is not None},
        prediction_timestamp=fv.prediction_timestamp.isoformat(),
    )
    if result["status"] != "SUCCESS":
        return CanonicalRecoveryResponse(status=result["status"], note=result.get("error"), **base)
    row = db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
    amount = float(row.amount_at_risk) if row else 0.0
    p = result["probability"]
    return CanonicalRecoveryResponse(
        status="SUCCESS",
        raw_probability=result.get("raw_probability"),
        calibrated_probability=p,
        calibrated_recovery_probability=p,
        expected_recoverable_value=round(amount * p, 2),
        note=(
            f"READ-ONLY analytical path; not the production decision. {fv.coverage_label()} "
            "features are genuinely derivable live; the rest are passed as missing (never "
            "fabricated) and handled by the trained pipeline's imputer. "
            f"DATA_BLOCKED: {', '.join(fv.blocked)}."
        ),
        **base,
    )


@app.get(
    "/cases/{case_id}/shadow-erv",
    dependencies=[Depends(verify_api_key)],
    response_model=ShadowERVResponse,
)
def get_shadow_erv(case_id: str, db: Session = Depends(get_db)):
    """Read-only diagnostic: current production ERV vs canonical recovery ERV.
    Executes nothing and changes no financial behaviour."""
    from application.recovery_probability_predictor import RecoveryProbabilityPredictor
    from ml.inference.recovery_features import derive_recovery_features

    repo = SqlAlchemyCaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    amount = float(case.amount_at_risk.amount) if case.amount_at_risk else 0.0
    cur_p = case.prediction.recovery_probability if case.prediction else None
    cur_src = case.prediction.model_version if case.prediction else None
    top = case.recommendation.top_candidate if case.recommendation else None
    cur_erv = (
        top.expected_recoverable_value
        if top
        else (round(amount * cur_p, 2) if cur_p is not None else None)
    )
    canon = _canonical_recovery(case_id, db, derive_recovery_features, RecoveryProbabilityPredictor)
    c_erv = canon.expected_recoverable_value
    return ShadowERVResponse(
        case_id=case_id,
        amount_at_risk=amount,
        current_probability=cur_p,
        current_probability_source=cur_src,
        current_erv=cur_erv,
        current_erv_probability=top.estimated_probability if top else None,
        canonical_status=canon.status,
        canonical_probability=canon.calibrated_probability,
        canonical_model_id=canon.model_id,
        canonical_erv=c_erv,
        difference_erv=(
            round(c_erv - cur_erv, 2) if c_erv is not None and cur_erv is not None else None
        ),
        feature_coverage_label=canon.feature_coverage_label,
        canonical_model_version=canon.model_version,
        canonical_calibrated=canon.calibrated,
        canonical_calibration_basis=canon.calibration_basis,
        canonical_integrity_status=canon.integrity_status,
        prediction_timestamp=canon.prediction_timestamp,
        approximations=canon.approximations,
        note="SHADOW / evaluation only. The canonical ERV never drives a recommendation, policy "
        "decision or execution; the production ERV path is unchanged. The production decision "
        "uses `current_erv`, not `canonical_erv`.",
    )


@app.get("/cases/{case_id}/recovery-prediction", response_model=RecoveryPredictionResponse)
def get_prediction(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case or not case.prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    p = case.prediction
    return RecoveryPredictionResponse(
        prediction_id=p.prediction_id,
        recovery_probability=p.recovery_probability,
        confidence=p.confidence,
        model_version=p.model_version,
        feature_version=p.feature_version,
        prediction_timestamp=p.prediction_timestamp.isoformat(),
        contributing_features=p.contributing_features,
        prediction_status=p.prediction_status,
        explanation=p.explanation,
        confidence_tier=p.confidence_tier,
        anomaly=p.anomaly,
    )


@app.post(
    "/cases/{case_id}/recommend-action",
    dependencies=[Depends(verify_api_key)],
    response_model=ActionRecommendationResponse,
)
def recommend_action(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)

    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from application.action_evaluator import DeterministicActionEvaluator
    from domain.models import CaseState

    evaluator = DeterministicActionEvaluator()
    recommendation = evaluator.evaluate(case)

    old_state = (
        case.current_state.value
        if hasattr(case.current_state, "value")
        else str(case.current_state)
    )
    try:
        CaseLifecycleManager.record_recommendation_result(case)
    except CaseClosedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    case.recommendation = recommendation
    if recommendation.top_candidate:
        _cur = case.amount_at_risk.currency if case.amount_at_risk else "INR"
        case.expected_recoverable_value = Money(
            amount=recommendation.top_candidate.expected_recoverable_value, currency=_cur
        )

    top_type = recommendation.top_candidate.action_type if recommendation.top_candidate else "NONE"

    audit.log_transition(
        case_id=case.case_id,
        from_state=old_state,
        to_state=CaseState.RECOMMENDING.value,
        evidence={
            "action": "action_recommendation",
            "top_candidate": top_type,
            "status": recommendation.status.value,
        },
    )

    repo.save(case)

    return ActionRecommendationResponse(
        recommendation_id=recommendation.recommendation_id,
        candidates=[c.model_dump() for c in recommendation.candidates],
        top_candidate=recommendation.top_candidate.model_dump()
        if recommendation.top_candidate
        else None,
        status=recommendation.status.value,
        rationale=recommendation.rationale,
        engine_version=recommendation.engine_version,
        timestamp=recommendation.timestamp.isoformat(),
    )


@app.get("/cases/{case_id}/recommendation", response_model=ActionRecommendationResponse)
def get_recommendation(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case or not case.recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    rec = case.recommendation
    return ActionRecommendationResponse(
        recommendation_id=rec.recommendation_id,
        candidates=[c.model_dump() for c in rec.candidates],
        top_candidate=rec.top_candidate.model_dump() if rec.top_candidate else None,
        status=rec.status.value,
        rationale=rec.rationale,
        engine_version=rec.engine_version,
        timestamp=rec.timestamp.isoformat(),
    )


@app.post(
    "/cases/{case_id}/policy-check",
    dependencies=[Depends(verify_api_key)],
    response_model=PolicyDecisionResponse,
)
def policy_check(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)

    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from application.policy_engine import DeterministicPolicyEngine
    from domain.models import CaseState

    engine = DeterministicPolicyEngine()
    try:
        _ctx = repo.get_policy_context(case_id)
    except Exception:
        _ctx = None
    decision = engine.evaluate(case, context=_ctx)

    old_state = (
        case.current_state.value
        if hasattr(case.current_state, "value")
        else str(case.current_state)
    )
    try:
        CaseLifecycleManager.record_policy_evaluation(case)
    except CaseClosedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    case.policy_decision = decision

    action_type = (
        case.recommendation.top_candidate.action_type
        if case.recommendation and case.recommendation.top_candidate
        else "NONE"
    )

    audit.log_transition(
        case_id=case.case_id,
        from_state=old_state,
        to_state=CaseState.POLICY_EVALUATED.value,
        evidence={
            "action": "policy_evaluation",
            "candidate_action": action_type,
            "decision": decision.status.value,
            "reason": decision.reason,
            "decision_id": decision.decision_id,
            "contact_permission": decision.contact_permission,
        },
    )

    repo.save(case)

    return PolicyDecisionResponse(
        decision_id=decision.decision_id,
        status=decision.status.value,
        policy_version=decision.policy_version,
        rules_evaluated=[r.model_dump() for r in decision.rules_evaluated],
        failed_rules=[r.model_dump() for r in decision.failed_rules],
        reason=decision.reason,
        timestamp=decision.timestamp.isoformat(),
        contact_permission=decision.contact_permission,
    )


@app.get("/cases/{case_id}/policy-decision", response_model=PolicyDecisionResponse)
def get_policy_decision(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case or not case.policy_decision:
        raise HTTPException(status_code=404, detail="Policy decision not found")

    dec = case.policy_decision
    return PolicyDecisionResponse(
        decision_id=dec.decision_id,
        status=dec.status.value,
        policy_version=dec.policy_version,
        rules_evaluated=[r.model_dump() for r in dec.rules_evaluated],
        failed_rules=[r.model_dump() for r in dec.failed_rules],
        reason=dec.reason,
        timestamp=dec.timestamp.isoformat(),
        contact_permission=dec.contact_permission,
    )


@app.post("/cases/{case_id}/execute", response_model=ExecutionRecordResponse)
def execute_action(
    case_id: str, db: Session = Depends(get_db), api_key: str = Depends(verify_api_key)
):
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)

    # Row-locked read: execution is the one operation with an external side effect,
    # so it takes the same lock verify and human-review already use. (On SQLite this
    # is a no-op; the authoritative protection is the execution-slot claim inside
    # AgentOrchestrator, which is enforced by a primary key on both backends.)
    case = repo.get_by_id_for_update(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from application.agents import AgentOrchestrator
    from application.policy_engine import DeterministicPolicyEngine
    from domain.models import CaseState, ExecutionRecord, ExecutionStatus

    # We execute whatever action the policy authorized
    if not case.policy_decision or not case.recommendation or not case.recommendation.top_candidate:
        raise HTTPException(status_code=400, detail="Missing policy or recommendation")

    requested_action = case.recommendation.top_candidate.action_type

    old_state = (
        case.current_state.value
        if hasattr(case.current_state, "value")
        else str(case.current_state)
    )

    # Replay first. If this exact authorization has already dispatched, return that
    # record unchanged. This must come BEFORE revalidation: after a successful
    # execution the ledger legitimately shows a prior attempt and a live cooldown, so
    # revalidating first would answer an idempotent retry with a fresh REJECTED record
    # instead of the original result.
    from application.agents import execution_idempotency_key

    _existing = None
    try:
        _cached = repo.get_execution_record_by_key(
            execution_idempotency_key(case, requested_action)
        )
        if _cached and not SqlAlchemyCaseRepository.is_claim_placeholder(_cached):
            _existing = ExecutionRecord(**_cached)
    except Exception:
        _existing = None
    if _existing is not None:
        return ExecutionRecordResponse(
            execution_id=_existing.execution_id,
            action_type=_existing.action_type.value,
            agent_type=_existing.agent_type,
            policy_decision_id=_existing.policy_decision_id,
            status=_existing.status.value,
            adapter_used=_existing.adapter_used,
            timestamp=_existing.timestamp.isoformat(),
            result_metadata=_existing.result_metadata,
            trace_id=trace_id_for_case(case_id),
            operation_id=_existing.execution_id,
        )

    # Deterministic revalidation immediately before the side effect.
    #
    # Human approval previously flipped policy_decision.status to PERMITTED and
    # execution simply trusted that flag, so nothing re-checked the hard limits
    # between approval and execution: a retry cap or contact cap could have been
    # reached, or the case stopped, in the meantime. The Policy Engine is re-run here
    # against fresh durable context. A human approval counts only for the decision it
    # was granted on, and only satisfies the automated-amount threshold - every other
    # rule still has to pass.
    try:
        _revalidation_ctx = repo.get_policy_context(
            case_id, approved_decision_id=case.policy_decision.decision_id
        )
    except Exception:
        _revalidation_ctx = None
    _revalidated = DeterministicPolicyEngine().evaluate(case, context=_revalidation_ctx)
    # Phase 17: record the revalidation as structured, append-only evidence - PASSED or
    # FAILED - before either branch acts on it. Nothing here changes the verdict.
    from application.revalidation import latest_approval, revalidation_evidence
    from infrastructure.orm import AuditModel as _AuditModel

    _approval = latest_approval(
        db.query(_AuditModel).filter(_AuditModel.case_id == case_id).all(),
        case.policy_decision.decision_id,
    )
    _reval_evidence = revalidation_evidence(
        case=case, requested_action=requested_action, revalidated=_revalidated, approval=_approval
    )
    audit.log_transition(
        case_id=case.case_id, from_state=old_state, to_state=old_state, evidence=_reval_evidence
    )
    if _revalidated.status != PolicyDecisionStatus.PERMITTED:
        record = ExecutionRecord(
            execution_id=f"exec_{uuid.uuid4().hex[:8]}",
            action_type=requested_action,
            agent_type="Orchestrator",
            policy_decision_id=case.policy_decision.decision_id,
            policy_version=case.policy_decision.policy_version,
            parameters={},
            idempotency_key="REJECTED_POLICY_REVALIDATION",
            trace_id=trace_id_for_case(case_id),
            case_id=case_id,
            status=ExecutionStatus.REJECTED,
            adapter_used="NONE",
            result_metadata={
                "error": "Policy revalidation before execution did not permit this action.",
                "revalidated_status": _revalidated.status.value,
                "reason": _revalidated.reason,
            },
        )
        case.execution_record = record
        audit.log_transition(
            case_id=case.case_id,
            from_state=old_state,
            to_state=old_state,
            # Keep the existing "execution" audit vocabulary - a blocked attempt is
            # still an execution attempt - and add why it was stopped.
            evidence={
                "action": "execution",
                "agent": record.agent_type,
                "status": record.status.value,
                "adapter": record.adapter_used,
                "execution_id": record.execution_id,
                "policy_decision_id": record.policy_decision_id,
                "revalidation_id": _reval_evidence["revalidation_id"],
                "blocked_by": "policy_revalidation",
                "revalidated_decision": _revalidated.status.value,
                "reason": _revalidated.reason,
            },
        )
        repo.save(case)
        return ExecutionRecordResponse(
            execution_id=record.execution_id,
            action_type=record.action_type.value,
            agent_type=record.agent_type,
            policy_decision_id=record.policy_decision_id,
            status=record.status.value,
            adapter_used=record.adapter_used,
            timestamp=record.timestamp.isoformat(),
            result_metadata=record.result_metadata,
            trace_id=record.trace_id,
            operation_id=record.operation_id,
        )

    orchestrator = AgentOrchestrator()
    record = orchestrator.execute(case, requested_action, repo=repo)

    if record.status == ExecutionStatus.COMPLETED_SIMULATED:
        target_state = CaseState.PENDING_VERIFICATION
    elif record.status == ExecutionStatus.REJECTED:
        target_state = case.current_state  # keep current state (existing behavior)
    else:
        target_state = CaseState.EXECUTING  # Wait or failed

    try:
        CaseLifecycleManager.record_execution_result(case, target_state)
    except CaseClosedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    case.execution_record = record

    new_state = (
        case.current_state.value
        if hasattr(case.current_state, "value")
        else str(case.current_state)
    )
    audit.log_transition(
        case_id=case.case_id,
        from_state=old_state,
        to_state=new_state,
        evidence={
            "action": "execution",
            "agent": record.agent_type,
            "status": record.status.value,
            "adapter": record.adapter_used,
            "execution_id": record.execution_id,
            "policy_decision_id": record.policy_decision_id,
            "revalidation_id": _reval_evidence["revalidation_id"],
        },
    )

    repo.save(case)

    return ExecutionRecordResponse(
        execution_id=record.execution_id,
        action_type=record.action_type.value,
        agent_type=record.agent_type,
        policy_decision_id=record.policy_decision_id,
        status=record.status.value,
        adapter_used=record.adapter_used,
        timestamp=record.timestamp.isoformat(),
        result_metadata=record.result_metadata,
        trace_id=record.trace_id,
        operation_id=record.operation_id,
    )


@app.post(
    "/cases/{case_id}/verify",
    dependencies=[Depends(verify_api_key)],
    response_model=RecoveryOutcomeResponse,
)
def verify_outcome(case_id: str, request: VerifyOutcomeRequest, db: Session = Depends(get_db)):
    """Phase 5: this is now a READ of the case's authoritative financial position, not a
    decision. `request.external_reference` is accepted for backward API compatibility
    but no longer selects or fabricates an outcome - the money moved only if a real
    outcome event (POST /outcomes, /outcomes/batch) already applied it
    (application/outcome_events.py). An executed action with no matching outcome event
    yet correctly reports PENDING_VERIFICATION with nothing recovered: ACTION ATTEMPT !=
    ACTION SUCCESS != REVENUE RECOVERED."""
    repo = SqlAlchemyCaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from application.verification_engine import VerificationEngine

    outcome = VerificationEngine().current_position(case)

    return RecoveryOutcomeResponse(
        outcome_id=outcome.outcome_id,
        status=outcome.status.value,
        expected_amount=outcome.expected_amount.amount,
        actual_amount_recovered=outcome.actual_amount_recovered.amount,
        currency=outcome.actual_amount_recovered.currency,
        verification_source=outcome.verification_source,
        external_reference=outcome.external_reference,
        reconciliation_status=outcome.reconciliation_status,
        timestamp=outcome.verification_timestamp.isoformat(),
        verification_basis=outcome.verification_basis,
        action_result=outcome.action_result,
        outstanding_amount=outcome.outstanding_amount.amount
        if outcome.outstanding_amount
        else None,
        cancelled_amount=outcome.cancelled_amount.amount if outcome.cancelled_amount else None,
        applied_outcome_event_ids=outcome.applied_outcome_event_ids,
        provenance=outcome.provenance,
    )


@app.get("/cases/{case_id}/outcome", response_model=RecoveryOutcomeResponse)
def get_outcome(case_id: str, db: Session = Depends(get_db)):
    repo = SqlAlchemyCaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case or not case.outcome:
        raise HTTPException(status_code=404, detail="Outcome not found")

    outcome = case.outcome
    return RecoveryOutcomeResponse(
        outcome_id=outcome.outcome_id,
        status=outcome.status.value,
        expected_amount=outcome.expected_amount.amount,
        actual_amount_recovered=outcome.actual_amount_recovered.amount,
        currency=outcome.actual_amount_recovered.currency,
        verification_source=outcome.verification_source,
        external_reference=outcome.external_reference,
        reconciliation_status=outcome.reconciliation_status,
        timestamp=outcome.verification_timestamp.isoformat(),
        verification_basis=outcome.verification_basis,
        action_result=outcome.action_result,
        outstanding_amount=outcome.outstanding_amount.amount
        if outcome.outstanding_amount
        else None,
        cancelled_amount=outcome.cancelled_amount.amount if outcome.cancelled_amount else None,
        applied_outcome_event_ids=outcome.applied_outcome_event_ids,
        provenance=outcome.provenance,
    )


from api.schemas import EvaluationRunResponse, IngestionRunResponse  # noqa: E402


def _evaluation_run_response(run, include_metrics: bool) -> EvaluationRunResponse:
    return EvaluationRunResponse(
        run_id=run.run_id,
        status=run.status,
        experiment_mode=run.experiment_mode,
        isolation=run.isolation,
        started_at=run.started_at.isoformat() if run.started_at else "",
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        total_scenarios=run.total_scenarios,
        passed_scenarios=run.passed_scenarios,
        error_message=run.error_message,
        metrics=run.metrics if include_metrics else None,
    )


@app.post("/evaluation/run", dependencies=[Depends(verify_api_key)])
def run_evaluation(db: Session = Depends(get_db)):
    """Run the gold-labelled evaluation scenarios.

    The scenarios run in an isolated, ephemeral store (EvaluationRunner.isolated):
    nothing they create reaches recovery_cases, revenue_events, audit_records or
    execution_attempts, so an evaluation run cannot move any operational figure. Only
    a summary row is kept, in evaluation_runs, which no operational aggregate reads.
    This route previously handed the runner the request's own session, which put
    every scenario into the live case table and changed dashboard money totals.
    """
    from evaluation.runner import AblationConfig, EvaluationRunner
    from evaluation.scenarios import SCENARIOS
    from infrastructure.orm import EvaluationRunModel

    ablation = AblationConfig()
    run = EvaluationRunModel(
        run_id=f"eval_{uuid.uuid4().hex[:16]}",
        status="RUNNING",
        experiment_mode=ablation.experiment_mode,
        isolation=EvaluationRunner.ISOLATION,
        started_at=datetime.now(UTC),
    )
    try:
        with EvaluationRunner.isolated(ablation) as runner:
            metrics = runner.run_all(SCENARIOS)
    except Exception as exc:
        logger.exception("evaluation run %s failed", run.run_id)
        run.status = "FAILED"
        run.completed_at = datetime.now(UTC)
        run.error_message = f"{type(exc).__name__}: {exc}"[:500]
        db.add(run)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Evaluation run {run.run_id} failed.") from exc

    payload = metrics.model_dump(mode="json")
    run.status = "COMPLETED"
    run.completed_at = datetime.now(UTC)
    run.total_scenarios = metrics.total_scenarios
    run.passed_scenarios = metrics.passed_scenarios
    run.metrics = payload
    db.add(run)
    db.commit()

    return {
        **payload,
        "evaluation_run_id": run.run_id,
        "data_class": "EVALUATION",
        "isolation": run.isolation,
    }


@app.get("/evaluation/runs", response_model=list[EvaluationRunResponse])
def list_evaluation_runs(limit: int = 20, db: Session = Depends(get_db)):
    """Evaluation runs, newest first. Summary only; fetch one run for its metrics."""
    from infrastructure.orm import EvaluationRunModel

    limit = max(1, min(limit, 100))
    rows = (
        db.query(EvaluationRunModel)
        .order_by(EvaluationRunModel.started_at.desc())
        .limit(limit)
        .all()
    )
    return [_evaluation_run_response(r, include_metrics=False) for r in rows]


@app.get("/evaluation/runs/{run_id}", response_model=EvaluationRunResponse)
def get_evaluation_run(run_id: str, db: Session = Depends(get_db)):
    from infrastructure.orm import EvaluationRunModel

    run = db.query(EvaluationRunModel).filter(EvaluationRunModel.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return _evaluation_run_response(run, include_metrics=True)


@app.get("/ingestion-runs/{run_id}", response_model=IngestionRunResponse)
def get_ingestion_run(run_id: str, db: Session = Depends(get_db)):
    """The persisted data-quality result of one ingestion run (a dataset replay or an
    /events/batch call)."""
    from application.ingestion import ingestion_run_to_dict
    from infrastructure.orm import IngestionRunModel

    run = db.query(IngestionRunModel).filter(IngestionRunModel.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Ingestion run not found")
    return ingestion_run_to_dict(run)


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    from infrastructure.redis_client import redis_enabled

    health_status = {"api": "ok", "db": "unknown", "redis": "disabled"}
    try:
        db.execute(text("SELECT 1"))
        health_status["db"] = "ok"
    except Exception:
        health_status["db"] = "error"

    if redis_enabled():
        try:
            get_redis_client().ping()
            health_status["redis"] = "ok"
        except Exception:
            health_status["redis"] = "error"
    return health_status


from infrastructure.orm import OutcomeEventModel


def _json_get(blob, *keys):
    cur = blob
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _scalar(v):
    """Unwrap a serialized Money dict ({'amount','currency'}) to its amount."""
    return v.get("amount") if isinstance(v, dict) else v


@app.get("/cases", response_model=list[CaseResponse])
def list_cases(limit: int = 200, offset: int = 0, db: Session = Depends(get_db)):
    """Read straight from the persisted JSON columns — one query, no per-row
    domain rehydration (was an N+1 over every case)."""
    limit = max(1, min(limit, 500))
    q = (
        db.query(CaseModel, func.count(EventModel.event_id))
        .outerjoin(EventModel, EventModel.case_id == CaseModel.case_id)
        .group_by(CaseModel.case_id)
        .order_by(CaseModel.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = []
    for c, event_count in q.all():
        result.append(
            CaseResponse(
                case_id=c.case_id,
                customer_id=c.customer_id,
                risk_category=c.risk_category,
                amount_at_risk=c.amount_at_risk,
                currency=c.currency,
                current_state=c.current_state,
                event_count=event_count or 0,
                risk_level=_json_get(c.risk_assessment, "risk_level"),
                cause_category=_json_get(c.diagnosis, "cause_category"),
                recovery_probability=_json_get(c.prediction, "recovery_probability"),
                expected_recoverable_value=c.expected_recoverable_value,
                recommended_action=_json_get(c.recommendation, "top_candidate", "action_type"),
                policy_status=_json_get(c.policy_decision, "status"),
                execution_status=_json_get(c.execution_record, "status"),
                outcome_status=_json_get(c.outcome, "status"),
                actual_amount_recovered=_scalar(_json_get(c.outcome, "actual_amount_recovered")),
            )
        )
    return result


@app.get("/dashboard/metrics", response_model=DashboardMetricsResponse)
def get_dashboard_metrics(db: Session = Depends(get_db)):
    cases = db.query(CaseModel).all()

    total_risk = 0.0
    active = high_crit = pending = waiting = 0
    opportunities = 0.0
    recovered = 0.0
    outstanding_total = 0.0
    cancelled_total = 0.0
    # Per-currency figures. RecoverChain performs no currency conversion, so the
    # headline totals are plain sums and only meaningful when one currency is present.
    breakdown: dict[str, dict[str, float]] = {}
    _terminal = {
        "FULLY_RECOVERED",
        "PARTIALLY_RECOVERED",
        "CLOSED_NOT_RECOVERED",
        "STOPPED",
        "DENIED",
        "CANCELLED",
    }

    for c in cases:
        ccy = (c.currency or "INR").strip().upper()
        bucket = breakdown.setdefault(
            ccy,
            {
                "total_revenue_at_risk": 0.0,
                "verified_recovery": 0.0,
                "recovery_opportunities": 0.0,
                "outstanding_amount": 0.0,
                "cancelled_amount": 0.0,
            },
        )
        risk = float(c.amount_at_risk or 0)
        total_risk += risk
        bucket["total_revenue_at_risk"] += risk

        state_val = (
            c.current_state.value if hasattr(c.current_state, "value") else str(c.current_state)
        )
        if state_val not in _terminal:
            active += 1

        if _json_get(c.risk_assessment, "risk_level") in ("HIGH", "CRITICAL"):
            high_crit += 1

        pol_val = _json_get(c.policy_decision, "status")
        if pol_val == "ESCALATE":
            pending += 1
        elif pol_val == "WAIT":
            waiting += 1

        erv = _json_get(c.recommendation, "top_candidate", "expected_recoverable_value")
        if erv is not None:
            opportunities += float(erv)
            bucket["recovery_opportunities"] += float(erv)

        # Phase 5: recovered_amount / outstanding_amount / cancelled_amount are
        # maintained only by application.outcome_events from real outcome events - see
        # that module's docstring. They agree with case.outcome.actual_amount_recovered
        # whenever an outcome has been reconciled, and additionally stay correct
        # (0 recovered, fully outstanding) for a case no outcome event has touched yet,
        # which the JSON outcome blob alone cannot represent.
        recovered_amt = float(c.recovered_amount or 0)
        outstanding_amt = float(c.outstanding_amount if c.outstanding_amount is not None else risk)
        cancelled_amt = float(c.cancelled_amount or 0)
        recovered += recovered_amt
        outstanding_total += outstanding_amt
        cancelled_total += cancelled_amt
        bucket["verified_recovery"] += recovered_amt
        bucket["outstanding_amount"] += outstanding_amt
        bucket["cancelled_amount"] += cancelled_amt

    unmatched_count = (
        db.query(OutcomeEventModel).filter(OutcomeEventModel.match_status == "UNMATCHED").count()
    )
    unmatched_amount = (
        db.query(func.coalesce(func.sum(OutcomeEventModel.amount), 0))
        .filter(OutcomeEventModel.match_status == "UNMATCHED")
        .scalar()
        or 0
    )

    return DashboardMetricsResponse(
        total_revenue_at_risk=total_risk,
        active_cases=active,
        high_critical_cases=high_crit,
        pending_human_review=pending,
        waiting_cases=waiting,
        recovery_opportunities=opportunities,
        verified_recovery=recovered,
        recovery_gap=total_risk - recovered,
        outstanding_amount=outstanding_total,
        cancelled_amount=cancelled_total,
        unmatched_outcome_events=unmatched_count,
        unmatched_outcome_amount=unmatched_amount,
        currency_breakdown=breakdown,
        mixed_currency=len(breakdown) > 1,
    )


from api.schemas import HumanReviewRequest
from domain.models import PolicyDecisionStatus


@app.post("/cases/{case_id}/human-review")
def submit_human_review(
    case_id: str,
    request: HumanReviewRequest,
    db: Session = Depends(get_db),
    principal=Depends(require_reviewer),
):
    """Approve / reject an escalated action. Phase 17: REVIEWER or ADMIN only, and the
    acting principal + role are written into the (append-only) approval evidence.
    Approval is NOT authorization forever - POST /execute revalidates the Policy
    Engine immediately before any side effect."""
    repo = SqlAlchemyCaseRepository(db)
    audit = SqlAlchemyAuditRecorder(db)

    case = repo.get_by_id_for_update(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not case.policy_decision or case.policy_decision.status != PolicyDecisionStatus.ESCALATE:
        raise HTTPException(status_code=400, detail="Case is not pending human escalation.")

    old_state = case.current_state.value
    approved = request.decision.upper() == "APPROVE"

    # The decision the human is actually reviewing, and the action it authorises.
    # Both are recorded in the audit evidence below so that execution can verify it
    # is running the exact thing that was approved, and so that re-running the policy
    # check (which mints a new decision_id) correctly invalidates this approval.
    reviewed_decision_id = case.policy_decision.decision_id
    reviewed_action = (
        case.recommendation.top_candidate.action_type.value
        if case.recommendation and case.recommendation.top_candidate
        else None
    )

    if approved:
        case.policy_decision.status = PolicyDecisionStatus.PERMITTED
        case.policy_decision.reason += f" [Human Approved: {request.note}]"
    else:
        case.policy_decision.status = PolicyDecisionStatus.DENIED
        case.policy_decision.reason += f" [Human Rejected: {request.note}]"
        # remains in escalated or moves to closed

    CaseLifecycleManager.record_human_review(
        case, approved
    )  # POLICY_REVIEW on approve, else unchanged

    audit.log_transition(
        case_id=case.case_id,
        from_state=old_state,
        to_state=case.current_state.value,
        evidence={
            "action": "human_review",
            "decision": request.decision,
            "note": request.note,
            "approved_decision_id": reviewed_decision_id,
            "approved_action": reviewed_action,
            # Phase 17 approval evidence (structured). The actor is the authenticated
            # principal - never a claim taken from the request body.
            "approval_id": f"appr_{uuid.uuid4().hex[:12]}",
            "case_id": case.case_id,
            "policy_decision_id": reviewed_decision_id,
            "policy_version": case.policy_decision.policy_version,
            "actor": principal.as_evidence(),
        },
    )

    repo.save(case)
    db.commit()
    return {"status": "success", "decision": case.policy_decision.status.value}
