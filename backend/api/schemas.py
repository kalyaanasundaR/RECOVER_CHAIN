from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ValidationInfo, field_validator

from application.ingestion import normalize_currency, normalize_identifier, parse_amount
from domain.models import CaseState, RiskCategory


class IngestEventRequest(BaseModel):
    """The API source adapter. Every field goes through the same ingestion contract
    as a dataset row (application/ingestion.py). Previously a negative, zero or 1e15
    amount, a blank customer or event id, and any currency string were all accepted."""

    customer_id: str
    risk_category: RiskCategory
    external_system: str
    external_event_id: str
    reference_id: str | None = None
    amount: Decimal
    currency: str = "INR"
    raw_payload: dict[str, Any]

    @field_validator("customer_id", "external_system", "external_event_id", mode="before")
    @classmethod
    def _required_identifier(cls, v: Any, info: ValidationInfo) -> str:
        return normalize_identifier(v, info.field_name)

    @field_validator("reference_id", mode="before")
    @classmethod
    def _optional_reference(cls, v: Any) -> str | None:
        if v is None or not str(v).strip():
            return None
        return str(v).strip()

    @field_validator("amount", mode="before")
    @classmethod
    def _amount(cls, v: Any) -> Decimal:
        return parse_amount(v)

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, v: Any) -> str:
        return normalize_currency(v, default="INR")


class IngestEventResponse(BaseModel):
    case_id: str
    is_new_case: bool
    status: str
    message: str


class RiskAssessmentResponse(BaseModel):
    score: float
    risk_level: str
    detection_status: str
    primary_risk_signals: dict[str, Any]
    contributing_evidence_references: list[str]
    assessment_timestamp: str
    detector_version: str


class DiagnosisResponse(BaseModel):
    diagnosis_id: str
    cause_category: str
    confidence: float
    status: str
    supporting_signals: dict[str, Any]
    evidence_references: list[str]
    diagnostic_method: str
    timestamp: str


class RecoveryPredictionResponse(BaseModel):
    prediction_id: str
    recovery_probability: float
    confidence: float
    model_version: str
    feature_version: str
    prediction_timestamp: str
    contributing_features: dict[str, Any]
    prediction_status: str
    # Phase 12 — advisory only, MODEL ASSOCIATION not business causation; see
    # application/explainability.py and application/anomaly_detection.py.
    explanation: list[dict] | None = None
    confidence_tier: str | None = None
    anomaly: dict | None = None


class CanonicalRecoveryResponse(BaseModel):
    """Lineage B - the canonical recovery-probability model. Deliberately a separate
    response shape from RecoveryPredictionResponse (Lineage A, payment-failure-risk
    inverted). See docs/ai_reconstruction/MODEL_LINEAGE_RECONCILIATION.md."""

    status: str  # SUCCESS | NO_MODEL | ERROR
    model_task: str = "RECOVERY_PROBABILITY"
    prediction_task: str = "RECOVERY_PROBABILITY"
    target: str = "recovery_success"
    model_id: str | None = None
    model_version: str | None = None
    calibrated: bool | None = None
    raw_probability: float | None = None
    calibrated_probability: float | None = None
    calibrated_recovery_probability: float | None = None
    expected_recoverable_value: float | None = None
    feature_coverage: float | None = None
    feature_coverage_label: str | None = None
    feature_vector: dict[str, Any] | None = None
    feature_provenance: dict[str, Any] | None = None
    features_used: dict[str, Any] | None = None
    prediction_timestamp: str | None = None
    calibration_basis: str | None = None
    integrity_status: str | None = None  # VERIFIED | REFUSED
    integrity_failure: dict[str, Any] | None = None
    artifact_sha256: str | None = None
    approximations: list[str] | None = None
    is_production_decision: bool = False  # this path is analytical, read-only
    note: str | None = None


class ShadowERVResponse(BaseModel):
    """READ-ONLY comparison of the current production ERV with the canonical recovery
    ERV. Never drives an action; the production ERV path is unchanged."""

    case_id: str
    amount_at_risk: float
    current_probability: float | None
    current_probability_source: str | None
    current_erv: float | None
    current_erv_probability: float | None = None  # the recommendation candidate's own estimate
    canonical_status: str
    canonical_probability: float | None
    canonical_model_id: str | None
    canonical_erv: float | None
    difference_erv: float | None
    feature_coverage_label: str | None
    canonical_model_version: str | None = None
    canonical_calibrated: bool | None = None
    canonical_calibration_basis: str | None = None
    canonical_integrity_status: str | None = None
    prediction_timestamp: str | None = None
    approximations: list[str] | None = None
    production_decision_source: str = "current_erv (unchanged)"
    shadow_only: bool = True
    note: str


class CandidateActionResponse(BaseModel):
    action_type: str
    estimated_probability: float
    expected_recoverable_value: Decimal
    rationale: str


class ActionRecommendationResponse(BaseModel):
    recommendation_id: str
    candidates: list[CandidateActionResponse]
    top_candidate: CandidateActionResponse | None
    status: str
    rationale: str
    engine_version: str
    timestamp: str


class RuleResultResponse(BaseModel):
    rule_name: str
    passed: bool
    details: str


class PolicyDecisionResponse(BaseModel):
    decision_id: str
    status: str
    policy_version: str
    rules_evaluated: list[RuleResultResponse]
    failed_rules: list[RuleResultResponse]
    reason: str
    timestamp: str
    contact_permission: dict[str, Any] | None = None


class ExecutionRecordResponse(BaseModel):
    execution_id: str
    action_type: str
    agent_type: str
    policy_decision_id: str
    status: str
    adapter_used: str
    timestamp: str
    result_metadata: dict
    # Phase 16: the journey this execution belongs to, and its own operation id.
    trace_id: str | None = None
    operation_id: str | None = None


class RecoveryOutcomeResponse(BaseModel):
    outcome_id: str
    status: str
    expected_amount: Decimal
    actual_amount_recovered: Decimal
    currency: str
    verification_source: str
    external_reference: str
    reconciliation_status: str
    timestamp: str
    verification_basis: str | None = None
    action_result: str | None = None
    outstanding_amount: Decimal | None = None
    cancelled_amount: Decimal | None = None
    applied_outcome_event_ids: list[str] = []
    # Phase 8B (Step 14): the provenance of the outcome event(s) that produced this
    # figure - "test", "external_source_simulated", "human_confirmed", etc, or "mixed"
    # if more than one kind contributed, or None if nothing has been applied yet. Lets
    # the UI show operators that a recovered amount is not (yet) real-world evidence,
    # without any redesign of the existing advisory framing.
    provenance: str | None = None


class VerifyOutcomeRequest(BaseModel):
    external_reference: str = ""


class CaseResponse(BaseModel):
    case_id: str
    customer_id: str
    risk_category: RiskCategory
    amount_at_risk: Decimal
    currency: str
    current_state: CaseState
    event_count: int
    risk_level: str | None = None
    cause_category: str | None = None
    recovery_probability: float | None = None
    expected_recoverable_value: Decimal | None = None
    recommended_action: str | None = None
    policy_status: str | None = None
    execution_status: str | None = None
    outcome_status: str | None = None
    actual_amount_recovered: Decimal | None = None


class AuditResponse(BaseModel):
    id: str
    case_id: str | None
    from_state: str | None
    to_state: str | None
    evidence: dict[str, Any]
    timestamp: str


class DashboardMetricsResponse(BaseModel):
    total_revenue_at_risk: Decimal
    active_cases: int
    high_critical_cases: int
    pending_human_review: int
    waiting_cases: int
    recovery_opportunities: Decimal
    verified_recovery: Decimal
    recovery_gap: Decimal
    # Phase 5: distinct from the above (never counted as "recovered"). outstanding is
    # still-owed money on obligations not yet fully settled or cancelled; cancelled is
    # the balance withdrawn at the source; unmatched_outcome_events is money that
    # arrived but could not be matched to any obligation - never silently added to
    # verified_recovery. See application/outcome_events.py.
    outstanding_amount: Decimal = 0
    cancelled_amount: Decimal = 0
    unmatched_outcome_events: int = 0
    unmatched_outcome_amount: Decimal = 0
    # RecoverChain performs no currency conversion. The totals above are plain sums
    # and are only meaningful when every case shares one currency; mixed_currency says
    # when they do not, and currency_breakdown gives the per-currency figures.
    currency_breakdown: dict[str, dict[str, Decimal]] = {}
    mixed_currency: bool = False


class HumanReviewRequest(BaseModel):
    decision: str
    note: str


class BatchIngestRequest(BaseModel):
    # Items are validated one by one against IngestEventRequest inside the route, so
    # one malformed item is REJECTED on its own instead of failing the whole batch.
    events: list[dict[str, Any]]
    auto_advance: bool = True


class BatchIngestItemResult(BaseModel):
    external_event_id: str
    case_id: str | None = None
    is_new_case: bool = False
    status: str  # success | ignored (duplicate) | rejected (invalid) | error
    current_state: str | None = None
    policy_status: str | None = None
    message: str = ""
    reasons: list[str] = []


class BatchIngestResponse(BaseModel):
    submitted: int
    ingested: int
    duplicates: int
    failed: int
    results: list[BatchIngestItemResult]
    rejected: int = 0
    ingestion_run_id: str | None = None


class AdvanceCaseResponse(BaseModel):
    case_id: str
    current_state: str
    risk_level: str | None = None
    cause_category: str | None = None
    recovery_probability: float | None = None
    prediction_status: str | None = None
    recommended_action: str | None = None
    expected_recoverable_value: Decimal | None = None
    policy_status: str | None = None
    policy_reason: str | None = None


class DatasetPredictionRequest(BaseModel):
    canonical_features: dict[str, Any]


class DatasetPredictionResponse(BaseModel):
    probability: float
    status: str
    model_metadata: dict[str, Any]


class GenerateCasesRequest(BaseModel):
    max_cases: int = 100


class GenerateCasesResponse(BaseModel):
    status: str
    cases_generated: int
    case_ids: list[str]
    counters: dict[str, int] | None = None
    ingestion_run_id: str | None = None
    truncated: bool = False  # the source held more rows than max_cases allowed
    rejected_samples: list[dict[str, Any]] = []
    domain_classification: dict[str, Any] | None = None  # how rows were assigned a domain


GenerateCasesResponse.model_rebuild()


class IngestionRunResponse(BaseModel):
    run_id: str
    source_type: str
    source_ref: str | None = None
    source_name: str | None = None
    status: str
    started_at: str
    completed_at: str | None = None
    rows_in_source: int | None = None
    rows_received: int
    rows_accepted: int
    rows_rejected: int
    rows_duplicate: int
    rows_not_at_risk: int
    rows_not_processed: int
    cases_created: int
    cases_updated: int
    rejection_counts: dict[str, int] = {}
    rejected_samples: list[dict[str, Any]] = []
    domain_classification: dict[str, Any] | None = None
    error_message: str | None = None


class EvaluationRunResponse(BaseModel):
    run_id: str
    data_class: str = "EVALUATION"
    status: str
    experiment_mode: str
    isolation: str
    started_at: str
    completed_at: str | None = None
    total_scenarios: int | None = None
    passed_scenarios: int | None = None
    error_message: str | None = None
    metrics: dict[str, Any] | None = None
