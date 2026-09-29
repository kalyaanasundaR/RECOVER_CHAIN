from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class Money(BaseModel):
    amount: Decimal
    currency: str = "INR"


class RiskCategory(str, Enum):
    FAILED_PAYMENT = "FAILED_PAYMENT"
    CHECKOUT_ABANDONMENT = "CHECKOUT_ABANDONMENT"
    FAILED_SUBSCRIPTION = "FAILED_SUBSCRIPTION"
    OVERDUE_INVOICE = "OVERDUE_INVOICE"
    BROKEN_PROMISE = "BROKEN_PROMISE"


class CaseState(str, Enum):
    DETECTED = "DETECTED"
    OPEN = "OPEN"
    DIAGNOSING = "DIAGNOSING"
    RECOMMENDING = "RECOMMENDING"
    POLICY_EVALUATED = "POLICY_EVALUATED"
    ASSESSED = "ASSESSED"
    ACTION_PROPOSED = "ACTION_PROPOSED"
    POLICY_REVIEW = "POLICY_REVIEW"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    WAITING = "WAITING"
    ESCALATED = "ESCALATED"
    EXECUTION_PREPARING = "EXECUTION_PREPARING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    FULLY_RECOVERED = "FULLY_RECOVERED"
    PARTIALLY_RECOVERED = "PARTIALLY_RECOVERED"
    CLOSED_NOT_RECOVERED = "CLOSED_NOT_RECOVERED"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    STOPPED = "STOPPED"
    # Phase 5: the obligation itself was withdrawn at the source (a voided invoice, a
    # cancelled subscription, a cancelled order) - distinct from CLOSED_NOT_RECOVERED
    # (which means recovery was attempted and failed) and from STOPPED (an internal
    # decision to stop trying). Terminal. See domain/lifecycle.py.
    CANCELLED = "CANCELLED"


class RevenueEvent(BaseModel):
    event_id: str
    customer_id: str
    risk_category: RiskCategory
    external_system: str
    external_event_id: str
    reference_id: str | None = None
    amount: Money
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_payload: dict
    # Provenance: set at the ingestion boundary (application/ingestion.py), persisted on
    # the event row. Optional so domain code and tests that build events directly
    # keep working.
    source_type: str | None = None
    source_ref: str | None = None
    ingestion_run_id: str | None = None
    # Why risk_category is what it is (domain.revenue_domains.DomainBasis). None for
    # events built directly by domain code and for events stored before Phase 4.
    domain_basis: str | None = None


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAssessment(BaseModel):
    score: float  # 0.0 to 1.0
    risk_level: RiskLevel
    detection_status: str
    primary_risk_signals: (
        dict  # Category-specific signals (e.g. {"failure_count": 3, "amount": 100})
    )
    contributing_evidence_references: list[str]  # e.g., ["evt_1", "evt_2"]
    assessment_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    detector_version: str = "deterministic-v1.0"
    confidence: float = 1.0


class DiagnosisStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class RootCauseCategory(str, Enum):
    # FAILED_PAYMENT
    NETWORK_FAILURE = "NETWORK_FAILURE"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    PAYMENT_METHOD_INVALID = "PAYMENT_METHOD_INVALID"
    # CHECKOUT_ABANDONMENT
    PAYMENT_FRICTION = "PAYMENT_FRICTION"
    # FAILED_SUBSCRIPTION
    MANDATE_FAILURE = "MANDATE_FAILURE"
    # OVERDUE_INVOICE
    UNRESOLVED_DISPUTE = "UNRESOLVED_DISPUTE"
    # BROKEN_PROMISE
    MISSED_COMMITMENT = "MISSED_COMMITMENT"
    # GENERAL
    UNKNOWN = "UNKNOWN"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class RootCauseDiagnosis(BaseModel):
    diagnosis_id: str
    cause_category: RootCauseCategory
    confidence: float
    status: DiagnosisStatus
    supporting_signals: dict
    evidence_references: list[str]
    diagnostic_method: str = "deterministic-v1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RecoveryPrediction(BaseModel):
    prediction_id: str
    recovery_probability: float  # 0.0 to 1.0
    confidence: float
    model_version: str
    feature_version: str
    prediction_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    contributing_features: dict
    prediction_status: str
    # Phase 12: MODEL ASSOCIATION, not business causation - see
    # application/explainability.py. Optional/defaulted so every pre-existing
    # constructor call (deterministic baseline, tests) keeps working unchanged.
    explanation: list[dict] | None = None
    # A documented-methodology confidence bucket (application/explainability.py
    # ::confidence_tier), never a fabricated statistical confidence interval.
    confidence_tier: str | None = None
    # Statistical distance from the model's own training distribution
    # (application/anomaly_detection.py) - separate from recovery_probability,
    # never influences ERV or the action recommendation.
    anomaly: dict | None = None


class ActionType(str, Enum):
    # FAILED_PAYMENT
    RETRY_PAYMENT = "RETRY_PAYMENT"
    REQUEST_PAYMENT_METHOD_UPDATE = "REQUEST_PAYMENT_METHOD_UPDATE"
    SEND_PAYMENT_REMINDER = "SEND_PAYMENT_REMINDER"
    # CHECKOUT_ABANDONMENT
    SEND_CHECKOUT_REMINDER = "SEND_CHECKOUT_REMINDER"
    OFFER_CHECKOUT_ASSISTANCE = "OFFER_CHECKOUT_ASSISTANCE"
    # FAILED_SUBSCRIPTION
    RETRY_BILLING = "RETRY_BILLING"
    SEND_SUBSCRIPTION_REMINDER = "SEND_SUBSCRIPTION_REMINDER"
    # OVERDUE_INVOICE
    SEND_INVOICE_REMINDER = "SEND_INVOICE_REMINDER"
    SEND_PAYMENT_LINK = "SEND_PAYMENT_LINK"
    ESCALATE_COLLECTION = "ESCALATE_COLLECTION"
    # BROKEN_PROMISE
    SEND_PROMISE_REMINDER = "SEND_PROMISE_REMINDER"
    REQUEST_NEW_COMMITMENT = "REQUEST_NEW_COMMITMENT"
    # GENERAL
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    NO_ACTION_POSSIBLE = "NO_ACTION_POSSIBLE"


class ActionChannel(str, Enum):
    """How an action reaches the customer. This is the single classification the
    Policy Engine rate-limits by and the execution ledger records, so the two can
    never drift apart."""

    PAYMENT = "payment"  # moves money: retry caps + cooldown apply
    COMMUNICATION = "communication"  # reaches the customer: contact-frequency cap applies
    INTERNAL = "internal"  # never leaves the system: no outreach to rate-limit


# Every action, stated explicitly: the channel it reaches the customer through and the
# recovery domains it is a valid response to. The Policy Engine rate-limits by channel
# and refuses an action outside its domains; the execution ledger and the simulated
# adapter read the same table. Nothing is inferred from an action's NAME - the old
# `"REMINDER" in action` test left five customer-facing actions unclassified.
_ALL_DOMAINS = frozenset(RiskCategory)
ACTION_SPECS: dict[ActionType, tuple[ActionChannel, frozenset[RiskCategory]]] = {
    # payment actions
    ActionType.RETRY_PAYMENT: (ActionChannel.PAYMENT, frozenset({RiskCategory.FAILED_PAYMENT})),
    ActionType.RETRY_BILLING: (
        ActionChannel.PAYMENT,
        frozenset({RiskCategory.FAILED_SUBSCRIPTION}),
    ),
    # communication actions, by the domain they belong to
    ActionType.SEND_PAYMENT_REMINDER: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.FAILED_PAYMENT}),
    ),
    ActionType.REQUEST_PAYMENT_METHOD_UPDATE: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.FAILED_PAYMENT, RiskCategory.FAILED_SUBSCRIPTION}),
    ),
    ActionType.SEND_CHECKOUT_REMINDER: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.CHECKOUT_ABANDONMENT}),
    ),
    ActionType.OFFER_CHECKOUT_ASSISTANCE: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.CHECKOUT_ABANDONMENT}),
    ),
    ActionType.SEND_SUBSCRIPTION_REMINDER: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.FAILED_SUBSCRIPTION}),
    ),
    ActionType.SEND_INVOICE_REMINDER: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.OVERDUE_INVOICE}),
    ),
    ActionType.SEND_PAYMENT_LINK: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.OVERDUE_INVOICE}),
    ),
    ActionType.ESCALATE_COLLECTION: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.OVERDUE_INVOICE, RiskCategory.BROKEN_PROMISE}),
    ),
    ActionType.SEND_PROMISE_REMINDER: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.BROKEN_PROMISE}),
    ),
    ActionType.REQUEST_NEW_COMMITMENT: (
        ActionChannel.COMMUNICATION,
        frozenset({RiskCategory.BROKEN_PROMISE}),
    ),
    # internal actions: never leave the system
    ActionType.ESCALATE_TO_HUMAN: (ActionChannel.INTERNAL, _ALL_DOMAINS),
    ActionType.NO_ACTION_POSSIBLE: (ActionChannel.INTERNAL, _ALL_DOMAINS),
}


def action_channel(action) -> ActionChannel:
    """Classify an action. Accepts an ActionType or its string value."""
    try:
        return ACTION_SPECS[ActionType(action)][0]
    except (ValueError, KeyError):
        # Unrecognised action: treat as customer-facing so it is rate-limited.
        return ActionChannel.COMMUNICATION


def action_domains(action) -> frozenset[RiskCategory]:
    """The recovery domains an action is valid for; empty for an unrecognised action."""
    try:
        return ACTION_SPECS[ActionType(action)][1]
    except (ValueError, KeyError):
        return frozenset()


def action_applies_to(action, category) -> bool:
    return RiskCategory(category) in action_domains(action)


class CandidateAction(BaseModel):
    action_type: ActionType
    estimated_probability: float
    expected_recoverable_value: float
    rationale: str


class RecommendationStatus(str, Enum):
    RECOMMENDED = "RECOMMENDED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ActionRecommendation(BaseModel):
    recommendation_id: str
    candidates: list[CandidateAction]
    top_candidate: CandidateAction | None
    status: RecommendationStatus
    rationale: str
    engine_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PolicyDecisionStatus(str, Enum):
    PERMITTED = "PERMITTED"
    DENIED = "DENIED"
    WAIT = "WAIT"
    ESCALATE = "ESCALATE"


class RuleResult(BaseModel):
    rule_name: str
    passed: bool
    details: str


class PolicyDecision(BaseModel):
    decision_id: str
    status: PolicyDecisionStatus
    policy_version: str
    rules_evaluated: list[RuleResult]
    failed_rules: list[RuleResult]
    reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # The contact permission this decision evaluated for the action's channel:
    # {channel, status, record_id, unknown_consent_behaviour}. None when the action is
    # internal or no durable context was supplied.
    contact_permission: dict | None = None
    # Phase 16 traceability. Set by DeterministicPolicyEngine.evaluate from the case (all
    # optional so pre-Phase-16 stored decisions still load). `reason_codes` are the
    # names of the rules that failed (or ALL_RULES_PASSED); references, not a case copy.
    trace_id: str | None = None
    case_id: str | None = None
    action: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    approval_required: bool = False
    lifecycle_state: str | None = None


class RecoveryOutcomeStatus(str, Enum):
    FULLY_RECOVERED = "FULLY_RECOVERED"
    PARTIALLY_RECOVERED = "PARTIALLY_RECOVERED"
    NOT_RECOVERED = "NOT_RECOVERED"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    # Phase 5: the obligation was withdrawn at the source. Distinct from NOT_RECOVERED
    # (which reports a settlement source that confirmed nothing came back).
    CANCELLED = "CANCELLED"


class RecoveryOutcome(BaseModel):
    outcome_id: str
    case_id: str
    execution_id: str | None = None
    status: RecoveryOutcomeStatus
    expected_amount: Money
    actual_amount_recovered: Money
    verification_source: str
    external_reference: str
    reconciliation_status: str
    verification_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # What the settlement source had to confirm for this domain
    # (DomainSpec.verification_basis), the ACTION RESULT at verification time (what the
    # adapter reported for the action - never itself recovery), and what is still owed.
    verification_basis: str | None = None
    action_result: str | None = None
    outstanding_amount: Money | None = None
    # Phase 5: the position derived from every authoritative outcome event applied to
    # this case (application/outcome_events.py), decomposing actual_amount_recovered.
    # cancelled_amount is the balance withdrawn at the source (never "recovered").
    cancelled_amount: Money | None = None
    # Ids of the OutcomeEventModel rows this outcome was (re)computed from, in
    # application order. Empty for an outcome that predates Phase 5 outcome events
    # (legacy simulated verification) or that reports "no outcome yet".
    applied_outcome_event_ids: list[str] = Field(default_factory=list)
    # Phase 8B: the provenance of the outcome event(s) behind this figure - "test",
    # "external_source_simulated", "human_confirmed", etc, "mixed" if more than one
    # kind contributed, or None for an outcome that predates Phase 8A provenance
    # tracking. Never assume this is "production" just because it is unset.
    provenance: str | None = None


class ExecutionStatus(str, Enum):
    PREPARED = "PREPARED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED_SIMULATED = "COMPLETED_SIMULATED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class ExecutionRecord(BaseModel):
    execution_id: str
    action_type: ActionType
    agent_type: str
    policy_decision_id: str
    policy_version: str
    parameters: dict
    idempotency_key: str
    status: ExecutionStatus
    adapter_used: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    result_metadata: dict = Field(default_factory=dict)
    # Phase 16: the journey this attempt belongs to (`trace_id`) and the attempt's own
    # identity (`operation_id` == execution_id). Optional so old records still load.
    trace_id: str | None = None
    case_id: str | None = None
    operation_id: str | None = None


class RecoveryCase(BaseModel):
    case_id: str
    customer_id: str
    risk_category: RiskCategory
    reference_id: str | None = None
    amount_at_risk: Money
    expected_recoverable_value: Money | None = None
    actual_amount_recovered: Money | None = None
    current_state: CaseState = CaseState.DETECTED
    linked_events: list[RevenueEvent] = Field(default_factory=list)
    risk_assessment: RiskAssessment | None = None
    diagnosis: RootCauseDiagnosis | None = None
    prediction: RecoveryPrediction | None = None
    recommendation: ActionRecommendation | None = None
    policy_decision: PolicyDecision | None = None
    execution_record: ExecutionRecord | None = None
    candidate_action: CandidateAction | None = None
    outcome: RecoveryOutcome | None = None
    # Phase 5: the case's current financial position, maintained by
    # application/outcome_events.py.reconcile_case from the append-only outcome_events
    # table - never recomputed by reprocessing the pipeline. All three start at 0 /
    # amount_at_risk and only move by an applied outcome event. Persisted on CaseModel
    # (infrastructure/orm.py) as a materialised cache of that table for cheap reads;
    # outcome_events remains the source of truth and audit trail.
    recovered_amount: Money | None = None
    outstanding_amount: Money | None = None
    cancelled_amount: Money | None = None
