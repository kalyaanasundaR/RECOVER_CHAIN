"""Phase 15 — the bounded agent contract.

Everything a specialist agent may receive, return, call, or do is defined here as
a closed, machine-checkable structure. Nothing an agent produces is ever parsed
from free text into a command: `validate_agent_output` accepts only a dict (or a
JSON *object* string) that satisfies `AgentRecommendation` exactly - unknown
fields (a smuggled `tool_calls`, `amount_at_risk`, `human_approved`,
`recovered_amount`...) are rejected, not ignored.

What is deliberately NOT here: any field an agent could use to change money or
authorization. `AgentRecommendation` has no amount-at-risk, no recovered amount, no
approval flag, no policy status. The Policy Engine (application/policy_engine.py)
remains the only authority; this module only defines what may be *proposed*.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from domain.models import (
    ActionChannel,
    ActionType,
    PolicyDecisionStatus,
    RiskCategory,
    action_applies_to,
    action_channel,
)

AGENT_CONTRACT_VERSION = "agent-contract-v1"


def new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------- structured output


class AgentStatus(str, Enum):
    RECOMMENDED = "RECOMMENDED"
    NO_ACTION = "NO_ACTION"  # e.g. obligation already settled - nothing to propose
    NEEDS_MAPPING = "NEEDS_MAPPING"  # no specialist for this domain
    FALLBACK = "FALLBACK"  # the agent failed/disagreed; deterministic result stands


class ReasonCode(str, Enum):
    """Closed vocabulary - the agent picks from this list, it does not invent
    reasons. Each code is derived from a real field (see specialist_agents)."""

    CAUSE_DIAGNOSED = "CAUSE_DIAGNOSED"
    CAUSE_UNKNOWN = "CAUSE_UNKNOWN"
    HIGH_RECOVERY_PROBABILITY = "HIGH_RECOVERY_PROBABILITY"
    LOW_RECOVERY_PROBABILITY = "LOW_RECOVERY_PROBABILITY"
    ANOMALY_SIGNAL_PRESENT = "ANOMALY_SIGNAL_PRESENT"
    HIGHEST_ERV_CANDIDATE = "HIGHEST_ERV_CANDIDATE"
    ONLY_APPLICABLE_CANDIDATE = "ONLY_APPLICABLE_CANDIDATE"
    HUMAN_REVIEW_FALLBACK = "HUMAN_REVIEW_FALLBACK"
    OBLIGATION_ALREADY_CLOSED = "OBLIGATION_ALREADY_CLOSED"
    PRIOR_ACTION_ON_RECORD = "PRIOR_ACTION_ON_RECORD"


class PolicyPrerequisite(str, Enum):
    """Checks the Policy Engine WILL apply to the proposed action's channel. Listed
    so a reader can see what stands between a recommendation and execution; the
    agent does not evaluate any of them."""

    CONTACT_PERMISSION_PAYMENT = "CONTACT_PERMISSION_PAYMENT"
    CONTACT_PERMISSION_COMMUNICATION = "CONTACT_PERMISSION_COMMUNICATION"
    PAYMENT_RETRY_LIMIT = "PAYMENT_RETRY_LIMIT"
    PAYMENT_COOLDOWN = "PAYMENT_COOLDOWN"
    COMMUNICATION_CONTACT_LIMIT = "COMMUNICATION_CONTACT_LIMIT"
    ACTION_DOMAIN_APPLICABILITY = "ACTION_DOMAIN_APPLICABILITY"
    AUTOMATED_AMOUNT_LIMIT_OR_HUMAN_APPROVAL = "AUTOMATED_AMOUNT_LIMIT_OR_HUMAN_APPROVAL"
    OBLIGATION_NOT_SETTLED = "OBLIGATION_NOT_SETTLED"


def prerequisites_for(action: ActionType) -> list[PolicyPrerequisite]:
    ch = action_channel(action)
    common = [
        PolicyPrerequisite.ACTION_DOMAIN_APPLICABILITY,
        PolicyPrerequisite.AUTOMATED_AMOUNT_LIMIT_OR_HUMAN_APPROVAL,
        PolicyPrerequisite.OBLIGATION_NOT_SETTLED,
    ]
    if ch == ActionChannel.PAYMENT:
        return [
            PolicyPrerequisite.CONTACT_PERMISSION_PAYMENT,
            PolicyPrerequisite.PAYMENT_RETRY_LIMIT,
            PolicyPrerequisite.PAYMENT_COOLDOWN,
            *common,
        ]
    if ch == ActionChannel.COMMUNICATION:
        return [
            PolicyPrerequisite.CONTACT_PERMISSION_COMMUNICATION,
            PolicyPrerequisite.COMMUNICATION_CONTACT_LIMIT,
            *common,
        ]
    return common  # internal actions reach nobody


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateRef(_Strict):
    action_type: ActionType
    estimated_probability: float = Field(ge=0.0, le=1.0)
    expected_recoverable_value: float = Field(ge=0.0)

    @field_validator("estimated_probability", "expected_recoverable_value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class ModelReference(_Strict):
    model_version: str
    prediction_status: str
    prediction_id: str | None = None
    confidence_tier: str | None = None


class AnomalyInfo(_Strict):
    level: str
    max_abs_z: float | None = None


class ExplanationItem(_Strict):
    """MODEL ASSOCIATION only (application/explainability.py). Never a causal claim."""

    feature: str
    direction: Literal["increased", "decreased", "no_effect", "UNKNOWN"]
    kind: Literal["MODEL_ASSOCIATION"] = "MODEL_ASSOCIATION"


class AgentRecommendation(_Strict):
    agent_id: str
    agent_version: str
    contract_version: str = AGENT_CONTRACT_VERSION
    case_id: str
    domain: RiskCategory | None  # None only for NEEDS_MAPPING (no recognised domain)
    domain_raw: str | None = None
    status: AgentStatus
    candidate_actions: list[CandidateRef] = Field(default_factory=list)
    recommended_action: ActionType | None = None
    model_reference: ModelReference | None = None
    recovery_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_recoverable_value: float | None = Field(default=None, ge=0.0)
    anomaly: AnomalyInfo | None = None
    explanation: list[ExplanationItem] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    # Display text only. Never parsed, never executed; capped so it cannot carry a payload.
    rationale: str = Field(default="", max_length=500)
    policy_prerequisites: list[PolicyPrerequisite] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str

    @field_validator("recovery_probability", "expected_recoverable_value")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("must be finite")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> AgentRecommendation:
        if self.domain is None and (self.candidate_actions or self.recommended_action):
            raise ValueError(
                "no candidate or recommended action is allowed without a recognised domain"
            )
        for c in self.candidate_actions:
            if not action_applies_to(c.action_type, self.domain):
                raise ValueError(
                    f"{c.action_type.value} is not a recovery action for {self.domain.value}"
                )
        if self.recommended_action is not None:
            if self.recommended_action not in {c.action_type for c in self.candidate_actions}:
                raise ValueError("recommended_action must be one of candidate_actions")
        if self.status == AgentStatus.RECOMMENDED and self.recommended_action is None:
            raise ValueError("RECOMMENDED status requires a recommended_action")
        return self


class AgentOutputRejected(ValueError):
    """Structured-output validation failed. The caller falls back to the
    deterministic recommendation; nothing from the rejected output is used."""


def validate_agent_output(
    raw: Any, *, expected_domain: RiskCategory | None = None
) -> AgentRecommendation:
    """The only door from agent output to the rest of the system. Accepts an
    AgentRecommendation, a dict, or a JSON string whose top level is an OBJECT.
    Plain prose (`"Please retry the payment"`) is rejected - never string-parsed
    into an action."""
    if isinstance(raw, AgentRecommendation):
        rec = raw
    else:
        if isinstance(raw, (str, bytes)):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError) as e:
                raise AgentOutputRejected(
                    "agent output is not valid JSON (free text is not accepted)"
                ) from e
        if not isinstance(raw, dict):
            raise AgentOutputRejected("agent output must be a JSON object")
        try:
            rec = AgentRecommendation.model_validate(raw)
        except ValidationError as e:
            raise AgentOutputRejected(f"schema violation: {e.errors()[0]['msg']}") from e
    if expected_domain is not None and rec.domain != expected_domain:
        raise AgentOutputRejected(
            f"agent output domain {getattr(rec.domain, 'value', rec.domain)} != "
            f"case domain {expected_domain.value}"
        )
    return rec


# ---------------------------------------------------------------- tool boundaries


class ToolClass(str, Enum):
    READ_ONLY = "READ_ONLY"
    RECOMMENDATION = "RECOMMENDATION"
    POLICY = "POLICY"
    EXECUTION = "EXECUTION"
    OUTCOME = "OUTCOME"


AGENT_PERMITTED_TOOL_CLASSES = frozenset({ToolClass.READ_ONLY, ToolClass.RECOMMENDATION})

# The complete tool inventory. Classification is the enforcement point: an agent's
# toolbox refuses anything outside AGENT_PERMITTED_TOOL_CLASSES, so POLICY / EXECUTION
# / OUTCOME tools exist only for the deterministic orchestration layer.
TOOL_REGISTRY: dict[str, ToolClass] = {
    "get_decision_context": ToolClass.READ_ONLY,
    "get_case_summary": ToolClass.READ_ONLY,
    "get_prediction": ToolClass.READ_ONLY,
    "get_anomaly": ToolClass.READ_ONLY,
    "request_context": ToolClass.READ_ONLY,  # structured handoff to the context layer
    "generate_candidate_actions": ToolClass.RECOMMENDATION,
    "emit_recommendation": ToolClass.RECOMMENDATION,
    "evaluate_policy": ToolClass.POLICY,  # DeterministicPolicyEngine.evaluate
    "record_human_review": ToolClass.POLICY,  # POST /cases/{id}/human-review
    "dispatch_execution": ToolClass.EXECUTION,  # AgentOrchestrator.execute
    "ingest_outcome_event": ToolClass.OUTCOME,  # application/outcome_events.py
}


class ToolDenied(PermissionError):
    pass


class AgentToolbox:
    """What one agent may call. Bound to a deep COPY of the case, so even a
    misbehaving read-only tool cannot mutate the operational record."""

    def __init__(
        self, agent_id: str, case, decision_context: dict, policy_signals: dict | None = None
    ):
        self.agent_id = agent_id
        self._case = case.model_copy(deep=True)
        self._ctx = dict(decision_context)
        self._signals = dict(policy_signals or {})
        self.calls: list[str] = []

    def call(self, tool_name: str, *args):
        cls = TOOL_REGISTRY.get(tool_name)
        if cls is None:
            raise ToolDenied(f"unknown tool {tool_name!r} - arbitrary tool calls are not accepted")
        if cls not in AGENT_PERMITTED_TOOL_CLASSES:
            raise ToolDenied(
                f"tool {tool_name!r} is {cls.value}; agents may only use READ_ONLY/RECOMMENDATION"
            )
        self.calls.append(tool_name)
        if tool_name == "get_decision_context":
            return dict(self._ctx)
        if tool_name == "get_case_summary":
            return {
                "case_id": self._case.case_id,
                "domain": self._case.risk_category.value,
                "obligation_state": self._case.current_state.value,
            }
        if tool_name == "get_prediction":
            return self._case.prediction.model_copy(deep=True) if self._case.prediction else None
        if tool_name == "get_anomaly":
            return (self._case.prediction.anomaly if self._case.prediction else None) or None
        if tool_name == "request_context":
            return dict(self._signals)
        if tool_name == "generate_candidate_actions":
            from application.action_evaluator import DeterministicActionEvaluator

            return self._case.recommendation or DeterministicActionEvaluator().evaluate(self._case)
        if tool_name == "emit_recommendation":
            return validate_agent_output(args[0], expected_domain=self._case.risk_category)
        raise ToolDenied(f"tool {tool_name!r} has no agent-side implementation")


# ---------------------------------------------------------------- state machine


class RunState(str, Enum):
    RECEIVED = "RECEIVED"
    CONTEXT_BUILT = "CONTEXT_BUILT"
    ANALYZED = "ANALYZED"
    RECOMMENDATION_READY = "RECOMMENDATION_READY"
    POLICY_REVIEW = "POLICY_REVIEW"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_MAPPING = "NEEDS_MAPPING"


TERMINAL_STATES = frozenset(
    {RunState.COMPLETED, RunState.REJECTED, RunState.FAILED, RunState.NEEDS_MAPPING}
)

_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.RECEIVED: frozenset({RunState.CONTEXT_BUILT, RunState.NEEDS_MAPPING, RunState.FAILED}),
    RunState.CONTEXT_BUILT: frozenset({RunState.ANALYZED, RunState.NEEDS_MAPPING, RunState.FAILED}),
    RunState.ANALYZED: frozenset({RunState.RECOMMENDATION_READY, RunState.FAILED}),
    RunState.RECOMMENDATION_READY: frozenset({RunState.POLICY_REVIEW, RunState.FAILED}),
    # POLICY_REVIEW -> APPROVED / APPROVAL_REQUIRED / REJECTED happen ONLY through
    # AgentRun.apply_policy_decision (a real PolicyDecision), never advance().
    RunState.POLICY_REVIEW: frozenset(
        {RunState.APPROVED, RunState.APPROVAL_REQUIRED, RunState.REJECTED}
    ),
    RunState.APPROVAL_REQUIRED: frozenset({RunState.APPROVED, RunState.REJECTED}),
    RunState.APPROVED: frozenset({RunState.EXECUTING, RunState.FAILED}),
    RunState.EXECUTING: frozenset({RunState.VERIFYING, RunState.FAILED}),
    RunState.VERIFYING: frozenset({RunState.COMPLETED, RunState.FAILED}),
}

_POLICY_GATED = frozenset({RunState.APPROVED, RunState.APPROVAL_REQUIRED, RunState.REJECTED})


class IllegalTransition(RuntimeError):
    pass


class AgentRun:
    """Bounded orchestration state for one case. Authorization states can only be
    entered by applying a real Policy Engine decision; terminal states stay terminal."""

    def __init__(self, case_id: str, trace_id: str | None = None):
        self.case_id = case_id
        self.trace_id = trace_id or new_trace_id()
        self.operation_id: str | None = None  # this run's own id; trace_id is the journey
        self.state = RunState.RECEIVED
        self.history: list[tuple[str, str]] = [(self.state.value, datetime.now(UTC).isoformat())]

    def _move(self, to: RunState) -> None:
        if self.state in TERMINAL_STATES:
            raise IllegalTransition(f"{self.state.value} is terminal")
        if to not in _TRANSITIONS.get(self.state, frozenset()):
            raise IllegalTransition(f"{self.state.value} -> {to.value} is not permitted")
        self.state = to
        self.history.append((to.value, datetime.now(UTC).isoformat()))

    def advance(self, to: RunState) -> None:
        if to in _POLICY_GATED:
            raise IllegalTransition(
                f"{to.value} can only be entered through apply_policy_decision - an agent cannot authorize itself"
            )
        self._move(to)

    def apply_policy_decision(self, status: PolicyDecisionStatus) -> None:
        if self.state != RunState.POLICY_REVIEW:
            raise IllegalTransition("a policy decision can only be applied from POLICY_REVIEW")
        target = {
            PolicyDecisionStatus.PERMITTED: RunState.APPROVED,
            PolicyDecisionStatus.ESCALATE: RunState.APPROVAL_REQUIRED,
            PolicyDecisionStatus.DENIED: RunState.REJECTED,
            PolicyDecisionStatus.WAIT: RunState.REJECTED,  # not now; a later run may retry
        }[status]
        self._move(target)

    def apply_human_review(self, revalidated_status: PolicyDecisionStatus) -> None:
        """Human approval never authorizes by itself: it only re-enters the flow if the
        Policy Engine's fresh revalidation of the approved decision is PERMITTED."""
        if self.state != RunState.APPROVAL_REQUIRED:
            raise IllegalTransition("human review applies only from APPROVAL_REQUIRED")
        self._move(
            RunState.APPROVED
            if revalidated_status == PolicyDecisionStatus.PERMITTED
            else RunState.REJECTED
        )


# ---------------------------------------------------------------- handoffs


class CustomerContextRequest(_Strict):
    kind: Literal["CUSTOMER_CONTEXT_REQUEST"] = "CUSTOMER_CONTEXT_REQUEST"
    fields: list[Literal["retry_history", "contact_history", "permission_status"]]


HANDOFF_PAYLOAD_SCHEMAS: dict[str, type[BaseModel]] = {
    "CUSTOMER_CONTEXT_REQUEST": CustomerContextRequest,
}
HANDOFF_DESTINATIONS = frozenset(
    {"CONTEXT_LAYER"}
)  # agents never hand off to another agent for authority


class Handoff(_Strict):
    source_agent: str
    destination: str
    case_id: str
    reason: str = Field(max_length=200)
    payload_schema: str
    payload: dict
    trace_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # A handoff transfers information, never authority.
    grants_authorization: Literal[False] = False

    def to_audit_evidence(self) -> dict:
        return {
            "action": "agent_handoff",
            "source_agent": self.source_agent,
            "destination": self.destination,
            "reason": self.reason,
            "payload_schema": self.payload_schema,
            "trace_id": self.trace_id,
            "grants_authorization": False,
        }


def make_handoff(
    *,
    source_agent: str,
    destination: str,
    case_id: str,
    reason: str,
    payload_schema: str,
    payload: dict,
    trace_id: str,
) -> Handoff:
    if destination not in HANDOFF_DESTINATIONS:
        raise AgentOutputRejected(f"handoff destination {destination!r} is not permitted")
    schema = HANDOFF_PAYLOAD_SCHEMAS.get(payload_schema)
    if schema is None:
        raise AgentOutputRejected(f"unknown handoff payload schema {payload_schema!r}")
    try:
        validated = schema.model_validate(payload)
    except ValidationError as e:
        raise AgentOutputRejected(
            f"handoff payload violates {payload_schema}: {e.errors()[0]['msg']}"
        ) from e
    return Handoff(
        source_agent=source_agent,
        destination=destination,
        case_id=case_id,
        reason=reason,
        payload_schema=payload_schema,
        payload=validated.model_dump(),
        trace_id=trace_id,
    )
