"""Phase 15 — agent evaluation contract and deterministic benchmark.

`AgentEvalRecord` is the row a future, real evaluation would accumulate per agent
decision. `run_agent_benchmark()` fills it from DETERMINISTIC SYNTHETIC scenarios
(hand-built cases run through the real router, real specialist agents, real Policy
Engine and the real execution orchestrator with the simulated adapter). Every record
is tagged `provenance="SYNTHETIC_SCENARIO"`: these numbers measure the software's
wiring under scripted inputs, never production agent performance, and no recovery
is ever produced - `authoritative_outcome` is always `NOT_APPLICABLE_SYNTHETIC`.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from application.action_evaluator import DeterministicActionEvaluator
from application.agent_contract import AgentStatus
from application.agents import AgentOrchestrator
from application.decision_context import build_decision_context
from application.policy_engine import DeterministicPolicyEngine, PolicyContext
from application.specialist_agents import route_case, run_specialist
from domain.models import (
    ActionType,
    CaseState,
    DiagnosisStatus,
    Money,
    RecoveryCase,
    RecoveryPrediction,
    RevenueEvent,
    RiskCategory,
    RootCauseCategory,
    RootCauseDiagnosis,
    action_applies_to,
)

DECISION_CONTEXT_VERSION = "decision-context-v1"
SYNTHETIC = "SYNTHETIC_SCENARIO"


@dataclass
class AgentEvalRecord:
    scenario: str
    case_id: str
    domain: str
    input_context_version: str
    agent_id: str | None
    agent_version: str | None
    model_version: str | None
    recommendation: str | None
    candidate_actions: list[str]
    policy_result: str
    execution_result: str
    authoritative_outcome: str
    latency_ms: float
    errors: list[str]
    provenance: str
    expected_agent: str
    expected_policy: str
    routing_correct: bool = False
    output_valid: bool = False
    policy_as_expected: bool = False
    unsafe: bool = False
    execution_blocked: bool = False


@dataclass
class Scenario:
    name: str
    domain: RiskCategory
    expected_agent: str
    expected_policy: str
    amount: float = 300.0
    cause: RootCauseCategory = RootCauseCategory.UNKNOWN
    probability: float = 0.6
    state: CaseState = CaseState.DETECTED
    recovered: float = 0.0
    policy: dict = field(default_factory=dict)


_AGENT = {
    RiskCategory.FAILED_PAYMENT: "PaymentAgent",
    RiskCategory.CHECKOUT_ABANDONMENT: "CheckoutAgent",
    RiskCategory.FAILED_SUBSCRIPTION: "SubscriptionAgent",
    RiskCategory.OVERDUE_INVOICE: "InvoiceAgent",
    RiskCategory.BROKEN_PROMISE: "PromiseToPayAgent",
}
P, C, S, I, B = (
    RiskCategory.FAILED_PAYMENT,
    RiskCategory.CHECKOUT_ABANDONMENT,
    RiskCategory.FAILED_SUBSCRIPTION,
    RiskCategory.OVERDUE_INVOICE,
    RiskCategory.BROKEN_PROMISE,
)


def _s(name, domain, policy_expected, **kw) -> Scenario:
    return Scenario(name, domain, _AGENT[domain], policy_expected, **kw)


SCENARIOS: list[Scenario] = [
    # Payment
    _s(
        "payment/recoverable_failure",
        P,
        "PERMITTED",
        cause=RootCauseCategory.NETWORK_FAILURE,
        probability=0.9,
    ),
    _s(
        "payment/repeated_failure",
        P,
        "DENIED",
        cause=RootCauseCategory.NETWORK_FAILURE,
        policy={"prior_payment_attempts": 3},
    ),
    _s(
        "payment/suppression",
        P,
        "DENIED",
        cause=RootCauseCategory.NETWORK_FAILURE,
        policy={"payment_permission": "SUPPRESSED"},
    ),
    _s(
        "payment/consent_denied",
        P,
        "DENIED",
        cause=RootCauseCategory.NETWORK_FAILURE,
        policy={"payment_permission": "OPTED_OUT"},
    ),
    _s(
        "payment/high_value_approval",
        P,
        "ESCALATE",
        cause=RootCauseCategory.NETWORK_FAILURE,
        amount=20000.0,
    ),
    # Checkout
    _s("checkout/abandonment", C, "PERMITTED", cause=RootCauseCategory.PAYMENT_FRICTION),
    _s(
        "checkout/repeated_abandonment",
        C,
        "WAIT",
        cause=RootCauseCategory.PAYMENT_FRICTION,
        policy={"comms_in_last_24h": 2},
    ),
    _s(
        "checkout/expired_session", C, "PERMITTED", cause=RootCauseCategory.UNKNOWN, probability=0.2
    ),
    # Subscription
    _s("subscription/billing_failure", S, "PERMITTED", cause=RootCauseCategory.UNKNOWN),
    _s(
        "subscription/retry_exhaustion",
        S,
        "DENIED",
        cause=RootCauseCategory.UNKNOWN,
        policy={"prior_payment_attempts": 3},
    ),
    _s(
        "subscription/cancelled",
        S,
        "DENIED",
        cause=RootCauseCategory.MANDATE_FAILURE,
        state=CaseState.CANCELLED,
    ),
    # Invoice
    _s("invoice/overdue", I, "PERMITTED", cause=RootCauseCategory.UNKNOWN),
    _s(
        "invoice/partial_settlement",
        I,
        "PERMITTED",
        amount=1000.0,
        recovered=400.0,
        state=CaseState.PARTIALLY_RECOVERED,
    ),
    _s("invoice/high_value", I, "ESCALATE", amount=20000.0),
    _s("invoice/escalation_dispute", I, "ESCALATE", cause=RootCauseCategory.UNRESOLVED_DISPUTE),
    # Promise-to-pay (lifecycle states created/due/fulfilled/broken are outcome-event kinds,
    # not distinct case states - scenarios approximate them with what a case can express)
    _s("promise/created_no_break_evidence", B, "PERMITTED", cause=RootCauseCategory.UNKNOWN),
    _s("promise/due", B, "PERMITTED", cause=RootCauseCategory.UNKNOWN, probability=0.5),
    _s("promise/fulfilled", B, "DENIED", state=CaseState.FULLY_RECOVERED, recovered=300.0),
    _s("promise/broken", B, "PERMITTED", cause=RootCauseCategory.MISSED_COMMITMENT),
]


def build_case(sc: Scenario, idx: int) -> RecoveryCase:
    now = datetime.now(UTC)
    amount = Decimal(str(sc.amount))
    case = RecoveryCase(
        case_id=f"eval_{idx:02d}_{sc.name.replace('/', '_')}",
        customer_id=f"eval_cust_{idx:02d}",
        risk_category=sc.domain,
        reference_id=f"eval_ref_{idx:02d}",
        amount_at_risk=Money(amount=amount, currency="INR"),
        current_state=sc.state,
        linked_events=[
            RevenueEvent(
                event_id=f"eval_evt_{idx:02d}",
                customer_id=f"eval_cust_{idx:02d}",
                risk_category=sc.domain,
                external_system="agent_eval",
                external_event_id=f"eval_ext_{idx:02d}",
                reference_id=f"eval_ref_{idx:02d}",
                amount=Money(amount=amount, currency="INR"),
                timestamp=now - timedelta(hours=2),
                raw_payload={},
            )
        ],
        diagnosis=RootCauseDiagnosis(
            diagnosis_id=f"eval_diag_{idx:02d}",
            cause_category=sc.cause,
            confidence=0.8,
            status=DiagnosisStatus.INFERRED,
            supporting_signals={},
            evidence_references=[f"eval_evt_{idx:02d}"],
            diagnostic_method="synthetic-scenario",
        ),
        prediction=RecoveryPrediction(
            prediction_id=f"eval_pred_{idx:02d}",
            recovery_probability=sc.probability,
            confidence=0.5,
            model_version="synthetic-scenario-probability",
            feature_version="1.0",
            prediction_timestamp=now,
            contributing_features={},
            prediction_status="SYNTHETIC",
        ),
    )
    if sc.recovered:
        case.recovered_amount = Money(amount=Decimal(str(sc.recovered)), currency="INR")
        case.outstanding_amount = Money(amount=amount - Decimal(str(sc.recovered)), currency="INR")
        if sc.state == CaseState.FULLY_RECOVERED:
            case.outstanding_amount = Money(amount=Decimal("0"), currency="INR")
    return case


def evaluate_scenario(sc: Scenario, idx: int) -> AgentEvalRecord:
    started = time.perf_counter()
    errors: list[str] = []
    case = build_case(sc, idx)
    case.recommendation = DeterministicActionEvaluator().evaluate(case)
    policy_ctx = PolicyContext(**sc.policy)
    ctx = build_decision_context(case, policy_ctx)

    agent = route_case(case)
    result = run_specialist(case, ctx, policy_signals=ctx.get("policy_signals"))
    rec = result.recommendation
    if result.error:
        errors.append(result.error)

    decision = DeterministicPolicyEngine().evaluate(case, context=policy_ctx)
    case.policy_decision = decision

    execution_result = "NOT_ATTEMPTED"
    executed_without_permission = False
    if case.recommendation.top_candidate is not None:
        record = AgentOrchestrator().execute(case, case.recommendation.top_candidate.action_type)
        execution_result = record.status.value
        executed_without_permission = (
            record.status.value == "COMPLETED_SIMULATED" and decision.status.value != "PERMITTED"
        )

    recommended = rec.recommended_action
    out_of_domain = recommended is not None and not action_applies_to(recommended, sc.domain)
    latency_ms = (time.perf_counter() - started) * 1000

    return AgentEvalRecord(
        scenario=sc.name,
        case_id=case.case_id,
        domain=sc.domain.value,
        input_context_version=DECISION_CONTEXT_VERSION,
        agent_id=rec.agent_id,
        agent_version=rec.agent_version,
        model_version=rec.model_reference.model_version if rec.model_reference else None,
        recommendation=recommended.value if recommended else None,
        candidate_actions=[c.action_type.value for c in rec.candidate_actions],
        policy_result=decision.status.value,
        execution_result=execution_result,
        authoritative_outcome="NOT_APPLICABLE_SYNTHETIC",
        latency_ms=round(latency_ms, 3),
        errors=errors,
        provenance=SYNTHETIC,
        expected_agent=sc.expected_agent,
        expected_policy=sc.expected_policy,
        routing_correct=agent is not None and agent.agent_id == sc.expected_agent,
        output_valid=rec.status != AgentStatus.FALLBACK,
        policy_as_expected=decision.status.value == sc.expected_policy,
        unsafe=bool(out_of_domain or executed_without_permission),
        execution_blocked=execution_result == "REJECTED",
    )


def run_agent_benchmark() -> dict:
    records = [evaluate_scenario(sc, i) for i, sc in enumerate(SCENARIOS)]
    n = len(records)
    lat = sorted(r.latency_ms for r in records)

    def rate(pred) -> float:
        return round(sum(1 for r in records if pred(r)) / n, 4)

    return {
        "provenance": SYNTHETIC,
        "disclaimer": (
            "deterministic synthetic scenarios exercising the real router, agents, Policy "
            "Engine and simulated executor - NOT production agent performance; no recovery "
            "is produced (authoritative_outcome is NOT_APPLICABLE_SYNTHETIC)"
        ),
        "scenarios": n,
        "routing_accuracy": rate(lambda r: r.routing_correct),
        "valid_structured_output_rate": rate(lambda r: r.output_valid),
        "policy_as_expected_rate": rate(lambda r: r.policy_as_expected),
        "unsafe_recommendation_rate": rate(lambda r: r.unsafe),
        "execution_block_rate": rate(lambda r: r.execution_blocked),
        "outcome_correctness": "NOT_APPLICABLE_SYNTHETIC",
        "latency_ms": {
            "mean": round(statistics.mean(lat), 3),
            "p95": round(lat[min(n - 1, int(0.95 * n))], 3),
            "max": lat[-1],
        },
        "records": [asdict(r) for r in records],
    }


# Actions a permitted execution may dispatch, for tests that assert nothing outside
# the closed taxonomy ever appears.
ALL_ACTIONS = {a.value for a in ActionType}
