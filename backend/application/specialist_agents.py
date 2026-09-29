"""Phase 15 — deterministic specialist agents, routing, and the safe runner.

Honest description of what these are: DETERMINISTIC specialists. Each one is bound
to exactly one recovery domain, reads a decision context through a restricted
toolbox (READ_ONLY + RECOMMENDATION tools only), and returns a schema-validated
`AgentRecommendation`. No LLM is involved (none is in the live path - see
docs/history/PHASE_15_MULTI_AGENT_ORCHESTRATION_REPORT.md). They wrap the existing
`DeterministicActionEvaluator` rather than re-deriving actions, and they never
re-implement consent, suppression, retry or contact rules - those stay in the Policy
Engine, and the agents only *list* which of them will apply.

The recommendation an agent produces is cross-checked against the deterministic
recommendation already on the case; on any disagreement, failure, timeout or invalid
output the deterministic result stands and the agent output is marked FALLBACK.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from application.agent_contract import (
    AGENT_CONTRACT_VERSION,
    AgentOutputRejected,
    AgentRecommendation,
    AgentRun,
    AgentStatus,
    AgentToolbox,
    AnomalyInfo,
    CandidateRef,
    ExplanationItem,
    Handoff,
    ModelReference,
    ReasonCode,
    RunState,
    make_handoff,
    new_trace_id,
    prerequisites_for,
    validate_agent_output,
)
from application.trace_ids import new_operation_id, trace_id_for_case
from domain.lifecycle import OUTCOME_FINAL_STATES
from domain.models import ActionType, RecoveryCase, RiskCategory, action_applies_to

logger = logging.getLogger(__name__)

AGENT_TIMEOUT_SECONDS = 2.0
HIGH_PROBABILITY = 0.7
LOW_PROBABILITY = 0.4


class SpecialistAgent:
    agent_id: str = "AbstractSpecialist"
    agent_version: str = "1.0.0"
    domain: RiskCategory

    def analyze(
        self,
        case: RecoveryCase,
        decision_context: dict,
        *,
        policy_signals: dict | None = None,
        trace_id: str | None = None,
    ) -> AgentRecommendation:
        trace_id = trace_id or new_trace_id()
        tools = AgentToolbox(self.agent_id, case, decision_context, policy_signals)
        base = {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "case_id": case.case_id,
            "domain": self.domain,
            "trace_id": trace_id,
        }

        summary = tools.call("get_case_summary")
        if case.current_state in OUTCOME_FINAL_STATES:
            return AgentRecommendation(
                **base,
                status=AgentStatus.NO_ACTION,
                reason_codes=[ReasonCode.OBLIGATION_ALREADY_CLOSED],
                rationale=f"obligation is {summary['obligation_state']}; nothing remains to recover",
            )

        prediction = tools.call("get_prediction")
        anomaly = tools.call("get_anomaly")
        recommendation = tools.call("generate_candidate_actions")

        candidates = [
            c
            for c in (recommendation.candidates or [])
            if action_applies_to(c.action_type, self.domain)
        ]
        refs = [
            CandidateRef(
                action_type=c.action_type,
                estimated_probability=c.estimated_probability,
                expected_recoverable_value=c.expected_recoverable_value,
            )
            for c in candidates
        ]

        reasons: list[ReasonCode] = []
        cause = (decision_context or {}).get("cause_category")
        reasons.append(
            ReasonCode.CAUSE_DIAGNOSED if cause and cause != "UNKNOWN" else ReasonCode.CAUSE_UNKNOWN
        )
        probability = prediction.recovery_probability if prediction else None
        if probability is not None:
            if probability >= HIGH_PROBABILITY:
                reasons.append(ReasonCode.HIGH_RECOVERY_PROBABILITY)
            elif probability < LOW_PROBABILITY:
                reasons.append(ReasonCode.LOW_RECOVERY_PROBABILITY)
        if anomaly and anomaly.get("level") in ("UNUSUAL", "ANOMALOUS"):
            reasons.append(ReasonCode.ANOMALY_SIGNAL_PRESENT)
        if (policy_signals or {}).get("prior_payment_attempts", 0) or (
            (policy_signals or {}).get("comms_in_last_24h", 0)
        ):
            reasons.append(ReasonCode.PRIOR_ACTION_ON_RECORD)

        top = recommendation.top_candidate
        chosen = (
            top.action_type
            if top and any(c.action_type == top.action_type for c in candidates)
            else None
        )
        if not refs or chosen is None:
            return AgentRecommendation(
                **base,
                status=AgentStatus.NO_ACTION,
                reason_codes=[*reasons, ReasonCode.HUMAN_REVIEW_FALLBACK],
                rationale="no applicable candidate action; a person should review",
                recovery_probability=probability,
            )

        reasons.append(
            ReasonCode.ONLY_APPLICABLE_CANDIDATE
            if len(refs) == 1
            else ReasonCode.HIGHEST_ERV_CANDIDATE
        )
        if chosen == ActionType.ESCALATE_TO_HUMAN:
            reasons.append(ReasonCode.HUMAN_REVIEW_FALLBACK)

        explanation = (
            [
                ExplanationItem(feature=e["feature"], direction=e["direction"])
                for e in (prediction.explanation or [])
                if prediction
                and e.get("direction") in ("increased", "decreased", "no_effect", "UNKNOWN")
            ]
            if prediction
            else []
        )

        return AgentRecommendation(
            **base,
            status=AgentStatus.RECOMMENDED,
            candidate_actions=refs,
            recommended_action=chosen,
            model_reference=ModelReference(
                model_version=prediction.model_version,
                prediction_status=prediction.prediction_status,
                prediction_id=prediction.prediction_id,
                confidence_tier=prediction.confidence_tier,
            )
            if prediction
            else None,
            recovery_probability=probability,
            expected_recoverable_value=top.expected_recoverable_value,
            anomaly=AnomalyInfo(level=anomaly["level"], max_abs_z=anomaly.get("max_abs_z"))
            if anomaly and anomaly.get("level")
            else None,
            explanation=explanation,
            reason_codes=reasons,
            rationale=(
                f"{self.agent_id} proposes {chosen.value}: "
                f"{'the only' if len(refs) == 1 else 'the highest-ERV'} applicable candidate "
                f"of {len(refs)}. Model association only; the Policy Engine decides."
            ),
            policy_prerequisites=prerequisites_for(chosen),
        )


class PaymentAgent(SpecialistAgent):
    agent_id = "PaymentAgent"
    domain = RiskCategory.FAILED_PAYMENT


class CheckoutAgent(SpecialistAgent):
    agent_id = "CheckoutAgent"
    domain = RiskCategory.CHECKOUT_ABANDONMENT


class SubscriptionAgent(SpecialistAgent):
    agent_id = "SubscriptionAgent"
    domain = RiskCategory.FAILED_SUBSCRIPTION


class InvoiceAgent(SpecialistAgent):
    agent_id = "InvoiceAgent"
    domain = RiskCategory.OVERDUE_INVOICE


class PromiseToPayAgent(SpecialistAgent):
    agent_id = "PromiseToPayAgent"
    domain = RiskCategory.BROKEN_PROMISE


_AGENTS: dict[RiskCategory, SpecialistAgent] = {
    a.domain: a
    for a in (
        PaymentAgent(),
        CheckoutAgent(),
        SubscriptionAgent(),
        InvoiceAgent(),
        PromiseToPayAgent(),
    )
}


def route_case(case) -> SpecialistAgent | None:
    """Deterministic domain -> specialist routing. An unrecognised domain returns
    None (=> NEEDS_MAPPING); there is no default agent."""
    raw = getattr(case, "risk_category", None)
    if isinstance(raw, RiskCategory):
        return _AGENTS.get(raw)
    try:
        return _AGENTS.get(RiskCategory(getattr(raw, "value", raw)))
    except ValueError:
        return None


# ---------------------------------------------------------------- safe runner


@dataclass
class AgentRunResult:
    recommendation: AgentRecommendation
    run: AgentRun
    latency_ms: float
    tools_called: list[str] = field(default_factory=list)
    error: str | None = None
    agrees_with_deterministic: bool | None = None


def _needs_mapping(case, trace_id: str) -> AgentRecommendation:
    raw = getattr(
        getattr(case, "risk_category", None), "value", getattr(case, "risk_category", None)
    )
    return AgentRecommendation(
        agent_id="Router",
        agent_version="1.0.0",
        case_id=getattr(case, "case_id", "unknown"),
        domain=None,
        domain_raw=str(raw),
        status=AgentStatus.NEEDS_MAPPING,
        reason_codes=[ReasonCode.HUMAN_REVIEW_FALLBACK],
        rationale="no specialist is registered for this domain; mapping is required",
        trace_id=trace_id,
    )


def _fallback(case, agent: SpecialistAgent | None, trace_id: str, why: str) -> AgentRecommendation:
    return AgentRecommendation(
        agent_id=agent.agent_id if agent else "Router",
        agent_version=agent.agent_version if agent else "1.0.0",
        case_id=case.case_id,
        domain=agent.domain if agent else None,
        status=AgentStatus.FALLBACK,
        reason_codes=[ReasonCode.HUMAN_REVIEW_FALLBACK],
        rationale=f"agent output unusable ({why}); the deterministic recommendation stands",
        trace_id=trace_id,
    )


def run_specialist(
    case: RecoveryCase,
    decision_context: dict,
    *,
    policy_signals: dict | None = None,
    timeout_s: float = AGENT_TIMEOUT_SECONDS,
    agent_override: SpecialistAgent | None = None,
) -> AgentRunResult:
    """Route, analyze (bounded by a timeout), validate, and cross-check against the
    deterministic recommendation already on the case. Never raises: every failure
    becomes a FALLBACK recommendation and the run ends FAILED (terminal)."""
    started = time.perf_counter()
    # The journey trace id is derived from the case (immutable across re-advances and
    # retries); this run's own identity is the operation_id.
    run = AgentRun(case.case_id, trace_id=trace_id_for_case(case.case_id))
    run.operation_id = new_operation_id("agentrun")
    trace_id = run.trace_id

    agent = agent_override or route_case(case)
    if agent is None:
        run.advance(RunState.NEEDS_MAPPING)
        return AgentRunResult(
            _needs_mapping(case, trace_id), run, (time.perf_counter() - started) * 1000
        )

    error = None
    try:
        run.advance(RunState.CONTEXT_BUILT)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(
                agent.analyze,
                case.model_copy(deep=True),
                decision_context,
                policy_signals=policy_signals,
                trace_id=trace_id,
            )
            raw = future.result(timeout=timeout_s)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        run.advance(RunState.ANALYZED)
        rec = validate_agent_output(raw, expected_domain=case.risk_category)
        run.advance(RunState.RECOMMENDATION_READY)
    except FutureTimeout:
        error = "timeout"
    except AgentOutputRejected as e:
        error = f"invalid_output: {e}"
    except Exception as e:  # noqa: BLE001 - any agent failure must degrade, never propagate
        error = f"{type(e).__name__}"

    if error is not None:
        if run.state not in (RunState.FAILED,):
            try:
                run.advance(RunState.FAILED)
            except Exception:  # noqa: BLE001
                pass
        return AgentRunResult(
            _fallback(case, agent, trace_id, error.split(":")[0]),
            run,
            (time.perf_counter() - started) * 1000,
            error=error,
        )

    agrees = None
    det = case.recommendation.top_candidate if case.recommendation else None
    if det is not None and rec.status == AgentStatus.RECOMMENDED:
        agrees = det.action_type == rec.recommended_action
        if not agrees:
            run.advance(
                RunState.FAILED
            )  # a disagreeing agent is a failed run; deterministic stands
            return AgentRunResult(
                _fallback(case, agent, trace_id, "disagrees with deterministic evaluator"),
                run,
                (time.perf_counter() - started) * 1000,
                error="disagreement",
                agrees_with_deterministic=False,
            )
    return AgentRunResult(
        rec, run, (time.perf_counter() - started) * 1000, agrees_with_deterministic=agrees
    )


def agent_evidence(result: AgentRunResult) -> dict:
    """The audit-log payload for an agent step (existing audit infrastructure)."""
    rec = result.recommendation
    return {
        "action": "agent_recommendation",
        "trace_id": rec.trace_id,
        "parent_trace_id": rec.trace_id,
        "operation_id": getattr(result.run, "operation_id", None),
        "agent_id": rec.agent_id,
        "agent_version": rec.agent_version,
        "contract_version": AGENT_CONTRACT_VERSION,
        "domain": rec.domain.value if rec.domain else None,
        "status": rec.status.value,
        "recommended_action": rec.recommended_action.value if rec.recommended_action else None,
        "recovery_probability": rec.recovery_probability,
        "expected_recoverable_value": rec.expected_recoverable_value,
        "model_version": rec.model_reference.model_version if rec.model_reference else None,
        "reason_codes": [r.value for r in rec.reason_codes],
        "anomaly_level": rec.anomaly.level if rec.anomaly else None,
        "policy_prerequisites": [p.value for p in rec.policy_prerequisites],
        "run_states": [s for s, _ in result.run.history],
        "latency_ms": round(result.latency_ms, 3),
        "error": result.error,
        "agrees_with_deterministic": result.agrees_with_deterministic,
    }


def request_context_handoff(
    agent: SpecialistAgent, case: RecoveryCase, trace_id: str, fields: list[str], reason: str
) -> Handoff:
    """A structured, schema-validated handoff from an agent to the CONTEXT layer.
    Transfers information only (`grants_authorization` is always False)."""
    return make_handoff(
        source_agent=agent.agent_id,
        destination="CONTEXT_LAYER",
        case_id=case.case_id,
        reason=reason,
        payload_schema="CUSTOMER_CONTEXT_REQUEST",
        payload={"fields": fields},
        trace_id=trace_id,
    )
