"""Phase 15 — bounded multi-agent orchestration tests.

contract / structured output / tool boundaries / routing / state machine / handoffs /
adversarial safety / idempotency / failure recovery / observability / benchmark.
"""

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import app
from application.action_evaluator import DeterministicActionEvaluator
from application.agent_contract import (
    AGENT_PERMITTED_TOOL_CLASSES,
    TOOL_REGISTRY,
    AgentOutputRejected,
    AgentRecommendation,
    AgentRun,
    AgentToolbox,
    IllegalTransition,
    RunState,
    ToolClass,
    ToolDenied,
    make_handoff,
    validate_agent_output,
)
from application.agent_evaluation import SCENARIOS, build_case, run_agent_benchmark
from application.agents import (
    AgentOrchestrator,
    CheckoutRecoveryAgent,
    PaymentRecoveryAgent,
    PromiseRecoveryAgent,
    ReceivablesRecoveryAgent,
    SubscriptionRecoveryAgent,
)
from application.decision_context import build_decision_context
from application.policy_engine import DeterministicPolicyEngine, PolicyContext
from application.specialist_agents import (
    PaymentAgent,
    SpecialistAgent,
    route_case,
    run_specialist,
)
from domain.models import (
    ACTION_SPECS,
    ActionChannel,
    ActionType,
    CaseState,
    PolicyDecisionStatus,
    RiskCategory,
    RootCauseCategory,
)
from infrastructure.db import SessionLocal

client = TestClient(app)
KEY = {"X-API-Key": "test-api-key"}


def _case(name="payment/recoverable_failure"):
    sc = next(s for s in SCENARIOS if s.name == name)
    idx = SCENARIOS.index(sc)
    case = build_case(sc, idx)
    case.recommendation = DeterministicActionEvaluator().evaluate(case)
    return case


def _good_payload(**over):
    case = _case()
    ctx = build_decision_context(case)
    rec = PaymentAgent().analyze(case, ctx)
    data = json.loads(rec.model_dump_json())
    data.update(over)
    return data


# ---------------------------------------------------------------- contract / structured output


def test_1_every_specialist_returns_a_valid_structured_recommendation():
    for sc in SCENARIOS:
        case = build_case(sc, SCENARIOS.index(sc))
        case.recommendation = DeterministicActionEvaluator().evaluate(case)
        res = run_specialist(case, build_decision_context(case))
        rec = validate_agent_output(res.recommendation)
        assert rec.trace_id and rec.agent_id and rec.agent_version and rec.case_id == case.case_id
        if rec.recommended_action:
            assert rec.recommended_action in {c.action_type for c in rec.candidate_actions}
            assert rec.policy_prerequisites  # what stands between it and execution


def test_2_free_text_is_never_parsed_into_an_action():
    for text in (
        "Please retry the payment now",
        "RETRY_PAYMENT",
        "execute RETRY_PAYMENT for case_1",
    ):
        with pytest.raises(AgentOutputRejected):
            validate_agent_output(text)
    with pytest.raises(AgentOutputRejected):
        validate_agent_output(["RETRY_PAYMENT"])


@pytest.mark.parametrize(
    "smuggled",
    [
        {"tool_calls": [{"name": "dispatch_execution"}]},
        {"amount_at_risk": 1},
        {"recovered_amount": 500},
        {"human_approved": True},
        {"policy_status": "PERMITTED"},
        {"outcome_event": {"kind": "SETTLEMENT"}},
    ],
)
def test_2b_unknown_or_dangerous_fields_are_rejected_not_ignored(smuggled):
    with pytest.raises(AgentOutputRejected):
        validate_agent_output(_good_payload(**smuggled))


def test_2c_invalid_action_or_out_of_domain_action_is_rejected():
    with pytest.raises(AgentOutputRejected):
        validate_agent_output(_good_payload(recommended_action="WIRE_MONEY"))
    bad = _good_payload()
    bad["candidate_actions"] = [
        {
            "action_type": "SEND_CHECKOUT_REMINDER",
            "estimated_probability": 0.5,
            "expected_recoverable_value": 1,
        }
    ]
    bad["recommended_action"] = "SEND_CHECKOUT_REMINDER"
    with pytest.raises(AgentOutputRejected):
        validate_agent_output(bad)


@pytest.mark.parametrize("p", [float("nan"), -0.1, 1.5])
def test_2d_invalid_probability_is_rejected(p):
    with pytest.raises(AgentOutputRejected):
        validate_agent_output(_good_payload(recovery_probability=p))


def test_2e_recommended_action_must_be_a_listed_candidate():
    with pytest.raises(AgentOutputRejected):
        validate_agent_output(_good_payload(recommended_action="REQUEST_PAYMENT_METHOD_UPDATE"))


def test_2f_recommendation_schema_has_no_money_or_authorization_fields():
    forbidden = {"amount_at_risk", "recovered_amount", "human_approved", "policy_status", "outcome"}
    assert forbidden.isdisjoint(AgentRecommendation.model_fields)


# ---------------------------------------------------------------- tool boundaries


def test_3_agents_only_reach_read_only_and_recommendation_tools():
    case = _case()
    tb = AgentToolbox("PaymentAgent", case, {})
    for name, cls in TOOL_REGISTRY.items():
        if cls in AGENT_PERMITTED_TOOL_CLASSES:
            if name in ("emit_recommendation",):
                continue
            tb.call(name)
        else:
            with pytest.raises(ToolDenied):
                tb.call(name)
    with pytest.raises(ToolDenied):
        tb.call("drop_table")  # arbitrary tool injection


def test_3b_every_execution_policy_and_outcome_tool_is_classified_and_denied():
    assert {ToolClass.POLICY, ToolClass.EXECUTION, ToolClass.OUTCOME} <= set(TOOL_REGISTRY.values())
    assert ToolClass.EXECUTION not in AGENT_PERMITTED_TOOL_CLASSES
    assert ToolClass.OUTCOME not in AGENT_PERMITTED_TOOL_CLASSES


def test_3c_agent_works_on_a_copy_and_cannot_mutate_the_case():
    case = _case()
    before = case.model_dump_json()
    PaymentAgent().analyze(case, build_decision_context(case))
    assert case.model_dump_json() == before
    tb = AgentToolbox("PaymentAgent", case, {})
    tb.call("get_prediction").recovery_probability = 0.0  # mutate the returned copy
    assert case.model_dump_json() == before


# ---------------------------------------------------------------- routing


@pytest.mark.parametrize(
    "domain,agent",
    [
        (RiskCategory.FAILED_PAYMENT, "PaymentAgent"),
        (RiskCategory.CHECKOUT_ABANDONMENT, "CheckoutAgent"),
        (RiskCategory.FAILED_SUBSCRIPTION, "SubscriptionAgent"),
        (RiskCategory.OVERDUE_INVOICE, "InvoiceAgent"),
        (RiskCategory.BROKEN_PROMISE, "PromiseToPayAgent"),
    ],
)
def test_4_domain_routes_to_exactly_its_specialist(domain, agent):
    sc = next(s for s in SCENARIOS if s.domain == domain)
    case = build_case(sc, 0)
    assert route_case(case).agent_id == agent


def test_4b_unknown_domain_is_needs_mapping_not_an_arbitrary_agent():
    class _Case:
        case_id = "c"
        risk_category = "SOMETHING_NEW"

    assert route_case(_Case()) is None
    res = run_specialist(_Case(), {})
    assert res.recommendation.status.value == "NEEDS_MAPPING"
    assert res.recommendation.candidate_actions == []
    assert res.run.state == RunState.NEEDS_MAPPING


def test_4c_a_specialist_never_proposes_an_action_outside_its_domain():
    for sc in SCENARIOS:
        case = build_case(sc, SCENARIOS.index(sc))
        case.recommendation = DeterministicActionEvaluator().evaluate(case)
        rec = run_specialist(case, build_decision_context(case)).recommendation
        for c in rec.candidate_actions:
            assert sc.domain in ACTION_SPECS[c.action_type][1]


def test_4d_executor_selection_is_domain_aware_and_unique():
    """Every non-internal (action, domain) pair has exactly one executor - a
    subscription case's REQUEST_PAYMENT_METHOD_UPDATE no longer goes to the payment
    executor (regression for the pre-Phase-15 first-match selection)."""
    orch = AgentOrchestrator()
    for action, (channel, domains) in ACTION_SPECS.items():
        if channel == ActionChannel.INTERNAL:
            continue
        for d in domains:
            owners = [a for a in orch.agents if a.can_handle(action) and d in a.domains]
            assert len(owners) == 1, (action, d, owners)
    assert isinstance(
        next(
            a
            for a in orch.agents
            if a.can_handle(ActionType.REQUEST_PAYMENT_METHOD_UPDATE)
            and RiskCategory.FAILED_SUBSCRIPTION in a.domains
        ),
        SubscriptionRecoveryAgent,
    )


def test_4e_specialist_and_executor_domains_agree():
    pairs = {
        "PaymentAgent": PaymentRecoveryAgent,
        "CheckoutAgent": CheckoutRecoveryAgent,
        "SubscriptionAgent": SubscriptionRecoveryAgent,
        "InvoiceAgent": ReceivablesRecoveryAgent,
        "PromiseToPayAgent": PromiseRecoveryAgent,
    }
    for sc in SCENARIOS:
        case = build_case(sc, 0)
        spec = route_case(case)
        assert spec.domain in pairs[spec.agent_id].domains


# ---------------------------------------------------------------- state machine


def test_5_state_machine_happy_path_and_terminal_states():
    run = AgentRun("c1")
    for s in (
        RunState.CONTEXT_BUILT,
        RunState.ANALYZED,
        RunState.RECOMMENDATION_READY,
        RunState.POLICY_REVIEW,
    ):
        run.advance(s)
    run.apply_policy_decision(PolicyDecisionStatus.PERMITTED)
    assert run.state == RunState.APPROVED
    for s in (RunState.EXECUTING, RunState.VERIFYING, RunState.COMPLETED):
        run.advance(s)
    with pytest.raises(IllegalTransition):
        run.advance(RunState.RECEIVED)  # terminal stays terminal


def test_5b_authorization_states_cannot_be_entered_directly_or_skipped():
    run = AgentRun("c2")
    for target in (RunState.APPROVED, RunState.APPROVAL_REQUIRED, RunState.REJECTED):
        with pytest.raises(IllegalTransition):
            run.advance(target)
    with pytest.raises(IllegalTransition):
        run.advance(RunState.EXECUTING)  # cannot skip straight to execution
    run.advance(RunState.CONTEXT_BUILT)
    run.advance(RunState.ANALYZED)
    run.advance(RunState.RECOMMENDATION_READY)
    with pytest.raises(IllegalTransition):
        run.advance(RunState.EXECUTING)  # RECOMMENDATION_READY -> EXECUTING is not a thing
    with pytest.raises(IllegalTransition):
        run.apply_policy_decision(PolicyDecisionStatus.PERMITTED)  # not yet in POLICY_REVIEW


def test_5c_escalation_needs_human_review_and_revalidation():
    run = AgentRun("c3")
    for s in (
        RunState.CONTEXT_BUILT,
        RunState.ANALYZED,
        RunState.RECOMMENDATION_READY,
        RunState.POLICY_REVIEW,
    ):
        run.advance(s)
    run.apply_policy_decision(PolicyDecisionStatus.ESCALATE)
    assert run.state == RunState.APPROVAL_REQUIRED
    with pytest.raises(IllegalTransition):
        run.advance(RunState.EXECUTING)
    run.apply_human_review(PolicyDecisionStatus.DENIED)  # approved, but fresh policy says no
    assert run.state == RunState.REJECTED


def test_5d_denied_and_wait_are_rejected_terminals():
    for st in (PolicyDecisionStatus.DENIED, PolicyDecisionStatus.WAIT):
        run = AgentRun("c4")
        for s in (
            RunState.CONTEXT_BUILT,
            RunState.ANALYZED,
            RunState.RECOMMENDATION_READY,
            RunState.POLICY_REVIEW,
        ):
            run.advance(s)
        run.apply_policy_decision(st)
        assert run.state == RunState.REJECTED
        with pytest.raises(IllegalTransition):
            run.advance(RunState.EXECUTING)


# ---------------------------------------------------------------- handoffs


def test_6_handoff_is_structured_recorded_and_never_grants_authority():
    h = make_handoff(
        source_agent="PaymentAgent",
        destination="CONTEXT_LAYER",
        case_id="c1",
        reason="need retry history",
        payload_schema="CUSTOMER_CONTEXT_REQUEST",
        payload={"fields": ["retry_history"]},
        trace_id="trace_x",
    )
    ev = h.to_audit_evidence()
    assert ev["grants_authorization"] is False and ev["trace_id"] == "trace_x"
    assert (
        ev["source_agent"] == "PaymentAgent" and ev["payload_schema"] == "CUSTOMER_CONTEXT_REQUEST"
    )
    with pytest.raises(ValidationError):
        h.model_copy(update={"grants_authorization": True}).model_validate(
            {**h.model_dump(), "grants_authorization": True}
        )


def test_6b_handoff_to_another_agent_or_unknown_schema_is_rejected():
    kw = dict(source_agent="a", case_id="c", reason="r", trace_id="t")
    with pytest.raises(AgentOutputRejected):
        make_handoff(
            destination="InvoiceAgent",
            payload_schema="CUSTOMER_CONTEXT_REQUEST",
            payload={"fields": ["retry_history"]},
            **kw,
        )
    with pytest.raises(AgentOutputRejected):
        make_handoff(
            destination="CONTEXT_LAYER", payload_schema="APPROVE_EXECUTION", payload={}, **kw
        )
    with pytest.raises(AgentOutputRejected):
        make_handoff(
            destination="CONTEXT_LAYER",
            payload_schema="CUSTOMER_CONTEXT_REQUEST",
            payload={"fields": ["retry_history"], "approve": True},
            **kw,
        )


# ---------------------------------------------------------------- adversarial safety


def _authorized(action=ActionType.RETRY_PAYMENT, status=PolicyDecisionStatus.PERMITTED):
    case = _case()
    case.recommendation = DeterministicActionEvaluator().evaluate(case)
    case.policy_decision = DeterministicPolicyEngine().evaluate(case, PolicyContext())
    case.policy_decision.status = status
    return case


def test_7_agent_recommendation_cannot_bypass_policy():
    case = _case("payment/suppression")
    ctx = PolicyContext(payment_permission="SUPPRESSED")
    rec = run_specialist(case, build_decision_context(case, ctx)).recommendation
    assert rec.recommended_action == ActionType.RETRY_PAYMENT  # agent proposes...
    case.policy_decision = DeterministicPolicyEngine().evaluate(case, ctx)
    assert case.policy_decision.status == PolicyDecisionStatus.DENIED  # ...policy refuses
    record = AgentOrchestrator().execute(case, ActionType.RETRY_PAYMENT)
    assert record.status.value == "REJECTED" and record.adapter_used == "NONE"


def test_7b_agent_cannot_override_consent_retry_limit_or_approval_threshold():
    for name, ctx, expected in (
        ("payment/consent_denied", PolicyContext(payment_permission="OPTED_OUT"), "DENIED"),
        ("payment/repeated_failure", PolicyContext(prior_payment_attempts=3), "DENIED"),
        ("payment/high_value_approval", PolicyContext(), "ESCALATE"),
    ):
        case = _case(name)
        run_specialist(case, build_decision_context(case, ctx))
        assert DeterministicPolicyEngine().evaluate(case, ctx).status.value == expected


def test_7c_unauthorized_or_mismatched_action_is_rejected():
    case = _authorized()
    rec = AgentOrchestrator().execute(
        case, ActionType.SEND_PAYMENT_REMINDER
    )  # not what policy authorized
    assert rec.status.value == "REJECTED" and rec.idempotency_key == "REJECTED_MISMATCH"


def test_7d_an_agent_cannot_approve_itself_or_skip_human_approval():
    case = _case("payment/high_value_approval")
    ctx = PolicyContext()
    decision = DeterministicPolicyEngine().evaluate(case, ctx)
    assert decision.status == PolicyDecisionStatus.ESCALATE
    with pytest.raises(ValidationError):
        PolicyContext(human_approved=True, agent_approved=True)  # extra fields refused
    # an agent payload claiming approval is refused at the contract boundary
    with pytest.raises(AgentOutputRejected):
        validate_agent_output(_good_payload(human_approved=True))
    case.policy_decision = decision  # still ESCALATE - nothing the agent did changed it
    assert AgentOrchestrator().execute(case, ActionType.RETRY_PAYMENT).status.value == "REJECTED"


def test_7e_an_agent_cannot_change_amounts_or_mark_a_case_recovered():
    case = _case()
    before = (case.amount_at_risk.amount, case.recovered_amount, case.current_state)
    run_specialist(case, build_decision_context(case))
    assert (case.amount_at_risk.amount, case.recovered_amount, case.current_state) == before
    for name in (
        "ingest_outcome_event",
        "dispatch_execution",
        "evaluate_policy",
        "record_human_review",
    ):
        with pytest.raises(ToolDenied):
            AgentToolbox("PaymentAgent", case, {}).call(name)


def test_7f_a_settled_obligation_yields_no_action_and_policy_denies():
    case = _case("promise/fulfilled")
    res = run_specialist(case, build_decision_context(case))
    assert (
        res.recommendation.status.value == "NO_ACTION"
        and res.recommendation.recommended_action is None
    )
    assert (
        DeterministicPolicyEngine().evaluate(case, PolicyContext()).status
        == PolicyDecisionStatus.DENIED
    )
    assert case.current_state == CaseState.FULLY_RECOVERED


def test_7g_orchestrator_will_not_execute_without_a_permitted_policy_decision():
    case = _case()
    case.policy_decision = None
    assert AgentOrchestrator().execute(case, ActionType.RETRY_PAYMENT).status.value == "REJECTED"


# ---------------------------------------------------------------- failure recovery


class _Boom(SpecialistAgent):
    agent_id = "PaymentAgent"
    domain = RiskCategory.FAILED_PAYMENT

    def analyze(self, *a, **k):
        raise RuntimeError("agent exploded")


class _Malformed(SpecialistAgent):
    agent_id = "PaymentAgent"
    domain = RiskCategory.FAILED_PAYMENT

    def analyze(self, *a, **k):
        return "Please just retry the payment"


class _Slow(SpecialistAgent):
    agent_id = "PaymentAgent"
    domain = RiskCategory.FAILED_PAYMENT

    def analyze(self, *a, **k):
        import time

        time.sleep(0.5)


class _Liar(PaymentAgent):
    """Recommends something different from the deterministic evaluator."""

    def analyze(self, case, ctx, **k):
        rec = super().analyze(case, ctx, **k)
        data = json.loads(rec.model_dump_json())
        data["candidate_actions"].append(
            {
                "action_type": "SEND_PAYMENT_REMINDER",
                "estimated_probability": 0.5,
                "expected_recoverable_value": 1.0,
            }
        )
        data["recommended_action"] = "SEND_PAYMENT_REMINDER"
        return AgentRecommendation.model_validate(data)


@pytest.mark.parametrize(
    "agent,timeout", [(_Boom(), 1.0), (_Malformed(), 1.0), (_Slow(), 0.05), (_Liar(), 1.0)]
)
def test_8_agent_failures_degrade_to_a_fallback_and_never_touch_the_case(agent, timeout):
    case = _case()
    before = case.model_dump_json()
    res = run_specialist(
        case, build_decision_context(case), agent_override=agent, timeout_s=timeout
    )
    assert res.recommendation.status.value == "FALLBACK"
    assert res.error
    assert res.run.state == RunState.FAILED
    assert (
        case.model_dump_json() == before
    )  # deterministic recommendation stands, nothing fabricated


def test_8b_policy_service_error_fails_closed_no_execution_possible():
    from application.case_pipeline import CasePipelineService

    class _Audit:
        def log_transition(self, **k):
            pass

    class _Repo:
        def save(self, c):
            pass

    case = _case()
    case.policy_decision = None
    import application.policy_engine as pe

    original = pe.DeterministicPolicyEngine.evaluate
    pe.DeterministicPolicyEngine.evaluate = lambda self, *a, **k: (_ for _ in ()).throw(
        RuntimeError("policy down")
    )
    try:
        with pytest.raises(RuntimeError):
            CasePipelineService(_Repo(), _Audit()).policy_check(case)
    finally:
        pe.DeterministicPolicyEngine.evaluate = original
    assert case.policy_decision is None
    assert AgentOrchestrator().execute(case, ActionType.RETRY_PAYMENT).status.value == "REJECTED"


def test_8c_database_failure_at_slot_claim_means_the_adapter_is_never_called():
    calls = []

    class _Adapter:
        def execute(self, params):
            calls.append(params)
            return {"adapter_status": "COMPLETED_SIMULATED"}

    class _Repo:
        def claim_execution_slot(self, key):
            raise RuntimeError("database unavailable")

    case = _authorized()
    with pytest.raises(RuntimeError):
        AgentOrchestrator(_Adapter()).execute(case, ActionType.RETRY_PAYMENT, repo=_Repo())
    assert calls == []


def test_8d_adapter_error_is_a_failed_record_not_a_success_and_creates_no_outcome():
    class _Adapter:
        def execute(self, params):
            raise TimeoutError("gateway down")

    case = _authorized()
    rec = AgentOrchestrator(_Adapter()).execute(case, ActionType.RETRY_PAYMENT)
    assert rec.status.value == "FAILED"
    assert case.recovered_amount is None or case.recovered_amount.amount == 0


def test_8e_llm_unavailable_is_the_normal_state_and_changes_nothing():
    import os

    from application.real_llm_adapter import RealGeminiAdapter

    saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        assert RealGeminiAdapter().available is False
    finally:
        if saved is not None:
            os.environ["GEMINI_API_KEY"] = saved
    case = _case()
    assert (
        run_specialist(case, build_decision_context(case)).recommendation.status.value
        == "RECOMMENDED"
    )


# ---------------------------------------------------------------- idempotency (real API)


def _open_permitted_case(tag):
    r = client.post(
        "/events",
        json={
            "customer_id": f"p15_cust_{tag}",
            "risk_category": "FAILED_PAYMENT",
            "external_system": "p15_sys",
            "external_event_id": f"p15_evt_{tag}",
            "amount": "300",
            "currency": "INR",
            "raw_payload": {"failure_code": "network_error"},
        },
        headers=KEY,
    )
    assert r.status_code == 200
    cid = r.json()["case_id"]
    adv = client.post(f"/cases/{cid}/advance", headers=KEY).json()
    assert adv["policy_status"] == "PERMITTED"
    return cid


def test_9_repeated_execute_requests_produce_at_most_one_side_effect():
    from infrastructure.orm import ExecutionAttemptModel

    cid = _open_permitted_case(uuid.uuid4().hex[:8])
    statuses = [
        client.post(f"/cases/{cid}/execute", headers=KEY).json()["status"] for _ in range(4)
    ]
    assert statuses[0] == "COMPLETED_SIMULATED"
    assert all(s == "COMPLETED_SIMULATED" for s in statuses)  # replay returns the original record
    db = SessionLocal()
    try:
        assert (
            db.query(ExecutionAttemptModel).filter(ExecutionAttemptModel.case_id == cid).count()
            == 1
        )
    finally:
        db.close()


def test_9b_orchestrator_level_duplicate_claims_never_reach_the_adapter_twice():
    calls = []

    class _Adapter:
        def execute(self, params):
            calls.append(1)
            return {"adapter_status": "COMPLETED_SIMULATED"}

    class _Repo:
        def __init__(self):
            self.slots = set()

        def claim_execution_slot(self, key):
            if key in self.slots:
                return False
            self.slots.add(key)
            return True

        def get_execution_record_by_key(self, key):
            return None

        def complete_execution_slot(self, key, rec):
            pass

        def record_execution_attempt(self, **k):
            pass

    repo = _Repo()
    case = _authorized()
    orch = AgentOrchestrator(_Adapter())
    results = [orch.execute(case, ActionType.RETRY_PAYMENT, repo=repo) for _ in range(3)]
    assert len(calls) == 1
    assert sum(1 for r in results if r.status.value == "COMPLETED_SIMULATED") == 1


# ---------------------------------------------------------------- observability / trace


def test_10_agent_step_is_audited_and_the_chain_is_reconstructable():
    cid = _open_permitted_case(uuid.uuid4().hex[:8])
    client.post(f"/cases/{cid}/execute", headers=KEY)
    body = client.get(f"/system/cases/{cid}/agent-trace").json()
    assert body["trace_id"] and body["trace_id"].startswith("trace_")
    assert body["agent"]["agent_id"] == "PaymentAgent"
    assert body["agent"]["agrees_with_deterministic"] is True
    stages = [s["stage"] for s in body["steps"]]
    for needed in ("CASE", "MODEL", "RECOMMENDATION", "AGENT", "POLICY", "EXECUTION"):
        assert needed in stages
    order = {
        s: i
        for i, s in enumerate(["CASE", "MODEL", "RECOMMENDATION", "AGENT", "POLICY", "EXECUTION"])
    }
    seq = [order[s] for s in stages if s in order]
    assert seq == sorted(seq)  # agent before policy before execution
    assert body["policy_decision_id"] and body["execution_id"]
    assert "OUTCOME" not in stages  # simulated execution is never an outcome


def test_10b_agent_step_cannot_alter_the_deterministic_recommendation_or_amounts():
    cid = _open_permitted_case(uuid.uuid4().hex[:8])
    snap = client.get(f"/system/cases/{cid}").json()
    assert snap["recommendation"]["top_candidate"]["action_type"] == "RETRY_PAYMENT"
    assert float(snap["amount_at_risk"]) == 300.0
    assert snap["outcome"] is None


def test_10c_trace_for_unknown_case_is_404():
    assert client.get("/system/cases/nope_nope/agent-trace").status_code == 404


# ---------------------------------------------------------------- benchmark


def test_11_benchmark_is_synthetic_flagged_and_safe():
    r = run_agent_benchmark()
    assert r["provenance"] == "SYNTHETIC_SCENARIO"
    assert r["outcome_correctness"] == "NOT_APPLICABLE_SYNTHETIC"
    assert r["routing_accuracy"] == 1.0
    assert r["valid_structured_output_rate"] == 1.0
    assert r["unsafe_recommendation_rate"] == 0.0
    assert r["policy_as_expected_rate"] == 1.0
    assert all(x["authoritative_outcome"] == "NOT_APPLICABLE_SYNTHETIC" for x in r["records"])
    assert all(x["provenance"] == "SYNTHETIC_SCENARIO" for x in r["records"])
    # nothing was executed unless policy PERMITTED
    for x in r["records"]:
        if x["execution_result"] == "COMPLETED_SIMULATED":
            assert x["policy_result"] == "PERMITTED"


def test_11b_benchmark_covers_every_required_scenario_family():
    names = " ".join(s.name for s in SCENARIOS)
    for needle in (
        "recoverable_failure",
        "repeated_failure",
        "suppression",
        "consent_denied",
        "high_value",
        "abandonment",
        "repeated_abandonment",
        "expired_session",
        "billing_failure",
        "retry_exhaustion",
        "cancelled",
        "overdue",
        "partial_settlement",
        "escalation",
        "created",
        "due",
        "fulfilled",
        "broken",
    ):
        assert needle in names, needle


def test_12_no_agent_module_imports_execution_outcome_or_policy_writers():
    """Import-level guarantee (AST, not text): the agent contract/specialists cannot
    even reach the executor, the policy engine, or the outcome writer."""
    import ast

    import application.agent_contract as ac
    import application.specialist_agents as sa

    banned = {
        "application.agents",
        "application.policy_engine",
        "application.outcome_events",
        "application.verification_engine",
        "infrastructure.repositories",
        "infrastructure.adapters",
    }
    for mod in (ac, sa):
        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        assert not (imported & banned), (mod.__name__, imported & banned)


def test_12b_domain_datetime_sanity():
    assert (
        isinstance(datetime.now(UTC), datetime) and Decimal("1") == 1 and RootCauseCategory.UNKNOWN
    )
