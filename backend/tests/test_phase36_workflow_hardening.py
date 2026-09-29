"""Phase 2 controlled-implementation regression tests.

Each test pins a defect that was reproduced against the running application before
being fixed. Grouped by the guarantee it protects:

  A. Every action is rate-limited (five customer-facing actions had no cap at all).
  B. An action that has been dispatched cannot be rewound by a later event.
  C. Human approval is revalidated against the hard limits before execution.
  D. A stop is honoured for the obligation, not just the one case row.
  E. Closed-case handling is consistent (409, not 500, and not a silent no-op).
  F. The execution ledger is a complete, append-only action history.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import app
from application.policy_engine import DeterministicPolicyEngine, MerchantPolicy, PolicyContext
from domain.lifecycle import CaseClosedError, CaseLifecycleManager
from domain.models import (
    ActionChannel,
    ActionRecommendation,
    ActionType,
    CandidateAction,
    CaseState,
    Money,
    RecommendationStatus,
    RecoveryCase,
    RiskCategory,
    action_applies_to,
    action_channel,
    action_domains,
)

client = TestClient(app)


def _ingest(
    customer=None,
    category="FAILED_PAYMENT",
    amount=100.0,
    failure_code="network_error",
    reference=None,
    external_id=None,
):
    payload = {
        "customer_id": customer or f"cust_{uuid.uuid4().hex[:6]}",
        "risk_category": category,
        "external_system": "phase36",
        "external_event_id": external_id or f"evt_{uuid.uuid4().hex[:8]}",
        "reference_id": reference or f"ref_{uuid.uuid4().hex[:8]}",
        "amount": amount,
        "currency": "USD",
        "raw_payload": {"failure_code": failure_code},
    }
    r = client.post("/events", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["case_id"], payload


_settlement_n = 0


def _settle(payload: dict, amount: float) -> dict:
    """Post an authoritative settlement outcome event for the obligation `payload`
    (as returned by _ingest) describes. Phase 5 removed the client-supplied
    sim_full/sim_partial/sim_fail reference from POST /cases/{id}/verify - it is now a
    pure read of whatever a real outcome event already reconciled."""
    global _settlement_n
    _settlement_n += 1
    outcome_type = {"FAILED_PAYMENT": "PAYMENT_SETTLED", "OVERDUE_INVOICE": "INVOICE_SETTLED"}[
        payload["risk_category"]
    ]
    res = client.post(
        "/outcomes",
        headers={"X-API-Key": "test-api-key"},
        json={
            "external_system": "phase36_settlement",
            "external_event_id": f"settle_{uuid.uuid4().hex[:8]}_{_settlement_n}",
            "domain": payload["risk_category"],
            "outcome_type": outcome_type,
            "customer_id": payload["customer_id"],
            "reference_id": payload["reference_id"],
            "amount": amount,
            "currency": payload["currency"],
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    )
    assert res.status_code == 200, res.text
    return res.json()


def _run_stages(case_id):
    for stage in (
        "assess-risk",
        "diagnose",
        "predict-recovery",
        "recommend-action",
        "policy-check",
    ):
        r = client.post(f"/cases/{case_id}/{stage}")
        assert r.status_code == 200, (stage, r.text)
    return client.get(f"/cases/{case_id}/policy-decision").json()


def _case(case_id):
    return client.get(f"/system/cases/{case_id}").json()


# ---------------------------------------------------------------------------
# A. Every action carries a rate limit
# ---------------------------------------------------------------------------


def test_A_every_action_type_has_a_channel_and_none_is_unlimited():
    """The old rule matched RETRY_* or the substring "REMINDER", leaving five
    customer-facing actions with no frequency limit whatsoever."""
    previously_unbounded = [
        ActionType.REQUEST_PAYMENT_METHOD_UPDATE,
        ActionType.SEND_PAYMENT_LINK,
        ActionType.OFFER_CHECKOUT_ASSISTANCE,
        ActionType.ESCALATE_COLLECTION,
        ActionType.REQUEST_NEW_COMMITMENT,
    ]
    for a in previously_unbounded:
        assert action_channel(a) == ActionChannel.COMMUNICATION, a
    assert action_channel(ActionType.RETRY_PAYMENT) == ActionChannel.PAYMENT
    assert action_channel(ActionType.RETRY_BILLING) == ActionChannel.PAYMENT
    # An unknown/new action must default to the rate-limited channel, never to
    # "no limit applies".
    assert action_channel("SOME_FUTURE_ACTION") == ActionChannel.COMMUNICATION


def _case_with_action(action: ActionType, amount=100.0) -> RecoveryCase:
    # Phase 4: the Policy Engine refuses an action outside its domain, so each action is
    # tested on a case of a domain it belongs to (FAILED_PAYMENT where it applies).
    category = (
        RiskCategory.FAILED_PAYMENT
        if action_applies_to(action, RiskCategory.FAILED_PAYMENT)
        else min(action_domains(action), key=lambda c: c.value)
    )
    return RecoveryCase(
        case_id="c1",
        customer_id="cust",
        risk_category=category,
        amount_at_risk=Money(amount=amount),
        recommendation=ActionRecommendation(
            recommendation_id="rec_1",
            candidates=[],
            top_candidate=CandidateAction(
                action_type=action,
                estimated_probability=0.5,
                expected_recoverable_value=50.0,
                rationale="t",
            ),
            status=RecommendationStatus.RECOMMENDED,
            rationale="t",
            engine_version="t",
        ),
    )


@pytest.mark.parametrize(
    "action",
    [
        ActionType.REQUEST_PAYMENT_METHOD_UPDATE,
        ActionType.SEND_PAYMENT_LINK,
        ActionType.ESCALATE_COLLECTION,
        ActionType.REQUEST_NEW_COMMITMENT,
        ActionType.OFFER_CHECKOUT_ASSISTANCE,
    ],
)
def test_A2_previously_unbounded_actions_now_hit_the_contact_cap(action):
    engine = DeterministicPolicyEngine(MerchantPolicy(communication_max_messages_24h=2))
    case = _case_with_action(action)
    # Under the cap: permitted.
    assert (
        engine.evaluate(case, context=PolicyContext(comms_in_last_24h=1)).status.value
        == "PERMITTED"
    )
    # At the cap: refused. Before this fix these actions matched no rule at all and
    # were permitted without bound (six consecutive real executions were observed).
    assert engine.evaluate(case, context=PolicyContext(comms_in_last_24h=2)).status.value == "WAIT"


def test_A3_contact_cap_is_reached_at_the_stated_maximum_not_one_past_it():
    """`comms` counts messages already sent, so a max of 2 must refuse the third.
    The rule used `>` and so permitted it; PaymentMaxRetries already used `>=`."""
    engine = DeterministicPolicyEngine(MerchantPolicy(communication_max_messages_24h=2))
    case = _case_with_action(ActionType.SEND_INVOICE_REMINDER)
    assert engine.evaluate(case, context=PolicyContext(comms_in_last_24h=2)).status.value == "WAIT"


# ---------------------------------------------------------------------------
# B. A dispatched action cannot be rewound
# ---------------------------------------------------------------------------


def test_B_second_event_cannot_rewind_a_case_that_already_acted():
    """Reproduced before the fix: a second failure event for the same obligation on
    the batch path reset PENDING_VERIFICATION back to POLICY_EVALUATED."""
    case_id, first = _ingest(amount=100.0)
    _run_stages(case_id)
    ex = client.post(f"/cases/{case_id}/execute").json()
    assert ex["status"] == "COMPLETED_SIMULATED"
    assert _case(case_id)["current_state"] == "PENDING_VERIFICATION"

    second = dict(first, external_event_id=f"evt_{uuid.uuid4().hex[:8]}")
    r = client.post("/events/batch", json={"auto_advance": True, "events": [second]})
    assert r.status_code == 200

    snap = _case(case_id)
    assert snap["current_state"] == "PENDING_VERIFICATION", "a dispatched case must not rewind"
    assert snap["execution_record"]["status"] == "COMPLETED_SIMULATED"
    assert snap["execution_record"]["execution_id"] == ex["execution_id"]


def test_B2_stages_are_refused_on_a_case_awaiting_verification():
    case_id, payload = _ingest(amount=100.0)
    _run_stages(case_id)
    assert client.post(f"/cases/{case_id}/execute").json()["status"] == "COMPLETED_SIMULATED"

    for stage in (
        "assess-risk",
        "diagnose",
        "predict-recovery",
        "recommend-action",
        "policy-check",
    ):
        r = client.post(f"/cases/{case_id}/{stage}")
        assert r.status_code == 400, (stage, r.status_code)

    # Verification itself must still be reachable - that is the whole point of the
    # case not being "closed" - and it reports the real settlement once one arrives.
    _settle(payload, 100.0)
    v = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""})
    assert v.status_code == 200
    assert v.json()["status"] == "FULLY_RECOVERED"


def test_B3_lifecycle_guard_names_the_dispatched_state():
    case = RecoveryCase(
        case_id="c_disp",
        customer_id="cust",
        risk_category=RiskCategory.FAILED_PAYMENT,
        amount_at_risk=Money(amount=10.0),
        current_state=CaseState.PENDING_VERIFICATION,
    )
    with pytest.raises(CaseClosedError):
        CaseLifecycleManager.record_policy_evaluation(case)
    assert case.current_state == CaseState.PENDING_VERIFICATION


# ---------------------------------------------------------------------------
# C. Human approval is revalidated, and only lifts the escalation trigger
# ---------------------------------------------------------------------------


def test_C_human_approval_satisfies_only_the_amount_threshold():
    engine = DeterministicPolicyEngine(MerchantPolicy(financial_max_automated_amount=5000.0))
    case = _case_with_action(ActionType.RETRY_PAYMENT, amount=9000.0)

    assert engine.evaluate(case, context=PolicyContext()).status.value == "ESCALATE"
    assert (
        engine.evaluate(case, context=PolicyContext(human_approved=True)).status.value
        == "PERMITTED"
    )

    # ...but approval must NOT buy past a hard limit.
    denied = engine.evaluate(
        case, context=PolicyContext(human_approved=True, prior_payment_attempts=3)
    )
    assert denied.status.value == "DENIED"
    stopped = engine.evaluate(case, context=PolicyContext(human_approved=True, stopped=True))
    assert stopped.status.value == "DENIED"


def test_C2_approved_high_value_case_executes_end_to_end():
    """The approval path must still work - this is the live-audited Tyrell flow."""
    case_id, _ = _ingest(amount=9000.0)
    pol = _run_stages(case_id)
    assert pol["status"] == "ESCALATE"

    ok = client.post(f"/cases/{case_id}/human-review", json={"decision": "APPROVE", "note": "ok"})
    assert ok.status_code == 200
    ex = client.post(f"/cases/{case_id}/execute")
    assert ex.status_code == 200
    assert ex.json()["status"] == "COMPLETED_SIMULATED", ex.json()


def test_C3_approval_is_bound_to_the_decision_that_was_reviewed():
    """Re-running the policy check mints a new decision_id, which must invalidate the
    earlier approval rather than silently carrying it forward."""
    case_id, _ = _ingest(amount=9000.0)
    _run_stages(case_id)
    client.post(f"/cases/{case_id}/human-review", json={"decision": "APPROVE", "note": "ok"})

    # A fresh policy evaluation does not inherit the approval.
    repolicy = client.post(f"/cases/{case_id}/policy-check")
    assert repolicy.status_code == 200
    assert repolicy.json()["status"] == "ESCALATE"

    blocked = client.post(f"/cases/{case_id}/execute")
    assert blocked.status_code == 200
    assert blocked.json()["status"] == "REJECTED"
    assert _case(case_id)["execution_record"]["status"] == "REJECTED"


# ---------------------------------------------------------------------------
# D. A stop is honoured for the obligation
# ---------------------------------------------------------------------------


def test_D_stop_rule_is_live_and_denies_further_action():
    case_id, _ = _ingest(amount=100.0)
    _run_stages(case_id)
    assert client.post(f"/cases/{case_id}/stop").status_code == 200
    # PolicyContext.stopped was never populated before, so StopRule was dead code.
    engine = DeterministicPolicyEngine()
    case = _case_with_action(ActionType.RETRY_PAYMENT)
    d = engine.evaluate(case, context=PolicyContext(stopped=True))
    assert d.status.value == "DENIED"
    assert any(r.rule_name == "StopRule" for r in d.failed_rules)


def test_D2_stopping_a_settled_case_is_refused_and_restopping_is_idempotent():
    case_id, payload = _ingest(amount=100.0)
    _run_stages(case_id)
    client.post(f"/cases/{case_id}/execute")
    _settle(payload, 100.0)
    assert _case(case_id)["current_state"] == "FULLY_RECOVERED"

    # Stamping STOPPED here would discard the terminal state while the outcome still
    # says the money came back.
    r = client.post(f"/cases/{case_id}/stop")
    assert r.status_code == 409
    assert _case(case_id)["current_state"] == "FULLY_RECOVERED"

    other, _ = _ingest(amount=100.0)
    assert client.post(f"/cases/{other}/stop").status_code == 200
    assert client.post(f"/cases/{other}/stop").status_code == 200  # idempotent
    assert _case(other)["current_state"] == "STOPPED"


# ---------------------------------------------------------------------------
# E. Closed-case handling is consistent
# ---------------------------------------------------------------------------


def test_E_advance_treats_every_closed_state_the_same_way():
    """PARTIALLY_RECOVERED and DENIED were missing from advance()'s inline closed
    set, so they fell through to the stage guard and escaped as an unhandled HTTP
    500, while FULLY_RECOVERED got a polite no-op. All closed states now behave
    identically - a safe no-op that leaves the settled case untouched."""
    partial, partial_payload = _ingest(amount=100.0)
    _run_stages(partial)
    client.post(f"/cases/{partial}/execute")
    _settle(partial_payload, 40.0)
    assert _case(partial)["current_state"] == "PARTIALLY_RECOVERED"

    full, full_payload = _ingest(amount=100.0)
    _run_stages(full)
    client.post(f"/cases/{full}/execute")
    _settle(full_payload, 100.0)
    assert _case(full)["current_state"] == "FULLY_RECOVERED"

    for case_id, expected in ((partial, "PARTIALLY_RECOVERED"), (full, "FULLY_RECOVERED")):
        r = client.post(f"/cases/{case_id}/advance")
        assert r.status_code == 200, (case_id, r.status_code, r.text)
        assert _case(case_id)["current_state"] == expected, "a settled case must not be reprocessed"


def test_E2_prediction_is_refused_on_a_settled_case():
    case_id, _ = _ingest(amount=100.0)
    _run_stages(case_id)
    client.post(f"/cases/{case_id}/execute")
    client.post(f"/cases/{case_id}/verify", json={"external_reference": "sim_full"})
    r = client.post(f"/cases/{case_id}/predict-recovery")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# F. The ledger is a complete action history
# ---------------------------------------------------------------------------


def test_F_execution_ledger_records_provenance_for_every_attempt():
    case_id, _ = _ingest(amount=100.0)
    _run_stages(case_id)
    ex = client.post(f"/cases/{case_id}/execute").json()

    rows = client.get("/system/executions?limit=100").json()["items"]
    mine = [r for r in rows if r["case_id"] == case_id]
    assert mine, "the ledger, not the single overwritten case column, backs this view"
    assert mine[0]["execution_id"] == ex["execution_id"]
    assert mine[0]["agent_type"] == ex["agent_type"]


def test_F2_repeated_execute_under_one_authorization_stays_a_single_attempt():
    case_id, _ = _ingest(amount=100.0)
    _run_stages(case_id)
    first = client.post(f"/cases/{case_id}/execute").json()
    again = client.post(f"/cases/{case_id}/execute").json()
    assert again["execution_id"] == first["execution_id"]

    rows = client.get("/system/executions?limit=100").json()["items"]
    completed = [
        r for r in rows if r["case_id"] == case_id and r["status"] == "COMPLETED_SIMULATED"
    ]
    assert len(completed) == 1, "one authorization must yield exactly one dispatched action"
