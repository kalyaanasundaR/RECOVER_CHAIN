"""Phase 1 controlled-implementation regression tests, updated for Phase 5.

Covers the two verified audit findings and nothing else:

1. Verification idempotency — a case that has already reached a *final* outcome
   (anything other than PENDING_VERIFICATION) reports that exact persisted result on
   every subsequent read; POST /cases/{id}/verify never draws a new one. Phase 5
   changed WHERE the outcome comes from (an authoritative outcome event, POST
   /outcomes - never a client-supplied reference) but not this guarantee.

2. Lifecycle authority — case-state mutation goes through
   domain.lifecycle.CaseLifecycleManager. A case that has already reached a
   closed (terminal) state can no longer be silently reprocessed by the core
   pipeline stages.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import app
from domain.lifecycle import CaseClosedError, CaseLifecycleManager
from domain.models import CaseState, Money, RecoveryCase, RiskCategory

client = TestClient(app)


def _new_case_at_policy_check(
    amount: float, failure_code: str = "network_error"
) -> tuple[str, str, dict]:
    """Ingest one event and drive it through assess -> diagnose -> predict ->
    recommend -> policy-check. Returns (case_id, reference_id, policy_decision_json)
    with no assumption about which way policy decided.

    `network_error` deterministically diagnoses as NETWORK_FAILURE, whose only
    recommended action is RETRY_PAYMENT (see application/action_evaluator.py),
    so the amount alone controls whether policy PERMITs or ESCALATEs.
    """
    reference_id = f"ref_{uuid.uuid4().hex[:8]}"
    payload = {
        "customer_id": f"cust_{uuid.uuid4().hex[:6]}",
        "risk_category": "FAILED_PAYMENT",
        "external_system": "phase35_test",
        "external_event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "reference_id": reference_id,
        "amount": amount,
        "currency": "USD",
        "raw_payload": {"failure_code": failure_code},
    }
    ingest = client.post("/events", json=payload)
    assert ingest.status_code == 200, ingest.text
    case_id = ingest.json()["case_id"]

    assert client.post(f"/cases/{case_id}/assess-risk").status_code == 200
    assert client.post(f"/cases/{case_id}/diagnose").status_code == 200
    assert client.post(f"/cases/{case_id}/predict-recovery").status_code == 200
    assert client.post(f"/cases/{case_id}/recommend-action").status_code == 200

    pol = client.post(f"/cases/{case_id}/policy-check")
    assert pol.status_code == 200, pol.text
    return case_id, reference_id, pol.json()


def _walk_case_to_pending_verification(amount: float = 100.0) -> tuple[str, str]:
    """A low-amount, zero-prior-attempts case: policy PERMITs, execution
    completes (simulated), and the case is ready for verification."""
    case_id, reference_id, pol = _new_case_at_policy_check(amount)
    assert pol["status"] == "PERMITTED", pol

    ex = client.post(f"/cases/{case_id}/execute")
    assert ex.status_code == 200, ex.text
    assert ex.json()["status"] == "COMPLETED_SIMULATED", ex.json()
    return case_id, reference_id


_settlement_n = 0


def _settle(case_id: str, reference_id: str, amount: float, currency: str = "USD") -> dict:
    """Post an authoritative PAYMENT_SETTLED outcome event for this case's obligation
    and return the reconciled position (POST /outcomes). Replaces the removed
    sim_full/sim_partial/sim_fail client-supplied-reference behaviour of
    POST /cases/{id}/verify."""
    global _settlement_n
    _settlement_n += 1
    customer_id = client.get(f"/system/cases/{case_id}").json()["customer_id"]
    res = client.post(
        "/outcomes",
        headers={"X-API-Key": "test-api-key"},
        json={
            "external_system": "phase35_settlement",
            "external_event_id": f"settle_{uuid.uuid4().hex[:8]}_{_settlement_n}",
            "domain": "FAILED_PAYMENT",
            "outcome_type": "PAYMENT_SETTLED",
            "customer_id": customer_id,
            "reference_id": reference_id,
            "amount": amount,
            "currency": currency,
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    )
    assert res.status_code == 200, res.text
    return res.json()


# ---------------------------------------------------------------------------
# Finding #1 — verification idempotency
# ---------------------------------------------------------------------------


def test_A_first_verification_persists_a_deterministic_result():
    case_id, reference_id = _walk_case_to_pending_verification(amount=100.0)
    settled = _settle(case_id, reference_id, 100.0)
    assert settled["match_status"] == "MATCHED" and settled["applied_amount"] == "100.0000"

    resp = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FULLY_RECOVERED"
    assert data["actual_amount_recovered"] == "100.0000"

    snap = client.get(f"/system/cases/{case_id}").json()
    assert snap["outcome"]["status"] == "FULLY_RECOVERED"
    assert snap["outcome"]["actual_amount_recovered"]["amount"] == "100.0000"
    assert snap["current_state"] == "FULLY_RECOVERED"


def test_B_repeated_verification_returns_the_exact_same_result():
    case_id, reference_id = _walk_case_to_pending_verification(amount=100.0)
    _settle(case_id, reference_id, 100.0)
    first = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""}).json()

    # A second read — even carrying an ignored, arbitrary reference, as a careless
    # retry or a stale frontend re-trigger might send — must not change anything: a
    # duplicate settlement for the same obligation is rejected as a duplicate
    # external_event_id long before it would reach reconciliation.
    dup = _settle(case_id, reference_id, 999.0)
    assert dup["status"] == "processed"  # a *different* external id is its own event...
    # ...but it arrives on an already FULLY_RECOVERED case, so it changes nothing:
    assert dup["effect"] == "EXCESS"
    second = client.post(f"/cases/{case_id}/verify", json={"external_reference": "ignored"}).json()

    assert second["outcome_id"] == first["outcome_id"]
    assert second["status"] == first["status"] == "FULLY_RECOVERED"
    assert second["actual_amount_recovered"] == first["actual_amount_recovered"] == "100.0000"
    assert second["timestamp"] == first["timestamp"], (
        "original verification timestamp must stay authoritative"
    )


def test_C_reload_shows_the_same_persisted_result_every_time():
    """Regression test for the exact live-audit symptom: reloading the result
    screen must never show a different recovered amount."""
    case_id, reference_id = _walk_case_to_pending_verification(amount=100.0)
    _settle(case_id, reference_id, 100.0)

    reads = [
        client.get(f"/system/cases/{case_id}").json()["outcome"]["actual_amount_recovered"][
            "amount"
        ]
        for _ in range(5)
    ]
    assert reads == ["100.0000"] * 5


def test_D_repeated_verification_never_mutates_the_financial_outcome():
    case_id, reference_id = _walk_case_to_pending_verification(amount=100.0)
    _settle(case_id, reference_id, 100.0)

    for _ in range(5):
        client.post(f"/cases/{case_id}/verify", json={"external_reference": ""})

    audits = client.get(f"/cases/{case_id}/audit").json()
    reconciliations = [a for a in audits if a["evidence"].get("action") == "outcome_reconciled"]
    assert len(reconciliations) == 1, "exactly one real determination, ever"
    # /verify is now a pure read (Phase 5): repeating it writes no audit at all, and
    # produces no new draw - checked directly against the persisted outcome below.

    snap = client.get(f"/system/cases/{case_id}").json()
    assert snap["outcome"]["actual_amount_recovered"]["amount"] == "100.0000"
    assert snap["current_state"] == "FULLY_RECOVERED"


def test_pending_outcome_is_not_frozen_but_the_final_answer_is():
    """PENDING_VERIFICATION is explicitly NOT 'already verified' — a
    follow-up check must still be able to reach a real determination. Once it
    does, *that* result freezes like any other."""
    case_id, reference_id = _walk_case_to_pending_verification(amount=100.0)

    pending = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""}).json()
    assert pending["status"] == "PENDING_VERIFICATION"

    _settle(case_id, reference_id, 100.0)
    final = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""}).json()
    assert final["status"] == "FULLY_RECOVERED"

    again = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""}).json()
    assert again["status"] == "FULLY_RECOVERED"
    assert again["outcome_id"] == final["outcome_id"]


# ---------------------------------------------------------------------------
# Finding #2 — lifecycle transitions are authoritative
# ---------------------------------------------------------------------------


def test_E_valid_lifecycle_transition_runs_through_the_domain_layer():
    case = RecoveryCase(
        case_id="case_e",
        customer_id="cust_e",
        risk_category=RiskCategory.FAILED_PAYMENT,
        amount_at_risk=Money(amount=100.0),
        current_state=CaseState.OPEN,
    )
    CaseLifecycleManager.record_risk_assessment(case)
    assert case.current_state == CaseState.ASSESSED


def test_F_invalid_lifecycle_transition_is_rejected_on_a_closed_case():
    case = RecoveryCase(
        case_id="case_f",
        customer_id="cust_f",
        risk_category=RiskCategory.FAILED_PAYMENT,
        amount_at_risk=Money(amount=100.0),
        current_state=CaseState.FULLY_RECOVERED,
    )
    for transition in (
        CaseLifecycleManager.record_risk_assessment,
        CaseLifecycleManager.record_diagnosis_result,
        CaseLifecycleManager.record_recommendation_result,
        CaseLifecycleManager.record_policy_evaluation,
    ):
        with pytest.raises(CaseClosedError):
            transition(case)
    assert case.current_state == CaseState.FULLY_RECOVERED, "closed case must stay untouched"

    # The pre-existing OPEN -> ... -> POLICY_REVIEW family (used at case
    # creation) is untouched by this phase and keeps its own guard.
    with pytest.raises(ValueError):
        CaseLifecycleManager.record_diagnosis(case)


def test_F2_api_rejects_reprocessing_a_case_that_already_closed():
    case_id, reference_id = _walk_case_to_pending_verification(amount=100.0)
    _settle(case_id, reference_id, 100.0)
    assert client.get(f"/system/cases/{case_id}").json()["current_state"] == "FULLY_RECOVERED"

    resp = client.post(f"/cases/{case_id}/diagnose")
    assert resp.status_code == 400

    still = client.get(f"/system/cases/{case_id}").json()
    assert still["current_state"] == "FULLY_RECOVERED", (
        "rejected call must not overwrite the closed case"
    )


def test_G_stop_route_still_works_through_the_lifecycle_layer():
    case_id, _reference_id, _pol = _new_case_at_policy_check(amount=100.0)
    resp = client.post(f"/cases/{case_id}/stop", params={"reason": "customer_opt_out"})
    assert resp.status_code == 200
    assert resp.json()["current_state"] == "STOPPED"
    # Stopping remains permitted from any state — including one already
    # somewhere mid-pipeline — matching existing (unrestricted) behavior.


def test_G2_human_review_approve_then_execute_then_verify_still_works():
    # A high amount forces ESCALATE so human review is reachable, exactly
    # like the live-audited Tyrell (Rs. 15,000) case.
    case_id, reference_id, pol = _new_case_at_policy_check(amount=9000.0)
    assert pol["status"] == "ESCALATE"

    approve = client.post(
        f"/cases/{case_id}/human-review", json={"decision": "APPROVE", "note": "ok"}
    )
    assert approve.status_code == 200
    assert approve.json()["decision"] == "PERMITTED"

    ex = client.post(f"/cases/{case_id}/execute")
    assert ex.status_code == 200
    assert ex.json()["status"] == "COMPLETED_SIMULATED"

    _settle(case_id, reference_id, 4500.0)  # half of 9000: a real partial settlement
    verify = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""})
    assert verify.status_code == 200
    assert verify.json()["status"] == "PARTIALLY_RECOVERED"


def test_G3_human_review_reject_leaves_state_unchanged_and_blocks_execution():
    case_id, _reference_id, pol = _new_case_at_policy_check(amount=9000.0)
    assert pol["status"] == "ESCALATE"
    state_before = client.get(f"/system/cases/{case_id}").json()["current_state"]

    reject = client.post(
        f"/cases/{case_id}/human-review", json={"decision": "REJECT", "note": "no"}
    )
    assert reject.status_code == 200
    assert reject.json()["decision"] == "DENIED"

    state_after = client.get(f"/system/cases/{case_id}").json()["current_state"]
    assert state_after == state_before, "rejection leaves current_state exactly as it was"

    ex = client.post(f"/cases/{case_id}/execute")
    assert ex.status_code == 200
    assert ex.json()["status"] == "REJECTED"


# ---------------------------------------------------------------------------
# Test H — no regression in existing Policy Engine behavior
# ---------------------------------------------------------------------------


def test_H_policy_engine_still_escalates_above_the_automated_threshold():
    _, _reference_id, pol = _new_case_at_policy_check(amount=9000.0)
    assert pol["status"] == "ESCALATE"
    assert any(
        r["rule_name"] == "FinancialAutomatedLimit" and not r["passed"] for r in pol["failed_rules"]
    )


def test_H2_policy_engine_still_permits_a_low_amount_first_attempt():
    _, _reference_id, pol = _new_case_at_policy_check(amount=100.0)
    assert pol["status"] == "PERMITTED"
    assert pol["failed_rules"] == []
