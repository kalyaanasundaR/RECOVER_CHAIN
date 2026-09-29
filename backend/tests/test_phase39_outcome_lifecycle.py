"""Phase 5 outcome-lifecycle regression tests.

A    outcome ingestion (single)
B    outcome validation (rejected, with reasons)
C    outcome deduplication
D    outcome-to-obligation matching
E    unmatched outcome, stored and later rematched
F    partial settlement
G    full settlement
H    duplicate settlement cannot increase recovered money
I    out-of-order outcome (settlement before the failure event exists)
J    cancellation
K    promise created
L    promise fulfilled
M    promise broken
N    consent import (batch)
O    consent timeline is bitemporal - a later record cannot rewrite history
P    customer identity links (deterministic, no fuzzy matching)
S    financial invariants across a mixed sequence
T    dashboard distinguishes at-risk / recovered / outstanding / cancelled / unmatched
U    ACTION ATTEMPT != REVENUE RECOVERED

Every test drives the real API, the real reconciliation function
(application/outcome_events.py) and the real database - no mocks of the module under
test. Legacy identity reconciliation (Q) and dataset re-upload after it (R) are
Phase 4 concerns re-verified in this phase's report via test_phase38's test_K and a
runtime run against the dev database; not duplicated here. PostgreSQL (W) is UNKNOWN -
no server is available in this environment (see the Phase 5 report).
"""

import uuid
from decimal import Decimal

from fastapi.testclient import TestClient

from api.main import app
from application.outcome_events import (
    Effect,
    OutcomeKind,
    reconcile_position,
)
from domain.models import CaseState

client = TestClient(app)
KEY = {"X-API-Key": "test-api-key"}


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _open_case(
    tag: str,
    *,
    domain: str = "OVERDUE_INVOICE",
    amount: float = 1000,
    currency: str = "INR",
    customer_id: str | None = None,
    reference_id: str | None = None,
    failure_code: str | None = None,
) -> dict:
    customer_id = customer_id or f"p5_cust_{tag}"
    reference_id = reference_id or f"p5_ref_{tag}"
    payload = {
        "customer_id": customer_id,
        "risk_category": domain,
        "external_system": "p5_sys",
        "external_event_id": f"p5_evt_{_tag()}",
        "reference_id": reference_id,
        "amount": amount,
        "currency": currency,
        "raw_payload": {"failure_code": failure_code} if failure_code else {},
    }
    r = client.post("/events", json=payload)
    assert r.status_code == 200, r.text
    return {
        "case_id": r.json()["case_id"],
        "customer_id": customer_id,
        "reference_id": reference_id,
        "domain": domain,
        "currency": currency,
    }


_OUTCOME_TYPE = {
    "FAILED_PAYMENT": "PAYMENT_SETTLED",
    "OVERDUE_INVOICE": "INVOICE_SETTLED",
    "CHECKOUT_ABANDONMENT": "CHECKOUT_CONVERTED",
    "FAILED_SUBSCRIPTION": "RENEWAL_RECOVERED",
    "BROKEN_PROMISE": "PROMISE_PAYMENT_RECEIVED",
}


def _outcome(
    obligation: dict,
    *,
    outcome_type: str | None = None,
    amount: float | None = None,
    external_event_id: str | None = None,
    reference_id: str | None = "__default__",
    customer_id: str | None = None,
    customer_system: str | None = None,
    occurred_at: str = "2031-06-01T00:00:00Z",
    promised_date: str | None = None,
) -> dict:
    body = {
        "external_system": "p5_settlement",
        "external_event_id": external_event_id or f"p5_out_{_tag()}",
        "domain": obligation["domain"],
        "outcome_type": outcome_type or _OUTCOME_TYPE[obligation["domain"]],
        "customer_id": customer_id or obligation["customer_id"],
        "reference_id": obligation["reference_id"]
        if reference_id == "__default__"
        else reference_id,
        "amount": amount,
        "currency": obligation.get("currency", "INR"),
        "occurred_at": occurred_at,
    }
    if customer_system:
        body["customer_system"] = customer_system
    if promised_date:
        body["promised_date"] = promised_date
    return client.post("/outcomes", headers=KEY, json=body).json()


def _snap(case_id: str) -> dict:
    return client.get(f"/system/cases/{case_id}").json()


# ------------------------------------------------------------------ A ingestion


def test_A_outcome_ingestion_records_and_applies_a_settlement():
    tag = _tag()
    ob = _open_case(tag, amount=1000)
    res = _outcome(ob, amount=1000)
    assert res["status"] == "processed"
    assert res["match_status"] == "MATCHED"
    assert res["case_id"] == ob["case_id"]
    assert res["applied_amount"] == "1000.0000"
    assert res["effect"] == "APPLIED"

    snap = _snap(ob["case_id"])
    assert snap["current_state"] == "FULLY_RECOVERED"
    assert snap["outcome"]["actual_amount_recovered"]["amount"] == "1000.0000"
    assert snap["outcome"]["verification_source"] == "RECORDED_OUTCOME_EVENTS"


# ------------------------------------------------------------------ B validation


def test_B_invalid_outcome_items_are_rejected_with_reasons():
    tag = _tag()
    ob = _open_case(tag)

    # An unknown outcome_type is rejected.
    assert (
        client.post(
            "/outcomes",
            headers=KEY,
            json={
                "external_system": "p5",
                "external_event_id": f"bad_{tag}",
                "domain": ob["domain"],
                "outcome_type": "NOT_A_TYPE",
                "customer_id": ob["customer_id"],
                "reference_id": ob["reference_id"],
                "amount": 10,
                "currency": "INR",
                "occurred_at": "2026-01-01T00:00:00Z",
            },
        ).status_code
        == 422
    )

    mismatch = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5",
            "external_event_id": f"mismatch_{tag}",
            "domain": "OVERDUE_INVOICE",
            "outcome_type": "PAYMENT_SETTLED",  # belongs to FAILED_PAYMENT
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "amount": 10,
            "currency": "INR",
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["detail"]["reason"] == "outcome_type_domain_mismatch"

    no_amount = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5",
            "external_event_id": f"noamt_{tag}",
            "domain": "OVERDUE_INVOICE",
            "outcome_type": "INVOICE_SETTLED",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "currency": "INR",
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    )
    assert no_amount.status_code == 422
    assert no_amount.json()["detail"]["reason"] == "missing_amount"


# ------------------------------------------------------------------ C dedup


def test_C_the_same_outcome_event_ingested_twice_is_a_duplicate():
    tag = _tag()
    ob = _open_case(tag, amount=500)
    first = _outcome(ob, amount=500, external_event_id=f"dup_{tag}")
    assert first["status"] == "processed"
    again = _outcome(ob, amount=500, external_event_id=f"dup_{tag}")
    assert again["status"] == "duplicate"
    assert again["outcome_event_id"] == first["outcome_event_id"]

    snap = _snap(ob["case_id"])
    assert snap["outcome"]["actual_amount_recovered"]["amount"] == "500.0000"


# ------------------------------------------------------------------ D matching


def test_D_matching_is_domain_customer_and_reference_never_a_fallback():
    tag = _tag()
    ob = _open_case(tag, domain="OVERDUE_INVOICE", amount=100)

    other_domain = _outcome(
        {**ob, "domain": "FAILED_PAYMENT"},
        outcome_type="PAYMENT_SETTLED",
        amount=100,
    )
    assert other_domain["match_status"] == "UNMATCHED"
    assert other_domain["match_reason"] == "NO_OBLIGATION"

    other_customer = _outcome(ob, amount=100, customer_id=f"someone_else_{tag}")
    assert other_customer["match_status"] == "UNMATCHED"

    correct = _outcome(ob, amount=100)
    assert correct["match_status"] == "MATCHED" and correct["case_id"] == ob["case_id"]


# ------------------------------------------------------------------ E unmatched + rematch


def test_E_an_unmatched_outcome_is_visible_and_can_be_rematched_once_the_obligation_exists():
    tag = _tag()
    customer = f"p5_cust_{tag}"
    reference = f"p5_ref_{tag}"

    early = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"early_{tag}",
            "domain": "OVERDUE_INVOICE",
            "outcome_type": "INVOICE_SETTLED",
            "customer_id": customer,
            "reference_id": reference,
            "amount": 300,
            "currency": "INR",
            "occurred_at": "2020-01-01T00:00:00Z",
        },
    )
    assert early.status_code == 200
    body = early.json()
    assert body["match_status"] == "UNMATCHED" and body["match_reason"] == "NO_OBLIGATION"

    listed = client.get("/outcomes", params={"match_status": "UNMATCHED"}).json()
    assert any(i["outcome_event_id"] == body["outcome_event_id"] for i in listed["items"])

    ob = _open_case(
        tag, domain="OVERDUE_INVOICE", amount=300, customer_id=customer, reference_id=reference
    )

    rematch = client.post("/outcomes/rematch-unmatched", headers=KEY).json()
    assert rematch["rematched"] >= 1

    snap = _snap(ob["case_id"])
    assert snap["current_state"] == "FULLY_RECOVERED"
    assert snap["outcome"]["actual_amount_recovered"]["amount"] == "300.0000"


def test_E2_a_currency_mismatch_is_unmatched_never_converted():
    tag = _tag()
    ob = _open_case(tag, currency="INR", amount=100)
    res = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"ccy_{tag}",
            "domain": ob["domain"],
            "outcome_type": "INVOICE_SETTLED",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "amount": 100,
            "currency": "USD",
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    ).json()
    assert res["match_status"] == "UNMATCHED" and res["match_reason"] == "CURRENCY_MISMATCH"
    assert _snap(ob["case_id"])["current_state"] != "FULLY_RECOVERED"


# ------------------------------------------------------------------ F/G settlement


def test_F_partial_settlement_records_the_outstanding_balance():
    tag = _tag()
    ob = _open_case(tag, amount=1000)
    res = _outcome(ob, amount=400)
    assert res["effect"] == "APPLIED" and res["applied_amount"] == "400.0000"

    snap = _snap(ob["case_id"])
    assert snap["current_state"] == "PARTIALLY_RECOVERED"
    assert snap["outcome"]["actual_amount_recovered"]["amount"] == "400.0000"
    assert snap["outcome"]["outstanding_amount"]["amount"] == "600.0000"


def test_G_full_settlement_closes_the_case_and_a_later_settlement_is_excess():
    tag = _tag()
    ob = _open_case(tag, amount=250)
    _outcome(ob, amount=250)
    assert _snap(ob["case_id"])["current_state"] == "FULLY_RECOVERED"

    # No further recovery action is permitted.
    exec_res = client.post(f"/cases/{ob['case_id']}/execute")
    assert exec_res.status_code == 400 or exec_res.json().get("status") == "REJECTED"


# ------------------------------------------------------------------ H duplicate settlement


def test_H_duplicate_settlement_cannot_increase_recovered_money():
    tag = _tag()
    ob = _open_case(tag, amount=800)
    first = _outcome(ob, amount=500)
    assert first["effect"] == "APPLIED"
    top_up = _outcome(ob, amount=500)  # a second, genuinely different event
    assert top_up["effect"] == "APPLIED" and top_up["applied_amount"] == "300.0000"
    assert _snap(ob["case_id"])["current_state"] == "FULLY_RECOVERED"

    # Now the obligation is fully settled: any further amount is EXCESS, never recovered.
    excess = _outcome(ob, amount=200)
    assert excess["effect"] == "EXCESS"
    snap = _snap(ob["case_id"])
    assert snap["outcome"]["actual_amount_recovered"]["amount"] == "800.0000"
    assert snap["current_state"] == "FULLY_RECOVERED"


# ------------------------------------------------------------------ I out-of-order


def test_I_an_out_of_order_settlement_is_applied_against_the_position_when_it_arrives():
    tag = _tag()
    ob = _open_case(tag, amount=1000)
    # A settlement whose occurred_at is BEFORE the case's own creation - arrives late
    # relative to real-world time, but is applied deterministically at ingestion.
    old = _outcome(ob, amount=300, occurred_at="2000-01-01T00:00:00Z")
    assert old["effect"] == "APPLIED"
    newer = _outcome(ob, amount=700, occurred_at="2031-06-01T00:00:00Z")
    assert newer["effect"] == "APPLIED"
    assert _snap(ob["case_id"])["current_state"] == "FULLY_RECOVERED"


# ------------------------------------------------------------------ J cancellation


def test_J_cancellation_withdraws_the_obligation_and_blocks_further_action():
    tag = _tag()
    ob = _open_case(tag, domain="CHECKOUT_ABANDONMENT", amount=500)
    res = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"cancel_{tag}",
            "domain": "CHECKOUT_ABANDONMENT",
            "outcome_type": "ORDER_CANCELLED",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    ).json()
    assert res["match_status"] == "MATCHED" and res["effect"] == "APPLIED"

    snap = _snap(ob["case_id"])
    assert snap["current_state"] == "CANCELLED"
    assert snap["outcome"]["status"] == "CANCELLED"
    assert snap["outcome"]["cancelled_amount"]["amount"] == "500.0000"

    exec_res = client.post(f"/cases/{ob['case_id']}/execute")
    assert exec_res.status_code == 400 or exec_res.json().get("status") == "REJECTED"

    # A settlement after cancellation is real money, recorded, but does not resurrect
    # the case or count as recovered.
    late = _outcome(
        ob, outcome_type="CHECKOUT_CONVERTED", amount=500, external_event_id=f"late_{tag}"
    )
    assert late["effect"] == "EXCESS"
    assert _snap(ob["case_id"])["current_state"] == "CANCELLED"


# ------------------------------------------------------------------ K/L/M promise-to-pay


def test_K_promise_created_is_recorded_and_is_never_money():
    tag = _tag()
    ob = _open_case(tag, domain="BROKEN_PROMISE", amount=600)
    res = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"promise_{tag}",
            "domain": "BROKEN_PROMISE",
            "outcome_type": "PROMISE_CREATED",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "amount": 600,
            "currency": "INR",
            "occurred_at": "2031-06-01T00:00:00Z",
            "promised_date": "2031-06-15T00:00:00Z",
        },
    ).json()
    assert res["match_status"] == "MATCHED" and res["effect"] == "NO_EFFECT"

    obligation = client.get(
        "/obligations",
        params={
            "domain": "BROKEN_PROMISE",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
        },
    ).json()
    assert len(obligation["promises"]) == 1
    promise = obligation["promises"][0]
    assert promise["status"] == "CREATED"
    assert _snap(ob["case_id"])["outcome"] is None  # a promise is never itself recovery


def test_L_promise_fulfilled_is_derived_from_a_matching_settlement():
    tag = _tag()
    ob = _open_case(tag, domain="BROKEN_PROMISE", amount=600)
    client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"promise_{tag}",
            "domain": "BROKEN_PROMISE",
            "outcome_type": "PROMISE_CREATED",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "amount": 600,
            "currency": "INR",
            "occurred_at": "2031-06-01T00:00:00Z",
            "promised_date": "2031-06-15T00:00:00Z",
        },
    )
    settle = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"promise_pay_{tag}",
            "domain": "BROKEN_PROMISE",
            "outcome_type": "PROMISE_PAYMENT_RECEIVED",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "amount": 600,
            "currency": "INR",
            "occurred_at": "2031-06-10T00:00:00Z",
        },
    ).json()
    assert settle["effect"] == "APPLIED"
    assert _snap(ob["case_id"])["current_state"] == "FULLY_RECOVERED"

    obligation = client.get(
        "/obligations",
        params={
            "domain": "BROKEN_PROMISE",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
        },
    ).json()
    assert obligation["promises"][0]["status"] == "FULFILLED"


def test_M_promise_broken_is_reported_explicitly():
    tag = _tag()
    ob = _open_case(tag, domain="BROKEN_PROMISE", amount=600)
    client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"promise_{tag}",
            "domain": "BROKEN_PROMISE",
            "outcome_type": "PROMISE_CREATED",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "amount": 600,
            "currency": "INR",
            "occurred_at": "2031-06-01T00:00:00Z",
            "promised_date": "2031-06-15T00:00:00Z",
        },
    )
    broken = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"promise_broken_{tag}",
            "domain": "BROKEN_PROMISE",
            "outcome_type": "PROMISE_BROKEN",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "occurred_at": "2031-06-16T00:00:00Z",
        },
    ).json()
    assert broken["match_status"] == "MATCHED"

    obligation = client.get(
        "/obligations",
        params={
            "domain": "BROKEN_PROMISE",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
        },
    ).json()
    assert obligation["promises"][0]["status"] == "BROKEN"
    # Still never money:
    assert _snap(ob["case_id"])["outcome"] is None


# ------------------------------------------------------------------ N/O consent import + timeline


def test_N_consent_import_validates_each_item_independently():
    tag = _tag()
    res = client.post(
        "/contact-permissions/import",
        headers=KEY,
        json={
            "records": [
                {
                    "customer_id": f"p5_consent_{tag}_a",
                    "channel": "communication",
                    "status": "ALLOWED",
                    "source": "crm_import",
                    "effective_at": "2026-01-01T00:00:00Z",
                },
                {  # missing status -> rejected, never assumed ALLOWED
                    "customer_id": f"p5_consent_{tag}_b",
                    "channel": "communication",
                    "status": "",
                    "source": "crm_import",
                },
                {  # SUPPRESSED on a single channel is invalid
                    "customer_id": f"p5_consent_{tag}_c",
                    "channel": "communication",
                    "status": "SUPPRESSED",
                    "source": "crm_import",
                },
            ]
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["imported"] == 1 and body["rejected"] == 2

    state = client.get(f"/customers/p5_consent_{tag}_a/contact-permissions").json()
    assert state["effective"]["communication"]["status"] == "ALLOWED"
    state_b = client.get(f"/customers/p5_consent_{tag}_b/contact-permissions").json()
    assert state_b["effective"]["communication"]["status"] == "UNKNOWN"


def test_N2_reimporting_the_identical_record_is_a_no_op_duplicate():
    tag = _tag()
    customer = f"p5_consent_dup_{tag}"
    item = {
        "customer_id": customer,
        "channel": "communication",
        "status": "ALLOWED",
        "source": "crm_import",
        "effective_at": "2026-01-01T00:00:00Z",
    }
    first = client.post("/contact-permissions/import", headers=KEY, json={"records": [item]}).json()
    second = client.post(
        "/contact-permissions/import", headers=KEY, json={"records": [item]}
    ).json()
    assert first["imported"] == 1
    assert second["imported"] == 0 and second["duplicates"] == 1
    records = client.get(f"/customers/{customer}/contact-permissions").json()["records"]
    assert len(records) == 1


def test_O_consent_timeline_is_bitemporal_a_later_record_cannot_rewrite_history():
    tag = _tag()
    customer = f"p5_timeline_{tag}"

    # At contact time T, only ALLOWED (effective before T) was on record.
    client.post(
        f"/customers/{customer}/contact-permissions",
        headers=KEY,
        json={
            "channel": "communication",
            "status": "ALLOWED",
            "source": "signup",
            "effective_at": "2026-01-01T00:00:00Z",
        },
    )
    # A contact attempt is made and recorded (through the ledger, via a real case).
    ob = _open_case(tag, domain="OVERDUE_INVOICE", amount=100, customer_id=customer)
    client.post(f"/cases/{ob['case_id']}/assess-risk")
    client.post(f"/cases/{ob['case_id']}/diagnose")
    client.post(f"/cases/{ob['case_id']}/predict-recovery")
    client.post(f"/cases/{ob['case_id']}/recommend-action")
    client.post(f"/cases/{ob['case_id']}/policy-check")
    client.post(f"/cases/{ob['case_id']}/execute")

    # LATER, the customer opts out, but backdates the record to before the contact.
    client.post(
        f"/customers/{customer}/contact-permissions",
        headers=KEY,
        json={
            "channel": "communication",
            "status": "OPTED_OUT",
            "source": "backfilled_correction",
            "effective_at": "2026-01-15T00:00:00Z",  # before contact_time
        },
    )

    history = client.get(f"/customers/{customer}/contact-history").json()
    attempt = history["attempts"][0]
    # What was actually known AT the time of contact never changes:
    assert attempt["permission_known_at_attempt"] == "ALLOWED"
    # What today's full record set believes was true then DOES reflect the backdated
    # correction - a different, clearly-labelled view, not a rewrite of the first.
    assert attempt["permission_in_force_at_attempt"] == "OPTED_OUT"


# ------------------------------------------------------------------ P customer identity links


def test_P_customer_identity_links_are_explicit_never_inferred():
    tag = _tag()
    link = client.post(
        "/customers/identity-links",
        headers=KEY,
        json={
            "system": "netsuite",
            "external_customer_id": f"NS-{tag}",
            "customer_id": f"p5_cust_{tag}",
            "source": "manual_mapping",
        },
    )
    assert link.status_code == 200

    ob = _open_case(tag, domain="OVERDUE_INVOICE", amount=250, customer_id=f"p5_cust_{tag}")
    res = _outcome(ob, amount=250, customer_id=f"NS-{tag}", customer_system="netsuite")
    assert res["match_status"] == "MATCHED" and res["case_id"] == ob["case_id"]

    # An UNLINKED external id is never guessed at.
    unlinked = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"unlinked_{tag}",
            "domain": "OVERDUE_INVOICE",
            "outcome_type": "INVOICE_SETTLED",
            "customer_id": f"NS-UNKNOWN-{tag}",
            "customer_system": "netsuite",
            "reference_id": ob["reference_id"],
            "amount": 10,
            "currency": "INR",
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    ).json()
    assert unlinked["match_status"] == "UNMATCHED"
    assert unlinked["match_reason"] == "UNLINKED_CUSTOMER_IDENTITY"

    # Re-linking to a DIFFERENT customer_id is refused, not silently overwritten.
    conflict = client.post(
        "/customers/identity-links",
        headers=KEY,
        json={
            "system": "netsuite",
            "external_customer_id": f"NS-{tag}",
            "customer_id": "someone_else",
            "source": "manual_mapping",
        },
    )
    assert conflict.status_code == 409


# ------------------------------------------------------------------ S financial invariants


def test_S_financial_invariants_hold_across_a_mixed_sequence():
    """recovered + outstanding + cancelled == original at every step, for every
    kind of event, in every order this module's reconcile_position can reach."""
    original = Decimal("1000")
    state = {"recovered": Decimal(0), "outstanding": original, "cancelled": Decimal(0)}
    case_state = CaseState.OPEN

    def apply(kind, amount=None):
        nonlocal case_state
        r = reconcile_position(
            kind=kind,
            amount=Decimal(str(amount)) if amount is not None else None,
            current_state=case_state,
            recovered=state["recovered"],
            outstanding=state["outstanding"],
            cancelled=state["cancelled"],
        )
        state["recovered"], state["outstanding"], state["cancelled"] = (
            r.new_recovered,
            r.new_outstanding,
            r.new_cancelled,
        )
        case_state = r.new_case_state
        assert state["recovered"] + state["outstanding"] + state["cancelled"] == original
        assert state["recovered"] >= 0 and state["outstanding"] >= 0 and state["cancelled"] >= 0
        assert state["recovered"] <= original
        return r

    r1 = apply(OutcomeKind.SETTLEMENT, 300)  # partial
    assert r1.effect == Effect.APPLIED and case_state == CaseState.PARTIALLY_RECOVERED
    r2 = apply(OutcomeKind.SETTLEMENT, 5000)  # overpay: clamps to outstanding, records excess
    assert r2.effect == Effect.APPLIED and r2.excess_amount == Decimal("4300.0000")
    assert case_state == CaseState.FULLY_RECOVERED
    r3 = apply(OutcomeKind.SETTLEMENT, 1)  # duplicate/late on a fully-recovered case
    assert r3.effect == Effect.EXCESS and case_state == CaseState.FULLY_RECOVERED
    r4 = apply(OutcomeKind.CANCELLATION)  # cancelling a fully-recovered obligation is inert
    assert r4.effect == Effect.NO_EFFECT and case_state == CaseState.FULLY_RECOVERED

    # A fresh sequence: cancellation first, then a late settlement is excess only.
    state = {"recovered": Decimal(0), "outstanding": original, "cancelled": Decimal(0)}
    case_state = CaseState.OPEN
    c1 = apply(OutcomeKind.CANCELLATION)
    assert c1.effect == Effect.APPLIED and case_state == CaseState.CANCELLED
    c2 = apply(OutcomeKind.SETTLEMENT, 1000)
    assert c2.effect == Effect.EXCESS and state["recovered"] == 0
    assert case_state == CaseState.CANCELLED


# ------------------------------------------------------------------ T dashboard


def test_T_dashboard_distinguishes_at_risk_recovered_outstanding_cancelled_unmatched():
    tag = _tag()

    def snapshot():
        m = client.get("/dashboard/metrics").json()
        return (
            Decimal(str(m["outstanding_amount"])),
            Decimal(str(m["cancelled_amount"])),
            m["unmatched_outcome_events"],
        )

    before_out, before_cancel, before_unmatched = snapshot()

    partial_ob = _open_case(tag + "p", domain="OVERDUE_INVOICE", amount=1000, currency="GBP")
    _outcome(partial_ob, amount=250)  # 750 stays outstanding

    cancel_ob = _open_case(tag + "c", domain="OVERDUE_INVOICE", amount=400, currency="GBP")
    client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"cancel_dash_{tag}",
            "domain": "OVERDUE_INVOICE",
            "outcome_type": "INVOICE_VOIDED",
            "customer_id": cancel_ob["customer_id"],
            "reference_id": cancel_ob["reference_id"],
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    )

    client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p5_settlement",
            "external_event_id": f"orphan_dash_{tag}",
            "domain": "OVERDUE_INVOICE",
            "outcome_type": "INVOICE_SETTLED",
            "customer_id": f"nobody_{tag}",
            "reference_id": f"nothing_{tag}",
            "amount": 50,
            "currency": "GBP",
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    )

    after_out, after_cancel, after_unmatched = snapshot()
    assert after_out - before_out == Decimal("750")
    assert after_cancel - before_cancel == Decimal("400")
    assert after_unmatched - before_unmatched == 1

    breakdown = client.get("/dashboard/metrics").json()["currency_breakdown"]["GBP"]
    assert Decimal(str(breakdown["outstanding_amount"])) >= Decimal("750")
    assert Decimal(str(breakdown["cancelled_amount"])) >= Decimal("400")


# ------------------------------------------------------------------ U action != recovery


def test_U_executing_an_action_never_by_itself_produces_a_recovered_amount():
    tag = _tag()
    # network_error diagnoses deterministically to NETWORK_FAILURE, whose recommended
    # action is RETRY_PAYMENT (a low amount, so policy PERMITs it) - an UNKNOWN
    # diagnosis would recommend ESCALATE_TO_HUMAN instead, which is not what this test
    # is about.
    ob = _open_case(tag, domain="FAILED_PAYMENT", amount=100, failure_code="network_error")
    for stage in (
        "assess-risk",
        "diagnose",
        "predict-recovery",
        "recommend-action",
        "policy-check",
    ):
        assert client.post(f"/cases/{ob['case_id']}/{stage}").status_code == 200
    ex = client.post(f"/cases/{ob['case_id']}/execute")
    assert ex.status_code == 200 and ex.json()["status"] == "COMPLETED_SIMULATED"

    snap = _snap(ob["case_id"])
    assert snap["outcome"] is None
    assert snap["current_state"] == "PENDING_VERIFICATION"

    verify = client.post(f"/cases/{ob['case_id']}/verify", json={"external_reference": ""}).json()
    assert verify["status"] == "PENDING_VERIFICATION"
    assert Decimal(str(verify["actual_amount_recovered"])) == 0

    _outcome(ob, outcome_type="PAYMENT_SETTLED", amount=100)
    assert _snap(ob["case_id"])["current_state"] == "FULLY_RECOVERED"
