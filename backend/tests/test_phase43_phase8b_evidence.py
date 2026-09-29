"""Phase 8B — independent evidence acquisition & data quality tests.

1  provenance cannot be falsely promoted (production/reserved values rejected)
2  external-source-simulated remains non-production
3  independent cause observation remains distinct (reserved sources rejected too)
4  duplicate external outcomes (still rejected/idempotent)
5  out-of-order outcomes (still reconcile correctly)
6  late outcomes (occurred_at far after ingestion is fine; before case existed is flagged)
7  training-data eligibility (ELIGIBLE only reachable with real provenance + complete case)
8  temporal eligibility (a non-terminal case is INSUFFICIENT_EVIDENCE at best)
9  dataset snapshot reproducibility (same inputs -> same checksum; version never reused)
10 operational/evaluation isolation (building a snapshot never touches case/outcome rows)
11 financial invariants (unchanged, re-verified)
12 multi-currency safety (unchanged, re-verified)
"""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from fastapi.testclient import TestClient

from api.main import app
from application.dataset_snapshot import build_snapshot
from application.eligibility import EligibilityStatus, evaluate_case_eligibility
from application.outcome_provenance import OutcomeProvenance
from infrastructure.db import SessionLocal
from infrastructure.orm import CaseModel

client = TestClient(app)
KEY = {"X-API-Key": "test-api-key"}


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _open_case(tag: str, *, domain: str = "OVERDUE_INVOICE", amount: float = 1000) -> dict:
    customer_id = f"p8b_cust_{tag}"
    reference_id = f"p8b_ref_{tag}"
    r = client.post(
        "/events",
        json={
            "customer_id": customer_id,
            "risk_category": domain,
            "external_system": "p8b_sys",
            "external_event_id": f"p8b_evt_{tag}",
            "reference_id": reference_id,
            "amount": amount,
            "currency": "INR",
            "raw_payload": {},
        },
    )
    assert r.status_code == 200, r.text
    case_id = r.json()["case_id"]
    r = client.post(f"/cases/{case_id}/advance", headers=KEY)
    assert r.status_code == 200
    return {
        "case_id": case_id,
        "customer_id": customer_id,
        "reference_id": reference_id,
        "domain": domain,
    }


def _outcome(
    ob,
    *,
    amount,
    provenance=None,
    external_event_id=None,
    occurred_at=None,
    external_system="p8b_out",
    evidence_reference=None,
    source_id=None,
):
    body = {
        "external_system": external_system,
        "external_event_id": external_event_id or f"p8b_out_{_tag()}",
        "domain": ob["domain"],
        "outcome_type": "INVOICE_SETTLED",
        "customer_id": ob["customer_id"],
        "reference_id": ob["reference_id"],
        "amount": amount,
        "currency": "INR",
        "occurred_at": occurred_at or datetime.now(UTC).isoformat(),
    }
    if provenance is not None:
        body["provenance"] = provenance
    # Phase 9: human_confirmed now requires a genuine external evidence_reference.
    if provenance == OutcomeProvenance.HUMAN_CONFIRMED and evidence_reference is None:
        evidence_reference = f"support-ticket-{_tag()}"
    if evidence_reference is not None:
        body["evidence_reference"] = evidence_reference
    if source_id is not None:
        body["source_id"] = source_id
    return client.post("/outcomes", headers=KEY, json=body)


# ------------------------------------------------------------------ 1 + 2


def test_1_production_provenance_cannot_be_claimed_via_shared_api_key():
    ob = _open_case(_tag(), amount=500)
    # Phase 9: production now requires a verified, enabled evidence source (source_id)
    # - the rejection point moved from the bare provenance value to the missing/
    # unusable source, which is the more precise, checkable reason.
    r = _outcome(ob, amount=500, provenance="production")
    assert r.status_code == 422
    assert r.json()["detail"]["field"] == "source_id"
    assert "VERIFIED" in r.json()["detail"]["detail"]


def test_1_reserved_integration_provenance_cannot_be_claimed_either():
    ob = _open_case(_tag(), amount=500)
    for reserved in (
        "payment_provider",
        "invoice_system",
        "subscription_system",
        "checkout_system",
    ):
        r = _outcome(
            ob, amount=500, provenance=reserved, external_event_id=f"p8b_{reserved}_{_tag()}"
        )
        assert r.status_code == 422, reserved


def test_2_external_source_simulated_is_accepted_but_never_counts_as_production():
    from application.data_readiness import build_data_readiness_report

    db = SessionLocal()
    try:
        before = build_data_readiness_report(db)
    finally:
        db.close()

    ob = _open_case(_tag(), amount=650)
    r = _outcome(ob, amount=650, provenance=OutcomeProvenance.EXTERNAL_SOURCE_SIMULATED)
    assert r.status_code == 200
    assert r.json()["provenance"] == "external_source_simulated"

    db = SessionLocal()
    try:
        after = build_data_readiness_report(db)
    finally:
        db.close()
    assert after["production_outcomes_count"] == before["production_outcomes_count"]
    assert (
        after["external_source_simulated_outcomes_count"]
        == before["external_source_simulated_outcomes_count"] + 1
    )


# ------------------------------------------------------------------ 3


def test_3_reserved_and_rule_derived_cause_sources_are_rejected():
    ob = _open_case(_tag(), amount=200)
    for bad_source in (
        "diagnosis_engine",
        "rule",
        "payment_gateway_response",
        "invoice_payment_history",
    ):
        r = client.post(
            f"/cases/{ob['case_id']}/observed-cause",
            headers=KEY,
            json={"observed_cause_category": "NETWORK_FAILURE", "source": bad_source},
        )
        assert r.status_code == 422, bad_source

    r = client.post(
        f"/cases/{ob['case_id']}/observed-cause",
        headers=KEY,
        json={
            "observed_cause_category": "NETWORK_FAILURE",
            "source": "human_confirmed",
            "evidence_reference": "ticket-4711",
            "confidence": 0.9,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["evidence_reference"] == "ticket-4711"
    assert float(body["confidence"]) == 0.9


# ------------------------------------------------------------------ 4


def test_4_duplicate_external_outcome_is_rejected_not_double_applied():
    ob = _open_case(_tag(), amount=300)
    eid = f"p8b_dup_{_tag()}"
    r1 = _outcome(ob, amount=300, external_event_id=eid)
    assert r1.json()["status"] == "processed"
    r2 = _outcome(ob, amount=300, external_event_id=eid)
    assert r2.json()["status"] == "duplicate"


# ------------------------------------------------------------------ 5 + 6


def test_5_6_out_of_order_and_late_outcomes_reconcile_correctly():
    ob = _open_case(_tag(), amount=1000)
    r1 = _outcome(
        ob,
        amount=400,
        occurred_at="2026-09-05T00:00:00Z",
        external_event_id=f"p8b_ooo_late_{_tag()}",
    )
    assert r1.json()["effect"] == "APPLIED"
    r2 = _outcome(
        ob,
        amount=600,
        occurred_at="2026-09-01T00:00:00Z",
        external_event_id=f"p8b_ooo_early_{_tag()}",
    )
    assert r2.json()["effect"] == "APPLIED"

    snap = client.get(f"/cases/{ob['case_id']}/outcome", headers=KEY).json()
    assert snap["status"] == "FULLY_RECOVERED"
    assert snap["actual_amount_recovered"] == "1000.0000"


def test_6_an_outcome_before_the_case_existed_is_flagged_not_silently_accepted():
    from application.data_readiness import run_quality_checks

    ob = _open_case(_tag(), amount=100)
    r = _outcome(
        ob,
        amount=100,
        occurred_at="2020-01-01T00:00:00Z",
        external_event_id=f"p8b_ancient_{_tag()}",
    )
    assert (
        r.status_code == 200
    )  # accepted (a genuinely late/backdated report is not itself invalid)

    db = SessionLocal()
    try:
        issues = run_quality_checks(db)
    finally:
        db.close()
    flagged = [i for i in issues if i.outcome_event_id == r.json()["outcome_event_id"]]
    assert any(i.issue == "outcome_occurred_before_case_existed" for i in flagged)


# ------------------------------------------------------------------ 7 + 8


def test_7_8_eligibility_requires_real_provenance_and_a_terminal_state():
    ob = _open_case(_tag(), amount=800)

    db = SessionLocal()
    try:
        case = db.query(CaseModel).filter(CaseModel.case_id == ob["case_id"]).first()
        result = evaluate_case_eligibility(db, case)
    finally:
        db.close()
    assert result.status == EligibilityStatus.INELIGIBLE
    assert "no authoritative outcome" in result.reasons[0]

    # A TEST-provenance settlement makes it terminal but still INELIGIBLE (provenance).
    r = _outcome(
        ob,
        amount=800,
        provenance=OutcomeProvenance.TEST,
        external_event_id=f"p8b_elig_test_{_tag()}",
    )
    assert r.json()["effect"] == "APPLIED"
    db = SessionLocal()
    try:
        case = db.query(CaseModel).filter(CaseModel.case_id == ob["case_id"]).first()
        result = evaluate_case_eligibility(db, case)
    finally:
        db.close()
    assert result.status == EligibilityStatus.INELIGIBLE
    assert any("non-production provenance" in r_ for r_ in result.reasons)

    # A HUMAN_CONFIRMED settlement on a fresh, complete, terminal case IS reachable as
    # ELIGIBLE - proving the classifier is not permanently stuck at INELIGIBLE by
    # construction, only by the current absence of real evidence.
    ob2 = _open_case(_tag(), amount=450)
    r = _outcome(
        ob2,
        amount=450,
        provenance=OutcomeProvenance.HUMAN_CONFIRMED,
        external_event_id=f"p8b_elig_human_{_tag()}",
    )
    assert r.json()["effect"] == "APPLIED"
    db = SessionLocal()
    try:
        case = db.query(CaseModel).filter(CaseModel.case_id == ob2["case_id"]).first()
        result = evaluate_case_eligibility(db, case)
    finally:
        db.close()
    assert result.status == EligibilityStatus.ELIGIBLE, result.reasons


# ------------------------------------------------------------------ 9 + 10


def test_9_10_snapshot_is_reproducible_and_does_not_mutate_operational_data():
    db = SessionLocal()
    try:
        case_count_before = db.query(CaseModel).count()
        snap1 = build_snapshot(db, dataset_name=f"p8b_test_dataset_{_tag()}", created_by="test")
        db.commit()
    finally:
        db.close()

    db = SessionLocal()
    try:
        case_count_after = db.query(CaseModel).count()
    finally:
        db.close()

    # Building a snapshot must never create/delete/alter an operational case row.
    assert case_count_after == case_count_before

    db = SessionLocal()
    try:
        snap2 = build_snapshot(db, dataset_name=snap1["dataset_name"], created_by="test")
        db.commit()
    finally:
        db.close()

    # Same eligible set -> same checksum; version always increments, never reused.
    assert snap2["checksum"] == snap1["checksum"]
    assert snap2["version"] == snap1["version"] + 1


# ------------------------------------------------------------------ 11 + 12


def test_11_financial_invariants_still_hold():
    ob = _open_case(_tag(), amount=500)
    r = _outcome(ob, amount=300, external_event_id=f"p8b_fi_{_tag()}")
    assert r.json()["effect"] == "APPLIED"
    snap = client.get(f"/cases/{ob['case_id']}/outcome", headers=KEY).json()
    recovered = Decimal(snap["actual_amount_recovered"])
    outstanding = Decimal(snap["outstanding_amount"])
    assert recovered == Decimal("300.0000")
    assert outstanding == Decimal("200.0000")
    assert recovered + outstanding == Decimal("500.0000")
    assert recovered <= Decimal("500.0000")


def test_12_multi_currency_mismatch_is_rejected_not_summed():
    ob = _open_case(_tag(), amount=500)  # opened in INR
    r = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p8b_out",
            "external_event_id": f"p8b_ccy_{_tag()}",
            "domain": ob["domain"],
            "outcome_type": "INVOICE_SETTLED",
            "customer_id": ob["customer_id"],
            "reference_id": ob["reference_id"],
            "amount": 500,
            "currency": "USD",  # wrong currency for this obligation
            "occurred_at": "2026-09-10T00:00:00Z",
        },
    )
    body = r.json()
    assert body["match_status"] == "UNMATCHED"
    assert body["match_reason"] == "CURRENCY_MISMATCH"
