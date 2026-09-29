"""Phase 9 — real evidence acquisition & integration readiness tests.

1  trusted-source registration (registered, starts NOT_CONFIGURED, never auto-trusted)
2  unauthenticated production rejection (no verified source -> always refused)
3  provider-event identity (external_event_id distinct from any RecoverChain id)
4  human-confirmation integrity (evidence_reference required, self-referential rejected)
5  provenance escalation prevention (an unverified/disabled source can't be forced usable)
6  duplicate external events (idempotent, not double-applied)
7  late external events (backdated occurred_at still reconciles; flagged if pre-dates case)
8  unmatched events (stay visible, excluded from recovered revenue)
9  partial recovery (stays partial, real outstanding balance)
10 cancellation (stays cancellation, never recovered)
11 eligibility (re-run; still requires real, verified-grade provenance)
12 ML-readiness gate (reasons are live-computed, not stale strings)
13 financial invariants (recovered<=at_risk, components reconcile)
14 operational/evaluation isolation (registry/snapshot ops never mutate case data)
"""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from fastapi.testclient import TestClient

from api.main import app
from application.data_readiness import build_data_readiness_report
from application.eligibility import EligibilityStatus, evaluate_case_eligibility
from application.evidence_sources import (
    IntegrationStatus,
    SourceRegistrationError,
    is_usable,
    register_source,
    set_enabled,
    verify_source,
)
from application.outcome_provenance import OutcomeProvenance
from infrastructure.db import SessionLocal
from infrastructure.orm import CaseModel, EvidenceSourceModel

client = TestClient(app)
KEY = {"X-API-Key": "test-api-key"}


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _open_case(tag: str, *, domain: str = "OVERDUE_INVOICE", amount: float = 1000) -> dict:
    customer_id = f"p9_cust_{tag}"
    reference_id = f"p9_ref_{tag}"
    r = client.post(
        "/events",
        json={
            "customer_id": customer_id,
            "risk_category": domain,
            "external_system": "p9_sys",
            "external_event_id": f"p9_evt_{tag}",
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
    amount=None,
    provenance=None,
    external_event_id=None,
    occurred_at=None,
    evidence_reference=None,
    source_id=None,
    outcome_type="INVOICE_SETTLED",
):
    body = {
        "external_system": "p9_out",
        "external_event_id": external_event_id or f"p9_out_{_tag()}",
        "domain": ob["domain"],
        "outcome_type": outcome_type,
        "customer_id": ob["customer_id"],
        "reference_id": ob["reference_id"],
        "currency": "INR",
        "occurred_at": occurred_at or datetime.now(UTC).isoformat(),
    }
    if amount is not None:
        body["amount"] = amount
    if provenance is not None:
        body["provenance"] = provenance
    if evidence_reference is not None:
        body["evidence_reference"] = evidence_reference
    if source_id is not None:
        body["source_id"] = source_id
    return client.post("/outcomes", headers=KEY, json=body)


# ------------------------------------------------------------------ 1


def test_1_trusted_source_registration_starts_not_configured_never_auto_trusted():
    db = SessionLocal()
    try:
        row = register_source(
            db,
            source_type="payment_provider",
            provider="test_psp",
            domain="FAILED_PAYMENT",
            schema_version="v1",
            registered_by="test",
        )
        db.commit()
        assert row.integration_status == IntegrationStatus.NOT_CONFIGURED
        assert row.enabled is False
        assert row.last_verified_at is None
    finally:
        db.close()


# ------------------------------------------------------------------ 2 + 5


def test_2_5_unauthenticated_production_is_always_rejected_and_cannot_be_forced():
    db = SessionLocal()
    try:
        row = register_source(
            db,
            source_type="payment_provider",
            provider="test_psp2",
            domain="FAILED_PAYMENT",
            schema_version="v1",
        )
        db.commit()
        source_id = row.source_id

        # Verification finds no real credential (none configured in this environment)
        # and leaves it NOT_CONFIGURED - never flips to VERIFIED on request alone.
        verified = verify_source(db, source_id)
        db.commit()
        assert verified.integration_status == IntegrationStatus.NOT_CONFIGURED

        # Enabling a NOT_CONFIGURED source is structurally refused.
        try:
            set_enabled(db, source_id, True)
            raised = False
        except SourceRegistrationError:
            raised = True
        db.rollback()
        assert raised

        # And therefore it can never back an outcome, regardless of caller intent.
        assert (
            is_usable(db, source_id, source_type="payment_provider", domain="FAILED_PAYMENT")
            is False
        )
    finally:
        db.close()

    ob = _open_case(_tag(), domain="FAILED_PAYMENT", amount=500)
    r = _outcome(
        ob,
        amount=500,
        provenance=OutcomeProvenance.PRODUCTION,
        source_id=source_id,
        outcome_type="PAYMENT_SETTLED",
    )
    assert r.status_code == 422
    assert r.json()["detail"]["field"] == "source_id"


# ------------------------------------------------------------------ 3


def test_3_provider_event_identity_is_distinct_from_recoverchain_ids():
    ob = _open_case(_tag(), amount=250)
    # A provider-shaped external_event_id (never one of RecoverChain's own prefixes)
    # is accepted and stored as-is - the dedup key is (external_system,
    # external_event_id), never confused with RecoverChain's internal oc_/case_ ids.
    r = _outcome(ob, amount=250, external_event_id="psp_sim_demo_evt_00042")
    assert r.status_code == 200
    body = r.json()
    assert body["external_event_id"] == "psp_sim_demo_evt_00042"
    assert body["outcome_event_id"].startswith("oc_")  # RecoverChain's own id, separate
    assert body["outcome_event_id"] != body["external_event_id"]


# ------------------------------------------------------------------ 4


def test_4_human_confirmation_requires_real_external_evidence():
    ob = _open_case(_tag(), amount=300)

    # Missing entirely.
    r = _outcome(ob, amount=300, provenance=OutcomeProvenance.HUMAN_CONFIRMED)
    assert r.status_code == 422
    assert r.json()["detail"]["field"] == "evidence_reference"

    # Self-referential (points at a RecoverChain-generated id, not external evidence).
    r = _outcome(
        ob,
        amount=300,
        provenance=OutcomeProvenance.HUMAN_CONFIRMED,
        evidence_reference="case_0123abcd",
    )
    assert r.status_code == 422
    assert r.json()["detail"]["field"] == "evidence_reference"

    # Genuine external reference - accepted.
    r = _outcome(
        ob,
        amount=300,
        provenance=OutcomeProvenance.HUMAN_CONFIRMED,
        evidence_reference="support-ticket-9981",
    )
    assert r.status_code == 200
    assert r.json()["evidence_reference"] == "support-ticket-9981"

    # A user cannot "click a button" to relabel: the same underlying observation
    # (same external_event_id) already exists as this human_confirmed row; posting a
    # DIFFERENT provenance for the SAME external_event_id is not a relabel - it is a
    # duplicate of an already-processed id and is refused as such.
    ob2 = _open_case(_tag(), amount=100)
    body = {
        "external_system": "p9_out",
        "external_event_id": "p9_relabel_attempt",
        "domain": ob2["domain"],
        "outcome_type": "INVOICE_SETTLED",
        "customer_id": ob2["customer_id"],
        "reference_id": ob2["reference_id"],
        "amount": 100,
        "currency": "INR",
        "occurred_at": "2026-09-10T00:00:00Z",
        "provenance": "test",
    }
    r1 = client.post("/outcomes", headers=KEY, json=body)
    assert r1.json()["status"] == "processed"
    body["provenance"] = "human_confirmed"
    body["evidence_reference"] = "ticket-1"
    r2 = client.post("/outcomes", headers=KEY, json=body)
    assert r2.json()["status"] == "duplicate"  # never re-processed under a new provenance


# ------------------------------------------------------------------ 6 + 7


def test_6_7_duplicate_and_late_external_events():
    ob = _open_case(_tag(), amount=400)
    eid = f"p9_dup_{_tag()}"
    r1 = _outcome(ob, amount=400, external_event_id=eid)
    assert r1.json()["status"] == "processed"
    r2 = _outcome(ob, amount=400, external_event_id=eid)
    assert r2.json()["status"] == "duplicate"

    ob2 = _open_case(_tag(), amount=200)
    r = _outcome(ob2, amount=200, occurred_at="2020-01-01T00:00:00Z")
    assert r.status_code == 200  # a genuinely late report is accepted, not rejected
    assert r.json()["effect"] == "APPLIED"


# ------------------------------------------------------------------ 8


def test_8_unmatched_events_stay_visible_and_excluded():
    r = _outcome(
        {
            "domain": "OVERDUE_INVOICE",
            "customer_id": f"p9_nope_{_tag()}",
            "reference_id": "p9_nope_ref",
        },
        amount=75,
    )
    assert r.json()["match_status"] == "UNMATCHED"
    assert r.json()["case_id"] is None

    listed = client.get("/outcomes", params={"match_status": "UNMATCHED"}).json()
    assert any(i["outcome_event_id"] == r.json()["outcome_event_id"] for i in listed["items"])


# ------------------------------------------------------------------ 9 + 10


def test_9_partial_recovery_stays_partial():
    ob = _open_case(_tag(), amount=1000)
    r = _outcome(ob, amount=400)
    assert r.json()["effect"] == "APPLIED"
    snap = client.get(f"/cases/{ob['case_id']}/outcome", headers=KEY).json()
    assert snap["status"] == "PARTIALLY_RECOVERED"
    assert snap["outstanding_amount"] == "600.0000"


def test_10_cancellation_stays_cancellation_never_recovered():
    ob = _open_case(_tag(), domain="CHECKOUT_ABANDONMENT", amount=500)
    r = _outcome(ob, outcome_type="ORDER_CANCELLED")
    assert r.json()["effect"] == "APPLIED"
    snap = client.get(f"/cases/{ob['case_id']}/outcome", headers=KEY).json()
    assert snap["status"] == "CANCELLED"
    assert snap["actual_amount_recovered"] == "0.0000"
    assert snap["cancelled_amount"] == "500.0000"


# ------------------------------------------------------------------ 11 + 12


def test_11_12_eligibility_and_ml_gate_reflect_live_evidence():
    from application.data_readiness import evaluate_model_readiness

    ob = _open_case(_tag(), amount=150)
    r = _outcome(
        ob,
        amount=150,
        provenance=OutcomeProvenance.HUMAN_CONFIRMED,
        evidence_reference=f"call-log-{_tag()}",
    )
    assert r.json()["effect"] == "APPLIED"

    db = SessionLocal()
    try:
        case = db.query(CaseModel).filter(CaseModel.case_id == ob["case_id"]).first()
        result = evaluate_case_eligibility(db, case)
        gates = evaluate_model_readiness(db)
        report = build_data_readiness_report(db)
    finally:
        db.close()

    assert result.status == EligibilityStatus.ELIGIBLE, result.reasons
    # The gate's own reasons must cite the CURRENT count, not a stale "zero" claim.
    gate_a = next(g for g in gates if g.task == "A_recovery_probability")
    assert str(report["production_outcomes_count"]) in gate_a.reasons[0]
    assert gate_a.status == "NOT_READY"  # still - a handful of rows is not enough


# ------------------------------------------------------------------ 13


def test_13_financial_invariants_hold():
    ob = _open_case(_tag(), amount=777)
    r = _outcome(ob, amount=300)
    assert r.json()["effect"] == "APPLIED"
    snap = client.get(f"/cases/{ob['case_id']}/outcome", headers=KEY).json()
    recovered = Decimal(snap["actual_amount_recovered"])
    outstanding = Decimal(snap["outstanding_amount"])
    assert recovered <= Decimal("777.0000")
    assert recovered + outstanding == Decimal("777.0000")
    assert recovered >= 0 and outstanding >= 0


# ------------------------------------------------------------------ 14


def test_14_registry_and_snapshot_operations_never_mutate_operational_case_data():
    from application.dataset_snapshot import build_snapshot

    db = SessionLocal()
    try:
        case_count_before = db.query(CaseModel).count()
        source_count_before = db.query(EvidenceSourceModel).count()

        row = register_source(
            db,
            source_type="invoice_system",
            provider="test_erp",
            domain="OVERDUE_INVOICE",
            schema_version="v1",
        )
        db.commit()
        verify_source(db, row.source_id)
        db.commit()
        build_snapshot(db, dataset_name=f"p9_isolation_{_tag()}", created_by="test")
        db.commit()

        case_count_after = db.query(CaseModel).count()
        source_count_after = db.query(EvidenceSourceModel).count()
    finally:
        db.close()

    assert case_count_after == case_count_before  # no case created/deleted/altered
    assert source_count_after == source_count_before + 1  # only the registry grew
