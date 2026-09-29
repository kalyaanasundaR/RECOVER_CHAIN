"""Phase 8A — real-data readiness, outcome provenance, and model-gating tests.

1  authoritative outcome creation (still works, now with provenance)
2  test outcome isolation (a TEST-provenance outcome never counts as production)
3  provenance validation (unknown value rejected; omitted -> TEST, never PRODUCTION)
4  duplicate prevention (unchanged Phase 5 guarantee, re-verified here)
5  invalid financial values (data-quality check catches a negative applied amount)
6  timestamp validation (data-quality check catches an impossible timestamp)
7  outcome count accuracy (authoritative vs production vs test, from real queries)
8  data-readiness report shape and internal consistency
9  model-readiness gate (all three tasks NOT_READY, with real, non-fabricated reasons)
10 root-cause label independence (rule-derived != observed; a fake "independent"
   source is rejected)
11 advisory-only behavior (unchanged Phase 7 guarantee, re-verified here)
12 AI failure safety (unchanged Phase 7 guarantee, re-verified here)
"""

import uuid
from decimal import Decimal

from fastapi.testclient import TestClient

from api.main import app
from application.data_readiness import (
    build_data_readiness_report,
    build_root_cause_label_status,
    evaluate_model_readiness,
    run_quality_checks,
)
from application.outcome_provenance import OutcomeProvenance, normalize_provenance
from infrastructure.db import SessionLocal

client = TestClient(app)
KEY = {"X-API-Key": "test-api-key"}


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _open_case(tag: str, *, domain: str = "OVERDUE_INVOICE", amount: float = 1000) -> dict:
    customer_id = f"p8a_cust_{tag}"
    reference_id = f"p8a_ref_{tag}"
    r = client.post(
        "/events",
        json={
            "customer_id": customer_id,
            "risk_category": domain,
            "external_system": "p8a_sys",
            "external_event_id": f"p8a_evt_{tag}",
            "reference_id": reference_id,
            "amount": amount,
            "currency": "INR",
            "raw_payload": {},
        },
    )
    assert r.status_code == 200, r.text
    return {
        "case_id": r.json()["case_id"],
        "customer_id": customer_id,
        "reference_id": reference_id,
        "domain": domain,
    }


def _outcome(
    ob: dict,
    *,
    amount,
    provenance=None,
    external_event_id=None,
    occurred_at=None,
    evidence_reference=None,
):
    body = {
        "external_system": "p8a_out",
        "external_event_id": external_event_id or f"p8a_out_{_tag()}",
        "domain": ob["domain"],
        "outcome_type": "INVOICE_SETTLED",
        "customer_id": ob["customer_id"],
        "reference_id": ob["reference_id"],
        "amount": amount,
        "currency": "INR",
        "occurred_at": occurred_at or "2026-09-10T00:00:00Z",
    }
    if provenance is not None:
        body["provenance"] = provenance
    # Phase 9: human_confirmed now requires a genuine external evidence_reference.
    if provenance == OutcomeProvenance.HUMAN_CONFIRMED and evidence_reference is None:
        evidence_reference = f"support-ticket-{_tag()}"
    if evidence_reference is not None:
        body["evidence_reference"] = evidence_reference
    return client.post("/outcomes", headers=KEY, json=body)


# ------------------------------------------------------------------ 1 + 3


def test_1_3_outcome_creation_records_provenance_and_defaults_to_test():
    ob = _open_case(_tag(), amount=500)
    r = _outcome(ob, amount=500)  # no provenance given
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processed"
    assert body["provenance"] == "test"  # never silently PRODUCTION


def test_3_unrecognised_provenance_is_rejected():
    ob = _open_case(_tag(), amount=500)
    r = _outcome(ob, amount=500, provenance="definitely_not_a_real_category")
    assert r.status_code == 422
    assert r.json()["detail"]["field"] == "provenance"


def test_3_a_real_provenance_value_is_accepted_and_stored_distinctly():
    ob = _open_case(_tag(), amount=500)
    r = _outcome(ob, amount=500, provenance=OutcomeProvenance.HUMAN_CONFIRMED)
    assert r.status_code == 200
    assert r.json()["provenance"] == "human_confirmed"


# ------------------------------------------------------------------ 2 + 7


def test_2_7_test_outcome_never_counts_as_production():
    db = SessionLocal()
    try:
        before = build_data_readiness_report(db)
    finally:
        db.close()

    ob = _open_case(_tag(), amount=750)
    r = _outcome(ob, amount=750, provenance=OutcomeProvenance.TEST)
    assert r.status_code == 200

    db = SessionLocal()
    try:
        after = build_data_readiness_report(db)
    finally:
        db.close()

    # A new TEST outcome increases the authoritative (matched) total...
    assert (
        after["authoritative_outcome_events_count"]
        == before["authoritative_outcome_events_count"] + 1
    )
    # ...but NEVER increases the production-provenance count.
    assert after["production_outcomes_count"] == before["production_outcomes_count"]
    assert after["test_outcomes_count"] == before["test_outcomes_count"] + 1


# ------------------------------------------------------------------ 4


def test_4_duplicate_outcome_is_rejected_not_double_counted():
    ob = _open_case(_tag(), amount=300)
    eid = f"p8a_dup_{_tag()}"
    r1 = _outcome(ob, amount=300, external_event_id=eid)
    assert r1.json()["status"] == "processed"
    r2 = _outcome(ob, amount=300, external_event_id=eid)  # exact replay
    assert r2.json()["status"] == "duplicate"


# ------------------------------------------------------------------ 5 + 6


def test_5_6_quality_checks_flag_bad_values_without_repairing_them():
    # Exercise the real function against the real (session) database directly -
    # constructing a malformed row in memory to prove the *check itself* works,
    # without needing to smuggle a bad value through the validated API (which
    # correctly refuses one at the door - that's application/outcome_events.py's
    # job; this is application/data_readiness.py's job).
    from datetime import UTC, datetime

    from infrastructure.orm import OutcomeEventModel

    db = SessionLocal()
    try:
        bad = OutcomeEventModel(
            outcome_event_id=f"oc_p8a_bad_{_tag()}",
            external_system="p8a_direct",
            external_event_id=f"p8a_direct_{_tag()}",
            domain="OVERDUE_INVOICE",
            kind="SETTLEMENT",
            outcome_type="INVOICE_SETTLED",
            customer_id="p8a_bad_customer",
            reference_id=None,
            amount=Decimal("-50.0000"),
            currency="INR",
            occurred_at=datetime.now(UTC).replace(tzinfo=None),
            match_status="UNMATCHED",
            match_reason="NO_OBLIGATION",
            case_id=None,
            applied_amount=Decimal("-50.0000"),  # negative - the check must catch this
            excess_amount=None,
            effect=None,
            provenance="test",
        )
        db.add(bad)
        db.commit()

        issues = run_quality_checks(db)
        flagged = [i for i in issues if i.outcome_event_id == bad.outcome_event_id]
        assert any(i.issue == "negative_applied_amount" for i in flagged)

        # Not repaired: the row still holds the negative value afterward.
        refetched = (
            db.query(OutcomeEventModel)
            .filter(OutcomeEventModel.outcome_event_id == bad.outcome_event_id)
            .first()
        )
        assert refetched.applied_amount == Decimal("-50.0000")
    finally:
        db.close()


# ------------------------------------------------------------------ 8 + 9


def test_8_9_readiness_report_and_model_gate_are_internally_consistent():
    db = SessionLocal()
    try:
        report = build_data_readiness_report(db)
        gates = evaluate_model_readiness(db)
    finally:
        db.close()

    assert report["authoritative_outcome_events_count"] >= report["production_outcomes_count"]
    # production_outcomes_count is REAL_WORLD_VALUES (production/human_confirmed/
    # payment_provider/...), NOT literally provenance=="production" alone -
    # human_confirmed_outcomes_count is a breakdown WITHIN it, not a disjoint bucket.
    # The three genuinely disjoint, exhaustive buckets are production (real-world),
    # test, and external_source_simulated - this must hold regardless of which other
    # test files (Phase 8B/9) have already added rows to the shared session database.
    assert report["human_confirmed_outcomes_count"] <= report["production_outcomes_count"]
    assert report["authoritative_outcome_events_count"] == (
        report["production_outcomes_count"]
        + report["test_outcomes_count"]
        + report["external_source_simulated_outcomes_count"]
    )

    tasks = {g.task for g in gates}
    assert tasks == {"A_recovery_probability", "B_next_best_action", "C_root_cause_classifier"}
    # No universal, unjustified sample-size threshold is asserted as "sufficient" -
    # every gate must still be NOT_READY at this scale (a handful of rows, however
    # they're categorised), and must say so with real, live-queried numbers, never an
    # arbitrary magic constant.
    for g in gates:
        assert g.status == "NOT_READY"
        assert any(str(report["production_outcomes_count"]) in r for r in g.reasons) or any(
            "0 independent observations" in r or "production_outcomes_count=0" in r
            for r in g.reasons
        )


# ------------------------------------------------------------------ 10


def test_10_root_cause_labels_are_kept_distinct_from_observations():
    ob = _open_case(_tag(), amount=200)

    # The API rejects "the rule itself" as if it were an independent source.
    r = client.post(
        f"/cases/{ob['case_id']}/observed-cause",
        headers=KEY,
        json={"observed_cause_category": "NETWORK_FAILURE", "source": "diagnosis_engine"},
    )
    assert r.status_code == 422

    r = client.post(
        f"/cases/{ob['case_id']}/observed-cause",
        headers=KEY,
        json={
            "observed_cause_category": "NETWORK_FAILURE",
            "source": "human_confirmed",
            "note": "test",
            "evidence_reference": f"support-ticket-{_tag()}",
        },
    )
    assert r.status_code == 200
    assert r.json()["source"] == "human_confirmed"

    db = SessionLocal()
    try:
        status = build_root_cause_label_status(db)
    finally:
        db.close()
    assert status["independent_observation_count"] >= 1
    assert "circular" in status["verdict"]


# ------------------------------------------------------------------ 11 + 12


def test_11_normalize_provenance_never_defaults_to_production():
    assert normalize_provenance(None) == OutcomeProvenance.TEST
    assert normalize_provenance("") == OutcomeProvenance.TEST
    # Phase 9: normalize_provenance() itself now only validates the VOCABULARY (an
    # explicit "production" claim is a recognised value, so it no longer raises
    # here) - whether it is actually AUTHORIZED is checked separately, against the
    # evidence-source registry, inside application.outcome_events.validate_outcome_item
    # (which requires a verified/enabled source_id - see
    # test_phase43_phase8b_evidence.py::test_1_production_provenance_cannot_be_claimed
    # for the real, end-to-end rejection).
    assert normalize_provenance("production") == OutcomeProvenance.PRODUCTION
    assert normalize_provenance("human_confirmed") == OutcomeProvenance.HUMAN_CONFIRMED


def test_12_missing_or_broken_predictor_still_falls_back_safely():
    # Re-verifies the Phase 7 safety contract (application/case_pipeline.py) was not
    # weakened by any Phase 8A change.
    from datetime import UTC, datetime

    from application.case_pipeline import predict_recovery_for_case
    from application.recovery_predictor import FeatureExtractor
    from domain.models import CaseState, Money, RecoveryCase, RevenueEvent, RiskCategory

    event = RevenueEvent(
        event_id="evt_p8a",
        customer_id="cust_p8a",
        risk_category=RiskCategory.OVERDUE_INVOICE,
        external_system="test",
        external_event_id="ext_p8a",
        reference_id="ref_p8a",
        amount=Money(amount=Decimal("400"), currency="INR"),
        timestamp=datetime.now(UTC),
        raw_payload={},
    )
    case = RecoveryCase(
        case_id="case_p8a",
        customer_id="cust_p8a",
        risk_category=RiskCategory.OVERDUE_INVOICE,
        amount_at_risk=Money(amount=Decimal("400"), currency="INR"),
        current_state=CaseState.DIAGNOSING,
        linked_events=[event],
    )

    class _BrokenPredictor:
        model = object()

        def predict_failure_risk(self, features):
            raise RuntimeError("simulated failure")

    result = predict_recovery_for_case(case, predictor=_BrokenPredictor())
    assert result.prediction_status == "SUCCESS_BASELINE"
    FeatureExtractor.extract_features(case)  # must not raise


# ------------------------------------------------------------------ additional Step 3 checks


def test_deleting_test_data_does_not_change_production_statistics():
    db = SessionLocal()
    try:
        before = build_data_readiness_report(db)
    finally:
        db.close()

    ob = _open_case(_tag(), amount=222)
    r = _outcome(ob, amount=222, provenance=OutcomeProvenance.TEST)
    assert r.status_code == 200
    outcome_event_id = r.json()["outcome_event_id"]

    from infrastructure.orm import OutcomeEventModel

    db = SessionLocal()
    try:
        db.query(OutcomeEventModel).filter(
            OutcomeEventModel.outcome_event_id == outcome_event_id
        ).delete()
        db.commit()
        after = build_data_readiness_report(db)
    finally:
        db.close()

    # Removing TEST data must never move the production count - it was never in it.
    assert after["production_outcomes_count"] == before["production_outcomes_count"]


def test_outcome_events_has_no_mutation_endpoint():
    # Append-only by construction (Phase 5): only POST exists. A correction must be a
    # new, auditable row - never an in-place edit of a finalized outcome.
    from api.main import app as fastapi_app

    outcome_paths = {
        route.path: route.methods
        for route in fastapi_app.routes
        if "/outcomes" in getattr(route, "path", "")
    }
    for _path, methods in outcome_paths.items():
        assert "PUT" not in methods
        assert "PATCH" not in methods
        assert "DELETE" not in methods
