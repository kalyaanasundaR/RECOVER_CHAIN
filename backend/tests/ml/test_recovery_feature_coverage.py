"""Live canonical-recovery feature derivation: cutoff, monotonicity, cold start, partial
history, duplicate outcomes, timestamp boundaries, missing telemetry, training/inference
parity, and the read-only shadow ERV path."""

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from api.main import app
from infrastructure.db import SessionLocal
from infrastructure.orm import (
    AuditModel,
    CaseModel,
    EventModel,
    OutcomeEventModel,
)
from ml.inference.recovery_features import (
    CANONICAL_RECOVERY_FEATURES,
    COLD_START_PRIOR_RECOVERY_RATE,
    DATA_BLOCKED_FEATURES,
    derive_recovery_features,
)

T0 = datetime(2026, 1, 1, 12, 0, 0)


def _uid():
    return uuid.uuid4().hex[:8]


def _case(db, cust, created, amount=100, state="POLICY_EVALUATED", diagnosis=None):
    cid = f"cov_{_uid()}"
    db.add(
        CaseModel(
            case_id=cid,
            customer_id=cust,
            risk_category="FAILED_PAYMENT",
            amount_at_risk=amount,
            current_state=state,
            created_at=created,
            diagnosis=diagnosis,
        )
    )
    db.flush()
    db.add(
        EventModel(
            event_id=f"e_{_uid()}",
            case_id=cid,
            customer_id=cust,
            external_system="t",
            external_event_id=f"x_{_uid()}",
            risk_category="FAILED_PAYMENT",
            amount=amount,
            timestamp=created,
            raw_payload={},
        )
    )
    db.flush()
    return cid


def _settle(db, cid, cust, amount, at):
    db.add(
        OutcomeEventModel(
            outcome_event_id=f"o_{_uid()}",
            external_system="t",
            external_event_id=f"s_{_uid()}",
            domain="FAILED_PAYMENT",
            kind="SETTLEMENT",
            outcome_type="settled",
            customer_id=cust,
            amount=amount,
            occurred_at=at,
            match_status="MATCHED",
            case_id=cid,
            applied_amount=amount,
            effect="APPLIED",
            provenance="test",
        )
    )
    db.flush()


def _closed_not_recovered(db, cid, at):
    db.add(
        AuditModel(
            id=_uid(),
            case_id=cid,
            from_state="VERIFYING",
            to_state="CLOSED_NOT_RECOVERED",
            evidence={},
            timestamp=at,
        )
    )
    db.flush()


@pytest.fixture
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


def test_cold_start_uses_documented_no_history_sentinel_not_a_prior(db):
    cid = _case(db, f"c_{_uid()}", T0)
    fv = derive_recovery_features(db, cid, T0 + timedelta(days=1))
    assert fv.values["prior_recovery_rate"] == COLD_START_PRIOR_RECOVERY_RATE
    assert fv.provenance["prior_recovery_rate"].status == "COLD_START"
    assert fv.values["number_of_previous_recoveries"] == 0
    assert fv.values["amount_vs_customer_average"] is None  # never a fabricated average
    assert fv.values["average_days_to_payment"] is None


def test_partial_history_one_recovery_gives_exact_statistics(db):
    cust = f"c_{_uid()}"
    p1 = _case(db, cust, T0, amount=200)
    _settle(db, p1, cust, 200, T0 + timedelta(days=4))
    cid = _case(db, cust, T0 + timedelta(days=30), amount=400)
    fv = derive_recovery_features(db, cid, T0 + timedelta(days=31))
    assert fv.values["number_of_previous_recoveries"] == 1
    assert fv.values["number_of_previous_failed_attempts"] == 0
    assert fv.values["prior_recovery_rate"] == 1.0
    assert fv.values["amount_vs_customer_average"] == pytest.approx(2.0)
    assert fv.values["average_days_to_payment"] == pytest.approx(4.0)
    assert fv.provenance["prior_recovery_rate"].records_used == 1


def test_future_recovery_does_not_leak_into_history(db):
    cust = f"c_{_uid()}"
    p1 = _case(db, cust, T0, amount=100)
    cid = _case(db, cust, T0 + timedelta(days=10))
    cutoff = T0 + timedelta(days=11)
    _settle(db, p1, cust, 100, cutoff + timedelta(days=5))  # recovers AFTER the prediction
    fv = derive_recovery_features(db, cid, cutoff)
    assert fv.values["number_of_previous_recoveries"] == 0
    assert fv.values["prior_recovery_rate"] == COLD_START_PRIOR_RECOVERY_RATE
    # ...and the same case re-predicted at a later cutoff DOES see it.
    later = derive_recovery_features(db, cid, cutoff + timedelta(days=10))
    assert later.values["number_of_previous_recoveries"] == 1


def test_event_exactly_at_cutoff_is_excluded_strictly_before_rule(db):
    cust = f"c_{_uid()}"
    p1 = _case(db, cust, T0)
    cid = _case(db, cust, T0 + timedelta(days=5))
    cutoff = T0 + timedelta(days=6)
    _settle(db, p1, cust, 100, cutoff)  # exactly at the cutoff
    assert derive_recovery_features(db, cid, cutoff).values["number_of_previous_recoveries"] == 0
    just_after = derive_recovery_features(db, cid, cutoff + timedelta(microseconds=1))
    assert just_after.values["number_of_previous_recoveries"] == 1


def test_later_created_sibling_cases_are_not_history(db):
    cust = f"c_{_uid()}"
    cid = _case(db, cust, T0)
    later = _case(db, cust, T0 + timedelta(days=3))
    _settle(db, later, cust, 100, T0 + timedelta(days=4))
    fv = derive_recovery_features(db, cid, T0 + timedelta(days=30))
    assert fv.values["number_of_previous_recoveries"] == 0  # created AFTER this case


def test_duplicate_settlement_events_do_not_double_count(db):
    cust = f"c_{_uid()}"
    p1 = _case(db, cust, T0, amount=100)
    _settle(db, p1, cust, 100, T0 + timedelta(days=2))
    _settle(db, p1, cust, 100, T0 + timedelta(days=3))  # a second event for the same obligation
    cid = _case(db, cust, T0 + timedelta(days=10))
    fv = derive_recovery_features(db, cid, T0 + timedelta(days=11))
    assert fv.values["number_of_previous_recoveries"] == 1  # one obligation, counted once


def test_history_counts_are_monotonic_over_chronological_obligations(db):
    cust = f"c_{_uid()}"
    seen = []
    for i in range(6):
        cid = _case(db, cust, T0 + timedelta(days=10 * i))
        fv = derive_recovery_features(db, cid, T0 + timedelta(days=10 * i + 1))
        seen.append(
            fv.values["number_of_previous_recoveries"]
            + fv.values["number_of_previous_failed_attempts"]
        )
        if i % 2 == 0:
            _settle(db, cid, cust, 100, T0 + timedelta(days=10 * i + 2))
        else:
            _closed_not_recovered(db, cid, T0 + timedelta(days=10 * i + 2))
    assert seen == sorted(seen) and seen[-1] > seen[0]


def test_missing_telemetry_stays_missing_and_is_reported_blocked(db):
    cid = _case(db, f"c_{_uid()}", T0)
    fv = derive_recovery_features(db, cid, T0 + timedelta(days=1))
    for name in DATA_BLOCKED_FEATURES:
        assert fv.values[name] is None
        assert fv.provenance[name].status == "DATA_BLOCKED"
    assert set(fv.blocked) == set(DATA_BLOCKED_FEATURES)
    assert fv.available_count + len(fv.blocked) <= fv.total_count == 23


def test_positive_diagnosis_evidence_is_used_absence_is_left_missing(db):
    a = _case(
        db,
        f"c_{_uid()}",
        T0,
        diagnosis={"supporting_signals": {"has_expired_card": True, "has_dispute": True}},
    )
    b = _case(
        db,
        f"c_{_uid()}",
        T0,
        diagnosis={"supporting_signals": {"has_expired_card": False, "has_dispute": False}},
    )
    fa = derive_recovery_features(db, a, T0 + timedelta(days=1))
    fb = derive_recovery_features(db, b, T0 + timedelta(days=1))
    assert fa.values["payment_method_valid"] == 0 and fa.values["dispute_indicator"] == 1
    assert fb.values["payment_method_valid"] is None and fb.values["dispute_indicator"] is None


def test_every_feature_has_provenance_with_the_cutoff(db):
    cid = _case(db, f"c_{_uid()}", T0)
    cutoff = T0 + timedelta(days=1)
    fv = derive_recovery_features(db, cid, cutoff)
    assert set(fv.provenance) == set(CANONICAL_RECOVERY_FEATURES)
    assert all(p.cutoff == cutoff.isoformat() for p in fv.provenance.values())


def test_training_inference_feature_schema_parity():
    """The live vector must carry exactly the features the model was trained on."""
    from ml.training.train_recovery_model import CATEGORICAL, FEATURES, NUMERIC

    assert set(CANONICAL_RECOVERY_FEATURES) == set(FEATURES)
    assert len(CANONICAL_RECOVERY_FEATURES) == len(FEATURES) == 23
    assert set(FEATURES) == set(NUMERIC) | set(CATEGORICAL)

    import json
    from pathlib import Path

    from application.recovery_probability_predictor import (
        DEFAULT_REGISTRY_DIR,
        RecoveryProbabilityPredictor,
    )

    pred = RecoveryProbabilityPredictor()
    if pred.model is None:
        pytest.skip("no canonical artifact present")
    meta = json.loads(Path(DEFAULT_REGISTRY_DIR, f"{pred.model_id}_metadata.json").read_text())
    assert set(meta["feature_columns"]) == set(CANONICAL_RECOVERY_FEATURES)
    # categorical features are strings, numeric ones numbers (or None) -- same dtypes as training
    db = SessionLocal()
    try:
        cid = _case(db, f"c_{_uid()}", T0)
        vals = derive_recovery_features(db, cid, T0 + timedelta(days=1)).values
        for k in CATEGORICAL:
            assert vals[k] is None or isinstance(vals[k], str)
        for k in NUMERIC:
            assert vals[k] is None or isinstance(vals[k], (int, float))
        assert (
            pred.predict(vals)["status"] == "SUCCESS"
        )  # the trained pipeline accepts the live vector
    finally:
        db.rollback()
        db.close()


def test_endpoint_contract_and_shadow_erv_are_read_only_and_distinct():
    from application.recovery_probability_predictor import RecoveryProbabilityPredictor

    if RecoveryProbabilityPredictor().model is None:
        pytest.skip("no canonical artifact present")
    client = TestClient(app)
    ing = client.post(
        "/events",
        json={
            "customer_id": f"shadow_{_uid()}",
            "risk_category": "FAILED_PAYMENT",
            "external_system": "t",
            "external_event_id": f"x_{_uid()}",
            "amount": "1000",
            "currency": "INR",
            "raw_payload": {"failure_code": "network_error"},
        },
        headers={"X-API-Key": "test-api-key"},
    )
    cid = ing.json()["case_id"]
    assert (
        client.post(f"/cases/{cid}/advance", headers={"X-API-Key": "test-api-key"}).status_code
        == 200
    )
    before = client.get(f"/system/cases/{cid}", headers={"X-API-Key": "test-api-key"}).json()

    r = client.get(
        f"/cases/{cid}/recovery-probability", headers={"X-API-Key": "test-api-key"}
    ).json()
    assert r["status"] == "SUCCESS" and r["prediction_task"] == "RECOVERY_PROBABILITY"
    assert r["target"] == "recovery_success" and r["calibrated"] is True
    assert r["raw_probability"] is not None and r["calibrated_probability"] is not None
    assert r["expected_recoverable_value"] == pytest.approx(
        1000 * r["calibrated_probability"], abs=0.01
    )
    assert set(r["feature_provenance"]) == set(CANONICAL_RECOVERY_FEATURES)
    assert r["feature_coverage_label"].endswith("/23")

    s = client.get(f"/cases/{cid}/shadow-erv", headers={"X-API-Key": "test-api-key"}).json()
    assert "current_erv_probability" in s
    assert s["shadow_only"] is True and s["canonical_status"] == "SUCCESS"
    assert s["canonical_erv"] == pytest.approx(r["expected_recoverable_value"], abs=0.01)
    assert s["current_probability_source"] != r["model_id"]  # a different model lineage

    after = client.get(f"/system/cases/{cid}", headers={"X-API-Key": "test-api-key"}).json()
    for k in (
        "policy_decision",
        "recommendation",
        "prediction",
        "ml_shadow_prediction",
        "current_state",
    ):
        assert before.get(k) == after.get(k)  # the shadow path changed nothing
