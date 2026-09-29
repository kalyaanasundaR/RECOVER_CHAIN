import os
import uuid
from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from application.dataset_lab import DatasetLabService
from infrastructure.dataset_orm import DatasetMetadataModel, DatasetStatus
from infrastructure.db import SessionLocal


@pytest.mark.fast
def test_malformed_dataset_case_generation_safety_and_counters():
    """
    Phase 34.1: Prove that the existing case-generation pathway strictly rejects
    malformed records and non-failed rows without fabricating financial values.
    """
    client = TestClient(app)
    db = SessionLocal()
    service = DatasetLabService(db)

    dataset_id = f"ds_phase34_1_{uuid.uuid4().hex[:8]}"

    # 11-row deterministic fixture
    data = {
        "client_ref": [
            "CUST_001",  # 1. Valid
            "CUST_002",  # 2. Valid
            "CUST_003",  # 3. Settled/Success (outcome 0 -> skipped)
            "CUST_004",  # 4. Missing amount
            "CUST_005",  # 5. Non-numeric amount
            "CUST_006",  # 6. Negative amount
            None,  # 7. Missing entity
            "CUST_008",  # 8. Missing timestamp
            "CUST_009",  # 9. Malformed timestamp
            "CUST_010",  # 10. Ambiguous target
            "CUST_011",  # 11. Missing target
        ],
        "monetary_amt": [
            150.00,  # 1. Valid
            300.50,  # 2. Valid
            200.00,  # 3. Valid
            None,  # 4. Missing
            "INVALID_AMT",  # 5. Non-numeric
            -50.00,  # 6. Negative
            100.00,  # 7. Valid amount, missing entity
            75.00,  # 8. Valid amount, missing timestamp
            85.00,  # 9. Valid amount, bad timestamp
            95.00,  # 10. Valid amount, ambiguous target
            110.00,  # 11. Valid amount, missing target
        ],
        "event_time": [
            "2026-08-01T10:00:00Z",  # 1. Valid
            "2026-08-02T11:00:00Z",  # 2. Valid
            "2026-08-03T12:00:00Z",  # 3. Valid
            "2026-08-04T13:00:00Z",  # 4. Valid
            "2026-08-05T14:00:00Z",  # 5. Valid
            "2026-08-06T15:00:00Z",  # 6. Valid
            "2026-08-07T16:00:00Z",  # 7. Valid
            None,  # 8. Missing
            "NOT_A_VALID_DATE_TIME",  # 9. Malformed
            "2026-08-10T19:00:00Z",  # 10. Valid
            "2026-08-11T20:00:00Z",  # 11. Valid
        ],
        "failure_target": [
            1,  # 1. Failed (Valid case)
            "failed",  # 2. Failed (Valid case)
            0,  # 3. Settled/Success (Skipped, no case)
            1,  # 4. Failed
            1,  # 5. Failed
            1,  # 6. Failed
            1,  # 7. Failed
            1,  # 8. Failed
            1,  # 9. Failed
            "MAYBE_DECLINED",  # 10. Ambiguous target
            None,  # 11. Missing target
        ],
        "actual_recovered": [
            0.0,
            0.0,
            200.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # Leakage column
        ],
        "region_code": [
            "US_EAST",
            "US_WEST",
            "EU_CENTRAL",
            "US_EAST",
            "US_WEST",
            "EU_CENTRAL",
            "US_EAST",
            "US_WEST",
            "EU_CENTRAL",
            "US_EAST",
            "US_WEST",
        ],
    }

    df = pd.DataFrame(data)
    file_path = os.path.join(service.dataset_dir, f"{dataset_id}.csv")
    df.to_csv(file_path, index=False)

    try:
        # Create Dataset Metadata with Confirmed Mappings and Leakage Exclusions
        meta = DatasetMetadataModel(
            dataset_id=dataset_id,
            name="Malformed Regression Fixture",
            filename=f"{dataset_id}.csv",
            file_type="csv",
            file_size_bytes=os.path.getsize(file_path),
            upload_timestamp=datetime.now(UTC),
            row_count=11,
            column_count=6,
            status=DatasetStatus.COMPLETED,
            # Phase 4: nothing in these columns identifies the revenue domain, and an
            # undeclared domain is no longer assumed to be a failed payment.
            declared_domain="FAILED_PAYMENT",
            recoverchain_signals=[
                {
                    "original_column": "client_ref",
                    "canonical_field": "ENTITY_ID",
                    "action": "confirm",
                    "confidence": "HIGH",
                },
                {
                    "original_column": "monetary_amt",
                    "canonical_field": "AMOUNT",
                    "action": "confirm",
                    "confidence": "HIGH",
                },
                {
                    "original_column": "event_time",
                    "canonical_field": "TIMESTAMP",
                    "action": "confirm",
                    "confidence": "HIGH",
                },
                {
                    "original_column": "failure_target",
                    "canonical_field": "OUTCOME",
                    "action": "confirm",
                    "confidence": "HIGH",
                },
                {
                    "original_column": "actual_recovered",
                    "canonical_field": "UNKNOWN",
                    "action": "unused",
                    "confidence": "LOW",
                },
                {
                    "original_column": "region_code",
                    "canonical_field": "UNKNOWN",
                    "action": "unused",
                    "confidence": "LOW",
                },
            ],
            leakage_detection=[
                {"column": "actual_recovered", "reason": "Post-outcome recovery value"}
            ],
            training_suitability={
                "prediction_problem": "payment-failure-risk",
                "target_column": "failure_target",
                "feature_columns": ["monetary_amt", "region_code"],
                "excluded_columns": ["actual_recovered"],
                "readiness_status": "ML_TRAINING_READY",
            },
        )
        db.add(meta)
        db.commit()

        # Execute Generate Cases
        res = client.post(f"/datasets/{dataset_id}/generate-cases", json={"max_cases": 25})
        assert res.status_code == 200
        body = res.json()

        assert body["status"] == "SUCCESS"
        assert body["cases_generated"] == 2
        assert len(body["case_ids"]) == 2

        counters = body["counters"]
        # Numerical counter verification against deterministic fixture:
        assert counters["rows_seen"] == 11, f"Expected 11 rows seen, got {counters['rows_seen']}"
        assert counters["rows_accepted"] == 2, (
            f"Expected 2 accepted cases, got {counters['rows_accepted']}"
        )
        assert counters["rows_skipped"] == 9, (
            f"Expected 9 skipped rows, got {counters['rows_skipped']}"
        )
        assert counters["invalid_amount"] == 3, (
            f"Expected 3 invalid amounts (null, string, negative), got {counters['invalid_amount']}"
        )
        assert counters["invalid_entity"] == 1, (
            f"Expected 1 invalid entity (null), got {counters['invalid_entity']}"
        )
        assert counters["invalid_timestamp"] == 2, (
            f"Expected 2 invalid timestamps (null, malformed), got {counters['invalid_timestamp']}"
        )
        assert counters["ambiguous_target"] == 1, (
            f"Expected 1 ambiguous target ('MAYBE_DECLINED'), got {counters['ambiguous_target']}"
        )
        assert counters["invalid_target"] == 1, (
            f"Expected 1 invalid target (null), got {counters['invalid_target']}"
        )

        # Verify the 2 accepted cases and non-fabrication
        for cid in body["case_ids"]:
            c_res = client.get(f"/cases/{cid}")
            assert c_res.status_code == 200
            c_data = c_res.json()

            # Non-fabrication verification: exact amounts and customer IDs preserved
            amt_float = float(c_data["amount_at_risk"])
            assert amt_float in [150.0, 300.5], (
                "Amount must match exact source row, not fabricated to 0"
            )
            assert c_data["customer_id"] in ["CUST_001", "CUST_002"], (
                "Customer ID must match exact source row, not fabricated"
            )
            assert c_data["risk_category"] == "FAILED_PAYMENT"

    finally:
        db.close()
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.mark.fast
def test_accepted_case_full_safety_lifecycle():
    """
    Verify that an accepted case follows the full safety chain:
    Case -> ML Shadow Prediction -> Deterministic Policy Engine -> Simulated Execution -> Verification
    with ML remaining SHADOW_ONLY and Policy Engine remaining sole authority.
    """
    client = TestClient(app)
    headers = {"X-API-Key": os.getenv("API_KEY", "test-api-key")}

    # 1. Ingest event
    ingest_payload = {
        "customer_id": "CUST_SAFETY_TEST",
        "risk_category": "FAILED_PAYMENT",
        "external_system": "STRIPE_MOCK_SANDBOX",
        "external_event_id": f"evt_safety_{uuid.uuid4().hex[:6]}",
        "reference_id": f"ref_safety_{uuid.uuid4().hex[:6]}",
        "amount": 250.0,
        "currency": "USD",
        "raw_payload": {"reason": "insufficient_funds"},
    }
    i_res = client.post("/events", json=ingest_payload, headers=headers)
    assert i_res.status_code == 200
    case_id = i_res.json()["case_id"]

    # 2. Risk Detection
    r_res = client.post(f"/cases/{case_id}/assess-risk")
    assert r_res.status_code == 200

    # 3. Diagnosis
    d_res = client.post(f"/cases/{case_id}/diagnose")
    assert d_res.status_code == 200

    # 4. Shadow ML Prediction
    p_res = client.post(f"/cases/{case_id}/predict-recovery")
    assert p_res.status_code == 200
    pred = p_res.json()
    # ML stays shadow-only; when no dataset-isolated model exists we fall back to
    # the deterministic baseline rather than a meaningless 0.0.
    assert pred["prediction_status"] in ["SHADOW_ONLY", "NO_MODEL", "SUCCESS_BASELINE"]
    assert 0.0 <= pred["recovery_probability"] <= 1.0

    # 5. Recommendation
    rec_res = client.post(f"/cases/{case_id}/recommend-action")
    assert rec_res.status_code == 200

    # 6. Policy Check (Sole Authority)
    pol_res = client.post(f"/cases/{case_id}/policy-check")
    assert pol_res.status_code == 200
    pol_data = pol_res.json()
    assert pol_data["status"] in ["PERMITTED", "WAIT", "ESCALATE", "DENIED"]

    # 7. Sandbox Execution (Simulated only)
    if pol_data["status"] == "PERMITTED":
        exec_res = client.post(f"/cases/{case_id}/execute", headers=headers)
        assert exec_res.status_code == 200
        assert exec_res.json()["status"] == "COMPLETED_SIMULATED"

        # 8. Verification
        v_res = client.post(
            f"/cases/{case_id}/verify", json={"external_reference": exec_res.json()["execution_id"]}
        )
        assert v_res.status_code == 200
        assert v_res.json()["status"] in ["FULLY_RECOVERED", "PARTIALLY_RECOVERED", "NOT_RECOVERED"]


@pytest.mark.fast
def test_dataset_isolation_and_no_model_behavior():
    """
    Verify that an un-modeled dataset safely returns NO_MODEL telemetry
    without fabricating probabilities or borrowing models from another dataset.
    """
    from application.recovery_predictor_ml import MLPaymentFailurePredictor

    # Non-existent isolated dataset ID
    pred = MLPaymentFailurePredictor(dataset_id="ds_isolated_unmodeled_9999")
    res = pred.predict_failure_risk({"monetary_amt": 500.0})

    assert res["status"] == "NO_MODEL"
    assert res["probability"] == 0.0
