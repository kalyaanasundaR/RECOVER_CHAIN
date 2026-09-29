import json
import os
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from application.dataset_intelligence import CanonicalField
from application.recovery_predictor_ml import MLPaymentFailurePredictor
from domain.models import ExecutionStatus, PolicyDecisionStatus, RiskCategory
from infrastructure.storage import model_registry_dir

client = TestClient(app)


# ---------------------------------------------------------------------------
# Phase 29: Real Universal Bank Dataset End-to-End Workflow Test
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_phase29_universal_bank_dataset_full_lifecycle(tmp_path):
    """
    Validates the complete universal workflow using a novel synthetic bank dataset:
    client_key, txn_ref, monetary_value, event_date, payment_result, payment_channel, region, actual_recovered_amount
    """
    # 1. Generate Synthetic Bank Dataset
    # 60 rows total: 50 valid training rows + 10 edge case/invalid rows
    test_uid = uuid.uuid4().hex[:6]
    valid_rows = []
    for i in range(50):
        # Alternate outcome: "failed" vs "success"
        res_val = "failed" if i % 2 == 0 else "success"
        valid_rows.append(
            {
                "client_key": f"CLI_{test_uid}_{1000 + i}",
                "txn_ref": f"TXN_{test_uid}_{9000 + i}",
                "monetary_value": 75.50 + (i * 3.25),
                "event_date": f"2024-01-{(i % 28) + 1:02d}",
                "payment_result": res_val,
                "payment_channel": "ach" if i % 3 == 0 else "wire" if i % 3 == 1 else "card",
                "region": "NA" if i % 2 == 0 else "EMEA",
                "actual_recovered_amount": 75.50 if res_val == "failed" and i % 4 == 0 else 0.0,
            }
        )

    invalid_rows = [
        # Invalid amount (string)
        {
            "client_key": f"CLI_{test_uid}_ERR1",
            "txn_ref": f"TXN_{test_uid}_E1",
            "monetary_value": "not_a_number",
            "event_date": "2024-02-01",
            "payment_result": "failed",
            "payment_channel": "card",
            "region": "NA",
            "actual_recovered_amount": 0.0,
        },
        # Invalid amount (negative)
        {
            "client_key": f"CLI_{test_uid}_ERR2",
            "txn_ref": f"TXN_{test_uid}_E2",
            "monetary_value": -150.0,
            "event_date": "2024-02-01",
            "payment_result": "failed",
            "payment_channel": "card",
            "region": "NA",
            "actual_recovered_amount": 0.0,
        },
        # Invalid timestamp (unparseable)
        {
            "client_key": f"CLI_{test_uid}_ERR3",
            "txn_ref": f"TXN_{test_uid}_E3",
            "monetary_value": 200.0,
            "event_date": "corrupted_timestamp",
            "payment_result": "failed",
            "payment_channel": "ach",
            "region": "EMEA",
            "actual_recovered_amount": 0.0,
        },
        # Ambiguous target (unknown string)
        {
            "client_key": f"CLI_{test_uid}_ERR4",
            "txn_ref": f"TXN_{test_uid}_E4",
            "monetary_value": 300.0,
            "event_date": "2024-02-02",
            "payment_result": "under_investigation",
            "payment_channel": "wire",
            "region": "APAC",
            "actual_recovered_amount": 0.0,
        },
        # Empty entity ID
        {
            "client_key": "",
            "txn_ref": f"TXN_{test_uid}_E5",
            "monetary_value": 400.0,
            "event_date": "2024-02-03",
            "payment_result": "failed",
            "payment_channel": "card",
            "region": "NA",
            "actual_recovered_amount": 0.0,
        },
    ]

    all_rows = valid_rows + invalid_rows
    df = pd.DataFrame(all_rows)
    csv_file = str(tmp_path / "universal_bank_data.csv")
    df.to_csv(csv_file, index=False)

    # 2. UPLOAD
    filename = f"bank_dataset_{uuid.uuid4().hex[:8]}.csv"
    with open(csv_file, "rb") as f:
        upload_res = client.post("/datasets/upload", files={"file": (filename, f, "text/csv")})
    assert upload_res.status_code == 200, upload_res.json()
    ds_id = upload_res.json()["dataset_id"]

    # 3. PROFILE / ANALYZE
    analyze_res = client.post(f"/datasets/{ds_id}/analyze")
    assert analyze_res.status_code == 200, analyze_res.json()

    ds_meta = client.get(f"/datasets/{ds_id}").json()
    signals = {s["original_column"]: s for s in ds_meta["recoverchain_signals"]}

    # Verify universal semantic inference
    assert signals["client_key"]["canonical_field"] == CanonicalField.CUSTOMER_ID.value
    assert signals["txn_ref"]["canonical_field"] == CanonicalField.TRANSACTION_ID.value
    assert signals["monetary_value"]["canonical_field"] == CanonicalField.AMOUNT.value
    assert signals["event_date"]["canonical_field"] == CanonicalField.TIMESTAMP.value
    assert signals["payment_result"]["canonical_field"] == CanonicalField.OUTCOME.value
    assert signals["payment_channel"]["canonical_field"] in [
        CanonicalField.PAYMENT_METHOD.value,
        CanonicalField.UNKNOWN.value,
    ]
    assert signals["region"]["canonical_field"] == CanonicalField.UNKNOWN.value

    # Verify leakage detection
    leakage = ds_meta.get("leakage_detection", [])
    assert any(l["column"] == "actual_recovered_amount" for l in leakage)

    # 4. PREVIEW ENDPOINT
    prev_res = client.get(f"/datasets/{ds_id}/preview?limit=15")
    assert prev_res.status_code == 200
    prev_data = prev_res.json()
    assert len(prev_data["rows"]) == 15
    assert len(prev_data["columns"]) == 8

    # 5. CONFIRM MAPPING
    mappings = []
    for s in ds_meta["recoverchain_signals"]:
        cf = s["canonical_field"]
        if s["original_column"] == "actual_recovered_amount":
            cf = CanonicalField.UNKNOWN.value
        mappings.append(
            {
                "original_column": s["original_column"],
                "canonical_field": cf,
                "action": "confirm" if cf != CanonicalField.UNKNOWN.value else "unused",
            }
        )
    map_res = client.post(f"/datasets/{ds_id}/mapping", json={"mappings": mappings})
    assert map_res.status_code == 200, map_res.json()

    # 6. ML READINESS
    ml_res = client.post(f"/datasets/{ds_id}/ml-readiness")
    assert ml_res.status_code == 200, ml_res.json()
    ml_spec = ml_res.json()

    assert ml_spec["readiness_status"] in [
        "ML_TRAINING_READY",
        "ML_TRAINING_READY_WITH_EXCLUSIONS",
        "ML_TRAINING_READY_WITH_WARNINGS",
    ]
    assert ml_spec["target_column"] == "payment_result"
    assert "actual_recovered_amount" in ml_spec.get("excluded_columns", [])
    assert "client_key" in ml_spec.get("excluded_columns", [])
    assert "txn_ref" in ml_spec.get("excluded_columns", [])
    assert "monetary_value" in ml_spec.get("feature_columns", [])
    assert "payment_channel" in ml_spec.get("feature_columns", [])
    assert "region" in ml_spec.get("feature_columns", [])

    # 7. TRAIN DATASET MODEL
    train_res = client.post(f"/datasets/{ds_id}/train")
    assert train_res.status_code == 200, train_res.json()
    train_data = train_res.json()
    assert "Training initiated" in train_data["status"] or "SUCCESS" in train_data["status"]

    # Wait for background training completion
    import time

    for _ in range(20):
        ds_status = client.get(f"/datasets/{ds_id}").json().get("status")
        if ds_status == "TRAINED":
            break
        time.sleep(0.5)
    assert client.get(f"/datasets/{ds_id}").json().get("status") == "TRAINED"

    # 8. VERIFY MODEL IN REGISTRY AND METADATA CONTRACT
    models_res = client.get(f"/datasets/{ds_id}/models")
    assert models_res.status_code == 200
    models_list = models_res.json()
    assert len(models_list) >= 1

    # Inspect persisted disk metadata
    reg_entry = models_list[0]
    assert reg_entry["dataset_id"] == ds_id
    assert reg_entry["task"] == "payment-failure-risk"
    assert "actual_recovered_amount" not in reg_entry["feature_columns"]
    assert "actual_recovered_amount" not in reg_entry.get("canonical_feature_mapping", {})

    # 9. SHADOW INFERENCE VIA PREDICTOR & STRICT FEATURE CONTRACT
    predictor = MLPaymentFailurePredictor(dataset_id=ds_id, registry_dir=model_registry_dir())
    assert predictor.model is not None
    assert predictor.metadata.get("task") == "payment-failure-risk"

    # Baseline prediction with exact required features
    pred_base = predictor.predict_failure_risk(
        {
            "monetary_value": 120.0,
            "payment_channel": "card",
            "region": "NA",
            "event_date": "2024-01-15",
        }
    )
    assert pred_base["status"] == "SUCCESS"
    assert 0.0 <= pred_base["probability"] <= 1.0
    assert pred_base["model_metadata"]["shadow_mode_active"] is True
    assert "actual_recovered_amount" not in pred_base["model_metadata"]["features_used"]

    # Leakage and extraneous fields cannot alter prediction or feature schema
    pred_with_leakage = predictor.predict_failure_risk(
        {
            "monetary_value": 120.0,
            "payment_channel": "card",
            "region": "NA",
            "event_date": "2024-01-15",
            "actual_recovered_amount": 999999.0,
            "unrelated_extra_attribute": "junk_value",
        }
    )
    assert pred_with_leakage["status"] == "SUCCESS"
    assert pred_with_leakage["probability"] == pred_base["probability"]
    assert "actual_recovered_amount" not in pred_with_leakage["model_metadata"]["features_used"]

    # Missing required feature must explicitly raise ValueError (never guessed)
    with pytest.raises(ValueError, match="Incomplete feature set"):
        predictor.predict_failure_risk(
            {
                "monetary_value": 120.0,
                "payment_channel": "card",
                # Missing event_date and region
            }
        )

    # 10. GENERATE CASES WITH PROCESSING COUNTERS
    gen_res = client.post(f"/datasets/{ds_id}/generate-cases", json={"max_cases": 55})
    assert gen_res.status_code == 200, gen_res.json()
    gen_data = gen_res.json()

    assert gen_data["status"] == "SUCCESS"
    assert "counters" in gen_data
    counters = gen_data["counters"]

    # Verify truthfulness of counters
    assert counters["rows_seen"] == 55
    assert counters["rows_accepted"] == 25  # 25 failed payments in valid rows
    assert counters["rows_skipped"] == 30  # 25 successful payments + 5 invalid rows
    assert counters["invalid_amount"] == 2  # 1 string + 1 negative
    assert counters["invalid_timestamp"] == 1
    assert counters["ambiguous_target"] == 1
    assert counters["invalid_entity"] == 1
    assert len(gen_data["case_ids"]) == 25

    # 11. INSPECT GENERATED CASE DETAILS
    case_id = gen_data["case_ids"][0]
    case_res = client.get(f"/cases/{case_id}")
    assert case_res.status_code == 200
    c = case_res.json()

    assert c["case_id"] == case_id
    assert c["risk_category"] == RiskCategory.FAILED_PAYMENT.value
    assert float(str(c["amount_at_risk"]).replace("$", "").strip()) > 0
    assert c["currency"] == "INR"

    # Verify shadow prediction attached
    assert c["recovery_probability"] is not None
    assert 0.0 <= c["recovery_probability"] <= 1.0

    # Verify deterministic Policy Decision
    assert c["policy_status"] in [
        PolicyDecisionStatus.PERMITTED.value,
        PolicyDecisionStatus.DENIED.value,
        "ESCALATE",
        "APPROVED",
        "REJECTED",
    ]

    # Verify simulated Execution Record
    if c["policy_status"] == PolicyDecisionStatus.PERMITTED.value:
        assert c["execution_status"] == ExecutionStatus.COMPLETED_SIMULATED.value


# ---------------------------------------------------------------------------
# Phase 29: Cross-Dataset Isolation, Multi-Task Separation, and Security Verification
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_phase29_dataset_isolation_and_security(tmp_path):
    """
    Verifies that:
    1. Dataset A and Dataset B maintain absolute model and inference isolation.
    2. Multi-task separation: models trained on the same dataset for other tasks (e.g. churn)
       cannot be loaded by MLPaymentFailurePredictor.
    3. Path traversal attacks and malformed dataset identifiers fail safely.
    """
    # Create Dataset A
    df_a = pd.DataFrame(
        {
            "client_key": [f"ALPHA_{i}" for i in range(50)],
            "monetary_value": [50.0 + i for i in range(50)],
            "event_date": ["2024-01-01" for _ in range(50)],
            "payment_result": ["failed" if i % 2 == 0 else "success" for i in range(50)],
            "region": ["US" for _ in range(50)],
        }
    )
    path_a = str(tmp_path / "ds_alpha.csv")
    df_a.to_csv(path_a, index=False)

    # Create Dataset B
    df_b = pd.DataFrame(
        {
            "account_no": [f"BETA_{i}" for i in range(50)],
            "payment_amt": [200.0 + i for i in range(50)],
            "timestamp": ["2024-02-01" for _ in range(50)],
            "outcome": [1 if i % 3 == 0 else 0 for i in range(50)],
            "channel": ["web" for _ in range(50)],
        }
    )
    path_b = str(tmp_path / "ds_beta.csv")
    df_b.to_csv(path_b, index=False)

    # Upload A and B
    fname_a = f"alpha_{uuid.uuid4().hex[:8]}.csv"
    with open(path_a, "rb") as f:
        ds_a_id = client.post("/datasets/upload", files={"file": (fname_a, f, "text/csv")}).json()[
            "dataset_id"
        ]

    fname_b = f"beta_{uuid.uuid4().hex[:8]}.csv"
    with open(path_b, "rb") as f:
        ds_b_id = client.post("/datasets/upload", files={"file": (fname_b, f, "text/csv")}).json()[
            "dataset_id"
        ]

    # Analyze, confirm, train Dataset A ONLY for payment-failure-risk
    client.post(f"/datasets/{ds_a_id}/analyze")
    ds_a_data = client.get(f"/datasets/{ds_a_id}").json()
    mappings_a = [
        {
            "original_column": s["original_column"],
            "canonical_field": s["canonical_field"],
            "action": "confirm",
        }
        for s in ds_a_data["recoverchain_signals"]
    ]
    client.post(f"/datasets/{ds_a_id}/mapping", json={"mappings": mappings_a})
    client.post(f"/datasets/{ds_a_id}/ml-readiness")
    client.post(f"/datasets/{ds_a_id}/train")

    # Verify Dataset A has payment-failure-risk model
    pred_a = MLPaymentFailurePredictor(dataset_id=ds_a_id, registry_dir=model_registry_dir())
    assert pred_a.model is not None
    assert pred_a.metadata.get("task") == "payment-failure-risk"

    # Verify Dataset B has NO model (isolation)
    pred_b = MLPaymentFailurePredictor(dataset_id=ds_b_id, registry_dir=model_registry_dir())
    assert pred_b.model is None

    # Dataset B cannot generate cases using Dataset A's model
    gen_b = client.post(f"/datasets/{ds_b_id}/generate-cases", json={"max_cases": 10})
    assert gen_b.status_code == 400
    assert "ML ready" in gen_b.json()["detail"]

    # MULTI-TASK ISOLATION ON SAME DATASET:
    # Create a metadata entry for ds_a_id with a different task (e.g. churn-prediction)
    churn_meta = {
        "model_id": f"{ds_a_id}_churn_run",
        "dataset_id": ds_a_id,
        "task": "churn-prediction",
        "target_column": "churn_label",
        "feature_columns": ["monetary_value"],
        "artifact_path": pred_a.metadata.get("artifact_path"),
    }
    churn_meta_path = os.path.join(model_registry_dir(), f"{ds_a_id}_churn_run_metadata.json")
    with open(churn_meta_path, "w") as f_meta:
        json.dump(churn_meta, f_meta)

    try:
        # Requesting payment-failure-risk must resolve ONLY payment-failure-risk, not the churn-prediction model
        pred_a_task = MLPaymentFailurePredictor(
            dataset_id=ds_a_id, registry_dir=model_registry_dir()
        )
        assert pred_a_task.model is not None
        assert pred_a_task.metadata.get("task") == "payment-failure-risk"

        # If a dataset ONLY has a churn-prediction model, MLPaymentFailurePredictor must NOT load it
        ds_dummy_id = f"ds_churn_only_{uuid.uuid4().hex[:6]}"
        dummy_meta = {
            "model_id": f"{ds_dummy_id}_run",
            "dataset_id": ds_dummy_id,
            "task": "churn-prediction",
            "feature_columns": ["monetary_value"],
            "artifact_path": pred_a.metadata.get("artifact_path"),
        }
        dummy_meta_path = os.path.join(model_registry_dir(), f"{ds_dummy_id}_run_metadata.json")
        with open(dummy_meta_path, "w") as f_d:
            json.dump(dummy_meta, f_d)

        try:
            pred_dummy = MLPaymentFailurePredictor(
                dataset_id=ds_dummy_id, registry_dir=model_registry_dir()
            )
            assert pred_dummy.model is None  # Must NOT load churn-prediction model
        finally:
            if os.path.exists(dummy_meta_path):
                os.remove(dummy_meta_path)
    finally:
        if os.path.exists(churn_meta_path):
            os.remove(churn_meta_path)

    # Path traversal and invalid identifiers rejected safely
    for attack_payload in [
        "../../etc/passwd",
        "..\\..\\some_other_dataset",
        "/etc/shadow",
        "C:\\Windows\\System32",
        "invalid id with spaces!",
        "ds_nonexistent_12345",
    ]:
        pred_trav = MLPaymentFailurePredictor(
            dataset_id=attack_payload, registry_dir=model_registry_dir()
        )
        assert pred_trav.model is None
        assert pred_trav.is_legacy is False
