import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def phase27_synthetic_bank_datasets(tmp_path):
    df_a = pd.DataFrame(
        {
            "account_no": ["CUST1", "CUST2", "CUST3", "CUST4"],
            "txn_amount": [10.5, 20.0, 15.0, 100.0],
            "transaction_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "payment_failed": [0, 1, 0, 0],
            "failure_code": ["", "insufficient_funds", "", ""],
        }
    )
    path_a = str(tmp_path / "bank_phase27.csv")
    df_a.to_csv(path_a, index=False)
    return {"A": path_a}


@pytest.mark.fast
def test_phase27_e2e_dataset_to_case_workflow(phase27_synthetic_bank_datasets):
    # 1. Upload
    with open(phase27_synthetic_bank_datasets["A"], "rb") as f:
        res = client.post(
            "/datasets/upload",
            files={"file": (f"ui_test_{uuid.uuid4().hex[:8]}.csv", f, "text/csv")},
        )
    assert res.status_code == 200, res.text
    ds_id = res.json()["dataset_id"]

    # 2. Analyze
    res = client.post(f"/datasets/{ds_id}/analyze")
    assert res.status_code == 200, (
        f"{res.text} | Suitability: {client.get(f'/datasets/{ds_id}').json().get('training_suitability')}"
    )

    # 3. Confirm Mappings
    ds = client.get(f"/datasets/{ds_id}").json()
    mappings = []
    for sig in ds["recoverchain_signals"]:
        mappings.append(
            {
                "original_column": sig["original_column"],
                "canonical_field": sig["canonical_field"],
                "action": "confirm",
            }
        )
    client.post(f"/datasets/{ds_id}/mapping", json={"mappings": mappings})

    # 4. ML Readiness
    res = client.post(f"/datasets/{ds_id}/ml-readiness")
    assert res.status_code == 200, (
        f"{res.text} | Suitability: {client.get(f'/datasets/{ds_id}').json().get('training_suitability')}"
    )

    # 5. Train
    res = client.post(f"/datasets/{ds_id}/train")
    assert res.status_code == 200, (
        f"{res.text} | Suitability: {client.get(f'/datasets/{ds_id}').json().get('training_suitability')}"
    )

    # Simulate wait for training
    import time

    for _ in range(10):
        if client.get(f"/datasets/{ds_id}").json()["status"] == "TRAINED":
            break
        time.sleep(1)

    # 6. Generate Cases
    res = client.post(f"/datasets/{ds_id}/generate-cases", json={"max_cases": 10})
    assert res.status_code == 200, (
        f"{res.text} | Suitability: {client.get(f'/datasets/{ds_id}').json().get('training_suitability')}"
    )
    data = res.json()
    assert data["cases_generated"] > 0
    case_id = data["case_ids"][0]

    # 7. Verify Case State
    case_res = client.get(f"/cases/{case_id}")
    assert case_res.status_code == 200
    case_data = case_res.json()

    # Check all requested boundaries exist and are distinct
    assert float(case_data["amount_at_risk"]) == 20.0
    assert case_data["customer_id"] == "CUST2"
    assert "cause_category" in case_data
    assert case_data["recovery_probability"] is not None
    assert case_data["expected_recoverable_value"] is not None, case_data
    assert case_data["recommended_action"] is not None
    assert case_data["policy_status"] is not None
    assert case_data["execution_status"] is not None
    # Phase 5: generate-cases simulates the ACTION (execution_status above) but never
    # fabricates a financial outcome. outcome_status stays unset until a real outcome
    # event is ingested (POST /outcomes) for this obligation; /verify honestly reports
    # PENDING_VERIFICATION in the meantime.
    assert case_data["outcome_status"] is None
    verify_res = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""})
    assert verify_res.status_code == 200
    assert verify_res.json()["status"] == "PENDING_VERIFICATION"

    # 8. Check Audit
    audit_res = client.get(f"/cases/{case_id}/audit")
    assert audit_res.status_code == 200
    audit_events = audit_res.json()
    assert len(audit_events) > 0
