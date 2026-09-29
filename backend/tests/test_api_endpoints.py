from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_ingest_event_api():
    payload = {
        "customer_id": "cust_api_1",
        "risk_category": "FAILED_PAYMENT",
        "external_system": "stripe",
        "external_event_id": "ch_12345",
        "reference_id": "tx_api_999",
        "amount": 99.99,
        "currency": "USD",
        "raw_payload": {"foo": "bar"},
    }

    response = client.post("/events", headers={"X-API-Key": "test-api-key"}, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_new_case"] is True
    assert data["status"] == "success"
    case_id = data["case_id"]

    # Test deduplication on API layer
    resp2 = client.post("/events", headers={"X-API-Key": "test-api-key"}, json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "ignored"

    # Test Retrieval
    resp3 = client.get(f"/cases/{case_id}")
    assert resp3.status_code == 200
    case_data = resp3.json()
    assert case_data["amount_at_risk"] == "99.9900"
    assert case_data["event_count"] == 1

    # Test related event attachment
    payload2 = payload.copy()
    payload2["external_event_id"] = "ch_67890"
    payload2["amount"] = 105.00
    resp4 = client.post("/events", headers={"X-API-Key": "test-api-key"}, json=payload2)
    assert resp4.json()["is_new_case"] is False
    assert resp4.json()["case_id"] == case_id

    resp5 = client.get(f"/cases/{case_id}")
    assert resp5.json()["amount_at_risk"] == "105.0000"
    assert resp5.json()["event_count"] == 2

    # Test Phase 5 Risk Assessment API
    resp_risk = client.post(f"/cases/{case_id}/assess-risk")
    assert resp_risk.status_code == 200
    risk_data = resp_risk.json()
    assert "score" in risk_data
    assert "risk_level" in risk_data
    assert risk_data["detector_version"] == "deterministic-v1.0"

    resp_risk_get = client.get(f"/cases/{case_id}/risk")
    assert resp_risk_get.status_code == 200
    assert resp_risk_get.json()["score"] == risk_data["score"]

    # Test Case response includes risk level now
    resp_case2 = client.get(f"/cases/{case_id}")
    assert resp_case2.json()["risk_level"] == risk_data["risk_level"]

    # Test Audit Trail includes risk assessment
    resp6 = client.get(f"/cases/{case_id}/audit")
    assert resp6.status_code == 200
    audits = resp6.json()
    assert len(audits) >= 3  # created, attached, risk_assessment
    assert any(a["evidence"].get("action") == "risk_assessment" for a in audits)

    # Test Phase 6 Diagnosis API
    resp_diag = client.post(f"/cases/{case_id}/diagnose")
    assert resp_diag.status_code == 200
    diag_data = resp_diag.json()
    assert "cause_category" in diag_data
    assert "confidence" in diag_data
    assert diag_data["diagnostic_method"] == "deterministic-v1.0"

    resp_diag_get = client.get(f"/cases/{case_id}/diagnosis")
    assert resp_diag_get.status_code == 200
    assert resp_diag_get.json()["cause_category"] == diag_data["cause_category"]

    # Test Case response includes diagnosis
    resp_case3 = client.get(f"/cases/{case_id}")
    assert resp_case3.json()["cause_category"] == diag_data["cause_category"]

    # Test Audit Trail includes diagnosis
    resp7 = client.get(f"/cases/{case_id}/audit")
    assert resp7.status_code == 200
    audits = resp7.json()
    assert any(a["evidence"].get("action") == "diagnosis" for a in audits)

    # Test Phase 7 Recovery Prediction API
    resp_pred = client.post(f"/cases/{case_id}/predict-recovery")
    assert resp_pred.status_code == 200
    pred_data = resp_pred.json()
    assert "recovery_probability" in pred_data
    assert "model_version" in pred_data

    resp_pred_get = client.get(f"/cases/{case_id}/recovery-prediction")
    assert resp_pred_get.status_code == 200
    assert resp_pred_get.json()["recovery_probability"] == pred_data["recovery_probability"]

    # Test Case response includes prediction
    resp_case4 = client.get(f"/cases/{case_id}")
    assert resp_case4.json()["recovery_probability"] == pred_data["recovery_probability"]

    # Test Audit Trail includes prediction
    resp8 = client.get(f"/cases/{case_id}/audit")
    assert resp8.status_code == 200
    audits = resp8.json()
    assert any(a["evidence"].get("action") == "recovery_prediction" for a in audits)

    # Test Phase 8 Recommendation API
    resp_rec = client.post(f"/cases/{case_id}/recommend-action")
    assert resp_rec.status_code == 200
    rec_data = resp_rec.json()
    assert "candidates" in rec_data
    assert "top_candidate" in rec_data
    assert rec_data["status"] == "RECOMMENDED"

    resp_rec_get = client.get(f"/cases/{case_id}/recommendation")
    assert resp_rec_get.status_code == 200
    assert (
        resp_rec_get.json()["top_candidate"]["action_type"]
        == rec_data["top_candidate"]["action_type"]
    )

    # Test Case response includes recommendation
    resp_case5 = client.get(f"/cases/{case_id}")
    assert resp_case5.json()["recommended_action"] == rec_data["top_candidate"]["action_type"]

    # Test Audit Trail includes recommendation
    resp9 = client.get(f"/cases/{case_id}/audit")
    assert resp9.status_code == 200
    audits = resp9.json()
    assert any(a["evidence"].get("action") == "action_recommendation" for a in audits)

    # Test Phase 9 Policy API
    resp_pol = client.post(f"/cases/{case_id}/policy-check")
    assert resp_pol.status_code == 200
    pol_data = resp_pol.json()
    assert "status" in pol_data
    assert "rules_evaluated" in pol_data
    assert pol_data["status"] == "ESCALATE"

    resp_pol_get = client.get(f"/cases/{case_id}/policy-decision")
    assert resp_pol_get.status_code == 200
    assert resp_pol_get.json()["status"] == pol_data["status"]

    # Test Case response includes policy status
    resp_case6 = client.get(f"/cases/{case_id}")
    assert resp_case6.json()["policy_status"] == pol_data["status"]

    # Test Audit Trail includes policy
    resp10 = client.get(f"/cases/{case_id}/audit")
    assert resp10.status_code == 200
    audits = resp10.json()
    assert any(a["evidence"].get("action") == "policy_evaluation" for a in audits)

    # Test Phase 10 Execution API
    # Force the policy to PERMITTED manually to test the execution
    # Actually, since it ESCALATED, if we try to execute, it should return REJECTED!
    resp_exec = client.post(f"/cases/{case_id}/execute", headers={"X-API-Key": "test-api-key"})
    assert resp_exec.status_code == 200
    exec_data = resp_exec.json()
    assert exec_data["status"] == "REJECTED"

    # Test Case response includes execution status
    resp_case7 = client.get(f"/cases/{case_id}")
    assert resp_case7.json()["execution_status"] == exec_data["status"]

    # Test Audit Trail includes execution
    resp11 = client.get(f"/cases/{case_id}/audit")
    assert resp11.status_code == 200
    audits = resp11.json()
    assert any(a["evidence"].get("action") == "execution" for a in audits)

    # Test Phase 11 Outcome Verification API. Verification is a READ of whatever an
    # authoritative outcome event has already reconciled (Phase 5) - it does not itself
    # decide anything, so first there must be a real settlement for this obligation.
    resp_before_outcome = client.post(
        f"/cases/{case_id}/verify", json={"external_reference": "anything"}
    )
    assert resp_before_outcome.status_code == 200
    assert resp_before_outcome.json()["status"] == "PENDING_VERIFICATION"

    resp_settle = client.post(
        "/outcomes",
        headers={"X-API-Key": "test-api-key"},
        json={
            "external_system": "stripe",
            "external_event_id": "ch_settlement_1",
            "domain": "FAILED_PAYMENT",
            "outcome_type": "PAYMENT_SETTLED",
            "customer_id": "cust_api_1",
            "reference_id": "tx_api_999",
            "amount": 105.00,
            "currency": "USD",
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    )
    assert resp_settle.status_code == 200, resp_settle.text
    assert resp_settle.json()["match_status"] == "MATCHED"
    assert resp_settle.json()["case_id"] == case_id

    resp_verify = client.post(f"/cases/{case_id}/verify", json={"external_reference": "sim_full"})
    assert resp_verify.status_code == 200
    verify_data = resp_verify.json()
    assert verify_data["status"] == "FULLY_RECOVERED"
    assert verify_data["actual_amount_recovered"] == "105.0000"

    # Test Case response includes outcome
    resp_case8 = client.get(f"/cases/{case_id}")
    assert resp_case8.json()["outcome_status"] == verify_data["status"]
    assert resp_case8.json()["actual_amount_recovered"] == verify_data["actual_amount_recovered"]

    # Test Audit Trail includes the outcome reconciliation
    resp12 = client.get(f"/cases/{case_id}/audit")
    assert resp12.status_code == 200
    audits = resp12.json()
    assert any(a["evidence"].get("action") == "outcome_reconciled" for a in audits)
