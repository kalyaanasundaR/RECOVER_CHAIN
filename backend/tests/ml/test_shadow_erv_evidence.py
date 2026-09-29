"""Shadow-ERV / canonical endpoint evidence quality: explicit separation of production vs
canonical values, calibration basis, integrity status, refusal without fallback, read-only."""

import uuid

import pytest
from fastapi.testclient import TestClient

import application.recovery_probability_predictor as rpp
from api.main import app

H = {"X-API-Key": "test-api-key"}


def _case(client):
    r = client.post(
        "/events",
        json={
            "customer_id": f"ev_{uuid.uuid4().hex[:8]}",
            "risk_category": "FAILED_PAYMENT",
            "external_system": "t",
            "external_event_id": f"x_{uuid.uuid4().hex[:8]}",
            "amount": "1500",
            "currency": "INR",
            "raw_payload": {"failure_code": "network_error"},
        },
        headers=H,
    )
    cid = r.json()["case_id"]
    assert client.post(f"/cases/{cid}/advance", headers=H).status_code == 200
    return cid


def test_shadow_erv_separates_production_and_canonical_and_is_never_the_decision():
    if rpp.RecoveryProbabilityPredictor().model is None:
        pytest.skip("no production canonical artifact present")
    c = TestClient(app)
    cid = _case(c)
    s = c.get(f"/cases/{cid}/shadow-erv", headers=H).json()
    for k in (
        "amount_at_risk",
        "current_probability",
        "current_probability_source",
        "current_erv",
        "current_erv_probability",
        "canonical_probability",
        "canonical_erv",
        "canonical_model_id",
        "canonical_model_version",
        "canonical_calibrated",
        "canonical_calibration_basis",
        "canonical_integrity_status",
        "feature_coverage_label",
        "prediction_timestamp",
        "approximations",
    ):
        assert k in s, k
    assert s["shadow_only"] is True and "unchanged" in s["production_decision_source"]
    assert s["canonical_integrity_status"] == "VERIFIED"
    assert "SYNTHETIC_VALIDATION_ONLY" in s["canonical_calibration_basis"]
    assert any("amount_vs_customer_average" in a for a in s["approximations"])
    assert "not `canonical_erv`" in s["note"]
    assert s["canonical_erv"] == pytest.approx(
        s["amount_at_risk"] * s["canonical_probability"], abs=0.01
    )

    r = c.get(f"/cases/{cid}/recovery-probability", headers=H).json()
    assert r["is_production_decision"] is False and r["integrity_status"] == "VERIFIED"
    assert r["artifact_sha256"] and "READ-ONLY" in r["note"]
    assert "SYNTHETIC_VALIDATION_ONLY" in r["calibration_basis"]  # exists != outcome-validated


def test_tampered_artifact_is_refused_by_the_api_with_no_fallback(
    make_recovery_artifact, monkeypatch
):
    d, _, artifact = make_recovery_artifact()
    with open(artifact, "ab") as f:
        f.write(b"tampered")
    monkeypatch.setattr(rpp, "DEFAULT_REGISTRY_DIR", str(d))
    c = TestClient(app)
    cid = _case(c)
    r = c.get(f"/cases/{cid}/recovery-probability", headers=H).json()
    assert r["status"] == "MODEL_INTEGRITY_FAILED" and r["integrity_status"] == "REFUSED"
    assert r["integrity_failure"]["reason_code"] == "CHECKSUM_MISMATCH"
    assert r["calibrated_probability"] is None and r["expected_recoverable_value"] is None
    s = c.get(f"/cases/{cid}/shadow-erv", headers=H).json()
    assert s["canonical_status"] == "MODEL_INTEGRITY_FAILED"
    assert s["canonical_probability"] is None and s["canonical_erv"] is None
    assert s["current_erv"] is not None  # production numbers are unaffected


def test_missing_registry_reports_no_model_not_a_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(rpp, "DEFAULT_REGISTRY_DIR", str(tmp_path / "empty"))
    c = TestClient(app)
    r = c.get(f"/cases/{_case(c)}/recovery-probability", headers=H).json()
    assert r["status"] == "NO_MODEL" and r["calibrated_probability"] is None
