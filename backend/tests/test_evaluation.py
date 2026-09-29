import pytest
from fastapi.testclient import TestClient

from api.main import app
from infrastructure.db import Base, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    # Ensure the schema exists (idempotent). This file must NOT drop it afterwards: the
    # session-scoped `_test_database` fixture in conftest.py owns the schema lifecycle.
    # A per-test `drop_all` here destroyed the shared operational-test schema for every
    # test that ran after this file and only "worked" in the full suite because later
    # modules happen to call create_all themselves (Phase 16 root-cause fix).
    Base.metadata.create_all(bind=engine)
    yield


def test_run_evaluation():
    resp = client.post("/evaluation/run")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_scenarios" in data
    assert data["total_scenarios"] == 12
    print(
        [
            r
            for r in data["scenario_results"]
            if not r["outcome_matched"]
            or not r["policy_matched"]
            or not r["diagnosis_matched"]
            or not r["action_matched"]
        ]
    )
    assert data["passed_scenarios"] == 12  # If our rules perfectly match gold
    assert data["unsafe_execution_rate"] == 0.0
    assert data["policy_bypass_rate"] == 0.0
    assert data["policy_bypass_rate"] == 0.0

    # Check results manually
    results = data["scenario_results"]
    assert len(results) == 12

    # 1. Standard
    assert results[0]["policy_decision_status"]
    assert results[5]["execution_status"] == "REJECTED"
