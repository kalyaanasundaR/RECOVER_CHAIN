import uuid
from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from application.agents import AgentOrchestrator
from application.dataset_intelligence import (
    CanonicalField,
    DatasetClassification,
    DatasetProfiler,
    DatasetValidator,
    SemanticMapper,
)
from application.recovery_predictor_ml import MLPaymentFailurePredictor
from domain.models import (
    ActionType,
    ExecutionStatus,
    Money,
    PolicyDecision,
    PolicyDecisionStatus,
    RecoveryCase,
    RiskCategory,
)
from infrastructure.storage import model_registry_dir

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1-4. Alternate Column Names for Entity, Amount, Timestamp, Outcome
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_alternate_entity_names():
    mapper = SemanticMapper()
    sample_entities = pd.Series([f"ENT_{i}" for i in range(50)])

    assert (
        mapper.map_column("account_no", sample_entities)["canonical_field"]
        == CanonicalField.ACCOUNT_ID.value
    )
    assert (
        mapper.map_column("acct_number", sample_entities)["canonical_field"]
        == CanonicalField.ACCOUNT_ID.value
    )
    assert (
        mapper.map_column("customer_ref", sample_entities)["canonical_field"]
        == CanonicalField.CUSTOMER_ID.value
    )
    assert (
        mapper.map_column("client_key", sample_entities)["canonical_field"]
        == CanonicalField.CUSTOMER_ID.value
    )
    assert (
        mapper.map_column("user_identifier", sample_entities)["canonical_field"]
        == CanonicalField.CUSTOMER_ID.value
    )


@pytest.mark.fast
def test_alternate_amount_names():
    mapper = SemanticMapper()
    sample_amounts = pd.Series([10.5, 20.0, 99.99, 1500.50, 42.0])

    assert (
        mapper.map_column("transaction_value", sample_amounts)["canonical_field"]
        == CanonicalField.AMOUNT.value
    )
    assert (
        mapper.map_column("debit_value", sample_amounts)["canonical_field"]
        == CanonicalField.AMOUNT.value
    )
    assert (
        mapper.map_column("payment_amt", sample_amounts)["canonical_field"]
        == CanonicalField.AMOUNT.value
    )
    assert (
        mapper.map_column("invoice_total", sample_amounts)["canonical_field"]
        == CanonicalField.AMOUNT.value
    )
    assert (
        mapper.map_column("monetary_value", sample_amounts)["canonical_field"]
        == CanonicalField.AMOUNT.value
    )
    assert (
        mapper.map_column("outstanding_balance", sample_amounts)["canonical_field"]
        == CanonicalField.BALANCE.value
    )


@pytest.mark.fast
def test_alternate_timestamp_names():
    mapper = SemanticMapper()
    sample_dates = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])

    assert (
        mapper.map_column("event_date", sample_dates)["canonical_field"]
        == CanonicalField.TIMESTAMP.value
    )
    assert (
        mapper.map_column("payment_timestamp", sample_dates)["canonical_field"]
        == CanonicalField.TIMESTAMP.value
    )
    assert (
        mapper.map_column("processed_on", sample_dates)["canonical_field"]
        == CanonicalField.TIMESTAMP.value
    )
    assert (
        mapper.map_column("txn_datetime", sample_dates)["canonical_field"]
        == CanonicalField.TIMESTAMP.value
    )
    assert (
        mapper.map_column("settlement_date", sample_dates)["canonical_field"]
        == CanonicalField.SETTLEMENT_DATE.value
    )


@pytest.mark.fast
def test_alternate_outcome_names():
    mapper = SemanticMapper()
    sample_binary = pd.Series([1, 0, 1, 0, 1])
    sample_str = pd.Series(["success", "failure", "success", "failure"])

    assert (
        mapper.map_column("payment_result", sample_str)["canonical_field"]
        == CanonicalField.OUTCOME.value
    )
    assert (
        mapper.map_column("failure_indicator", sample_binary)["canonical_field"]
        == CanonicalField.OUTCOME.value
    )
    assert (
        mapper.map_column("collection_status", sample_str)["canonical_field"]
        == CanonicalField.OUTCOME.value
    )
    assert (
        mapper.map_column("recovery_flag", sample_binary)["canonical_field"]
        == CanonicalField.OUTCOME.value
    )
    assert (
        mapper.map_column("settlement_result", sample_str)["canonical_field"]
        == CanonicalField.OUTCOME.value
    )


# ---------------------------------------------------------------------------
# 5. Ambiguous Mappings
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_ambiguous_mappings():
    mapper = SemanticMapper()
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "transaction_amount": [100.0, 200.0, 300.0],
            "invoice_total": [100.0, 200.0, 300.0],  # Conflicting AMOUNT
            "event_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "payment_result": ["success", "failed", "success"],
        }
    )
    mappings = mapper.map_schema(df)
    sig_map = {m["original_column"]: m for m in mappings}

    assert sig_map["transaction_amount"]["canonical_field"] == CanonicalField.UNKNOWN.value
    assert sig_map["invoice_total"]["canonical_field"] == CanonicalField.UNKNOWN.value
    assert "AMBIGUOUS" in sig_map["transaction_amount"]["reason"]
    assert "AMBIGUOUS" in sig_map["invoice_total"]["reason"]


# ---------------------------------------------------------------------------
# 6. Amount vs Balance Distinction
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_amount_vs_balance_distinction():
    mapper = SemanticMapper()
    sample_nums = pd.Series([100.0, 250.0, 500.0, 1000.0])

    # Amounts
    for col in [
        "transaction_value",
        "debit_value",
        "payment_amt",
        "invoice_total",
        "monetary_value",
    ]:
        res = mapper.map_column(col, sample_nums)
        assert res["canonical_field"] == CanonicalField.AMOUNT.value, f"Failed for {col}"

    # Balances
    for col in ["outstanding_balance", "current_balance", "avail_bal"]:
        res = mapper.map_column(col, sample_nums)
        assert res["canonical_field"] == CanonicalField.BALANCE.value, f"Failed for {col}"

    # Leakage fields must not become valid amounts
    for col in ["post_recovery_balance", "actual_recovered", "settled_amount"]:
        leak = DatasetValidator.detect_leakage(col, "AMOUNT")
        assert leak["status"] == "WARNING", f"Failed for {col}"


# ---------------------------------------------------------------------------
# 7. Categorical Non-Targets and Numeric IDs
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_categorical_non_target_features():
    mapper = SemanticMapper()

    # Region/country/device/channel must NOT become TARGET
    s_region = pd.Series(["US", "EU", "APAC", "US", "EU"])
    assert mapper.map_column("region", s_region)["canonical_field"] == CanonicalField.UNKNOWN.value

    s_country = pd.Series(["USA", "CAN", "GBR", "USA"])
    assert (
        mapper.map_column("country", s_country)["canonical_field"] == CanonicalField.UNKNOWN.value
    )

    s_device = pd.Series(["mobile", "desktop", "tablet", "mobile"])
    assert (
        mapper.map_column("device_type", s_device)["canonical_field"]
        == CanonicalField.UNKNOWN.value
    )

    s_channel = pd.Series(["web", "ios", "android", "web"])
    assert (
        mapper.map_column("channel", s_channel)["canonical_field"] == CanonicalField.UNKNOWN.value
    )

    # Payment method channel
    s_pm = pd.Series(["ach", "card", "wire", "ach"])
    assert (
        mapper.map_column("payment_method", s_pm)["canonical_field"]
        == CanonicalField.PAYMENT_METHOD.value
    )


@pytest.mark.fast
def test_numeric_ids_and_timestamps():
    mapper = SemanticMapper()

    # Numeric count cannot be AMOUNT
    s_count = pd.Series([1, 2, 1, 3, 5])
    assert (
        mapper.map_column("login_count", s_count)["canonical_field"] == CanonicalField.UNKNOWN.value
    )

    # Numeric ID is an entity ID, not AMOUNT
    s_id_num = pd.Series([100234, 100235, 100236, 100237])
    assert mapper.map_column("customer_no", s_id_num)["canonical_field"] in [
        CanonicalField.CUSTOMER_ID.value,
        CanonicalField.ACCOUNT_ID.value,
    ]

    # Timestamp cannot become ID
    s_ts = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    assert mapper.map_column("timestamp", s_ts)["canonical_field"] == CanonicalField.TIMESTAMP.value


# ---------------------------------------------------------------------------
# 8. Leakage Detection
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_leakage_detection():
    leakage_cols = [
        "actual_recovered",
        "recovered_amount",
        "actual_recovered_amount",
        "recovery_date",
        "final_status",
        "settled_amount",
        "post_recovery_balance",
        "final_settlement_status",
    ]
    for col in leakage_cols:
        leak = DatasetValidator.detect_leakage(col, "AMOUNT")
        assert leak["status"] == "WARNING", f"Failed for {col}"
        assert "POST_OUTCOME" in leak["reason"], f"Failed for {col}"


# ---------------------------------------------------------------------------
# 9-11. Malformed Data, Timestamps, and Target Handling in Case Gen
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_malformed_data_and_counters(tmp_path):
    df = pd.DataFrame(
        {
            "account_no": ["ACC1", "ACC2", "ACC3", "ACC4", "ACC5", "ACC6", "ACC7"],
            "payment_amt": ["100.0", "invalid_amt", "-50.0", "200.0", "300.0", "400.0", "500.0"],
            "event_date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "bad_date",
                "2023-01-05",
                "2023-01-06",
                "2023-01-07",
            ],
            "payment_result": [
                "failed",
                "failed",
                "failed",
                "failed",
                "ambiguous_status",
                "success",
                "failed",
            ],
        }
    )

    extra_rows = []
    for i in range(50):
        extra_rows.append(
            {
                "account_no": f"EXTRA_{i}",
                "payment_amt": 50.0 + i,
                "event_date": "2023-01-10",
                "payment_result": "failed" if i % 2 == 0 else "success",
            }
        )
    df_full = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    csv_path = str(tmp_path / "robustness_test.csv")
    df_full.to_csv(csv_path, index=False)

    fname = f"robust_{uuid.uuid4().hex[:8]}.csv"
    with open(csv_path, "rb") as f:
        res = client.post("/datasets/upload", files={"file": (fname, f, "text/csv")})
    ds_id = res.json()["dataset_id"]

    client.post(f"/datasets/{ds_id}/analyze")

    ds = client.get(f"/datasets/{ds_id}").json()
    mappings = []
    for sig in ds["recoverchain_signals"]:
        cf = sig["canonical_field"]
        if sig["original_column"] == "payment_amt":
            cf = CanonicalField.AMOUNT.value
        mappings.append(
            {"original_column": sig["original_column"], "canonical_field": cf, "action": "confirm"}
        )
    client.post(f"/datasets/{ds_id}/mapping", json={"mappings": mappings})

    client.post(f"/datasets/{ds_id}/ml-readiness")
    client.post(f"/datasets/{ds_id}/train")

    gen_res = client.post(f"/datasets/{ds_id}/generate-cases", json={"max_cases": 7})
    assert gen_res.status_code == 200, gen_res.json()
    data = gen_res.json()

    assert data["cases_generated"] == 2
    assert "counters" in data
    counters = data["counters"]
    assert counters["rows_seen"] == 7
    assert counters["rows_accepted"] == 2
    assert counters["rows_skipped"] == 5
    assert counters["invalid_amount"] >= 2
    assert counters["invalid_timestamp"] >= 1
    assert counters["ambiguous_target"] >= 1


# ---------------------------------------------------------------------------
# 12. Settlement Date Cannot Substitute for Event Timestamp
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_settlement_date_cannot_substitute_for_event_timestamp(tmp_path):
    rows = []
    for i in range(50):
        rows.append(
            {
                "account_no": f"ACC_{i}",
                "amount": 100.0 + i,
                "settlement_date": "2023-08-01",
                "outcome": 1 if i % 2 == 0 else 0,
            }
        )
    df = pd.DataFrame(rows)
    csv_file = str(tmp_path / "settle_only.csv")
    df.to_csv(csv_file, index=False)

    fname = f"settle_{uuid.uuid4().hex[:8]}.csv"
    with open(csv_file, "rb") as f:
        res = client.post("/datasets/upload", files={"file": (fname, f, "text/csv")})
    ds_id = res.json()["dataset_id"]

    client.post(f"/datasets/{ds_id}/analyze")

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
    # Mapping confirmation correctly detects missing core timestamp and blocks progression
    map_res = client.post(f"/datasets/{ds_id}/mapping", json={"mappings": mappings})
    assert map_res.status_code == 400
    assert (
        "Unsafe mappings" in map_res.json()["detail"]
        or "timestamp" in map_res.json()["detail"].lower()
    )


# ---------------------------------------------------------------------------
# 13. ML Leakage Isolation During Generate Cases
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_ml_leakage_isolated_during_generate_cases(tmp_path):
    rows = []
    for i in range(50):
        rows.append(
            {
                "account_no": f"ACC_{i}",
                "amount": 100.0 + i,
                "timestamp": "2023-01-01",
                "outcome": 1 if i % 2 == 0 else 0,
                "actual_recovered_amount": 100.0 if i % 2 == 0 else 0.0,
                "recovery_date": "2023-01-15",
                "final_status": "recovered" if i % 2 == 0 else "unrecovered",
            }
        )
    df = pd.DataFrame(rows)
    csv_file = str(tmp_path / "leakage_dataset.csv")
    df.to_csv(csv_file, index=False)

    fname = f"leak_{uuid.uuid4().hex[:8]}.csv"
    with open(csv_file, "rb") as f:
        res = client.post("/datasets/upload", files={"file": (fname, f, "text/csv")})
    ds_id = res.json()["dataset_id"]

    client.post(f"/datasets/{ds_id}/analyze")
    ds = client.get(f"/datasets/{ds_id}").json()

    leakage = ds.get("leakage_detection", [])
    assert any("actual_recovered_amount" in l["column"] for l in leakage)
    assert any("recovery_date" in l["column"] for l in leakage)

    # Confirm mapping (leaked columns confirmed as unused/unknown)
    mappings = []
    for sig in ds["recoverchain_signals"]:
        cf = sig["canonical_field"]
        if (
            "actual_recovered" in sig["original_column"]
            or "recovery_date" in sig["original_column"]
        ):
            cf = CanonicalField.UNKNOWN.value
        mappings.append(
            {
                "original_column": sig["original_column"],
                "canonical_field": cf,
                "action": "confirm" if cf != CanonicalField.UNKNOWN.value else "unused",
            }
        )
    # Phase 4: nothing in these columns identifies the revenue domain, and an undeclared
    # domain is no longer assumed to be a failed payment - so it is declared explicitly.
    map_res = client.post(
        f"/datasets/{ds_id}/mapping",
        json={"mappings": mappings, "declared_domain": "FAILED_PAYMENT"},
    )
    assert map_res.status_code == 200, map_res.json()

    # ML readiness must exclude leaked columns
    ml_spec = client.post(f"/datasets/{ds_id}/ml-readiness").json()
    assert "actual_recovered_amount" in ml_spec.get("excluded_columns", [])
    assert "recovery_date" in ml_spec.get("excluded_columns", [])

    # Train
    train_res = client.post(f"/datasets/{ds_id}/train")
    assert train_res.status_code == 200

    # Generate cases
    gen_res = client.post(f"/datasets/{ds_id}/generate-cases", json={"max_cases": 5})
    assert gen_res.status_code == 200


# ---------------------------------------------------------------------------
# 14. Large CSV Bounded Ingestion
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_csv_bounded_processing(tmp_path):
    rows = []
    for i in range(10000):
        rows.append(
            {
                "client_key": f"C_{i}",
                "transaction_value": 10.0 + (i % 100),
                "txn_datetime": "2023-05-01",
                "payment_result": 1 if i % 3 == 0 else 0,
            }
        )
    df_large = pd.DataFrame(rows)
    csv_file = str(tmp_path / "large_dataset.csv")
    df_large.to_csv(csv_file, index=False)

    profile = DatasetProfiler.profile_file(csv_file, "text/csv")
    assert profile["row_count"] == 10000
    assert profile["column_count"] == 4
    assert profile["validation"]["classification"] == DatasetClassification.ML_TRAINING_READY.value


# ---------------------------------------------------------------------------
# 15. Parquet Metadata / Row-Group Ingestion
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_parquet_metadata_processing(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(
        {
            "account_no": [f"ACC_{i}" for i in range(1000)],
            "monetary_value": [25.0 * (i % 10) for i in range(1000)],
            "event_date": ["2023-06-01" for _ in range(1000)],
            "failure_indicator": [1 if i % 4 == 0 else 0 for i in range(1000)],
        }
    )
    table = pa.Table.from_pandas(df)
    parquet_path = str(tmp_path / "test_data.parquet")
    pq.write_table(table, parquet_path, row_group_size=200)

    profile = DatasetProfiler.profile_file(parquet_path, "application/parquet")
    assert profile["row_count"] == 1000
    assert profile["column_count"] == 4
    assert profile["validation"]["classification"] == DatasetClassification.ML_TRAINING_READY.value


# ---------------------------------------------------------------------------
# 16. XLSX Ingestion
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_xlsx_processing(tmp_path):
    df = pd.DataFrame(
        {
            "user_identifier": [f"USR_{i}" for i in range(50)],
            "debit_value": [100.0 + i for i in range(50)],
            "processed_on": ["2023-07-01" for _ in range(50)],
            "settlement_result": ["failure" if i % 2 == 0 else "success" for i in range(50)],
        }
    )
    xlsx_path = str(tmp_path / "test_data.xlsx")
    df.to_excel(xlsx_path, index=False)

    profile = DatasetProfiler.profile_file(
        xlsx_path, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert profile["row_count"] == 50
    assert profile["column_count"] == 4
    assert profile["validation"]["classification"] == DatasetClassification.ML_TRAINING_READY.value


# ---------------------------------------------------------------------------
# 17. Preview Maximum Bounded Endpoint
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_preview_endpoint_and_bounds(tmp_path):
    df = pd.DataFrame(
        {
            "account_no": [f"A_{i}" for i in range(200)],
            "payment_amt": [10.0 + i for i in range(200)],
            "txn_datetime": ["2023-01-01" for _ in range(200)],
        }
    )
    csv_file = str(tmp_path / "preview_test.csv")
    df.to_csv(csv_file, index=False)

    fname = f"prev_{uuid.uuid4().hex[:8]}.csv"
    with open(csv_file, "rb") as f:
        res = client.post("/datasets/upload", files={"file": (fname, f, "text/csv")})
    ds_id = res.json()["dataset_id"]

    prev_default = client.get(f"/datasets/{ds_id}/preview")
    assert prev_default.status_code == 200
    data_default = prev_default.json()
    assert len(data_default["rows"]) == 25
    assert len(data_default["columns"]) == 3
    assert data_default["columns"][0]["original_name"] == "account_no"

    prev_50 = client.get(f"/datasets/{ds_id}/preview?limit=50")
    assert prev_50.status_code == 200
    assert len(prev_50.json()["rows"]) == 50

    prev_cap = client.get(f"/datasets/{ds_id}/preview?limit=500")
    assert prev_cap.status_code == 200
    assert len(prev_cap.json()["rows"]) == 100


# ---------------------------------------------------------------------------
# 18. Minimum Information Contract
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_minimum_information_contract_explanations():
    res1 = DatasetValidator.classify_dataset(
        [
            {"canonical_field": CanonicalField.AMOUNT.value, "confidence": "HIGH"},
            {"canonical_field": CanonicalField.TIMESTAMP.value, "confidence": "HIGH"},
            {"canonical_field": CanonicalField.OUTCOME.value, "confidence": "HIGH"},
        ]
    )
    assert res1["classification"] == DatasetClassification.PARTIALLY_USABLE.value
    assert any("Entity/Account Identifier" in c for c in res1["diagnostic"]["missing_concepts"])

    res2 = DatasetValidator.classify_dataset(
        [
            {"canonical_field": CanonicalField.ACCOUNT_ID.value, "confidence": "HIGH"},
            {"canonical_field": CanonicalField.AMOUNT.value, "confidence": "HIGH"},
            {"canonical_field": CanonicalField.TIMESTAMP.value, "confidence": "HIGH"},
        ]
    )
    assert res2["classification"] == DatasetClassification.ANALYSIS_READY.value
    assert any("Outcome/Target Label" in c for c in res2["diagnostic"]["missing_concepts"])

    res3 = DatasetValidator.classify_dataset(
        [{"canonical_field": CanonicalField.UNKNOWN.value, "confidence": "LOW"}]
    )
    assert res3["classification"] == DatasetClassification.INSUFFICIENT.value


# ---------------------------------------------------------------------------
# 19. Dataset Isolation
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_dataset_isolation_strict(tmp_path):
    df_a = pd.DataFrame(
        {
            "account_no": [f"A_{i}" for i in range(50)],
            "amount": [10.0 + i for i in range(50)],
            "timestamp": ["2023-01-01" for _ in range(50)],
            "outcome": [1 if i % 2 == 0 else 0 for i in range(50)],
        }
    )
    path_a = str(tmp_path / "ds_a.csv")
    df_a.to_csv(path_a, index=False)

    df_b = pd.DataFrame(
        {
            "customer_id": [f"B_{i}" for i in range(50)],
            "price": [500.0 + i for i in range(50)],
            "created_at": ["2023-02-01" for _ in range(50)],
            "target": [1 if i % 3 == 0 else 0 for i in range(50)],
        }
    )
    path_b = str(tmp_path / "ds_b.csv")
    df_b.to_csv(path_b, index=False)

    fname_a = f"ds_a_{uuid.uuid4().hex[:8]}.csv"
    with open(path_a, "rb") as f:
        ds_a_id = client.post("/datasets/upload", files={"file": (fname_a, f, "text/csv")}).json()[
            "dataset_id"
        ]

    fname_b = f"ds_b_{uuid.uuid4().hex[:8]}.csv"
    with open(path_b, "rb") as f:
        ds_b_id = client.post("/datasets/upload", files={"file": (fname_b, f, "text/csv")}).json()[
            "dataset_id"
        ]

    client.post(f"/datasets/{ds_a_id}/analyze")
    ds_a_data = client.get(f"/datasets/{ds_a_id}").json()
    mappings_a = [
        {
            "original_column": sig["original_column"],
            "canonical_field": sig["canonical_field"],
            "action": "confirm",
        }
        for sig in ds_a_data["recoverchain_signals"]
    ]
    client.post(f"/datasets/{ds_a_id}/mapping", json={"mappings": mappings_a})
    client.post(f"/datasets/{ds_a_id}/ml-readiness")
    client.post(f"/datasets/{ds_a_id}/train")

    # Predictor for B should NOT find A's model
    pred_b = MLPaymentFailurePredictor(dataset_id=ds_b_id, registry_dir=model_registry_dir())
    assert pred_b.model is None

    # Predictor for non-existent dataset should be None
    pred_fake = MLPaymentFailurePredictor(
        dataset_id="non_existent_dataset_123", registry_dir=model_registry_dir()
    )
    assert pred_fake.model is None


# ---------------------------------------------------------------------------
# 20. Sandbox Execution Boundary
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_sandbox_execution_cannot_become_production():
    orchestrator = AgentOrchestrator()

    case = RecoveryCase(
        case_id="case_sandbox_test",
        customer_id="cust_123",
        risk_category=RiskCategory.FAILED_PAYMENT,
        amount_at_risk=Money(amount=150.0, currency="USD"),
        reference_id="tx_123",
    )
    case.policy_decision = PolicyDecision(
        decision_id="pol_123",
        status=PolicyDecisionStatus.PERMITTED,
        policy_version="v1.0",
        rules_evaluated=[],
        failed_rules=[],
        reason="Permitted in sandbox",
        timestamp=datetime.now(UTC),
    )
    from domain.models import ActionRecommendation, CandidateAction

    case.recommendation = ActionRecommendation(
        recommendation_id="rec_123",
        candidates=[
            CandidateAction(
                action_type=ActionType.RETRY_PAYMENT,
                estimated_probability=0.8,
                expected_recoverable_value=120.0,
                rationale="Test",
            )
        ],
        top_candidate=CandidateAction(
            action_type=ActionType.RETRY_PAYMENT,
            estimated_probability=0.8,
            expected_recoverable_value=120.0,
            rationale="Test",
        ),
        status="RECOMMENDED",
        rationale="Test",
        engine_version="v1",
    )

    record = orchestrator.execute(case, ActionType.RETRY_PAYMENT)
    assert record.status == ExecutionStatus.COMPLETED_SIMULATED
    assert record.adapter_used == "MockExecutionAdapter"
    assert record.result_metadata.get("adapter_status") == "COMPLETED_SIMULATED"
    assert record.result_metadata.get("metadata", {}).get("gateway") == "sandbox"


# ---------------------------------------------------------------------------
# 21. Path Traversal and Registry Escape Protection
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_path_traversal_and_registry_isolation():
    pred_traversal = MLPaymentFailurePredictor(
        dataset_id="../../legacy_billing_v3", registry_dir=model_registry_dir()
    )
    assert pred_traversal.model is None

    pred_traversal_win = MLPaymentFailurePredictor(
        dataset_id="..\\..\\some_model", registry_dir=model_registry_dir()
    )
    assert pred_traversal_win.model is None

    pred_slash = MLPaymentFailurePredictor(
        dataset_id="/etc/passwd", registry_dir=model_registry_dir()
    )
    assert pred_slash.model is None

    pred_missing = MLPaymentFailurePredictor(
        dataset_id="ds_does_not_exist_9999", registry_dir=model_registry_dir()
    )
    assert pred_missing.model is None
