"""Phase 3 data-foundation regression tests.

1. Evaluation isolation - an evaluation run cannot change any operational figure, and
   evaluation results stay identifiable in their own store.
2. The ingestion contract - the API and dataset files apply the same validation, and
   invalid records are rejected with reasons instead of being accepted or dropped.
3. Event identity - the same event arriving twice (same API id, same file re-uploaded,
   same dataset re-processed) is recognised; file rows no longer get a fabricated
   reference id that correlates unrelated obligations.
4. Provenance and run records - every event says where it came from, and every batch
   ingestion leaves a durable record of what happened to its rows.
"""

import uuid
import zipfile

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ----------------------------------------------------------------- helpers


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _snapshot() -> dict:
    """Every operational figure an evaluation run must not be able to move."""
    m = client.get("/dashboard/metrics").json()
    s = client.get("/system/summary").json()
    return {
        "at_risk": m["total_revenue_at_risk"],
        "recovered": m["verified_recovery"],
        "opportunities": m["recovery_opportunities"],
        "active": m["active_cases"],
        "pending": m["pending_human_review"],
        "breakdown": m["currency_breakdown"],
        "cases": s["cases_count"],
        "events": s["revenue_events_count"],
        "audit": s["audit_records_count"],
        "outcomes": s["outcomes_count"],
        "executions": client.get("/system/executions?limit=1").json()["total_count"],
        "case_list": len(client.get("/cases?limit=500").json()),
    }


def _event(tag: str, **override) -> dict:
    ev = {
        "customer_id": f"p3_cust_{tag}",
        "risk_category": "FAILED_PAYMENT",
        "external_system": f"p3_sys_{tag}",
        "external_event_id": f"p3_evt_{tag}",
        "amount": 250,
        "currency": "INR",
        "raw_payload": {"failure_code": "network_error"},
    }
    ev.update(override)
    return ev


HEADER = "customer_id,amount,invoice_date,status,reason,currency"


def _csv(rows: list[str]) -> bytes:
    return (HEADER + "\n" + "\n".join(rows) + "\n").encode()


def _padding(tag: str, n_failed: int = 5, n_paid: int = 3) -> list[str]:
    rows = [
        f"pad_f{i}_{tag},{120 + i}.00,2026-07-{i + 1:02d},failed,insufficient_funds,INR"
        for i in range(n_failed)
    ]
    rows += [f"pad_p{i}_{tag},{60 + i}.00,2026-07-{i + 1:02d},paid,,INR" for i in range(n_paid)]
    return rows


def _ready_dataset(content: bytes, name: str) -> tuple[str, dict]:
    """Upload, profile, confirm the proposed mapping and pass ML readiness - the same
    calls the guided workflow makes before it can generate cases."""
    up = client.post("/datasets/upload", files={"file": (name, content, "text/csv")})
    assert up.status_code == 200, up.text
    ds = up.json()["dataset_id"]
    assert client.post(f"/datasets/{ds}/analyze").status_code == 200
    detail = client.get(f"/datasets/{ds}").json()
    mappings = [
        {
            "original_column": s["original_column"],
            "canonical_field": s["canonical_field"],
            "action": "confirm",
        }
        for s in detail["recoverchain_signals"]
    ]
    r = client.post(f"/datasets/{ds}/mapping", json={"mappings": mappings})
    assert r.status_code == 200, r.text
    r = client.post(f"/datasets/{ds}/ml-readiness")
    assert r.status_code == 200, r.text
    return ds, up.json()


def _generate(ds: str, max_cases: int = 100) -> dict:
    r = client.post(f"/datasets/{ds}/generate-cases", json={"max_cases": max_cases})
    assert r.status_code == 200, r.text
    return r.json()


# ----------------------------------------------------------------- 1. evaluation isolation


def test_A_evaluation_run_cannot_change_operational_figures():
    """The live defect: one POST /evaluation/run added 12 scenario cases to the live
    table and moved revenue-at-risk, recovered money and active-case counts."""
    client.post("/events", json=_event(_tag()))  # a real operational case exists first
    before = _snapshot()

    r = client.post("/evaluation/run")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data_class"] == "EVALUATION"
    assert body["total_scenarios"] == 12
    assert body["total_amount_at_risk"] > 0  # the scenarios really ran on money figures

    assert _snapshot() == before
    events = client.get("/system/events?limit=100").json()["items"]
    assert not [e for e in events if e["external_system"] == "system_test"]


def test_B_evaluation_runs_are_repeatable_and_remain_identifiable():
    # A second run used to fail with HTTP 500 on the fixed scenario event ids.
    ids = []
    for _ in range(2):
        r = client.post("/evaluation/run")
        assert r.status_code == 200, r.text
        ids.append(r.json()["evaluation_run_id"])
    assert ids[0] != ids[1]

    listed = {x["run_id"]: x for x in client.get("/evaluation/runs").json()}
    for run_id in ids:
        assert listed[run_id]["status"] == "COMPLETED"
        assert listed[run_id]["data_class"] == "EVALUATION"
        assert listed[run_id]["isolation"] == "ISOLATED_EPHEMERAL_STORE"
        assert listed[run_id]["metrics"] is None  # list view carries summaries only

    detail = client.get(f"/evaluation/runs/{ids[1]}").json()
    assert detail["metrics"]["total_scenarios"] == 12
    assert client.get("/evaluation/runs/eval_does_not_exist").status_code == 404


def test_B2_evaluation_runner_refuses_the_operational_database():
    from evaluation.runner import EvaluationRunner, OperationalStoreError
    from infrastructure.db import SessionLocal

    session = SessionLocal()
    try:
        with pytest.raises(OperationalStoreError):
            EvaluationRunner(session)
    finally:
        session.close()


def test_L_evaluation_aggregation_is_internally_consistent_and_stored():
    body = client.post("/evaluation/run").json()
    results = body["scenario_results"]
    assert body["total_amount_at_risk"] == pytest.approx(sum(r["amount_at_risk"] for r in results))
    assert body["total_simulated_recovered"] == pytest.approx(
        sum(r["actual_recovered"] for r in results)
    )
    stored = client.get(f"/evaluation/runs/{body['evaluation_run_id']}").json()["metrics"]
    assert stored["total_amount_at_risk"] == body["total_amount_at_risk"]
    assert stored["passed_scenarios"] == body["passed_scenarios"]


# ----------------------------------------------------------------- 2. API ingestion contract


@pytest.mark.parametrize(
    "override",
    [
        {"amount": -500},
        {"amount": 0},
        {"amount": 1e15},
        {"amount": "abc"},
        {"customer_id": ""},
        {"customer_id": "   "},
        {"external_event_id": ""},
        {"external_system": ""},
        {"currency": "XYZ"},
    ],
)
def test_E_api_rejects_invalid_events_without_storing_them(override):
    before = _snapshot()["events"]
    r = client.post("/events", json=_event(_tag(), **override))
    assert r.status_code == 422, r.text
    assert _snapshot()["events"] == before


def test_F_api_rejects_a_missing_required_field():
    ev = _event(_tag())
    del ev["external_event_id"]
    r = client.post("/events", json=ev)
    assert r.status_code == 422
    assert any("external_event_id" in d["loc"] for d in r.json()["detail"])


def test_I_api_event_carries_provenance_and_a_normalised_currency():
    tag = _tag()
    r = client.post("/events", json=_event(tag, currency="usd"))
    assert r.status_code == 200, r.text
    snap = client.get(f"/system/cases/{r.json()['case_id']}").json()
    assert snap["currency"] == "USD"
    ev = snap["data_events"][0]
    assert ev["source_type"] == "API"
    assert ev["source_ref"] == f"p3_sys_{tag}"
    assert ev["ingested_at"]


def test_C_duplicate_external_event_is_ignored_but_identity_is_per_system():
    tag = _tag()
    first = client.post("/events", json=_event(tag)).json()
    assert first["status"] == "success"
    events_before = _snapshot()["events"]

    again = client.post("/events", json=_event(tag)).json()
    assert again["status"] == "ignored"
    assert _snapshot()["events"] == events_before

    # The same event id from a DIFFERENT external system is a different event.
    other = client.post("/events", json=_event(tag, external_system=f"p3_other_{tag}")).json()
    assert other["status"] == "success"
    assert other["case_id"] != first["case_id"]


def test_M_batch_rejects_invalid_items_individually_and_records_the_run():
    tag = _tag()
    good = _event(tag)
    bad_amount = _event(f"{tag}a", amount=-1)
    bad_currency = _event(f"{tag}b", currency="XYZ")
    missing = _event(f"{tag}c")
    del missing["external_event_id"]

    r = client.post(
        "/events/batch",
        json={"auto_advance": False, "events": [good, bad_amount, bad_currency, missing, good]},
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert (b["submitted"], b["ingested"], b["duplicates"], b["rejected"], b["failed"]) == (
        5,
        1,
        1,
        3,
        0,
    )
    assert [x["status"] for x in b["results"]] == [
        "success",
        "rejected",
        "rejected",
        "rejected",
        "ignored",
    ]
    assert "non_positive_amount" in b["results"][1]["reasons"]
    assert "invalid_currency" in b["results"][2]["reasons"]
    assert "missing_field" in b["results"][3]["reasons"]

    run = client.get(f"/ingestion-runs/{b['ingestion_run_id']}").json()
    assert run["source_type"] == "API_BATCH"
    assert run["status"] == "COMPLETED"
    assert run["rows_received"] == 5
    assert run["rows_received"] == (
        run["rows_accepted"] + run["rows_rejected"] + run["rows_duplicate"]
    )


# ----------------------------------------------------------------- 3/4. file ingestion


def test_GH_file_rows_are_validated_explained_and_the_run_is_persisted():
    tag = _tag()
    rows = [
        f"ok1_{tag},240.00,2026-08-20,failed,insufficient_funds,INR",
        f"ok2_{tag},980.00,2026-08-21,failed,network_error,INR",
        f"ok3_{tag},150.00,2026-08-22,failed,expired_card,INR",
        f"ok4_{tag},320.00,2026-08-24,failed,network_error,usd",
        f"ok5_{tag},760.00,2026-08-25,failed,insufficient_funds,INR",
        f"ok6_{tag},410.00,2026-08-26,failed,network_error,INR",
        f"paid1_{tag},180.00,2026-08-26,paid,,INR",
        f"paid2_{tag},540.00,2026-08-28,paid,,INR",
        f"paid3_{tag},90.00,2026-08-28,paid,,INR",
        f"bad_amt_{tag},abc,2026-08-27,failed,network_error,INR",  # data row 10
        f"neg_amt_{tag},-5,2026-08-27,failed,network_error,INR",
        f"zero_amt_{tag},0,2026-08-27,failed,network_error,INR",
        f"bad_date_{tag},88.50,not-a-date,failed,insufficient_funds,INR",
        ",300.00,2026-08-29,failed,network_error,INR",
        f"bad_ccy_{tag},72.00,2026-08-30,failed,expired_card,XYZ",
    ]
    name = f"quality_{tag}.csv"
    ds, _ = _ready_dataset(_csv(rows), name)
    b = _generate(ds)
    c = b["counters"]

    assert c["rows_seen"] == 15
    assert c["rows_accepted"] == 6 and c["cases_created"] == 6
    assert c["not_at_risk"] == 3
    assert c["invalid_amount"] == 3  # "abc", -5 and 0
    assert c["invalid_timestamp"] == 1
    assert c["invalid_entity"] == 1
    assert c["invalid_currency"] == 1
    assert c["rows_skipped"] == 9
    assert c["duplicate_event"] == 0
    assert b["truncated"] is False

    reasons = {s["reason"] for s in b["rejected_samples"]}
    assert {
        "invalid_amount",
        "non_positive_amount",
        "invalid_timestamp",
        "missing_identifier",
        "invalid_currency",
    } <= reasons
    assert next(s for s in b["rejected_samples"] if s["reason"] == "invalid_amount")["row"] == 10

    run = next(
        x
        for x in client.get(f"/datasets/{ds}/ingestion-runs").json()
        if x["run_id"] == b["ingestion_run_id"]
    )
    assert run["status"] == "COMPLETED"
    assert run["source_type"] == "FILE_UPLOAD" and run["source_name"] == name
    assert run["rows_received"] == 15
    assert run["rows_received"] == (
        run["rows_accepted"]
        + run["rows_rejected"]
        + run["rows_duplicate"]
        + run["rows_not_at_risk"]
    )
    assert run["rejection_counts"]["non_positive_amount"] == 2

    currencies = set()
    for case_id in b["case_ids"]:
        snap = client.get(f"/system/cases/{case_id}").json()
        currencies.add(snap["currency"])
        ev = snap["data_events"][0]
        assert ev["source_type"] == "FILE_UPLOAD"
        assert ev["source_ref"] == ds
        assert ev["ingestion_run_id"] == b["ingestion_run_id"]
        assert ev["external_system"] == "file_upload"
        # Content identity. Phase 4 moved file rows to identity v2 ("fp2:"), which leaves
        # the re-declarable domain out; stored v1 "fp:" ids are still recognised.
        assert ev["external_event_id"].startswith("fp2:")
        assert ev["reference_id"] is None  # no transaction column: nothing fabricated
    assert currencies == {"INR", "USD"}


def test_N_reprocessing_the_same_dataset_creates_nothing_new():
    tag = _tag()
    rows = [
        f"rep{i}_{tag},{200 + i}.00,2026-08-{i + 1:02d},failed,network_error,INR" for i in range(6)
    ]
    ds, _ = _ready_dataset(_csv(rows + _padding(tag, n_failed=0)), f"reprocess_{tag}.csv")
    first = _generate(ds)
    assert first["counters"]["cases_created"] == 6
    before = _snapshot()

    second = _generate(ds)
    c = second["counters"]
    assert c["cases_created"] == 0 and c["rows_accepted"] == 0
    assert c["duplicate_event"] == 6
    assert sorted(second["case_ids"]) == sorted(first["case_ids"])
    after = _snapshot()
    for key in ("cases", "events", "at_risk", "recovered", "breakdown"):
        assert after[key] == before[key], key


def test_D_the_same_file_uploaded_twice_reuses_existing_cases():
    tag = _tag()
    rows = [
        f"twice{i}_{tag},{300 + i}.00,2026-08-{i + 1:02d},failed,insufficient_funds,INR"
        for i in range(5)
    ]
    content = _csv(rows + _padding(tag, n_failed=0))
    name = f"twice_{tag}.csv"
    ds1, up1 = _ready_dataset(content, name)
    first = _generate(ds1)
    assert up1["duplicate_of"] is None
    before = _snapshot()

    ds2, up2 = _ready_dataset(content, name)
    assert up2["duplicate_of"] == ds1
    assert up2["content_sha256"] == up1["content_sha256"]
    second = _generate(ds2)
    assert second["counters"]["cases_created"] == 0
    assert second["counters"]["duplicate_event"] == first["counters"]["rows_accepted"] == 5
    assert sorted(second["case_ids"]) == sorted(first["case_ids"])
    after = _snapshot()
    for key in ("cases", "events", "at_risk", "recovered"):
        assert after[key] == before[key], key


def test_P_rows_without_a_transaction_id_never_correlate_across_files():
    """File A's row escalates (amount above the 5000 automated limit), so its case stays
    open. File B has a row for the same customer and category. With the old positional
    "tx_<row>" reference, B's row 0 attached to A's open case and overwrote its amount."""
    tag = _tag()
    customer = f"shared_{tag}"
    file_a = [f"{customer},9000.00,2026-08-01,failed,network_error,INR"] + _padding(f"a{tag}")
    file_b = [f"{customer},777.00,2026-08-02,failed,network_error,INR"] + _padding(f"b{tag}")
    ds_a, _ = _ready_dataset(_csv(file_a), f"corr_a_{tag}.csv")
    ds_b, _ = _ready_dataset(_csv(file_b), f"corr_b_{tag}.csv")

    def case_for(result):
        for case_id in result["case_ids"]:
            snap = client.get(f"/system/cases/{case_id}").json()
            if snap["customer_id"] == customer:
                return snap
        raise AssertionError("customer case not found")

    case_a = case_for(_generate(ds_a))
    case_b = case_for(_generate(ds_b))
    assert case_a["case_id"] != case_b["case_id"]
    assert client.get(f"/system/cases/{case_a['case_id']}").json()["amount_at_risk"] == 9000.0
    assert len(case_a["data_events"]) == 1


def test_O_truncation_is_reported_and_a_larger_run_resumes_without_duplicates():
    tag = _tag()
    failed = [
        f"t{i}_{tag},{100 + i}.00,2026-08-{(i % 27) + 1:02d},failed,network_error,INR"
        for i in range(24)
    ]
    paid = [f"tp{i}_{tag},{50 + i}.00,2026-08-01,paid,,INR" for i in range(6)]
    ds, _ = _ready_dataset(_csv(failed + paid), f"trunc_{tag}.csv")

    first = _generate(ds, max_cases=10)
    assert first["truncated"] is True
    c1 = first["counters"]
    assert (c1["rows_seen"], c1["rows_in_file"], c1["rows_not_processed"]) == (10, 30, 20)
    assert c1["cases_created"] == 10
    run = client.get(f"/ingestion-runs/{first['ingestion_run_id']}").json()
    assert run["rows_not_processed"] == 20

    second = _generate(ds, max_cases=30)
    c2 = second["counters"]
    assert second["truncated"] is False
    assert c2["duplicate_event"] == 10  # the rows the first run already took in
    assert c2["cases_created"] == 14
    assert c2["not_at_risk"] == 6


def test_K_dashboard_reports_per_currency_totals_and_flags_mixing():
    before = client.get("/dashboard/metrics").json()
    tag = _tag()
    assert (
        client.post("/events", json=_event(f"{tag}i", amount=1000, currency="INR")).status_code
        == 200
    )
    assert (
        client.post("/events", json=_event(f"{tag}u", amount=50, currency="USD")).status_code == 200
    )
    after = client.get("/dashboard/metrics").json()

    def value(metrics, ccy):
        return float(metrics["currency_breakdown"].get(ccy, {}).get("total_revenue_at_risk", 0))

    assert value(after, "INR") - value(before, "INR") == pytest.approx(1000)
    assert value(after, "USD") - value(before, "USD") == pytest.approx(50)
    assert after["mixed_currency"] is True
    total = sum(float(b["total_revenue_at_risk"]) for b in after["currency_breakdown"].values())
    assert float(after["total_revenue_at_risk"]) == pytest.approx(total)


def test_Q_zip_holding_one_csv_is_profiled_and_a_multi_file_zip_fails_clearly(tmp_path):
    from application.dataset_intelligence import DatasetProfiler

    one = tmp_path / "one.zip"
    with zipfile.ZipFile(one, "w") as z:
        z.writestr(
            "a.csv",
            "customer_id,amount,invoice_date,status\n"
            + "".join(f"c{i},10,2026-08-01,failed\n" for i in range(7)),
        )
    profile = DatasetProfiler.profile_file(str(one), "zip")
    assert profile["row_count"] == 7
    assert profile["column_count"] == 4

    two = tmp_path / "two.zip"
    with zipfile.ZipFile(two, "w") as z:
        z.writestr("a.csv", "x\n1\n")
        z.writestr("b.csv", "x\n2\n")
    with pytest.raises(ValueError, match="Failed to read dataset"):
        DatasetProfiler.profile_file(str(two), "zip")


def test_ingestion_contract_units():
    from decimal import Decimal

    from application.ingestion import (
        ValidationFailure,
        event_fingerprint,
        normalize_currency,
        parse_amount,
    )

    assert parse_amount("12.50") == Decimal("12.50")
    for bad in ("1,200", "$45.10", "nan", float("inf"), True, None, ""):
        with pytest.raises(ValidationFailure):
            parse_amount(bad)
    assert normalize_currency(" usd ") == "USD"
    assert normalize_currency(None, default="INR") == "INR"

    from datetime import UTC, datetime

    kwargs = dict(
        customer_id="c1",
        risk_category="FAILED_PAYMENT",
        amount=Decimal("10"),
        currency="INR",
        timestamp=datetime(2026, 8, 1, tzinfo=UTC),
        reference_id=None,
        failure_code="NSF",
    )
    same = event_fingerprint(**{**kwargs, "amount": Decimal("10.0000"), "failure_code": "nsf"})
    assert event_fingerprint(**kwargs) == same  # normalised amount and failure code
    assert event_fingerprint(**kwargs) != event_fingerprint(**{**kwargs, "amount": Decimal("11")})


def test_dataset_detail_endpoint_serves_an_analyzed_dataset():
    """Pre-existing defect found during Phase 3 runtime verification: the profiler stores
    columns_profile as a LIST, but the response model declared a dict, so this endpoint
    returned HTTP 500 for every analyzed dataset. It also reports cases linked through
    Phase 3 provenance and the upload's content hash."""
    tag = _tag()
    rows = [
        f"det{i}_{tag},{150 + i}.00,2026-08-{i + 1:02d},failed,network_error,INR" for i in range(4)
    ]
    ds, up = _ready_dataset(_csv(rows + _padding(tag, n_failed=0)), f"detail_{tag}.csv")
    generated = _generate(ds)

    r = client.get(f"/system/datasets/{ds}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["profile_summary"], list) and body["profile_summary"]
    assert body["associated_cases_count"] == generated["counters"]["cases_created"] == 4
    assert body["content_sha256"] == up["content_sha256"]
