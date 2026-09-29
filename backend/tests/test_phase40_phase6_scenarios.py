"""Phase 6 regression tests — industrial dataset & end-to-end scenario validation.

These capture, as fast unit-level regressions, the scenarios verified live against
the running application during Phase 6 (see docs/history/phase6_final_report.md):

A  human approval does not bypass a suppression recorded after the original policy
   decision but before execution (revalidation at execute-time is what catches it)
B  PROMISE_CREATED requires its own amount; CANCELLATION never reads one
C  out-of-order settlement receipt still produces the correct final total
D  a dataset file may mix valid and invalid rows; each row is judged independently
E  re-uploading the same dataset file creates no new cases (row+file level dedup)
F  a duplicate outcome event (same external_event_id) cannot double-count recovery
"""

import io
import uuid

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
KEY = {"X-API-Key": "test-api-key"}


def _tag() -> str:
    return uuid.uuid4().hex[:8]


# ------------------------------------------------------------------ A


def test_A_suppression_recorded_after_policy_but_before_execution_still_blocks_it():
    tag = _tag()
    customer_id = f"p6_susp_{tag}"
    reference_id = f"p6_susp_ref_{tag}"

    # A high-value case -> ESCALATE (requires human approval), so it is NOT
    # auto-executed at creation time.
    r = client.post(
        "/events",
        json={
            "customer_id": customer_id,
            "risk_category": "FAILED_SUBSCRIPTION",
            "external_system": "p6_sys",
            "external_event_id": f"p6_evt_{tag}",
            "reference_id": reference_id,
            "amount": 6000,
            "currency": "INR",
            "raw_payload": {"failure_code": "insufficient_funds"},
        },
    )
    # (raw_payload already included above)
    assert r.status_code == 200, r.text
    case_id = r.json()["case_id"]
    r = client.post(f"/cases/{case_id}/advance", headers=KEY)
    assert r.status_code == 200, r.text

    before = client.get(f"/cases/{case_id}").json()
    assert before["policy_status"] == "ESCALATE"
    assert before["execution_status"] is None

    # Suppression is recorded NOW, before any approval or execution.
    r = client.post(
        "/contact-permissions/import",
        headers=KEY,
        json={
            "records": [
                {
                    "customer_id": customer_id,
                    "channel": "all",
                    "status": "SUPPRESSED",
                    "source": "compliance_hold",
                    "reason": "phase6 regression test",
                }
            ]
        },
    )
    assert r.status_code == 200 and r.json()["imported"] == 1

    # A human approves the threshold escalation...
    r = client.post(
        f"/cases/{case_id}/human-review",
        headers=KEY,
        json={"decision": "APPROVE", "note": "phase6 test"},
    )
    assert r.status_code == 200
    assert r.json()["decision"] == "PERMITTED"

    # ...but execution must revalidate policy against the now-suppressed consent and
    # refuse, even though a human already approved the threshold.
    r = client.post(f"/cases/{case_id}/execute", headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REJECTED"
    assert body["result_metadata"]["revalidated_status"] == "DENIED"
    assert "SUPPRESSED" in body["result_metadata"]["reason"]


# ------------------------------------------------------------------ B


def test_B_promise_created_requires_amount_cancellation_does_not():
    tag = _tag()
    customer_id = f"p6_promise_{tag}"
    reference_id = f"p6_promise_ref_{tag}"
    r = client.post(
        "/events",
        json={
            "customer_id": customer_id,
            "risk_category": "BROKEN_PROMISE",
            "external_system": "p6_sys",
            "external_event_id": f"p6_evt_{tag}",
            "reference_id": reference_id,
            "amount": 500,
            "currency": "INR",
            "raw_payload": {},
        },
    )
    assert r.status_code == 200

    # PROMISE_CREATED without an amount is rejected, not silently accepted as free.
    r = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p6_out",
            "external_event_id": f"p6_promise_noamt_{tag}",
            "domain": "BROKEN_PROMISE",
            "outcome_type": "PROMISE_CREATED",
            "customer_id": customer_id,
            "reference_id": reference_id,
            "occurred_at": "2031-06-01T00:00:00Z",
            "promised_date": "2031-06-15T00:00:00Z",
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"]["field"] == "amount"

    # A CANCELLATION with NO amount at all still fully cancels the outstanding balance.
    r = client.post(
        "/outcomes",
        headers=KEY,
        json={
            "external_system": "p6_out",
            "external_event_id": f"p6_cancel_{tag}",
            "domain": "BROKEN_PROMISE",
            "outcome_type": "PROMISE_BROKEN",
            "customer_id": customer_id,
            "reference_id": reference_id,
            "occurred_at": "2031-06-20T00:00:00Z",
        },
    )
    assert r.status_code == 200
    assert r.json()["effect"] == "NO_EFFECT"  # PROMISE_BROKEN never touches money


# ------------------------------------------------------------------ C


def test_C_out_of_order_settlement_receipt_still_totals_correctly():
    tag = _tag()
    customer_id = f"p6_ooo_{tag}"
    reference_id = f"p6_ooo_ref_{tag}"
    r = client.post(
        "/events",
        json={
            "customer_id": customer_id,
            "risk_category": "OVERDUE_INVOICE",
            "external_system": "p6_sys",
            "external_event_id": f"p6_evt_{tag}",
            "reference_id": reference_id,
            "amount": 1000,
            "currency": "INR",
            "raw_payload": {},
        },
    )
    case_id = r.json()["case_id"]

    def settle(amount, occurred_at, eid):
        return client.post(
            "/outcomes",
            headers=KEY,
            json={
                "external_system": "p6_out",
                "external_event_id": eid,
                "domain": "OVERDUE_INVOICE",
                "outcome_type": "INVOICE_SETTLED",
                "customer_id": customer_id,
                "reference_id": reference_id,
                "amount": amount,
                "currency": "INR",
                "occurred_at": occurred_at,
            },
        )

    # The event that occurred LATER in the real world is received FIRST.
    r1 = settle(400, "2026-09-05T00:00:00Z", f"p6_ooo_late_{tag}")
    assert r1.json()["effect"] == "APPLIED"
    # The event that occurred EARLIER is received SECOND.
    r2 = settle(600, "2026-09-01T00:00:00Z", f"p6_ooo_early_{tag}")
    assert r2.json()["effect"] == "APPLIED"

    snap = client.get(f"/cases/{case_id}/outcome", headers=KEY).json()
    assert snap["status"] == "FULLY_RECOVERED"
    assert snap["actual_amount_recovered"] == "1000.0000"
    assert snap["outstanding_amount"] == "0.0000"


# ------------------------------------------------------------------ D


def _ingest_mixed_validity_dataset(tag: str) -> str:
    df = pd.DataFrame(
        [
            {
                "customer_id": f"p6ds_{tag}_1",
                "amount": 100,
                "currency": "INR",
                "dt": "2026-09-01",
                "status": "failed",
                "ref": f"p6ds_ref_{tag}_1",
            },
            {
                "customer_id": f"p6ds_{tag}_2",
                "amount": None,
                "currency": "INR",  # invalid: missing amount
                "dt": "2026-09-01",
                "status": "failed",
                "ref": f"p6ds_ref_{tag}_2",
            },
            {
                "customer_id": f"p6ds_{tag}_3",
                "amount": 50,
                "currency": "INR",
                "dt": "2026-09-01",
                "status": "maybe",
                "ref": f"p6ds_ref_{tag}_3",
            },  # invalid: ambiguous
        ]
    )
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)

    r = client.post(
        "/datasets/upload",
        headers=KEY,
        files={
            "file": (
                f"p6_mixed_{tag}.xlsx",
                buf,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    dsid = r.json()["dataset_id"]
    client.post(f"/datasets/{dsid}/analyze", headers=KEY)
    r = client.post(
        f"/datasets/{dsid}/mapping",
        headers=KEY,
        json={
            "declared_domain": "FAILED_PAYMENT",
            "mappings": [
                {
                    "original_column": "customer_id",
                    "canonical_field": "CUSTOMER_ID",
                    "action": "confirm",
                },
                {"original_column": "amount", "canonical_field": "AMOUNT", "action": "confirm"},
                {"original_column": "dt", "canonical_field": "TIMESTAMP", "action": "confirm"},
                {"original_column": "status", "canonical_field": "OUTCOME", "action": "confirm"},
                {
                    "original_column": "ref",
                    "canonical_field": "TRANSACTION_ID",
                    "action": "confirm",
                },
                {"original_column": "currency", "canonical_field": "UNKNOWN", "action": "unused"},
            ],
        },
    )
    assert r.status_code == 200
    client.post(f"/datasets/{dsid}/ml-readiness", headers=KEY)
    r = client.post(f"/datasets/{dsid}/generate-cases", headers=KEY, json={"max_cases": 50})
    counters = r.json()["counters"]
    assert counters["rows_accepted"] == 1
    assert counters["invalid_amount"] == 1
    assert counters["ambiguous_target"] == 1
    return dsid


def test_D_a_dataset_file_can_mix_valid_and_invalid_rows():
    _ingest_mixed_validity_dataset(_tag())


def test_E_reuploading_the_same_dataset_file_creates_no_new_cases():
    dsid = _ingest_mixed_validity_dataset(_tag())
    r = client.post(f"/datasets/{dsid}/generate-cases", headers=KEY, json={"max_cases": 50})
    counters = r.json()["counters"]
    # Re-running generate-cases on the SAME dataset must recognise every accepted row
    # as already ingested - not create a duplicate case.
    assert counters["cases_created"] == 0
    assert counters["duplicate_event"] == 1


# ------------------------------------------------------------------ F


def test_F_duplicate_outcome_event_cannot_double_count_recovery():
    tag = _tag()
    customer_id = f"p6_dup_{tag}"
    reference_id = f"p6_dup_ref_{tag}"
    r = client.post(
        "/events",
        json={
            "customer_id": customer_id,
            "risk_category": "FAILED_PAYMENT",
            "external_system": "p6_sys",
            "external_event_id": f"p6_evt_{tag}",
            "reference_id": reference_id,
            "amount": 300,
            "currency": "INR",
            "raw_payload": {},
        },
    )
    case_id = r.json()["case_id"]
    body = {
        "external_system": "p6_out",
        "external_event_id": f"p6_dup_evt_{tag}",
        "domain": "FAILED_PAYMENT",
        "outcome_type": "PAYMENT_SETTLED",
        "customer_id": customer_id,
        "reference_id": reference_id,
        "amount": 300,
        "currency": "INR",
        "occurred_at": "2026-09-10T00:00:00Z",
    }
    r1 = client.post("/outcomes", headers=KEY, json=body)
    assert r1.json()["status"] == "processed"
    r2 = client.post("/outcomes", headers=KEY, json=body)  # exact replay
    assert r2.json()["status"] == "duplicate"

    snap = client.get(f"/cases/{case_id}/outcome", headers=KEY).json()
    assert snap["actual_amount_recovered"] == "300.0000"  # unchanged by the replay
