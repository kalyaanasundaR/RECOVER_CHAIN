"""Phase 4 domain-semantics regression tests.

A-E  each revenue domain is classified from explicit, named evidence
F    a dataset whose domain cannot be established is not silently a failed payment
G    an explicit domain overrides inference, and the declaration is audited
H-J  obligation identity; the same obligation correlates, different ones stay apart
K    legacy identity reconciliation (and Phase 3 v1 ids are still recognised)
L-O  contact permission from durable records: allowed, opted out, unknown, suppressed
P-Q  contact history from the execution ledger; contact frequency per customer
R    explicit action classification
S-T  verification semantics; partial recovery
U    currency-aware aggregation
V    the dataset category contract
W    Phase 1-3 guarantees on the new paths

Every test drives the real implementation: the HTTP API, the Policy Engine, the
classifier, the verification engine and the database.
"""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import app
from application.action_evaluator import DeterministicActionEvaluator
from application.case_engine import CaseEngine
from application.domain_classification import (
    DECLARED,
    INFERRED,
    MODE_DATASET,
    MODE_PER_ROW,
    NEEDS_MAPPING,
    classify_dataset,
)
from application.ingestion import event_fingerprint, fingerprint_event_id
from application.policy_engine import DeterministicPolicyEngine, MerchantPolicy, PolicyContext
from application.verification_engine import VerificationEngine
from domain.models import (
    ACTION_SPECS,
    ActionChannel,
    ActionRecommendation,
    ActionType,
    CandidateAction,
    DiagnosisStatus,
    ExecutionRecord,
    ExecutionStatus,
    Money,
    RecommendationStatus,
    RecoveryCase,
    RecoveryOutcomeStatus,
    RevenueEvent,
    RiskCategory,
    RootCauseCategory,
    RootCauseDiagnosis,
    action_applies_to,
    action_channel,
)
from domain.revenue_domains import DOMAIN_SPECS, DomainBasis, obligation_key
from infrastructure.adapters import MockExecutionAdapter
from infrastructure.db import SessionLocal
from infrastructure.orm import (
    AuditModel,
    EventIdentityAliasModel,
    EventModel,
    ExecutionAttemptModel,
)
from infrastructure.repositories import SqlAlchemyAuditRecorder, SqlAlchemyCaseRepository

client = TestClient(app)
TEST_DATASETS = Path(__file__).resolve().parents[2] / "data" / "test_datasets"

# ----------------------------------------------------------------- helpers

BASE = ["customer_id", "amount", "event_timestamp", "payment_result", "failure_reason"]


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _csv(header: list[str], rows: list[list]) -> bytes:
    lines = [",".join(header)] + [",".join("" if v is None else str(v) for v in r) for r in rows]
    return ("\n".join(lines) + "\n").encode()


def _file(tag, extra_cols=(), extra=lambda i: [], n_failed=6, n_paid=3, code="insufficient_funds"):
    """A small file: n_failed failed rows (distinct customers) and n_paid paid rows."""
    header = BASE + list(extra_cols)
    rows = [
        [f"{tag}_f{i}", f"{100 + i}.00", f"2026-08-{i + 1:02d}", "failed", code, *extra(i)]
        for i in range(n_failed)
    ]
    rows += [
        [f"{tag}_p{i}", f"{50 + i}.00", f"2026-07-{i + 1:02d}", "paid", "", *extra(i)]
        for i in range(n_paid)
    ]
    return header, rows


def _mappings(ds: str, overrides: dict | None = None) -> list[dict]:
    overrides = overrides or {}
    out = []
    for s in client.get(f"/datasets/{ds}").json()["recoverchain_signals"]:
        field = overrides.get(s["original_column"], s["canonical_field"])
        out.append(
            {
                "original_column": s["original_column"],
                "canonical_field": field,
                "action": "confirm" if field != "UNKNOWN" else "unused",
            }
        )
    return out


def _ready(content: bytes, name: str, overrides: dict | None = None, **contract) -> str:
    """Upload, profile, confirm the proposed mapping (plus any contract fields) and pass
    ML readiness - the calls the guided workflow makes before it can generate cases."""
    up = client.post(
        "/datasets/upload", files={"file": (f"{name}_{_tag()}.csv", content, "text/csv")}
    )
    assert up.status_code == 200, up.text
    ds = up.json()["dataset_id"]
    assert client.post(f"/datasets/{ds}/analyze").status_code == 200
    r = client.post(
        f"/datasets/{ds}/mapping", json={"mappings": _mappings(ds, overrides), **contract}
    )
    assert r.status_code == 200, r.text
    r = client.post(f"/datasets/{ds}/ml-readiness")
    assert r.status_code == 200, r.text
    return ds


def _generate(ds: str) -> dict:
    r = client.post(f"/datasets/{ds}/generate-cases", json={"max_cases": 100})
    assert r.status_code == 200, r.text
    return r.json()


def _snap(case_id: str) -> dict:
    return client.get(f"/system/cases/{case_id}").json()


def _categories(body: dict) -> set[str]:
    return {_snap(cid)["risk_category"] for cid in body["case_ids"]}


def _event(tag: str, **override) -> dict:
    ev = {
        "customer_id": f"p4_cust_{tag}",
        "risk_category": "OVERDUE_INVOICE",
        "external_system": "p4_sys",
        "external_event_id": f"p4_evt_{_tag()}",
        "amount": 400,
        "currency": "INR",
        "raw_payload": {},
    }
    ev.update(override)
    return ev


def _ingest(**ev) -> dict:
    r = client.post("/events", json=ev)
    assert r.status_code == 200, r.text
    return r.json()


def _advanced_case(**ev) -> str:
    """Ingest one event and run the deterministic pipeline up to the policy decision."""
    r = client.post("/events/batch", json={"events": [ev], "auto_advance": True})
    assert r.status_code == 200, r.text
    item = r.json()["results"][0]
    assert item["status"] == "success", item
    return item["case_id"]


def _decision(case_id: str) -> dict:
    return client.get(f"/cases/{case_id}/policy-decision").json()


def _permit(customer: str, channel: str, status: str, source: str = "test_suite", **kw):
    return client.post(
        f"/customers/{customer}/contact-permissions",
        json={"channel": channel, "status": status, "source": source, **kw},
    )


def _hand_built_case(action: ActionType, category: RiskCategory, amount=100.0) -> RecoveryCase:
    return RecoveryCase(
        case_id="c_p4",
        customer_id="cust_p4",
        risk_category=category,
        amount_at_risk=Money(amount=amount),
        recommendation=ActionRecommendation(
            recommendation_id="rec_p4",
            candidates=[],
            top_candidate=CandidateAction(
                action_type=action,
                estimated_probability=0.5,
                expected_recoverable_value=50.0,
                rationale="t",
            ),
            status=RecommendationStatus.RECOMMENDED,
            rationale="t",
            engine_version="t",
        ),
    )


def _stored_event(
    *, system, external_id, customer, amount, ts, reference, payload, source_ref=None
):
    """Store an event exactly as an older ingestion scheme did, through the CaseEngine."""
    db = SessionLocal()
    try:
        engine = CaseEngine(SqlAlchemyCaseRepository(db), SqlAlchemyAuditRecorder(db))
        case, _ = engine.ingest_normalized_event(
            RevenueEvent(
                event_id=f"evt_{_tag()}",
                customer_id=customer,
                risk_category=RiskCategory.FAILED_PAYMENT,
                external_system=system,
                external_event_id=external_id,
                reference_id=reference,
                amount=Money(amount=Decimal(amount), currency="INR"),
                timestamp=ts,
                raw_payload=payload,
                source_ref=source_ref,
            )
        )
        return case.case_id, case.linked_events[-1].event_id
    finally:
        db.close()


# ------------------------------------------------------------ A-F classification


def test_A_payment_failure_is_classified_from_payment_evidence():
    tag = _tag()
    header, rows = _file(tag)
    c = classify_dataset(pd.DataFrame(rows, columns=header))
    assert (c.status, c.mode, c.domain, c.basis) == (
        INFERRED,
        MODE_DATASET,
        RiskCategory.FAILED_PAYMENT,
        DomainBasis.PAYMENT_FAILURE_EVIDENCE,
    )
    assert {e["column"] for e in c.evidence} == {"payment_result", "failure_reason"}

    body = _generate(_ready(_csv(header, rows), "p4_payment"))
    assert body["domain_classification"]["domain"] == "FAILED_PAYMENT"
    assert _categories(body) == {"FAILED_PAYMENT"}
    snap = _snap(body["case_ids"][0])
    assert snap["data_events"][0]["domain_basis"] == DomainBasis.PAYMENT_FAILURE_EVIDENCE
    assert snap["domain"]["canonical_name"] == "PAYMENT_FAILURE"
    run = client.get(f"/ingestion-runs/{body['ingestion_run_id']}").json()
    assert run["domain_classification"]["accepted_by_domain"] == {"FAILED_PAYMENT": 6}


def test_B_checkout_is_classified_from_obligation_columns_regardless_of_column_order():
    tag = _tag()
    header, rows = _file(tag, ["checkout_phase"], lambda i: ["payment_entry"])
    df = pd.DataFrame(rows, columns=header)
    c = classify_dataset(df)
    # Decline codes are generic (every domain is paid somehow); a checkout column is not.
    assert (c.status, c.domain, c.basis) == (
        INFERRED,
        RiskCategory.CHECKOUT_ABANDONMENT,
        DomainBasis.SCHEMA_EVIDENCE,
    )
    reordered = classify_dataset(df[list(reversed(header))])
    assert (reordered.status, reordered.domain, reordered.basis) == (c.status, c.domain, c.basis)

    body = _generate(_ready(_csv(header, rows), "p4_checkout"))
    assert _categories(body) == {"CHECKOUT_ABANDONMENT"}


def test_C_subscription_file_is_no_longer_ingested_as_a_failed_payment():
    """The Phase 3 defect: a billing_cycle workbook became FAILED_PAYMENT because the
    keyword heuristic only looked for category-like column names."""
    tag = _tag()
    header, rows = _file(tag, ["billing_cycle"], lambda i: ["monthly" if i % 2 else "annual"])
    body = _generate(_ready(_csv(header, rows), "p4_subscription"))
    assert body["domain_classification"]["domain"] == "FAILED_SUBSCRIPTION"
    assert _categories(body) == {"FAILED_SUBSCRIPTION"}


@pytest.mark.parametrize(
    "name, expected",
    [
        ("RecoverChain_Payment_Failure.xlsx", ("FAILED_PAYMENT", MODE_DATASET)),
        ("RecoverChain_Checkout_Risk.xlsx", ("CHECKOUT_ABANDONMENT", MODE_DATASET)),
        ("RecoverChain_Subscription_Risk.xlsx", ("FAILED_SUBSCRIPTION", MODE_DATASET)),
        ("RecoverChain_Invoice_Risk.xlsx", ("OVERDUE_INVOICE", MODE_DATASET)),
        ("RecoverChain_Comprehensive_Mixed.xlsx", (None, MODE_PER_ROW)),
    ],
)
def test_C2_shipped_domain_workbooks_classify_to_their_domain(name, expected):
    path = TEST_DATASETS / name
    if not path.exists():
        pytest.skip(f"{name} is not present in this checkout")
    df = pd.read_excel(path, nrows=500)  # read-only
    c = classify_dataset(df)
    assert c.status == INFERRED
    assert ((c.domain.value if c.domain else None), c.mode) == expected


def test_D_invoice_is_classified_from_named_invoice_columns():
    c = classify_dataset(
        pd.DataFrame(
            {
                "customerID": ["c1"],
                "invoiceNumber": ["INV-1"],
                "InvoiceAmount": [10],
                "DueDate": ["2026-08-01"],
                "target_late": [1],
            }
        )
    )
    assert (c.status, c.domain) == (INFERRED, RiskCategory.OVERDUE_INVOICE)
    assert set(c.evidence[-1]["columns"]) == {"invoiceNumber", "InvoiceAmount", "DueDate"}

    tag = _tag()
    header = ["customer_id", "invoice_number", "invoice_amount", "due_date", "status", "notes"]
    rows = [
        [f"{tag}_f{i}", f"INV-{tag}-{i}", f"{300 + i}.00", f"2026-08-{i + 1:02d}", "unpaid", "po"]
        for i in range(6)
    ]
    rows += [
        [f"{tag}_p{i}", f"INV-{tag}-p{i}", f"{90 + i}.00", f"2026-07-{i + 1:02d}", "paid", "ok"]
        for i in range(3)
    ]
    body = _generate(_ready(_csv(header, rows), "p4_invoice"))
    assert _categories(body) == {"OVERDUE_INVOICE"}


def test_E_promise_to_pay_is_classified_by_columns_and_by_category_values():
    tag = _tag()
    header, rows = _file(tag, ["promise_date"], lambda i: [f"2026-08-{i + 2:02d}"])
    c = classify_dataset(pd.DataFrame(rows, columns=header))
    assert (c.status, c.domain, c.basis) == (
        INFERRED,
        RiskCategory.BROKEN_PROMISE,
        DomainBasis.SCHEMA_EVIDENCE,
    )

    per_row = classify_dataset(
        pd.DataFrame({"customer_id": ["a", "b"], "revenue_stream": ["Promise to Pay", "invoice"]})
    )
    assert (per_row.status, per_row.mode, per_row.category_column) == (
        INFERRED,
        MODE_PER_ROW,
        "revenue_stream",
    )
    assert per_row.resolve({"revenue_stream": "promise-to-pay"})[0] == RiskCategory.BROKEN_PROMISE
    assert per_row.resolve({"revenue_stream": "invoice"})[0] == RiskCategory.OVERDUE_INVOICE

    ds = _ready(
        _csv(header, rows),
        "p4_promise",
        overrides={"promise_date": "UNKNOWN", "event_timestamp": "TIMESTAMP"},
    )
    assert _categories(_generate(ds)) == {"BROKEN_PROMISE"}


def test_F_an_unidentifiable_dataset_is_not_silently_a_failed_payment():
    tag = _tag()
    header = ["customer_id", "amount", "event_timestamp", "status", "notes", "sales_channel"]
    rows = [
        [
            f"{tag}_f{i}",
            f"{100 + i}.00",
            f"2026-08-{i + 1:02d}",
            "failed",
            "subscription renewal",
            "web",
        ]
        for i in range(6)
    ]
    rows += [[f"{tag}_p{i}", "60.00", "2026-07-01", "paid", "", "store"] for i in range(3)]
    df = pd.DataFrame(rows, columns=header)
    # Incidental words: "subscription" inside a value, "channel" inside a column name
    # (the retired heuristic matched both), and a category-named column with no domain.
    df["subscription_notes"] = "see crm"
    df["category"] = "retail"
    c = classify_dataset(df)
    assert c.status == NEEDS_MAPPING and c.domain is None and c.resolve({}) is None

    ds = _ready(_csv(header, rows), "p4_unknown")
    runs_before = client.get(f"/datasets/{ds}/ingestion-runs").json()
    r = client.post(f"/datasets/{ds}/generate-cases", json={"max_cases": 100})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "DOMAIN_NEEDS_MAPPING"
    assert detail["domain_classification"]["status"] == NEEDS_MAPPING
    db = SessionLocal()
    try:
        assert db.query(EventModel).filter(EventModel.source_ref == ds).count() == 0
    finally:
        db.close()
    assert client.get(f"/datasets/{ds}/ingestion-runs").json() == runs_before


def test_F2_a_confirmed_category_column_rejects_rows_it_cannot_resolve():
    tag = _tag()
    header, rows = _file(
        tag, ["event_type"], lambda i: ["subscription_renewal" if i % 3 else "marketing_blast"]
    )
    ds = _ready(
        _csv(header, rows),
        "p4_per_row",
        overrides={"event_type": "EVENT_CATEGORY", "event_timestamp": "TIMESTAMP"},
    )
    body = _generate(ds)
    dc = body["domain_classification"]
    assert (dc["status"], dc["mode"], dc["basis"]) == (
        DECLARED,
        MODE_PER_ROW,
        DomainBasis.CATEGORY_COLUMN_MAPPED,
    )
    assert body["counters"]["unresolved_domain"] == 2  # failed rows 0 and 3
    assert {s["reason"] for s in body["rejected_samples"]} == {"unresolved_domain"}
    assert _categories(body) == {"FAILED_SUBSCRIPTION"}


# ------------------------------------------------------------------ G explicit wins


def test_G_an_explicit_domain_overrides_inference_and_is_audited():
    tag = _tag()
    header, rows = _file(tag, ["billing_cycle"], lambda i: ["monthly"])
    ds = _ready(_csv(header, rows), "p4_declared")
    put = client.put(f"/datasets/{ds}/domain", json={"declared_domain": "OVERDUE_INVOICE"})
    assert put.status_code == 200, put.text
    dc = put.json()["domain_classification"]
    assert (dc["status"], dc["domain"], dc["basis"]) == (
        DECLARED,
        "OVERDUE_INVOICE",
        DomainBasis.DECLARED_FOR_DATASET,
    )
    assert dc["inferred"]["domain"] == "FAILED_SUBSCRIPTION" and dc["warnings"]

    body = _generate(ds)
    assert _categories(body) == {"OVERDUE_INVOICE"}
    assert _snap(body["case_ids"][0])["data_events"][0]["domain_basis"] == (
        DomainBasis.DECLARED_FOR_DATASET
    )
    db = SessionLocal()
    try:
        declared = [
            a.evidence
            for a in db.query(AuditModel).filter(AuditModel.case_id.is_(None)).all()
            if (a.evidence or {}).get("action") == "dataset_domain_declared"
            and a.evidence.get("dataset_id") == ds
        ]
    finally:
        db.close()
    assert declared and declared[-1]["declared_domain"] == "OVERDUE_INVOICE"

    api_case = _ingest(**_event(_tag(), risk_category="BROKEN_PROMISE"))["case_id"]
    assert _snap(api_case)["data_events"][0]["domain_basis"] == DomainBasis.DECLARED_BY_SOURCE


# ------------------------------------------------------------ H-J obligation identity


def test_H_obligation_identity_is_domain_customer_and_reference_never_customer_alone():
    assert obligation_key("c1", RiskCategory.OVERDUE_INVOICE, None) is None
    assert obligation_key("c1", "OVERDUE_INVOICE", "") is None
    key = obligation_key("c1", "OVERDUE_INVOICE", "INV-1")
    assert key != obligation_key("c1", "CHECKOUT_ABANDONMENT", "INV-1")
    assert key != obligation_key("c2", "OVERDUE_INVOICE", "INV-1")
    assert key != obligation_key("c1", "OVERDUE_INVOICE", "INV-2")
    assert set(DOMAIN_SPECS) == set(RiskCategory)
    assert {s.canonical_name for s in DOMAIN_SPECS.values()} == {
        "PAYMENT_FAILURE",
        "CHECKOUT_ABANDONMENT",
        "SUBSCRIPTION_FAILURE",
        "INVOICE_OVERDUE",
        "PROMISE_TO_PAY",
    }


def test_I_same_obligation_correlates_and_a_row_is_recognised_across_domain_changes():
    tag = _tag()
    first = _ingest(**_event(tag, reference_id=f"INV-{tag}"))
    second = _ingest(**_event(tag, reference_id=f"INV-{tag}", amount=380))
    assert second["case_id"] == first["case_id"] and second["is_new_case"] is False

    # The same rows in a second upload whose domain is DECLARED differently: identity no
    # longer depends on the domain, so they are recognised and nothing new is created.
    header, rows = _file(tag)
    content = _csv(header, rows)
    first_run = _generate(_ready(content, "p4_same_a"))
    again = _generate(_ready(content, "p4_same_b", declared_domain="OVERDUE_INVOICE"))
    assert again["counters"]["cases_created"] == 0
    assert again["counters"]["duplicate_event"] == 6
    assert set(again["case_ids"]) == set(first_run["case_ids"])
    assert _categories(again) == {"FAILED_PAYMENT"}


def test_J_different_obligations_stay_separate():
    tag = _tag()
    customer = f"p4_sep_{tag}"
    results = [
        _ingest(**_event(tag, customer_id=customer, reference_id="REF-1")),
        _ingest(**_event(tag, customer_id=customer, reference_id="REF-2")),  # other reference
        _ingest(
            **_event(
                tag,
                customer_id=customer,
                reference_id="REF-1",
                risk_category="CHECKOUT_ABANDONMENT",
            )
        ),  # other domain
        _ingest(**_event(_tag(), reference_id="REF-1")),  # other customer
        _ingest(**_event(tag, customer_id=customer)),  # no reference ...
        _ingest(**_event(tag, customer_id=customer)),  # ... is never correlated
    ]
    assert len({r["case_id"] for r in results}) == 6
    assert all(r["is_new_case"] for r in results)


# ------------------------------------------------------------ K legacy reconciliation


def test_K_legacy_identities_are_reconciled_without_rewriting_or_merging():
    tag = _tag()
    customer = f"p4_legacy_{tag}"
    ts = datetime(2026, 8, 3, tzinfo=UTC)
    payload = {
        "customer_id": customer,
        "amount": 321.0,
        "event_timestamp": "2026-08-03",
        "payment_result": "failed",
        "failure_reason": "insufficient_funds",
    }
    legacy = dict(
        external_id="row_0",
        customer=customer,
        amount="321.00",
        ts=ts,
        reference="tx_0",
        payload=payload,
    )
    case_a, event_a = _stored_event(system=f"Dataset-legacyA_{tag}", **legacy)
    # the same file uploaded again under the legacy scheme
    _, event_b = _stored_event(system=f"Dataset-legacyB_{tag}", **legacy)
    scope = f"dataset_id=legacyA_{tag}&dataset_id=legacyB_{tag}"

    dry = client.post(f"/events/identity-reconciliation?{scope}").json()
    assert dry["mode"] == "DRY_RUN" and dry["aliases_added"] == 0
    assert dry["scanned"]["legacy_dataset_row"] == 2
    assert dry["eligible"] == 1 and dry["eligible_sample"][0]["event_id"] == event_a
    assert dry["conflict_count"] == 1
    conflict = dry["conflicts"][0]
    assert conflict["event_id"] == event_b and conflict["other_event_id"] == event_a
    assert conflict["kind"] == "same_identity_as_another_older_event"
    assert dry["limitations"]
    db = SessionLocal()
    try:
        assert db.query(EventIdentityAliasModel).filter_by(event_id=event_a).count() == 0
    finally:
        db.close()

    applied = client.post(f"/events/identity-reconciliation?{scope}&apply=true").json()
    assert applied["mode"] == "APPLIED" and applied["aliases_added"] == 1
    db = SessionLocal()
    try:
        alias = db.query(EventIdentityAliasModel).filter_by(event_id=event_a).one()
        assert alias.method == "legacy_dataset_row"
        assert alias.reconciliation_run_id == applied["run_id"]
        assert alias.details["fabricated_reference_ignored"] == "tx_0"
        original = db.query(EventModel).filter_by(event_id=event_a).one()
        assert (original.external_system, original.external_event_id) == (
            f"Dataset-legacyA_{tag}",
            "row_0",
        )  # never rewritten
        assert db.query(EventModel).filter_by(event_id=event_b).count() == 1  # never merged
        assert any(
            (a.evidence or {}).get("run_id") == applied["run_id"]
            for a in db.query(AuditModel).filter(AuditModel.case_id.is_(None)).all()
        )
    finally:
        db.close()

    again = client.post(f"/events/identity-reconciliation?{scope}&apply=true").json()
    assert (again["aliases_added"], again["already_aliased"], again["conflict_count"]) == (0, 1, 1)

    # A re-upload of that row is now recognised instead of becoming another case.
    header, rows = _file(f"pad{tag}")
    row = [customer, "321.00", "2026-08-03", "failed", "insufficient_funds"]
    body = _generate(
        _ready(
            _csv(header, [row, *rows]),
            "p4_legacy_reupload",
            overrides={"payment_result": "OUTCOME"},
        )
    )
    assert body["counters"]["duplicate_event"] == 1
    assert case_a in body["case_ids"]


def test_K2_rows_stored_under_the_phase3_v1_identity_are_still_recognised():
    tag = _tag()
    customer = f"p4_v1_{tag}"
    ts = datetime(2026, 8, 4, tzinfo=UTC)
    v1_id = fingerprint_event_id(
        event_fingerprint(
            customer_id=customer,
            risk_category=RiskCategory.FAILED_PAYMENT,
            amount=Decimal("222.00"),
            currency="INR",
            timestamp=ts,
            reference_id=None,
            failure_code="network_error",
        ),
        0,
    )
    case_id, event_id = _stored_event(
        system="file_upload",
        external_id=v1_id,
        customer=customer,
        amount="222.00",
        ts=ts,
        reference=None,
        payload={"failure_code": "network_error"},
        source_ref=f"v1ds_{tag}",
    )
    dry = client.post(f"/events/identity-reconciliation?dataset_id=v1ds_{tag}").json()
    assert dry["eligible_sample"][0] == {
        "event_id": event_id,
        "case_id": case_id,
        "identity": dry["eligible_sample"][0]["identity"],
        "method": "content_identity_v1",
    }
    assert dry["eligible_sample"][0]["identity"].startswith("fp2:")

    header, rows = _file(f"pad{tag}")
    row = [customer, "222.00", "2026-08-04", "failed", "network_error"]
    body = _generate(
        _ready(_csv(header, [row, *rows]), "p4_v1", overrides={"payment_result": "OUTCOME"})
    )
    assert body["counters"]["duplicate_event"] == 1
    assert case_id in body["case_ids"]


# ------------------------------------------------------------ L-O contact permission


def test_L_a_recorded_permission_allows_contact_and_is_cited():
    tag = _tag()
    customer = f"p4_allowed_{tag}"
    rec = _permit(customer, "communication", "ALLOWED", reason="opted in at signup")
    assert rec.status_code == 200, rec.text
    record_id = rec.json()["record"]["id"]

    decision = _decision(_advanced_case(**_event(tag, customer_id=customer)))
    assert decision["status"] == "PERMITTED"
    assert decision["contact_permission"] == {
        "channel": "communication",
        "status": "ALLOWED",
        "record_id": record_id,
        "unknown_consent_behaviour": None,
    }
    rule = next(r for r in decision["rules_evaluated"] if r["rule_name"] == "ContactPermission")
    assert rule["passed"] and record_id in rule["details"]


def test_M_an_opted_out_customer_is_not_messaged_but_an_authorised_retry_is_unaffected():
    tag = _tag()
    customer = f"p4_optout_{tag}"
    assert (
        _permit(
            customer, "communication", "OPTED_OUT", source="customer_request", reason="replied STOP"
        ).status_code
        == 200
    )

    invoice_case = _advanced_case(**_event(tag, customer_id=customer))
    decision = _decision(invoice_case)
    assert decision["status"] == "DENIED"
    failed = next(r for r in decision["failed_rules"] if r["rule_name"] == "ContactPermission")
    assert "OPTED_OUT" in failed["details"] and "customer_request" in failed["details"]
    assert client.post(f"/cases/{invoice_case}/execute").json()["status"] == "REJECTED"
    db = SessionLocal()
    try:
        assert db.query(ExecutionAttemptModel).filter_by(case_id=invoice_case).count() == 0
    finally:
        db.close()

    retry_case = _advanced_case(
        **_event(
            tag,
            customer_id=customer,
            risk_category="FAILED_PAYMENT",
            raw_payload={"failure_code": "network_error"},
        )
    )
    retry = _decision(retry_case)
    assert retry["status"] == "PERMITTED"
    assert retry["contact_permission"]["channel"] == "payment"


def test_N_unknown_consent_is_recorded_as_unknown_and_its_handling_is_explicit():
    decision = _decision(_advanced_case(**_event(_tag())))
    assert decision["status"] == "PERMITTED"
    assert decision["contact_permission"]["status"] == "UNKNOWN"
    assert decision["contact_permission"]["unknown_consent_behaviour"] == "ALLOW"
    rule = next(r for r in decision["rules_evaluated"] if r["rule_name"] == "ContactPermission")
    assert rule["passed"] and "not recorded consent" in rule["details"]

    case = _hand_built_case(ActionType.SEND_INVOICE_REMINDER, RiskCategory.OVERDUE_INVOICE)
    unknown = PolicyContext()
    escalate = MerchantPolicy(unknown_consent_communication="ESCALATE")
    deny = MerchantPolicy(unknown_consent_communication="DENY")
    assert DeterministicPolicyEngine(escalate).evaluate(case, unknown).status.value == "ESCALATE"
    assert DeterministicPolicyEngine(deny).evaluate(case, unknown).status.value == "DENIED"
    allowed = PolicyContext(communication_permission="ALLOWED")
    assert DeterministicPolicyEngine(deny).evaluate(case, allowed).status.value == "PERMITTED"
    with pytest.raises(ValidationError):
        MerchantPolicy(unknown_consent_communication="ASSUME_YES")
    with pytest.raises(ValidationError):
        PolicyContext(consent_ok=True)  # the never-populated flag is gone, not ignored


def test_O_suppression_blocks_every_automated_action_and_survives_human_approval():
    tag = _tag()
    customer = f"p4_supp_{tag}"
    escalated = _advanced_case(**_event(tag, customer_id=customer, amount=6000))
    assert _decision(escalated)["status"] == "ESCALATE"

    assert _permit(customer, "communication", "SUPPRESSED").status_code == 422  # customer-wide
    assert (
        _permit(
            customer, "all", "SUPPRESSED", source="compliance_hold", reason="legal hold"
        ).status_code
        == 200
    )

    payment_case = _advanced_case(
        **_event(
            tag,
            customer_id=customer,
            risk_category="FAILED_PAYMENT",
            raw_payload={"failure_code": "network_error"},
        )
    )
    invoice_case = _advanced_case(**_event(tag, customer_id=customer))
    for case_id in (payment_case, invoice_case):
        decision = _decision(case_id)
        assert decision["status"] == "DENIED"
        assert decision["contact_permission"]["status"] == "SUPPRESSED"

    # A person approves the earlier escalation; execution revalidates and still refuses.
    review = client.post(
        f"/cases/{escalated}/human-review", json={"decision": "APPROVE", "note": "ok"}
    )
    assert review.status_code == 200, review.text
    executed = client.post(f"/cases/{escalated}/execute").json()
    assert executed["status"] == "REJECTED"
    assert executed["result_metadata"]["revalidated_status"] == "DENIED"
    assert "SUPPRESSED" in executed["result_metadata"]["reason"]

    # A channel-specific ALLOWED does not lift a suppression; a later customer-wide one does.
    assert _permit(customer, "communication", "ALLOWED").status_code == 200
    state = client.get(f"/customers/{customer}/contact-permissions").json()
    assert state["effective"]["communication"]["status"] == "SUPPRESSED"
    assert _permit(customer, "all", "ALLOWED", reason="hold released").status_code == 200
    state = client.get(f"/customers/{customer}/contact-permissions").json()
    assert state["effective"]["communication"]["status"] == "ALLOWED"
    assert state["effective"]["payment"]["status"] == "ALLOWED"
    assert len(state["records"]) == 3


# ------------------------------------------------------------ P-Q contact history


def test_P_contact_history_is_answered_from_the_execution_ledger():
    tag = _tag()
    customer = f"p4_hist_{tag}"
    case_id = _advanced_case(**_event(tag, customer_id=customer, reference_id=f"INV-{tag}"))
    decision_id = _decision(case_id)["decision_id"]
    executed = client.post(f"/cases/{case_id}/execute").json()
    assert executed["status"] == "COMPLETED_SIMULATED", executed

    history = client.get(f"/customers/{customer}/contact-history").json()
    assert history["source"] == "execution_attempts"
    assert (history["communication_in_window"], history["payment_attempts_in_window"]) == (1, 0)
    (attempt,) = history["attempts"]
    assert attempt["case_id"] == case_id and attempt["reference_id"] == f"INV-{tag}"
    assert attempt["channel"] == "communication"
    assert attempt["action_type"] == "SEND_INVOICE_REMINDER"
    assert attempt["policy_decision_id"] == executed["policy_decision_id"] == decision_id
    assert attempt["policy_reason"]
    assert attempt["permission_at_authorisation"] == "UNKNOWN"
    assert attempt["permission_in_force_at_attempt"] == "UNKNOWN"


def test_Q_contact_frequency_is_limited_per_customer_across_obligations():
    tag = _tag()
    customer = f"p4_freq_{tag}"
    first = _advanced_case(**_event(tag, customer_id=customer, reference_id="INV-A"))
    second = _advanced_case(**_event(tag, customer_id=customer, reference_id="INV-B"))
    for case_id in (first, second):
        assert client.post(f"/cases/{case_id}/execute").json()["status"] == "COMPLETED_SIMULATED"

    third = _decision(_advanced_case(**_event(tag, customer_id=customer, reference_id="INV-C")))
    assert third["status"] == "WAIT"
    assert any(r["rule_name"] == "CommunicationMaxMessages" for r in third["failed_rules"])
    other_customer = _decision(_advanced_case(**_event(_tag(), reference_id="INV-C")))
    assert other_customer["status"] == "PERMITTED"


# ------------------------------------------------------------ R action classification


def test_R_actions_are_classified_explicitly_not_by_their_names():
    assert set(ACTION_SPECS) == set(ActionType)
    assert {a for a in ActionType if action_channel(a) == ActionChannel.PAYMENT} == {
        ActionType.RETRY_PAYMENT,
        ActionType.RETRY_BILLING,
    }

    adapter = MockExecutionAdapter()
    # The old adapter "completed" anything whose name contained REMINDER.
    assert adapter.execute({"action": "SEND_FAKE_REMINDER"})["adapter_status"] == "FAILED"
    gateways = {ActionChannel.PAYMENT: "sandbox", ActionChannel.COMMUNICATION: "mock_twilio"}
    for action in ActionType:
        result = adapter.execute({"action": action.value})
        if action_channel(action) in gateways:
            assert result["metadata"]["gateway"] == gateways[action_channel(action)]
        else:
            assert result["adapter_status"] == "FAILED"

    evaluator = DeterministicActionEvaluator()
    for category in RiskCategory:
        for cause in RootCauseCategory:
            case = RecoveryCase(
                case_id="c",
                customer_id="x",
                risk_category=category,
                amount_at_risk=Money(amount=100),
                diagnosis=RootCauseDiagnosis(
                    diagnosis_id="d",
                    cause_category=cause,
                    confidence=1.0,
                    status=DiagnosisStatus.INFERRED,
                    supporting_signals={},
                    evidence_references=[],
                ),
            )
            for action in evaluator.generate_candidates(case):
                assert action_applies_to(action, category), (category, cause, action)

    wrong = _hand_built_case(ActionType.SEND_CHECKOUT_REMINDER, RiskCategory.OVERDUE_INVOICE)
    decision = DeterministicPolicyEngine().evaluate(
        wrong, PolicyContext(communication_permission="ALLOWED")
    )
    assert decision.status.value == "DENIED"
    assert [r.rule_name for r in decision.failed_rules] == ["ActionDomainApplicability"]


# ------------------------------------------------------------ S-T verification


class _SettlementSource:
    def __init__(self, status, amount):
        self.status, self.amount = status, amount

    def verify(self, case, external_reference):
        return {"status": self.status, "amount": self.amount, "source": "TEST_SETTLEMENT"}


def test_S_a_completed_action_is_not_recovered_money():
    case = _hand_built_case(
        ActionType.SEND_INVOICE_REMINDER, RiskCategory.OVERDUE_INVOICE, amount=1000.0
    )
    case.execution_record = ExecutionRecord(
        execution_id="exec_p4",
        action_type=ActionType.SEND_INVOICE_REMINDER,
        agent_type="t",
        policy_decision_id="pol_p4",
        policy_version="t",
        parameters={},
        idempotency_key="k",
        status=ExecutionStatus.COMPLETED_SIMULATED,
        adapter_used="MockExecutionAdapter",
    )
    nothing = VerificationEngine(
        adapter=_SettlementSource(RecoveryOutcomeStatus.NOT_RECOVERED, 0.0)
    ).reconcile(case, "ref")
    assert nothing.status == RecoveryOutcomeStatus.NOT_RECOVERED
    assert nothing.actual_amount_recovered.amount == 0
    assert nothing.action_result == "COMPLETED_SIMULATED"  # the action DID complete
    assert nothing.verification_basis == "INVOICE_SETTLED"
    assert nothing.outstanding_amount.amount == Decimal("1000")
    overpaid = VerificationEngine(
        adapter=_SettlementSource(RecoveryOutcomeStatus.FULLY_RECOVERED, 5000.0)
    ).reconcile(case, "ref")
    assert overpaid.actual_amount_recovered.amount == Decimal("1000")
    assert overpaid.outstanding_amount.amount == 0
    assert len({s.verification_basis for s in DOMAIN_SPECS.values()}) == len(RiskCategory)

    # Through the API, in a currency no other test uses. Phase 5: an executed action
    # with no matching outcome event is honestly PENDING_VERIFICATION, not a fabricated
    # NOT_RECOVERED - ACTION ATTEMPT != REVENUE RECOVERED holds even for the negative case.
    case_id = _advanced_case(**_event(_tag(), currency="CHF"))
    assert client.post(f"/cases/{case_id}/execute").json()["status"] == "COMPLETED_SIMULATED"
    before = client.get("/dashboard/metrics").json()["currency_breakdown"]["CHF"]
    verified = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""}).json()
    assert verified["status"] == "PENDING_VERIFICATION"
    assert Decimal(str(verified["actual_amount_recovered"])) == 0
    assert verified["action_result"] == "COMPLETED_SIMULATED"
    assert verified["verification_basis"] == "INVOICE_SETTLED"
    after = client.get("/dashboard/metrics").json()["currency_breakdown"]["CHF"]
    assert Decimal(str(after["verified_recovery"])) == Decimal(str(before["verified_recovery"]))


def test_T_partial_recovery_closes_with_the_balance_recorded_and_the_obligation_continues():
    tag = _tag()
    customer = f"p4_partial_{tag}"
    reference = f"INV-{tag}"
    first = _advanced_case(**_event(tag, customer_id=customer, reference_id=reference, amount=1000))
    assert client.post(f"/cases/{first}/execute").json()["status"] == "COMPLETED_SIMULATED"
    settle = client.post(
        "/outcomes",
        headers={"X-API-Key": "test-api-key"},
        json={
            "external_system": "p4_settlement",
            "external_event_id": f"settle_{tag}",
            "domain": "OVERDUE_INVOICE",
            "outcome_type": "INVOICE_SETTLED",
            "customer_id": customer,
            "reference_id": reference,
            "amount": 500,
            "currency": "INR",
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    )
    assert settle.status_code == 200, settle.text
    assert settle.json()["match_status"] == "MATCHED" and settle.json()["case_id"] == first
    verified = client.post(f"/cases/{first}/verify", json={"external_reference": ""}).json()
    assert verified["status"] == "PARTIALLY_RECOVERED"
    assert Decimal(str(verified["actual_amount_recovered"])) == Decimal("500")
    assert Decimal(str(verified["outstanding_amount"])) == Decimal("500")
    assert _snap(first)["current_state"] == "PARTIALLY_RECOVERED"

    # The balance arrives as a new event for the same invoice: a new case, not a reopened
    # one - and the obligation's contact history carries over.
    remainder = _ingest(**_event(tag, customer_id=customer, reference_id=reference, amount=500))
    assert remainder["is_new_case"] is True and remainder["case_id"] != first
    assert _snap(first)["current_state"] == "PARTIALLY_RECOVERED"
    db = SessionLocal()
    try:
        context = SqlAlchemyCaseRepository(db).get_policy_context(remainder["case_id"])
    finally:
        db.close()
    assert context.comms_in_last_24h == 1


# ------------------------------------------------------------ U currency


def test_U_money_is_aggregated_per_currency_never_across():
    def breakdown():
        return client.get("/dashboard/metrics").json()["currency_breakdown"]

    def total(b, ccy):
        return Decimal(str(b.get(ccy, {}).get("total_revenue_at_risk", 0)))

    before = breakdown()
    for amount in (100, 250):
        _ingest(**_event(_tag(), currency="SEK", amount=amount))
    _ingest(**_event(_tag(), currency="NOK", amount=70))
    after = breakdown()
    assert total(after, "SEK") - total(before, "SEK") == Decimal("350")
    assert total(after, "NOK") - total(before, "NOK") == Decimal("70")
    assert client.get("/dashboard/metrics").json()["mixed_currency"] is True


# ------------------------------------------------------------ V dataset contract


def test_V_the_dataset_category_contract_is_declared_stored_and_reported():
    tag = _tag()
    header, rows = _file(tag, ["aging_bucket"], lambda i: ["d31_60"])
    ds = _ready(_csv(header, rows), "p4_contract", source_system="netsuite")

    contract = client.get(f"/datasets/{ds}/domain").json()
    assert contract["declared_domain"] is None and contract["source_system"] == "netsuite"
    assert contract["content_sha256"]
    assert {m["original_column"] for m in contract["schema"]} == set(header)
    assert [d["domain"] for d in contract["domains"]] == [c.value for c in RiskCategory]
    classification = contract["domain_classification"]
    assert (classification["status"], classification["domain"]) == (INFERRED, "OVERDUE_INVOICE")
    assert classification["classifier_version"]
    assert {
        "kind": "obligation_columns",
        "domain": "OVERDUE_INVOICE",
        "columns": ["aging_bucket"],
    } in classification["evidence"]

    confirmed = client.post(
        f"/datasets/{ds}/mapping",
        json={"mappings": _mappings(ds), "declared_domain": "OVERDUE_INVOICE"},
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["domain_classification"]["status"] == DECLARED
    detail = client.get(f"/datasets/{ds}").json()
    assert detail["declared_domain"] == "OVERDUE_INVOICE"
    assert detail["domain_classification"]["basis"] == DomainBasis.DECLARED_FOR_DATASET
    assert client.post(f"/datasets/{ds}/ml-readiness").status_code == 200

    body = _generate(ds)
    run = client.get(f"/ingestion-runs/{body['ingestion_run_id']}").json()
    assert run["domain_classification"]["classification"]["status"] == DECLARED
    assert run["domain_classification"]["accepted_by_domain"] == {"OVERDUE_INVOICE": 6}

    cleared = client.put(f"/datasets/{ds}/domain", json={"declared_domain": None}).json()
    assert cleared["declared_domain"] is None
    assert cleared["domain_classification"]["status"] == INFERRED
    bad = client.put(f"/datasets/{ds}/domain", json={"declared_domain": "NOT_A_DOMAIN"})
    assert bad.status_code == 422


# ------------------------------------------------------------ W earlier guarantees


def test_W_a_declared_domain_cannot_bypass_validation_and_verification_stays_idempotent():
    tag = _tag()
    header = BASE + ["currency"]
    rows = [
        [f"{tag}_ok{i}", f"{120 + i}.00", f"2026-08-{i + 1:02d}", "failed", "network_error", "INR"]
        for i in range(5)
    ]
    rows += [
        [f"{tag}_bad_amt", "abc", "2026-08-10", "failed", "network_error", "INR"],
        [f"{tag}_neg_amt", "-5", "2026-08-11", "failed", "network_error", "INR"],
        [f"{tag}_bad_ccy", "80.00", "2026-08-12", "failed", "network_error", "XYZ"],
        [f"{tag}_bad_date", "90.00", "not-a-date", "failed", "network_error", "INR"],
    ]
    rows += [[f"{tag}_p{i}", "60.00", "2026-07-01", "paid", "", "INR"] for i in range(3)]
    body = _generate(
        _ready(_csv(header, rows), "p4_validation", declared_domain="FAILED_SUBSCRIPTION")
    )
    counters = body["counters"]
    assert counters["rows_accepted"] == 5
    assert (
        counters["invalid_amount"],
        counters["invalid_currency"],
        counters["invalid_timestamp"],
    ) == (
        2,
        1,
        1,
    )

    reference = f"INV-W-{tag}"
    customer = f"p4_cust_{tag}"
    case_id = _advanced_case(**_event(tag, customer_id=customer, reference_id=reference))
    assert client.post(f"/cases/{case_id}/execute").json()["status"] == "COMPLETED_SIMULATED"
    settle = client.post(
        "/outcomes",
        headers={"X-API-Key": "test-api-key"},
        json={
            "external_system": "p4_settlement",
            "external_event_id": f"settle_w_{tag}",
            "domain": "OVERDUE_INVOICE",
            "outcome_type": "INVOICE_SETTLED",
            "customer_id": customer,
            "reference_id": reference,
            "amount": 400,
            "currency": "INR",
            "occurred_at": "2026-01-01T00:00:00Z",
        },
    )
    assert settle.status_code == 200 and settle.json()["case_id"] == case_id
    once = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""}).json()
    twice = client.post(f"/cases/{case_id}/verify", json={"external_reference": ""}).json()
    assert once["outcome_id"] == twice["outcome_id"] and twice["status"] == "FULLY_RECOVERED"
