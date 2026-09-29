"""Phase 16 - end-to-end traceability, authenticated observability, redaction, isolation."""

import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from api.access import require_operator, resolve_role
from api.main import app
from application.agent_trace import load_trace_data, reconstruct_trace
from application.trace_ids import (
    case_id_candidates_for_trace,
    is_trace_id,
    redact,
    trace_id_for_case,
)
from application.trace_validation import validate_all, validate_trace
from infrastructure.db import SessionLocal, engine
from infrastructure.orm import (
    AuditModel,
    CaseModel,
    ExecutionAttemptModel,
    OutcomeEventModel,
)

client = TestClient(app)
KEY = {"X-API-Key": "test-api-key"}
BACKEND = Path(__file__).resolve().parent.parent


def _tag():
    return uuid.uuid4().hex[:8]


def _open(amount="300", code="network_error", ref=None, customer=None, category="FAILED_PAYMENT"):
    t = _tag()
    body = {
        "customer_id": customer or f"p16_cust_{t}",
        "risk_category": category,
        "external_system": "p16_sys",
        "external_event_id": f"p16_evt_{t}",
        "amount": amount,
        "currency": "INR",
        "raw_payload": {"failure_code": code},
    }
    if ref:
        body["reference_id"] = ref
    r = client.post("/events", json=body, headers=KEY)
    assert r.status_code == 200, r.text
    cid = r.json()["case_id"]
    adv = client.post(f"/cases/{cid}/advance", headers=KEY)
    assert adv.status_code == 200, adv.text
    return cid, adv.json()


def _trace(cid):
    r = client.get(f"/system/cases/{cid}/agent-trace", headers=KEY)
    assert r.status_code == 200, r.text
    return r.json()


def _counts():
    db = SessionLocal()
    try:
        return {
            "cases": db.query(CaseModel).count(),
            "audit": db.query(AuditModel).count(),
            "ledger": db.query(ExecutionAttemptModel).count(),
            "outcomes": db.query(OutcomeEventModel).count(),
        }
    finally:
        db.close()


# ------------------------------------------------------------------ trace identity


def test_1_trace_id_is_a_pure_reversible_function_of_the_case_id():
    assert trace_id_for_case("case_ab12cd34") == "trace_ab12cd34"
    assert trace_id_for_case("case_ab12cd34") == trace_id_for_case("case_ab12cd34")
    assert case_id_candidates_for_trace("trace_ab12cd34")[0] == "case_ab12cd34"
    assert is_trace_id("trace_x") and not is_trace_id("trace_") and not is_trace_id(None)
    assert case_id_candidates_for_trace("not-a-trace") == []


def test_1b_the_same_trace_id_reaches_every_stage_of_a_real_journey():
    cid, adv = _open("20000")  # above the automated limit -> approval path
    assert adv["policy_status"] == "ESCALATE"
    assert (
        client.post(
            f"/cases/{cid}/human-review", json={"decision": "APPROVE", "note": "ok"}, headers=KEY
        ).status_code
        == 200
    )
    ex = client.post(f"/cases/{cid}/execute", headers=KEY).json()
    tid = trace_id_for_case(cid)
    assert ex["trace_id"] == tid and ex["operation_id"] == ex["execution_id"]

    db = SessionLocal()
    try:
        audits = db.query(AuditModel).filter(AuditModel.case_id == cid).all()
        assert audits and all((a.evidence or {}).get("trace_id") == tid for a in audits), [
            ((a.evidence or {}).get("action"), (a.evidence or {}).get("trace_id")) for a in audits
        ]
        ledger = db.query(ExecutionAttemptModel).filter(ExecutionAttemptModel.case_id == cid).all()
        assert ledger and all(row.trace_id == tid for row in ledger)
        case = db.query(CaseModel).filter(CaseModel.case_id == cid).first()
        assert case.policy_decision["trace_id"] == tid and case.execution_record["trace_id"] == tid
    finally:
        db.close()

    t = _trace(cid)
    assert t["trace_id"] == tid
    assert {s["detail"].get("trace_id") for s in t["steps"]} == {tid}


def test_2_policy_decision_carries_its_trace_references():
    cid, _ = _open("20000")
    db = SessionLocal()
    try:
        dec = db.query(CaseModel).filter(CaseModel.case_id == cid).first().policy_decision
    finally:
        db.close()
    assert dec["trace_id"] == trace_id_for_case(cid) and dec["case_id"] == cid
    assert dec["action"] == "RETRY_PAYMENT" and dec["policy_version"]
    assert dec["approval_required"] is True and "FinancialAutomatedLimit" in dec["reason_codes"]
    assert dec["lifecycle_state"] and dec["timestamp"] and dec["status"] == "ESCALATE"
    assert "linked_events" not in dec  # references, not a case copy


def test_2b_permitted_decision_states_it_passed_everything():
    cid, adv = _open("300")
    db = SessionLocal()
    try:
        dec = db.query(CaseModel).filter(CaseModel.case_id == cid).first().policy_decision
    finally:
        db.close()
    assert adv["policy_status"] == "PERMITTED"
    assert dec["reason_codes"] == ["ALL_RULES_PASSED"] and dec["approval_required"] is False


def test_2c_stamping_never_changes_a_policy_verdict():
    from decimal import Decimal

    from application.agent_evaluation import SCENARIOS, build_case
    from application.policy_engine import DeterministicPolicyEngine, PolicyContext

    sc = next(s for s in SCENARIOS if s.name == "payment/suppression")
    case = build_case(sc, 1)
    d = DeterministicPolicyEngine().evaluate(case, PolicyContext(payment_permission="SUPPRESSED"))
    assert Decimal("1") and d.status.value in ("DENIED", "ESCALATE")  # engine verdicts unchanged
    assert d.trace_id == trace_id_for_case(case.case_id) and d.reason_codes


# ------------------------------------------------------------------ immutability / idempotency


def test_3_trace_id_survives_re_advance_reload_approval_execution_and_repeat_reads():
    cid, _ = _open("20000")
    tid = trace_id_for_case(cid)
    t0 = _trace(cid)
    client.post(f"/cases/{cid}/advance", headers=KEY)  # re-advance: a NEW agent operation
    client.post(f"/cases/{cid}/advance", headers=KEY)
    t1 = _trace(cid)
    assert t0["trace_id"] == t1["trace_id"] == tid
    assert t1["stages"]["agent"]["runs"] >= 3
    ops = t1["stages"]["agent"]["operation_ids"]
    assert len(set(ops)) == len(ops)  # each run has its own operation id ...
    assert {s["detail"]["trace_id"] for s in t1["steps"] if s["stage"] == "AGENT"} == {
        tid
    }  # ... one trace
    client.post(
        f"/cases/{cid}/human-review", json={"decision": "APPROVE", "note": "ok"}, headers=KEY
    )
    client.post(f"/cases/{cid}/execute", headers=KEY)
    t2 = _trace(cid)
    assert t2["trace_id"] == tid
    assert _trace(cid) == t2  # reads are repeatable and side-effect free
    assert client.get(f"/system/traces/{tid}", headers=KEY).json()["case_id"] == cid


def test_3b_repeated_verification_does_not_change_the_trace():
    cid, _ = _open("300")
    client.post(f"/cases/{cid}/execute", headers=KEY)
    before = _trace(cid)["trace_id"]
    for _i in range(3):
        client.post(f"/cases/{cid}/verify", json={"external_reference": "x"}, headers=KEY)
    assert _trace(cid)["trace_id"] == before == trace_id_for_case(cid)


def test_4_idempotent_execute_retries_share_one_execution_and_one_trace():
    cid, _ = _open("300")
    runs = [client.post(f"/cases/{cid}/execute", headers=KEY).json() for _ in range(3)]
    assert len({r["execution_id"] for r in runs}) == 1
    assert {r["trace_id"] for r in runs} == {trace_id_for_case(cid)}
    db = SessionLocal()
    try:
        assert (
            db.query(ExecutionAttemptModel).filter(ExecutionAttemptModel.case_id == cid).count()
            == 1
        )
    finally:
        db.close()
    execs = _trace(cid)["stages"]["execution"]["executions"]
    assert len([e for e in execs if e["status"] == "COMPLETED_SIMULATED"]) == 1


def test_4b_a_new_authorized_attempt_is_a_new_operation_in_the_same_trace():
    """Two DIFFERENT policy decisions on one case -> two idempotency keys -> two
    operations, one trace. (The single-decision case above stays one operation.)"""
    from application.action_evaluator import DeterministicActionEvaluator
    from application.agent_evaluation import SCENARIOS, build_case
    from application.agents import AgentOrchestrator
    from application.policy_engine import DeterministicPolicyEngine, PolicyContext
    from domain.models import ActionType

    class _Repo:
        def __init__(self):
            self.slots, self.ledger = set(), []

        def claim_execution_slot(self, key):
            if key in self.slots:
                return False
            self.slots.add(key)
            return True

        def get_execution_record_by_key(self, key):
            return None

        def complete_execution_slot(self, key, rec):
            pass

        def record_execution_attempt(self, **kw):
            self.ledger.append(kw)

    case = build_case(next(s for s in SCENARIOS if s.name == "payment/recoverable_failure"), 0)
    case.recommendation = DeterministicActionEvaluator().evaluate(case)
    repo, orch = _Repo(), AgentOrchestrator()
    recs = []
    for _i in range(2):
        case.policy_decision = DeterministicPolicyEngine().evaluate(
            case, PolicyContext()
        )  # new decision id
        recs.append(orch.execute(case, ActionType.RETRY_PAYMENT, repo=repo))
    assert recs[0].execution_id != recs[1].execution_id
    assert {r.trace_id for r in recs} == {trace_id_for_case(case.case_id)}
    assert {k["trace_id"] for k in repo.ledger} == {trace_id_for_case(case.case_id)}


# ------------------------------------------------------------------ complete trace / no invention


def test_5_a_freshly_ingested_case_reports_missing_stages_as_not_occurred():
    t = client.post(
        "/events",
        json={
            "customer_id": f"p16_c_{_tag()}",
            "risk_category": "FAILED_PAYMENT",
            "external_system": "p16",
            "external_event_id": f"p16_{_tag()}",
            "amount": "50",
            "currency": "INR",
            "raw_payload": {"failure_code": "network_error"},
        },
        headers=KEY,
    ).json()
    tr = _trace(t["case_id"])
    st = tr["stages"]
    for stage in ("agent", "ml_prediction", "policy", "verification", "execution"):
        assert st[stage]["status"] == "NOT_OCCURRED", stage
    assert st["case"]["status"] == "OCCURRED" and st["outcome"]["status"] == "NOT_OCCURRED"
    assert tr["journey"]["executed"] is False and tr["journey"]["label"] == "NOT EXECUTED"


def test_5b_executed_but_unconfirmed_is_never_reported_as_recovered():
    cid, _ = _open("300")
    client.post(f"/cases/{cid}/execute", headers=KEY)
    tr = _trace(cid)
    assert tr["journey"]["label"] == "EXECUTED — RECOVERY NOT YET CONFIRMED"
    assert tr["journey"]["executed"] is True and tr["journey"]["recovery_confirmed"] is False
    assert tr["journey"]["execution_is_simulated"] is True
    assert tr["stages"]["outcome"]["status"] == "PENDING"
    assert tr["stages"]["outcome"]["recovered_amount"] == 0.0
    assert tr["stages"]["execution"]["status"] == "OCCURRED"
    assert [s["stage"] for s in tr["steps"] if s["stage"] in ("AGENT", "POLICY", "EXECUTION")] == [
        "AGENT",
        "POLICY",
        "EXECUTION",
    ]


def test_5c_ml_stage_links_model_probability_erv_and_explanation_reference_without_authority():
    cid, _ = _open("300")
    ml = _trace(cid)["stages"]["ml_prediction"]
    assert ml["model_version"] and ml["prediction_id"] and 0 <= ml["recovery_probability"] <= 1
    assert ml["expected_recoverable_value"] is not None
    assert ml["explanation_reference"]["prediction_id"] == ml["prediction_id"]
    assert "authorized nothing" in ml["note"]


def test_5d_agent_and_policy_stages_carry_their_references():
    cid, _ = _open("300")
    st = _trace(cid)["stages"]
    assert st["agent"]["agent_id"] == "PaymentAgent" and st["agent"]["reason_codes"]
    assert st["policy"]["policy_version"] and st["policy"]["decision"] == "PERMITTED"
    assert (
        st["policy"]["reason_codes"] == ["ALL_RULES_PASSED"]
        and st["policy"]["approval_required"] is False
    )
    assert st["approval"]["status"] == "NOT_REQUIRED"


def test_5e_human_approval_is_bound_to_the_decision_it_reviewed():
    cid, _ = _open("20000")
    assert _trace(cid)["stages"]["approval"]["status"] == "PENDING_APPROVAL"
    assert _trace(cid)["journey"]["label"] == "AWAITING HUMAN APPROVAL — NOT EXECUTED"
    client.post(
        f"/cases/{cid}/human-review", json={"decision": "APPROVE", "note": "fine"}, headers=KEY
    )
    client.post(f"/cases/{cid}/execute", headers=KEY)
    ap = _trace(cid)["stages"]["approval"]
    assert ap["status"] == "OCCURRED" and ap["required"] is True
    assert ap["reviews"][0]["bound_to_current_policy_decision"] is True
    assert "UNKNOWN" in ap["approver_identity"]  # no fabricated approver


def test_5f_policy_denial_and_suppression_show_no_execution_stage():
    cid, _ = _open("300", customer=f"p16_supp_{_tag()}")
    # suppress AFTER opening: revalidation at execute time must block
    client.post(
        f"/customers/{client.get(f'/system/cases/{cid}', headers=KEY).json()['customer_id']}/contact-permissions",
        json={"channel": "all", "status": "SUPPRESSED", "source": "p16_test", "reason": "t"},
        headers=KEY,
    )
    ex = client.post(f"/cases/{cid}/execute", headers=KEY).json()
    assert ex["status"] == "REJECTED"
    tr = _trace(cid)
    assert tr["stages"]["execution"]["status"] == "BLOCKED_OR_FAILED"
    assert tr["stages"]["execution"]["revalidation"] == "BLOCKED"
    assert tr["journey"]["executed"] is False and "BLOCKED" in tr["journey"]["label"]
    assert tr["stages"]["outcome"]["status"] == "NOT_OCCURRED"


# ------------------------------------------------------------------ outcome traceability


def _post_outcome(customer, ref, amount, tag=None):
    tag = tag or _tag()
    r = client.post(
        "/outcomes",
        json={
            "external_system": "p16_out",
            "external_event_id": f"p16_o_{tag}",
            "domain": "FAILED_PAYMENT",
            "outcome_type": "PAYMENT_SETTLED",
            "customer_id": customer,
            "reference_id": ref,
            "amount": amount,
            "currency": "INR",
            "occurred_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            "provenance": "test",
        },
        headers=KEY,
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_6_authoritative_outcome_confirms_recovery_with_provenance_and_amount():
    t = _tag()
    cust, ref = f"p16_oc_{t}", f"p16_ref_{t}"
    cid, _ = _open("300", ref=ref, customer=cust)
    client.post(f"/cases/{cid}/execute", headers=KEY)
    assert _trace(cid)["stages"]["outcome"]["status"] == "PENDING"
    assert _post_outcome(cust, ref, 300)["match_status"] == "MATCHED"
    tr = _trace(cid)
    o = tr["stages"]["outcome"]
    assert o["status"] == "RECOVERY_CONFIRMED" and o["recovered_amount"] == 300.0
    ev = o["matched_events"][0]
    assert ev["provenance"] == "test" and ev["applied_amount"] and ev["outcome_event_id"]
    assert tr["journey"]["recovery_confirmed"] is True
    assert "AUTHORITATIVE" in tr["journey"]["label"]
    assert any(s["stage"] == "OUTCOME" for s in tr["steps"])


def test_6b_an_unmatched_outcome_is_listed_but_never_counted_as_recovery():
    t = _tag()
    cust, ref = f"p16_um_{t}", f"p16_umref_{t}"
    cid, _ = _open("300", ref=ref, customer=cust)
    client.post(f"/cases/{cid}/execute", headers=KEY)
    out = _post_outcome(cust, "some_other_reference", 300)
    assert out["match_status"] == "UNMATCHED"
    o = _trace(cid)["stages"]["outcome"]
    assert o["status"] == "PENDING" and o["recovered_amount"] == 0.0
    assert o["matched_events"] == []


def test_6c_execution_alone_never_creates_recovery_or_outcome_rows():
    before = _counts()["outcomes"]
    cid, _ = _open("300")
    client.post(f"/cases/{cid}/execute", headers=KEY)
    db = SessionLocal()
    try:
        assert db.query(CaseModel).filter(CaseModel.case_id == cid).first().recovered_amount == 0
    finally:
        db.close()
    assert _counts()["outcomes"] == before


# ------------------------------------------------------------------ consistency validator


def _validation(cid):
    db = SessionLocal()
    try:
        return validate_trace(db, cid)
    finally:
        db.close()


def test_7_a_normal_journey_validates_clean():
    cid, _ = _open("20000")
    client.post(
        f"/cases/{cid}/human-review", json={"decision": "APPROVE", "note": "ok"}, headers=KEY
    )
    client.post(f"/cases/{cid}/execute", headers=KEY)
    v = _validation(cid)
    assert v["consistent"] is True and v["counts"]["ERROR"] == 0, v["issues"]
    assert v["read_only"] is True and v["repaired"] is False
    assert client.get(f"/system/traces/{trace_id_for_case(cid)}/validation", headers=KEY).json()[
        "consistent"
    ]


def _plant(cid, fn):
    db = SessionLocal()
    try:
        fn(db)
        db.commit()
    finally:
        db.close()


def _codes(v, severity=None):
    return {i["code"] for i in v["issues"] if severity is None or i["severity"] == severity}


def test_7b_validator_flags_missing_conflicting_and_malformed_trace_ids_without_repairing():
    cid, _ = _open("300")
    tid = trace_id_for_case(cid)

    def plant(db):
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="x",
                to_state="x",
                evidence={"action": "diagnosis"},
            )
        )
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="x",
                to_state="x",
                evidence={"action": "diagnosis", "trace_id": "trace_other"},
            )
        )
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="x",
                to_state="x",
                evidence={"action": "diagnosis", "trace_id": "garbage"},
            )
        )

    _plant(cid, plant)
    v = _validation(cid)
    assert {"MISSING_TRACE_ID", "CONFLICTING_TRACE_ID", "MALFORMED_TRACE_ID"} <= _codes(v)
    assert v["consistent"] is False
    db = SessionLocal()
    try:  # nothing was repaired: the planted rows are exactly as inserted
        rows = [
            (a.evidence or {}).get("trace_id")
            for a in db.query(AuditModel).filter(AuditModel.case_id == cid)
        ]
        assert None in rows and "trace_other" in rows and "garbage" in rows and tid in rows
    finally:
        db.close()


def test_7c_validator_flags_execution_without_policy_authorization():
    cid, _ = _open("20000")  # ESCALATE, never approved

    def plant(db):
        db.add(
            ExecutionAttemptModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                action_type="RETRY_PAYMENT",
                channel="payment",
                status="COMPLETED_SIMULATED",
                execution_id="exec_planted",
                policy_decision_id="pol_nonexistent",
                trace_id=trace_id_for_case(cid),
            )
        )

    _plant(cid, plant)
    assert "EXECUTION_WITHOUT_POLICY_AUTHORIZATION" in _codes(_validation(cid), "ERROR")
    assert "ORPHANED_EXECUTION" in _codes(_validation(cid))


def test_7d_validator_flags_orphaned_policy_decision_and_recovery_without_outcome():
    cid, _ = _open("300")

    def plant(db):
        c = db.query(CaseModel).filter(CaseModel.case_id == cid).first()
        dec = dict(c.policy_decision)
        dec["decision_id"] = "pol_ghost"
        c.policy_decision = dec
        c.recovered_amount = 123

    _plant(cid, plant)
    codes = _codes(_validation(cid), "ERROR")
    assert "ORPHANED_POLICY_DECISION" in codes and "OUTCOME_WITHOUT_CORRELATION" in codes


def test_7e_validator_flags_stage_ordering_violations():
    cid, _ = _open("300")

    def plant(db):
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="x",
                to_state="x",
                evidence={"action": "execution", "trace_id": trace_id_for_case(cid)},
                timestamp=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1),
            )
        )

    _plant(cid, plant)
    assert "STAGE_ORDER_VIOLATION" in _codes(_validation(cid), "ERROR")


def test_7f_bulk_scan_reports_and_does_not_mutate():
    before = _counts()
    db = SessionLocal()
    try:
        rep = validate_all(db, limit=25)
    finally:
        db.close()
    assert rep["read_only"] is True and rep["repaired"] is False and rep["cases_checked"] <= 25
    assert _counts() == before
    assert client.get("/system/trace-consistency?limit=5", headers=KEY).status_code == 200


# ------------------------------------------------------------------ authentication / authorization


@pytest.fixture
def real_auth(monkeypatch):
    """Remove the suite-wide bypass for the read-side dependency and pin the keys."""
    app.dependency_overrides.pop(require_operator, None)
    monkeypatch.setenv("API_KEY", "user-key-123")
    monkeypatch.delenv("OPERATOR_API_KEY", raising=False)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides[require_operator] = lambda: "operator"


PROTECTED = [
    "/system/cases/case_x/agent-trace",
    "/system/traces/trace_x",
    "/system/traces/trace_x/validation",
    "/system/trace-consistency",
    "/system/model-monitoring",
    "/system/models",
    "/system/agent-benchmark",
    "/system/datasets",
    "/system/executions",
    "/system/audit",
    "/cases/case_x/audit",
]


@pytest.mark.parametrize("path", PROTECTED)
def test_8_observability_endpoints_reject_anonymous_and_unknown_credentials(real_auth, path):
    assert real_auth.get(path).status_code == 401
    assert real_auth.get(path, headers={"X-API-Key": "wrong"}).status_code == 401


@pytest.mark.parametrize("path", PROTECTED)
def test_8b_a_nonexistent_resource_does_not_leak_to_anonymous_callers(real_auth, path):
    # unauthenticated must be 401 (never 404): existence is not disclosed
    assert real_auth.get(path).status_code == 401


def test_8c_operator_role_is_separate_from_user_role_when_configured(real_auth, monkeypatch):
    monkeypatch.setenv("OPERATOR_API_KEY", "operator-key-456")
    assert resolve_role("operator-key-456") == "operator"
    assert resolve_role("user-key-123") == "user"
    assert real_auth.get("/system/audit", headers={"X-API-Key": "user-key-123"}).status_code == 403
    assert (
        real_auth.get("/system/audit", headers={"X-API-Key": "operator-key-456"}).status_code == 200
    )
    # ACTION endpoints still use the existing mechanism: the user key may act...
    assert real_auth.get("/health").status_code == 200


def test_8d_without_a_separate_operator_key_api_key_holders_are_operators(real_auth):
    assert resolve_role("user-key-123") == "operator"
    assert real_auth.get("/system/audit", headers={"X-API-Key": "user-key-123"}).status_code == 200


def test_8e_authenticated_lookups_of_missing_things_are_404_or_400(real_auth):
    h = {"X-API-Key": "user-key-123"}
    assert real_auth.get("/system/cases/case_missing/agent-trace", headers=h).status_code == 404
    assert real_auth.get("/system/traces/trace_missing", headers=h).status_code == 404
    assert real_auth.get("/system/traces/not-a-trace", headers=h).status_code == 400


def test_8f_every_operator_route_is_read_only():
    from fastapi.routing import APIRoute

    from api.system_router import router as system_router

    guarded = []
    for r in [*app.routes, *system_router.routes]:
        if isinstance(r, APIRoute) and any(
            d.call is require_operator for d in r.dependant.dependencies
        ):
            guarded.append(r)
    assert len(guarded) >= len(PROTECTED) - 1
    for r in guarded:
        assert r.methods <= {"GET", "HEAD"}, (r.path, r.methods)


# ------------------------------------------------------------------ no side effects


def test_9_observability_reads_never_mutate_anything():
    cid, _ = _open("300")
    client.post(f"/cases/{cid}/execute", headers=KEY)
    tid = trace_id_for_case(cid)
    db = SessionLocal()
    try:
        snap = db.query(CaseModel).filter(CaseModel.case_id == cid).first()
        before_case = (
            snap.current_state,
            snap.recovered_amount,
            str(snap.policy_decision),
            str(snap.execution_record),
        )
    finally:
        db.close()
    before = _counts()
    for path in (
        f"/system/cases/{cid}/agent-trace",
        f"/system/traces/{tid}",
        f"/system/traces/{tid}/validation",
        "/system/trace-consistency?limit=10",
        "/system/model-monitoring",
        "/system/agent-benchmark",
        "/system/audit",
        "/system/executions",
        f"/cases/{cid}/audit",
        "/system/models",
        "/system/datasets",
    ):
        assert client.get(path, headers=KEY).status_code == 200, path
    assert _counts() == before
    db = SessionLocal()
    try:
        snap = db.query(CaseModel).filter(CaseModel.case_id == cid).first()
        assert (
            snap.current_state,
            snap.recovered_amount,
            str(snap.policy_decision),
            str(snap.execution_record),
        ) == before_case
    finally:
        db.close()


def test_9b_evaluation_cannot_touch_the_operational_store():
    from evaluation.runner import EvaluationRunner, OperationalStoreError

    before = _counts()
    assert client.post("/evaluation/run", headers=KEY).status_code == 200
    assert _counts() == before
    db = SessionLocal()
    try:
        with pytest.raises(OperationalStoreError):
            EvaluationRunner(db)
    finally:
        db.close()


# ------------------------------------------------------------------ redaction


def test_10_redact_removes_secrets_and_pii_but_keeps_diagnostics():
    out = redact(
        {
            "api_key": "sk-abcdef1234567890abcd",
            "Authorization": "Bearer abc.def.ghi",
            "nested": {
                "password": "hunter2",
                "note": "card 4111 1111 1111 1111 owner a.b@example.com",
            },
            "token_value": "zzz",
            "decision": "PERMITTED",
            "amount": "300.0000",
            "trace_id": "trace_ab12cd34",
            "list": ["call Bearer tok123", "fine"],
        }
    )
    assert out["api_key"] == "[REDACTED]" and out["Authorization"] == "[REDACTED]"
    assert out["nested"]["password"] == "[REDACTED]" and out["token_value"] == "[REDACTED]"
    assert "4111" not in out["nested"]["note"] and "example.com" not in out["nested"]["note"]
    assert "[REDACTED_PAN]" in out["nested"]["note"] and "[EMAIL]" in out["nested"]["note"]
    assert (
        out["decision"] == "PERMITTED"
        and out["amount"] == "300.0000"
        and out["trace_id"] == "trace_ab12cd34"
    )
    assert "tok123" not in out["list"][0] and out["list"][1] == "fine"


def test_10b_trace_responses_redact_planted_secrets():
    cid, _ = _open("300")

    def plant(db):
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="a",
                to_state="a",
                evidence={
                    "action": "human_review",
                    "decision": "APPROVE",
                    "trace_id": trace_id_for_case(cid),
                    "api_key": "sk-supersecretkey1234567890",
                    "approved_decision_id": "pol_x",
                    "note": "paid with 4242 4242 4242 4242, contact john.doe@corp.com, Bearer abc123token",
                },
            )
        )

    _plant(cid, plant)
    body = client.get(f"/system/cases/{cid}/agent-trace", headers=KEY).text
    for leaked in ("sk-supersecretkey", "4242 4242", "john.doe@corp.com", "abc123token"):
        assert leaked not in body, leaked
    assert "PERMITTED" in body or "ESCALATE" in body  # diagnostics survive


def test_10c_customer_identifiers_that_are_emails_are_masked():
    cid, _ = _open("300", customer="someone.private@example.com")
    body = client.get(f"/system/cases/{cid}/agent-trace", headers=KEY).text
    assert "someone.private@example.com" not in body


# ------------------------------------------------------------------ failure safety


def test_11_partial_and_malformed_history_never_crashes_or_invents_stages():
    cid, _ = _open("300")

    def plant(db):
        db.add(
            AuditModel(
                id=uuid.uuid4().hex, case_id=cid, from_state=None, to_state=None, evidence=None
            )
        )
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="a",
                to_state="a",
                evidence={"action": "totally_unknown_action"},
            )
        )
        db.add(
            ExecutionAttemptModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                action_type="RETRY_PAYMENT",
                channel="payment",
                status="FAILED",
                execution_id="exec_failed_planted",
                policy_decision_id="pol_x",
                trace_id=None,
            )
        )

    _plant(cid, plant)
    tr = _trace(cid)
    ex = tr["stages"]["execution"]
    assert ex["status"] == "BLOCKED_OR_FAILED"
    assert any(e["status"] == "FAILED" for e in ex["executions"])
    assert (
        tr["journey"]["executed"] is False and tr["stages"]["outcome"]["status"] == "NOT_OCCURRED"
    )
    assert all(s["action"] != "totally_unknown_action" for s in tr["steps"])


def test_11b_a_failed_agent_run_is_shown_as_fallback_not_as_a_recommendation():
    cid, _ = _open("300")

    def plant(db):
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="a",
                to_state="a",
                evidence={
                    "action": "agent_recommendation",
                    "status": "FALLBACK",
                    "agent_id": "PaymentAgent",
                    "trace_id": trace_id_for_case(cid),
                    "operation_id": "agentrun_x",
                    "error": "timeout",
                },
            )
        )

    _plant(cid, plant)
    assert _trace(cid)["stages"]["agent"]["recommendation_status"] == "FALLBACK"


def test_11c_reconstruction_of_an_unknown_case_is_none():
    db = SessionLocal()
    try:
        assert reconstruct_trace(db, "case_does_not_exist") is None
        assert load_trace_data(db, "case_does_not_exist") is None
    finally:
        db.close()


# ------------------------------------------------------------------ performance / isolation


def _count_selects(fn):
    n = {"q": 0}

    def hook(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            n["q"] += 1

    event.listen(engine, "before_cursor_execute", hook)
    try:
        fn()
    finally:
        event.remove(engine, "before_cursor_execute", hook)
    return n["q"]


def test_12_trace_reconstruction_uses_a_constant_number_of_queries():
    small, _ = _open("300")
    big, _ = _open("300")

    def plant(db):
        for i in range(150):
            db.add(
                AuditModel(
                    id=uuid.uuid4().hex,
                    case_id=big,
                    from_state="a",
                    to_state="a",
                    evidence={
                        "action": "agent_recommendation",
                        "trace_id": trace_id_for_case(big),
                        "operation_id": f"op{i}",
                    },
                )
            )
        for i in range(60):
            db.add(
                ExecutionAttemptModel(
                    id=uuid.uuid4().hex,
                    case_id=big,
                    action_type="RETRY_PAYMENT",
                    channel="payment",
                    status="FAILED",
                    execution_id=f"exec_{i}",
                    policy_decision_id="pol_x",
                    trace_id=trace_id_for_case(big),
                )
            )

    _plant(big, plant)

    def run(cid):
        db = SessionLocal()
        try:
            return reconstruct_trace(db, cid)
        finally:
            db.close()

    q_small, q_big = _count_selects(lambda: run(small)), _count_selects(lambda: run(big))
    assert q_small == q_big <= 6, (q_small, q_big)
    assert len(run(big)["stages"]["execution"]["executions"]) >= 60


def _run_pytest(*args):
    # The child session owns (and deletes) its database: give it a private file so it can never
    # drop the parent session's schema mid-run.
    import os
    import tempfile

    child_db = os.path.join(tempfile.mkdtemp(prefix="rc_child_db_"), "child.db").replace("\\", "/")
    env = {**os.environ, "RECOVERCHAIN_TEST_DATABASE_URL": "sqlite:///" + child_db}
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider", "--no-header"],
        env=env,
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_13_test_evaluation_passes_on_its_own():
    r = _run_pytest("tests/test_evaluation.py")
    assert r.returncode == 0, r.stdout[-1500:]


def test_13b_evaluation_before_api_tests_no_longer_breaks_them():
    """The Phase 15 finding: this order used to fail with 'no such table'."""
    r = _run_pytest("tests/test_evaluation.py", "tests/test_phase36_workflow_hardening.py")
    assert r.returncode == 0, r.stdout[-1500:]


def test_13c_test_and_operational_databases_are_distinct_files():
    from infrastructure.db import DATABASE_URL

    assert "_pytest_recoverchain.db" in DATABASE_URL
    assert "recoverchain.db" != Path(DATABASE_URL.split("///")[-1]).name
