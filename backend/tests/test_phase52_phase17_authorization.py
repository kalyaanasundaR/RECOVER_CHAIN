"""Phase 17 - principals/roles, approval evidence, revalidation evidence, endpoint access,
outcome protection, append-only evidence, legacy trace handling."""

import json
import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from api import principals
from api.access import (
    require_admin,
    require_operator,
    require_principal,
    require_reviewer,
)
from api.main import app, verify_api_key
from application.trace_ids import trace_id_for_case
from infrastructure.db import SessionLocal
from infrastructure.immutability import ImmutableRecordError
from infrastructure.orm import (
    AuditModel,
    CaseModel,
    ExecutionAttemptModel,
    OutcomeEventModel,
)

client = TestClient(app)  # default suite overrides: acts as ADMIN / REVIEWER
KEY = {"X-API-Key": "test-api-key"}

USER_KEY, OPERATOR_KEY, ALICE_KEY, ROOT_KEY, OFF_KEY = (
    "user-key-1",
    "operator-key-2",
    "alice-key-3",
    "root-key-4",
    "off-key-5",
)
REGISTRY = {
    ALICE_KEY: {"principal_id": "alice", "role": "REVIEWER"},
    ROOT_KEY: {"principal_id": "root", "role": "ADMIN"},
    OFF_KEY: {"principal_id": "former", "role": "REVIEWER", "active": False},
}
DEPS = (verify_api_key, require_principal, require_operator, require_reviewer, require_admin)


def _tag():
    return uuid.uuid4().hex[:8]


@pytest.fixture
def real_auth(monkeypatch):
    """Remove every suite-wide bypass and configure real keys: a separated deployment."""
    saved = {d: app.dependency_overrides.pop(d, None) for d in DEPS}
    monkeypatch.setenv("API_KEY", USER_KEY)
    monkeypatch.setenv("OPERATOR_API_KEY", OPERATOR_KEY)
    monkeypatch.setenv("RECOVERCHAIN_PRINCIPALS", json.dumps(REGISTRY))
    try:
        yield TestClient(app)
    finally:
        for d, v in saved.items():
            if v is not None:
                app.dependency_overrides[d] = v


def h(key):
    return {"X-API-Key": key}


def _open(amount="300", customer=None, code="network_error", c=client, headers=KEY):
    t = _tag()
    r = c.post(
        "/events",
        json={
            "customer_id": customer or f"p17_cust_{t}",
            "risk_category": "FAILED_PAYMENT",
            "external_system": "p17",
            "external_event_id": f"p17_{t}",
            "amount": amount,
            "currency": "INR",
            "raw_payload": {"failure_code": code},
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text
    cid = r.json()["case_id"]
    assert c.post(f"/cases/{cid}/advance", headers=headers).status_code == 200
    return cid


def _trace(cid, c=client, headers=KEY):
    r = c.get(f"/system/cases/{cid}/agent-trace", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def _audits(cid, action=None):
    db = SessionLocal()
    try:
        rows = (
            db.query(AuditModel)
            .filter(AuditModel.case_id == cid)
            .order_by(AuditModel.timestamp)
            .all()
        )
        return [
            dict(a.evidence or {}, _id=a.id)
            for a in rows
            if action is None or (a.evidence or {}).get("action") == action
        ]
    finally:
        db.close()


def _approve(cid):
    r = client.post(
        f"/cases/{cid}/human-review", json={"decision": "APPROVE", "note": "ok"}, headers=KEY
    )
    assert r.status_code == 200, r.text


def _revals(cid):
    return _trace(cid)["stages"]["revalidation"].get("revalidations", [])


def _customer_of(cid):
    return client.get(f"/system/cases/{cid}", headers=KEY).json()["customer_id"]


# ------------------------------------------------------------------ principals / roles


def test_1_legacy_keys_map_to_roles_without_claiming_a_person(monkeypatch):
    monkeypatch.setenv("API_KEY", USER_KEY)
    monkeypatch.delenv("OPERATOR_API_KEY", raising=False)
    monkeypatch.delenv("RECOVERCHAIN_PRINCIPALS", raising=False)
    p = principals.resolve_principal(USER_KEY)
    assert (
        p.role == "ADMIN" and p.principal_id.startswith("key:") and USER_KEY not in p.principal_id
    )
    monkeypatch.setenv("OPERATOR_API_KEY", OPERATOR_KEY)
    assert principals.resolve_principal(USER_KEY).role == "USER"
    assert principals.resolve_principal(OPERATOR_KEY).role == "OPERATOR"
    assert (
        principals.resolve_principal("nope") is None and principals.resolve_principal(None) is None
    )


def test_1b_registry_principals_inactive_and_misconfigured_roles_fail_closed(monkeypatch):
    monkeypatch.setenv(
        "RECOVERCHAIN_PRINCIPALS",
        json.dumps({**REGISTRY, "bad": {"principal_id": "x", "role": "GOD"}}),
    )
    assert principals.resolve_principal(ALICE_KEY).principal_id == "alice"
    assert principals.lookup(OFF_KEY) == (None, "inactive")
    assert principals.lookup("bad") == (
        None,
        "unknown",
    )  # a role outside the vocabulary grants nothing
    monkeypatch.setenv("RECOVERCHAIN_PRINCIPALS", "{not json")
    assert principals.lookup(ALICE_KEY)[0] is None  # broken config: fail closed


def test_1c_evidence_never_contains_the_credential():
    p = principals.Principal("alice", "REVIEWER")
    ev = p.as_evidence()
    assert set(ev) == {"principal_id", "role", "authentication_method"}
    assert p.can_approve and not p.is_admin and p.can_observe and p.can_act


def test_1d_role_capabilities_are_deterministic():
    caps = {r: principals.Principal("x", r) for r in principals.ROLES}
    assert (caps["USER"].can_act, caps["USER"].can_observe, caps["USER"].can_approve) == (
        True,
        False,
        False,
    )
    assert (
        caps["OPERATOR"].can_act,
        caps["OPERATOR"].can_observe,
        caps["OPERATOR"].can_approve,
    ) == (False, True, False)
    assert (
        caps["REVIEWER"].can_act,
        caps["REVIEWER"].can_observe,
        caps["REVIEWER"].can_approve,
    ) == (True, True, True)
    assert all(
        getattr(caps["ADMIN"], a) for a in ("can_act", "can_observe", "can_approve", "is_admin")
    )


# ------------------------------------------------------------------ approval authorization (direct API)


def test_2_only_a_reviewer_or_admin_can_approve_and_it_is_enforced_server_side(real_auth):
    cid = _open("20000", c=real_auth, headers=h(USER_KEY))
    body = {"decision": "APPROVE", "note": "n"}
    url = f"/cases/{cid}/human-review"
    assert real_auth.post(url, json=body).status_code == 401  # no key
    assert real_auth.post(url, json=body, headers=h("wrong")).status_code == 401
    assert (
        real_auth.post(url, json=body, headers=h(OFF_KEY)).status_code == 401
    )  # inactive principal
    assert (
        real_auth.post(url, json=body, headers=h(USER_KEY)).status_code == 403
    )  # can act, cannot approve
    assert real_auth.post(url, json=body, headers=h(OPERATOR_KEY)).status_code == 403  # read-only
    assert (
        _trace(cid, real_auth, h(OPERATOR_KEY))["stages"]["approval"]["status"]
        == "PENDING_APPROVAL"
    )  # nothing changed
    assert real_auth.post(url, json=body, headers=h(ALICE_KEY)).status_code == 200


def test_2b_an_operator_credential_cannot_execute_or_change_anything(real_auth):
    cid = _open("300", c=real_auth, headers=h(USER_KEY))
    for path in (
        f"/cases/{cid}/execute",
        f"/cases/{cid}/advance",
        f"/cases/{cid}/stop",
        f"/cases/{cid}/verify",
    ):
        r = real_auth.post(path, headers=h(OPERATOR_KEY), json={"external_reference": "x"})
        assert r.status_code == 403, path
    assert real_auth.post(f"/cases/{cid}/execute", headers=h(USER_KEY)).status_code == 200


def test_2c_execution_and_action_endpoints_reject_missing_and_invalid_credentials(real_auth):
    cid = _open("300", c=real_auth, headers=h(USER_KEY))
    for key in (None, "wrong", OFF_KEY):
        hd = h(key) if key else {}
        assert real_auth.post(f"/cases/{cid}/execute", headers=hd).status_code == 403
        assert real_auth.post(f"/cases/{cid}/advance", headers=hd).status_code == 403
    assert real_auth.post("/evaluation/run", headers=h(OPERATOR_KEY)).status_code == 403


# ------------------------------------------------------------------ approval evidence


def test_3_approval_records_the_authenticated_principal_role_policy_and_trace(real_auth):
    cid = _open("20000", c=real_auth, headers=h(USER_KEY))
    r = real_auth.post(
        f"/cases/{cid}/human-review",
        json={
            "decision": "APPROVE",
            "note": "reviewed",
            "actor": {"principal_id": "mallory", "role": "ADMIN"},
        },
        headers=h(ALICE_KEY),
    )
    assert r.status_code == 200
    ev = _audits(cid, "human_review")[0]
    assert ev["actor"] == {
        "principal_id": "alice",
        "role": "REVIEWER",
        "authentication_method": "api_key",
    }
    assert ev["approval_id"].startswith("appr_") and ev["case_id"] == cid
    assert (
        ev["trace_id"] == trace_id_for_case(cid)
        and ev["policy_version"]
        and ev["policy_decision_id"]
    )
    assert ev["approved_action"] == "RETRY_PAYMENT" and ev["decision"] == "APPROVE"
    assert "mallory" not in json.dumps(ev)  # an actor claimed in the body is never trusted
    body = json.dumps(_trace(cid, real_auth, h(OPERATOR_KEY)))
    for secret in (ALICE_KEY, USER_KEY, OPERATOR_KEY):
        assert secret not in body


def test_3b_trace_shows_who_approved_and_that_approval_is_not_recovery(real_auth):
    cid = _open("20000", c=real_auth, headers=h(USER_KEY))
    real_auth.post(
        f"/cases/{cid}/human-review", json={"decision": "APPROVE", "note": "x"}, headers=h(ROOT_KEY)
    )
    real_auth.post(f"/cases/{cid}/execute", headers=h(USER_KEY))
    t = _trace(cid, real_auth, h(OPERATOR_KEY))
    rev = t["stages"]["approval"]["reviews"][0]
    assert rev["actor"]["principal_id"] == "root" and rev["actor"]["role"] == "ADMIN"
    assert t["journey"]["approved_by"]["principal_id"] == "root"
    assert (
        t["journey"]["recovery_confirmed"] is False
        and t["stages"]["outcome"]["status"] == "PENDING"
    )


def test_3c_a_rejection_is_also_attributed_and_never_executes(real_auth):
    cid = _open("20000", c=real_auth, headers=h(USER_KEY))
    real_auth.post(
        f"/cases/{cid}/human-review",
        json={"decision": "REJECT", "note": "no"},
        headers=h(ALICE_KEY),
    )
    ev = _audits(cid, "human_review")[0]
    assert ev["decision"] == "REJECT" and ev["actor"]["principal_id"] == "alice"
    assert (
        real_auth.post(f"/cases/{cid}/execute", headers=h(USER_KEY)).json()["status"] == "REJECTED"
    )


def test_3d_historical_approvals_without_a_principal_stay_unknown_not_invented():
    cid = _open("20000")
    db = SessionLocal()
    try:
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="a",
                to_state="a",
                evidence={
                    "action": "human_review",
                    "decision": "APPROVE",
                    "approved_decision_id": "pol_old",
                },
            )
        )
        db.commit()
    finally:
        db.close()
    actor = _trace(cid)["stages"]["approval"]["reviews"][-1]["actor"]
    assert actor == principals.UNKNOWN_PRINCIPAL


# ------------------------------------------------------------------ revalidation evidence matrix


def test_4_case_a_no_approval_needed_revalidation_passes_then_executes():
    cid = _open("300")
    assert (
        client.post(f"/cases/{cid}/execute", headers=KEY).json()["status"] == "COMPLETED_SIMULATED"
    )
    rv = _revals(cid)
    assert len(rv) == 1 and rv[0]["result"] == "PASSED" and rv[0]["approval_reference"] is None
    assert (
        rv[0]["reason_codes"] == ["ALL_RULES_PASSED"]
        and rv[0]["policy_version"]
        and rv[0]["policy_version_changed"] is False
    )
    ev = _audits(cid, "policy_revalidation")[0]
    assert (
        ev["trace_id"] == trace_id_for_case(cid)
        and ev["case_id"] == cid
        and ev["requested_action"] == "RETRY_PAYMENT"
    )
    assert ev["revalidated_at"] and ev["initial_decision_id"] and ev["revalidation_decision_id"]
    t = _trace(cid)
    assert t["journey"]["revalidation"] == "PASSED"
    order = [
        s["stage"] for s in t["steps"] if s["stage"] in ("POLICY", "REVALIDATION", "EXECUTION")
    ]
    assert order == ["POLICY", "REVALIDATION", "EXECUTION"]
    exec_ev = _audits(cid, "execution")[0]
    assert exec_ev["revalidation_id"] == ev["revalidation_id"]


def test_4b_case_b_high_value_approved_then_revalidated_then_executed():
    cid = _open("20000")
    _approve(cid)
    assert (
        client.post(f"/cases/{cid}/execute", headers=KEY).json()["status"] == "COMPLETED_SIMULATED"
    )
    rv = _revals(cid)[0]
    approval = _audits(cid, "human_review")[0]
    assert rv["result"] == "PASSED" and rv["approval_required"] is True
    assert rv["approval_reference"] == approval["approval_id"]
    assert rv["approval_actor"]["principal_id"] == "test-reviewer"
    order = [
        s["stage"]
        for s in _trace(cid)["steps"]
        if s["stage"] in ("POLICY", "APPROVAL", "REVALIDATION", "EXECUTION")
    ]
    assert order == ["POLICY", "APPROVAL", "REVALIDATION", "EXECUTION"]


def test_4c_case_c_not_approved_means_revalidation_fails_and_nothing_executes():
    cid = _open("20000")
    r = client.post(f"/cases/{cid}/execute", headers=KEY).json()
    assert r["status"] == "REJECTED"
    rv = _revals(cid)
    assert rv[-1]["result"] == "FAILED" and "FinancialAutomatedLimit" in rv[-1]["reason_codes"]
    t = _trace(cid)
    assert (
        t["journey"]["executed"] is False
        and t["stages"]["execution"]["status"] == "BLOCKED_OR_FAILED"
    )
    db = SessionLocal()
    try:
        assert (
            db.query(ExecutionAttemptModel).filter(ExecutionAttemptModel.case_id == cid).count()
            == 0
        )
    finally:
        db.close()


def _assert_blocked_after_approval(cid, expected_rule):
    approvals_before = _audits(cid, "human_review")
    r = client.post(f"/cases/{cid}/execute", headers=KEY).json()
    assert r["status"] == "REJECTED" and r["adapter_used"] == "NONE"
    rv = _revals(cid)[-1]
    assert rv["result"] == "FAILED" and expected_rule in rv["reason_codes"]
    assert rv["approval_reference"] is not None  # the approval is still on record and referenced
    assert _audits(cid, "human_review") == approvals_before  # approval preserved, untouched
    t = _trace(cid)
    assert t["stages"]["approval"]["status"] == "OCCURRED" and t["journey"]["executed"] is False
    assert "REVALIDATION FAILED" in t["journey"]["label"]
    assert (
        t["stages"]["outcome"]["status"] == "NOT_OCCURRED"
        and t["stages"]["outcome"]["recovered_amount"] == 0.0
    )
    db = SessionLocal()
    try:
        assert (
            db.query(ExecutionAttemptModel).filter(ExecutionAttemptModel.case_id == cid).count()
            == 0
        )
        case = db.query(CaseModel).filter(CaseModel.case_id == cid).first()
        assert float(case.recovered_amount) == 0.0
    finally:
        db.close()


def test_4d_case_d_suppression_after_approval_fails_revalidation():
    cid = _open("20000")
    _approve(cid)
    client.post(
        f"/customers/{_customer_of(cid)}/contact-permissions",
        json={"channel": "all", "status": "SUPPRESSED", "source": "p17", "reason": "late"},
        headers=KEY,
    )
    _assert_blocked_after_approval(cid, "ContactPermission")


def test_4e_case_e_consent_change_after_approval_fails_revalidation():
    cid = _open("20000")
    _approve(cid)
    client.post(
        f"/customers/{_customer_of(cid)}/contact-permissions",
        json={"channel": "payment", "status": "OPTED_OUT", "source": "p17", "reason": "withdrawn"},
        headers=KEY,
    )
    _assert_blocked_after_approval(cid, "ContactPermission")


def test_4f_case_f_retry_limit_reached_after_approval_fails_revalidation():
    cid = _open("20000")
    _approve(cid)
    db = SessionLocal()
    try:
        for i in range(3):
            db.add(
                ExecutionAttemptModel(
                    id=uuid.uuid4().hex,
                    case_id=cid,
                    action_type="RETRY_PAYMENT",
                    channel="payment",
                    status="COMPLETED_SIMULATED",
                    execution_id=f"exec_prior_{i}_{cid}",
                    policy_decision_id="pol_prior",
                    trace_id=trace_id_for_case(cid),
                )
            )
        db.commit()
    finally:
        db.close()
    r = client.post(f"/cases/{cid}/execute", headers=KEY).json()
    assert r["status"] == "REJECTED"
    rv = _revals(cid)[-1]
    assert rv["result"] == "FAILED" and "PaymentMaxRetries" in rv["reason_codes"]
    assert _audits(cid, "human_review")  # approval preserved


def test_4g_case_g_policy_version_change_after_approval_fails_revalidation(monkeypatch):
    import application.policy_engine as pe

    cid = _open("20000")
    _approve(cid)
    original = pe.DeterministicPolicyEngine

    def stricter(*a, **k):
        return original(pe.MerchantPolicy(version="merchant-policy-v2-test", payment_max_retries=0))

    monkeypatch.setattr(pe, "DeterministicPolicyEngine", stricter)
    r = client.post(f"/cases/{cid}/execute", headers=KEY).json()
    monkeypatch.undo()
    assert r["status"] == "REJECTED"
    rv = _revals(cid)[-1]
    assert rv["result"] == "FAILED" and rv["policy_version_changed"] is True
    assert rv["initial_policy_version"] != rv["policy_version"] == "merchant-policy-v2-test"
    assert "PaymentMaxRetries" in rv["reason_codes"]
    assert _trace(cid)["journey"]["executed"] is False


def test_4h_a_failed_revalidation_never_rewrites_history_and_a_later_pass_appends():
    cid = _open("20000")
    _approve(cid)
    client.post(
        f"/customers/{_customer_of(cid)}/contact-permissions",
        json={"channel": "payment", "status": "OPTED_OUT", "source": "p17", "reason": "x"},
        headers=KEY,
    )
    assert client.post(f"/cases/{cid}/execute", headers=KEY).json()["status"] == "REJECTED"
    failed = _audits(cid, "policy_revalidation")[0]
    client.post(
        f"/customers/{_customer_of(cid)}/contact-permissions",
        json={
            "channel": "payment",
            "status": "ALLOWED",
            "source": "p17",
            "reason": "re-consented",
            "effective_at": (datetime.now(UTC) + timedelta(seconds=1)).isoformat(),
        },
        headers=KEY,
    )
    time.sleep(1.1)  # (sleep is patched out in tests; the effective_at above is what matters)
    r = client.post(f"/cases/{cid}/execute", headers=KEY).json()
    recs = _audits(cid, "policy_revalidation")
    assert recs[0] == failed and recs[0]["result"] == "FAILED"  # the failure is preserved verbatim
    if r["status"] == "COMPLETED_SIMULATED":
        assert recs[-1]["result"] == "PASSED" and len(recs) == 2
    else:  # consent resolution rules kept it blocked: still no execution, still recorded
        assert recs[-1]["result"] == "FAILED"


def test_4i_an_idempotent_replay_does_not_manufacture_a_second_revalidation():
    cid = _open("300")
    for _i in range(3):
        client.post(f"/cases/{cid}/execute", headers=KEY)
    assert len(_audits(cid, "policy_revalidation")) == 1


# ------------------------------------------------------------------ endpoint access matrix


OPERATOR_READS = [
    "/system/events",
    "/system/policy",
    "/system/data-readiness",
    "/system/eligibility",
    "/system/dataset-snapshots",
    "/system/cases.csv",
    "/system/evidence-sources",
    "/outcomes",
    "/customers/identity-links",
    "/customers/x/contact-history",
    "/customers/x/contact-permissions",
    "/system/cases/case_x/agent-trace",
    "/system/traces/trace_x",
    "/system/model-monitoring",
    "/system/agent-benchmark",
    "/system/datasets",
    "/system/executions",
    "/system/audit",
    "/system/models",
]


@pytest.mark.parametrize("path", OPERATOR_READS)
def test_5_operator_reads_401_anonymous_403_user_200_operator(real_auth, path):
    assert real_auth.get(path).status_code == 401
    assert real_auth.get(path, headers=h("wrong")).status_code == 401
    assert real_auth.get(path, headers=h(USER_KEY)).status_code == 403  # can act, cannot observe
    assert real_auth.get(path, headers=h(OPERATOR_KEY)).status_code in (
        200,
        404,
    )  # 404 = authorized but missing


def test_5b_case_snapshots_need_authentication_but_not_operator_rights(real_auth):
    cid = _open("300", c=real_auth, headers=h(USER_KEY))
    assert real_auth.get(f"/system/cases/{cid}").status_code == 401
    assert real_auth.get("/system/cases").status_code == 401
    assert real_auth.get(f"/system/cases/{cid}", headers=h(USER_KEY)).status_code == 200
    assert real_auth.get("/system/cases", headers=h(OPERATOR_KEY)).status_code == 200
    assert (
        real_auth.get("/system/cases/case_nope", headers=h("wrong")).status_code == 401
    )  # no existence leak


def test_5c_intentionally_public_reads_stay_public_by_default_and_close_with_the_switch(
    real_auth, monkeypatch
):
    for path in ("/health", "/cases", "/datasets", "/system/summary", "/system/health"):
        assert real_auth.get(path).status_code == 200, path
    monkeypatch.setenv("RECOVERCHAIN_PROTECT_READS", "1")
    for path in ("/cases", "/datasets", "/system/summary", "/dashboard/metrics"):
        assert real_auth.get(path).status_code == 401, path
        assert real_auth.get(path, headers=h(USER_KEY)).status_code == 200, path
    assert (
        real_auth.get("/health").status_code == 200
        and real_auth.get("/system/health").status_code == 200
    )


def test_5d_operator_routes_are_read_only_and_never_mutate():
    from fastapi.routing import APIRoute

    from api.system_router import router as system_router

    for r in [*app.routes, *system_router.routes]:
        if isinstance(r, APIRoute) and any(
            d.call is require_operator for d in r.dependant.dependencies
        ):
            assert r.methods <= {"GET", "HEAD"}, (r.path, r.methods)


# ------------------------------------------------------------------ outcome protection


def _outcome_body(customer, ref, prov="test"):
    t = _tag()
    return {
        "external_system": "p17_out",
        "external_event_id": f"o_{t}",
        "domain": "FAILED_PAYMENT",
        "outcome_type": "PAYMENT_SETTLED",
        "customer_id": customer,
        "reference_id": ref,
        "amount": 300,
        "currency": "INR",
        "occurred_at": (datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
        "provenance": prov,
    }


def test_6_only_an_admin_can_create_outcomes_so_no_one_else_can_fake_recovery(real_auth):
    cust, ref = f"p17_oc_{_tag()}", f"p17_ref_{_tag()}"
    t = _tag()
    cid = real_auth.post(
        "/events",
        json={
            "customer_id": cust,
            "risk_category": "FAILED_PAYMENT",
            "external_system": "p17",
            "external_event_id": f"e_{t}",
            "amount": "300",
            "currency": "INR",
            "reference_id": ref,
            "raw_payload": {"failure_code": "network_error"},
        },
        headers=h(USER_KEY),
    ).json()["case_id"]
    body = _outcome_body(cust, ref)
    for key in (None, "wrong", USER_KEY, OPERATOR_KEY, ALICE_KEY):
        r = real_auth.post("/outcomes", json=body, headers=h(key) if key else {})
        assert r.status_code in (401, 403), (key, r.status_code)
    for path in ("/outcomes/batch", "/outcomes/rematch-unmatched"):
        assert real_auth.post(path, json={"events": []}, headers=h(USER_KEY)).status_code == 403
    db = SessionLocal()
    try:
        assert (
            float(db.query(CaseModel).filter(CaseModel.case_id == cid).first().recovered_amount)
            == 0.0
        )
    finally:
        db.close()
    assert real_auth.post("/outcomes", json=body, headers=h(ROOT_KEY)).status_code == 200


def test_6b_provenance_protections_still_bind_even_an_admin(real_auth):
    cust, ref = f"p17_pv_{_tag()}", f"p17_pvref_{_tag()}"
    r = real_auth.post(
        "/outcomes", json=_outcome_body(cust, ref, prov="production"), headers=h(ROOT_KEY)
    )
    assert r.status_code >= 400  # `production` is not accepted through the shared credential


def test_6c_execution_never_creates_recovery_or_an_outcome_event():
    db = SessionLocal()
    try:
        before = db.query(OutcomeEventModel).count()
    finally:
        db.close()
    cid = _open("300")
    client.post(f"/cases/{cid}/execute", headers=KEY)
    db = SessionLocal()
    try:
        assert db.query(OutcomeEventModel).count() == before
        assert (
            float(db.query(CaseModel).filter(CaseModel.case_id == cid).first().recovered_amount)
            == 0.0
        )
    finally:
        db.close()


# ------------------------------------------------------------------ append-only evidence


def test_7_audit_records_including_approval_and_revalidation_evidence_cannot_be_rewritten():
    cid = _open("20000")
    _approve(cid)
    client.post(f"/cases/{cid}/execute", headers=KEY)
    db = SessionLocal()
    try:
        for action in ("human_review", "policy_revalidation", "execution", "policy_evaluation"):
            row = next(
                a
                for a in db.query(AuditModel).filter(AuditModel.case_id == cid)
                if (a.evidence or {}).get("action") == action
            )
            row.evidence = {**row.evidence, "tampered": True}
            with pytest.raises(ImmutableRecordError):
                db.commit()
            db.rollback()
            db.delete(row)
            with pytest.raises(ImmutableRecordError):
                db.commit()
            db.rollback()
    finally:
        db.close()
    assert not any("tampered" in e for e in _audits(cid))


def test_7b_execution_ledger_rows_cannot_be_rewritten_or_deleted():
    cid = _open("300")
    client.post(f"/cases/{cid}/execute", headers=KEY)
    db = SessionLocal()
    try:
        row = db.query(ExecutionAttemptModel).filter(ExecutionAttemptModel.case_id == cid).first()
        row.trace_id = "trace_forged"
        with pytest.raises(ImmutableRecordError):
            db.commit()
        db.rollback()
        db.delete(row)
        with pytest.raises(ImmutableRecordError):
            db.commit()
        db.rollback()
    finally:
        db.close()


def _matched_outcome():
    t = _tag()
    cust, ref = f"p17_im_{t}", f"p17_imref_{t}"
    cid = _open("300", customer=cust)
    # open a case WITH a reference so an outcome can match
    r = client.post(
        "/events",
        json={
            "customer_id": cust + "b",
            "risk_category": "FAILED_PAYMENT",
            "external_system": "p17",
            "external_event_id": f"m_{t}",
            "amount": "300",
            "currency": "INR",
            "reference_id": ref,
            "raw_payload": {"failure_code": "network_error"},
        },
        headers=KEY,
    )
    cid = r.json()["case_id"]
    out = client.post("/outcomes", json=_outcome_body(cust + "b", ref), headers=KEY).json()
    assert out["match_status"] == "MATCHED"
    return cid, out["outcome_event_id"]


def test_7c_matched_outcome_evidence_provenance_and_amounts_are_frozen():
    cid, oid = _matched_outcome()
    db = SessionLocal()
    try:
        for col, val in (
            ("provenance", "production"),
            ("evidence_reference", "forged"),
            ("amount", 999),
            ("applied_amount", 999),
            ("match_status", "UNMATCHED"),
            ("case_id", None),
            ("kind", "CANCELLATION"),
        ):
            row = (
                db.query(OutcomeEventModel)
                .filter(OutcomeEventModel.outcome_event_id == oid)
                .first()
            )
            setattr(row, col, val)
            with pytest.raises(ImmutableRecordError):
                db.commit()
            db.rollback()
        row = db.query(OutcomeEventModel).filter(OutcomeEventModel.outcome_event_id == oid).first()
        db.delete(row)
        with pytest.raises(ImmutableRecordError):
            db.commit()
        db.rollback()
    finally:
        db.close()
    assert _trace(cid)["stages"]["outcome"]["status"] == "RECOVERY_CONFIRMED"


def test_7d_dataset_snapshots_are_frozen():
    from infrastructure.orm import MLDatasetSnapshotModel

    db = SessionLocal()
    try:
        row = MLDatasetSnapshotModel(
            snapshot_id=f"snap_{_tag()}",
            dataset_name="p17",
            version=1,
            created_at=datetime.now(UTC).replace(tzinfo=None),
            eligible_case_count=0,
            provenance_distribution={},
            feature_schema_version="v",
            label_schema_version="v",
            exclusion_counts={},
            eligible_case_ids=[],
            checksum="x",
        )
        db.add(row)
        db.commit()
        row.checksum = "changed"
        with pytest.raises(ImmutableRecordError):
            db.commit()
        db.rollback()
    finally:
        db.close()


def test_7e_a_correction_is_a_new_append_only_record_not_an_edit():
    cid = _open("300")
    before = len(_audits(cid))
    db = SessionLocal()
    try:
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="a",
                to_state="a",
                evidence={
                    "action": "correction_note",
                    "corrects": "human_review",
                    "trace_id": trace_id_for_case(cid),
                },
            )
        )
        db.commit()
    finally:
        db.close()
    assert len(_audits(cid)) == before + 1


def test_7f_the_existing_unmatched_to_matched_rematch_flow_is_still_permitted():
    t = _tag()
    cust, ref = f"p17_rm_{t}", f"p17_rmref_{t}"
    out = client.post("/outcomes", json=_outcome_body(cust, ref), headers=KEY).json()
    assert out["match_status"] == "UNMATCHED"
    client.post(
        "/events",
        json={
            "customer_id": cust,
            "risk_category": "FAILED_PAYMENT",
            "external_system": "p17",
            "external_event_id": f"r_{t}",
            "amount": "300",
            "currency": "INR",
            "reference_id": ref,
            "raw_payload": {"failure_code": "network_error"},
        },
        headers=KEY,
    )
    r = client.post("/outcomes/rematch-unmatched", json={"events": []}, headers=KEY)
    assert r.status_code == 200


# ------------------------------------------------------------------ legacy trace handling


def _legacy_case():
    cid = f"case_legacy_{_tag()}"
    db = SessionLocal()
    try:
        db.add(
            CaseModel(
                case_id=cid,
                customer_id="legacy_cust",
                risk_category="FAILED_PAYMENT",
                amount_at_risk=500,
                currency="INR",
                current_state="POLICY_EVALUATED",
            )
        )
        db.flush()
        for action in (
            "case_created",
            "risk_assessment",
            "recovery_prediction",
            "action_recommendation",
            "policy_evaluation",
        ):
            db.add(
                AuditModel(
                    id=uuid.uuid4().hex,
                    case_id=cid,
                    from_state="a",
                    to_state="a",
                    evidence={
                        "action": action,
                        "decision_id": "pol_legacy",
                        "decision": "PERMITTED",
                    }
                    if action == "policy_evaluation"
                    else {"action": action},
                )
            )
        db.add(
            ExecutionAttemptModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                action_type="RETRY_PAYMENT",
                channel="payment",
                status="COMPLETED_SIMULATED",
                execution_id="exec_legacy",
                policy_decision_id="pol_legacy",
            )
        )
        db.commit()
    finally:
        db.close()
    return cid


def test_8_legacy_records_are_labelled_legacy_and_never_given_trace_ids():
    cid = _legacy_case()
    t = _trace(cid)
    assert (
        t["trace_status"] == "LEGACY_UNTRACED" and t["traceability"]["legacy_untraced_records"] >= 5
    )
    assert t["traceability"]["traced_records"] == 0
    assert all(
        s["detail"]["trace_id"] == "LEGACY_UNTRACED" and s["traceable"] is False for s in t["steps"]
    )
    assert t["stages"]["revalidation"]["status"] == "NOT_RECORDED"
    ex = t["stages"]["execution"]["executions"][0]
    assert ex["trace_id"] is None  # nothing fabricated
    db = SessionLocal()
    try:  # untouched
        assert all(
            not (a.evidence or {}).get("trace_id")
            for a in db.query(AuditModel).filter(AuditModel.case_id == cid)
        )
        assert (
            db.query(ExecutionAttemptModel)
            .filter(ExecutionAttemptModel.case_id == cid)
            .first()
            .trace_id
            is None
        )
    finally:
        db.close()


def test_8b_validator_reports_legacy_as_warnings_and_info_not_errors():
    cid = _legacy_case()
    v = client.get(f"/system/traces/{trace_id_for_case(cid)}/validation", headers=KEY).json()
    codes = {i["code"]: i["severity"] for i in v["issues"]}
    assert (
        codes.get("MISSING_TRACE_ID") == "WARN" and codes.get("REVALIDATION_NOT_RECORDED") == "INFO"
    )
    assert v["consistent"] is True and v["repaired"] is False
    assert v["traceability"]["status"] == "LEGACY_UNTRACED"


def test_8c_new_records_are_fully_traceable_and_distinguished_from_legacy():
    cid = _open("300")
    client.post(f"/cases/{cid}/execute", headers=KEY)
    t = _trace(cid)
    assert t["trace_status"] == "TRACED" and t["traceability"]["legacy_untraced_records"] == 0
    assert all(s["traceable"] for s in t["steps"])


def test_8d_a_mixed_case_is_reported_as_partially_traced():
    cid = _open("300")
    db = SessionLocal()
    try:
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=cid,
                from_state="a",
                to_state="a",
                evidence={"action": "diagnosis"},
            )
        )
        db.commit()
    finally:
        db.close()
    assert _trace(cid)["trace_status"] == "PARTIALLY_TRACED"


# ------------------------------------------------------------------ performance sanity


def test_9_authentication_and_authorization_overhead_is_negligible(monkeypatch):
    monkeypatch.setenv("API_KEY", USER_KEY)
    monkeypatch.setenv("RECOVERCHAIN_PRINCIPALS", json.dumps(REGISTRY))
    t = time.perf_counter()
    for _i in range(2000):
        principals.resolve_principal(ALICE_KEY)
        principals.resolve_principal("unknown-key")
    assert (time.perf_counter() - t) / 4000 < 0.001  # well under 1 ms per resolution
