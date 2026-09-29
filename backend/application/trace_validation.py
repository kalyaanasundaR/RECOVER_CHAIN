"""Phase 16 - trace consistency validator (read-only, never repairs).

Reports problems in a case's recorded history; it does NOT fix, back-fill, delete or
reinterpret anything, and never touches a financial field. Issues carry a severity:

  ERROR  - the history contradicts itself or an invariant (needs a human)
  WARN   - incomplete/legacy data that limits what can be proven
  INFO   - expected legacy shape (e.g. a Phase 15 per-run trace id)
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from application.agent_trace import TraceData, load_trace_data, traceability
from application.trace_ids import is_trace_id, trace_id_for_case
from infrastructure.orm import CaseModel

# canonical journey order for first-occurrence timing checks
_CHAIN = [
    ("recovery_prediction", "MODEL"),
    ("action_recommendation", "RECOMMENDATION"),
    ("policy_evaluation", "POLICY"),
    ("human_review", "APPROVAL"),
    ("execution", "EXECUTION"),
]


def _issue(code: str, severity: str, message: str, ref=None) -> dict:
    return {"code": code, "severity": severity, "message": message, "ref": ref}


def _ev(a) -> dict:
    return a.evidence or {}


def validate_trace_data(d: TraceData) -> dict:
    case = d.case
    canonical = trace_id_for_case(case.case_id)
    issues: list[dict] = []

    # --- missing / conflicting trace ids ---------------------------------------------
    for a in d.audits:
        ev = _ev(a)
        if not ev.get("action"):
            continue
        tid = ev.get("trace_id")
        if tid is None:
            issues.append(
                _issue(
                    "MISSING_TRACE_ID",
                    "WARN",
                    f"audit record {ev['action']} has no trace_id (pre-Phase-16?)",
                    a.id,
                )
            )
        elif not is_trace_id(tid):
            issues.append(
                _issue(
                    "MALFORMED_TRACE_ID",
                    "ERROR",
                    f"audit record has malformed trace_id {tid!r}",
                    a.id,
                )
            )
        elif tid != canonical:
            legacy = ev.get("action") == "agent_recommendation" and "operation_id" not in ev
            issues.append(
                _issue(
                    "LEGACY_TRACE_ID" if legacy else "CONFLICTING_TRACE_ID",
                    "INFO" if legacy else "ERROR",
                    f"{ev['action']} carries trace_id {tid}, journey trace is {canonical}",
                    a.id,
                )
            )
    for led in d.attempts:
        if led.trace_id is None:
            issues.append(
                _issue(
                    "MISSING_TRACE_ID",
                    "WARN",
                    "execution ledger row has no trace_id (pre-Phase-16?)",
                    led.execution_id,
                )
            )
        elif led.trace_id != canonical:
            issues.append(
                _issue(
                    "CONFLICTING_TRACE_ID",
                    "ERROR",
                    f"ledger row trace_id {led.trace_id} != {canonical}",
                    led.execution_id,
                )
            )
    for label, blob in (
        ("policy_decision", case.policy_decision),
        ("execution_record", case.execution_record),
    ):
        tid = (blob or {}).get("trace_id")
        if blob and tid is None:
            issues.append(
                _issue("MISSING_TRACE_ID", "WARN", f"case {label} has no trace_id (pre-Phase-16?)")
            )
        elif blob and tid != canonical:
            issues.append(
                _issue(
                    "CONFLICTING_TRACE_ID", "ERROR", f"case {label} trace_id {tid} != {canonical}"
                )
            )

    # --- policy decisions -------------------------------------------------------------
    eval_ids = {
        _ev(a).get("decision_id") for a in d.audits if _ev(a).get("action") == "policy_evaluation"
    }
    eval_by_id = {
        _ev(a).get("decision_id"): _ev(a).get("decision")
        for a in d.audits
        if _ev(a).get("action") == "policy_evaluation"
    }
    cur = (case.policy_decision or {}).get("decision_id")
    if cur and cur not in eval_ids:
        issues.append(
            _issue(
                "ORPHANED_POLICY_DECISION",
                "ERROR",
                "case policy decision has no policy_evaluation audit record",
                cur,
            )
        )
    if not case.policy_decision and eval_ids:
        issues.append(
            _issue(
                "ORPHANED_POLICY_DECISION",
                "WARN",
                "policy_evaluation audit records exist but the case holds no policy decision",
            )
        )

    for a in d.audits:
        ev = _ev(a)
        if ev.get("action") == "human_review" and not ev.get("actor"):
            issues.append(
                _issue(
                    "APPROVAL_WITHOUT_ACTOR",
                    "INFO",
                    "approval recorded before Phase 17: acting principal is UNKNOWN",
                    a.id,
                )
            )

    approvals = {
        _ev(a).get("approved_decision_id")
        for a in d.audits
        if _ev(a).get("action") == "human_review"
        and str(_ev(a).get("decision", "")).upper() == "APPROVE"
    }

    # --- executions -------------------------------------------------------------------
    audit_exec_ids = {
        _ev(a).get("execution_id") for a in d.audits if _ev(a).get("action") == "execution"
    }
    last_eid = (case.execution_record or {}).get("execution_id")
    for led in d.attempts:
        if (
            led.execution_id
            and led.execution_id not in audit_exec_ids
            and led.execution_id != last_eid
        ):
            issues.append(
                _issue(
                    "ORPHANED_EXECUTION",
                    "WARN",
                    "ledger row has no matching execution audit record",
                    led.execution_id,
                )
            )
        if led.status == "COMPLETED_SIMULATED":
            # Phase 17: revalidation evidence for this decision, up to the execution time
            revals = [
                _ev(a)
                for a in d.audits
                if _ev(a).get("action") == "policy_revalidation"
                and _ev(a).get("initial_decision_id") == led.policy_decision_id
                and (led.created_at is None or a.timestamp <= led.created_at)
            ]
            if not revals:
                issues.append(
                    _issue(
                        "REVALIDATION_NOT_RECORDED",
                        "INFO",
                        "completed execution has no policy_revalidation record (pre-Phase-17?)",
                        led.execution_id,
                    )
                )
            elif revals[-1].get("result") == "FAILED":
                issues.append(
                    _issue(
                        "EXECUTION_AFTER_FAILED_REVALIDATION",
                        "ERROR",
                        "execution completed although the latest revalidation for its decision FAILED",
                        led.execution_id,
                    )
                )
            pid = led.policy_decision_id
            permitted = eval_by_id.get(pid) == "PERMITTED"
            approved = pid in approvals and eval_by_id.get(pid) == "ESCALATE"
            if not (permitted or approved):
                issues.append(
                    _issue(
                        "EXECUTION_WITHOUT_POLICY_AUTHORIZATION",
                        "ERROR",
                        "completed execution is not backed by a PERMITTED policy evaluation or a human approval of an ESCALATE decision",
                        led.execution_id,
                    )
                )

    # --- outcomes ---------------------------------------------------------------------
    matched = [o for o in d.matched_outcomes if o.match_status == "MATCHED"]
    domain = getattr(case.risk_category, "value", case.risk_category)
    for o in matched:
        if o.domain != domain:
            issues.append(
                _issue(
                    "OUTCOME_WITHOUT_CORRELATION",
                    "ERROR",
                    f"matched outcome domain {o.domain} != case domain {domain}",
                    o.outcome_event_id,
                )
            )
        if not o.provenance:
            issues.append(
                _issue(
                    "OUTCOME_WITHOUT_CORRELATION",
                    "ERROR",
                    "matched outcome has no provenance",
                    o.outcome_event_id,
                )
            )
    recovered = float(case.recovered_amount or 0)
    applied = sum(float(o.applied_amount or 0) for o in matched if o.kind == "SETTLEMENT")
    if recovered > 0 and not matched:
        issues.append(
            _issue(
                "OUTCOME_WITHOUT_CORRELATION",
                "ERROR",
                "case shows recovered money but no matched authoritative outcome event exists",
            )
        )
    elif recovered > 0 and abs(recovered - applied) > 0.0001:
        issues.append(
            _issue(
                "OUTCOME_WITHOUT_CORRELATION",
                "ERROR",
                f"case recovered_amount {recovered} != applied settlements {applied}",
            )
        )

    # --- stage ordering (first occurrence; same-instant is fine) ------------------------
    first: dict[str, object] = {}
    for a in d.audits:
        act = _ev(a).get("action")
        if act and act not in first:
            first[act] = a.timestamp
    prev_name, prev_ts = None, None
    for act, _stage in _CHAIN:
        ts = first.get(act)
        if ts is None:
            continue
        if prev_ts is not None and ts < prev_ts:
            issues.append(
                _issue(
                    "STAGE_ORDER_VIOLATION", "ERROR", f"{act} first occurs before {prev_name}", act
                )
            )
        prev_name, prev_ts = act, max(ts, prev_ts) if prev_ts is not None else ts

    errors = [i for i in issues if i["severity"] == "ERROR"]
    return {
        "trace_id": canonical,
        "case_id": case.case_id,
        "consistent": not errors,
        "traceability": traceability(d),
        "counts": {
            "ERROR": len(errors),
            "WARN": sum(1 for i in issues if i["severity"] == "WARN"),
            "INFO": sum(1 for i in issues if i["severity"] == "INFO"),
        },
        "issues": issues,
        "read_only": True,
        "repaired": False,
    }


def validate_trace(db: Session, case_id: str) -> dict | None:
    d = load_trace_data(db, case_id)
    return validate_trace_data(d) if d else None


def validate_all(db: Session, limit: int = 200) -> dict:
    """Bounded scan (one 5-query load per case). Reports; never repairs."""
    ids = [r[0] for r in db.query(CaseModel.case_id).order_by(CaseModel.case_id).limit(limit).all()]
    results = [validate_trace(db, cid) for cid in ids]
    bad = [r for r in results if r and not r["consistent"]]
    return {
        "cases_checked": len(ids),
        "inconsistent_cases": len(bad),
        "errors": sum(r["counts"]["ERROR"] for r in results if r),
        "warnings": sum(r["counts"]["WARN"] for r in results if r),
        "inconsistent": [
            {
                "case_id": r["case_id"],
                "trace_id": r["trace_id"],
                "issues": [i for i in r["issues"] if i["severity"] == "ERROR"],
            }
            for r in bad
        ],
        "read_only": True,
        "repaired": False,
    }
