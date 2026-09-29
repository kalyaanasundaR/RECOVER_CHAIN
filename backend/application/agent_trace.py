"""Phase 16 - end-to-end trace reconstruction (read-only).

Given ONE case (== one journey == one `trace_id`, application/trace_ids.py) rebuild:

    Case -> Agent -> ML prediction -> Recommendation -> Policy -> Approval ->
    Execution -> Verification -> Outcome

from data that already exists: the case row (its prediction / recommendation / policy /
execution / outcome JSON), the append-only audit trail, the execution ledger, and the
outcome_events table. Nothing is written, back-filled, or repaired. A stage that did not
happen is reported as `NOT_OCCURRED` (or `PENDING` / `NOT_REQUIRED`), never invented.

Query budget is CONSTANT (case + audit + ledger + matched outcomes + unmatched
candidates = 5 statements) regardless of trace size - no per-row queries.

Everything returned is passed through `redact` (application/trace_ids.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from api.principals import UNKNOWN_PRINCIPAL
from application.trace_ids import mask_identifier, redact, trace_id_for_case
from infrastructure.orm import AuditModel, CaseModel, ExecutionAttemptModel, OutcomeEventModel

# audit `action` -> coarse stage (kept from Phase 15 for the chronological `steps` list)
_STAGE = {
    "risk_assessment": "CASE",
    "diagnosis": "CASE",
    "recovery_prediction": "MODEL",
    "action_recommendation": "RECOMMENDATION",
    "agent_recommendation": "AGENT",
    "agent_handoff": "AGENT",
    "policy_evaluation": "POLICY",
    "human_review": "APPROVAL",
    "policy_revalidation": "REVALIDATION",
    "execution": "EXECUTION",
    "outcome_reconciled": "OUTCOME",
    "outcome_recorded": "OUTCOME",
}
_ORDER = [
    "CASE",
    "MODEL",
    "RECOMMENDATION",
    "AGENT",
    "POLICY",
    "APPROVAL",
    "REVALIDATION",
    "EXECUTION",
    "OUTCOME",
]
_MAX_UNMATCHED = 5
_STEP_KEYS = (
    "trace_id",
    "parent_trace_id",
    "operation_id",
    "agent_id",
    "agent_version",
    "status",
    "recommended_action",
    "model_version",
    "probability",
    "decision",
    "decision_id",
    "blocked_by",
    "reason_codes",
    "policy_prerequisites",
    "agrees_with_deterministic",
    "approved_decision_id",
    "approved_action",
    "adapter",
    "execution_id",
    "policy_decision_id",
    "outcome_event_id",
    "effect",
    "kind",
    "revalidation_id",
    "result",
    "policy_version",
    "approval_id",
    "approval_reference",
    "actor",
    "requested_action",
    "policy_version_changed",
)


@dataclass
class TraceData:
    case: CaseModel
    audits: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    matched_outcomes: list = field(default_factory=list)
    unmatched_candidates: list = field(default_factory=list)


def load_trace_data(db: Session, case_id: str) -> TraceData | None:
    """The 5 constant queries. Returns None if the case does not exist."""
    case = db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
    if case is None:
        return None
    audits = (
        db.query(AuditModel)
        .filter(AuditModel.case_id == case_id)
        .order_by(AuditModel.timestamp.asc(), AuditModel.id.asc())
        .all()
    )
    attempts = (
        db.query(ExecutionAttemptModel)
        .filter(ExecutionAttemptModel.case_id == case_id)
        .order_by(ExecutionAttemptModel.created_at.asc())
        .all()
    )
    matched = (
        db.query(OutcomeEventModel)
        .filter(OutcomeEventModel.case_id == case_id)
        .order_by(OutcomeEventModel.occurred_at.asc())
        .all()
    )
    q = db.query(OutcomeEventModel).filter(
        OutcomeEventModel.match_status == "UNMATCHED",
        OutcomeEventModel.customer_id == case.customer_id,
    )
    if case.reference_id:
        q = q.filter(OutcomeEventModel.reference_id == case.reference_id)
    unmatched = q.order_by(OutcomeEventModel.occurred_at.asc()).limit(_MAX_UNMATCHED).all()
    return TraceData(case, audits, attempts, matched, unmatched)


def _iso(dt):
    return dt.isoformat() if dt else None


def _f(v):
    return float(v) if v is not None else None


def _ev(a: AuditModel) -> dict:
    return a.evidence or {}


def _stage_case(d: TraceData) -> dict:
    c = d.case
    created = next(
        (_iso(a.timestamp) for a in d.audits if _ev(a).get("action") == "case_created"), None
    )
    return {
        "status": "OCCURRED",
        "case_id": c.case_id,
        "domain": getattr(c.risk_category, "value", c.risk_category),
        "customer_id": mask_identifier(c.customer_id),
        "reference_id": c.reference_id,
        "amount_at_risk": _f(c.amount_at_risk),
        "currency": c.currency,
        "obligation_state": getattr(c.current_state, "value", c.current_state),
        "created_at": created,
    }


def _stage_agent(d: TraceData) -> dict:
    runs = [a for a in d.audits if _ev(a).get("action") == "agent_recommendation"]
    if not runs:
        return {"status": "NOT_OCCURRED"}
    last = _ev(runs[-1])
    return {
        "status": "OCCURRED",
        "agent_id": last.get("agent_id"),
        "agent_version": last.get("agent_version"),
        "domain": last.get("domain"),
        "recommendation_status": last.get("status"),
        "recommended_action": last.get("recommended_action"),
        "reason_codes": last.get("reason_codes"),
        "policy_prerequisites": last.get("policy_prerequisites"),
        "agrees_with_deterministic": last.get("agrees_with_deterministic"),
        "runs": len(runs),
        "operation_ids": [_ev(a).get("operation_id") for a in runs],
        "latest_operation_id": last.get("operation_id"),
        "note": "AGENT REASONING - proposes only; it did not authorize anything",
    }


def _stage_ml(d: TraceData) -> dict:
    p = d.case.prediction
    if not p:
        return {"status": "NOT_OCCURRED"}
    expl = p.get("explanation") or []
    return {
        "status": "OCCURRED",
        "model_version": p.get("model_version"),
        "prediction_id": p.get("prediction_id"),
        "prediction_status": p.get("prediction_status"),
        "recovery_probability": p.get("recovery_probability"),
        "expected_recoverable_value": _f(d.case.expected_recoverable_value),
        "confidence_tier": p.get("confidence_tier"),
        "anomaly_level": (p.get("anomaly") or {}).get("level"),
        "explanation_reference": {
            "prediction_id": p.get("prediction_id"),
            "factor_count": len(expl),
            "top_factors": [
                {"feature": e.get("feature"), "direction": e.get("direction")} for e in expl[:3]
            ],
        },
        "note": "MODEL ASSOCIATION - advisory input to the recommendation; the model authorized nothing",
    }


def _stage_recommendation(d: TraceData) -> dict:
    r = d.case.recommendation
    if not r:
        return {"status": "NOT_OCCURRED"}
    top = r.get("top_candidate") or {}
    return {
        "status": "OCCURRED",
        "engine_version": r.get("engine_version"),
        "recommendation_status": r.get("status"),
        "top_action": top.get("action_type"),
        "candidates": [
            {
                "action": c.get("action_type"),
                "estimated_probability": c.get("estimated_probability"),
                "expected_recoverable_value": c.get("expected_recoverable_value"),
            }
            for c in (r.get("candidates") or [])
        ],
    }


def _stage_policy(d: TraceData) -> dict:
    evals = [a for a in d.audits if _ev(a).get("action") == "policy_evaluation"]
    dec = d.case.policy_decision
    if not dec and not evals:
        return {"status": "NOT_OCCURRED"}
    dec = dec or {}
    reason_codes = (
        dec.get("reason_codes")
        or [r.get("rule_name") for r in (dec.get("failed_rules") or [])]
        or ["ALL_RULES_PASSED"]
    )
    return {
        "status": "OCCURRED",
        "decision_id": dec.get("decision_id"),
        "policy_version": dec.get("policy_version"),
        "decision": dec.get("status"),
        "action": dec.get("action"),
        "reason_codes": reason_codes,
        "approval_required": dec.get("approval_required")
        if dec.get("approval_required") is not None
        else any(_ev(a).get("decision") == "ESCALATE" for a in evals),
        "lifecycle_state": dec.get("lifecycle_state"),
        "timestamp": dec.get("timestamp"),
        "trace_id": dec.get("trace_id"),
        "evaluations": [
            {
                "decision_id": _ev(a).get("decision_id"),
                "decision": _ev(a).get("decision"),
                "reason": _ev(a).get("reason"),
                "timestamp": _iso(a.timestamp),
            }
            for a in evals
        ],
        "note": "POLICY DECISION - the only authorization authority",
    }


def _stage_approval(d: TraceData) -> dict:
    reviews = [a for a in d.audits if _ev(a).get("action") == "human_review"]
    escalated = any(
        _ev(a).get("action") == "policy_evaluation" and _ev(a).get("decision") == "ESCALATE"
        for a in d.audits
    )
    cur = (d.case.policy_decision or {}).get("decision_id")
    if not reviews:
        return {
            "status": "PENDING_APPROVAL" if escalated else "NOT_REQUIRED",
            "required": escalated,
        }
    return {
        "status": "OCCURRED",
        "required": escalated,
        "reviews": [
            {
                "audit_id": a.id,
                "decision": _ev(a).get("decision"),
                "approved_decision_id": _ev(a).get("approved_decision_id"),
                "approved_action": _ev(a).get("approved_action"),
                "approval_id": _ev(a).get("approval_id"),
                "policy_version": _ev(a).get("policy_version"),
                # who approved: recorded from Phase 17; older approvals have no principal
                "actor": _ev(a).get("actor") or dict(UNKNOWN_PRINCIPAL),
                "bound_to_current_policy_decision": _ev(a).get("approved_decision_id") == cur,
                "timestamp": _iso(a.timestamp),
                # free-text note: kept for audit, redacted like every other field
                "note": (_ev(a).get("note") or "")[:200],
            }
            for a in reviews
        ],
        "approver_identity": (
            "see reviews[].actor (authenticated principal, Phase 17+); UNKNOWN for older approvals"
        ),
    }


def _stage_revalidation(d: TraceData, executed: bool) -> dict:
    recs = [a for a in d.audits if _ev(a).get("action") == "policy_revalidation"]
    if not recs:
        if executed:
            return {
                "status": "NOT_RECORDED",
                "note": "an execution happened but no revalidation record exists (pre-Phase-17 "
                "execution); revalidation is enforced in POST /execute but was not recorded then",
            }
        return {"status": "NOT_OCCURRED"}
    items = []
    for a in recs:
        e = _ev(a)
        items.append(
            {
                "revalidation_id": e.get("revalidation_id"),
                "result": e.get("result"),
                "requested_action": e.get("requested_action"),
                "policy_version": e.get("policy_version"),
                "initial_decision_id": e.get("initial_decision_id"),
                "initial_policy_version": e.get("initial_policy_version"),
                "policy_version_changed": e.get("policy_version_changed"),
                "revalidation_decision_id": e.get("revalidation_decision_id"),
                "reason_codes": e.get("reason_codes"),
                "reason": e.get("reason"),
                "approval_required": e.get("approval_required"),
                "approval_reference": e.get("approval_reference"),
                "approval_actor": e.get("approval_actor"),
                "revalidated_at": e.get("revalidated_at"),
                "audit_id": a.id,
                "trace_id": e.get("trace_id"),
            }
        )
    return {
        "status": "OCCURRED",
        "latest_result": items[-1]["result"],
        "revalidations": items,
        "note": "a FRESH Policy Engine evaluation immediately before the side effect; "
        "approval never substitutes for it",
    }


def traceability(d: TraceData) -> dict:
    """NEW TRACEABLE records (carry a trace_id) vs LEGACY UNTRACED records (written before
    Phase 16). Legacy records are counted and labelled - never given a synthetic id."""
    known = [a for a in d.audits if _ev(a).get("action") in _STAGE]
    traced = sum(1 for a in known if _ev(a).get("trace_id")) + sum(
        1 for x in d.attempts if x.trace_id
    )
    total = len(known) + len(d.attempts)
    legacy = total - traced
    if total and traced == 0:
        status = "LEGACY_UNTRACED"
    elif legacy == 0:
        status = "TRACED"
    else:
        status = "PARTIALLY_TRACED"
    return {
        "status": status,
        "traced_records": traced,
        "legacy_untraced_records": legacy,
        "note": (
            "legacy records predate Phase 16 tracing: shown as LEGACY / UNKNOWN, not rewritten, "
            "not given a trace id"
            if legacy
            else "every recorded stage carries the journey trace id"
        ),
    }


def _stage_execution(d: TraceData) -> dict:
    by_ledger = {a.execution_id: a for a in d.attempts if a.execution_id}
    last_rec = d.case.execution_record or {}
    entries, seen = [], set()
    for a in d.audits:
        ev = _ev(a)
        if ev.get("action") != "execution":
            continue
        eid = ev.get("execution_id") or (
            last_rec.get("execution_id") if not ev.get("blocked_by") else None
        )
        led = by_ledger.get(eid) if eid else None
        if eid:
            seen.add(eid)
        entries.append(
            {
                "execution_id": eid,
                "operation_id": (led.execution_id if led else eid),
                "action": led.action_type
                if led
                else last_rec.get("action_type")
                if eid == last_rec.get("execution_id")
                else None,
                "agent": ev.get("agent"),
                "status": ev.get("status"),
                "adapter": ev.get("adapter"),
                "policy_decision_id": ev.get("policy_decision_id")
                or (led.policy_decision_id if led else None),
                "trace_id": led.trace_id if led else ev.get("trace_id"),
                "blocked_by": ev.get("blocked_by"),
                "revalidation_id": ev.get("revalidation_id"),
                "timestamp": _iso(a.timestamp),
                "source": "audit+ledger" if led else "audit",
            }
        )
    for led in d.attempts:
        if led.execution_id and led.execution_id in seen:
            continue
        entries.append(
            {
                "execution_id": led.execution_id,
                "operation_id": led.execution_id,
                "action": led.action_type,
                "agent": led.agent_type,
                "status": led.status,
                "adapter": None,
                "policy_decision_id": led.policy_decision_id,
                "trace_id": led.trace_id,
                "blocked_by": None,
                "timestamp": _iso(led.created_at),
                "source": "ledger_only",
            }
        )
    if not entries:
        return {"status": "NOT_OCCURRED", "executions": []}
    executed = [e for e in entries if e["status"] == "COMPLETED_SIMULATED"]
    return {
        "status": "OCCURRED" if executed else "BLOCKED_OR_FAILED",
        "executions": entries,
        "revalidation": (
            "BLOCKED"
            if any(e["blocked_by"] == "policy_revalidation" for e in entries)
            else "ENFORCED_IN_POST_EXECUTE (not recorded per successful attempt)"
        ),
        "note": "an execution is an ACTION ATTEMPT - it is never evidence that money was recovered",
    }


def _stage_verification(d: TraceData) -> dict:
    o = d.case.outcome
    if not o:
        return {
            "status": "NOT_OCCURRED",
            "note": "verification is a read of the authoritative position",
        }
    return {
        "status": "OCCURRED",
        "outcome_status": o.get("status"),
        "reconciliation_status": o.get("reconciliation_status"),
        "verification_basis": o.get("verification_basis"),
        "action_result": o.get("action_result"),
        "note": "verification READS the authoritative position; it decides nothing",
    }


def _stage_outcome(d: TraceData, executed: bool) -> dict:
    events = [
        {
            "outcome_event_id": o.outcome_event_id,
            "kind": o.kind,
            "outcome_type": o.outcome_type,
            "provenance": o.provenance,
            "evidence_reference": o.evidence_reference,
            "applied_amount": str(o.applied_amount) if o.applied_amount is not None else None,
            "excess_amount": str(o.excess_amount) if o.excess_amount is not None else None,
            "effect": o.effect,
            "occurred_at": _iso(o.occurred_at),
        }
        for o in d.matched_outcomes
        if o.match_status == "MATCHED"
    ]
    unmatched = [
        {
            "outcome_event_id": o.outcome_event_id,
            "provenance": o.provenance,
            "match_reason": o.match_reason,
            "amount": str(o.amount) if o.amount is not None else None,
            "occurred_at": _iso(o.occurred_at),
            "applied_to_this_trace": False,
        }
        for o in d.unmatched_candidates
    ]
    recovered = _f(d.case.recovered_amount) or 0.0
    settled = [
        e for e in events if e["kind"] == "SETTLEMENT" and float(e["applied_amount"] or 0) > 0
    ]
    if settled:
        status = "RECOVERY_CONFIRMED"
    elif any(e["kind"] == "CANCELLATION" for e in events):
        status = "OBLIGATION_CANCELLED"
    elif events:
        status = "OUTCOME_REPORTED_NO_RECOVERY_APPLIED"
    elif executed:
        status = "PENDING"
    else:
        status = "NOT_OCCURRED"
    return {
        "status": status,
        "recovered_amount": recovered,
        "outstanding_amount": _f(d.case.outstanding_amount),
        "matched_events": events,
        "unmatched_candidates": unmatched,
        "note": "ACTUAL OUTCOME - only authoritative outcome events establish recovery",
    }


def _journey(d: TraceData, stages: dict) -> dict:
    ex = stages["execution"]
    executed = any(e["status"] == "COMPLETED_SIMULATED" for e in ex.get("executions", []))
    confirmed = stages["outcome"]["status"] == "RECOVERY_CONFIRMED"
    pol = stages["policy"]
    if executed and confirmed:
        label = "EXECUTED — RECOVERY CONFIRMED BY AUTHORITATIVE OUTCOME"
    elif executed:
        label = "EXECUTED — RECOVERY NOT YET CONFIRMED"
    elif stages["approval"]["status"] == "PENDING_APPROVAL":
        label = "AWAITING HUMAN APPROVAL — NOT EXECUTED"
    elif stages.get("revalidation", {}).get("latest_result") == "FAILED":
        label = "NOT EXECUTED — BLOCKED: POLICY REVALIDATION FAILED"
    elif ex["status"] == "BLOCKED_OR_FAILED" or pol.get("decision") in ("DENIED", "WAIT"):
        label = "NOT EXECUTED — BLOCKED"
    else:
        label = "NOT EXECUTED"
    reviews = stages["approval"].get("reviews") or []
    last_review = reviews[-1] if reviews else None
    return {
        "executed": executed,
        "revalidation": stages.get("revalidation", {}).get("latest_result")
        or stages.get("revalidation", {}).get("status"),
        "approved_by": (last_review or {}).get("actor"),
        "execution_is_simulated": any(
            e.get("adapter") == "MockExecutionAdapter" for e in ex.get("executions", [])
        ),
        "recovery_confirmed": confirmed,
        "label": label,
    }


def reconstruct_trace(db: Session, case_id: str) -> dict | None:
    d = load_trace_data(db, case_id)
    if d is None:
        return None
    stages = {
        "case": _stage_case(d),
        "agent": _stage_agent(d),
        "ml_prediction": _stage_ml(d),
        "recommendation": _stage_recommendation(d),
        "policy": _stage_policy(d),
        "approval": _stage_approval(d),
        "execution": _stage_execution(d),
        "verification": _stage_verification(d),
    }
    executed_any = any(
        e["status"] == "COMPLETED_SIMULATED" for e in stages["execution"].get("executions", [])
    )
    stages["revalidation"] = _stage_revalidation(d, executed_any)
    stages["outcome"] = _stage_outcome(d, executed_any)

    steps = []
    for a in d.audits:
        ev = _ev(a)
        stage = _STAGE.get(ev.get("action"))
        if stage is None:
            continue
        steps.append(
            {
                "stage": stage,
                "action": ev.get("action"),
                "audit_id": a.id,
                "timestamp": _iso(a.timestamp),
                "detail": {k: ev[k] for k in _STEP_KEYS if k in ev}
                | ({} if ev.get("trace_id") else {"trace_id": "LEGACY_UNTRACED"}),
                "traceable": bool(ev.get("trace_id")),
            }
        )
    steps.sort(key=lambda s: (s["timestamp"] or "", _ORDER.index(s["stage"])))

    agent = stages["agent"]
    out = {
        "trace_id": trace_id_for_case(case_id),
        "case_id": case_id,
        # Phase 15 compatibility: the compact agent summary the UI reads
        "agent": (
            {
                "agent_id": agent.get("agent_id"),
                "agent_version": agent.get("agent_version"),
                "domain": agent.get("domain"),
                "status": agent.get("recommendation_status"),
                "recommended_action": agent.get("recommended_action"),
                "reason_codes": agent.get("reason_codes"),
                "policy_prerequisites": agent.get("policy_prerequisites"),
                "agrees_with_deterministic": agent.get("agrees_with_deterministic"),
            }
            if agent["status"] == "OCCURRED"
            else None
        ),
        "policy_decision_id": stages["policy"].get("decision_id"),
        "execution_id": (d.case.execution_record or {}).get("execution_id"),
        "stages": stages,
        "traceability": traceability(d),
        "trace_status": traceability(d)["status"],
        "journey": _journey(d, stages),
        "steps": steps,
        "read_only": True,
        "note": (
            "MODEL ASSOCIATION, AGENT REASONING, POLICY DECISION and ACTUAL OUTCOME are "
            "separate stages: an agent recommendation is never an authorization, an "
            "execution is never an outcome, and missing stages are reported, not invented"
        ),
    }
    return redact(out)
