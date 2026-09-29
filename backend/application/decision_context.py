"""Phase 13 — normalized decision context.

A single, explicit read-model of everything legitimately available AT DECISION
TIME for a case: no new data source, no new table — this assembles fields that
already exist on RecoveryCase / RecoveryPrediction / PolicyContext into one
object so a future candidate-scoring or offline-evaluation pass has a documented,
leakage-audited contract to build from, instead of each caller re-deriving its
own ad-hoc subset (which is how the split-column and event-count leaks in
earlier phases happened).

Nothing here is a new prediction, a new policy rule, or a new execution path -
it is read-only assembly of already-computed fields (application/case_pipeline.py
for prediction/explanation/anomaly, application/policy_engine.py's PolicyContext
for consent/retry/contact history, domain/models.py for the case's own state).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from domain.models import RecoveryCase

# Fields that would only be known AFTER a decision is made - if any of these
# ever appear as a key in a context dict, that dict is leaking the future.
FORBIDDEN_POST_DECISION_KEYS = frozenset(
    {
        "recovered_amount",
        "actual_amount_recovered",
        "outcome_status",
        "outcome_timestamp",
        "verification_timestamp",
        "execution_result",
        "action_result",
        "reconciliation_status",
    }
)


def build_decision_context(
    case: RecoveryCase, policy_context: Any | None = None, *, as_of: datetime | None = None
) -> dict:
    """Assembles the normalized decision context for ONE case at the given
    instant (defaults to now). `policy_context` is the PolicyContext this
    decision will actually be evaluated against (application.policy_engine) -
    passed in, not rebuilt here, so this module never becomes a second source
    of truth for retry/contact/consent state."""
    as_of = as_of or datetime.now(UTC)
    pred = case.prediction
    diag = case.diagnosis
    rec = case.recommendation

    time_since_event_hours = None
    if case.linked_events:
        first = min(e.timestamp for e in case.linked_events)
        time_since_event_hours = round((as_of - first).total_seconds() / 3600.0, 2)

    context = {
        "case_id": case.case_id,
        "customer_id": case.customer_id,
        "risk_category": case.risk_category.value,
        "domain": case.risk_category.value,
        "amount_at_risk": float(case.amount_at_risk.amount) if case.amount_at_risk else None,
        "currency": case.amount_at_risk.currency if case.amount_at_risk else None,
        "obligation_state": case.current_state.value,
        "outstanding_amount": float(case.outstanding_amount.amount)
        if case.outstanding_amount
        else None,
        "cause_category": diag.cause_category.value if diag else None,
        "diagnosis_confidence": diag.confidence if diag else None,
        # Prediction signals - advisory, never authorization.
        "recovery_probability": pred.recovery_probability if pred else None,
        "prediction_status": pred.prediction_status if pred else None,
        "confidence_tier": pred.confidence_tier if pred else None,
        "anomaly_level": (pred.anomaly or {}).get("level") if pred and pred.anomaly else None,
        "model_version": pred.model_version if pred else None,
        # ERV - a calculation, not a prediction and not money.
        "expected_recoverable_value": float(case.expected_recoverable_value.amount)
        if case.expected_recoverable_value
        else None,
        # Prior recommendation, if this is a re-decision.
        "previous_action": rec.top_candidate.action_type.value
        if rec and rec.top_candidate
        else None,
        "time_since_first_event_hours": time_since_event_hours,
        "decision_timestamp": as_of.isoformat(),
    }

    if policy_context is not None:
        context["policy_signals"] = {
            "prior_payment_attempts": policy_context.prior_payment_attempts,
            "comms_in_last_24h": policy_context.comms_in_last_24h,
            "stopped": policy_context.stopped,
            "communication_permission": policy_context.communication_permission,
            "payment_permission": policy_context.payment_permission,
        }

    return context


def audit_context_leakage(context: dict) -> list[str]:
    """Returns a list of violations (empty = clean). Checks for forbidden
    post-decision keys at any nesting level, and for any ISO timestamp value
    that falls after decision_timestamp (a context that "knows" something from
    the future relative to its own stated decision instant)."""
    violations: list[str] = []
    decision_ts = context.get("decision_timestamp")
    decision_dt = datetime.fromisoformat(decision_ts) if decision_ts else None

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in FORBIDDEN_POST_DECISION_KEYS:
                    violations.append(f"{path}.{k}: forbidden post-decision field present")
                if isinstance(v, str) and decision_dt is not None:
                    try:
                        v_dt = datetime.fromisoformat(v)
                    except ValueError:
                        v_dt = None
                    if v_dt is not None and v_dt > decision_dt:
                        violations.append(f"{path}.{k}: timestamp {v} is after decision time")
                _walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{path}[{i}]")

    _walk(context, "context")
    return violations
