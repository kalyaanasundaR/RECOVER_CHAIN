"""Phase 17 - structured evidence for execution-time policy revalidation.

The revalidation itself already exists (api/main.py POST /cases/{id}/execute re-runs the
DeterministicPolicyEngine against fresh durable context immediately before any side
effect). This module only builds the *record* of it - it never re-implements a rule and
never influences the verdict. The record is written as an append-only audit entry
(`action: "policy_revalidation"`), so a successful execution reads:

    initial policy decision -> [human approval] -> fresh revalidation (PASSED) -> execution

and a blocked one:

    initial policy decision -> human approval -> fresh revalidation (FAILED) -> no execution

Approval is preserved untouched in both cases: revalidation fails *forward* (a new
record), it never rewrites the approval or the original decision.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from domain.models import PolicyDecisionStatus


def latest_approval(audits, decision_id: str | None) -> dict | None:
    """The most recent APPROVE that reviewed exactly this decision id, else None."""
    found = None
    for a in audits:
        ev = a.evidence or {}
        if (
            ev.get("action") == "human_review"
            and str(ev.get("decision", "")).upper() == "APPROVE"
            and ev.get("approved_decision_id") == decision_id
        ):
            found = ev
    return found


def revalidation_evidence(*, case, requested_action, revalidated, approval: dict | None) -> dict:
    initial = case.policy_decision
    passed = revalidated.status == PolicyDecisionStatus.PERMITTED
    return {
        "action": "policy_revalidation",
        "revalidation_id": f"reval_{uuid.uuid4().hex[:12]}",
        "case_id": case.case_id,
        "requested_action": getattr(requested_action, "value", str(requested_action)),
        "result": "PASSED" if passed else "FAILED",
        "revalidated_status": revalidated.status.value,
        "reason_codes": list(revalidated.reason_codes),
        "reason": revalidated.reason,
        "policy_version": revalidated.policy_version,
        "initial_decision_id": initial.decision_id if initial else None,
        "initial_policy_version": initial.policy_version if initial else None,
        "policy_version_changed": bool(
            initial and initial.policy_version != revalidated.policy_version
        ),
        "revalidation_decision_id": revalidated.decision_id,
        "approval_required": bool(initial.approval_required) if initial else False,
        "approval_reference": (approval or {}).get("approval_id"),
        "approval_actor": (approval or {}).get("actor"),
        "revalidated_at": datetime.now(UTC).isoformat(),
    }
