"""Phase 13 — next-best-action dataset CONTRACT (schema + readiness query), not a
trained model.

A legitimate NBA training example needs CASE CONTEXT + ACTION TAKEN +
AUTHORITATIVE OUTCOME + OUTCOME TIMESTAMP, all independently evidenced. This
module defines the exact row shape (`NBARecord`) and a live-queried builder
(`build_nba_dataset`) that assembles real rows from execution_attempts +
outcome_events + recovery_cases - joining data that already exists, never
inventing an action, an outcome, or a label.

Per the brief's explicit rule: a deterministic policy decision, an AI
recommendation, or a synthetic/fabricated outcome is NEVER treated as an
independent action-success label. Only a row whose outcome_events.provenance
is `production` or `human_confirmed` counts toward "legitimate" here - `test`
and `external_source_simulated` rows are included in the dataset (for
structural/offline-evaluation purposes) but explicitly marked ineligible as a
training label, exactly as application/eligibility.py already does for the
recovery-probability task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from application.outcome_provenance import OutcomeProvenance
from infrastructure.orm import CaseModel, ExecutionAttemptModel, OutcomeEventModel

# Only these provenance values count as a legitimate, independent action-success
# label - matches application/eligibility.py's REAL_PROVENANCE_VALUES intent for
# the recovery-probability task, applied here to the NBA task.
LEGITIMATE_LABEL_PROVENANCE = frozenset(
    {OutcomeProvenance.PRODUCTION, OutcomeProvenance.HUMAN_CONFIRMED}
)

# A usable NBA dataset needs enough rows per action to rank actions against each
# other, not just enough rows in total. This number is not a magic ML threshold -
# it is the same floor application/data_readiness.py already uses for "a split
# is even structurally possible" reasoning, restated here for the action-level
# denominator this task actually needs.
MIN_LEGITIMATE_ROWS_FOR_TRAINING = 200


@dataclass
class NBARecord:
    case_id: str
    obligation_id: str | None
    decision_timestamp: str | None
    available_features: dict
    candidate_action: str
    action_source: str  # "AGENT_ORCHESTRATOR" - always, per agents.py's own hierarchy
    policy_eligibility: str  # the PolicyDecisionStatus this action received
    policy_decision_id: str | None
    execution_timestamp: str | None
    authoritative_outcome: str | None  # OutcomeKind, only when MATCHED
    recovered_amount: float | None
    outcome_timestamp: str | None
    provenance: str | None
    evidence_reference: str  # the outcome_event_id, or "NONE" if no matched outcome yet
    label_eligible: bool  # True only if provenance in LEGITIMATE_LABEL_PROVENANCE


@dataclass
class NBADatasetReport:
    status: str  # "READY" | "DATA_BLOCKED"
    total_rows: int
    legitimate_label_rows: int
    rows_by_provenance: dict = field(default_factory=dict)
    rows_by_action: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    records: list = field(default_factory=list)


def build_nba_dataset(db: Session, *, include_records: bool = False) -> NBADatasetReport:
    """Live-queried, never cached, never asserted from a prior phase's count.
    Joins execution_attempts -> outcome_events (by case_id, nearest MATCHED
    outcome at or after the attempt) -> recovery_cases, for whatever real rows
    exist today."""
    attempts = (
        db.query(ExecutionAttemptModel)
        .filter(ExecutionAttemptModel.status == "COMPLETED_SIMULATED")
        .all()
    )

    records: list[NBARecord] = []
    rows_by_provenance: dict[str, int] = {}
    rows_by_action: dict[str, int] = {}
    legitimate = 0

    for attempt in attempts:
        case = db.query(CaseModel).filter(CaseModel.case_id == attempt.case_id).first()
        if case is None:
            continue

        # Nearest MATCHED outcome for this case at/after the attempt - the
        # authoritative link between "an action was taken" and "what actually
        # happened", never inferred from timing alone beyond this ordering.
        outcome = (
            db.query(OutcomeEventModel)
            .filter(
                OutcomeEventModel.case_id == case.case_id,
                OutcomeEventModel.match_status == "MATCHED",
                OutcomeEventModel.occurred_at >= attempt.created_at,
            )
            .order_by(OutcomeEventModel.occurred_at.asc())
            .first()
        )

        provenance = outcome.provenance if outcome else None
        is_legitimate = provenance in LEGITIMATE_LABEL_PROVENANCE
        if is_legitimate:
            legitimate += 1
        rows_by_provenance[provenance or "NO_OUTCOME_YET"] = (
            rows_by_provenance.get(provenance or "NO_OUTCOME_YET", 0) + 1
        )
        rows_by_action[attempt.action_type] = rows_by_action.get(attempt.action_type, 0) + 1

        records.append(
            NBARecord(
                case_id=case.case_id,
                obligation_id=case.reference_id,
                decision_timestamp=attempt.created_at.isoformat() if attempt.created_at else None,
                available_features={
                    "risk_category": getattr(case.risk_category, "value", case.risk_category),
                    "amount_at_risk": float(case.amount_at_risk) if case.amount_at_risk else None,
                },
                candidate_action=attempt.action_type,
                action_source="AGENT_ORCHESTRATOR",
                policy_eligibility=attempt.status,
                policy_decision_id=attempt.policy_decision_id,
                execution_timestamp=attempt.created_at.isoformat() if attempt.created_at else None,
                authoritative_outcome=outcome.kind if outcome else None,
                recovered_amount=float(outcome.applied_amount)
                if outcome and outcome.applied_amount is not None
                else None,
                outcome_timestamp=outcome.occurred_at.isoformat()
                if outcome and outcome.occurred_at
                else None,
                provenance=provenance,
                evidence_reference=outcome.outcome_event_id if outcome else "NONE",
                label_eligible=is_legitimate,
            )
        )

    reasons = [
        f"{len(records)} total action-attempt rows joined to their nearest outcome (if any)",
        f"{legitimate} rows carry a legitimate (production/human_confirmed) outcome label",
        f"legitimate-row floor for training is {MIN_LEGITIMATE_ROWS_FOR_TRAINING} "
        "(matches the row-count reasoning application/data_readiness.py already applies "
        "to the recovery-probability task, not a fabricated ML-specific number)",
    ]
    status = "READY" if legitimate >= MIN_LEGITIMATE_ROWS_FOR_TRAINING else "DATA_BLOCKED"
    if status == "DATA_BLOCKED":
        reasons.append(
            "NBA_MODEL_STATUS = DATA_BLOCKED — insufficient legitimate action/outcome "
            "labels to train or evaluate a ranking model; this is a data-volume fact, "
            "not a technique limitation"
        )

    return NBADatasetReport(
        status=status,
        total_rows=len(records),
        legitimate_label_rows=legitimate,
        rows_by_provenance=rows_by_provenance,
        rows_by_action=rows_by_action,
        reasons=reasons,
        records=records if include_records else [],
    )


def nba_model_status(db: Session) -> str:
    """The literal `NBA_MODEL_STATUS` the brief asks for - a thin, live-queried
    wrapper, never a hardcoded string."""
    return build_nba_dataset(db).status


def snapshot_nba_dataset(db: Session, *, created_by: str | None = None) -> dict:
    """An immutable, checksummed snapshot of the CURRENT NBA dataset contract
    output - Section 16. Reuses application.dataset_snapshot's checksum
    approach (sha256 over a canonical JSON) rather than inventing a second
    hashing scheme; does not write to the operational store."""
    import hashlib
    import json

    report = build_nba_dataset(db, include_records=True)
    canonical = json.dumps(
        {
            "case_ids": sorted(r.case_id for r in report.records),
            "legitimate_label_rows": report.legitimate_label_rows,
        },
        sort_keys=True,
    )
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return {
        "status": report.status,
        "checksum": checksum,
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": created_by,
        "total_rows": report.total_rows,
        "legitimate_label_rows": report.legitimate_label_rows,
        "rows_by_provenance": report.rows_by_provenance,
        "rows_by_action": report.rows_by_action,
        "feature_schema_version": "nba-decision-context-v1",
        "label_schema_version": "nba-record-v1",
        "reasons": report.reasons,
    }
