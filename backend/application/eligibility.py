"""Phase 8B — the training-data eligibility classifier (Step 8).

A case does not become training data merely because it exists. Every case with at
least one MATCHED outcome_events row is evaluated against a fixed set of criteria and
assigned exactly one of three statuses, each with the concrete reasons behind it:

ELIGIBLE              - genuinely usable as a future training/evaluation label
INELIGIBLE            - actively disqualified (non-real provenance, a data-quality
                         issue, a duplicate, an impossible temporal relationship)
INSUFFICIENT_EVIDENCE - real provenance and no disqualifying issue, but something
                         needed for the label is still missing/incomplete

Nothing here trains anything. This module only decides which case_ids WOULD be handed
to a future experiment, and records exactly why the rest were not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from application.data_readiness import run_quality_checks
from application.outcome_provenance import REAL_WORLD_VALUES
from infrastructure.orm import CaseModel, OutcomeEventModel

FEATURE_SCHEMA_VERSION = "features-v1.0"  # matches recovery_predictor.FeatureExtractor.VERSION
LABEL_SCHEMA_VERSION = "label-v1.0"  # case.current_state -> RECOVERED/PARTIAL/NOT/CANCELLED


class EligibilityStatus:
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class EligibilityResult:
    case_id: str
    status: str
    reasons: list[str] = field(default_factory=list)


def evaluate_case_eligibility(db: Session, case: CaseModel) -> EligibilityResult:
    reasons: list[str] = []

    outcomes = (
        db.query(OutcomeEventModel)
        .filter(
            OutcomeEventModel.case_id == case.case_id,
            OutcomeEventModel.match_status == "MATCHED",
        )
        .all()
    )
    if not outcomes:
        return EligibilityResult(
            case.case_id, EligibilityStatus.INELIGIBLE, ["no authoritative outcome event"]
        )

    # provenance: EVERY outcome touching this case must be real-world evidence. A
    # case with even one test/external-source-simulated event mixed in among real
    # ones is not a clean label (which figure would a future experiment even trust?).
    provenances = {o.provenance for o in outcomes}
    non_real = provenances - REAL_WORLD_VALUES
    if non_real:
        reasons.append(f"non-production provenance present: {sorted(non_real)}")

    # duplicate status: a genuinely duplicated external id should be structurally
    # impossible (unique constraint), but re-verify against the live quality checks
    # rather than assuming.
    quality_issues = {i.outcome_event_id: i for i in run_quality_checks(db)}
    case_issue_ids = {o.outcome_event_id for o in outcomes} & quality_issues.keys()
    if case_issue_ids:
        reasons.append(
            f"data-quality issue(s) on outcome_event(s): "
            f"{sorted(quality_issues[i].issue for i in case_issue_ids)}"
        )

    # obligation identity: a case whose matched outcome carries no reference_id could
    # not have been deterministically correlated to this obligation with confidence
    # (domain/revenue_domains.py::obligation_key requires one) - such a case only
    # reaches here via an obligation-less single-event correlation, which is weaker
    # identity evidence for a training row.
    if any(o.reference_id is None for o in outcomes):
        reasons.append(
            "at least one matched outcome has no reference_id (weak obligation identity)"
        )

    # temporal validity: every outcome must have occurred at/after the case's own
    # creation (the same rule run_quality_checks already applies) - already covered
    # above via quality_issues, not duplicated here.

    # currency: every outcome must carry a currency when it carries money.
    for o in outcomes:
        if o.kind == "SETTLEMENT" and o.amount is not None and not o.currency:
            reasons.append(f"missing currency on {o.outcome_event_id}")

    # outcome completeness / cancellation: the case's financial position must be
    # internally consistent (recovered + outstanding + cancelled == amount_at_risk),
    # otherwise the "label" itself is not trustworthy.
    if case.amount_at_risk is not None:
        recovered = case.recovered_amount or 0
        outstanding = (
            case.outstanding_amount if case.outstanding_amount is not None else case.amount_at_risk
        )
        cancelled = case.cancelled_amount or 0
        if recovered + outstanding + cancelled != case.amount_at_risk:
            reasons.append("financial position does not reconcile to amount_at_risk")

    # required features: the deterministic feature set needs a diagnosis and a risk
    # assessment to have actually run (both required for FeatureExtractor).
    if case.diagnosis is None:
        reasons.append("missing diagnosis (required feature source)")
    if case.risk_assessment is None:
        reasons.append("missing risk_assessment (required feature source)")

    # label availability: the case must be in a genuinely TERMINAL state for its
    # outcome to be a stable label - PENDING_VERIFICATION is real evidence of
    # "nothing yet", not a usable positive or negative label.
    terminal_states = {
        "FULLY_RECOVERED",
        "PARTIALLY_RECOVERED",
        "CANCELLED",
        "CLOSED_NOT_RECOVERED",
    }
    if case.current_state not in terminal_states:
        reasons.append(f"case not in a terminal state (current: {case.current_state})")

    if non_real or case_issue_ids:
        return EligibilityResult(case.case_id, EligibilityStatus.INELIGIBLE, reasons)
    if reasons:  # real provenance, no disqualifying issue, but something is missing
        return EligibilityResult(case.case_id, EligibilityStatus.INSUFFICIENT_EVIDENCE, reasons)
    return EligibilityResult(case.case_id, EligibilityStatus.ELIGIBLE, ["all criteria satisfied"])


def evaluate_all(db: Session) -> list[EligibilityResult]:
    """Every case that has at least one MATCHED outcome event, evaluated."""
    case_ids = (
        db.query(OutcomeEventModel.case_id)
        .filter(OutcomeEventModel.match_status == "MATCHED", OutcomeEventModel.case_id.isnot(None))
        .distinct()
        .all()
    )
    results = []
    for (case_id,) in case_ids:
        case = db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
        if case is not None:
            results.append(evaluate_case_eligibility(db, case))
    return results
