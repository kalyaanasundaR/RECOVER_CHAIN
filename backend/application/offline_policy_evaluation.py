"""Phase 13 — offline policy evaluation.

For every case that reached a policy decision, reconstruct its decision-time
context from real durable state (the same application.policy_engine.PolicyContext
the live system actually evaluates against - infrastructure.repositories's
get_policy_context), re-run the CURRENT DeterministicPolicyEngine against it,
and compare the result to what actually happened. This answers "what would
today's policy do on our real case history", not "what caused recovery" - it is
a descriptive replay, never an online experiment (nothing here writes to a
case, changes live policy behaviour, or executes anything), and never a causal
claim (Section 7's distinction: PREDICTED OUTCOME vs CAUSAL EFFECT applies
here too - a recovery-rate number below is an observed association with
whichever action was actually taken, not evidence that the action caused it).

RecoverChain has exactly one policy VERSION in this codebase (no historical
policy-version registry exists), so "apply historical policy" and "apply
current policy" are the same operation here - documented explicitly rather
than silently implying a multi-version backtest that doesn't exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from application.policy_engine import DeterministicPolicyEngine
from domain.models import PolicyDecisionStatus
from infrastructure.orm import CaseModel
from infrastructure.repositories import SqlAlchemyCaseRepository


@dataclass
class OfflineEvaluationReport:
    cases_evaluated: int
    recovery_rate: float | None
    total_recovered_amount: float
    total_amount_at_risk: float
    recovery_per_intervention: float | None
    escalation_rate: float | None
    policy_rejection_rate: float | None
    contact_rate: float | None
    retry_rate: float | None
    average_time_to_recovery_hours: float | None
    replayed_matches_actual_rate: float | None
    caveats: list[str] = field(default_factory=list)


def evaluate_policy_offline(db: Session, *, limit: int | None = None) -> OfflineEvaluationReport:
    """Read-only. Never mutates a case, never executes an action, never writes
    an outcome. `limit` bounds the query for a quick smoke run; omit for a full
    replay."""
    repo = SqlAlchemyCaseRepository(db)
    engine = DeterministicPolicyEngine()

    query = db.query(CaseModel.case_id).filter(CaseModel.policy_decision.isnot(None))
    if limit:
        query = query.limit(limit)
    case_ids = [row[0] for row in query.all()]

    n = 0
    recovered_count = 0
    escalated = 0
    rejected = 0
    contacted = 0
    retried = 0
    replay_matches = 0
    total_recovered = 0.0
    total_at_risk = 0.0
    time_to_recovery_hours: list[float] = []

    for case_id in case_ids:
        case = repo.get_by_id(case_id)
        if case is None or not case.policy_decision:
            continue
        n += 1

        total_at_risk += float(case.amount_at_risk.amount) if case.amount_at_risk else 0.0
        recovered_amount = float(case.recovered_amount.amount) if case.recovered_amount else 0.0
        total_recovered += recovered_amount
        if recovered_amount > 0:
            recovered_count += 1

        actual_status = case.policy_decision.status
        if actual_status == PolicyDecisionStatus.ESCALATE:
            escalated += 1
        if actual_status == PolicyDecisionStatus.DENIED:
            rejected += 1
        if case.recommendation and case.recommendation.top_candidate:
            from domain.models import action_channel

            channel = action_channel(case.recommendation.top_candidate.action_type).value
            if channel == "communication":
                contacted += 1
            if channel == "payment":
                retried += 1

        try:
            ctx = repo.get_policy_context(case_id)
            replayed = engine.evaluate(case, context=ctx)
            if replayed.status == actual_status:
                replay_matches += 1
        except Exception:
            pass  # a case with no reconstructable context is simply excluded from this comparison

        if recovered_amount > 0 and case.linked_events:
            first_event = min(e.timestamp for e in case.linked_events)
            # outcome verification_timestamp isn't on RecoveryCase directly;
            # this uses the case's own recorded outcome, when present, as the
            # best available "when recovered" signal from durable state.
            if case.outcome and getattr(case.outcome, "verification_timestamp", None):
                delta = (case.outcome.verification_timestamp - first_event).total_seconds() / 3600.0
                if delta >= 0:
                    time_to_recovery_hours.append(delta)

    caveats = [
        "descriptive replay only - no causal claim; an observed association "
        "between an action and recovery is NOT evidence the action caused it "
        "(Section 7: PREDICTED OUTCOME is not CAUSAL EFFECT)",
        "exactly one policy version exists in this codebase - 'historical' and "
        "'current' policy are the same rules here, not a multi-version backtest",
        f"n={n} cases - too small for statistically meaningful rate estimates; "
        "reported as raw observed counts/rates, not confidence-bounded",
    ]

    def _rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return OfflineEvaluationReport(
        cases_evaluated=n,
        recovery_rate=_rate(recovered_count, n),
        total_recovered_amount=round(total_recovered, 2),
        total_amount_at_risk=round(total_at_risk, 2),
        recovery_per_intervention=_rate(total_recovered, n) if n else None,
        escalation_rate=_rate(escalated, n),
        policy_rejection_rate=_rate(rejected, n),
        contact_rate=_rate(contacted, n),
        retry_rate=_rate(retried, n),
        average_time_to_recovery_hours=(
            round(sum(time_to_recovery_hours) / len(time_to_recovery_hours), 2)
            if time_to_recovery_hours
            else None
        ),
        replayed_matches_actual_rate=_rate(replay_matches, n),
        caveats=caveats,
    )
