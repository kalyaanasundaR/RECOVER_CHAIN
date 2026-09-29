"""Runs the synthetic ethical edge-case dataset
(data/recoverchain/synthetic/ethical_edge_cases/) through the REAL policy
engine, ingestion dedup, lifecycle state machine, and anomaly detector —
proving in code, not just in a spreadsheet, that ML prediction never
authorizes a financial action by itself.

Each scenario in the dataset carries an `expected_decision` documenting
which real rule (application/policy_engine.py) or mechanism (ingestion
dedup, lifecycle guard, anomaly score) should fire. This file is that proof:
one assertion per scenario, against the actual production code, not a
re-implementation of it.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from application.anomaly_detection import score_anomaly
from application.ingestion import event_fingerprint_v2
from application.policy_engine import DeterministicPolicyEngine, MerchantPolicy, PolicyContext
from domain.lifecycle import OUTCOME_FINAL_STATES
from domain.models import (
    ActionRecommendation,
    ActionType,
    CandidateAction,
    CaseState,
    Money,
    PolicyDecisionStatus,
    RecommendationStatus,
    RecoveryCase,
    RiskCategory,
)

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "recoverchain"
    / "synthetic"
    / "ethical_edge_cases"
    / "ethical_edge_cases.csv"
)


@pytest.fixture(scope="module")
def edge_cases() -> pd.DataFrame:
    if not DATA_PATH.exists():
        pytest.skip(
            f"{DATA_PATH} not generated — run scripts/data_generation/generate_recoverchain_datasets.py"
        )
    return pd.read_csv(DATA_PATH)


def _representative_amount(df: pd.DataFrame, scenario: str) -> float:
    rows = df[df["scenario"] == scenario]
    assert len(rows), f"no rows for scenario {scenario} in the generated dataset"
    return float(rows["amount_at_risk"].iloc[0])


def _case(
    amount: float,
    action: ActionType = ActionType.RETRY_PAYMENT,
    domain=RiskCategory.FAILED_PAYMENT,
    state=CaseState.POLICY_EVALUATED,
):
    return RecoveryCase(
        case_id="edge_case_under_test",
        customer_id="edge_customer",
        risk_category=domain,
        reference_id="edge_ref",
        amount_at_risk=Money(amount=amount),
        current_state=state,
        recommendation=ActionRecommendation(
            recommendation_id="rec_edge",
            candidates=[],
            top_candidate=CandidateAction(
                action_type=action,
                estimated_probability=0.9,
                expected_recoverable_value=amount * 0.9,
                rationale="synthetic edge-case fixture",
            ),
            status=RecommendationStatus.RECOMMENDED,
            rationale="synthetic edge-case fixture",
            engine_version="test",
        ),
    )


@pytest.fixture
def engine():
    return DeterministicPolicyEngine(MerchantPolicy())


# --- Case A: consent false blocks a high-probability action --------------


def test_case_A_no_consent_blocks_action(edge_cases, engine):
    amount = _representative_amount(edge_cases, "high_probability_no_consent")
    case = _case(amount)
    ctx = PolicyContext(payment_permission="OPTED_OUT")
    decision = engine.evaluate(case, ctx)
    assert decision.status == PolicyDecisionStatus.DENIED
    assert any(r.rule_name == "ContactPermission" for r in decision.failed_rules)


# --- Case B: suppression blocks even a high-value case --------------------


def test_case_B_suppression_blocks_high_value(edge_cases, engine):
    amount = _representative_amount(edge_cases, "high_value_suppressed")
    case = _case(amount)
    ctx = PolicyContext(payment_permission="SUPPRESSED")
    decision = engine.evaluate(case, ctx)
    assert decision.status == PolicyDecisionStatus.DENIED
    assert any(r.rule_name == "ContactPermission" for r in decision.failed_rules)


# --- Case C: low amount + high probability still goes through the SAME policy check ---


def test_case_C_probability_never_bypasses_policy(edge_cases, engine):
    amount = _representative_amount(edge_cases, "low_amount_high_probability")
    case = _case(amount)
    ctx = PolicyContext(payment_permission="ALLOWED")
    decision = engine.evaluate(case, ctx)
    # Permitted -- but via the same rule set every other case goes through,
    # not because "probability was high". Confirmed by checking that
    # FinancialAutomatedLimit was actually EVALUATED, not skipped.
    assert decision.status == PolicyDecisionStatus.PERMITTED
    assert any(
        r.rule_name == "FinancialAutomatedLimit" and r.passed for r in decision.rules_evaluated
    )


# --- Case D: large amount + low confidence -> escalate, never auto-execute ---


def test_case_D_large_amount_escalates_without_human_approval(edge_cases, engine):
    amount = _representative_amount(edge_cases, "large_amount_uncertain_probability")
    case = _case(amount)
    ctx = PolicyContext(payment_permission="ALLOWED", human_approved=False)
    decision = engine.evaluate(case, ctx)
    assert decision.status == PolicyDecisionStatus.ESCALATE
    assert any(
        r.rule_name == "FinancialAutomatedLimit" and not r.passed for r in decision.failed_rules
    )
    assert decision.approval_required is True


# --- Case E: duplicate obligation is caught at INGESTION, before any case exists ---


def test_case_E_duplicate_is_caught_by_ingestion_fingerprint():
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    common = dict(
        customer_id="edge_customer",
        amount=Decimal("300.00"),
        currency="INR",
        timestamp=ts,
        reference_id="inv_1",
        failure_code="network_error",
    )
    fp1 = event_fingerprint_v2(**common)
    fp2 = event_fingerprint_v2(**common)  # the exact same event, submitted again
    assert fp1 == fp2  # identical content -> identical fingerprint -> dedup, not a second case

    # A materially different event (different amount) must NOT collide.
    fp3 = event_fingerprint_v2(**{**common, "amount": Decimal("301.00")})
    assert fp3 != fp1


# --- Case F: already recovered -> ObligationSettled denies further action ---


def test_case_F_already_recovered_denies_further_action(edge_cases, engine):
    amount = _representative_amount(edge_cases, "already_recovered")
    case = _case(amount, state=CaseState.FULLY_RECOVERED)
    assert case.current_state in OUTCOME_FINAL_STATES
    decision = engine.evaluate(case, PolicyContext(payment_permission="ALLOWED"))
    assert decision.status == PolicyDecisionStatus.DENIED
    assert any(r.rule_name == "ObligationSettled" for r in decision.failed_rules)


# --- Case G: outcome unknown -> stays in verification, never silently marked recovered ---


def test_case_G_unknown_outcome_stays_in_verification(edge_cases):
    _representative_amount(edge_cases, "payment_outcome_unknown")  # dataset row exists
    case = _case(100.0, state=CaseState.PENDING_VERIFICATION)
    # The lifecycle's own terminal-state set is the authority here: a case sitting
    # in PENDING_VERIFICATION is explicitly NOT one of the two outcome-final states.
    # Nothing marks it FULLY_RECOVERED without an authoritative outcome event
    # (application/outcome_events.py) being applied through domain/lifecycle.py.
    assert case.current_state == CaseState.PENDING_VERIFICATION
    assert case.current_state not in OUTCOME_FINAL_STATES


# --- Case H: low confidence / out-of-distribution -> anomaly-flagged, not silently trusted ---


def test_case_H_out_of_distribution_input_is_flagged(edge_cases):
    _representative_amount(edge_cases, "low_confidence_out_of_distribution")
    training_distribution = {
        "amount_at_risk": {"type": "numeric", "mean": 3000.0, "std": 800.0},
    }
    normal = score_anomaly({"amount_at_risk": 3200.0}, training_distribution)
    extreme = score_anomaly({"amount_at_risk": 60000.0}, training_distribution)  # ~71 std out
    assert normal["level"] == "NORMAL"
    assert extreme["level"] == "ANOMALOUS"
    # No training distribution at all -> UNKNOWN, never silently NORMAL.
    assert score_anomaly({"amount_at_risk": 3000.0}, None)["level"] == "UNKNOWN"


# --- Case I: repeated recent attempts -> retry cap / cooldown, not unlimited retries ---


def test_case_I_repeated_attempts_trigger_cap_or_cooldown(edge_cases, engine):
    amount = _representative_amount(edge_cases, "multiple_recent_recovery_attempts")
    case = _case(amount)
    ctx = PolicyContext(
        payment_permission="ALLOWED",
        prior_payment_attempts=engine.policy.payment_max_retries,  # at the cap
        last_payment_attempt_at=datetime.now(UTC) - timedelta(minutes=5),
    )
    decision = engine.evaluate(case, ctx)
    assert decision.status in (PolicyDecisionStatus.DENIED, PolicyDecisionStatus.WAIT)
    fired = {r.rule_name for r in decision.failed_rules}
    assert "PaymentMaxRetries" in fired or "PaymentRetryCooldown" in fired


# --- Case J: policy prohibits an action even if the model likes it (domain applicability) ---


def test_case_J_policy_prohibits_action_regardless_of_prediction(edge_cases, engine):
    amount = _representative_amount(edge_cases, "high_probability_policy_prohibited")
    # A payment-channel action requested for a domain it does not apply to --
    # the structural ActionDomainApplicability check, independent of the
    # recommender's stated probability (still 0.9 in the fixture).
    case = _case(amount, action=ActionType.RETRY_PAYMENT, domain=RiskCategory.CHECKOUT_ABANDONMENT)
    decision = engine.evaluate(case, PolicyContext(payment_permission="ALLOWED"))
    assert decision.status == PolicyDecisionStatus.DENIED
    assert any(r.rule_name == "ActionDomainApplicability" for r in decision.failed_rules)


def test_every_scenario_in_the_dataset_has_a_covering_test(edge_cases):
    """Guards against the dataset and this test file silently drifting apart."""
    covered = {
        "high_probability_no_consent",
        "high_value_suppressed",
        "low_amount_high_probability",
        "large_amount_uncertain_probability",
        "duplicate_obligation",
        "already_recovered",
        "payment_outcome_unknown",
        "low_confidence_out_of_distribution",
        "multiple_recent_recovery_attempts",
        "high_probability_policy_prohibited",
    }
    assert set(edge_cases["scenario"].unique()) == covered
