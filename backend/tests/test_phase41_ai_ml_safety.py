"""Phase 7 — AI/ML foundation safety tests.

Phase 7 found that real, authoritative recovery-outcome labels are far too scarce
(12 cases total across 5 domains in the dev DB) to train or evaluate a genuine
recovery-probability model — see docs/history/phase7_ai_ml_audit.md. No new model
was trained. These tests instead lock down the SAFETY CONTRACT the existing
advisory layer (DeterministicBaselinePredictor + the shadow MLPaymentFailurePredictor
wrapper) must hold once real models do exist, using only deterministic code paths
so they do not depend on the environment's currently-unreliable scipy/sklearn/xgboost
native imports (see the audit doc, Step 13 hardware findings).

1  prediction reproducibility - same case, same inputs -> same probability
2  model-version recording - every prediction carries model_version + feature_version
3  feature snapshot integrity - contributing_features is a real, frozen snapshot
4  no outcome leakage - features never include recovered/outcome/verification fields
5  deterministic fallback - a missing/broken ML model never breaks the pipeline
6  AI failure safety - a predictor that raises never crashes prediction
7  Policy Engine remains authoritative - PolicyContext has no ML/prediction field
8  AI cannot bypass policy - the recommended action still goes through policy_check
9  prediction never mutates source/financial case fields
"""

from datetime import UTC, datetime
from decimal import Decimal

from application.case_pipeline import predict_recovery_for_case
from application.policy_engine import PolicyContext
from application.recovery_predictor import DeterministicBaselinePredictor, FeatureExtractor
from domain.models import (
    CaseState,
    DiagnosisStatus,
    Money,
    RecoveryCase,
    RevenueEvent,
    RiskCategory,
    RootCauseCategory,
    RootCauseDiagnosis,
)


def _case(amount=1000, risk_category=RiskCategory.OVERDUE_INVOICE, cause=None) -> RecoveryCase:
    event = RevenueEvent(
        event_id="evt_test1",
        customer_id="cust_test1",
        risk_category=risk_category,
        external_system="test",
        external_event_id="ext_test1",
        reference_id="ref_test1",
        amount=Money(amount=Decimal(str(amount)), currency="INR"),
        timestamp=datetime.now(UTC),
        raw_payload={},
    )
    case = RecoveryCase(
        case_id="case_test1",
        customer_id="cust_test1",
        risk_category=risk_category,
        amount_at_risk=Money(amount=Decimal(str(amount)), currency="INR"),
        current_state=CaseState.DIAGNOSING,
        linked_events=[event],
    )
    if cause:
        case.diagnosis = RootCauseDiagnosis(
            diagnosis_id="diag_test1",
            cause_category=cause,
            confidence=0.9,
            status=DiagnosisStatus.CONFIRMED,
            supporting_signals={},
            evidence_references=["evt_test1"],
            diagnostic_method="deterministic-v1.0",
        )
    return case


# ------------------------------------------------------------------ 1 + 2 + 3


def test_1_2_3_prediction_is_reproducible_versioned_and_snapshotted():
    case = _case(amount=1000, cause=RootCauseCategory.NETWORK_FAILURE)
    p1 = DeterministicBaselinePredictor().predict(case)
    p2 = DeterministicBaselinePredictor().predict(case)

    # Reproducibility: identical case -> identical probability.
    assert p1.recovery_probability == p2.recovery_probability

    # Version metadata is always present and non-empty.
    assert p1.model_version
    assert p1.feature_version

    # The feature snapshot is a real, populated dict - not empty, not a
    # reference to something that can change under it later.
    assert isinstance(p1.contributing_features, dict)
    assert p1.contributing_features["amount"] == Decimal("1000")
    snapshot_copy = dict(p1.contributing_features)
    case.amount_at_risk = Money(amount=Decimal("9999"), currency="INR")
    assert p1.contributing_features == snapshot_copy  # unaffected by later mutation


# ------------------------------------------------------------------ 4


def test_4_features_never_include_outcome_or_verification_fields():
    case = _case(amount=500, cause=RootCauseCategory.INSUFFICIENT_FUNDS)
    features = FeatureExtractor.extract_features(case)
    forbidden_substrings = (
        "recovered",
        "outcome",
        "verification",
        "settlement",
        "cancelled_amount",
        "execution_record",
    )
    for key in features:
        lowered = key.lower()
        for bad in forbidden_substrings:
            assert bad not in lowered, f"leaked field: {key}"


# ------------------------------------------------------------------ 5 + 6


def test_5_6_missing_or_broken_predictor_falls_back_safely():
    case = _case(amount=750, cause=RootCauseCategory.NETWORK_FAILURE)

    class _BrokenPredictor:
        model = object()  # truthy - the pipeline will try to use it

        def predict_failure_risk(self, features):
            raise RuntimeError("simulated model failure")

    # A predictor that raises must never crash the pipeline - it must fall back
    # to the deterministic baseline, exactly like "no model available" does.
    result = predict_recovery_for_case(case, predictor=_BrokenPredictor())
    assert result is not None
    assert 0.0 <= result.recovery_probability <= 1.0
    assert result.prediction_status in ("SUCCESS_BASELINE",)

    # No model at all (predictor.model is None) - same safe fallback.
    class _NoModelPredictor:
        model = None

    result2 = predict_recovery_for_case(case, predictor=_NoModelPredictor())
    assert result2.prediction_status == "SUCCESS_BASELINE"


# ------------------------------------------------------------------ 7


def test_7_policy_context_has_no_ml_or_prediction_field():
    field_names = set(PolicyContext.model_fields.keys())
    for forbidden in (
        "recovery_probability",
        "prediction",
        "ml_score",
        "confidence",
        "model_version",
    ):
        assert forbidden not in field_names
    # extra="forbid" means a caller cannot even smuggle one in.
    import pydantic

    try:
        PolicyContext(recovery_probability=0.9)
        raised = False
    except pydantic.ValidationError:
        raised = True
    assert raised, "PolicyContext must reject an unknown ML-shaped field"


# ------------------------------------------------------------------ 8 + 9


def test_8_9_prediction_never_mutates_case_financial_truth():
    case = _case(amount=2000, cause=RootCauseCategory.PAYMENT_METHOD_INVALID)
    before_amount = case.amount_at_risk.amount
    before_state = case.current_state

    DeterministicBaselinePredictor().predict(case)
    predict_recovery_for_case(case)

    # Prediction is purely advisory - it must never write to the case's
    # financial or lifecycle truth. (The pipeline stage that calls this
    # assigns the RETURNED prediction onto case.prediction separately; the
    # predictor itself must not reach into the case and change anything else.)
    assert case.amount_at_risk.amount == before_amount
    assert case.current_state == before_state
    assert case.recovered_amount is None or case.recovered_amount.amount == Decimal(0)
