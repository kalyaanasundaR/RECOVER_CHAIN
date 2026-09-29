"""Phase 12 — advanced ML/DL: explainability, confidence, anomaly, and the new
model-failure-safety guards (NaN / out-of-range probability).

1  explain_sklearn_pipeline: perturbation-based contributions are real (derived
   from actual predict_proba calls on the actual trained pipeline), sorted by
   magnitude, and change when the underlying feature actually matters
2  explain_legacy_xgboost: exact native contributions from a real Booster
3  confidence_tier: honest, documented-methodology bucketing (not fabricated)
4  to_human_readable: MODEL ASSOCIATION language only, never causal claims,
   only ever describes features that are actually in the explanation
5  score_anomaly: NORMAL / UNUSUAL / ANOMALOUS / UNKNOWN from real z-scores
   against a real training distribution
6  end-to-end: a live trained shadow model's predict_failure_risk() carries
   explanation + confidence_tier + anomaly in its metadata
7  NaN / out-of-range probability is treated as a prediction FAILURE (falls
   back to the deterministic baseline via case_pipeline), never silently
   clamped or passed through as if valid
8  reproducibility: same case, same model -> same probability AND same
   explanation (determinism, not just "runs")
"""

import math

import pandas as pd
import pytest

from application.anomaly_detection import score_anomaly
from application.explainability import (
    confidence_tier,
    explain_sklearn_pipeline,
    to_human_readable,
)
from application.ml_training import MLTrainingEngine
from application.recovery_predictor_ml import MLPaymentFailurePredictor


@pytest.fixture
def mock_spec():
    return {
        "dataset_id": "ds_p12",
        "prediction_problem": "payment-failure-risk",
        "target_column": "target_recovered",
        "feature_columns": ["amount", "noise"],
        "canonical_feature_mapping": {"amount": "AMOUNT", "noise": "UNKNOWN"},
    }


@pytest.fixture
def mock_data(tmp_path):
    # `amount` is genuinely predictive (strictly separates the classes);
    # `noise` is random and should contribute far less.
    import random

    random.seed(42)
    n = 60
    rows = []
    for i in range(n):
        recovered = i % 2
        amount = 100.0 + recovered * 500.0  # clean separation
        rows.append({"amount": amount, "noise": random.random(), "target_recovered": recovered})
    df = pd.DataFrame(rows)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def trained_predictor(mock_spec, mock_data, tmp_path):
    engine = MLTrainingEngine(mock_spec, str(mock_data), str(tmp_path))
    engine.train_and_evaluate()
    return MLPaymentFailurePredictor(dataset_id="ds_p12", registry_dir=str(tmp_path))


# ------------------------------------------------------------------ 1


def test_1_explain_sklearn_pipeline_identifies_the_predictive_feature(trained_predictor):
    mapped = {"amount": 600.0, "noise": 0.5}
    explanation = explain_sklearn_pipeline(
        trained_predictor.model, trained_predictor.features, mapped
    )

    assert explanation, "expected at least one contribution"
    by_feature = {c["feature"]: c for c in explanation}
    assert "amount" in by_feature
    # `amount` was constructed to be strongly predictive; `noise` was not.
    if "noise" in by_feature and by_feature["noise"]["contribution"] is not None:
        assert abs(by_feature["amount"]["contribution"]) >= abs(by_feature["noise"]["contribution"])
    assert by_feature["amount"]["direction"] in ("increased", "decreased", "no_effect")


# ------------------------------------------------------------------ 3


def test_3_confidence_tier_is_low_without_quality_gate_evidence():
    assert confidence_tier(0.5, {}) == "LOW"
    assert confidence_tier(0.5, {"quality_gate": {}}) == "LOW"


def test_3b_confidence_tier_is_high_only_with_wide_margin_and_full_quality_gate():
    good_gate = {"quality_gate": {"passed": True, "small_sample": False}}
    assert confidence_tier(0.95, good_gate) == "HIGH"
    assert confidence_tier(0.50, good_gate) == "LOW"  # near decision boundary, still risky


# ------------------------------------------------------------------ 4


def test_4_human_readable_explanation_only_describes_real_contributions_and_never_claims_causation():
    explanation = [
        {"feature": "amount", "value": 600.0, "contribution": 0.12, "direction": "increased"},
        {"feature": "noise", "value": 0.5, "contribution": 0.0001, "direction": "increased"},
    ]
    lines = to_human_readable(explanation, probability=0.7)
    text = " ".join(lines)
    assert "amount" in text
    # noise's contribution (0.0001) is below the surfacing threshold
    assert "noise" not in text
    assert "MODEL ASSOCIATION" in text
    assert "caused" not in text.lower()
    assert "causation" in text.lower()  # explicitly disclaimed, not silently omitted


def test_4b_human_readable_explanation_handles_no_measurable_effect_honestly():
    lines = to_human_readable([], probability=0.5)
    assert lines
    assert "no" in lines[0].lower()


# ------------------------------------------------------------------ 5


def test_5_score_anomaly_flags_a_genuine_outlier():
    training_distribution = {
        "amount": {"type": "numeric", "mean": 100.0, "std": 10.0, "count": 50},
        "noise": {"type": "categorical", "count": 50},  # non-numeric, ignored
    }
    normal = score_anomaly({"amount": 105.0}, training_distribution)
    assert normal["level"] == "NORMAL"

    outlier = score_anomaly({"amount": 1000.0}, training_distribution)
    assert outlier["level"] == "ANOMALOUS"
    assert outlier["max_abs_z"] > 3.5


def test_5b_score_anomaly_is_unknown_without_a_training_distribution():
    assert score_anomaly({"amount": 1.0}, None)["level"] == "UNKNOWN"
    assert score_anomaly({"amount": 1.0}, {})["level"] == "UNKNOWN"


# ------------------------------------------------------------------ 6


def test_6_live_prediction_carries_explanation_confidence_and_anomaly(trained_predictor):
    res = trained_predictor.predict_failure_risk({"amount": 600.0, "noise": 0.5})
    assert res["status"] == "SUCCESS"
    meta = res["model_metadata"]
    assert "explanation" in meta and meta["explanation"] is not None
    assert meta["confidence_tier"] in ("LOW", "MEDIUM", "HIGH")
    assert "anomaly" in meta and meta["anomaly"]["level"] in (
        "NORMAL",
        "UNUSUAL",
        "ANOMALOUS",
        "UNKNOWN",
    )


def test_6b_extreme_feature_value_is_flagged_anomalous_end_to_end(trained_predictor):
    res = trained_predictor.predict_failure_risk({"amount": 999999.0, "noise": 0.5})
    assert res["status"] == "SUCCESS"
    assert res["model_metadata"]["anomaly"]["level"] in ("UNUSUAL", "ANOMALOUS")


# ------------------------------------------------------------------ 7


class _NaNModel:
    is_legacy = False
    features = ["amount"]
    canonical_mapping = {"amount": "AMOUNT"}
    metadata = {}

    def __init__(self, bad_value):
        self.model = self
        self.bad_value = bad_value

    def predict_proba(self, df):
        import numpy as np

        return np.array([[1 - self.bad_value, self.bad_value]])

    _map_canonical_to_original_features = (
        MLPaymentFailurePredictor._map_canonical_to_original_features
    )
    predict_failure_risk = MLPaymentFailurePredictor.predict_failure_risk


@pytest.mark.parametrize("bad_value", [float("nan"), -0.2, 1.5, float("inf")])
def test_7_invalid_probability_is_rejected_not_silently_clamped(bad_value):
    predictor = _NaNModel(bad_value)
    result = predictor.predict_failure_risk({"AMOUNT": 100.0})
    assert result["status"] == "INVALID_PROBABILITY"
    assert result["probability"] == 0.0  # a safe, explicit non-prediction, not the bad value


def test_7b_invalid_probability_falls_back_to_deterministic_baseline_via_pipeline():
    from datetime import UTC, datetime
    from decimal import Decimal

    from application.case_pipeline import predict_recovery_for_case
    from domain.models import CaseState, Money, RecoveryCase, RevenueEvent, RiskCategory

    event = RevenueEvent(
        event_id="evt_nan",
        customer_id="cust_nan",
        risk_category=RiskCategory.FAILED_PAYMENT,
        external_system="test",
        external_event_id="ext_nan",
        reference_id="ref_nan",
        amount=Money(amount=Decimal("100"), currency="INR"),
        timestamp=datetime.now(UTC),
        raw_payload={},
    )
    case = RecoveryCase(
        case_id="case_nan",
        customer_id="cust_nan",
        risk_category=RiskCategory.FAILED_PAYMENT,
        amount_at_risk=Money(amount=Decimal("100"), currency="INR"),
        current_state=CaseState.DIAGNOSING,
        linked_events=[event],
    )
    predictor = _NaNModel(float("nan"))
    result = predict_recovery_for_case(case, predictor=predictor)
    assert result.prediction_status == "SUCCESS_BASELINE"
    assert 0.0 <= result.recovery_probability <= 1.0
    assert not math.isnan(result.recovery_probability)


# ------------------------------------------------------------------ 8


def test_8_prediction_and_explanation_are_reproducible(trained_predictor):
    features = {"amount": 600.0, "noise": 0.5}
    r1 = trained_predictor.predict_failure_risk(dict(features))
    r2 = trained_predictor.predict_failure_risk(dict(features))
    assert r1["probability"] == r2["probability"]
    assert r1["model_metadata"]["explanation"] == r2["model_metadata"]["explanation"]


# ------------------------------------------------------------------ 2


def test_2_explain_legacy_xgboost_contributions_are_exact_and_additive():
    """XGBoost's pred_contribs are exact by construction: sum(contribs) + bias
    == the raw margin prediction. This tests that invariant survives our sign
    flip (failure-risk contribs -> recovery-probability contribs) and feature
    ordering, using a tiny real trained Booster (not a mock)."""
    import numpy as np
    import xgboost as xgb

    from application.explainability import explain_legacy_xgboost

    rng = np.random.default_rng(7)
    X = rng.random((200, 3))
    y = (X[:, 0] > 0.5).astype(int)  # feature 0 is genuinely predictive
    dtrain = xgb.DMatrix(X, label=y, feature_names=["f0", "f1", "f2"])
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3}, dtrain, num_boost_round=20
    )

    mapped = {"f0": 0.9, "f1": 0.3, "f2": 0.1}
    explanation = explain_legacy_xgboost(booster, ["f0", "f1", "f2"], mapped, top_k=3)
    assert len(explanation) == 3
    by_feature = {c["feature"]: c["contribution"] for c in explanation}

    # Recompute raw contribs directly to check our sign-flip against ground truth.
    dmat = xgb.DMatrix(np.array([[0.9, 0.3, 0.1]]), feature_names=["f0", "f1", "f2"])
    raw = booster.predict(dmat, pred_contribs=True)[0]
    for i, f in enumerate(["f0", "f1", "f2"]):
        assert by_feature[f] == pytest.approx(-float(raw[i]), abs=1e-6)

    # f0 is the genuinely predictive feature - it should dominate.
    assert abs(by_feature["f0"]) >= abs(by_feature["f1"])
    assert abs(by_feature["f0"]) >= abs(by_feature["f2"])
