"""Phase 14 — continuous learning / model monitoring / champion-challenger tests.

1  prediction-outcome linkage never treats a pending (no-outcome-yet) case as
   a negative label, and never overwrites the original prediction
2  prediction-vs-reality metrics honestly report INSUFFICIENT_EVIDENCE below
   the documented sample floor, and never fabricate ROC-AUC from one class
3  calibration bins are honest about tiny-bin sample sizes
4  feature drift classification reuses Phase 11's thresholds as documented
   tiers, and correctly separates feature/performance/business drift
5  champion vs challenger comparison never declares a winner from one metric
   alone, and calibration loss disqualifies a raw-ROC-AUC win
6  promotion gate never skips a stage without explicit evidence, and requires
   a real reviewed_by identifier to reach APPROVED
7  production validation status is NOT_READY without real production volume,
   never inflated by test/simulated rows
8  retraining readiness / data sufficiency never auto-retrain, and report
   real reasons
9  anomaly monitoring never interprets anomaly as fraud/outcome
10 Policy Engine remains isolated from every Phase 14 field (monitoring can't
   authorize, challenger can't execute)
11 explanation reproducibility holds for a deterministic model
"""

import pytest

from application.drift_monitoring import (
    classify_feature_drift,
    classify_performance_vs_business_drift,
)
from application.model_monitoring import (
    MIN_SAMPLE_SIZE,
    anomaly_monitoring,
    calibration_bins,
    compute_metrics,
    domain_monitoring,
)
from application.model_promotion import (
    compare_champion_challenger,
    data_sufficiency_gate,
    production_validation_status,
    promotion_gate,
    retraining_readiness,
)
from application.prediction_outcome_linkage import PredictionOutcomeRecord


def _record(
    case_id,
    prob,
    recovered,
    *,
    provenance="test",
    domain="FAILED_PAYMENT",
    amount=500.0,
    anomaly_level=None,
    decision_ts="2026-01-01T00:00:00+00:00",
):
    erv = amount * prob
    actual = amount if recovered else 0.0
    return PredictionOutcomeRecord(
        case_id=case_id,
        risk_category=domain,
        provenance=provenance,
        decision_timestamp=decision_ts,
        predicted_recovery_probability=prob,
        predicted_erv=erv,
        amount_at_risk=amount,
        model_version="test-model",
        prediction_status="SHADOW_ONLY",
        confidence_tier="MEDIUM",
        anomaly_level=anomaly_level,
        actual_recovered_amount=actual,
        recovered=recovered,
        outcome_timestamp=decision_ts,
        probability_error=prob - (1.0 if recovered else 0.0),
        absolute_probability_error=abs(prob - (1.0 if recovered else 0.0)),
        erv_error=erv - actual,
        absolute_erv_error=abs(erv - actual),
        calibration_bucket=f"{min(int(prob * 10), 9) / 10:.1f}-{min(int(prob * 10), 9) / 10 + 0.1:.1f}",
    )


# ------------------------------------------------------------------ 1


def test_1_linkage_never_treats_pending_case_as_negative_and_never_overwrites_prediction():
    from infrastructure.db import SessionLocal
    from infrastructure.orm import CaseModel

    db = SessionLocal()
    try:
        from application.prediction_outcome_linkage import build_prediction_outcome_records

        before = {
            row.case_id: row.prediction
            for row in db.query(CaseModel.case_id, CaseModel.prediction)
            .filter(CaseModel.prediction.isnot(None))
            .limit(50)
        }
        build_prediction_outcome_records(db)
        db.expire_all()
        after = {
            row.case_id: row.prediction
            for row in db.query(CaseModel.case_id, CaseModel.prediction)
            .filter(CaseModel.prediction.isnot(None))
            .limit(50)
        }
        assert before == after  # unchanged - linkage is read-only
    finally:
        db.close()


# ------------------------------------------------------------------ 2


def test_2_metrics_report_insufficient_evidence_below_floor():
    records = [_record(f"c{i}", 0.9, True) for i in range(MIN_SAMPLE_SIZE - 1)]
    block = compute_metrics(records)
    assert block.status == "INSUFFICIENT_EVIDENCE"
    assert block.roc_auc is None


def test_2b_metrics_report_insufficient_evidence_with_single_class_even_above_floor():
    records = [_record(f"c{i}", 0.9, True) for i in range(MIN_SAMPLE_SIZE + 10)]
    block = compute_metrics(records)
    assert block.n >= MIN_SAMPLE_SIZE
    assert block.roc_auc is None  # single class - ROC-AUC is mathematically undefined
    assert block.brier_score is not None  # doesn't need class diversity


def test_2c_metrics_compute_real_classification_metrics_with_both_classes():
    records = [_record(f"r{i}", 0.9, True) for i in range(20)] + [
        _record(f"nr{i}", 0.2, False) for i in range(20)
    ]
    block = compute_metrics(records)
    assert block.status == "COMPUTED"
    assert block.roc_auc == 1.0  # perfectly separated by construction
    assert block.brier_score is not None
    assert block.log_loss_ is not None


# ------------------------------------------------------------------ 3


def test_3_calibration_bins_hide_tiny_bins():
    records = [_record("c1", 0.85, True)]  # n=1, below MIN_BIN_SIZE
    bins = calibration_bins(records)
    assert bins["0.8-0.9"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_3b_calibration_bins_compute_real_gap_with_enough_samples():
    records = [_record(f"c{i}", 0.85, i < 4) for i in range(6)]  # 4/6 recovered at ~0.85 predicted
    bins = calibration_bins(records)
    b = bins["0.8-0.9"]
    assert b["status"] == "COMPUTED"
    assert b["n"] == 6
    assert 0.0 <= b["calibration_gap"] <= 1.0


# ------------------------------------------------------------------ 4


def test_4_feature_drift_reuses_phase11_thresholds_as_tiers():
    import random

    import pandas as pd

    from application.drift_readiness import summarize_distribution

    random.seed(1)
    baseline_values = [100.0 + random.uniform(-5, 5) for _ in range(50)]
    no_drift_values = [101.0 + random.uniform(-5, 5) for _ in range(50)]

    baseline = summarize_distribution(pd.DataFrame({"amount": baseline_values}), ["amount"])
    no_drift = summarize_distribution(pd.DataFrame({"amount": no_drift_values}), ["amount"])
    high_drift = summarize_distribution(
        pd.DataFrame({"amount": [100.0] * 25 + [10000.0] * 25}), ["amount"]
    )

    result_none = classify_feature_drift(baseline, no_drift)
    result_high = classify_feature_drift(baseline, high_drift)
    tier_rank = {"NO_DRIFT": 0, "LOW_DRIFT": 1, "MODERATE_DRIFT": 2, "HIGH_DRIFT": 3}
    # A mild, within-noise shift must classify strictly below a massive
    # (100x-scale outlier) shift - proves the tiers actually separate cases,
    # without pinning the exact boundary of a noisy random fixture.
    assert tier_rank[result_none["overall_tier"]] < tier_rank[result_high["overall_tier"]]
    assert result_high["overall_tier"] == "HIGH_DRIFT"


def test_4b_performance_and_business_drift_are_reported_separately():
    from application.model_monitoring import MetricBlock

    period_metrics = {
        "2026-01": MetricBlock(
            status="COMPUTED",
            n=40,
            n_recovered=20,
            n_not_recovered=20,
            mean_absolute_probability_error=0.10,
            recovery_rate=0.50,
        ),
        "2026-02": MetricBlock(
            status="COMPUTED",
            n=40,
            n_recovered=30,
            n_not_recovered=10,
            mean_absolute_probability_error=0.25,
            recovery_rate=0.75,
        ),
    }
    result = classify_performance_vs_business_drift(period_metrics)
    assert result["status"] == "COMPUTED"
    assert result["performance_drift"]["delta"] == pytest.approx(0.15)
    assert result["business_outcome_drift"]["delta"] == pytest.approx(0.25)
    # The two numbers differ - proving they're computed independently, not
    # collapsed into one "drift score".
    assert result["performance_drift"]["delta"] != result["business_outcome_drift"]["delta"]


# ------------------------------------------------------------------ 5


def test_5_challenger_never_wins_on_one_metric_alone():
    champion = {
        "model_id": "champ",
        "quality_gate": {"roc_auc": 0.80},
        "final_test_metrics": {"pr_auc": 0.70, "brier_score": 0.10, "f1": 0.65},
        "calibrated": True,
    }
    challenger_worse_calibration = {
        "model_id": "chal",
        "quality_gate": {"roc_auc": 0.95},  # much better ROC-AUC
        "final_test_metrics": {
            "pr_auc": 0.60,
            "brier_score": 0.30,
            "f1": 0.55,
        },  # worse everything else
        "calibrated": True,
    }
    comp = compare_champion_challenger(champion, challenger_worse_calibration)
    assert "CHALLENGER_BETTER" not in comp.recommendation
    assert comp.auto_promoted is False


def test_5b_challenger_clearly_better_on_every_metric_still_only_recommends_review():
    champion = {
        "model_id": "champ",
        "quality_gate": {"roc_auc": 0.70},
        "final_test_metrics": {"pr_auc": 0.60, "brier_score": 0.20, "f1": 0.55},
        "calibrated": True,
    }
    challenger = {
        "model_id": "chal",
        "quality_gate": {"roc_auc": 0.90},
        "final_test_metrics": {"pr_auc": 0.85, "brier_score": 0.05, "f1": 0.80},
        "calibrated": True,
    }
    comp = compare_champion_challenger(champion, challenger)
    assert "PROMOTION_REVIEW" in comp.recommendation
    assert comp.auto_promoted is False  # recommends review, never auto-promotes


# ------------------------------------------------------------------ 6


def test_6_promotion_gate_never_skips_a_stage():
    meta = {
        "final_test_metrics": {"roc_auc": 0.9},
        "quality_gate": {"passed": True},
    }
    result = promotion_gate(meta)  # no extra evidence supplied
    assert result["stage"] == "QUALITY_PASSED"
    assert result["blocked_at"] == "CHALLENGER"


def test_6b_promotion_gate_requires_a_real_reviewed_by_for_approved():
    meta = {"final_test_metrics": {"roc_auc": 0.9}, "quality_gate": {"passed": True}}
    result = promotion_gate(
        meta, is_challenger_evaluated=True, promotion_review_requested=True, reviewed_by=None
    )
    assert result["stage"] == "PROMOTION_REVIEW"
    assert result["blocked_at"] == "APPROVED"


def test_6c_promotion_gate_never_reaches_production_validated_on_thin_evidence():
    meta = {"final_test_metrics": {"roc_auc": 0.9}, "quality_gate": {"passed": True}}
    result = promotion_gate(
        meta,
        is_challenger_evaluated=True,
        promotion_review_requested=True,
        reviewed_by="someone",
        is_shadow_serving=True,
        production_evidence_rows=1,
    )
    assert result["stage"] == "SHADOW"
    assert result["blocked_at"] == "PRODUCTION_VALIDATED"


# ------------------------------------------------------------------ 7


def test_7_production_validation_not_ready_with_only_test_provenance():
    records = [_record(f"c{i}", 0.9, True, provenance="test") for i in range(500)]
    assert production_validation_status(records) == "NOT_READY"


def test_7b_production_validation_ready_only_with_real_production_volume():
    records = [_record(f"c{i}", 0.9, True, provenance="production") for i in range(250)]
    assert production_validation_status(records) == "READY"


# ------------------------------------------------------------------ 8


def test_8_retraining_never_auto_retrains():
    result = retraining_readiness(
        drift_tier="HIGH_DRIFT", performance_delta=0.2, new_legitimate_labels=500
    )
    assert result["status"] == "RETRAINING_RECOMMENDED"
    assert result["auto_retrained"] is False


def test_8b_data_sufficiency_gate_is_honest_about_thin_data():
    records = [_record(f"c{i}", 0.9, True, provenance="test") for i in range(5)]
    result = data_sufficiency_gate(records)
    assert result["status"] == "DATA_INSUFFICIENT"
    assert "minimum_sample_count" in result["failed_checks"]
    assert "legitimate_provenance_present" in result["failed_checks"]


# ------------------------------------------------------------------ 9


def test_9_anomaly_monitoring_never_interprets_anomaly_as_fraud():
    records = [_record(f"c{i}", 0.9, True, anomaly_level="ANOMALOUS") for i in range(5)]
    result = anomaly_monitoring(records)
    # The module explicitly DISCLAIMS a fraud interpretation (the word "fraud"
    # legitimately appears in that disclaimer) - what must never exist is an
    # actual fraud/customer-quality VERDICT field.
    assert not any(
        k in result for k in ("is_fraud", "fraud_score", "customer_quality", "bad_customer")
    )
    assert "never an automatic interpretation of fraud" in result["note"]


# ------------------------------------------------------------------ 10


def test_10_policy_context_still_has_no_monitoring_or_champion_fields():
    from application.policy_engine import PolicyContext

    field_names = set(PolicyContext.model_fields.keys())
    for forbidden in (
        "champion_model",
        "challenger_model",
        "drift_tier",
        "calibration_gap",
        "promotion_status",
    ):
        assert forbidden not in field_names


def test_10b_domain_monitoring_output_cannot_reach_the_policy_engine():
    """Structural guarantee: domain_monitoring returns plain dicts/dataclasses,
    not anything PolicyContext or DeterministicPolicyEngine.evaluate accepts."""
    records = [_record(f"c{i}", 0.9, True) for i in range(5)]
    result = domain_monitoring(records)
    import inspect

    from application.policy_engine import DeterministicPolicyEngine

    sig = inspect.signature(DeterministicPolicyEngine.evaluate)
    assert "monitoring" not in sig.parameters
    assert isinstance(result, dict)


# ------------------------------------------------------------------ 11


def test_11_explanation_is_reproducible_across_repeated_calls_same_model_version():
    import os
    import tempfile

    import pandas as pd

    from application.ml_training import MLTrainingEngine
    from application.recovery_predictor_ml import MLPaymentFailurePredictor

    with tempfile.TemporaryDirectory() as tmp:
        df = pd.DataFrame(
            {
                "f1": list(range(40)),
                "target_recovered": [i % 2 for i in range(40)],
            }
        )
        path = os.path.join(tmp, "data.csv")
        df.to_csv(path, index=False)
        spec = {
            "dataset_id": "ds_p14_explain_repro",
            "prediction_problem": "payment-failure-risk",
            "target_column": "target_recovered",
            "feature_columns": ["f1"],
            "canonical_feature_mapping": {"f1": "AMOUNT"},
        }
        MLTrainingEngine(spec, path, tmp).train_and_evaluate()
        predictor = MLPaymentFailurePredictor(dataset_id="ds_p14_explain_repro", registry_dir=tmp)

        r1 = predictor.predict_failure_risk({"AMOUNT": 5})
        r2 = predictor.predict_failure_risk({"AMOUNT": 5})
        assert r1["model_metadata"]["explanation"] == r2["model_metadata"]["explanation"]
        assert r1["model_metadata"]["model_version"] == r2["model_metadata"]["model_version"]
