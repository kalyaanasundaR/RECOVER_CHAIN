"""Phase 11 — ML production hardening regression tests.

1  registry metadata carries the new lineage fields (checksum, temporal
   boundaries, lifecycle status, provenance status, drift summary)
2  lifecycle_status distinguishes EVALUATED (failed quality gate) from
   QUALITY_PASSED, independent of the legacy SELECTED/REJECTED_LOW_QUALITY
   `status` field (kept for backward compatibility, not replaced)
3  dataset_snapshot_checksum is a real sha256 of the exact bytes trained on,
   deterministic for the same file
4  drift_readiness.summarize_distribution / compare_distributions behave
   correctly on both numeric and categorical columns, including a genuine
   flagged shift
5  ERV calculation uses full-precision probability, not the rounded display
   value (regression for the ERV financial-safety requirement)
6  a synthetic/UNKNOWN-provenance dataset is never silently promoted to
   PRODUCTION_VALIDATED by the training pipeline
"""

import hashlib
import json
import os

import pandas as pd
import pytest

from application.drift_readiness import compare_distributions, summarize_distribution
from application.ml_training import MLTrainingEngine

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def mock_spec():
    return {
        "dataset_id": "ds_p11",
        "prediction_problem": "payment-failure-risk",
        "target_column": "target_recovered",
        "feature_columns": ["f1", "f2"],
        "canonical_feature_mapping": {"f1": "AMOUNT", "f2": "CUSTOMER_ID"},
        "temporal_split": {"strategy": "TEMPORAL_CHRONOLOGICAL", "split_column": "date"},
    }


@pytest.fixture
def mock_data(tmp_path):
    df = pd.DataFrame(
        {
            "date": [f"2023-{m:02d}-01" for m in range(1, 11)],
            "f1": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target_recovered": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ 1, 2


def test_1_registry_metadata_carries_lineage_fields(mock_spec, mock_data, tmp_path):
    engine = MLTrainingEngine(mock_spec, str(mock_data), str(tmp_path))
    metadata = engine.train_and_evaluate()

    for field in (
        "dataset_snapshot_checksum",
        "temporal_boundaries",
        "lifecycle_status",
        "provenance_status",
        "train_test_distribution_drift",
    ):
        assert field in metadata, f"missing {field}"

    assert metadata["lifecycle_status"] in ("QUALITY_PASSED", "EVALUATED")
    # New field is additive, doesn't replace the pre-existing status field
    # test_7 (test_phase45) still checks a required-subset of keys.
    assert metadata["status"] in ("SELECTED", "REJECTED_LOW_QUALITY")
    assert metadata["provenance_status"] == "UNKNOWN"  # no provenance supplied in spec


def test_2_lifecycle_status_reflects_quality_gate_independent_of_legacy_status(
    mock_spec, mock_data, tmp_path
):
    # Force a rejection by setting an impossibly high ROC-AUC bar.
    engine = MLTrainingEngine(mock_spec, str(mock_data), str(tmp_path), min_roc_auc=1.1)
    metadata = engine.train_and_evaluate()
    assert metadata["status"] == "REJECTED_LOW_QUALITY"
    assert metadata["lifecycle_status"] == "EVALUATED"


# ------------------------------------------------------------------ 3


def test_3_dataset_snapshot_checksum_is_deterministic_sha256(mock_spec, mock_data, tmp_path):
    expected = hashlib.sha256(mock_data.read_bytes()).hexdigest()
    engine = MLTrainingEngine(mock_spec, str(mock_data), str(tmp_path))
    metadata = engine.train_and_evaluate()
    assert metadata["dataset_snapshot_checksum"] == expected


# ------------------------------------------------------------------ 4


def test_4_drift_readiness_flags_a_genuine_numeric_shift():
    baseline = pd.DataFrame({"amount": [10.0] * 50})
    shifted = pd.DataFrame({"amount": [10.0] * 25 + [1000.0] * 25})

    b = summarize_distribution(baseline, ["amount"])
    c = summarize_distribution(shifted, ["amount"])
    result = compare_distributions(b, c)

    assert result["any_drifted"] is True
    assert result["columns"]["amount"]["drifted"] is True


def test_4b_drift_readiness_does_not_flag_identical_distributions():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0, 5.0] * 10, "cat": ["a", "b"] * 25})
    b = summarize_distribution(df, ["amount", "cat"])
    c = summarize_distribution(df, ["amount", "cat"])
    result = compare_distributions(b, c)
    assert result["any_drifted"] is False


# ------------------------------------------------------------------ 5


def test_5_erv_uses_full_precision_probability_not_rounded_display_value():
    from decimal import Decimal

    from application.action_evaluator import DeterministicActionEvaluator
    from domain.models import (
        ActionRecommendation,
        CaseState,
        DiagnosisStatus,
        Money,
        RecoveryCase,
        RecoveryPrediction,
        RiskCategory,
        RootCauseCategory,
        RootCauseDiagnosis,
    )

    unrounded_prob = 0.593971175447785123  # more precision than the 3dp display value
    case = RecoveryCase(
        case_id="case_erv_precision",
        customer_id="cust_erv_precision",
        risk_category=RiskCategory.OVERDUE_INVOICE,
        amount_at_risk=Money(amount=Decimal("250.00"), currency="INR"),
        current_state=CaseState.RECOMMENDING,
        diagnosis=RootCauseDiagnosis(
            diagnosis_id="diag_erv",
            cause_category=RootCauseCategory.UNKNOWN,
            confidence=0.5,
            status=DiagnosisStatus.INFERRED,
            supporting_signals={},
            evidence_references=[],
            diagnostic_method="test",
        ),
        prediction=RecoveryPrediction(
            prediction_id="pred_erv",
            recovery_probability=unrounded_prob,
            confidence=0.8,
            model_version="test",
            feature_version="1.0",
            prediction_timestamp=__import__("datetime").datetime.now(__import__("datetime").UTC),
            contributing_features={},
            prediction_status="SHADOW_ONLY",
        ),
    )

    rec: ActionRecommendation = DeterministicActionEvaluator().evaluate(case)
    top = rec.top_candidate
    assert top is not None
    # The stored ERV must equal amount * full-precision probability, rounded
    # only at the END - not amount * the already-rounded display probability.
    full_precision_erv = round(250.00 * unrounded_prob, 2)
    display_probability_erv = round(250.00 * round(unrounded_prob, 3), 2)
    assert top.expected_recoverable_value == full_precision_erv
    # These differ in this fixture on purpose, proving the two calculations
    # are NOT interchangeable and the code path takes the correct one.
    if full_precision_erv != display_probability_erv:
        assert top.expected_recoverable_value != display_probability_erv


# ------------------------------------------------------------------ 6


def test_6_unknown_provenance_dataset_never_auto_promoted_to_production_validated(
    mock_spec, mock_data, tmp_path
):
    mock_spec["dataset_provenance_status"] = "SYNTHETIC"
    engine = MLTrainingEngine(mock_spec, str(mock_data), str(tmp_path))
    metadata = engine.train_and_evaluate()
    assert metadata["provenance_status"] == "SYNTHETIC"
    assert metadata["lifecycle_status"] != "PRODUCTION_VALIDATED"
    # The training pipeline has no code path that ever writes this value -
    # grep-level guarantee, re-asserted here as a behavioural one.
    assert "PRODUCTION_VALIDATED" not in json.dumps(metadata)
