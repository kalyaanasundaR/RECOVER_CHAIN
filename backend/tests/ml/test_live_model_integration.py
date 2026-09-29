"""Proves, against real code (not assumed from reading it), the findings in
docs/ai_reconstruction/LIVE_MODEL_FORENSIC_AUDIT.md:

  1. Once a dataset-isolated model is trained, predict_recovery_for_case()
     genuinely loads and uses it (not the deterministic baseline).
  2. The result is the CALIBRATED, INVERTED recovery probability
     (1 - P(failure)), never the raw failure probability.
  3. Without a dataset_id, the same case falls back to the deterministic
     baseline -- the exact gap this session found live via the running API
     (POST /cases/{id}/predict-recovery with no ?dataset_id=... query param).
  4. A model trained for a non-revenue-recovery task (churn/fraud) is never
     loaded as a recovery predictor, so ERV can never be built from the
     wrong kind of probability.
"""

import pandas as pd
import pytest

from application.case_pipeline import predict_recovery_for_case
from application.ml_training import MLTrainingEngine
from application.recovery_predictor_ml import MLPaymentFailurePredictor
from domain.models import CaseState, Money, RecoveryCase, RevenueEvent, RiskCategory


@pytest.fixture
def trained_registry(tmp_path):
    """A real trained-and-calibrated model, isolated to dataset_id='ds_live_test'."""
    spec = {
        "dataset_id": "ds_live_test",
        "prediction_problem": "payment-failure-risk",
        "target_column": "status",
        "feature_columns": ["amount", "reason"],
        "canonical_feature_mapping": {"amount": "AMOUNT", "reason": "UNKNOWN"},
        "excluded_columns": ["id", "split"],
    }
    n = 120
    df = pd.DataFrame(
        {
            "id": range(n),
            "amount": [100 + (i * 37) % 900 for i in range(n)],
            "reason": (
                ["declined", "network_error", "expired_card", "insufficient_funds"] * (n // 4)
            ),
            # Alternating, not blocked, so every split fold gets both classes --
            # a too-small/single-class validation fold skips calibration by
            # design (ml_training.py's own documented behaviour), which would
            # make this fixture nondeterministically fail to exercise it.
            "status": (["failed", "failed", "failed", "paid"] * (n // 4)),
        }
    )
    data_path = tmp_path / "ds_live_test.csv"
    df.to_csv(data_path, index=False)
    engine = MLTrainingEngine(spec, str(data_path), str(tmp_path), min_roc_auc=0.0, min_test_rows=0)
    metadata = engine.train_and_evaluate()
    assert metadata["status"] == "SELECTED"
    assert metadata["calibrated"] is True  # the bug fixed this session must stay fixed
    return tmp_path


def _case(amount=500.0, reason="declined", source_ref=None):
    ev = RevenueEvent(
        event_id="evt_live_test",
        customer_id="cust_live_test",
        risk_category=RiskCategory.FAILED_PAYMENT,
        external_system="test",
        external_event_id="ext_1",
        reference_id=None,
        amount=Money(amount=amount),
        raw_payload={"amount": amount, "reason": reason},
        source_ref=source_ref,
    )
    return RecoveryCase(
        case_id="case_live_test",
        customer_id="cust_live_test",
        risk_category=RiskCategory.FAILED_PAYMENT,
        reference_id=None,
        amount_at_risk=Money(amount=amount),
        current_state=CaseState.DIAGNOSING,
        linked_events=[ev],
    )


def test_trained_model_is_genuinely_used_when_dataset_id_is_supplied(trained_registry):
    # Mirrors how api/dataset_router.py's shared_predictor is built for a real
    # dataset_id -- predict_recovery_for_case's own `predictor` parameter exists
    # exactly for this (reuse instead of re-scanning the registry per row).
    predictor = MLPaymentFailurePredictor(
        dataset_id="ds_live_test", registry_dir=str(trained_registry)
    )
    assert predictor.model is not None  # sanity: the fixture's model actually loaded
    prediction = predict_recovery_for_case(_case(), dataset_id="ds_live_test", predictor=predictor)
    assert prediction.model_version not in (None, "baseline-deterministic-v1.0")
    assert prediction.model_version in ("logistic_regression", "xgboost")


def test_recovery_probability_is_the_inverted_failure_probability(trained_registry):
    """§12: ERV must never be built from a raw failure probability labeled as recovery."""
    predictor = MLPaymentFailurePredictor(
        dataset_id="ds_live_test", registry_dir=str(trained_registry)
    )
    # Same feature-dict shape predict_recovery_for_case itself builds (raw
    # original column names alongside the canonical keys) -- "reason" is the
    # ORIGINAL column name the model was trained on; "UNKNOWN" in the
    # canonical_feature_mapping fixture above just means "no canonical field
    # applies to it", not that the raw name changes.
    raw = predictor.predict_failure_risk({"amount": 500.0, "reason": "declined", "AMOUNT": 500.0})
    prediction = predict_recovery_for_case(_case(), dataset_id="ds_live_test", predictor=predictor)
    # case_pipeline.py:87-88's own documented contract: recovery = 1 - failure.
    assert prediction.recovery_probability == pytest.approx(1.0 - raw["probability"], abs=1e-6)


def test_without_dataset_id_or_source_ref_falls_back_to_deterministic_baseline():
    """A case with no dataset lineage at all (e.g. created via the plain /events
    API, not a file upload) has nothing to auto-resolve and correctly stays on
    the deterministic baseline -- this is not a bug, just no dataset to look up."""
    prediction = predict_recovery_for_case(_case(source_ref=None), dataset_id=None)
    assert prediction.model_version == "baseline-deterministic-v1.0"


def test_dataset_id_is_auto_resolved_from_the_case_own_event(trained_registry, monkeypatch):
    """THE FIX: a caller (like the live POST /cases/{id}/predict-recovery
    endpoint, or the automatic /cases/{id}/advance pipeline) that supplies
    neither dataset_id nor predictor no longer silently gets the baseline --
    it's resolved from the case's own RevenueEvent.source_ref, exactly the
    field infrastructure/orm.py documents as "dataset_id for file replay"."""
    monkeypatch.setattr("infrastructure.storage.model_registry_dir", lambda: str(trained_registry))
    case = _case(source_ref="ds_live_test")
    prediction = predict_recovery_for_case(case)  # no dataset_id, no predictor -- must self-resolve
    assert prediction.model_version not in (None, "baseline-deterministic-v1.0")
    assert prediction.model_version in ("logistic_regression", "xgboost")


def test_a_non_revenue_recovery_task_model_is_never_loaded_as_a_recovery_predictor(tmp_path):
    spec = {
        "dataset_id": "ds_churn_test",
        "prediction_problem": "churn-prediction",  # deliberately NOT a recovery task
        "target_column": "status",
        "feature_columns": ["amount"],
        "canonical_feature_mapping": {"amount": "AMOUNT"},
        "excluded_columns": ["id"],
    }
    df = pd.DataFrame(
        {"id": range(30), "amount": range(30), "status": (["failed"] * 20 + ["paid"] * 10)}
    )
    data_path = tmp_path / "ds_churn_test.csv"
    df.to_csv(data_path, index=False)
    MLTrainingEngine(
        spec, str(data_path), str(tmp_path), min_roc_auc=0.0, min_test_rows=0
    ).train_and_evaluate()

    predictor = MLPaymentFailurePredictor(dataset_id="ds_churn_test", registry_dir=str(tmp_path))
    assert predictor.model is None  # churn model exists on disk but must never be loaded here
