import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from application.recovery_predictor_ml import MLPaymentFailurePredictor
from domain.models import (
    CaseState,
    Money,
    RecoveryCase,
    RecoveryPrediction,
    RevenueEvent,
    RiskCategory,
)


def apply_shadow_prediction(case, db):
    predictor = MLPaymentFailurePredictor()

    # Extract available features from the case
    case_features = {
        "amount": case.amount_at_risk.amount if case.amount_at_risk else 0.0,
        "customer_id": case.customer_id,
        # fallback mapping for testing
        "f1": 1,
        "f2": 0.5,
    }

    try:
        res = predictor.predict_failure_risk(case_features)
        prob = res.get("probability", 0.0)
        status = res.get("status", "FAILED")
        meta = res.get("model_metadata", {})
    except ValueError as e:
        prob = 0.0
        status = f"FAILED: {str(e)}"
        meta = {}

    prediction = RecoveryPrediction(
        prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
        recovery_probability=prob,
        confidence=0.7,
        model_version=meta.get("model_version", "unknown"),
        feature_version="1.0",
        prediction_timestamp=datetime.now(UTC),
        contributing_features=case_features,
        prediction_status=status,
    )

    return prediction


# ---------------------------------------------------------------------------
# Phase 11 — live shadow integration audit (spec §10): six failure modes, each
# must degrade safely to deterministic behaviour. 1 (valid model) and 6
# (prediction exception) already have direct coverage in
# test_phase45_phase10_ml_evaluation.py / test_ml_training.py; these cover the
# remaining modes plus the trained_at staleness-visibility field added above.
# ---------------------------------------------------------------------------


def _case(case_id="case_shadow", amount=Decimal("400")):
    event = RevenueEvent(
        event_id=f"evt_{case_id}",
        customer_id=f"cust_{case_id}",
        risk_category=RiskCategory.FAILED_PAYMENT,
        external_system="test",
        external_event_id=f"ext_{case_id}",
        reference_id=f"ref_{case_id}",
        amount=Money(amount=amount, currency="INR"),
        timestamp=datetime.now(UTC),
        raw_payload={},
    )
    return RecoveryCase(
        case_id=case_id,
        customer_id=f"cust_{case_id}",
        risk_category=RiskCategory.FAILED_PAYMENT,
        amount_at_risk=Money(amount=amount, currency="INR"),
        current_state=CaseState.DIAGNOSING,
        linked_events=[event],
    )


def test_shadow_2_no_model_available_falls_back_to_deterministic():
    """Mode 2: a dataset_id with no registry entry at all."""
    from application.case_pipeline import predict_recovery_for_case

    result = predict_recovery_for_case(_case("case_no_model"), dataset_id="ds_definitely_missing")
    assert result.prediction_status == "SUCCESS_BASELINE"
    assert 0.0 <= result.recovery_probability <= 1.0


def test_shadow_3a_malformed_model_artifact_raises_at_construction(tmp_path):
    """Loading a corrupt artifact (not a valid joblib pickle - a truncated or
    corrupted write) raises at MLPaymentFailurePredictor construction time
    (joblib.load on a non-pickle blob raises KeyError, not a clean custom
    error). Documents the exact failure shape the pipeline-level fix below
    must catch."""
    import json

    meta = {
        "model_id": "train_bad",
        "dataset_id": "ds_bad_artifact",
        "task": "payment-failure-risk",
        "feature_columns": ["f1"],
        "canonical_feature_mapping": {"f1": "AMOUNT"},
        "artifact_path": str(tmp_path / "train_bad_model.joblib"),
        "status": "SELECTED",
    }
    (tmp_path / "train_bad_metadata.json").write_text(json.dumps(meta))
    (tmp_path / "train_bad_model.joblib").write_bytes(b"not a real joblib pickle")

    with pytest.raises(KeyError):
        MLPaymentFailurePredictor(dataset_id="ds_bad_artifact", registry_dir=str(tmp_path))


def test_shadow_3b_malformed_model_artifact_falls_back_via_pipeline(tmp_path, monkeypatch):
    """Mode 3, end to end: a case pipeline run against a dataset whose registry
    entry points at a corrupt artifact must still return a safe deterministic
    prediction, not crash. Regression for a real defect found this phase:
    predict_recovery_for_case only wrapped the predict_failure_risk() call in
    try/except, not MLPaymentFailurePredictor's own construction (which is
    where the artifact is actually loaded from disk) - fixed in
    application/case_pipeline.py."""
    import json

    from infrastructure import storage

    meta = {
        "model_id": "train_bad",
        "dataset_id": "ds_bad_artifact_e2e",
        "task": "payment-failure-risk",
        "feature_columns": ["f1"],
        "canonical_feature_mapping": {"f1": "AMOUNT"},
        "artifact_path": str(tmp_path / "train_bad_model.joblib"),
        "status": "SELECTED",
    }
    (tmp_path / "train_bad_metadata.json").write_text(json.dumps(meta))
    (tmp_path / "train_bad_model.joblib").write_bytes(b"not a real joblib pickle")
    monkeypatch.setattr(storage, "model_registry_dir", lambda: str(tmp_path))

    from application.case_pipeline import predict_recovery_for_case

    result = predict_recovery_for_case(_case("case_bad_artifact"), dataset_id="ds_bad_artifact_e2e")
    assert result.prediction_status == "SUCCESS_BASELINE"
    assert 0.0 <= result.recovery_probability <= 1.0


def test_shadow_4_incompatible_feature_schema_falls_back_via_pipeline(tmp_path):
    """Mode 4: a real, loadable model whose required features the live case
    genuinely cannot supply (e.g. a schema drift between training and serving).
    predict_failure_risk raises ValueError; the pipeline must still produce a
    safe deterministic prediction, never propagate the error to the caller."""
    from application.case_pipeline import predict_recovery_for_case

    class _IncompatibleSchema:
        model = object()
        features = ["f1", "f2", "f3_never_present"]

        def predict_failure_risk(self, features):
            missing = [f for f in self.features if f not in features]
            if missing:
                raise ValueError(f"Incomplete feature set: {missing}")
            return {"probability": 0.1, "status": "SUCCESS", "model_metadata": {}}

    result = predict_recovery_for_case(_case("case_incompatible"), predictor=_IncompatibleSchema())
    assert result.prediction_status == "SUCCESS_BASELINE"
    assert 0.0 <= result.recovery_probability <= 1.0


def test_shadow_5_model_metadata_exposes_trained_at_for_staleness_visibility(tmp_path):
    """Mode 5 (stale model): this repo has no automatic staleness cutoff (a
    genuine, disclosed gap — see the Phase 11 report), but every real
    prediction now carries `trained_at` so a caller CAN judge staleness. This
    guards against silently losing that field in a future refactor."""
    from application.ml_training import MLTrainingEngine

    df_path = tmp_path / "data.csv"
    import pandas as pd

    pd.DataFrame(
        {
            "f1": list(range(40)),
            "target_recovered": [i % 2 for i in range(40)],
        }
    ).to_csv(df_path, index=False)

    spec = {
        "dataset_id": "ds_staleness",
        "prediction_problem": "payment-failure-risk",
        "target_column": "target_recovered",
        "feature_columns": ["f1"],
        "canonical_feature_mapping": {"f1": "AMOUNT"},
    }
    MLTrainingEngine(spec, str(df_path), str(tmp_path)).train_and_evaluate()

    predictor = MLPaymentFailurePredictor(dataset_id="ds_staleness", registry_dir=str(tmp_path))
    res = predictor.predict_failure_risk({"AMOUNT": 1})
    assert res["model_metadata"]["trained_at"] is not None
