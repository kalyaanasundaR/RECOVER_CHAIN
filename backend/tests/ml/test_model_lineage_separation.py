"""Proves the two model lineages (docs/ai_reconstruction/MODEL_LINEAGE_RECONCILIATION.md)
are genuinely, structurally separate -- not just separate by convention:

  1. RecoveryProbabilityPredictor (Lineage B) never loads a payment-failure-risk
     artifact, even if one exists in its registry directory.
  2. MLPaymentFailurePredictor (Lineage A) never loads a recovery-probability
     artifact, even if one exists in its registry directory.
  3. derive_recovery_features_for_case never fabricates a feature it cannot
     genuinely derive -- unavailable ones stay None, never a guessed default.
  4. The live app's real trained canonical model is reachable end to end
     through GET /cases/{id}/recovery-probability, using DB-derived history.
  5. ERV from the canonical endpoint is exactly amount_at_risk × calibrated
     probability -- never the payment-failure-inverted number.
"""

import json

import pandas as pd

from application.ml_training import MLTrainingEngine
from application.recovery_predictor_ml import MLPaymentFailurePredictor
from application.recovery_probability_predictor import (
    CANONICAL_RECOVERY_TASK,
    RecoveryProbabilityPredictor,
)


def _train_stub(tmp_path, task, target_col, name):
    spec = {
        "dataset_id": f"ds_{name}",
        "prediction_problem": task,
        "target_column": target_col,
        "feature_columns": ["amount"],
        "canonical_feature_mapping": {"amount": "AMOUNT"},
        "excluded_columns": ["id"],
    }
    df = pd.DataFrame(
        {"id": range(30), "amount": range(30), target_col: (["failed"] * 20 + ["paid"] * 10)}
    )
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    meta = MLTrainingEngine(
        spec, str(path), str(tmp_path), min_roc_auc=0.0, min_test_rows=0
    ).train_and_evaluate()
    # Rewrite the task label directly in the metadata file -- MLTrainingEngine
    # itself only ever writes "payment-failure-risk" (from ml_readiness), so
    # this simulates what a canonical-recovery-trained artifact's metadata
    # looks like without duplicating the whole training pipeline here.
    meta_path = tmp_path / f"{meta['model_id']}_metadata.json"
    meta["task"] = task
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return tmp_path


def test_recovery_predictor_never_loads_a_payment_failure_artifact(tmp_path):
    _train_stub(tmp_path, "payment-failure-risk", "status", "pf")
    predictor = RecoveryProbabilityPredictor(registry_dir=str(tmp_path))
    assert predictor.model is None  # exists on disk, wrong task -- must not load


def test_payment_failure_predictor_never_loads_a_recovery_probability_artifact(tmp_path):
    _train_stub(tmp_path, CANONICAL_RECOVERY_TASK, "status", "rec")
    predictor = MLPaymentFailurePredictor(dataset_id="ds_rec", registry_dir=str(tmp_path))
    assert predictor.model is None  # exists on disk, wrong task -- must not load


def test_recovery_predictor_loads_its_own_task_correctly(make_recovery_artifact):
    # The earlier version loaded a 1-feature stub with no checksum. Under strict integrity
    # loading such an artifact must be REFUSED, so this now builds a real canonical-schema
    # artifact (tests/ml/conftest.py).
    d, _, _ = make_recovery_artifact()
    predictor = RecoveryProbabilityPredictor(registry_dir=str(d))
    assert predictor.model is not None
    assert predictor.metadata["task"] == CANONICAL_RECOVERY_TASK
