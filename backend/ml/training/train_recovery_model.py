"""Canonical RecoverChain recovery-probability training entry point.

Trains on the GENUINE recovery target (recovery_success, a real recovered/
not-recovered-in-a-30-day-window label — see
docs/ai_reconstruction/RECOVERY_DATASET_VALIDATION.md), not the payment
status column any per-dataset upload happens to carry. This is a separate
model lineage from application/recovery_predictor_ml.py's
MLPaymentFailurePredictor (payment-failure-risk, per-upload) — see
docs/ai_reconstruction/MODEL_LINEAGE_RECONCILIATION.md for exactly why they
must not be merged.

Saves its artifact to its OWN registry directory
(backend/ml/models/recovery_registry/), structurally separate from the
payment-failure registry, with task="recovery-probability" so
application/recovery_predictor_ml.MLPaymentFailurePredictor can never load
it (disjoint task vocabulary) and vice versa.

Usage:
    venv/Scripts/python.exe ml/training/train_recovery_model.py
"""

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from ml.inference.model_integrity import (  # noqa: E402
    CANONICAL_CATEGORICAL,
    CANONICAL_NUMERIC,
    CANONICAL_TARGET_DEFINITION,
    register_artifact,
)

DATA_PATH = (
    BACKEND_ROOT.parent
    / "data"
    / "recoverchain"
    / "synthetic"
    / "recovery_outcome"
    / "recovery_outcome.xlsx"
)
REGISTRY_DIR = BACKEND_ROOT / "ml" / "models" / "recovery_registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_RECOVERY_TASK = "recovery-probability"
TARGET = "recovery_success"
CATEGORICAL = [
    "domain",
    "payment_method",
    "account_status",
    "consent_status",
    "historical_recovery_behavior",
]
NUMERIC = [
    "amount_at_risk",
    "age_of_obligation",
    "customer_tenure",
    "prior_payment_success_rate",
    "prior_recovery_rate",
    "prior_late_rate",
    "number_of_previous_failed_attempts",
    "number_of_previous_recoveries",
    "average_days_to_payment",
    "amount_vs_customer_average",
    "invoice_age",
    "payment_method_valid",
    "customer_engagement",
    "recent_login",
    "reminder_history",
    "previous_contact_success",
    "dispute_indicator",
    "suppression_status",
]
FEATURES = NUMERIC + CATEGORICAL
assert NUMERIC == CANONICAL_NUMERIC and CATEGORICAL == CANONICAL_CATEGORICAL, (
    "training feature lists drifted from ml/inference/model_integrity.py"
)


def _metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_rate": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out["brier_score"] = float(brier_score_loss(y_true, y_prob))
        out["log_loss"] = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)))
    return out


def main():
    df = pd.read_excel(DATA_PATH)
    df["suppression_status"] = df["suppression_status"].astype(int)
    checksum = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "validation"]
    te = df[df["split"] == "test"]

    X_tr, y_tr = tr[FEATURES], tr[TARGET].to_numpy()
    X_va, y_va = va[FEATURES], va[TARGET].to_numpy()
    X_te, y_te = te[FEATURES], te[TARGET].to_numpy()

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    candidates = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=(y_tr == 0).sum() / max(1, (y_tr == 1).sum()),
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        ),
    }

    results = {}
    fitted = {}
    for name, clf in candidates.items():
        t0 = time.time()
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        try:
            model = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
            model.fit(X_va, y_va)
            calibrated_ok = True
        except Exception:
            model = pipe
            calibrated_ok = False
        fitted[name] = model
        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "calibrated": calibrated_ok,
            "training_time_sec": round(time.time() - t0, 3),
            "validation_metrics": _metrics(y_va, p_va),
            "test_metrics": _metrics(y_te, p_te),
        }

    best_name = max(results, key=lambda n: results[n]["validation_metrics"]["pr_auc"])
    best_model = fitted[best_name]

    run_id = f"recovery_model_{int(time.time())}"
    artifact_path = REGISTRY_DIR / f"{run_id}_model.joblib"
    joblib.dump(best_model, artifact_path)

    metadata = {
        "model_id": run_id,
        "task": CANONICAL_RECOVERY_TASK,  # disjoint from REVENUE_RECOVERY_TASKS by design
        "target_column": TARGET,
        "target_definition": CANONICAL_TARGET_DEFINITION,
        "feature_columns": FEATURES,
        "selected_model": best_name,
        "status": "SELECTED",
        "calibrated": results[best_name]["calibrated"],
        "candidate_results": results,
        "final_test_metrics": results[best_name]["test_metrics"],
        "dataset_path": str(DATA_PATH),
        "dataset_checksum": checksum,
        "split_row_counts": {"train": len(tr), "validation": len(va), "test": len(te)},
        "artifact_path": str(artifact_path),
        "created_at": datetime.now(UTC).isoformat(),
    }
    metadata_path = REGISTRY_DIR / f"{run_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    # Register checksum + schema + preprocessing metadata from the ACTUAL artifact bytes.
    register_artifact(str(REGISTRY_DIR), str(metadata_path))

    print(
        f"selected_model={best_name}  test_roc_auc={results[best_name]['test_metrics'].get('roc_auc')}"
    )
    print(f"artifact: {artifact_path}")
    print(f"metadata: {metadata_path}")
    return metadata


if __name__ == "__main__":
    main()
