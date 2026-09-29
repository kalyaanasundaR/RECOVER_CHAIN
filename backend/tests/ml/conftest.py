"""Builds small but REAL canonical-schema recovery artifacts (calibrated pipeline, joblib,
registered checksum/schema) in a temp dir - so integrity tests never mutate the production
artifact."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.inference.model_integrity import (
    CANONICAL_CATEGORICAL,
    CANONICAL_FEATURES,
    CANONICAL_NUMERIC,
    CANONICAL_TARGET,
    CANONICAL_TARGET_DEFINITION,
    CANONICAL_TASK,
    register_artifact,
)

CATS = {
    "domain": ["FAILED_PAYMENT", "OVERDUE_INVOICE"],
    "payment_method": ["card", "upi"],
    "account_status": ["active", "closed"],
    "consent_status": ["ALLOWED", "OPTED_OUT", "UNKNOWN"],
    "historical_recovery_behavior": ["mixed", "rarely_recovers"],
}


def _frame(n=200, seed=0, column_order=None):
    rng = np.random.default_rng(seed)
    d = {c: rng.normal(size=n) for c in CANONICAL_NUMERIC}
    for c, vals in CATS.items():
        d[c] = rng.choice(vals, size=n)
    df = pd.DataFrame(d)
    y = (d["amount_at_risk"] + rng.normal(scale=1.0, size=n) > 0).astype(int)
    return df[column_order or CANONICAL_FEATURES], y


@pytest.fixture
def make_recovery_artifact(tmp_path):
    def _make(
        model_id="recovery_model_test1", *, column_order=None, register=True, created="2026-01-01"
    ):
        X, y = _frame(column_order=column_order)
        cols = list(X.columns)
        num = [c for c in cols if c in CANONICAL_NUMERIC]
        cat = [c for c in cols if c in CANONICAL_CATEGORICAL]
        pre = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                    ),
                    num,
                ),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ]
        )
        pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300))]).fit(X, y)
        model = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid").fit(X, y)
        artifact = tmp_path / f"{model_id}_model.joblib"
        joblib.dump(model, artifact)
        meta = {
            "model_id": model_id,
            "task": CANONICAL_TASK,
            "target_column": CANONICAL_TARGET,
            "target_definition": CANONICAL_TARGET_DEFINITION,
            "feature_columns": list(CANONICAL_FEATURES),
            "selected_model": "logistic_regression",
            "status": "SELECTED",
            "calibrated": True,
            "artifact_path": str(artifact),
            "created_at": created,
        }
        meta_path = tmp_path / f"{model_id}_metadata.json"
        meta_path.write_text(json.dumps(meta))
        if register:
            register_artifact(str(tmp_path), str(meta_path))
        return tmp_path, meta_path, artifact

    return _make
