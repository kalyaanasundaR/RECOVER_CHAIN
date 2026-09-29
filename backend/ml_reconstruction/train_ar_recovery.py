"""RecoverChain AI reconstruction — §7-20 baseline ladder for ar_recovery_v1.

Standalone, independent of the app's live MLTrainingEngine (never touches the
operational DB or the live shadow-model registry). Runs against the dataset's
OWN pre-computed temporal split (train/validation/test, documented in
evaluation/datasets/ar_recovery_v1_metadata.json) and its own documented
leakage exclusions (DaysToSettle, DaysLate, SettledDate — all only knowable
after settlement, per that file).

Usage:
    venv/Scripts/python.exe ml_reconstruction/train_ar_recovery.py
"""

import json
import time
from pathlib import Path

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
DATA_PATH = HERE.parent / "evaluation" / "datasets" / "ar_recovery_v1.csv"
OUT_DIR = HERE / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_late"
# Exactly the feature list documented in ar_recovery_v1_metadata.json. Does NOT
# include DaysToSettle / DaysLate / SettledDate (the file's own documented
# leakage exclusions — all only knowable after the invoice settles) and does
# NOT include identifiers (customerID, invoiceNumber, InvoiceDate, DueDate).
FEATURES = [
    "InvoiceAmount",
    "countryCode",
    "is_disputed",
    "payment_window_days",
    "invoice_seq",
    "prior_late_count",
    "prior_invoice_count",
    "prior_late_rate",
    "prior_avg_amount",
    "prior_disputes",
    "is_paperless",
]
CATEGORICAL = ["countryCode"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


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
    df = pd.read_csv(DATA_PATH)
    assert set(df["split"].unique()) >= {"train", "validation", "test"}

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "validation"]
    te = df[df["split"] == "test"]
    print(
        f"train={len(tr)} validation={len(va)} test={len(te)}  (dataset's own pre-computed split)"
    )

    X_tr, y_tr = tr[FEATURES], tr[TARGET].to_numpy()
    X_va, y_va = va[FEATURES], va[TARGET].to_numpy()
    X_te, y_te = te[FEATURES], te[TARGET].to_numpy()

    results = {}

    # --- Baseline 0: no-skill (majority class) -----------------------------
    majority = int(round(y_tr.mean()))
    results["baseline_0_majority"] = {
        "description": f"always predicts {majority} (the majority class in train)",
        "test_metrics": _metrics(y_te, np.full(len(y_te), float(majority))),
    }

    # --- Baseline 1: historical late-rate as the probability ---------------
    # Uses the row's own prior_late_rate feature as the predicted probability
    # directly (no model fit) — the honest "what would a merchant's gut-feel
    # heuristic get" baseline. -1 sentinel (first invoice, no history) is
    # treated as the train base rate.
    hist = te["prior_late_rate"].replace(-1, y_tr.mean()).clip(0, 1).to_numpy()
    results["baseline_1_historical_rate"] = {
        "description": "uses the invoice's own prior_late_rate as the probability, no fitting",
        "test_metrics": _metrics(y_te, hist),
    }

    # --- Shared preprocessing ------------------------------------------------
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
            max_depth=6,
            min_samples_leaf=5,
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

    fitted = {}
    for name, clf in candidates.items():
        t0 = time.time()
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)

        # Calibrate on validation (prefit — the already-fitted pipe is frozen so
        # calibration fits only the sigmoid, never re-fits the classifier).
        try:
            calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
            calibrated.fit(X_va, y_va)
            model = calibrated
            calibrated_ok = True
        except Exception as e:
            model = pipe
            calibrated_ok = False
            print(f"  [{name}] calibration skipped: {e}")

        fitted[name] = model
        train_time = time.time() - t0

        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "calibrated": calibrated_ok,
            "training_time_sec": round(train_time, 3),
            "validation_metrics": _metrics(y_va, p_va),
            "test_metrics": _metrics(y_te, p_te),
        }

    # --- Model selection: by validation PR-AUC, exactly the app's own rule --
    ranked = sorted(
        (
            (name, r["validation_metrics"]["pr_auc"])
            for name, r in results.items()
            if "validation_metrics" in r
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_val_pr_auc = ranked[0]
    results["_selection"] = {
        "criterion": "validation PR-AUC (imbalanced positive class — 35.6% late)",
        "ranked": ranked,
        "selected_model": best_name,
    }

    print(json.dumps({k: v for k, v in results.items() if k != "_selection"}, indent=2))
    print(json.dumps(results["_selection"], indent=2))

    run_id = f"ar_recovery_v1_{int(time.time())}"
    out_path = OUT_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": "ar_recovery_v1",
                "dataset_metadata_ref": "evaluation/datasets/ar_recovery_v1_metadata.json",
                "split_row_counts": {"train": len(tr), "validation": len(va), "test": len(te)},
                "features": FEATURES,
                "target": TARGET,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
