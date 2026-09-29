"""RecoverChain AI reconstruction — §21-22 DL comparison point for ar_recovery_v1.

Per Rule 0.3/0.4 (do not force DL, do not force a heavier framework than the
problem justifies): this venv has no torch/tensorflow installed, and
ar_recovery_v1 is 2,466 rows — installing a GPU-oriented DL framework for
that is not defensible. sklearn.neural_network.MLPClassifier IS a real
neural network trained by backpropagation with the Adam optimizer; it is the
right-sized tool for this comparison, not a shortcut around "real DL".

Only run after the classical-ML ladder (train_ar_recovery.py) has already
produced a baseline to beat — per your own instruction, this is a documented
comparison point, not a default choice.

Usage:
    venv/Scripts/python.exe ml_reconstruction/train_ar_recovery_mlp.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "evaluation" / "datasets" / "ar_recovery_v1.csv"
OUT_DIR = HERE / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_late"
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

    # Architecture: input -> Dense(32, ReLU) -> Dense(16, ReLU) -> sigmoid output.
    # Small on purpose: 1,594 training rows and ~15 encoded features after
    # one-hot -- a bigger network would just memorize (params >> data, per
    # Rule/§65). alpha is L2 weight decay; early_stopping holds out 15% of
    # TRAIN internally (not touching our validation/test folds) to stop
    # before overfitting.
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=42,
    )

    t0 = time.time()
    pipe = Pipeline([("pre", pre), ("clf", mlp)])
    pipe.fit(X_tr, y_tr)
    train_time = time.time() - t0

    loss_curve = pipe.named_steps["clf"].loss_curve_
    n_iter = pipe.named_steps["clf"].n_iter_
    print(f"trained {n_iter} epochs in {train_time:.2f}s")
    print(f"final training loss: {loss_curve[-1]:.4f}  (first: {loss_curve[0]:.4f})")
    overfit_flag = None
    if pipe.named_steps["clf"].validation_scores_ is not None:
        val_scores = pipe.named_steps["clf"].validation_scores_
        overfit_flag = val_scores[-1] < max(val_scores) - 0.02  # meaningfully off its own peak
        print(
            f"internal early-stopping val accuracy: first={val_scores[0]:.3f} last={val_scores[-1]:.3f} peak={max(val_scores):.3f}"
        )

    try:
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
        calibrated.fit(X_va, y_va)
        model = calibrated
        calibrated_ok = True
    except Exception as e:
        model = pipe
        calibrated_ok = False
        print(f"calibration skipped: {e}")

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    result = {
        "model": "mlp_sklearn",
        "architecture": "Dense(32,relu) -> Dense(16,relu) -> sigmoid",
        "framework": "sklearn.neural_network.MLPClassifier (no torch/tensorflow installed; "
        "dataset is 2,466 rows so a heavier framework is not justified)",
        "optimizer": "adam",
        "learning_rate_init": 1e-3,
        "alpha_l2": 1e-3,
        "epochs_run": int(n_iter),
        "max_epochs": 500,
        "early_stopping": True,
        "training_time_sec": round(train_time, 3),
        "loss_curve_first": float(loss_curve[0]),
        "loss_curve_last": float(loss_curve[-1]),
        "internal_val_overfit_flag": overfit_flag,
        "calibrated": calibrated_ok,
        "validation_metrics": _metrics(y_va, p_va),
        "test_metrics": _metrics(y_te, p_te),
    }

    print(json.dumps(result, indent=2))

    # Compare against the classical ladder's best result on this same dataset.
    ladder_runs = sorted((HERE / "runs").glob("ar_recovery_v1_*.json"))
    comparison = None
    if ladder_runs:
        classical = json.load(open(ladder_runs[-1]))["results"]
        best_classical = max(
            (
                (n, r["test_metrics"]["roc_auc"])
                for n, r in classical.items()
                if n not in ("_selection",) and "test_metrics" in r
            ),
            key=lambda x: x[1],
        )
        comparison = {
            "best_classical_model": best_classical[0],
            "best_classical_test_roc_auc": best_classical[1],
            "mlp_test_roc_auc": result["test_metrics"]["roc_auc"],
            "mlp_wins": result["test_metrics"]["roc_auc"] > best_classical[1],
        }
        print(json.dumps(comparison, indent=2))

    run_id = f"ar_recovery_v1_mlp_{int(time.time())}"
    out_path = OUT_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump({"result": result, "comparison_vs_classical_ladder": comparison}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
