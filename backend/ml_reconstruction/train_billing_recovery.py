"""RecoverChain AI reconstruction — §7-20 baseline ladder for billing_recovery_v3.

Standalone, independent of the app's live MLTrainingEngine. Runs against the
dataset's own pre-computed split and the exact feature list already used by
backend/run_phase19_training.py (a prior phase's one-off run against this
same file — this script repeats that experiment properly, with baselines,
RF, calibration and an honest report, rather than just an XGBoost .json with
no surviving metrics).

3.6M rows / 94.2% positive class (target_recovered=1) — per this dataset's
own metadata.json, the target is a *per-cycle* failure proxy and the file
notes 89% of failing subscriber-months recover in a later month. Expect
this to be a genuinely hard, weak-signal problem; report whatever comes out,
do not tune until it looks good.

Usage:
    venv/Scripts/python.exe ml_reconstruction/train_billing_recovery.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "evaluation" / "datasets" / "billing_recovery_v3.csv"
OUT_DIR = HERE / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_recovered"
# Identical to backend/run_phase19_training.py's spec, minus bill_seq (that
# prior run excluded it as an identifier-adjacent counter, not a feature).
FEATURES = [
    "bill_amount_excl_late",
    "base_charge_usd",
    "data_overage_usd",
    "intl_roaming_usd",
    "tax_usd",
    "has_overage",
    "has_roaming",
    "prior_failures",
    "prior_unpaid",
    "prior_late",
    "prior_ontime",
    "prior_nonrecovery_count",
    "prior_nonrecovery_rate",
    "prior_late_rate",
    "prior_avg_amount",
]
# All-numeric — no categorical columns in this feature set (has_overage /
# has_roaming are already 0/1 ints, not category labels).


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
    t_load = time.time()
    cols = [*FEATURES, TARGET, "split"]
    df = pd.read_csv(DATA_PATH, usecols=cols)
    print(f"loaded {len(df)} rows in {time.time() - t_load:.1f}s")

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "validation"]
    te = df[df["split"] == "test"]
    print(
        f"train={len(tr)} validation={len(va)} test={len(te)}  (dataset's own pre-computed split)"
    )

    X_tr, y_tr = tr[FEATURES], tr[TARGET].to_numpy()
    X_va, y_va = va[FEATURES], va[TARGET].to_numpy()
    X_te, y_te = te[FEATURES], te[TARGET].to_numpy()
    print(
        f"train positive rate={y_tr.mean():.4f}  (94%+ recovered -> failure/non-recovery is the rare class)"
    )

    results = {}

    # --- Baseline 0: majority class -----------------------------------------
    majority = int(round(y_tr.mean()))
    results["baseline_0_majority"] = {
        "description": f"always predicts {majority} (the majority class in train)",
        "test_metrics": _metrics(y_te, np.full(len(y_te), float(majority))),
    }

    # --- Baseline 1: 1 - prior_nonrecovery_rate as the probability ----------
    # This dataset's analogue of ar_recovery_v1's "historical rate" baseline:
    # target=1 means recovered, so 1 - prior_nonrecovery_rate is the no-fit
    # probability estimate. -1 sentinel (no history) -> train base rate.
    hist = (1 - te["prior_nonrecovery_rate"].replace(-1, 1 - y_tr.mean())).clip(0, 1).to_numpy()
    results["baseline_1_historical_rate"] = {
        "description": "uses 1 - prior_nonrecovery_rate as the probability, no fitting",
        "test_metrics": _metrics(y_te, hist),
    }

    pre = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])

    candidates = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=200, random_state=42, n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=(y_tr == 0).sum() / max(1, (y_tr == 1).sum()),
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            n_jobs=-1,
        ),
    }

    for name, clf in candidates.items():
        t0 = time.time()
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        print(f"  [{name}] fit in {fit_time:.1f}s")

        try:
            calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
            calibrated.fit(X_va, y_va)
            model = calibrated
            calibrated_ok = True
        except Exception as e:
            model = pipe
            calibrated_ok = False
            print(f"  [{name}] calibration skipped: {e}")

        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "calibrated": calibrated_ok,
            "training_time_sec": round(fit_time, 2),
            "validation_metrics": _metrics(y_va, p_va),
            "test_metrics": _metrics(y_te, p_te),
        }

    ranked = sorted(
        (
            (name, r["validation_metrics"]["pr_auc"])
            for name, r in results.items()
            if "validation_metrics" in r
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    results["_selection"] = {
        "criterion": "validation PR-AUC",
        "ranked": ranked,
        "selected_model": ranked[0][0] if ranked else None,
        "note": (
            "Target is a per-cycle non-recovery proxy; this dataset's own "
            "metadata.json documents that 89% of failing subscriber-months "
            "recover in a later month, so a high test score here does not "
            "mean absolute non-recovery is predictable — only this cycle's "
            "proxy is."
        ),
    }

    print(json.dumps({k: v for k, v in results.items() if k != "_selection"}, indent=2))
    print(json.dumps(results["_selection"], indent=2))

    run_id = f"billing_recovery_v3_{int(time.time())}"
    out_path = OUT_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": "billing_recovery_v3",
                "dataset_metadata_ref": "evaluation/datasets/billing_recovery_v3_metadata.json",
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
