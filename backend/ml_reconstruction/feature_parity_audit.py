"""Training-vs-inference definition audit for amount_vs_customer_average (and a permutation
sensitivity check of the PRODUCTION canonical model on the SYNTHETIC validation split).

Not a live evaluation: no real outcomes exist. This only answers "does the model depend on
the feature whose live definition is an approximation?".

Usage: venv/Scripts/python.exe ml_reconstruction/feature_parity_audit.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, ".")
from application.recovery_probability_predictor import RecoveryProbabilityPredictor  # noqa: E402
from ml.inference.model_integrity import CANONICAL_FEATURES  # noqa: E402

DATA = Path("../data/recoverchain/synthetic/recovery_outcome/recovery_outcome.csv")


def main():
    df = pd.read_csv(DATA)
    df["suppression_status"] = df["suppression_status"].astype(int)
    df = df[~df["case_id"].str.endswith("_dup")].sort_values(
        ["customer_id", "obligation_created_at"]
    )

    # Live definition: amount / mean(customer's EARLIER obligation amounts)
    df["rolling_prior_mean"] = df.groupby("customer_id")["amount_at_risk"].transform(
        lambda s: s.shift().expanding().mean()
    )
    live_def = df["amount_at_risk"] / df["rolling_prior_mean"].clip(lower=1.0)
    has_prior = live_def.notna()
    train_def = df["amount_vs_customer_average"]
    out = {
        "rows": int(len(df)),
        "rows_with_a_prior_obligation": int(has_prior.sum()),
        "pearson_training_vs_live_definition": float(
            np.corrcoef(train_def[has_prior], live_def[has_prior])[0, 1]
        ),
        "spearman_training_vs_live_definition": float(
            pd.Series(train_def[has_prior].to_numpy()).corr(
                pd.Series(live_def[has_prior].to_numpy()), method="spearman"
            )
        ),
        "training_null_share": float(train_def.isna().mean()),
        "live_cold_start_null_share": float((~has_prior).mean()),
    }

    pred = RecoveryProbabilityPredictor()
    assert pred.model is not None, pred.load_failure
    val = df[df["split"] == "validation"]
    X = val[CANONICAL_FEATURES]
    y = val["recovery_success"].to_numpy()
    base_p = pred.model.predict_proba(X)[:, 1]
    out["synthetic_validation_pr_auc_baseline"] = float(average_precision_score(y, base_p))
    rng = np.random.default_rng(0)
    for name, mode in (("permuted", "perm"), ("set_missing", "nan")):
        deltas = []
        for _ in range(5 if mode == "perm" else 1):
            Xp = X.copy()
            Xp["amount_vs_customer_average"] = (
                rng.permutation(Xp["amount_vs_customer_average"].to_numpy())
                if mode == "perm"
                else np.nan
            )
            deltas.append(
                average_precision_score(y, pred.model.predict_proba(Xp)[:, 1])
                - out["synthetic_validation_pr_auc_baseline"]
            )
        out[f"pr_auc_change_when_amount_vs_customer_average_{name}"] = float(np.mean(deltas))
    out["roc_auc_baseline"] = float(roc_auc_score(y, base_p))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
