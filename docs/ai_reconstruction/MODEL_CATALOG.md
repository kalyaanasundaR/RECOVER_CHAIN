# Model Catalog

| Model | Task | Target | Dataset | Status |
|---|---|---|---|---|
| Logistic Regression / Random Forest / XGBoost (per-upload) | `payment-failure-risk` | uploaded file's own `status`-type column | whatever the merchant uploads | **SHADOW / PRODUCTION-ADJACENT** — genuinely wired to `GET/POST .../predict-recovery`, feeds the live recommendation's ERV today |
| Logistic Regression / Random Forest / XGBoost (`ml_reconstruction/train_ar_recovery.py`) | late-payment risk | `target_late` | `ar_recovery_v1` (real IBM Watson sample data) | **EXPERIMENTAL / RESEARCH** — not wired to any live path; a comparison-ladder artifact only |
| MLP (`ml_reconstruction/train_ar_recovery_mlp.py`) | late-payment risk | `target_late` | `ar_recovery_v1` | **EXPERIMENTAL** — DL comparison point only, edges out classical models within small-sample noise |
| XGBoost (`run_phase19_training.py`, `scripts/legacy/phase17_train.py`) | payment-failure-risk | `billing_recovery_v3`'s `status` | `billing_recovery_v3` (3.6M rows) | **RETIRED / WEAK-SIGNAL** — near-random (ROC-AUC ≈ 0.54-0.55), independently reproduced twice; deployment-blocked |
| **Logistic Regression / Random Forest / XGBoost (`ml/training/train_recovery_model.py`)** | **`recovery-probability`** | **`recovery_success`** | **`recovery_outcome.xlsx` (validated, see `RECOVERY_DATASET_VALIDATION.md`)** | **CANONICAL — production-reachable via `GET /cases/{id}/recovery-probability`, calibrated, but low live feature coverage (see `FEATURE_CONTRACT.md`)** |

**Selected canonical model:** Random Forest (`random_forest`), chosen by
validation PR-AUC, test ROC-AUC ≈ 0.88. Logistic Regression and XGBoost are
statistically competitive (within a few points) — reported as competitive,
not as a manufactured decisive win.

## What "canonical" means here, precisely

The Payment Failure Model remains the one actually driving today's live
recommendation ERV for every case — unchanged, not deprecated, still
useful as an upstream payment-mechanics signal. The Recovery Model is the
scientifically correct target for the business question RecoverChain
actually cares about (will the money come back), is real, trained,
calibrated, and live-reachable — but is not yet the sole driver of the
automated path, because its live feature coverage is currently too low for
most cases to responsibly replace the existing signal outright. See
`MODEL_LINEAGE_RECONCILIATION.md` for the full reasoning.

## Action-conditioned model

`ACTION_CONDITIONED_RECOVERY = DATA_BLOCKED`, unchanged from Phase 17 and
this session's earlier work. `action_outcome.xlsx` remains observational
and confounded by construction (kept specifically to demonstrate why naive
action-outcome comparisons are invalid — see
`data/recoverchain/manifests/DATASET_CATALOG.md` §3).
