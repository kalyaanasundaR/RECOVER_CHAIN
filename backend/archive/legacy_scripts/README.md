# Archived — superseded

- `run_phase19_training.py` — a one-off standalone training script against
  `billing_recovery_v3.csv`. Its trained artifact (`ml/models/registry/failure_risk_tree.json`)
  survives but its metadata companion file is empty (0 bytes) — no accuracy/
  ROC-AUC/split info recoverable for it. Superseded by
  `backend/ml_reconstruction/train_billing_recovery.py`, which repeats this
  experiment properly with baselines, Random Forest, calibration, and an
  honest report (see `docs/ai_reconstruction/01_TRAINING_RESULTS.md`).
  Kept here as research evidence per the reconstruction's "archive, don't
  silently destroy" rule — not deleted.
