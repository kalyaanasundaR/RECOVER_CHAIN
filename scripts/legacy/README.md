# scripts/legacy

One-off scripts from the phase-by-phase build history (dataset construction,
validation, bias checks, ad-hoc inspection, and the `patch_*` migration helpers).

They are **not** part of the application and are not run by CI or `run.py`. Kept
for provenance only. Dataset build history lives in `docs/history/`.

`phase17_train.py` / `phase17_train_out.txt` here are this folder's own
original training run against the billing dataset (note: this project's
older internal phase numbering — unrelated to the later, current "Phase 17"
authorization work under `docs/history/PHASE_17_*`). It already correctly
named the target **Payment Failure Risk**, not recovery, and already found
the same near-random result (ROC-AUC ~0.545) independently reproduced in
`backend/ml_reconstruction/train_billing_recovery.py` — see
`docs/ai_reconstruction/01_TRAINING_RESULTS.md`.

Training entry points, current:
- `backend/run_phase19_training.py` — a standalone XGBoost-only run against
  `billing_recovery_v3.csv` (its metadata sidecar file is empty; no
  surviving metrics for that specific run).
- `backend/ml_reconstruction/train_ar_recovery.py`,
  `train_billing_recovery.py`, `train_ar_recovery_mlp.py` — the same
  experiments redone with a full baseline ladder, Random Forest, working
  calibration, and a written report. Prefer these when reproducing results.
