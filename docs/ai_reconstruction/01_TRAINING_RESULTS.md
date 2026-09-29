# RecoverChain AI Reconstruction — Real Training Results

Everything below is a genuine, executed training run. Run scripts and raw
JSON results are in `backend/ml_reconstruction/` (`train_ar_recovery.py`,
`train_billing_recovery.py`, `train_ar_recovery_mlp.py`; results under
`ml_reconstruction/runs/`). All models trained/evaluated on the dataset's
own pre-computed `train`/`validation`/`test` split — never on the app's
live operational data (`evaluation.runner.EvaluationRunner` and this
reconstruction both stay off that store).

## 1. Bug found and fixed during this work

`backend/application/ml_training.py` calibrated every trained model with
`CalibratedClassifierCV(..., cv="prefit")`. scikit-learn removed `cv="prefit"`
in 1.6+; this repo pins **scikit-learn 1.9.0**. Every calibration attempt
raised and was silently swallowed by a broad `except Exception`, so the
live shadow model (the one behind the UI's "recovery probability") has been
serving **uncalibrated** probabilities while its metadata still recorded
`calibrated: true` whenever the earlier steps happened to succeed. Fixed
with `sklearn.frozen.FrozenEstimator`, the current replacement API. Verified:
full 617-test backend suite still passes; calibration now measurably runs
(Brier score improves on every model in both ladders below).

## 2. `ar_recovery_v1` — classical ML ladder (2,466 invoices, real IBM Watson sample data)

Target: `target_late` (1 = settled late). Features and leakage exclusions
per the dataset's own pre-existing `ar_recovery_v1_metadata.json` (not
re-derived here — it was already correct). Selection rule: validation PR-AUC
(imbalanced positive class, 35.6% late), matching the app's own selection
criterion in `ml_training.py`.

| Model | Test ROC-AUC | Test PR-AUC | Test Accuracy | Test F1 | Calibrated |
|---|---|---|---|---|---|
| Baseline 0 — majority class | 0.500 | 0.277 | 0.723 | 0.000 | — |
| Baseline 1 — historical rate, no fitting | 0.847 | 0.650 | 0.802 | 0.679 | — |
| **Logistic Regression (selected)** | 0.894 | **0.750** | 0.823 | 0.681 | yes |
| Random Forest | 0.887 | 0.740 | 0.830 | 0.714 | yes |
| XGBoost | 0.885 | 0.728 | 0.834 | 0.708 | yes |

**Interpretation:** the three trained models are statistically indistinguishable
on a 429-row test set (1-2 points apart) — this is exactly the outcome
Gorishniy et al. predict: no tabular model family dominates by default.
Logistic Regression wins the app's own selection rule and is the most
interpretable — a defensible pick, not an arbitrary one. Notably, the
**no-fit historical-rate heuristic already reaches 0.847 ROC-AUC** — most of
the exploitable signal is in `prior_late_rate` alone; the trained models add
real but modest lift on top of it.

## 3. `ar_recovery_v1` — MLP comparison point (§21-22)

Framework: `sklearn.neural_network.MLPClassifier` — this venv has no
torch/tensorflow installed, and 2,466 rows does not justify installing one
(Rule 0.3/0.4: don't force DL, don't force a heavier framework than the
problem needs). This is still a genuine neural network trained by
backpropagation with the Adam optimizer, not a stand-in.

- Architecture: `Dense(32, ReLU) → Dense(16, ReLU) → sigmoid`, small on
  purpose (params vs. 1,594 training rows).
- Trained 38 epochs (early-stopped on an internal 15% validation split of
  train, not touching our real validation/test folds); training loss fell
  0.769 → 0.362; internal validation accuracy rose 0.379 → 0.821 with no
  late-epoch drop (`overfit_flag: false`).
- **Test ROC-AUC: 0.906, PR-AUC: 0.767, accuracy: 0.846** — on this run the
  MLP is the best single model on all three metrics, ahead of Logistic
  Regression (0.894 / 0.750 / 0.823) and XGBoost (0.885 / 0.728 / 0.834).
  This is a genuine result, not manufactured — but on a 429-row test set,
  a several-point gap is within the range small-sample noise can produce.

**Honest read:** the MLP is not decisively better. On this dataset size the
difference between 0.894 and 0.906 ROC-AUC is not something a professor
should be told is "deep learning wins" — it's "roughly tied, DL used no more
data or unfair advantage, and happens to land slightly ahead this run."
Re-running with a different `random_state` would plausibly flip the
ranking. If this needs a single production choice, Logistic Regression
remains the recommendation for interpretability at this data size.

## 4. `billing_recovery_v3` — weak-signal result at scale (3,601,892 rows)

Target: `target_recovered` (1 = all rows for that subscriber-month were
Paid/Paid Late; 0 = any Failed/Unpaid). Per this dataset's own
pre-existing `billing_recovery_v3_metadata.json`: *"89% of subscribers with
Failed/Unpaid status in any month later show successful payment in
subsequent months. The target captures per-cycle payment failures, which
are mostly temporary in this dataset."* — flagged as a likely weak-signal
label before this run happened, exactly as your brief's §69 anticipated.

| Model | Test ROC-AUC | Test PR-AUC | Notes |
|---|---|---|---|
| Baseline 0 — majority class | 0.500 | 0.930 | 93% positive rate makes PR-AUC misleadingly high |
| Baseline 1 — historical rate, no fitting | 0.517 | 0.937 | barely above random |
| Logistic Regression (selected by val PR-AUC) | 0.543 | 0.937 | — |
| Random Forest | 0.547 | 0.939 | — |
| XGBoost | 0.545 | 0.938 | — |

**Result: near-random.** Every trained model lands at ROC-AUC 0.54-0.55 —
barely above the 0.50 coin-flip line, and at the default 0.5 threshold every
model (including the trained ones) collapses to "always predict recovered"
(`recall=1.0, tn=0`) despite `class_weight="balanced"`. PR-AUC looks high
only because 93% of rows are the positive class — the majority-class
baseline alone gets 0.930 PR-AUC by doing nothing. **This confirms, rather
than manufactures, the weak-signal finding already on record.** No
threshold tuning or feature engineering was applied to try to improve this
number — reporting it as found.

**Verdict: `MODEL DEPLOYMENT BLOCKED` for `billing_recovery_v3` as a
non-recovery predictor.** The per-cycle proxy target doesn't carry enough
signal at this feature set to be useful; a defensible next step (not
attempted here — out of this pass's scope) would be redefining the label
around genuine multi-month non-recovery rather than a single noisy cycle.

## 5. `recovery_outcome` (new synthetic dataset) — first genuine recovery target in this repo

Built this turn: `scripts/data_generation/generate_recoverchain_datasets.py`
→ `data/recoverchain/synthetic/recovery_outcome/`. See
`data/recoverchain/manifests/DATASET_CATALOG.md` for the full dataset
description; this is the training-ladder sanity check.

Target: `recovery_success` (1 = recovered within a 30-day window). 6,030
rows, 17.2% positive rate, temporal split (4,221 / 904 / 905).

| Model | Test ROC-AUC | Test PR-AUC | Test Accuracy | Test F1 |
|---|---|---|---|---|
| Baseline 0 — majority class | 0.500 | 0.168 | 0.832 | 0.000 |
| Baseline 1 — historical rate, no fitting | 0.720 | 0.360 | 0.726 | 0.421 |
| **Logistic Regression (selected)** | **0.845** | **0.561** | 0.865 | 0.483 |
| Random Forest | 0.846 | 0.561 | 0.872 | 0.496 |
| XGBoost | 0.840 | 0.567 | 0.863 | 0.483 |

**Interpretation:** all three trained models land in a tight 0.840-0.846
ROC-AUC band, clearly ahead of both baselines and nowhere near a suspicious
~1.0 — the signature of a realistic, honestly-noised label rather than a
leaked or trivial one. This confirms the generator's label mechanism
(§ documented in the script itself) produces a genuinely learnable-but-hard
problem, exactly as the reconstruction brief requires ("realistic signal +
noise + ambiguity," "a 70% model with valid evaluation is preferable to a
fake 98% model").

## 6. What this does and does not establish

**Established, with evidence:** a real classical-ML ladder with an honest
baseline floor, a real leakage-respecting split, real calibration (after
fixing the bug above), a real — if modest and single-run — DL comparison,
and a real negative result on the large dataset reported without spin.

**Not established by this pass:** GRU/LSTM/Transformer (no dataset here has
a genuine per-entity event sequence — both datasets are one-row-per-
obligation, not sequences, so there is no data to justify them, per Rule
0.4/§23); action-conditioned/causal recovery modelling (§30-32 — this repo
has no treatment/action/outcome-labeled data to support it, so NBA stays
`DATA_BLOCKED`, unchanged from Phase 17); LLM/RAG rebuild (§51-56 — the
forensic audit already found the LLM path is real-but-unwired; fixing that
wiring is separate work, not attempted here); explainability, error
analysis, OOD detection, model versioning/registry unification, and the
remaining 19 documents (§75) are all out of scope for this pass.
