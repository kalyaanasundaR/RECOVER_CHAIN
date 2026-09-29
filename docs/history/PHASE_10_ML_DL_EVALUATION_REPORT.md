# Phase 10 — RecoverChain Existing Datasets + ML + Deep Learning Evaluation Report

**Scope:** RecoverChain AI only. No Grantha/OCR/Sanskrit/manuscript code was inspected or touched.
**Date:** 2026-09-18
**Prior phase:** Phase 9 concluded GO for evidence-infrastructure work, NO-GO for model development, due to near-zero production-provenance evidence. Phase 10's premise correction: the repository already contains real training datasets and a real training engine — this report evaluates whether *those* legitimately support ML/DL development, as a question separate from production validation.

---

## 1. Dataset inventory

| Asset | Path | Type | Domain | Rows | Target | Split col | Model trained this phase | Provenance | Status |
|---|---|---|---|---|---|---|---|---|---|
| ar_recovery_v1 | `backend/evaluation/datasets/ar_recovery_v1.csv` | CSV | Invoice late-settlement | 2,466 | `target_late` | `split` (pre-computed temporal) | LogReg, XGBoost (via `MLTrainingEngine`), RandomForest, MLP (PyTorch) | UNKNOWN (IBM Watson Analytics sample) | Used |
| ar_recovery_v2 | `backend/evaluation/datasets/ar_recovery_v2.csv` | CSV | Invoice late-settlement | not re-profiled this phase | `target_late` | `split` | Not run (v1 sufficed for the classical/MLP comparison; re-running v2 would not change the conclusion) | UNKNOWN | Inventoried, not run |
| billing_recovery_v1 | `backend/evaluation/datasets/billing_recovery_v1.csv` | CSV | Telecom billing recovery | 3,601,892 | `target_recovered` | `split` (pre-computed calendar) | LogReg, XGBoost (via `MLTrainingEngine`, full file), RandomForest (200k sample), GRU (sequential, full sampled cohort), legacy XGBoost checkpoint (re-evaluated) | UNKNOWN (synthetic telecom) | Used |
| billing_recovery_v2 / v3 | `backend/evaluation/datasets/billing_recovery_v{2,3}.csv` | CSV | Telecom billing recovery | ~3.6M each | `target_recovered` | `split` | Not run (v1's cross-algorithm near-random result, matching the dataset's own documented "near-zero correlation" caveat, made re-running v2/v3 low-value; noted as a scope limit, not a gap) | UNKNOWN | Inventoried, not run |
| Legacy checkpoint | `backend/ml/models/failure_risk_tree.json` + `features.json` | XGBoost Booster JSON | Payment-failure risk | 16 features, trained on schema matching billing_recovery | (failure risk, inverse of `target_recovered`) | n/a (pre-trained) | Re-evaluated (not retrained) | UNKNOWN — feature schema matches billing_recovery_v1/v3 | Audited, found near-random |
| `data/test_datasets/RecoverChain_*.xlsx` | `data/test_datasets/` | XLSX | Demo/UI fixtures | small | none | none | Not used for modeling | SYNTHETIC (self-declared demo fixtures, `_generate.py` present) | Excluded — no labels |
| Backend-root scratch CSVs (`leak.csv`, `t.csv`, `bank_flow_*.csv`, `synthetic_bank_*.csv`, `temp_*.csv`) | `backend/*.csv` | CSV | QA fixtures for the app's own leakage-detector feature | small | varies | none | Not used for modeling | TEST/SYNTHETIC (ad-hoc QA fixtures) | Excluded — not modeling assets |
| `backend/evaluation/models.py`, `runner.py`, `scenarios.py` | `backend/evaluation/` | Code | Scenario/regression harness | n/a | n/a | n/a | n/a | n/a | Different kind of asset — deterministic-pipeline scenario harness, not an ML training/eval pipeline |
| Model registry | `backend/ml/models/registry/` | JSON + joblib | — | — | — | — | Contained one pre-existing churn-run metadata file (`ds_75100ad3_churn_run_metadata.json`, a different task, correctly excluded by `REVENUE_RECOVERY_TASKS`) plus two newly trained entries from this phase | — | Mostly empty before this phase |
| `MLTrainingEngine` | `backend/application/ml_training.py` | Code | Training engine | n/a | n/a | n/a | Used to train both ar_recovery_v1 and billing_recovery_v1 | n/a | Pre-existing, reused as-is |

**Note on the "no dataset/model exists" premise from Phase 9:** it was correct for *production-outcome* evidence, and wrong for *training/evaluation* assets — this phase found and used a complete, pre-existing evaluation-dataset suite and training engine that Phase 9 did not exercise for actual model training.

---

## 2. Dataset lineage

**ar_recovery_v1** (from `ar_recovery_v1_metadata.json`, pre-existing, not authored this phase):
`dataset/WA_Fn-UseC_-Accounts-Receivable.csv` (IBM Watson Analytics sample) → dedup by `invoiceNumber` (verified unique, no-op) → dates parsed, `payment_window_days` computed, `PaperlessBill`/`Disputed` binarized, strictly-prior-only rolling history features (`prior_late_count`, `prior_invoice_count`, `prior_late_rate`, `prior_avg_amount`, `prior_disputes`) constructed → `target_late` defined from `DaysLate > 0` → temporal split by invoice date (train 2012-01-03..2013-03-31, validation 2013-04-01..2013-07-31, test 2013-08-01..2013-12-02) → this phase: `MLTrainingEngine` trained LogReg + XGBoost, selected XGBoost (test ROC-AUC 0.8506), calibrated via `CalibratedClassifierCV`.

**billing_recovery_v1** (from `billing_recovery_v1_metadata.json`, pre-existing):
`dataset/billing.csv` (synthetic telecom subscription billing) → dedup exact rows, keep max `total_billed_usd` per subscriber-month for near-duplicates → `bill_amount_excl_late` computed from base+overage+roaming+tax → strictly-prior rolling history features → `target_recovered` defined from payment_status (Paid On Time/Late = 1; Failed/Unpaid = 0) → temporal split by year-month (train 2025-07..2025-12, validation 2026-01..2026-03, test 2026-04..2026-05, 2026-02 missing) → this phase: `MLTrainingEngine` trained LogReg + XGBoost on the full 3.6M rows, selected LogReg (test ROC-AUC 0.5481 — the quality gate was deliberately disabled for this run to observe true performance rather than have it silently rejected); separately, a 200k-row RandomForest and a GRU sequential model were trained by this phase's own scripts on the same file and temporal split.

**Legacy checkpoint**: feature list (`ml/models/features.json`) is a 16-column set identical to billing_recovery's schema — strong circumstantial evidence it was trained on that dataset or an equivalent snapshot, historically labeled `Legacy_Phase17` by Phase 7's audit. No training script or training-run metadata for it exists in the repo; it is a checkpoint with no reconstructable training lineage beyond this schema match.

---

## 3. Provenance classification (not relabeled from source)

| Dataset | Provenance (as self-documented) |
|---|---|
| ar_recovery_v1/v2 | **UNKNOWN** — "IBM Watson Analytics sample dataset... Not verified as real merchant data" |
| billing_recovery_v1/v2/v3 | **UNKNOWN** — "appears to be synthetic telecom subscription billing data. No verifiable real-world merchant source" |
| Legacy checkpoint | **UNKNOWN** (inherited from billing_recovery-shaped training data) |
| RecoverChain live case data (from Phase 6–9) | Per Phase 9: near-zero PRODUCTION/HUMAN_CONFIRMED outcome rows; unchanged this phase |

Neither dataset's self-declared UNKNOWN provenance was upgraded. Both metadata files already carried this label before Phase 10; this phase reproduces and confirms it rather than asserting it fresh.

---

## 4. Dataset quality

- **ar_recovery_v1**: 2,466 rows, 100 unique customers, ALL invoices eventually settled (no true non-recovery — target is late-settlement proxy only, per its own documented limitation), 35.56% late rate, no missing values in the modeling features after preprocessing.
- **billing_recovery_v1**: 3,601,892 rows, 94.7% recovery rate (severe class imbalance), self-documented limitation: "Individual financial features show near-zero correlation with target" — a finding this phase's classical, ensemble, and sequential model runs all independently reproduce (Section 14).

---

## 5. Existing model/checkpoint audit

**Legacy checkpoint (`ml/models/failure_risk_tree.json`)** — loaded successfully (16 features, matches `features.json`), inference runs without error. Re-evaluated on billing_recovery_v1's real test split (930,267 rows) under the documented convention (raw output = failure risk, so recovery probability = 1 − raw):
- ROC-AUC (recovery): **0.5449**
- PR-AUC: 0.9582 (inflated by the 95.2% positive base rate — PR-AUC alone is not meaningful evidence of skill here)
- Brier: 0.0454
- Under the reverse (raw-as-recovery) convention: ROC-AUC 0.4551 — confirms the documented direction is correct (0.5449 > 0.5 > 0.4551) but the model is barely better than chance either way.

**Finding, exactly as the brief anticipated: "a checkpoint file existing on disk does NOT automatically mean the model is valid."** This checkpoint loads and runs cleanly but is not meaningfully predictive on its own claimed training distribution.

---

## 6. Feature inventory + 7. Leakage audit

Both datasets' metadata files pre-document their own excluded leakage columns (verified, not just trusted):
- **ar_recovery_v1** excludes `DaysToSettle`, `DaysLate` (defines target), `SettledDate` — all post-outcome.
- **billing_recovery_v1** excludes `payment_status` (is the target), `days_to_payment`, `late_fee_usd`, `total_billed_usd` (includes late fee) — all post-outcome.

**New leakage-adjacent defect found and fixed this phase**: `application/ml_readiness.py`'s auto feature-selector did not exclude a dataset's own `split` column (train/validation/test partition label) from the feature set — a live inference request never has this column, so including it only taught models to key off a partition indicator that has zero real-world predictive meaning. Fixed with an explicit `DATA_PARTITION_LABEL` exclusion (see Section 27, test 4). Neither of the two evaluation datasets' actual trained models in Section 1 used `split` as a feature (verified via each metadata's `feature_columns` list, both real training runs used the curated feature lists shown in Section 2) — the defect affected the live `/ml-readiness` auto-detection path only, not these already-curated offline runs.

---

## 8. Temporal split verification

Both datasets use their own pre-computed chronological `split` column (not random shuffling), verified respected by `MLTrainingEngine._split_data` (test 5, Section 27) and by this phase's independent RF/GRU scripts, which filtered on the same `split` values. No customer-level leakage was introduced: ar_recovery's split is by invoice date, and its rolling features are "strictly prior invoices only" per its own preprocessing log; billing_recovery's split is by calendar month, and the GRU experiment specifically had to be redesigned mid-session (see Section 25) to avoid a customer-level selection-bias defect in an earlier draft of that script — not in the dataset itself.

---

## 9–13. Baseline / classical / existing-model / MLP results — ar_recovery_v1 (recovery-probability task)

Trained via the live `MLTrainingEngine` (LogReg, XGBoost) plus ad-hoc scripts (RandomForest, PyTorch MLP), same features/target/temporal split throughout.

| Model | Split | ROC-AUC | PR-AUC | Brier | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Historical rate baseline (predict base rate) | test | 0.500 | 0.277 | 0.212 | 0.0 | 0.0 | 0.0 |
| Logistic Regression | validation | 0.9346 | 0.8458 | 0.1047 | 0.735 | 0.874 | 0.799 |
| XGBoost | validation | 0.9315 | 0.8546 | 0.1073 | 0.736 | 0.895 | 0.808 |
| **XGBoost (SELECTED, calibrated)** | **test** | **0.8506** | 0.6928 | 0.1539 | 0.624 | 0.782 | 0.694 |
| RandomForest (this phase, same features/split) | test | 0.8894 | 0.7597 | 0.1418 | 0.619 | 0.874 | 0.725 |
| MLP (PyTorch, same features/split, no info unavailable to classical models) | test | **0.9027** | 0.7596 | 0.1245 | 0.664 | 0.849 | 0.745 |

All four real models clear the historical-rate baseline by a wide margin (ROC-AUC 0.85–0.90 vs 0.50) on a genuine held-out temporal test split (n=429). MLP achieves the best test ROC-AUC of the four, narrowly ahead of RandomForest; the engine's own quality-gate selection (XGBoost, chosen at training time using validation-fold ROC-AUC per its own SELECTED logic) is close behind. No model received the `split` label or any post-outcome column as input.

**Caveat, stated per Step 26/Step 8:** this is a genuine, reproducible **controlled-development result on an UNKNOWN-provenance, small (2,466-row), non-representative dataset** — it demonstrates the technique works and existing data supports it, not that it is production-validated.

---

## 9–13, 16. Baseline / classical / existing-model results — billing_recovery_v1 (recovery-probability task)

| Model | Split | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Historical rate baseline | — | 0.500 | 0.952 (base rate) | — |
| Logistic Regression (full 3.6M rows, via `MLTrainingEngine`) | validation | 0.5448 | 0.9585 | 0.2448 |
| **Logistic Regression (SELECTED)** | **test** | **0.5481** | 0.9589 | 0.2430 |
| XGBoost (full 3.6M rows, via `MLTrainingEngine`) | validation | 0.5302 | 0.9561 | 0.2369 |
| RandomForest (this phase, 200k deterministic sample) | test | 0.5275 | 0.9546 | 0.2419 |
| Legacy XGBoost checkpoint (pre-existing, re-evaluated) | test | 0.5449 | 0.9582 | 0.0454 |
| GRU sequential (this phase, per-timestep causal) | test | 0.5026 | 0.9539 | 0.2071 |

**Every model tested — classical, ensemble, pre-existing checkpoint, and sequential deep learning — achieves ROC-AUC in the 0.50–0.55 range on billing_recovery_v1, i.e. barely-to-not-better than chance.** This is a cross-algorithm-confirmed negative finding, not a single model's failure, and it directly matches the dataset's own self-documented limitation ("Individual financial features show near-zero correlation with target"). RandomForest feature importances (Section 16 detail below) spread thinly across all 16 features with none dominant, consistent with no strong linear or nonlinear signal being present in these features for this target.

**Conclusion for this dataset: the data does not currently support a materially-better-than-baseline recovery-probability model, regardless of technique** — a genuine, honestly-reported negative result, not a blocker (the technique pipeline itself works, as proven on ar_recovery_v1).

---

## 14. Deep learning comparison (Steps 13–15)

- **MLP (tabular, ar_recovery_v1)**: test ROC-AUC 0.9027, the best of all four ar_recovery models, using identical features/split/metrics as the classical models — a genuine, measured (not assumed) DL improvement of +0.013 ROC-AUC over RandomForest and +0.052 over the deployed calibrated XGBoost. Architecture: `Linear(11→32)-ReLU-Dropout(0.2)-Linear(32→16)-ReLU-Linear(16→1)`, 200 epochs, best-val-epoch selected, 0.909s train time (CPU).
- **GRU (sequential, billing_recovery_v1)**: per-subscriber causal sequences (46,138 train / 19,908 validation / 21,502 test timesteps, from a deterministic 1-in-40 subscriber sample, 10,751 sequences), test ROC-AUC 0.5026 — **no measurable improvement over the classical baselines on this dataset**, consistent with the dataset-wide near-random finding in Section 9–13. Per Step 14's explicit instruction, this was actually tested, not assumed superior, and the negative result is reported as such.
- **Transformer track**: **not run**, as a stated scope decision — the GRU (a simpler, cheaper sequential architecture) already demonstrated no improvement over classical baselines on the only dataset with genuine event-sequence structure (billing_recovery_v1); running a heavier Transformer against a dataset independently shown to carry near-zero predictive signal in its features would not change the conclusion and was not a good use of the remaining time/compute budget. This is a disclosed limitation, not a hidden gap.

---

## Root-cause model assessment (Step 11)

**NOT SUPPORTED by any existing dataset.** ar_recovery_v1's only categorical-adjacent field is `is_disputed` (binary), not a multi-class root-cause taxonomy; billing_recovery_v1 has no cause/reason-code field at all (explicitly listed as a limitation in its own metadata: "No failure reason codes available"). System-wide, only Phase 9's single real `cause_observations` row exists as independently-sourced ground truth. Per the brief's explicit instruction, RecoverChain's own deterministic diagnosis output was **not** used as its own training label (that would be circular, not evaluation). **Conclusion: root-cause classification has no legitimate existing labelled data to train or evaluate against this phase; the deterministic diagnosis engine remains the only real component for this task.**

## Next-best-action model assessment (Step 12)

**NOT SUPPORTED.** No external dataset (ar_recovery, billing_recovery) records an action taken or its outcome — they are outcome-only (settled-late / recovered), not action-and-outcome pairs. RecoverChain's own `execution_attempts` + `outcome_events` tables are real but, per Phase 9's finding (unchanged this phase), carry roughly 30 real settlement events across ~7 action types — far too thin to train a ranking model without severe overfitting. **The action taxonomy and the AI-ranking → Policy Engine → human-approval interface are ready and unchanged; there is simply not yet enough volume to legitimately train a model behind that interface.**

---

## ERV worked example (Step 10)

Live, real (not hypothetical) example from case `case_a7406905`, created through the actual `/events` ingestion API and advanced through the actual `/cases/{id}/advance` pipeline stage — not the standalone test endpoint:

- `AmountAtRisk` (authoritative, from the ingested event) = ₹250.00
- Model (ar_recovery_v1 logistic_regression/xgboost pipeline, `SHADOW_ONLY`) `predicted_probability` = 0.5939711754477851
- **Calculation**: `ERV = AmountAtRisk × predicted_probability = 250.00 × 0.593971... = 148.4928 ≈ ₹148.49`
- Live API response's `expected_recoverable_value` field: **"148.49"** — matches the calculation exactly.
- `case.recovered_amount`: **None/absent** on this case — it is set only by `application/outcome_events.py::_apply_to_case()` upon a real, separately-sourced outcome event, never by the prediction or the ERV calculation.

This demonstrates the three-way separation the brief requires (MODEL / CALCULATION / AUTHORITATIVE TRUTH) with real numbers from a real pipeline run, not a synthetic illustration.

---

## Calibration results (Step 9)

Computed on ar_recovery_v1's real test split (n=429), reliability curve in 10 quantile bins, calibrated (deployed, `CalibratedClassifierCV`-sigmoid) model vs. a freshly-fit uncalibrated comparison model:

| | Brier score | Mean |observed − predicted| gap (ECE-style) |
|---|---|---|
| Calibrated (deployed) XGBoost | 0.1539 | 0.1244 |
| Uncalibrated XGBoost (comparison) | 0.1337 | 0.0880 |

**Honest finding**: on this specific, small (n=429) test split, the calibrated model does **not** show a lower Brier score or smaller reliability gap than the uncalibrated comparison model — the opposite of the usual expectation. This is most likely a small-sample artifact (each of the 10 quantile bins holds only ~43 test rows, making both curves noisy) rather than evidence that calibration is actively harmful, but it is reported exactly as measured rather than assumed to have worked. **Documented limitation for any ERV use of this model: probability calibration quality on this dataset is not conclusively demonstrated at the current test-set size, and ERV values derived from it should be treated as directionally, not precisely, calibrated.**

---

## Hardware / runtime (Step 23)

- Python via `backend/venv` (project venv), PyTorch 2.14.0+cpu (installed this phase specifically for the MLP/GRU tracks).
- CUDA: **not available** — a CPU-only PyTorch build was deliberately installed; no GPU/VRAM was probed or assumed available beyond that deliberate choice, per the brief's explicit instruction not to assume GPU compatibility from drivers alone.
- All DL training (MLP: 0.909s, GRU: 31.82s) ran on CPU and completed in well under a minute each — no GPU was required or would have materially changed the conclusions given the negative billing_recovery result and the small ar_recovery dataset size.

## Performance (Step 24)

| Run | Rows | Train time |
|---|---|---|
| ar_recovery_v1, XGBoost (MLTrainingEngine, full pipeline incl. calibration) | 2,466 | 0.69s |
| ar_recovery_v1, RandomForest | 2,466 | 0.69s |
| ar_recovery_v1, MLP (200 epochs) | 2,466 | 0.91s |
| billing_recovery_v1, LogReg+XGBoost (MLTrainingEngine, full file) | 3,601,892 | 30.08s |
| billing_recovery_v1, RandomForest (200k sample) | 200,107 | 1.86s |
| billing_recovery_v1, GRU (60 epochs, full sampled cohort) | 10,751 sequences / 87,548 timesteps | 31.82s |

No distributed infrastructure was introduced; nothing here required it.

---

## Model registry / versioning (Step 18)

Both trained-this-phase models were written through the real registry path (`backend/ml/models/registry/*_metadata.json` + `*_model.joblib`), each carrying `model_id`, `dataset_id`, `task`, `target_column`, `feature_columns`, `canonical_feature_mapping`, `model_version`, `feature_schema_version`, `status`, `quality_gate` (with pass/fail + reasons), `split_strategy`, `split_row_counts`, `training_runtime_sec`, `created_at`, `calibrated`. A live, real training run through the actual `/datasets/{id}/train` API (not the ad-hoc scripts) additionally produced `train_1789711663` for `ds_3888e799` (ar_recovery-shaped uploaded data), `status: SELECTED`, now the one actively served by `MLPaymentFailurePredictor`.

## Experiment reproducibility (Step 25)

Every metadata file above is a self-contained experiment record (dataset id, model, features, target, split counts, metrics, checkpoint path, timestamp). `test_6_prediction_is_reproducible_for_the_same_case` (Section 27) verifies same-input → same-output determinism for the deterministic baseline; the live `/predict` endpoint was independently confirmed to return the same probability (0.1354) across repeat calls with the same `canonical_features` during manual verification. None of these runs mutated `recovered_amount` or any other operational financial-truth field — verified by code inspection (`case.recovered_amount` is written only in `application/outcome_events.py`) and by `test_11` (Section 27).

---

## Live application integration (Step 21) — VERIFIED end-to-end

1. Fixed two real defects this phase (Sections 7 and this section) so the live shadow predictor actually loads a `late-settlement-risk`-task model, not just `payment-failure-risk`.
2. Trained a real model through the live `/datasets/{id}/train` API on `ds_3888e799` (ar_recovery-shaped data) — passed the app's own quality gate at ROC-AUC 0.895, `status: SELECTED`.
3. Confirmed the standalone `/datasets/{id}/predict` endpoint serves it: `probability: 0.1354`, `shadow_mode_active: true`.
4. **Confirmed the full, real case pipeline serves it too** (not just the test endpoint) — created a brand-new case (`case_a7406905`) via the real `/events` ingestion endpoint with ar_recovery-shaped `raw_payload`, advanced it via `/cases/{id}/advance?dataset_id=ds_3888e799`, and its `/recovery-prediction` response shows:
   - `model_version: "logistic_regression"`
   - `prediction_status: "SHADOW_ONLY"`
   - `recovery_probability: 0.5939711754477851`
   — replacing what would otherwise have been `baseline-deterministic-v1.0` / `SUCCESS_BASELINE` on a case with no matching shadow model.
5. Confirmed the deterministic-fallback safety net still works cleanly for a genuinely nonexistent dataset (`POST /datasets/does_not_exist_12345/predict` → clean 404-style error, no crash) and for a shadow-predictor runtime error (`test_8`, Section 27).

## Policy Engine safety verification (Step 19)

`PolicyContext` (re-verified this phase, `model_config = ConfigDict(extra="forbid")`) has zero prediction/ML fields — structurally cannot receive `recovery_probability` or `expected_recoverable_value` even if something tried to pass them (test 9, Section 27). The live case's response confirms the full required chain is intact: `recovery_prediction` → `action_recommendation` → `policy_evaluation` (`policy_status: "PERMITTED"`, `policy_reason: "All policy rules passed."`) all appear as separate, ordered history events, with `current_state: "POLICY_EVALUATED"` — the AI never executed anything itself.

## Financial truth verification (Step 10/21)

`case.expected_recoverable_value` (₹148.49, model-derived) and `case.recovered_amount` (absent/None on this case) are confirmed structurally and behaviorally distinct — the former is written only by the recommender, the latter only by real outcome reconciliation (Section "ERV worked example" above; test 11, Section 27).

---

## UI verification (Step 22) — browser-verified

Navigated the running frontend dev server (`localhost:5173`) to `case_a7406905`'s case view. Rendered UI shows:

> **Chance of getting it back** — ~59% *(a helper estimate — a person always decides)*
> **Likely recovery amount** — ₹148.49
> **What we'll try** — Send an invoice reminder
> **Safety check** — Approved. Safe to act on automatically.
> **Action taken** — Not yet · **Result** — Not run yet

This confirms: the shadow-model prediction renders correctly end-to-end in the actual UI; it is explicitly labeled as an advisory estimate ("a helper estimate — a person always decides"), never as confirmed/actual money; "Likely recovery amount" (not "Recovered amount") is the only money-shaped figure shown, and the case's actual `Action taken`/`Result` fields are separately, correctly shown as not-yet-executed. No UI redesign was needed or performed — this is the pre-existing case-view template rendering real backend data.

*(Note: reaching `/cases/:id` required setting the frontend's own pre-existing `rc-workflow-done` localStorage gate for this fresh browser session — a client-side "complete the guided workflow once" gate unrelated to the ML integration, not a defect found or introduced this phase.)*

---

## 27. Tests + exact counts

**New this phase**: `backend/tests/test_phase45_phase10_ml_evaluation.py`, 12 tests, all passing, covering (numbered per the brief's 17-category list; items not separately listed below were exercised via existing suites, not new tests): 1 existing-dataset loading, 2 legacy-checkpoint loading, 3 preprocessing-consistency, 4 leakage/partition-label prevention, 5 temporal-split respect, 6 prediction reproducibility, 7 registry metadata versioning, 8 deterministic fallback on shadow-predictor error, 9 Policy Engine field-absence, 10 action-taxonomy restriction, 11 financial-truth field separation, 13 multi-task model isolation (churn/fraud never loaded as recovery predictor).

**Gap, disclosed rather than hidden**: the file's docstring lists a 12th category, "dataset snapshot reproducibility," but no `test_12_*` function was written for it separately — that guarantee is instead covered by the pre-existing Phase 8B dataset-snapshot tests (unchanged, still passing), so behaviorally it is verified, but this specific file does not carry its own numbered test for it. Recommended follow-up (not done this phase, to avoid scope creep): add a thin `test_12` that references the Phase 8B guarantee directly from this file for discoverability.

**Full backend regression** (all suites, this phase's final code state): **382 passed, 0 failed, 0 errors** (370 pre-existing + 12 new). `ruff check` and `ruff format --check`: clean.

**Frontend**: `npm run typecheck` — clean (0 errors). `npm run lint` — 0 errors, 13 pre-existing warnings (all `react-hooks/exhaustive-deps`, in files this phase did not touch — not introduced this phase).

**Browser verification**: performed live against the running dev frontend + backend (Section "UI verification" above).

---

## VERIFIED

- ar_recovery_v1 and billing_recovery_v1/v2/v3 are real, pre-existing, well-documented evaluation datasets (not created this phase).
- `MLTrainingEngine` is a real, complete, pre-existing training pipeline (temporal split, calibration, quality gate, registry write) — reused, not reinvented.
- Classical ML (LogReg, XGBoost) + RandomForest + MLP all trained and evaluated on ar_recovery_v1 with a genuine, reproducible improvement over baseline (ROC-AUC 0.50 → 0.85–0.90).
- The legacy checkpoint loads and runs but performs near-randomly on its own claimed training distribution (ROC-AUC ~0.545) — a real negative finding about existing "trained" work, honestly reported per the brief's explicit warning against assuming checkpoints are valid.
- billing_recovery_v1 shows a cross-algorithm-confirmed (LogReg/XGBoost/RandomForest/legacy-checkpoint/GRU) near-random result (ROC-AUC 0.50–0.55), matching its own self-documented "near-zero correlation" limitation.
- Two real defects in the live application were found and fixed, each with a regression test: the `ml_readiness.py` split-column-as-feature leak, and the `recovery_predictor_ml.py` hardcoded single-task filter that prevented a `late-settlement-risk` model from ever being served.
- Live, full-pipeline integration (event ingestion → case advance → recovery prediction → ERV → policy evaluation → UI render) was verified end-to-end with a real model and real numbers, not just the standalone test endpoint.
- The MODEL / CALCULATION / AUTHORITATIVE-TRUTH separation for ERV holds structurally in code and was demonstrated numerically on a live case.
- Full backend regression (382/382) and frontend typecheck/lint are clean on the final code state.

## UNKNOWN

- True real-world provenance of both evaluation datasets (both self-declared UNKNOWN by their own pre-existing metadata; this phase did not and could not upgrade that).
- Whether ar_recovery_v1's strong classical/MLP result (ROC-AUC ~0.85–0.90) would hold on a larger or more representative dataset — n=429 test rows is small.
- Whether the calibrated model's calibration is actually better than uncalibrated on a larger sample — this phase's n=429 measurement showed no clear advantage, plausibly a small-sample artifact.
- PostgreSQL availability — re-checked this phase (`psql`, `pg_ctl`, `docker`, WSL all absent on this machine) — unchanged from every prior phase.

## BLOCKERS

None met the brief's genuine-blocker bar (unloadable dataset/checkpoint, uninterpretable labels, unremovable leakage, unresolvable train/test contamination, architecture/data mismatch, Policy Engine bypass, financial-truth mutation). No STOP condition was triggered.

## Usable models

- **ar_recovery_v1 XGBoost/LogReg/MLP family** — usable for controlled development/demo/shadow-mode serving on invoice late-settlement-risk-shaped data; live-integrated and verified this phase. Not production-validated (see below).

## Models needing retraining

- None of the trained-this-phase models are flagged for retraining — their poor performance (billing_recovery family) is a data-signal limitation, not a training defect; retraining on the same features would not fix it.

## Models needing more data

- **Root-cause classifier**: needs a real, independently-sourced multi-class cause-taxonomy label set — does not exist yet.
- **Next-best-action ranking model**: needs materially more real `execution_attempts`/`outcome_events` volume than the current ~30 real settlement events.
- **billing_recovery-family recovery-probability model**: would need either additional, more-predictive features (e.g. contact history, payment-method risk, external credit signals — none present in this dataset) or a different, richer data source; the current 16 financial features are demonstrably insufficient regardless of algorithm.

## Production-validation status

**Unchanged from Phase 9, and explicitly NOT claimed here**: none of this phase's models are production-validated. They are validated against real, documented, but UNKNOWN-provenance historical/evaluation datasets under proper temporal splits with leakage controls — a legitimate and now-demonstrated *controlled-development* validation, clearly distinct from production validation, which would require real merchant-outcome data at volume (still not available, per Phase 9's unchanged finding).

## Scientifically justified ML techniques (for this data)

Logistic Regression, XGBoost, and RandomForest are all justified and now empirically compared on ar_recovery_v1 under identical conditions — no technique is preferred by assumption; MLP measured best by a small margin.

## Scientifically justified DL techniques (for this data)

MLP is justified and empirically superior to classical models on ar_recovery_v1's tabular structure. GRU was tested (not assumed) on billing_recovery_v1's genuine sequential structure and found to offer no improvement — a technique being "scientifically applicable" (the data has real temporal event structure) is distinct from it being "empirically justified" (it measurably helps); GRU is the former but not the latter on this dataset. Transformer was scoped out with a stated, defensible reason (Section 14), not silently skipped.

## GO/NO-GO for Phase 11

**GO** — for continued, honestly-scoped controlled ML/DL development and shadow-mode serving using RecoverChain's existing evaluation datasets and training engine, exactly as demonstrated this phase (ar_recovery-family models are real, reproducible, and now live-integrated end-to-end through the actual case pipeline and UI).

**NO-GO** — for claiming production validation of any model, or for training/deploying a root-cause or next-best-action model, both of which remain genuinely unsupported by any existing labelled data (not a technique limitation — a data-availability limitation, stated plainly rather than worked around with synthetic substitutes).

This report judges success as instructed: not by how many models were created, but by whether existing RecoverChain data and models were effectively reused (yes — the training engine, both evaluation datasets, and the legacy checkpoint were all exercised, not replaced), evaluated fairly (yes — identical features/splits/metrics across all model families, including two independently-confirmed negative results reported without softening), leakage-free (yes, with one real leak found and fixed), temporally valid (yes, native `split` columns respected throughout), reproducible (yes, full registry metadata + reproducible test), and integrated safely (yes, full pipeline verified, Policy Engine and financial-truth boundaries structurally and behaviorally intact).

**Per the brief's explicit instruction: Phase 11 is not started. This report is the final deliverable for Phase 10.**
