# ERV Migration Criteria (operational)

Purpose: define what evidence must exist before the canonical recovery model could ever
replace the current production ERV probability. **No migration has happened.**
`PRODUCTION_ERV_CHANGED = NO`. Production ERV is still `amount × candidate probability`
(`application/action_evaluator.py`), and the canonical model is a read-only analytical
path (`GET /cases/{id}/recovery-probability`, `GET /cases/{id}/shadow-erv`).

Status vocabulary: PASS / PARTIAL / DATA_BLOCKED / FAIL. No overall score or ranking is
assigned, and passing tests is never treated as production readiness. No numeric
threshold below is invented: where one is needed the business/risk owner must fix it,
in writing, **before** the evaluation is run.

## 1. Artifact integrity
- **Evidence required:** registered artifact checksum verified before deserialisation; task, target and target definition; feature count/names/order; preprocessing metadata; the artifact's own internal schema; registry consistency; refusal with no fallback.
- **Current status:** **PASS** (`ml/inference/model_integrity.py`; 19 tests in `tests/ml/test_model_artifact_integrity.py`; production artifact `recovery_model_1790147239` sha256 `dd19d3af…fb68` registered from the real bytes and confirmed against an independent `sha256sum`).
- **Exact blocker:** none for integrity itself. Note: integrity proves the file is the registered one and schema-compatible; it says nothing about predictive quality.
- **Would resolve / keep valid:** re-register on every retrain (the training script now does this automatically).

## 2. Training/inference parity
- **Evidence required:** identical names/order/dtypes/vocabulary/preprocessing; identical semantic definition for every feature.
- **Current status:** **PARTIAL.** Names, order, dtypes, categorical vocabulary, scaler statistics and the dict→pipeline path are proven identical (`tests/ml/test_training_inference_parity.py`, 9 tests; predictions through the inference path equal direct training-path predictions to 1e-12). Rolling-history counters equal the generator's definition step by step. **Not identical:** `amount_vs_customer_average` (training = amount ÷ a fixed latent per-customer typical amount; live = amount ÷ mean of the customer's earlier amounts; Pearson 0.32 / Spearman 0.62 between the two on the synthetic data; never missing in training vs 17% cold-start missing live). Also: live history counts only *settled* prior cases, training counts every earlier obligation resolved inside the 30-day window.
- **Exact blocker:** the definition mismatch above. Permutation/ablation on the synthetic validation split shows the model does not depend on the feature (PR-AUC change −0.0006 permuted, −0.0002 set missing) — but that is *by construction* (the label generator never uses this feature), so it does not establish real-world insensitivity.
- **Would resolve:** a new model version trained with the live definition (or without the feature) and re-verified; or real outcome data showing insensitivity. Not done here (no retrain).

## 3. Feature coverage
- **Evidence required:** coverage on real cases per domain; owner-defined sufficiency; plan for blocked features.
- **Current status:** **FAIL** as a migration gate. Observed **8–12 of 23** on 228 dev cases (8:4, 9:150, 10:59, 11:11, 12:4); maximum possible today 14/23; 9 features `DATA_BLOCKED` (`MISSING_RECOVERY_TELEMETRY.md`).
- **Exact blocker:** nine features have no data source; five more are conditional.
- **Would resolve:** new telemetry (tenure, payment history, engagement, contact response, account status, invoice date) collected for enough history, or a retrain on a feature set the live data can actually supply.

## 4. Temporal integrity
- **Evidence required:** strictly-before-cutoff history; boundary, duplicate, future-event and monotonicity tests.
- **Current status:** **PASS** for the implemented features (`tests/ml/test_recovery_feature_coverage.py`, parity tests). Recovery is dated by outcome-event business time; a recovery after the prediction time is not history.
- **Exact blocker:** none for derived features; blocked features have no history at all.
- **Would resolve:** n/a; re-run on any new feature.

## 5. Calibration
- **Evidence required:** reliability/Brier/calibration error on held-out **live** outcomes.
- **Current status:** **DATA_BLOCKED.** A sigmoid calibrator object exists, fitted on the *synthetic* validation split. Existence of the object is not calibration evidence; every response states `calibration_basis = SYNTHETIC_VALIDATION_ONLY`. The raw score (e.g. 0.50) and calibrated score (0.14) differ a lot, so the raw score must never be read as a probability.
- **Exact blocker:** 0 production outcomes.
- **Would resolve:** authoritative production outcomes at a volume the owner defines, then an outcome-based calibration evaluation.

## 6. Outcome availability
- **Evidence required:** authoritative outcomes (`provenance = production`) for cases that carry a canonical prediction.
- **Current status:** **DATA_BLOCKED.** Production-provenance outcomes: **0**. On the 228 cases: 4 have an applied settlement, all `test` provenance (not evidence).
- **Exact blocker:** no real settlement feed.
- **Would resolve:** a registered evidence source delivering real settlements.

## 7. Financial evaluation
- **Evidence required:** ERV error, recovered-amount error, and false-positive/negative exposure of canonical vs current ERV on real outcomes, protocol fixed in advance.
- **Current status:** **DATA_BLOCKED.** Not computed; no accuracy, ROC-AUC, PR-AUC, Brier, calibration error or lift is reported against live outcomes.
- **Exact blocker:** criterion 6.
- **Would resolve:** criterion 6, then the pre-registered evaluation.

## 8. Shadow stability
- **Evidence required:** shadow ERV logged over a defined period; stability and unexplained-drift review; a real (not baseline-dominated) comparison.
- **Current status:** **PARTIAL.** The read-only endpoint and breakdown exist. 225/228 stored current probabilities are the deterministic baseline and only 3 come from a trained shadow model, so the comparison is mostly canonical-vs-baseline-heuristic. No logging period exists.
- **Exact blocker:** no persisted shadow log; baseline-dominated comparison.
- **Would resolve:** persist shadow outputs per case; run on cases scored by the trained shadow model.

## 9. Policy compatibility
- **Evidence required:** full policy/authorization/edge-case suites green with canonical ERV in a staged path; the model still cannot authorize anything.
- **Current status:** **PARTIAL.** Policy Engine untouched and its suites pass (679 tests); the canonical path is read-only and changes no case field (tested). Not exercised with canonical ERV inside the recommendation path because that path was deliberately not built.
- **Exact blocker:** no staged path (correct at this stage).
- **Would resolve:** a feature-flagged staging path, then the same suites.

## 10. Explainability / provenance
- **Evidence required:** every feature carries value, source, cutoff, records used, status; approximations and blocked features are disclosed in every response.
- **Current status:** **PASS** for provenance/disclosure. **Not done:** per-prediction feature attribution for the canonical model (only provenance, not contribution).
- **Exact blocker:** none for provenance; attribution absent.
- **Would resolve:** an explainability pass on the canonical model if migration is pursued.

## 11. Operational reliability
- **Evidence required:** deterministic refusal on integrity failure; no fallback; restart verification; latency/availability targets.
- **Current status:** **PARTIAL.** Refusal and restart verification are tested (API returns `MODEL_INTEGRITY_FAILED`; fresh-process load verified). No latency/availability targets or monitoring exist.
- **Exact blocker:** no operational SLOs/monitoring.
- **Would resolve:** defined targets and monitoring; a rollback switch to the current signal (also required before any migration).

## Also required before any migration
A written decision record by an accountable person; the rollback switch above; sign-off on the
disagreement review (`ml_reconstruction/lineage_disagreement.py`); resolution or explicit
acceptance of the `amount_vs_customer_average` difference. The Policy Engine remains
authoritative regardless of which probability feeds ERV.

**LIVE_OUTCOME_EVALUATION = DATA_BLOCKED.**
