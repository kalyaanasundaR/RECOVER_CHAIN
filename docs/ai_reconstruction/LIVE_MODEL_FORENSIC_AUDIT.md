# Live Model Forensic Audit — real code trace, not inference

Every row below is cited to a file/line I actually read and, where marked
**[LIVE]**, an API call I actually made against the running app this
session. Nothing here is assumed.

## 1. Stage-by-stage trace

| Stage | File | Function/Class | Actual implementation | Status |
|---|---|---|---|---|
| XLSX parsing | `api/dataset_router.py:600-613` | `_read_source_frame()` | `pd.read_excel(file_path, nrows=nrows)` for `.xlsx`/`.xls` | VERIFIED |
| Schema mapping | `api/dataset_router.py` `/mapping` route | `confirm_mapping` (`DatasetLabService`) | User-confirmed column→canonical field mapping, stored on `DatasetMetadataModel.recoverchain_signals` | VERIFIED |
| ML readiness / target selection | `application/ml_readiness.py` | `MLReadinessAnalyzer.analyze_readiness()` | Auto-picks `target_column` from the mapped `OUTCOME` column, infers `prediction_problem` from the column name (`payment-failure-risk` by default) | VERIFIED |
| Model training | `application/ml_training.py` | `MLTrainingEngine.train_and_evaluate()` | Trains LR + XGBoost (+ calibration, now working — see below), gated at ROC-AUC≥0.55 / adaptive min test rows | VERIFIED |
| Trigger | `api/dataset_router.py:923-937` | `generate_cases_from_dataset()` | `background_tasks.add_task(run_ml_training_task, ...)` fired for every dataset with ≥20 rows and an ML-ready status | VERIFIED |
| Model artifact | `ml/models/registry/<run_id>_model.joblib` + `_metadata.json` | `MLTrainingEngine` (`joblib.dump`) | Real sklearn/XGBoost pipeline, real metadata (target, features, metrics, `calibrated: true`) | **VERIFIED LIVE** — `train_1790086713_metadata.json`, `dataset_id=ds_bc361e26` |
| Model loading | `application/recovery_predictor_ml.py:17-` | `MLPaymentFailurePredictor.__init__` | Scans `model_registry_dir()` for the newest non-`REJECTED_LOW_QUALITY` metadata file whose `dataset_id` matches and whose `task` is in `REVENUE_RECOVERY_TASKS` (`payment-failure-risk`, `late-settlement-risk`) | VERIFIED |
| Prediction (inference-time) | `application/case_pipeline.py:26-113` | `predict_recovery_for_case()` | Builds a feature dict from the case's own linked event, calls `predictor.predict_failure_risk()`, **inverts**: `recovery_probability = 1.0 - failure_probability` (line 87-88) | **VERIFIED LIVE** |
| Calibration | `application/ml_training.py:365-379` | `CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")` | Real (fixed this session — previously silently failing on scikit-learn 1.9.0; see `01_TRAINING_RESULTS.md` §1) | **VERIFIED LIVE**: `calibrated: true` in the ds_bc361e26 artifact |
| ERV | `application/case_pipeline.py` / recommendation builder | `CandidateAction.expected_recoverable_value` | `amount_at_risk × estimated_probability` ("Gross ERV = X * Y" in the stored rationale text) | VERIFIED |
| Policy | `application/policy_engine.py` | `DeterministicPolicyEngine.evaluate()` | Fully deterministic; unaffected by which probability source fed the recommendation | VERIFIED (unchanged this session) |
| Outcome | `application/outcome_events.py` | outcome-event ingestion | Only an authoritative outcome event marks a case recovered | VERIFIED (unchanged this session) |

## 2. Is the live path currently ML or deterministic fallback?

**Both exist and the code correctly chooses between them — but with an
important timing caveat, found live, not assumed.**

- `predict_recovery_for_case()` (`application/case_pipeline.py:69-107`) always
  tries the ML path first; only on `predictor.model is None`, a `ValueError`
  (incomplete features), or any other exception does it fall back to
  `DeterministicBaselinePredictor` (`application/recovery_predictor.py`).
  This is a genuine safety fallback, not a permanent stand-in.
- **Confirmed live:** cases generated in the *same request* that kicks off
  background training score against `baseline-deterministic-v1.0`, because
  the background training task cannot finish before the synchronous
  case-creation loop in that same request does. This matches this
  project's own code comment (`dataset_router.py:923-928`): *"This run's
  cases are scored by whatever model is on disk now... and the trained
  model is picked up on the next run."*
- **Also confirmed live — the actual gap:** `POST /cases/{id}/predict-recovery`
  (`api/main.py:621`) takes `dataset_id` as an **optional** query parameter.
  Calling it without `dataset_id` — which the frontend never supplies
  (`grep` across `frontend/src` found zero calls to this endpoint at all;
  it is API-only, not used by the UI) — returns the baseline even when a
  trained model exists for that exact dataset. Calling it **with**
  `dataset_id` explicitly returns the real model:

```
POST /cases/case_0a8a1665/predict-recovery                          -> model_version: baseline-deterministic-v1.0
POST /cases/case_a5ad9931/predict-recovery?dataset_id=ds_bc361e26   -> model_version: xgboost, prediction_status: SHADOW_ONLY,
                                                                        recovery_probability: 0.0477, real per-feature explanation,
                                                                        real anomaly score (level: NORMAL)
```

**Verdict: `LIVE_ML_STATUS = WIRED_AND_WORKING`.**
The wiring is not fake and not absent — it is a real capability. It was
never wrong to call the earlier guide trace's `baseline-deterministic-v1.0`
an honest result; it was correct for the case scored at that exact moment.

**FIXED, same session:** `predict_recovery_for_case()`
(`application/case_pipeline.py:26-`) now auto-resolves `dataset_id` from
the case's own `RevenueEvent.source_ref` — literally documented at the ORM
level as *"dataset_id for file replay"* (`infrastructure/orm.py:104`) —
whenever neither `dataset_id` nor `predictor` is supplied by the caller. An
explicitly passed `dataset_id` still wins, so the existing
`shared_predictor` call site in `dataset_router.py`'s case-generation loop
is unaffected. Verified live against the running app: the exact case that
previously returned `baseline-deterministic-v1.0` when called with zero
query parameters now returns `model_version: xgboost` on the identical
call. Also proven with a dedicated test,
`tests/ml/test_live_model_integration.py::test_dataset_id_is_auto_resolved_from_the_case_own_event`.
Full backend suite: 633 passed.

## 3. Why `baseline-deterministic-v1.0` appeared in the earlier guide trace

Not a broken integration — a race between synchronous case creation and an
async background training task, both correctly designed, that this repo's
own code comments already anticipated. No code change was required to make
the model exist; it already trained successfully (`quality_gate.passed:
true`, `calibrated: true`) within the same session, just after the
first-pass cases were scored.

## 4. Critical target semantics — what the LIVE shadow model actually predicts

**This is the real finding your brief is after.** The live per-dataset
shadow model (`ds_bc361e26`'s `train_1790086713`) has:
```
task: "payment-failure-risk"
target_column: "status"
```
`status` is the SAME `_NOT_FAILED`/`_FAILED` vocabulary column used to
decide which rows become cases at all (see this session's earlier answer
on payment success/failure classification). The model is trained on the
**whole uploaded file** (147 rows, including the 20 already-paid ones) to
predict each row's own success/failure — then, at inference time,
`case_pipeline.py` inverts it (`1 - P(failure)`) and calls the result
"recovery probability."

**Classification (per `02_SEMANTIC_TARGET_CLASSIFICATION.md`'s vocabulary):
this live shadow model is a `VALID_PROXY` for payment-failure risk, inverted
— not the `DIRECT_RECOVERY_TARGET` this session separately built in
`recovery_outcome.xlsx`.** It answers "how likely was this row/case to be a
payment failure" (inverted to a recovery-flavored number), not "will this
specific at-risk obligation actually be recovered within a defined
window." Those are related but distinct questions, exactly as §2-3 of the
brief warns against conflating.

**These are two separate, currently unconnected model lineages:**
1. **Live per-dataset shadow model** (`ml_training.py` → `recovery_predictor_ml.py`)
   — trains automatically on whatever a user uploads, task inferred from
   the file's own OUTCOME column. This is what the running app actually
   serves today, and it is real, calibrated, and correctly labeled
   `SHADOW_ONLY` — but its target is a failure-risk proxy, not a direct
   recovery target.
2. **`recovery_outcome` research model** (`backend/ml_reconstruction/train_recovery_outcome.py`)
   — trained on the genuine `recovery_success` target this session built
   specifically because nothing else in the repo had one. It is **not**
   wired into `recovery_predictor_ml.py` or the live serving path.

**Merging these is out of scope for this pass, and forcing it would be
dishonest.** `recovery_outcome.xlsx`'s schema (prior recovery rate,
customer engagement, consent status, dispute indicators, etc.) does not
match what a merchant's raw transaction upload provides
(`customer_id, amount, date, status, reason`); the live training pipeline
is architected around whatever a real upload contains, not a fixed feature
contract. Wiring the `recovery_outcome`-shaped model into live serving
would require either (a) deriving its exact feature set from real
operational history per customer (a genuine, nontrivial feature-engineering
project — not just a code change), or (b) accepting a second, differently-
shaped upload flow specifically for recovery-outcome-labeled historical
data. Neither is attempted here.

**Verdict: `TARGET_MERGE = DATA_BLOCKED — architecturally, not for lack of
a trained model.`** The live app's own "recovery probability" label should
be read as *"the complement of this dataset's own trained failure-risk
estimate"* — an honest, useful signal, but not literally
`P(recovery within N days | X)` until the two lineages are deliberately
unified, which is future work.

## 5. ERV type safety (§12)

Checked: `CandidateAction.expected_recoverable_value` is always computed as
`amount_at_risk × estimated_probability`, and `estimated_probability` is
always whatever `predict_recovery_for_case()` returned (baseline or
shadow-ML, always already the "recovery" complement, never a raw failure
probability). No code path multiplies amount by a failure probability and
labels it ERV. No separate `LatePaymentProbability` type exists in the
domain model — worth adding if the two model lineages above are ever
unified, but not a live bug today since only one probability type
(recovery-flavored, post-inversion) ever reaches ERV.
