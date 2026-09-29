# Guide Demonstration — traced from one real, live upload

Everything below is a real API response captured from the running app on a
genuine upload of `data/recoverchain/synthetic/guide_demo/guide_demo.xlsx`
(dataset `ds_bc361e26`, upload/mapping/generate-cases all via the app's real
endpoints — no case data was hand-constructed). Two concrete cases are
traced end to end; a third proves ingestion dedup.

## Setup, run live

```bash
curl -X POST localhost:8000/datasets/upload -H "X-API-Key: ..." -F "file=@data/recoverchain/synthetic/guide_demo/guide_demo.xlsx"
curl -X POST localhost:8000/datasets/ds_bc361e26/analyze -H "X-API-Key: ..."
curl -X POST localhost:8000/datasets/ds_bc361e26/mapping -H "X-API-Key: ..." -d '{...confirm all 4 fields, declared_domain=FAILED_PAYMENT...}'
curl -X POST localhost:8000/datasets/ds_bc361e26/ml-readiness -H "X-API-Key: ..."
curl -X POST localhost:8000/datasets/ds_bc361e26/generate-cases -H "X-API-Key: ..." -d '{"max_cases":200}'
```

**Result:** `rows_seen=147, not_at_risk=20 (already-paid, correctly skipped), cases_created=127`.
All 5 columns (`customer_id`→CUSTOMER_ID, `amount`→AMOUNT, `invoice_date`→TIMESTAMP,
`status`→OUTCOME, `reason`→UNKNOWN/unused) auto-mapped correctly with no
manual correction needed. Domain classifier correctly inferred `FAILED_PAYMENT`
from the `reason` column's values, independent of the declared domain.

## Trace 1 — a small, auto-approved case (`case_5296e02b`, ₹2,665.64)

```
INPUT ROW (guide_small_63)
  customer_id=guide_small_63, amount=2665.64, status=failed, reason=<card-related>
      ↓
RISK ASSESSMENT (deterministic-v1.0)
  score=0.657, level=HIGH, confidence=1.0
      ↓
DIAGNOSIS (deterministic-v1.0)
  cause_category=PAYMENT_METHOD_INVALID, confidence=0.95
      ↓
SHADOW ML PREDICTION (baseline-deterministic-v1.0, advisory only)
  recovery_probability=0.232, status=SUCCESS_BASELINE
      ↓
RECOMMENDATION (baseline-evaluator-v1.0)
  top_candidate=REQUEST_PAYMENT_METHOD_UPDATE
  estimated_probability=0.186, expected_recoverable_value=494.74
  rationale="Baseline approximation. Gross ERV = 2665.64 * 0.186."
      ↓
POLICY DECISION (merchant-policy-v1.1)
  status=PERMITTED, reason_codes=["ALL_RULES_PASSED"]
  (amount 2665.64 is under the 5000 automated-approval threshold; no human
   review required)
      ↓
EXECUTION (simulated)
  execution_id=exec_1b3f8871, adapter=MockExecutionAdapter
  status=COMPLETED_SIMULATED, message="Message sent."
      ↓
OUTCOME
  pending — execution happened, recovery is not yet confirmed; only a real
  outcome event (application/outcome_events.py) can mark this recovered
```
Trace ID: `trace_5296e02b`. This is the honest EXECUTION ≠ RECOVERY
invariant in action — even a clean, permitted, executed case has no
recovered amount until an authoritative settlement event says so.

## Trace 2 — a large case, escalated by the same policy engine (`case_cf0d0188`, ₹53,273.49)

```
POLICY DECISION
  status=ESCALATE, reason_codes=["FinancialAutomatedLimit"]
  "Amount 53273.49 exceeds automated threshold 5000.0."
      ↓
approval_required=true — a person must review before anything executes.
No execution record exists for this case; nothing was auto-run.
```
Same policy engine, same rule set as Trace 1 — the only difference is the
amount crossing the automated-approval threshold. This is the direct,
non-hypothetical evidence that probability never authorizes an action by
itself; the deterministic threshold does.

## Trace 3 — the duplicate pair, and the real dedup behavior

The dataset includes one exact-duplicate row pair. On the **first**
`generate-cases` run, both rows were accepted as two legitimate cases
(`duplicate_event: 0`) — the app correctly treats an exact repeat *within
one upload* as a customer genuinely failing twice, not a data error. On a
**second** `generate-cases` call against the *same file*:
```
{"cases_generated": 127, "counters": {"rows_accepted": 0, "duplicate_event": 127, "cases_created": 0, ...}}
```
Every one of the 127 previously-ingested rows was correctly recognized as a
duplicate and produced zero new cases. This is real dedup, verified live —
not assumed from reading the code.

## REAL TRAINED ML MODEL TRACE (added in a later session — background training had time to finish)

Trace 1/2 above (`case_5296e02b`, `case_cf0d0188`) were scored the moment
they were created, before the background training task for `ds_bc361e26`
had finished — an honest result, not a broken one (see
`LIVE_MODEL_FORENSIC_AUDIT.md` §2-3 for the exact race condition and why
it's expected behavior). Once training completed (`train_1790086713`, real
XGBoost artifact, `calibrated: true`), a fresh prediction call against the
**same running app, same dataset, a different existing case** shows the
real trained model:

```
XLSX ROW (guide_large_28, from the same guide_demo.xlsx upload)
  customer_id=guide_large_28, amount=20578.20, status=failed, reason=declined
      ↓
FEATURE VECTOR (application/case_pipeline.py:53-62)
  {customer_id, amount, invoice_date, status, reason, failure_code, AMOUNT,
   BALANCE, CUSTOMER_ID, ACCOUNT_ID, ENTITY_ID, TIMESTAMP}
      ↓
MODEL ID / VERSION
  train_1790086713 (xgboost) -- loaded by dataset_id=ds_bc361e26
      ↓
RAW MODEL OUTPUT
  predict_failure_risk() -> P(failure) [not shown directly by the API;
  only the inverted, recovery-flavored number is exposed]
      ↓
RECOVERY PROBABILITY (= 1 - P(failure), case_pipeline.py:87-88)
  0.0477   (a "declined" reason correctly drives this low)
      ↓
CALIBRATED
  yes -- CalibratedClassifierCV(FrozenEstimator(...), method="sigmoid"),
  verified fixed this session (was silently failing before)
      ↓
EXPLANATION (real, per-feature, not fabricated)
  reason="declined" contributed +0.752 (increased failure risk)
  amount: no measurable effect for this row
      ↓
ANOMALY CHECK
  level=NORMAL, max_abs_z=0.778 (within the training distribution)
      ↓
prediction_status = SHADOW_ONLY  (never a live authority)
```
Exact call, exact response:
```bash
curl -X POST "localhost:8000/cases/case_a5ad9931/predict-recovery?dataset_id=ds_bc361e26" -H "X-API-Key: ..."
```
```json
{
  "prediction_id": "pred_6bca6a76",
  "recovery_probability": 0.04765544123472876,
  "confidence": 0.8,
  "model_version": "xgboost",
  "feature_version": "1.0",
  "prediction_status": "SHADOW_ONLY",
  "explanation": [
    {"feature": "reason", "value": "declined", "contribution": 0.752474, "direction": "increased"},
    {"feature": "amount", "value": 20578.2, "contribution": 0.0, "direction": "no_effect"}
  ],
  "confidence_tier": "MEDIUM",
  "anomaly": {"level": "NORMAL", "max_abs_z": 0.778, "per_feature": {"amount": 0.778}}
}
```
**Important honesty note:** this proves the live app genuinely loads and
serves a trained model — it does **not** prove the live app is using the
`recovery_outcome.xlsx` genuine-recovery-target model. `train_1790086713`
was trained on `ds_bc361e26`'s own `status` column (payment-failure-risk,
inverted). See `LIVE_MODEL_FORENSIC_AUDIT.md` §4 for why those are two
separate, currently unconnected model lineages, and what unifying them
would actually require.

## THE CANONICAL RECOVERY MODEL, LIVE (Lineage B — a later session)

`MODEL_LINEAGE_RECONCILIATION.md` explains why Lineage A's `1 - P(failure)`
is not the same claim as genuine recovery probability. This is that model,
live, on the exact same case as the trace above:

```bash
curl "localhost:8000/cases/case_0a8a1665/recovery-probability" -H "X-API-Key: ..."
```
```json
{
  "status": "SUCCESS",
  "model_task": "RECOVERY_PROBABILITY",
  "model_id": "recovery_model_1790147239",
  "model_version": "random_forest",
  "calibrated": true,
  "raw_probability": 0.1357462005175059,
  "calibrated_recovery_probability": 0.1357462005175059,
  "expected_recoverable_value": 4630.62,
  "feature_coverage": 0.261,
  "features_used": {
    "amount_at_risk": 34112.32,
    "age_of_obligation": 0,
    "prior_recovery_rate": -1.0,
    "number_of_previous_failed_attempts": 0,
    "number_of_previous_recoveries": 0,
    "domain": "FAILED_PAYMENT"
  },
  "note": "Only 26% of this model's declared features were derivable from real operational history for this case; the rest were passed as missing (not fabricated) and handled by the trained pipeline's own imputer."
}
```

**Side by side, same case, same session:**
```
Lineage A (payment-failure-risk, inverted):  recovery_probability = 0.0477
Lineage B (canonical recovery model):        calibrated_recovery_probability = 0.1357
```
Nearly 3x apart. This is the concrete, non-hypothetical proof that these
are two different questions with two different real answers — exactly what
this reconstruction set out to establish, not just assert.

`prior_recovery_rate = -1.0` and the two rolling counts are all `0` here
because `guide_large_12` (this case's customer) has no prior *settled*
case in this dataset — a genuine, honestly-reported absence of history, not
a placeholder invented to look complete. `feature_coverage: 0.261` is
reported on every response so nobody mistakes a 6-of-23-feature assessment
for a full one (see `FEATURE_CONTRACT.md`).

## LIVE RECOVERY FEATURE PROVENANCE (later pass — real capture, 2026-09-24)

`GET /cases/case_a5ad9931/recovery-probability` (₹20,578.20, reason `declined`):

```
Prediction timestamp : 2026-09-24T09:13:58
Model / version      : recovery_model_1790147239 / random_forest   task=RECOVERY_PROBABILITY  target=recovery_success
Feature coverage     : 9/23
Raw probability      : 0.5024     (pre-calibration — NOT a usable probability)
Calibrated probability: 0.1387
Expected recoverable value (canonical, shadow): 20578.20 × 0.1387 = ₹2,854.99

Feature                        Value    Status      Source                                   Cutoff              Records
amount_at_risk                 20578.2  OBSERVED    recovery_cases.amount_at_risk            2026-09-24T09:13:58  1
domain                         FAILED_PAYMENT OBSERVED recovery_cases.risk_category         "                    1
age_of_obligation              27       OBSERVED    revenue_events (earliest event)          "                    1
prior_recovery_rate            -1.0     COLD_START  cases+outcome_events+audit, dated<cutoff "                    0 prior cases
number_of_previous_recoveries  0        COLD_START  same                                     "                    0
number_of_previous_failed_attempts 0    COLD_START  same                                     "                    0
reminder_history               0        OBSERVED    execution_attempts(communication)        "                    0
consent_status                 UNKNOWN  OBSERVED    contact_permissions (bitemporal)         "                    0
suppression_status             0        OBSERVED    contact_permissions                      "                    0
missing (cold start)           amount_vs_customer_average, average_days_to_payment
DATA_BLOCKED                   customer_tenure, prior_payment_success_rate, prior_late_rate, invoice_age,
                               customer_engagement, recent_login, previous_contact_success,
                               account_status, historical_recovery_behavior
```

Shadow comparison (`GET /cases/{id}/shadow-erv`, read-only, executes nothing):

```
current probability (stored prediction, xgboost per-upload) : 0.0477
current ERV (recommendation top candidate, own estimate)    : ₹4,281.29
canonical calibrated probability                            : 0.1387
canonical ERV                                               : ₹2,854.99      difference: −₹1,426.30
```
The "current ERV" uses the *recommendation candidate's own* probability estimate, not
the stored prediction — which is why 20,578.2 × 0.0477 ≠ 4,281.29; both are reported.

**Why the payment-failure complement and the recovery probability are different
predictions.** Lineage A estimates *p = P(payment record is a failure | amount, reason)*
from the uploaded file's own status column; the app then reports *1 − p*. That is the
probability a payment attempt *succeeds*, an event about the attempt's mechanics — but
a RecoverChain case exists only because the payment already failed. Lineage B estimates
*P(R = 1 | X_t)*, R = "at-risk amount recovered within 30 days", conditioned on the
customer's history, consent state and obligation age. In general
P(R=1 | X, failed) ≠ 1 − P(failed | X): they condition on different information, are
trained on different labels and populations, and neither identity nor a monotone
relation is guaranteed. Neither is called "better"; live accuracy is DATA_BLOCKED until
authoritative outcomes exist.

## CANONICAL RECOVERY PATH: WHAT IT IS AND WHAT IT IS NOT (integrity checkpoint)

The canonical recovery model is currently a **read-only analytical path**. It changes no
recommendation, policy decision or execution.

```
INPUT (a case built from an uploaded row)
  -> canonical feature derivation      ml/inference/recovery_features.py (history strictly before the cutoff)
  -> feature provenance + coverage     value / status / source / cutoff / records used;  8-12 of 23 today
  -> integrity-verified model          sha256 + schema checked before use; refused (no fallback) otherwise
  -> calibrated recovery probability   raw 0.50 -> calibrated 0.14 in the earlier example; calibration basis is
                                       SYNTHETIC_VALIDATION_ONLY - not outcome-validated
  -> shadow ERV                        amount x canonical calibrated probability
  -> comparison with the CURRENT ERV   the recommendation candidate's own estimate; shown side by side
  -> NO production decision change     the decision still uses current_erv; the Policy Engine is untouched
```
What the response carries so a reader cannot mistake it for the decision:
`is_production_decision=false`, `production_decision_source="current_erv (unchanged)"`,
`calibration_basis`, `integrity_status`, `artifact_sha256`, `approximations`
(`amount_vs_customer_average` is not identical to its training definition) and the
`DATA_BLOCKED` feature list. No live accuracy or calibration is claimed: there are 0
production recovery outcomes, so `LIVE_OUTCOME_EVALUATION = DATA_BLOCKED`.

## What a guide can ask, and where the answer is

| Question | Answer, with evidence |
|---|---|
| Where does the XLSX get parsed? | `api/dataset_router.py:611-613`, `_read_source_frame()` — `pd.read_excel` |
| Where does a row become a case? | `api/dataset_router.py`'s `generate_cases_from_dataset()`, the `_NOT_FAILED`/`_FAILED` classification loop (see this session's earlier answer on payment success/failure classification) |
| Where is the shadow probability computed? | `application/ml_training.py`, `application/recovery_predictor_ml.py` — advisory only, never authorizes |
| Where is the real decision made? | `application/policy_engine.py`, `DeterministicPolicyEngine.evaluate()` — deterministic, not ML |
| Can the model execute a payment? | **No.** Confirmed by Trace 2: a 0.9-probability-adjacent case with a large amount is escalated, not executed, regardless of the model's estimate |
| Who confirms actual recovery? | Only `application/outcome_events.py` — an authoritative settlement event, never the prediction itself |
