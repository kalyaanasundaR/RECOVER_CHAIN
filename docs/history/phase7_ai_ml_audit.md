# Phase 7 — AI/ML Intelligence Foundation: Pre-Flight Audit

This document is Steps 1–13 of the Phase 7 brief. It is an audit and a design, not
a trained model — see [phase7_final_report.md](phase7_final_report.md) for the
governing conclusion and GO/NO-GO.

## Step 1 — AI/ML pre-flight audit: what labelled data actually exists

Queried directly against the live dev database (`backend/test_recoverchain.db`),
not estimated:

```
Total cases:                396
Cases with >=1 REAL, matched outcome_events row: 12
  (application/outcome_events.py — Phase 5's authoritative outcome model)
Distribution across domains:
  OVERDUE_INVOICE        5
  CHECKOUT_ABANDONMENT   2
  BROKEN_PROMISE         2
  FAILED_SUBSCRIPTION    2
  FAILED_PAYMENT         1
```

**Every one of these 12 labels was produced by me, manually, during Phase 5 and
Phase 6 runtime/dataset verification** — deliberately posting settlement,
cancellation, and promise outcome events to prove the reconciliation mechanism
works. They are not organic merchant behaviour. Two consequences:

1. The *count* (12) is the honest ceiling on what a supervised model could be
   fit or evaluated on today, and it is far below any usable threshold.
2. Even if the count were larger, this specific set of 12 has severe selection
   bias (I chose which obligations got a full settlement, a partial, a
   cancellation, etc., to exercise every code path) — it does not reflect a
   real population's recovery behaviour and must never be treated as
   representative even for descriptive statistics.

The other 384 cases (`CLOSED_NOT_RECOVERED`: 89, `FULLY_RECOVERED`: 73 minus the
12 real ones = 61 of those are legacy, etc.) carry **pre-Phase-5 simulated**
outcomes — `case.outcome` set by the old `MockOutcomeVerificationAdapter`, which
Phase 5 established is never authoritative. Using them as labels would be
training a model to reproduce a random/scripted simulator, not real recovery
behaviour. They are correctly excluded.

### Per-task audit

| Task | Input features | Target/label | Source of truth | Sample count | Verdict |
|---|---|---|---|---|---|
| A. Recovery probability | risk, diagnosis, amount, age, domain, permitted action | binary/continuous "recovered" | `outcome_events` (Phase 5+, authoritative only) | **12**, all domains combined; 1–5 per domain | **UNAVAILABLE** — orders of magnitude short |
| B. Expected recoverable value | Task A's probability × amount | (derived, not separately labelled) | same as A | 12 | **UNAVAILABLE** — depends entirely on A |
| C. Failure/root-cause classification | `failure_code` / `reason` text | `cause_category` | `application/diagnosis_engine.py`'s own deterministic rule | hundreds of rows (all domains, Phases 4–6) | **UNAVAILABLE for a different reason** — see below, not a count problem |
| D. Next-best-action advisory (ranking) | case features + candidate actions | which action actually recovered money | `outcome_events` joined to `execution_attempts` | **11** SETTLEMENT-matched events with a known action | **UNAVAILABLE** — far too sparse to rank anything |

**Task C is not a sample-size problem — it's a circularity problem.** The
"target" (`cause_category`) is not an independently observed fact; it is
itself 100% deterministically computed from the same `failure_code`/`reason`
text a classifier would use as input (`application/diagnosis_engine.py`'s
`EvidenceBuilder` + `DeterministicDiagnosisEngine`, both pure rule-based, no
randomness). Training a supervised classifier to predict cause_category from
failure_code would just be learning to re-derive a lookup table that already
exists in code — it cannot add information, and reporting an accuracy score
for it would be misleading (it would trivially approach 100% because the
"label" IS a function of the input). The one place a classifier COULD add
value — resolving the `UNKNOWN`/`CONFLICTING_EVIDENCE` cases where the rules
don't have enough signal — has no ground truth at all (no human ever
confirmed the true cause for those), so there is still no legitimate target.
**Marked UNAVAILABLE, but for the correct reason: not a data-volume gap, a
label-source gap.**

## Step 2 — the four ML tasks, evaluated

- **A. Recovery probability** — blocked by data (12 samples). See above.
- **B. Expected recoverable value (ERV)** — already implemented, but observe
  the label carefully: `case_pipeline.py`'s `predict_recovery_for_case()`
  currently sources "recovery probability" from one of two places:
  1. `DeterministicBaselinePredictor` (hand-tuned heuristic, honestly labelled
     `prediction_status: SUCCESS_BASELINE`, `model_version:
     baseline-deterministic-v1.0`) — used whenever no per-dataset model exists.
  2. `MLPaymentFailurePredictor` (a real, trained XGBoost/sklearn model, but
     labelled `SHADOW_ONLY`) — **when one does exist**. Its target, however,
     is `TARGET`/`OUTCOME` from the *uploaded dataset's own `status` column*
     (did this row already say "paid" or "failed" at ingestion time) — i.e.
     it is a **payment-failure risk classifier**, not a recovery model. The
     pipeline then computes `recovery_probability = 1 − P(row_was_a_failure)`.
     This is a genuine, pre-existing (not introduced this phase) semantic
     mismatch: "the complement of whether this row historically failed" is
     not the same question as "will this already-failed obligation now
     recover." It happens to correlate loosely (harder-to-fail customers are
     often also easier to recover from), but it is not the right target, and
     it was never trained or evaluated against real recovery outcomes. This
     is flagged here as a genuine limitation of the existing (pre-Phase-7)
     shadow model, not fixed this phase (see §Step 11 and the final report's
     recommendations) — fixing it would require exactly the recovery-outcome
     labels that don't yet exist in sufficient volume.
- **C. Root-cause classification** — not beneficial; see Step 1.
- **D. Next-best-action advisory** — blocked by data (11 action-outcome pairs).

**No model was trained this phase for any of A/B/D.** ERV as currently computed
(`amount_at_risk × recovery_probability`, where the probability comes from the
baseline heuristic in the overwhelming majority of real cases) continues to
run unchanged — it was already correctly labelled as advisory before Phase 7
and needed no change to remain honest.

## Step 3 — data leakage audit

Performed against `FeatureExtractor.extract_features()` (`application/
recovery_predictor.py`, the ONLY feature-extraction path currently feeding any
prediction) and against what a legitimate Task A model's feature set would need
to be, since none exists yet:

**Currently used features** (verified by reading the function, and locked down
by `test_phase41_ai_ml_safety.py::test_4`): `risk_score`, `risk_category`,
`amount`, `event_count`, `cause_category`, `diagnosis_confidence`,
`diagnosis_status`, `age_hours`. Every one of these is knowable at the moment
a case is diagnosed — before any action, before any outcome. **None of the
following ever appears in this feature set**, verified by direct inspection
and now enforced by a regression test that greps every produced feature key
for these substrings: `recovered`, `outcome`, `verification`, `settlement`,
`cancelled_amount`, `execution_record`.

**If/when Task A becomes buildable** (real data accumulates), the leakage
risks below must be excluded from its feature set — documented now so the
design is ready, not invented as an afterthought:

| Leakage source | Why it leaks | Rule |
|---|---|---|
| `outcome_events.occurred_at` / `ingested_at` | Only exists after settlement | never a feature |
| `case.recovered_amount` / `outstanding_amount` / `cancelled_amount` | IS the (derived) label | never a feature |
| `execution_attempts.status` (COMPLETED/REJECTED) *at or after prediction time* | An action outcome, not an obligation fact | only usable if strictly prior to the prediction timestamp (e.g., "3rd attempt on this obligation" is fine; "this attempt's own result" is not) |
| `verification_source` / `reconciliation_status` | Post-hoc, by definition | never a feature |
| Any duplicate/replayed `external_event_id` | Reflects operational noise, not the obligation | dedupe before feature computation |
| Future events on the same obligation (a later re-open, a later duplicate) | Time-travel | features computed strictly from events with `timestamp <= prediction_timestamp` |

## Step 4 — prediction timestamp and temporal split design (not yet executable)

**Explicit prediction timestamp, per task**: the moment `case.diagnosis` is
first set (i.e., right after diagnosis, before recommendation/policy/execution)
— this is the existing pipeline's natural cut point and matches what
`predict_recovery_for_case()` already runs at.

**Temporal split design**, ready for when volume exists: split by
`case.linked_events[0].timestamp` (the triggering event's time), not by row
index or a random shuffle, so no case's outcome can leak into the training
window for an earlier case. Recommended shape once ≥500 labelled cases exist
per domain:

```
TRAIN period:      earliest available -> T - 60 days
VALIDATION period:  T - 60 days -> T - 30 days
TEST period:        T - 30 days -> most recent complete outcome window
```

The test period boundary must lag "now" by at least one full obligation's
typical resolution time (per domain) so that cases in the test window have
actually had time to reach a terminal outcome — otherwise the test set is
itself censored (most rows would still be "pending," not a real negative).

**This cannot be executed today** — 12 total labelled cases cannot be split
three ways and produce anything statistically meaningful. The design above is
documented so it is not invented under time pressure once data exists.

## Step 5 — baselines first (computed, not trained)

Per the brief's own instruction ("historical recovery rate," "action-specific
recovery rate" as simple baselines before any model): computed directly from
the 12 real labels, **explicitly marked NOT USABLE as a predictive baseline**,
shown only to demonstrate the computation a real baseline would use once
enough data exists:

```
Overall "recovered something" rate (12 samples): 9/12 = 75%
  FULLY_RECOVERED: 8   PARTIALLY_RECOVERED: 1   CANCELLED: 2   PENDING: 1

Per-domain (too small to report a rate meaningfully):
  OVERDUE_INVOICE (5):        4 fully, 1 partial
  CHECKOUT_ABANDONMENT (2):   1 fully, 1 cancelled
  FAILED_SUBSCRIPTION (2):    1 fully, 1 cancelled
  BROKEN_PROMISE (2):         1 fully, 1 pending
  FAILED_PAYMENT (1):         1 fully
```

75% is almost certainly **not** the real recovery rate of anything — it
reflects that I, testing the mechanism, posted mostly successful settlements
on purpose. Presenting this number as a system capability would be exactly
the kind of "invented label" the brief prohibits. It is reported here purely
as the baseline *computation method* (a simple historical rate, no model),
ready to be recomputed honestly once real merchant outcome data accumulates.

**No logistic regression, tree model, or any trained baseline was fit.**
Fitting one on N=12 (and then claiming it "beat" anything on a test set that
can't exist) would be fabricating a result. This is the direct application of
the brief's own STOP condition: *"training data insufficient."*

## Step 6 — calibration

Not applicable — there is no trained probability to calibrate. The existing
`DeterministicBaselinePredictor` is a hand-tuned heuristic, not a probabilistic
model fit to data, so ROC-AUC/PR-AUC/log-loss/Brier score/reliability curves
have no meaning against it (there is no held-out label set it was fit toward).
Once Task A has enough data to train against, its evaluation must include all
of the metrics the brief lists — noted here as a requirement for that future
work, not computed now.

## Step 7 — action-conditioned modeling

`P(recovery | action, context)` cannot be legitimately estimated: only 11
SETTLEMENT-matched outcome events exist with a known prior recommended
action, spread across 5 distinct action types — 1–3 observations per action.
No causal claim is possible even in principle from this data (no
randomization, no counterfactual, tiny N) and no *observational* estimate is
credible either at this volume. **Marked UNAVAILABLE.** When real data
accumulates, any estimate here must be explicitly labelled `observational`
(never `causal`) in its output metadata — recovery following an action is not
evidence the action caused it (e.g., a customer might have paid anyway).

## Step 8 — model contract (template, since no model is trained)

This is the contract every future Task A/B/D model must satisfy before being
wired into the advisory layer — written now so it isn't improvised later:

```
model_name:            recovery_probability_v1  (not yet built)
task:                  P(recovery | obligation, context, permitted_action)
feature_schema:        risk_score, risk_category, amount, currency,
                        event_count, cause_category, diagnosis_confidence,
                        diagnosis_status, age_hours, domain, permitted_action
                        (see Step 3's leakage table for exclusions)
target:                case reaches FULLY_RECOVERED or PARTIALLY_RECOVERED
                        via an authoritative outcome_events row within a
                        domain-specific resolution window
prediction_timestamp:  moment of first diagnosis (before recommend/policy)
training_dataset_id:   <dataset export id + row range, once real>
evaluation_dataset_id: <held-out temporal window, once real>
metrics:                ROC-AUC, PR-AUC, log-loss, Brier score,
                        reliability curve, vs. historical-rate baseline
calibration_status:    UNCALIBRATED until proven otherwise
known_limitations:     <filled in per model>
confidence/uncertainty: model must emit a confidence alongside probability
                        (RecoveryPrediction.confidence already exists in the
                        domain model - reuse it)
fallback_behavior:     any error/timeout -> DeterministicBaselinePredictor
                        (already implemented, now hardened - see Step 10)
reproducibility:        prediction = f(model_version, feature_snapshot,
                        input_case) - the existing RecoveryPrediction schema
                        already carries model_version + feature_version +
                        contributing_features (a frozen snapshot dict); this
                        was true before Phase 7 and is now covered by a
                        regression test (test_phase41_ai_ml_safety.py::test_1_2_3)
```

## Step 9 — AI output safety (already largely satisfied by the existing schema)

`domain.models.RecoveryPrediction` already carries: `prediction_id`,
`recovery_probability`, `confidence`, `model_version`, `feature_version`,
`prediction_timestamp`, `contributing_features` (the feature snapshot),
`prediction_status`. This is essentially the full metadata contract the
brief asks for, and it predates Phase 7 — no schema change was needed.
Verified this phase: AI output never overwrites raw source data, never
overwrites `case.outcome` (only `application/outcome_events.py`'s
authoritative reconciliation path does that — confirmed unchanged), and
never changes `case.policy_decision` (confirmed: `application/policy_engine
.py` contains zero references to `prediction` or `recovery_probability`
anywhere in its decision logic).

## Step 10 — integration and fallback (one real defect found and fixed)

The conceptual flow the brief specifies was already the actual flow:
`DETECTED → normalize → diagnose → [AI advisory: predict_recovery_for_case]
→ recommend → Policy Engine → [human approval if required] → execute →
authoritative outcome → reconciliation`. Advisory placement is correct and
was not changed.

**Defect found**: `predict_recovery_for_case()` only caught `ValueError` from
a shadow model's `predict_failure_risk()` call. Any *other* exception — a
corrupted model artifact, a native-library crash (relevant given Step 13's
findings below), an unexpected data shape — would propagate uncaught and
crash the pipeline stage instead of falling back to the deterministic
baseline, directly violating "SYSTEM MUST CONTINUE SAFELY USING DETERMINISTIC
FALLBACKS." Caught by a new regression test
(`test_phase41_ai_ml_safety.py::test_5_6`) that injects a predictor which
raises `RuntimeError`. **Fixed**: broadened the catch in
`application/case_pipeline.py` to also catch general `Exception`, falling
through to the baseline exactly as a missing model already did. This is the
one production-code change made this phase, and it is a strict safety
hardening, not a new capability.

## Step 11 — existing heuristics audit

| Heuristic | Where used | Labelled clearly? | Replaceable now? | Fallback if unavailable |
|---|---|---|---|---|
| `DeterministicBaselinePredictor` | `case_pipeline.py`, whenever no per-dataset shadow model exists (the overwhelming majority of cases) | Yes — `prediction_status: SUCCESS_BASELINE`, `model_version: baseline-deterministic-v1.0` | No — this IS the fallback; nothing to fall back to below it | N/A, it is the floor |
| `MLPaymentFailurePredictor` (XGBoost/sklearn, per-dataset) | Same call site, when a trained model exists for the dataset | Yes — `prediction_status: SHADOW_ONLY` | Not this phase — wrong target (Step 2), and blocked by Step 13's environment findings | Falls to the baseline above (now hardened, Step 10) |

Neither heuristic is silently relabelled as "ML" when it isn't, or vice versa.
No change was needed here beyond the Step 10 fallback hardening.

## Step 12 — LLM: do not train yet

**LLM TRAINING = BLOCKED BY DATA.** Audited `application/real_llm_adapter.py`
and `application/llm_adapters.py`: both are thin adapters that call an
external LLM API (Gemini) for structured assistance — there is no local
model, no fine-tuning code, and critically **no labelled corpus of any kind**
in this repository for any candidate LLM task (explanation generation,
domain-text normalization, structured diagnostic assistance). No corpus was
manufactured. If a future phase wants LLM-assisted explanation generation, it
can use the existing API-adapter pattern (advisory, structured output) without
training anything — that is a legitimate near-term option distinct from
fine-tuning, and is not blocked by data, only by product decision (out of
scope here).

## Step 13 — hardware / runtime (a real, significant finding)

```
Python:     3.14.7
PyTorch:    NOT INSTALLED
xgboost:    installed (venv), version unresolved to a stable import this session
scikit-learn: installed (venv), version unresolved to a stable import this session
scipy:      installed (venv) - native extensions INTERMITTENTLY BLOCKED
GPU:        NVIDIA GeForce RTX 5050, driver 592.82, CUDA 13.1, 8151 MiB, idle
```

**The significant finding**: scipy's compiled extension modules (used
transitively by both scikit-learn and xgboost) are being **intermittently
blocked by a Windows Application Control policy** on this machine —
`ImportError: DLL load failed while importing <module>: An Application
Control policy has blocked this file.` This was reproduced repeatedly this
session, with a *different* specific sub-DLL failing each time
(`_ccallback_c`, `_sparsetools`, `_ufuncs`, `XGBClassifier`'s own native
loader) — consistent with a real-time security control (not a broken
install) nondeterministically intercepting native-code loads. Across 4
consecutive full-suite runs today: 2 failed at test *collection* (12 then 2
import errors), 1 failed 16 tests at runtime with the same `ImportError`
family, and 1 ran completely clean (336/336). This did **not** occur in the
Phase 5/Phase 6 sessions two days ago — it is new environment drift between
2026-09-15 and 2026-09-17, not caused by any code in this repository.

**What this means concretely**: this environment cannot currently be trusted
to reliably train or serve any scikit-learn/xgboost-based model — even one
with sufficient data, training could fail non-deterministically partway
through, or a served model could intermittently fail at inference time (which
is exactly why Step 10's fallback hardening matters regardless of the data
question). No attempt was made to disable, bypass, or route around the
Application Control policy — that would be inappropriate. **Recommendation**:
ask IT/endpoint-security to allow-list this venv's `scipy`, `scikit-learn`,
and `xgboost` native extension DLLs, or provision a separate, dedicated ML
environment (a devcontainer, a cloud training environment, or a machine
without this policy) for any future model-training work. Do not silently
work around this by changing dependency versions or package sources.
