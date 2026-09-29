# Phase 7 — AI/ML Intelligence Foundation: Final Report

Companion document: [phase7_ai_ml_audit.md](phase7_ai_ml_audit.md) (Steps 1–13,
full detail). This report is the required 23-section summary + verdict.

**Headline: no new model was trained. Real, authoritative recovery-outcome
labels are too scarce (12 total, across all 5 domains) to legitimately train
or evaluate anything. Per the brief's own STOP conditions, this is reported
as a blocker rather than worked around with fabricated data.** One real
safety defect in the existing advisory fallback path was found and fixed.
One real environment defect (intermittent native-DLL blocking by a Windows
security policy) was found and documented, not worked around.

---

## 1. AI/ML task inventory

Four tasks were evaluated per the brief: (A) recovery probability, (B)
expected recoverable value, (C) failure/root-cause classification, (D)
next-best-action advisory. See §2 for verdicts.

## 2. Available labelled data

**12 cases** with a real, authoritative outcome event
(`OutcomeEventModel.case_id IS NOT NULL AND match_status='MATCHED'`), out of
396 total cases in the dev DB. All 12 were produced by me during Phase 5/6
verification, not organic merchant behaviour — severely selection-biased in
addition to being far too few. Per-domain: OVERDUE_INVOICE 5,
CHECKOUT_ABANDONMENT 2, BROKEN_PROMISE 2, FAILED_SUBSCRIPTION 2,
FAILED_PAYMENT 1. Full detail: audit doc §Step 1.

## 3. Data gaps

- No real merchant recovery-outcome history exists anywhere in this system —
  Phase 5 (which introduced authoritative outcomes) is only ~2 days old in
  this dev environment.
- No independent ground-truth label exists for root-cause beyond what the
  deterministic rule engine already computes (a circularity gap, not a volume
  gap — see audit doc §Step 1).
- No LLM training corpus of any kind exists for any candidate LLM task.
- No held-out temporal test window can exist yet (nothing has had time to
  "age" through a realistic resolution period under the new outcome model).

## 4. Feature definitions

Current production feature set (`FeatureExtractor.extract_features`):
`risk_score`, `risk_category`, `amount`, `event_count`, `cause_category`,
`diagnosis_confidence`, `diagnosis_status`, `age_hours` — all computed
strictly at/before the diagnosis stage, before any action or outcome exists.
Full leakage-safe feature design for a future real model: audit doc §Step 3.

## 5. Leakage audit

No leakage found in the current feature-extraction path (verified by
inspection and now locked down by `test_phase41_ai_ml_safety.py::test_4`, which
asserts no produced feature key contains `recovered`/`outcome`/`verification`/
`settlement`/`cancelled_amount`/`execution_record`). Full leakage-source table
for future model-building: audit doc §Step 3.

## 6. Prediction timestamps

Defined as the moment `case.diagnosis` is first set — before recommendation,
policy, or execution. This matches where the existing pipeline already runs
`predict_recovery_for_case()`; no change was needed.

## 7. Temporal split design

Documented (train/validation/test by triggering-event timestamp, with a
resolution-window-aware test boundary) but **not executable today** — 12
labelled cases cannot be usefully split three ways. Full design: audit doc
§Step 4.

## 8. Baseline results

A simple historical recovery rate was computed from the 12 real labels
(75% "recovered something") **and is explicitly flagged as not usable** — it
reflects the fact that I, testing the reconciliation mechanism, deliberately
posted mostly-successful settlements, not any real population's behaviour.
No baseline model (logistic regression, tree, or otherwise) was fit — fitting
one against N=12 and reporting a "result" would itself be a fabrication.
Full detail: audit doc §Step 5.

## 9. Model results

**None. No model was trained this phase.** Per the brief's STOP condition:
*"legitimate ML labels are unavailable" / "training data is insufficient."*
This condition was triggered honestly rather than worked around.

## 10. Calibration results

Not applicable — no trained probability exists to calibrate. See audit doc
§Step 6.

## 11. Action-conditioned analysis

Not legitimately possible: only 11 SETTLEMENT-matched outcomes have a known
prior action, spread across 5 action types (1–3 observations each). Marked
**UNAVAILABLE**. When real data accumulates, any future estimate here must be
labelled `observational`, never `causal`. Full detail: audit doc §Step 7.

## 12. Model contract

A complete template contract (feature schema, target, prediction timestamp,
metrics, calibration status, fallback behaviour, reproducibility guarantee)
was written for the **first future model that does get built**, so that work
starts from a ready specification rather than improvising one under time
pressure. See audit doc §Step 8. `RecoveryPrediction`'s existing schema
(`model_version`, `feature_version`, `prediction_timestamp`,
`contributing_features`, `confidence`) already satisfies most of the
reproducibility/metadata requirement and predates this phase.

## 13. AI safety / integration architecture

Unchanged conceptual flow, confirmed correct: `DETECTED → deterministic
normalization → AI advisory (diagnosis/prediction) → deterministic Policy
Engine → permitted action → human approval if required → execution →
authoritative outcome → reconciliation`. Verified this phase:
- `PolicyContext` (the sole input to policy decisions) has zero ML/prediction
  fields, and `extra="forbid"` structurally prevents one from being smuggled
  in (test_phase41::test_7).
- `application/policy_engine.py` contains no reference to `prediction` or
  `recovery_probability` anywhere in its decision logic (grep-verified).
- A prediction can never mutate a case's financial/lifecycle truth
  (test_phase41::test_8_9).
- AI output (`RecoveryPrediction`) always carries full provenance metadata
  (test_phase41::test_1_2_3).

## 14. Fallback behavior

**One real defect found and fixed**: the shadow-model call path only caught
`ValueError`; any other exception (a corrupted model file, a native-library
crash — directly relevant given §16's findings) would have crashed the case
pipeline instead of falling back to the deterministic baseline. Fixed in
`application/case_pipeline.py` by broadening the catch to `Exception`, so
*any* AI/ML failure now falls through to `DeterministicBaselinePredictor`,
exactly like "no model available" already did. Covered by a new regression
test that injects a raising predictor and confirms safe fallback.

## 15. LLM feasibility assessment

**LLM TRAINING = BLOCKED BY DATA.** No training corpus exists for any
candidate task. No corpus was manufactured. The existing API-based adapter
pattern (`real_llm_adapter.py`) remains a legitimate non-training path for
future advisory/explanation features, but building on it was out of this
phase's scope (Phase 7's brief is about the evaluation/training foundation,
which this finding shows isn't ready yet regardless).

## 16. Hardware/runtime findings

Python 3.14.7, no PyTorch installed, an idle RTX 5050 GPU (8GB, CUDA 13.1)
available but irrelevant to this phase's tree-based/linear tooling. **The
significant finding**: scipy's native extensions (used transitively by
scikit-learn and xgboost, both already present in the venv from earlier
phases) are being **intermittently blocked by a Windows Application Control
policy** — reproduced across 4 full-suite runs today with different specific
DLLs failing each time, consistent with a real-time security control, not a
broken install. This is new environment drift since the Sep 15 session (2
days ago), not caused by anything in this repository. No attempt was made to
bypass it. Recommendation: an IT allow-list exception for this venv's
scipy/scikit-learn/xgboost native DLLs, or a dedicated ML environment for any
future training work. Full detail: audit doc §Step 13.

## 17. Phase 6 carry-forward status

- **A. `outcomes_count` staleness — FIXED.** `GET /system/summary` now
  returns both the legacy `outcomes_count` (396, unchanged meaning, kept for
  backward compatibility, now documented in-code as counting legacy simulated
  outcomes too) and a new `authoritative_outcomes_count` (12 — real, matched
  outcome events only). Verified live against the running dev server.
- **B. PostgreSQL — still UNKNOWN.** Re-checked: no `psql`, `pg_ctl`, or
  `docker` on PATH; WSL itself is not installed on this machine (confirmed via
  `wsl -l -v`, not just "no distro"). No infrastructure was installed to force
  a check, per instruction. Carried forward as UNKNOWN, now with slightly
  more certainty about why.
- **C. No production-scale claim made.** Phase 6 was semantic validation (61
  rows); this phase touched no new scale question. Explicitly not claiming
  production readiness or production-scale performance anywhere in this
  report.

## 18. Tests and exact counts

New file: `backend/tests/test_phase41_ai_ml_safety.py` — 5 tests covering
reproducibility/versioning/snapshot integrity, no-leakage, deterministic
fallback under both "no model" and "model raises," Policy Engine's structural
immunity to ML fields, and prediction non-mutation of case truth. All use
only deterministic code paths (no sklearn/xgboost dependency), so they are
unaffected by §16's environment flakiness.

**Full backend suite: 336 passed, 0 failed** (331 pre-existing Phase 1–6
tests + 5 new Phase 7 tests), in a clean run this session. Given §16's
documented flakiness, this specific run was clean; three earlier same-day
runs hit the intermittent DLL-block issue in ML-dependent test files
(unrelated to any Phase 7 code — reproduced identically before any Phase 7
change was made). No existing test was weakened, skipped, or deleted.

`ruff check` / `ruff format --check` on all changed files: clean. Frontend
`tsc --noEmit`, `prettier --check`, `eslint`: all clean/unchanged (0 errors,
13 pre-existing warnings — no frontend file was touched this phase, since no
UI defect required a fix).

## 19. Browser verification

Backend and frontend dev servers were restarted fresh this phase (both had
stopped since the last session) and confirmed working: `GET /health` returns
`{"api":"ok","db":"ok"}`; `GET /system/summary` shows the new
`authoritative_outcomes_count: 12` field live. In the browser, a previously
fully-recovered case (`case_a37fa855`) still renders correctly — critically,
its recovery-probability estimate is still clearly labelled **"~54% (a helper
estimate — a person always decides)"**, unchanged, confirming the advisory
framing survived this phase's one code change untouched. No new AI-facing UI
was built this phase (none was needed), so no further browser verification
was performed beyond confirming the app runs and the existing advisory
labelling is intact.

## 20. VERIFIED

- Exact count and provenance of every real, authoritative outcome label in
  the system (12, all self-generated test data).
- No feature-leakage in the current advisory prediction path.
- The Policy Engine is structurally immune to ML input (schema-enforced, not
  just by convention).
- AI predictions never mutate case financial/lifecycle truth.
- AI predictions carry full reproducibility metadata.
- A previously-unhardened AI failure path (non-`ValueError` exceptions) is now
  caught and falls back safely — closing a real gap between the stated
  Phase 7 safety requirement and the actual code.
- `outcomes_count` staleness (Phase 6 carry-forward) is fixed and verified
  live.
- Existing heuristics are honestly labelled; none was silently relabelled.
- 336/336 backend tests in a clean run; frontend fully clean.

## 21. UNKNOWN

- Whether the intermittent Application Control DLL blocking is a
  configuration issue fixable by an allow-list, or a broader policy that
  will also affect a from-scratch venv rebuild — not diagnosable without
  IT/security involvement.
- PostgreSQL behaviour (still no server available).
- What a real recovery-probability model's performance would look like —
  genuinely unknown until real data exists; no simulation of this was
  attempted.
- Whether the existing shadow XGBoost model (trained on payment-failure risk,
  not recovery) is doing more harm than good by nudging `recovery_probability`
  away from the baseline heuristic in datasets where it happens to be active —
  not measured this phase (would itself require the same missing outcome
  labels to evaluate).

## 22. BLOCKERS

1. **Task A (recovery probability) is BLOCKED BY DATA** — 12 real labels
   system-wide, need on the order of hundreds per domain minimum for any
   credible train/validation/test split with calibration.
2. **Task B (ERV)** is blocked transitively through Task A.
3. **Task D (next-best-action)** is BLOCKED BY DATA — 11 action-outcome pairs.
4. **Task C (root-cause classification)** is not a data problem but a label
   problem — the target is circular with the existing deterministic rule;
   not recommended even once more data exists, unless a genuine independent
   ground-truth source (e.g., human-confirmed root cause) is introduced.
5. **LLM fine-tuning is BLOCKED BY DATA** — no corpus exists.
6. **The environment's ML toolchain (scipy/scikit-learn/xgboost) is
   unreliable** due to an external Windows security policy, independent of
   data availability — any future training work should not proceed until this
   is resolved or a dedicated environment is provisioned.

None of these blockers required inventing an assumption to route around; each
is reported here per the brief's explicit STOP-and-report instruction.

## 23. GO/NO-GO for Phase 8

**NO-GO for building any new AI/ML predictive model in Phase 8, on the
current data.** The system is not blocked architecturally — the advisory
integration point, safety contract, and model contract are all ready (§12,
§13) — it is blocked by data volume and by an environment issue outside this
repository's control. Recommended before any further AI/ML modeling work:

1. Let the Phase 5 authoritative-outcome mechanism run in real (or
   realistic, sustained) use long enough to accumulate genuinely observed
   recovery labels — hundreds per domain, not dozens system-wide.
2. Resolve or work around the Application Control DLL blocking (§16) so
   training and serving are reliable.
3. Revisit the shadow XGBoost model's target definition (§2 of the audit doc)
   before trusting its output for anything beyond the current shadow/advisory
   role it already correctly stays in.

Non-modeling Phase 7 deliverables — the audit, the leakage-safe design, the
model contract, the hardened fallback path, the Phase 6 carry-forward fix —
are complete and real. Per the explicit instruction, **no Phase 8 work has
been started.** This report is the end of Phase 7.
