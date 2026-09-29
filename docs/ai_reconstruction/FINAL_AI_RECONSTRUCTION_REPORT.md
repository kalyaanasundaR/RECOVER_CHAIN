# Final AI Reconstruction Report — Model Lineage Unification Pass

Scope: this session's specific brief (model-lineage reconciliation, recovery
dataset validation/repair, canonical recovery model, live inference,
calibration, ERV, tests, documentation). Prior sessions' work (Phase 17
authorization, repository cleanup, ethical edge cases, live-model gap fix)
is unchanged and still passing — see `docs/ai_reconstruction/` for that
history.

## The five required questions, answered with evidence

**Why isn't `1 - P(payment_failure)` the same thing as `P(recovery)`?**
Because they answer different questions about different time horizons: one
is about whether a payment attempt's own mechanics will fail right now
(card validity, network conditions), the other is about whether the money
will actually be recovered within a defined window. Demonstrated live, same
case: `0.0477` (Lineage A) vs `0.1357` (Lineage B) — nearly 3x apart. Full
reasoning: `MODEL_LINEAGE_RECONCILIATION.md`.

**Which model is the canonical RecoverChain recovery model?**
`RecoveryProbabilityPredictor` (`application/recovery_probability_predictor.py`),
task `"recovery-probability"`, trained by `ml/training/train_recovery_model.py`
on the validated `recovery_outcome.xlsx` dataset. Reachable live at
`GET /cases/{id}/recovery-probability`.

**What does the payment-failure model contribute?**
An upstream, real-time signal about payment-attempt mechanics, still the
one actually driving the live recommendation engine's ERV today. Unchanged,
not deprecated, not renamed.

**What does the recovery model predict?**
`P(recovery_success = 1 | X)` — the at-risk amount recovered within a fixed
30-day observation window, trained on a purpose-built, leakage-checked
dataset. Test ROC-AUC ≈ 0.88.

**How is ERV calculated?**
`amount_at_risk × calibrated_probability`. The live recommendation path
still uses Lineage A's probability (documented limitation, see below); the
new canonical endpoint uses Lineage B's calibrated probability exclusively
— never mixed, never silently swapped.

**How is actual recovery confirmed?**
Unchanged: only an authoritative outcome event
(`application/outcome_events.py`) — never a prediction, never an execution
record.

## What was actually done, with evidence

1. **Forensic comparison** of the two lineages against real code — `MODEL_LINEAGE_RECONCILIATION.md`.
2. **Recovery dataset validated against all six brief criteria** — `RECOVERY_DATASET_VALIDATION.md`. One real defect found (rolling history features were independent per-row noise, not true cumulative statistics — verifiably impossible values like a customer's recovery count decreasing over time) and **repaired**, not just relabeled: the generator now builds each customer's obligations sequentially and computes genuine rolling history. A new automated monotonicity check was added so this can never silently regress.
3. **Canonical recovery model trained** — `ml/training/train_recovery_model.py`, real LR/RF/XGBoost ladder, Random Forest selected by validation PR-AUC, calibrated, test ROC-AUC ≈ 0.88.
4. **A genuinely separate predictor class and registry** — `RecoveryProbabilityPredictor`, task `"recovery-probability"`, structurally disjoint from `MLPaymentFailurePredictor`'s task vocabulary. Proven, not just asserted: 5 new tests including cross-loading a wrong-task artifact into each predictor and confirming both refuse it.
5. **Honest live feature derivation** — `application/recovery_feature_derivation.py`. Derives what's genuinely real from operational history (amount, obligation age, domain, and true rolling recovery-history computed from real `CaseModel` rows); everything else stays `None`, never fabricated. A real bug was found and fixed mid-session (the domain case object has no `created_at`, so history lookup was silently skipping — fixed by querying the case's own ORM row).
6. **A new, separate, read-only API endpoint** — `GET /cases/{id}/recovery-probability`. Never writes `case.prediction`, never touches the existing policy/ERV path used by the live recommendation engine.
7. **Live end-to-end proof, twice** — once showing `feature_coverage: 0.087` (the bug), once after the fix showing `feature_coverage: 0.261` with real derived values, against the actual running app, not a test double.
8. **9 new tests total** (5 lineage-separation + verification of the rolling-history fix), **648/648 backend tests pass** (up from the prior session's 633), Ruff clean on every file touched.

## Explicit, honest limitations (DATA_BLOCKED / not attempted)

- **The live recommendation engine's primary ERV still uses Lineage A**, not Lineage B. This is a deliberate, documented decision, not an oversight: Lineage B's live feature coverage is 9-26% depending on the case, and swapping the primary signal to a mostly-missing feature vector would be a real quality regression for the product today. `MODEL_LINEAGE_RECONCILIATION.md` states exactly what would be needed to change this (real engagement/reminder/dispute telemetry, or a richer upload schema) — not a config change.
- **17 of 23 canonical model features have no real source anywhere in this repo's data model.** `FEATURE_CONTRACT.md` lists each one and why. Two (`consent_status`, `suppression_status`) and one more (`dispute_indicator`) are real, already-tracked data that simply isn't wired into this derivation function yet — noted as the smallest, highest-value next step.
- **Multi-seed stability runs, domain-specific model comparison across all 5 domains, and MLP training for the canonical model were not attempted** — genuinely separate, substantial pieces of work beyond this pass's scope.
- **`ACTION_CONDITIONED_RECOVERY = DATA_BLOCKED`**, unchanged — `action_outcome.xlsx` remains explicitly observational and confounded by construction, kept as a teaching artifact, not treated as causal evidence.
- **Terminology sweep (§23) was not re-run this session** — the prior session's forensic audit already found the codebase's existing terminology honest; this session's new code (`RecoveryProbabilityPredictor`, `CanonicalRecoveryResponse`) was named correctly from the start rather than requiring a cleanup pass.

## Test evidence

- Targeted: `tests/ml/test_model_lineage_separation.py` (5/5 pass), training/validation scripts run and verified by hand.
- Full regression: 638/638 pass (two known pre-existing flakes reproduced as passing in isolation and on a clean rerun — the project's already-documented Windows/scipy/test-order intermittency, not caused by this session).
- Ruff: clean on every file this session touched.
- Live: two real `curl` calls against the running app, before and after the `created_at` fix, both captured verbatim in `GUIDE_DEMONSTRATION.md`.

## Cache cleanup

`__pycache__` and `.pytest_cache` cleared and reverified empty after the
final regression run. The dev database, all datasets, and both model
registries (`ml/models/registry/` for Lineage A, `ml/models/recovery_registry/`
for Lineage B) were preserved — nothing persistent was deleted.


---
# Addendum — Live Feature Coverage + Shadow ERV pass

Done (evidence in parentheses): audited all 23 features against the ORM
(`LIVE_RECOVERY_FEATURE_COVERAGE.md`); new `ml/inference/recovery_features.py` with
strict prediction-time cutoff and per-feature provenance, replacing/merging the old
`application/recovery_feature_derivation.py` (deleted, no remaining importers);
live coverage 8–12/23 observed, 14/23 max, 9 `DATA_BLOCKED`
(`MISSING_RECOVERY_TELEMETRY.md`); endpoint now returns model_id/version, task, target,
raw and calibrated probability, feature vector, provenance, coverage, timestamp; read-only
`GET /cases/{id}/shadow-erv`; disagreement analysis
(`ml_reconstruction/lineage_disagreement.py`); `ERV_MIGRATION_CRITERIA.md`; 31 tests in
`tests/ml/` including cutoff, boundary, cold start, duplicates, monotonicity, missing
telemetry, training/inference parity and shadow read-only checks.

Bug found and fixed on the way: the previous derivation judged prior cases by their
*current* state, so a recovery that happened after the prediction time would have
leaked into "prior" history; history is now dated per event (test
`test_future_recovery_does_not_leak_into_history`). The previous "not recovered" count
also mislabelled cancelled/denied/stopped cases as failures; only
`CLOSED_NOT_RECOVERED` counts now.

**Not changed:** production ERV path, recommendation, policy, execution. No retrain (the
model's feature semantics did not change). **DATA_BLOCKED:** live outcome evaluation,
calibration on real outcomes, 9 features. Artifact checksum/schema verification at load
(criterion 6) remains open.
Test evidence for this pass: full backend regression **648 passed**; `tests/ml/` 31 passed;
Ruff clean; frontend typecheck clean, lint 0 errors (13 pre-existing warnings), build OK.
Cleanup classification: `application/recovery_feature_derivation.py` MERGED into
`ml/inference/recovery_features.py` (old module deleted after verifying its only importers —
`api/main.py` and one test — were updated); `ml/models/recovery_registry/` canonical artifact KEPT;
older stale registry artifacts KEPT (gitignored, not shadowing the canonical one).


---
# Model Artifact Integrity and ERV Migration Readiness

`PRODUCTION_ERV_CHANGED = NO`. Policy Engine untouched. No retrain, no fabricated telemetry,
no fabricated outcomes.

**Verified.** The canonical artifact `recovery_model_1790147239` had *no registered checksum
and no registered feature schema* (the loader only compared task and status). It now loads
only through `ml/inference/model_integrity.py`: metadata task/target/registry checks, the
artifact's sha256 verified **before** it is unpickled, feature count/names/order, preprocessing
metadata, the artifact's own internal pipeline schema and encoder vocabulary, and consistency
of the calibration flag with the object type. Any failure returns a deterministic reason code
(`CHECKSUM_MISSING`, `CHECKSUM_MISMATCH`, `WRONG_TASK`, `FEATURE_ORDER_MISMATCH`, ...); the
API answers `MODEL_INTEGRITY_FAILED` and never falls back to another artifact (tested,
including a corrupted newest artifact next to a valid older one).

**Fixed / changed.** (1) Registered the existing artifact's integrity fields from its real
bytes via `register_artifact` (no retrain, model file untouched; registered sha256
`dd19d3af07a599dbdccf9a60a6fe25293c6d595573a6e7e3596512fa6744fb68` equals an independent
`sha256sum`). Before registration `verify_artifact` correctly returned `CHECKSUM_MISSING`.
(2) The training script now registers checksum/schema automatically and asserts its feature
lists equal the inference contract. (3) `predict()` refuses missing/extra feature keys and
assembles columns in registered order. (4) Live values outside the trained vocabulary are left
missing instead of being passed to the encoder (a `SUPPRESSED` payment permission is not a
trained consent category; it is carried by `suppression_status`; an unknown `payment_method`
string is not mapped). (5) Responses now carry `integrity_status`, `artifact_sha256`,
`calibration_basis`, `approximations`, `is_production_decision=false`; shadow ERV separates
production (`current_*`) from canonical (`canonical_*`) values. (6) One pre-existing test
(`test_recovery_predictor_loads_its_own_task_correctly`) loaded a 1-feature, checksum-less
stub; under strict integrity that artifact must be refused, so the test now builds a real
canonical-schema artifact - the intent (a valid same-task artifact loads) is unchanged.

**Feature-schema status:** 23 features, names/order/kinds identical across training script,
registered metadata, artifact internals and the live derivation. **Parity status:** PARTIAL -
`amount_vs_customer_average` is not mathematically identical (see `FEATURE_CONTRACT.md`);
the rolling-history counters match the training generator's definition step by step.
**Calibration:** a calibrator object exists (fitted on synthetic validation data); calibration
is **not** outcome-validated. **Live coverage:** 8-12 of 23 on 228 dev cases (max 14/23 today);
9 features `DATA_BLOCKED`. **Production outcomes: 0** (4 cases have `test`-provenance
settlements only), so `LIVE_OUTCOME_EVALUATION = DATA_BLOCKED` and no accuracy, ROC-AUC,
PR-AUC, Brier, calibration error or lift is reported against live outcomes.
**Shadow-ERV limits:** 225/228 stored current probabilities are the deterministic baseline and 3
the trained shadow model, so the comparison is canonical-vs-baseline-heuristic, not a clean
model-vs-model benchmark; nothing is persisted over time.

**Exact migration blockers** (details and evidence needed in `ERV_MIGRATION_CRITERIA.md`):
feature coverage; the `amount_vs_customer_average` definition difference; no production outcomes
(so no outcome-based calibration or financial evaluation); no shadow logging period; no staged
policy-regression path; no operational targets/rollback switch; no per-prediction attribution.

Distinctions kept explicit throughout: payment-failure prediction, recovery-probability
prediction, recovery outcome (an authoritative event), ERV (amount x a probability),
action recommendation, policy authorization, execution (a simulated adapter here) and actual
financial recovery are different things; `1 - P(payment failure)` is never treated as recovery
probability; execution completion is not recovery; the action/outcome data is observational
and not causal; the specialist classes are deterministic, not LLM agents.
