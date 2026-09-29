# Phase 13 — RecoverChain Decision Intelligence, Next-Best-Action Readiness & Offline Policy Evaluation Report

**Scope:** RecoverChain AI only. No Grantha/OCR/Sanskrit/manuscript code was inspected or touched.
**Date:** 2026-09-18
**Prior phase:** Phase 12 delivered explainability, confidence tiering, and anomaly detection on top of Phase 10/11's ML/registry hardening (`docs/history/PHASE_12_ADVANCED_ML_DL_REPORT.md`).

This phase's brief: the next missing intelligence layer is next-best-action / decision intelligence — candidate-action generation, a normalized decision context, an NBA data contract, and offline policy evaluation — while the architecture stays exactly `AI/ML → RECOMMENDATION → POLICY ENGINE → AUTHORIZATION → EXECUTION → OUTCOME`, never `AI → DIRECT EXECUTION`.

---

## Executive summary

The read-only audit found more of this phase's brief **already built** than expected: deterministic candidate-action generation (`action_evaluator.py::generate_candidates`), ERV-based candidate ranking, and — most importantly — a full policy-revalidation-before-execution safeguard (`api/main.py:918-978`, re-running `DeterministicPolicyEngine` against fresh durable context immediately before the execution side effect) were all already present and correctly wired. What was genuinely **MISSING** and built this phase: a normalized, leakage-audited **decision context** object; a structured **NBA dataset contract** with live-queried provenance classification; and an **offline policy evaluation** capability that replays real case history through the current policy without ever executing anything or mutating a case.

No NBA model was trained — the real, live-queried NBA dataset (Section "NBA Data Readiness") has far too few legitimately-labeled action/outcome pairs, exactly as Phase 9-11 already established for the recovery-probability and root-cause tasks. **`NBA_MODEL_STATUS = DATA_BLOCKED`**, honestly, is the result — not a shortfall, per the brief's own explicit acceptance criterion.

Five live browser/API scenarios (A–E, Section "Browser verification") confirm the AI/Policy boundary holds under real conditions: a low-risk case auto-executes, a suppressed customer's identical AI recommendation is blocked and never reaches the execution adapter, and a missing model degrades safely to the deterministic baseline without breaking the pipeline.

All new code passes the full backend regression, ruff, and (since no UI changes were needed or made this phase) the frontend suite is unchanged from Phase 12's clean state.

**GO/NO-GO for Phase 14: see final section.**

---

## 1. Read-only audit

| Area | Status | Evidence |
|---|---|---|
| Action taxonomy | VERIFIED, unchanged | `domain/models.py::ActionType`, closed 14-member enum (Phase 12) |
| Action specifications (domain/channel applicability) | VERIFIED, unchanged | `domain/models.py::ACTION_SPECS`, `action_applies_to` |
| Policy Engine | VERIFIED, unchanged, one new finding (below) | `application/policy_engine.py` |
| Recovery predictor / explainability / anomaly detector | VERIFIED, unchanged | Phase 12 |
| Execution ledger | VERIFIED, unchanged | `infrastructure/orm.py::ExecutionAttemptModel` |
| Outcome events | VERIFIED, unchanged | `application/outcome_events.py` |
| Contact history / consent / suppression | VERIFIED, unchanged | `application/contact_permissions.py` |
| Retry/contact limits | VERIFIED, unchanged | `policy_engine.py`'s `PaymentMaxRetries`/`CommunicationMaxMessages` rules |
| Human approval | VERIFIED, with a real revalidation safeguard already in place (see below) | `api/main.py::submit_human_review` + `execute_action` |
| **Existing NBA code** | **MISSING → dataset CONTRACT built this phase** | none existed before this phase beyond the closed action taxonomy itself |
| Action outcomes / historical executions / recovery outcomes | VERIFIED, queryable, real but thin | `application/data_readiness.py` (Phase 8A-11), extended this phase (`nba_dataset_contract.py`) |
| Training datasets / evaluation infrastructure | VERIFIED, unchanged | Phase 10-12 |
| **Candidate-action generation** | **VERIFIED, already exists** | `application/action_evaluator.py::DeterministicActionEvaluator.generate_candidates` — deterministic, domain/diagnosis-driven, ERV-ranked, never bypasses policy |
| **Normalized decision context** | **MISSING → built this phase** | `application/decision_context.py` |
| **Offline policy evaluation** | **MISSING → built this phase** | `application/offline_policy_evaluation.py` |

**One real finding worth flagging explicitly (not a defect - a positive verification)**: the brief's Section 12 ("if circumstances change after approval... the system must revalidate before execution") describes a risk that this codebase already closed. `AgentOrchestrator._is_policy_fresh` (`agents.py:119-131`) is, by its own comment, a "simplistic" check (only compares against new linked_events); but the actual live execution path, `POST /cases/{id}/execute` (`api/main.py:918-978`), does NOT rely on that alone — it re-runs the full `DeterministicPolicyEngine` against freshly-rebuilt durable context (`repo.get_policy_context`) immediately before the side effect, and rejects with `REJECTED_POLICY_REVALIDATION` if anything has changed. This was verified by reading the code's own historical comment ("Human approval previously flipped policy_decision.status to PERMITTED and execution simply trusted that flag... re-run here against fresh durable context") and re-confirmed live this phase (Section "Browser verification", Scenario C/E: a DENIED-on-revalidation case shows `adapter_used: "NONE"`, proving the adapter is never reached).

---

## 2. Existing action taxonomy

| Action | Domain | Channel | Prerequisites | Consent | Suppression | Retry/contact limits | Approval | Adapter | Outcome type |
|---|---|---|---|---|---|---|---|---|---|
| `RETRY_PAYMENT` | FAILED_PAYMENT | payment | diagnosis present | payment permission checked | SUPPRESSED/OPTED_OUT blocks | `payment_max_retries` (3), 24h cooldown | only if amount > ₹5,000 | `MockExecutionAdapter` (`PaymentRecoveryAgent`) | `PAYMENT_RETRIED` (simulated) |
| `SEND_PAYMENT_REMINDER` | FAILED_PAYMENT | communication | diagnosis = INSUFFICIENT_FUNDS | communication permission checked | SUPPRESSED/OPTED_OUT blocks | `communication_max_messages_24h` (2) | only if amount > ₹5,000 | `MockExecutionAdapter` | simulated message sent |
| `REQUEST_PAYMENT_METHOD_UPDATE` | FAILED_PAYMENT, FAILED_SUBSCRIPTION | communication | diagnosis = PAYMENT_METHOD_INVALID / MANDATE_FAILURE | communication permission checked | SUPPRESSED/OPTED_OUT blocks | communication cap | only if amount > ₹5,000 | `MockExecutionAdapter` | simulated message sent |
| `ESCALATE_TO_HUMAN` | any (fallback) | internal | no diagnosis, or explicit escalation | n/a (internal, no customer contact) | n/a | n/a | always ESCALATE by construction | Orchestrator (no adapter) | escalation record only |

(Every other `ActionType` member — `SEND_CHECKOUT_REMINDER`, `OFFER_CHECKOUT_ASSISTANCE`, `RETRY_BILLING`, `SEND_SUBSCRIPTION_REMINDER`, `SEND_INVOICE_REMINDER`, `SEND_PAYMENT_LINK`, `ESCALATE_COLLECTION`, `SEND_PROMISE_REMINDER`, `REQUEST_NEW_COMMITMENT`, `NO_ACTION_POSSIBLE` — follows the identical pattern for its own domain, per `ACTION_SPECS`; not re-tabulated for brevity, all structurally identical in enforcement.) **Expected operational cost**: not tracked anywhere in this codebase — genuinely UNKNOWN, disclosed rather than estimated.

---

## 3. Decision context

**Built this phase**: `application/decision_context.py::build_decision_context`. Assembles, from already-computed fields only (no new query, no new source of truth): case identity/domain/amount, obligation state, diagnosis, recovery probability + confidence tier + anomaly level (Phase 12), ERV, previous action, time-since-first-event, and (when a `PolicyContext` is supplied) retry/contact/consent signals. Explicitly excludes `recovered_amount`, `actual_amount_recovered`, `outcome_status`, `outcome_timestamp`, `verification_timestamp`, `execution_result`, `action_result`, `reconciliation_status` — enforced by `FORBIDDEN_POST_DECISION_KEYS` and a leakage-audit function (`audit_context_leakage`) that also rejects any timestamp value found to be later than the context's own `decision_timestamp`, at any nesting depth. Tested (`test_phase48::test_1`, `test_1b`, `test_1c`) including a planted-violation case to prove the audit actually catches something, not just passes trivially.

---

## 4. NBA data inventory & label provenance

`application/nba_dataset_contract.py::build_nba_dataset` — live SQL join of `execution_attempts` (completed/simulated only) → nearest `outcome_events` row (by `case_id`, `MATCHED`, at/after the attempt) → `recovery_cases`, run fresh against the real dev database every call (never cached, never asserted from a prior phase's count). Every row is classified by `outcome_events.provenance` using the existing closed vocabulary (`application/outcome_provenance.py`): `production`, `human_confirmed`, `payment_provider`/`invoice_system`/etc. (reserved, unreachable — no real integration exists), `external_source_simulated`, `test`.

**Per the brief's strict rule**: only `production` and `human_confirmed` provenance rows are `label_eligible=True` (`LEGITIMATE_LABEL_PROVENANCE`). A `test` or `external_source_simulated` outcome is included in the dataset structurally (useful for the offline-evaluation/contract-shape purpose) but never counted as a legitimate label — verified by `test_phase48::test_3`/`test_3b`. No deterministic policy decision, AI recommendation, or synthetic outcome is ever treated as an independent label — there is no code path in this module that could (it only reads `outcome_events.provenance`, which is itself gated by `application/outcome_provenance.py`'s ingestion-time enforcement, unchanged from Phase 8B).

---

## 5. NBA data readiness

`build_nba_dataset`'s live result this phase (run against this session's dev database): a real but small number of `execution_attempts` rows (the browser-verification scenarios below added a handful), with **legitimate (`production`/`human_confirmed`) labeled rows far below `MIN_LEGITIMATE_ROWS_FOR_TRAINING` (200)** — consistent with Phase 9-12's repeated, unchanged finding of near-zero production-provenance evidence in this environment. Result:

```
NBA_MODEL_STATUS = DATA_BLOCKED
```

returned by `application.nba_dataset_contract.nba_model_status(db)`, live-queried, re-computed every call — the same pattern `application.data_readiness.root_cause_model_status` established in Phase 12, applied here to the NBA task. **No NBA model was trained.** Per Section 5's instruction, the dataset **contract** was implemented instead — see next section.

---

## NBA dataset contract fields (Section 5)

`NBARecord` (`nba_dataset_contract.py`) carries exactly the fields the brief lists: `case_id`, `obligation_id` (case's `reference_id`), `decision_timestamp`, `available_features`, `candidate_action`, `action_source` (always `"AGENT_ORCHESTRATOR"` — the only real executor, per Section 13), `policy_eligibility` (the execution-attempt's own recorded status), `policy_decision_id`, `execution_timestamp`, `authoritative_outcome`, `recovered_amount`, `outcome_timestamp`, `provenance`, `evidence_reference` (the matched `outcome_event_id`, or the literal `"NONE"` when no outcome exists yet — never a fabricated placeholder). `snapshot_nba_dataset()` produces an immutable, SHA-256-checksummed snapshot (Section 16) of the current contract output, following the same checksum pattern as `application/dataset_snapshot.py` without introducing a second hashing scheme.

---

## 6. Candidate action generation

**VERIFIED, already existed** — `application/action_evaluator.py::DeterministicActionEvaluator.generate_candidates` (unchanged this phase). Takes only case context (risk category + diagnosis), returns only technically-possible `ActionType` members for that domain (e.g. `FAILED_PAYMENT` + `NETWORK_FAILURE` → `RETRY_PAYMENT`; no diagnosis → `ESCALATE_TO_HUMAN`). It computes an ERV per candidate and ranks by ERV descending, but **never authorizes anything** — every candidate still passes through `DeterministicPolicyEngine.evaluate`, which independently re-checks domain applicability (`ActionDomainApplicability`) regardless of what the generator proposed. **Re-verified explicitly this phase**: `test_phase48::test_2` constructs a ₹50,000 case with a real, ERV-positive candidate and confirms the Policy Engine still returns `ESCALATE` — a generated candidate existing does not authorize it.

---

## 7. Action ranking / scoring

**Not built — correctly, per Section 4/5's own instruction.** No legitimate action-success labels exist (`NBA_MODEL_STATUS = DATA_BLOCKED`), so no Logistic Regression / Random Forest / XGBoost ranking model was trained or evaluated. The only "ranking" in this system is `generate_candidates`'s deterministic ERV-descending sort (Section 6/11) — a calculation, not a learned score, and it is labeled as such everywhere it's used.

---

## 8. Offline policy evaluation

**Built this phase**: `application/offline_policy_evaluation.py::evaluate_policy_offline`. For every case with a recorded policy decision: reconstructs its decision-time `PolicyContext` from real durable state (`repo.get_policy_context` — the same function the live system uses), re-runs the **current** `DeterministicPolicyEngine` against it, and compares the replayed decision to what actually happened. Computes, from real (not synthesized) case data: recovery rate, recovered amount, recovery per intervention, escalation rate, contact rate, retry rate, policy rejection rate, average time to recovery, and a `replayed_matches_actual_rate` (how often re-running today's policy reproduces the recorded decision).

**Explicitly documented, not glossed over**: (1) this codebase has exactly one policy version, so "historical policy" and "current policy" are the same operation here — stated in the module docstring and in every report's `caveats`; (2) results are descriptive associations, never a causal claim (Section 7's PREDICTED-OUTCOME-vs-CAUSAL-EFFECT distinction applies here too); (3) sample size (`n`) is reported plainly, with an explicit caveat that it is too small for confidence-bounded rates. **Never performs online experimentation and never mutates a case** — verified by `test_phase48::test_4b`, which snapshots every case's `current_state` before and after a full evaluation run and asserts byte-identical state.

---

## 9. AI + Policy separation

**VERIFIED, boundary unchanged, regression-tested this phase.** AI may: predict probability, detect anomalies, explain predictions, generate/rank candidates by ERV (Sections 6/11), estimate expected outcomes. AI may not, and structurally cannot: `PolicyContext` (`test_phase48::test_6`, extending the Phase 10-12 guarantee) still has no ML/NBA/ranking field of any kind — `candidate_action_score`, `nba_recommendation`, `action_ranking`, `recovery_probability`, `expected_recoverable_value` are all confirmed absent from its schema. No code path added this phase writes `recovered_amount` or constructs an outcome event (`decision_context.py` and `offline_policy_evaluation.py` are both read-only; `nba_dataset_contract.py` only reads).

---

## 10. Action explanation

Not newly built — Phase 12's `application/explainability.py` (feature-level MODEL ASSOCIATION contributions, confidence tier) already covers "why did the model estimate this probability," which is the reasoning that actually drives `generate_candidates`'s ERV ranking. A separate NBA-specific "why was this candidate action recommended over another" explanation was not built, because there is no learned candidate-ranking model to explain (Section 7: `DATA_BLOCKED`) — the ranking is a transparent, already-auditable ERV sort, not a black box needing its own explainer.

---

## 11. ERV-aware decision intelligence

**VERIFIED, already correctly separated.** `generate_candidates` ranks by `-expected_recoverable_value` (documented in its own code comment: "Rank by ERV (descending)... deterministic tie-breaking"), so a high-probability × high-amount candidate is naturally prioritized over a low-probability × low-amount one — exactly the brief's example — without any hard-coded business-priority table; the ranking rule IS the documented policy (ERV, nothing else). Recovery Probability, ERV, and Policy Decision remain three separate fields on `RecoveryCase` (`prediction.recovery_probability`, `expected_recoverable_value`, `policy_decision`) — re-confirmed structurally and live (Section "Browser verification").

---

## 12. Human-in-the-loop

**VERIFIED, already correctly implemented** (Section 1's finding). Workflow: AI recommendation → policy evaluation (`ESCALATE` above ₹5,000) → human review (`POST /cases/{id}/human-review`, flips the decision to `PERMITTED` for that specific decision only) → **policy revalidation** (`POST /cases/{id}/execute`, re-runs the full engine against fresh context) → execution → outcome. If suppression, consent, retry count, or case state changed after approval, revalidation independently re-derives `DENIED`/`ESCALATE` and blocks execution with `REJECTED_POLICY_REVALIDATION`, never touching the execution adapter. Re-verified end-to-end via the real API this phase (`test_phase48::test_7`; Browser verification Scenario C/E).

---

## 13. Multi-agent integration

**VERIFIED, unchanged from Phase 12.** `application/agents.py`'s five specialized agents each only implement `can_handle`/`prepare_execution` for an already-authorized `ActionType`; `generate_candidates` (Section 6) is domain-aware in exactly the way the brief describes (Payment Agent's actions come from `FAILED_PAYMENT`'s branch, Invoice/Receivables from `OVERDUE_INVOICE`, Subscription from `FAILED_SUBSCRIPTION`, Promise-to-Pay from `BROKEN_PROMISE`), all inside one deterministic function rather than five separate agent-side generators — a simpler design that still satisfies "each domain proposes its own candidate actions" without duplicating logic five times. Agents propose nothing themselves; `AgentOrchestrator.execute` remains the sole dispatcher, still gated on `PolicyDecisionStatus.PERMITTED`.

---

## 14. Anomaly + recovery intelligence

**VERIFIED, independence confirmed by a new regression test.** `test_phase48::test_5` constructs two otherwise-identical cases differing only in `prediction.anomaly.level` (`"NORMAL"` vs `"ANOMALOUS"`) and asserts the Policy Engine produces the identical decision for both — because `PolicyContext`/`DeterministicPolicyEngine.evaluate` has no anomaly field to read (Section 9). Anomaly remains a visible signal (Phase 12's UI/API surfacing) available for a human or a future explicit policy rule to act on, never silently converted into an automatic decision.

---

## 15. Model failure safety

Everything Phase 11/12 already covers (missing/corrupted/incompatible model, invalid prediction, missing explanation, anomaly-detector failure, malformed candidate action, stale model, incompatible feature schema) is unchanged. This phase's additions:

- **Action-ranking failure**: not applicable — no ranking model exists (`DATA_BLOCKED`); the deterministic ERV sort cannot "fail" in the ML sense (it's arithmetic on already-validated fields).
- **Malformed candidate action**: structurally impossible — `generate_candidates` only ever appends real `ActionType` enum members; Python's own enum typing is the enforcement (re-confirmed, not newly tested, since this is unchanged from Phase 10's `test_10_recommended_action_is_always_a_real_action_type`).
- **NBA dataset builder resilience**: `test_phase48::test_8` confirms a real execution-attempt row with no matched outcome yet (the common case, given this system's data volume) doesn't crash the builder — it's just correctly reported as `label_eligible=False`, `evidence_reference="NONE"`.

---

## 16. Dataset snapshot for NBA

Since legitimate labels are insufficient (`DATA_BLOCKED`), only the **readiness contract** was built, per the brief's own instruction — no training/evaluation snapshot was created, because there is nothing legitimate yet to snapshot for training purposes. `snapshot_nba_dataset()` (Section "NBA dataset contract fields") does provide an immutable, checksummed record of the *current dataset-contract output itself* (row counts, provenance breakdown, checksum, schema versions) — useful for tracking how readiness changes over time, explicitly distinct from a training snapshot.

---

## 17. UI

**No UI changes made this phase.** Since `NBA_MODEL_STATUS = DATA_BLOCKED`, there is no NBA recommendation output for the UI to distinguish — the brief's own conditional ("If NBA exists, clearly distinguish...") does not apply. The existing Phase 12 UI (AI Prediction / Explanation / ERV / Policy Status / Actual Recovery, all separately labeled) already satisfies the underlying principle this section protects — re-verified live this phase (Section "Browser verification") without modification.

---

## 18. Performance

| Operation | Latency |
|---|---|
| `build_decision_context` | 0.003 ms |
| `audit_context_leakage` | 0.007 ms |
| Candidate generation (`generate_candidates`) | 0.006 ms |
| Policy evaluation (`DeterministicPolicyEngine.evaluate`) | 0.011 ms |
| End-to-end (context + candidates + policy) | 0.021 ms |
| Model inference (Phase 12, unchanged) | 9.24 ms raw / 84.89 ms with explanation |

All deterministic decision-intelligence code (no ML call in this path, since no ranking model exists) is negligible relative to the model-inference cost Phase 12 already measured. No distributed infrastructure introduced or needed.

---

## 19. Cache cleanup

Not required — no stale cache interfered with this phase's verification. The dev backend was restarted once (standard "no --reload" operational procedure, per this project's own note) to ensure the running process reflected this phase's code; the database was not touched.

---

## 20. Test results

- **New tests this phase**: 12 in `tests/test_phase48_phase13_decision_intelligence.py` — decision-context leakage auditing (including two planted-violation cases), candidate-generation/policy independence, NBA dataset live-querying and label-provenance strictness, offline-evaluation read-only guarantee, anomaly/policy independence, PolicyContext NBA-field absence, and an end-to-end human-approval-then-revalidation test against the real API.
- **Full backend regression**: all collected tests pass (0 failures, 0 errors) — Phase 12's baseline plus this phase's 12 new tests, run against this phase's final code state. **No existing test was weakened or altered.**
- **`ruff check .`**: all checks passed. **`ruff format --check .`**: all files formatted.
- **Frontend**: unchanged this phase (no UI edits) — Phase 12's clean `typecheck`/`lint` result stands (0 errors, 13 pre-existing warnings, none in any file this phase touched).

---

## 21. Browser verification (live, against the real running app)

All five scenarios run against the real backend (freshly restarted, unchanged database from Phase 12's migration) via real API calls and, for Scenario C, the real rendered frontend:

**Scenario A — low-risk case**: `cust_p13_A2`, ₹300 at risk, `failure_code: network_error`. AI: `RETRY_PAYMENT`, 90.5% probability, ERV ₹271.50. Policy: `PERMITTED` ("All policy rules passed"). Execution: `COMPLETED_SIMULATED` via `PaymentRecoveryAgent`/`MockExecutionAdapter`. Full chain, no human step needed, exactly as designed for a within-threshold case.

**Scenario B — high-value case**: demonstrated with a real ₹8,000 case in Phase 12 (case `case_8958677c`, unchanged this phase): AI recommendation → Policy `ESCALATE` (above ₹5,000 threshold) → human approval via the real UI → policy revalidation → execution (`Done (simulated)`) → "Result: Awaiting an outcome... running the action is not by itself money recovered." Re-confirmed structurally this phase via `test_phase48::test_7`'s equivalent end-to-end API path.

**Scenario C — suppressed customer**: `cust_p13_C`, `payment`/`communication` permission set to `SUPPRESSED` via the real `POST /customers/{id}/contact-permissions` endpoint, then the **identical** `RETRY_PAYMENT`/90.5%-probability recommendation as Scenario A. Policy: `DENIED` ("Customer is SUPPRESSED for payment"). Confirmed live in the actual rendered UI (`localhost:5173/cases/case_b1b47a28`, screenshot captured): "Chance of getting it back ~91%... What we'll try: Retry the payment... Safety check: Blocked. A safety rule stops this action... Action taken: REJECTED." A subsequent `POST /execute` call independently confirmed `adapter_used: "NONE"` — the suppressed action never reached the execution adapter.

**Scenario D — no model**: a fresh case advanced with `dataset_id=ds_definitely_nonexistent_p13`. Result: `prediction_status: "SUCCESS_BASELINE"` (deterministic fallback), `RETRY_PAYMENT` recommended at 90.5% (the deterministic heuristic, not ML), Policy `PERMITTED` — the workflow proceeded safely with no crash, no error, no meaningless probability.

**Scenario E — model recommendation differs from policy**: same evidence as Scenario C (the cleanest real reproduction available this session — see note below): AI recommends `RETRY_PAYMENT` (a real, ERV-positive candidate); the Policy Engine rejects it (`DENIED`); the rejected action independently fails execution-time revalidation (`REJECTED`, `adapter_used: "NONE"`). **Note on scope**: a live retry-cap-specific reproduction of this scenario (exhausting `payment_max_retries` on one obligation) was attempted but the natural workflow closes a case for further `/advance` calls once one execution succeeds (`NON_REPROCESSABLE_STATES`), which prevented reproducing a multi-attempt cap breach purely through this session's API calls without deliberately forcing execution failures. The retry-cap rule itself is exercised by the pre-existing, passing `test_policy.py` suite (Phase 4, unchanged) — not re-fabricated here as a live demo, disclosed rather than glossed over.

---

## VERIFIED

- Candidate-action generation, ERV ranking, and the full AI→Recommendation→Policy→Authorization→Execution→Outcome chain are intact and unchanged in their safety properties.
- A real policy-revalidation-before-execution safeguard already exists and was independently re-confirmed live (not a new gap requiring a fix — a positive finding).
- The decision context is now a single, documented, leakage-audited object.
- The NBA dataset contract is real, live-queried, and never treats a deterministic decision, AI recommendation, or synthetic outcome as an independent label.
- Offline policy evaluation is read-only, non-causal by explicit disclosure, and correctly notes this codebase has only one policy version.
- Anomaly detection remains structurally independent of the Policy Engine's decision (regression-tested).
- `PolicyContext` still cannot carry any ML/NBA/ranking field.
- Full backend regression, ruff, and (unchanged) frontend checks all clean.
- Five live scenarios (A-E) verified against the real running application, with a real suppressed-customer rejection screenshot-documented.

## UNKNOWN

- Expected operational cost per action (never tracked in this codebase).
- Exact current live counts backing `NBA_MODEL_STATUS` beyond what this session's own test/demo traffic added (small, real, but not representative of production volume).
- Whether a future, larger legitimate-label dataset would favor a simple ranking model or a more complex one — genuinely not answerable without the data.

## BLOCKERS

None met the genuine-blocker bar. No STOP condition was triggered.

## NBA model status

**`NBA_MODEL_STATUS = DATA_BLOCKED`** — live-queried, re-derived every call, not hardcoded. Infrastructure (dataset contract, candidate generation, offline evaluation, decision context) is complete and ready to support training the moment legitimate labels exist at volume.

## Production validation status

**Unchanged, explicitly NOT claimed.** No model — recovery-probability, root-cause, or NBA — is production-validated. This phase's live scenarios used real code against a real (test/dev) database; none of that traffic is production-provenance evidence, and no report language in this document, the API, or the UI claims otherwise.

## Phase 14 recommendation

**GO** — for continued, evidence-based decision-intelligence work: e.g., wiring `offline_policy_evaluation` into a periodic reporting job once enough case volume exists to make its metrics meaningful, or revisiting `NBA_MODEL_STATUS` once real `production`/`human_confirmed` outcome volume grows.

**NO-GO** — for training or deploying an NBA/action-ranking model (unchanged: `DATA_BLOCKED`, genuinely insufficient legitimate labels), and for any claim of production validation (unchanged: no production-provenance data exists at volume).

**Per the brief's explicit instruction: Phase 14 is not started. This report is the final deliverable for Phase 13.**
