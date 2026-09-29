# Phase 16 — End-to-End Traceability, Authenticated Observability & Hardening

**Date:** 2026-09-19. **Scope:** RecoverChain only. No LLM added, no NBA training, no production data fabricated, UI changed minimally.

## Executive summary
One `trace_id` now follows a recovery journey through agent, policy, approval, execution and outcome, and a reviewer can reconstruct the whole chain — including what did *not* happen — from that id alone. Observability endpoints (trace, audit, execution history, model metadata, datasets, benchmark, monitoring) now require an operator credential using the **existing** `X-API-Key` mechanism. A trace-consistency validator reports (never repairs) history problems. The Phase 15 test-order dependency was root-caused (a test fixture dropping the shared schema) and fixed. Full backend regression, Ruff and frontend typecheck/lint are clean; 62 new tests.

## 1. Read-only audit (before changes)
| Item | Verdict |
|---|---|
| Agent audit record with random per-run `trace_id` | VERIFIED — but regenerated on every `/advance`, so **not** a journey id |
| Policy decision / execution record / ledger row trace fields | **MISSING** (correlated only by `decision_id` / `execution_id`) |
| Audit recorder | VERIFIED — single choke point (`SqlAlchemyAuditRecorder.log_transition`) |
| Verification | VERIFIED as a **read** (`VerificationEngine.current_position`), no audit record of its own |
| Auth | PARTIALLY — `X-API-Key` guarded actions; all `/system/*` and `/cases/{id}/audit` reads were anonymous; no roles |
| Trace endpoint | PARTIALLY — audit-only, flat step list, no NOT_OCCURRED representation, unauthenticated |
| Redaction | **MISSING** |
| Test DB | VERIFIED isolated (`_pytest_recoverchain.db`, temp storage); evaluation uses an in-memory store and refuses the operational DB |
| Test-order dependency | root cause found (§Database/Test Isolation) |

## Existing trace architecture / end-to-end trace
**Design:** `trace_id = "trace_" + case_id-suffix` (`application/trace_ids.py`). One journey = one case. It is a *pure function* of the case id, so it is immutable by construction: no stored value can drift, retries/re-advances/approval/verification/page reloads cannot regenerate it, and every stage derives the same value independently. Per-operation ids are separate: agent runs carry `operation_id` (+ `parent_trace_id`); executions carry `operation_id == execution_id`.
**Trade-off:** uniqueness equals case-id uniqueness; pre-Phase-16 records carry random or no ids and are reported as legacy/missing, not rewritten.

**Propagation (identifiers only — no verdict, amount or state changes):**
- Audit: every case-scoped audit record gets `trace_id` in the recorder (all stages, including outcome reconciliation).
- Policy: `PolicyDecision` gains `trace_id, case_id, action, reason_codes, approval_required, lifecycle_state` (stamped by a thin wrapper around the unchanged rule engine).
- Execution: `ExecutionRecord` gains `trace_id, case_id, operation_id`; the execution ledger gains a `trace_id` column (migration `e5f6a7b8c9d0`, additive/nullable, downgrade round-trip verified); execution audit evidence adds `execution_id`/`policy_decision_id`.
- API: `ExecutionRecordResponse` adds `trace_id`, `operation_id`.

## Agent / ML-DL / policy / approval / execution / outcome traceability
`GET /system/cases/{id}/agent-trace` and `GET /system/traces/{trace_id}` return a `stages` tree (`case, agent, ml_prediction, recommendation, policy, approval, execution, verification, outcome`), a `journey` summary and the chronological `steps`.
- **Agent:** id/version, reason codes, policy prerequisites, agrees-with-deterministic, run count and operation ids. Labelled AGENT REASONING; authorizes nothing.
- **ML/DL:** model version, prediction id, probability, ERV, confidence tier, anomaly level and an *explanation reference* (prediction id + top factor directions) — no payload duplication. Labelled MODEL ASSOCIATION.
- **Policy:** decision id, policy version, decision, reason codes, approval requirement, lifecycle state, plus every evaluation (so an ESCALATE later flipped to PERMITTED by review stays visible).
- **Approval:** `NOT_REQUIRED` / `PENDING_APPROVAL` / `OCCURRED`, bound to the decision it reviewed; approver identity reported as `UNKNOWN` (no user-identity model exists).
- **Execution:** every attempt (audit + ledger merged), status, adapter, policy decision id, trace id, `blocked_by`. Revalidation is reported `BLOCKED` when it blocked; otherwise "ENFORCED_IN_POST_EXECUTE (not recorded per successful attempt)" — I did not claim a record that does not exist.
- **Verification:** a read of the authoritative position (`NOT_OCCURRED` if none).
- **Outcome:** matched authoritative events with provenance, evidence reference and applied amount; statuses `RECOVERY_CONFIRMED | OBLIGATION_CANCELLED | OUTCOME_REPORTED_NO_RECOVERY_APPLIED | PENDING | NOT_OCCURRED`; unmatched candidates (same customer/reference) are listed with `applied_to_this_trace: false`.
- **Journey label** never infers recovery from execution: `EXECUTED — RECOVERY NOT YET CONFIRMED` until a settlement outcome is applied.

## Trace consistency
`application/trace_validation.py` (`/system/traces/{id}/validation`, `/system/trace-consistency`) reports: missing / conflicting / malformed trace ids, legacy Phase-15 ids (INFO), orphaned policy decisions, orphaned executions, **execution without policy authorization**, outcome-without-correlation (domain mismatch, no provenance, recovered money with no matched outcome, recovered ≠ applied settlements), and stage-order violations. It returns `repaired: false` and tests assert planted rows are untouched afterwards.
Run over the dev database (39 cases, mostly pre-Phase-16): **0 errors, 304 warnings** — all missing-trace-id warnings on legacy records, honestly reported and not back-filled.

## Authentication & authorization
Reused `X-API-Key` (`api/access.py`); no second mechanism. Roles: `operator` (holder of `OPERATOR_API_KEY`) and `user` (holder of `API_KEY`); when `OPERATOR_API_KEY` is unset (dev default) API-key holders are operators and startup logs a warning. **READ vs ACTION:** action endpoints keep `verify_api_key`; `require_operator` guards read-only observability. 401 = missing/unknown credential, 403 = valid but not operator; anonymous callers get 401 even for nonexistent resources (no existence leak). Guarded: agent-trace, traces, trace validation, trace-consistency, model-monitoring, `/system/models`, agent-benchmark, `/system/datasets` (+detail), `/system/executions`, `/system/audit`, `/cases/{id}/audit`. A test asserts every operator-guarded route is GET/HEAD only.
**Not protected (unchanged, disclosed):** `/cases`, `/cases/{id}`, `/datasets` (dataset router), `/system/cases*`, `/system/events`, `/system/policy`, `/system/data-readiness`, `/system/eligibility` — existing tests define these reads as open; changing them is a wider product decision. **Limitation:** there is no user model; "authenticated" means "holds a configured key", and the frontend ships its key in the bundle (`VITE_API_KEY`), so that key is not confidential.

## Data redaction
`redact()` recursively replaces secret-named keys (api key, token, password, authorization, card/account numbers…) and scrubs card-number shapes, bearer tokens, API-key shapes and emails from strings; email-shaped customer ids are masked. Raw `raw_payload` and adapter raw responses are not included in traces. Diagnostics (ids, decisions, amounts, reason codes) are preserved. Tests plant secrets/PANs/emails in audit evidence and assert none appear in the response.

## Database / test isolation
**Root cause of the Phase 15 finding:** `tests/test_evaluation.py` had a function-scoped autouse fixture that called `Base.metadata.drop_all(engine)` on the shared session database after each test, destroying the schema owned by the session fixture in `conftest.py`. Every test that ran afterwards failed with "no such table" unless its own module happened to `create_all`, which is why the full suite passed by accident. Not a migration, import or evaluation-DB problem. **Fix:** the fixture only ensures the schema exists and never drops it. Verified: `pytest tests/test_evaluation.py` alone passes; `test_evaluation.py` + `test_phase36_workflow_hardening.py` in that order passes; both are now regression tests (subprocess). Isolation also asserted: the pytest DB is a distinct file, `/evaluation/run` leaves operational row counts unchanged, and `EvaluationRunner` refuses an operational session.

## Performance
Trace reconstruction uses a **constant 5 SELECTs** regardless of size (asserted: a case with 150 extra audit rows and 60 ledger rows uses the same count as a small one, ≤ 6). Real dev DB, warm: reconstruct 1.8–2.0 ms, validate ~1.2 ms per case.

## Browser verification (real app, operator key sent by the frontend)
- **A** `p16_A_auto` (₹300): Policy PERMITTED → executed → "EXECUTED — RECOVERY NOT YET CONFIRMED", outcome PENDING.
- **B** `p16_B_high` (₹20,000): ESCALATE → approval → revalidation → execution; trace card: Agent/AI prediction/Policy·PERMITTED/Approval/Execution OCCURRED, Outcome PENDING; "Handled by PaymentAgent (… proposes only — the Policy Engine decides)", "Trace trace_7251e10f".
- **C** (₹20,000, not approved): "AWAITING HUMAN APPROVAL — NOT EXECUTED", execution NOT_OCCURRED.
- **D** suppressed: Policy DENIED, "Action taken: REJECTED", execution BLOCKED_OR_FAILED, outcome NOT_OCCURRED, "NOT EXECUTED — BLOCKED".
- **E** pending outcome: executed, outcome PENDING, recovered 0.
A/B/C/E/D validate `consistent: true` with 0 issues. Anonymous access to the trace endpoint returned 401. Scenarios were created through the real API; approval in B was made through the same endpoints the button calls (the button path itself was exercised in Phase 15).

## Test results
- New: `tests/test_phase51_phase16_traceability.py` — 62 tests (identity, propagation, policy/execution/outcome traceability, immutability, idempotency incl. a second authorized attempt = new operation in the same trace, no-invention, validator with planted defects, authn/authz matrix, read-only guarantee, redaction, failure safety, constant query count, order-independence, DB isolation).
- Full backend regression: pass. `ruff check`/`ruff format`: clean. Frontend `typecheck`: clean; `lint`: 0 errors, same 13 pre-existing warnings.
- Migration: upgrade `c3d4e5f6a7b8 → e5f6a7b8c9d0`, downgrade, upgrade again verified on a scratch SQLite; dev DB upgraded additively (backup `recoverchain.db.bak_pre_phase16`).
- No existing test assertion was weakened. Changes to existing test files: `conftest.py` adds a suite-wide override for `require_operator` in the same style as the existing `verify_api_key` bypass; `test_evaluation.py` fixture no longer drops the schema.
- Cache cleanup: not needed; backend/frontend processes restarted, database not deleted.

## VERIFIED
One immutable journey id reaches every stage; executions and policy decisions reference it; outcome and execution stay separate; missing stages are `NOT_OCCURRED`, not invented; validator reports without repairing; operator role separation works; secrets/PII redacted; observability is read-only and mutation-free; test-order dependency fixed at its root.

## UNKNOWN
Real user identity of approvers; behaviour under multiple tenants; per-attempt proof that revalidation ran (enforced in code, not recorded); production-scale trace volumes.

## BLOCKERS
None.

## Production validation status
`NOT_READY` — unchanged. Traces of test/simulated executions are not production evidence.

## NBA status
`DATA_BLOCKED` — unchanged.

## Phase 17 recommendation
**GO** for: recording revalidation results and the acting role on approval events; extending operator protection to the remaining sensitive reads (`/system/cases*`, `/system/events`, `/system/policy`) with an explicit product decision; a real per-user auth model if the system leaves single-tenant dev use.
**NO-GO** for: back-filling or rewriting legacy audit history; presenting trace/benchmark data as production evidence; adding an LLM to the live path; NBA training.
