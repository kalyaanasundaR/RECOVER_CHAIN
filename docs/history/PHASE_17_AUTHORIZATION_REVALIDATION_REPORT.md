# Phase 17 — Authorization Evidence, Policy Revalidation Audit, Operational Security

Status: COMPLETE. Invariants held: **APPROVAL ≠ AUTHORIZATION FOREVER**, **EXECUTION ≠ RECOVERY**.
No LLM in the live path. No NBA training. No fabricated identities, trace IDs or outcomes. No UI redesign. Legacy history untouched.

## 1. Executive Summary
- Every execution now writes an append-only, structured `policy_revalidation` audit record (PASSED **and** FAILED) before it acts or refuses.
- Approvals record the authenticated principal id and role, an `approval_id`, the policy version and the policy decision reference. Historical approvals show `UNKNOWN`.
- A bounded role vocabulary (USER / OPERATOR / REVIEWER / ADMIN) is enforced server-side. Human review needs REVIEWER, outcome/evidence ingestion needs ADMIN, and an OPERATOR credential is read-only.
- Sensitive reads were closed or reclassified; the remaining public reads are an explicit decision with a production switch.
- Audit rows, execution-ledger rows, dataset snapshots and outcome evidence are append-only at the ORM layer.
- Result: 617 backend tests pass, Ruff clean, frontend typecheck clean, lint 0 errors (13 pre-existing warnings), migration round trip OK, browser verified.

## 2. Authentication Model
- Unchanged transport: `X-API-Key`. New identity abstraction in `backend/api/principals.py`; keys resolve with `hmac.compare_digest`.
- Legacy keys: `OPERATOR_API_KEY` → OPERATOR; `API_KEY` → USER (when an operator key is set) or ADMIN (dev default when it is not).
- Real principals: env `RECOVERCHAIN_PRINCIPALS` (JSON `key → {principal_id, role, active}`). Bad JSON or a role outside the vocabulary fails closed.
- Legacy-key principal id is a credential fingerprint `key:<sha256[:10]>` — explicitly **a credential, not a person**.
- Audit evidence stores only `principal_id`, `role`, `authentication_method`; never the credential.
- Status codes: new dependencies return 401 (missing / unknown / inactive) and 403 (role lacking); `verify_api_key` keeps its historical 403.
- Limits: static shared keys, no expiry, no rotation, no per-user login. This is not an identity provider.

### Frontend key confidentiality
| Item | Class |
|---|---|
| `VITE_API_KEY` / bundled fallback | DEVELOPMENT-ONLY, **public** — anyone loading the app can read it |
| Reviewer key | CONFIDENTIAL — typed at runtime, kept in `sessionStorage` for one tab, never bundled or persisted |
| Operator key | OPERATOR-ONLY — never shipped to the browser |
| Server-side role checks | the only authorization; hiding a button is not one |

## 3. Authorization Model
| Role | act (execute/advance/…) | observe (traces, audit, policy…) | approve | admin (outcomes, evidence sources) |
|---|---|---|---|---|
| USER | yes | no | no | no |
| OPERATOR | no | yes | no | no |
| REVIEWER | yes | yes | yes | no |
| ADMIN | yes | yes | yes | yes |

Gap (documented, not solved): the proposer of a case is not tracked, so **no self-approval segregation** is enforced.

## 4. Endpoint Access Matrix (89 endpoints, generated from the route table)
Sensitive-read decisions:

| Endpoint | Decision | Why |
|---|---|---|
| `/system/events`, `/system/policy`, `/system/data-readiness`, `/system/eligibility`, dataset-snapshots, `cases.csv`, `GET /outcomes`, evidence-sources, identity-links, contact-history/permissions | OPERATOR | expose bulk, policy or customer-linked data |
| `/system/cases`, `/system/cases/{id}` | AUTHENTICATED | the UI needs it; still needs a principal |
| `/system/cases/{id}/agent-trace`, `/system/traces/*` | OPERATOR (Phase 16) | 401 anonymous, 403 non-operator, uniform for unknown ids |
| `/cases`, `/datasets`, `/system/summary`, `/dashboard/metrics`, `/health` | PUBLIC in dev | existing tests and the guided workflow rely on it; closed by `RECOVERCHAIN_PROTECT_READS=1` (health/docs stay open) |

Full generated table:

| Method | Path | Access | Class |
|---|---|---|---|
| GET | `/cases` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/cases/{case_id}` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/cases/{case_id}/advance` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| POST | `/cases/{case_id}/assess-risk` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/cases/{case_id}/audit` | OPERATOR | READ_ONLY |
| POST | `/cases/{case_id}/diagnose` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/cases/{case_id}/diagnosis` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/cases/{case_id}/events` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/cases/{case_id}/execute` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| POST | `/cases/{case_id}/human-review` | REVIEWER | MUTATING |
| GET | `/cases/{case_id}/observed-cause` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/cases/{case_id}/observed-cause` | ADMIN | MUTATING |
| GET | `/cases/{case_id}/outcome` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/cases/{case_id}/policy-check` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/cases/{case_id}/policy-decision` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/cases/{case_id}/predict-recovery` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| POST | `/cases/{case_id}/recommend-action` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/cases/{case_id}/recommendation` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/cases/{case_id}/recovery-prediction` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/cases/{case_id}/risk` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/cases/{case_id}/stop` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/cases/{case_id}/timeline` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/cases/{case_id}/verify` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| POST | `/contact-permissions/import` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/customers/identity-links` | OPERATOR | READ_ONLY |
| POST | `/customers/identity-links` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| DELETE | `/customers/identity-links/{system}/{external_customer_id}` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/customers/{customer_id}/contact-history` | OPERATOR | READ_ONLY |
| GET | `/customers/{customer_id}/contact-permissions` | OPERATOR | READ_ONLY |
| POST | `/customers/{customer_id}/contact-permissions` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/dashboard/metrics` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/datasets` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/datasets/prune` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| POST | `/datasets/sync` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| POST | `/datasets/upload` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/datasets/{dataset_id}` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/datasets/{dataset_id}/analyze` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/datasets/{dataset_id}/domain` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| PUT | `/datasets/{dataset_id}/domain` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| POST | `/datasets/{dataset_id}/generate-cases` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/datasets/{dataset_id}/ingestion-runs` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/datasets/{dataset_id}/mapping` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/datasets/{dataset_id}/ml-readiness` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/datasets/{dataset_id}/ml-readiness` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/datasets/{dataset_id}/models` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/datasets/{dataset_id}/predict` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/datasets/{dataset_id}/preview` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/datasets/{dataset_id}/train` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/datasets/{dataset_id}/workflow-status` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/evaluation/run` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/evaluation/runs` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/evaluation/runs/{run_id}` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| POST | `/events` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| POST | `/events/batch` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| POST | `/events/identity-reconciliation` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/health` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/ingestion-runs/{run_id}` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/obligations` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/outcomes` | OPERATOR | READ_ONLY |
| POST | `/outcomes` | ADMIN | MUTATING |
| POST | `/outcomes/batch` | ADMIN | MUTATING |
| POST | `/outcomes/rematch-unmatched` | ADMIN | MUTATING |
| GET | `/system/agent-benchmark` | OPERATOR | READ_ONLY |
| GET | `/system/audit` | OPERATOR | READ_ONLY |
| GET | `/system/cases` | AUTHENTICATED | READ_ONLY |
| GET | `/system/cases.csv` | OPERATOR | READ_ONLY |
| GET | `/system/cases/{case_id}` | AUTHENTICATED | READ_ONLY |
| GET | `/system/cases/{case_id}/agent-trace` | OPERATOR | READ_ONLY |
| GET | `/system/data-readiness` | OPERATOR | READ_ONLY |
| GET | `/system/dataset-snapshots` | OPERATOR | READ_ONLY |
| POST | `/system/dataset-snapshots` | ACT (USER/REVIEWER/ADMIN) | MUTATING |
| GET | `/system/dataset-snapshots/{snapshot_id}` | OPERATOR | READ_ONLY |
| GET | `/system/datasets` | OPERATOR | READ_ONLY |
| GET | `/system/datasets/{dataset_id}` | OPERATOR | READ_ONLY |
| GET | `/system/eligibility` | OPERATOR | READ_ONLY |
| GET | `/system/events` | OPERATOR | READ_ONLY |
| GET | `/system/evidence-sources` | OPERATOR | READ_ONLY |
| POST | `/system/evidence-sources` | ADMIN | MUTATING |
| POST | `/system/evidence-sources/{source_id}/enabled` | ADMIN | MUTATING |
| POST | `/system/evidence-sources/{source_id}/verify` | ADMIN | MUTATING |
| GET | `/system/executions` | OPERATOR | READ_ONLY |
| GET | `/system/health` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/system/model-monitoring` | OPERATOR | READ_ONLY |
| GET | `/system/models` | OPERATOR | READ_ONLY |
| GET | `/system/policy` | OPERATOR | READ_ONLY |
| GET | `/system/summary` | PUBLIC in dev; authenticated when `RECOVERCHAIN_PROTECT_READS=1` | READ_ONLY |
| GET | `/system/trace-consistency` | OPERATOR | READ_ONLY |
| GET | `/system/traces/{trace_id}` | OPERATOR | READ_ONLY |
| GET | `/system/traces/{trace_id}/validation` | OPERATOR | READ_ONLY |

TOTAL=89

## 5. Approval Evidence
`human_review` audit evidence: `approval_id` (`appr_<hex12>`), `case_id`, `trace_id`, `decision`, `approved_action`, `policy_decision_id`, `policy_version`, `actor{principal_id, role, authentication_method}`, note, timestamp. An `actor` supplied in the request body is ignored (tested).

## 6. Acting Identity / Role
Taken only from the authenticated credential. Approvals recorded before this phase render `actor = UNKNOWN` in traces; nothing is backfilled.

## 7. Policy Revalidation
Sequence: initial decision → human approval if required → fresh revalidation → `policy_revalidation` record → execution. Record fields: `revalidation_id`, `trace_id`, `case_id`, `requested_action`, `result`, `reason_codes`, `policy_version`, `initial_policy_version`, `policy_version_changed`, both decision ids, `approval_required`, `approval_reference`, `approval_actor`, `revalidated_at`. The execution record carries `revalidation_id`. Idempotent replays do not add revalidation records.

## 8. Revalidation Failure
The approval stays and is referenced; the failed revalidation is appended; no ledger row, no adapter call, recovered amount unchanged; the trace says `NOT EXECUTED — BLOCKED: POLICY REVALIDATION FAILED`. A later pass appends a new record and never edits the failure.

Matrix (tests `test_4*`; A, B and D also run live): A auto-permitted → PASSED + executed; B high-value approved → PASSED + executed; C not approved → FAILED; D suppression after approval → FAILED; E consent change → FAILED; F retry limit → FAILED; G stricter policy version → FAILED with `policy_version_changed=true`.

## 9. Trace Integration
New `REVALIDATION` stage between APPROVAL and EXECUTION (`NOT_OCCURRED` / `NOT_RECORDED` for legacy / `OCCURRED` with `latest_result`). Journey adds `revalidation` and `approved_by`. Validator adds `REVALIDATION_NOT_RECORDED` (INFO), `EXECUTION_AFTER_FAILED_REVALIDATION` (ERROR), `APPROVAL_WITHOUT_ACTOR` (INFO). Traceability: `TRACED` / `PARTIALLY_TRACED` / `LEGACY_UNTRACED`.

## 10. Execution Protection
Enforced by dependencies on the API, independent of the UI. A missing, wrong or inactive key and an OPERATOR credential cannot execute, advance, stop, verify or run evaluation. Tested through direct API calls.

## 11. Outcome Protection
`POST /outcomes*` and evidence-source writes need ADMIN. Execution never creates an outcome or a recovered amount. `production` provenance through the shared credential is still rejected. Matched outcomes have frozen provenance, evidence, amounts and match columns.

## 12. Legacy Trace Handling
Legacy audit/ledger rows show `trace_id: LEGACY_UNTRACED`, `traceable: false`, revalidation `NOT_RECORDED`, approval actor `UNKNOWN`. Nothing is rewritten or assigned an id (tested by planting legacy rows and checking they are unchanged).

## 13. Audit Immutability
`backend/infrastructure/immutability.py` blocks ORM update/delete on `AuditModel`, `ExecutionAttemptModel`, `MLDatasetSnapshotModel`; blocks delete and frozen-column changes on `OutcomeEventModel`; blocks re-matching an already MATCHED outcome. UNMATCHED→MATCHED rematch still works. Corrections are new appended rows.

**Limitation:** this is an application-layer guard. Bulk `Query.update/delete`, raw SQL and direct DB access bypass it; DB triggers or a write-once store would be needed for tamper evidence.

## 14. Security Negative Tests
`tests/test_phase52_phase17_authorization.py` (56 tests): principals/roles; approval 401/403/200 by role; operator cannot act; operator reads 401/403/200 across 19 endpoints; authenticated case snapshots; public reads and the protect-reads switch; ADMIN-only outcomes; immutability; legacy traces; revalidation matrix A–G; credentials never appear in audit or trace bodies.

## 15. Browser Verification
Backend restarted with a registry REVIEWER. Live: A (auto) executed with PASSED; B approved by `asha.reviewer` executed with PASSED; D approved, then contact suppressed → REJECTED, adapter NONE. The case page showed *Approved by asha.reviewer (REVIEWER · approval is not recovery)*, *Revalidation FAILED · policy merchant-policy-v1.1*, the Trace label `NOT EXECUTED — BLOCKED: POLICY REVALIDATION FAILED`, and stages Approval OCCURRED / Revalidation OCCURRED / Execution BLOCKED_OR_FAILED / Outcome NOT_OCCURRED. An anonymous trace request returned 401. No console errors.

Pre-existing wording issue found (left unchanged, no redesign): the "Safety check" line still reads "Approved. Safe to act on automatically." after a failed revalidation. The Trace and Revalidation rows are correct.

## 16. Performance (median / p95 ms, TestClient, SQLite)
| Operation | Median | p95 |
|---|---|---|
| principal resolution | 0.01 | 0.01 |
| `/health` (no auth) | 9.16 | 10.59 |
| operator read `/system/policy` | 12.29 | 13.77 |
| denied 403 | 7.98 | 9.11 |
| approval recording | 15.34 | 19.48 |
| execute incl. revalidation | 35.91 | 39.69 |
| trace reconstruction | 18.92 | 20.69 |

Authentication and authorization cost is negligible next to request overhead.

## 17. Test Results
- Backend: 617 passed. Ruff check and format clean.
- Frontend: `tsc --noEmit` clean; lint 0 errors, 13 pre-existing warnings.
- Migration: upgrade head → downgrade -1 → upgrade head on a scratch DB; head `e5f6a7b8c9d0` (Phase 17 added no migration).
- Test-isolation fix: Phase 16's subprocess pytest helper shared the parent's test database and dropped its schema at teardown, breaking any later DB test. It now uses a private DB via `RECOVERCHAIN_TEST_DATABASE_URL`. No assertion was weakened.
- The Phase 17 trace label keeps the substring `BLOCKED`, so the Phase 16 test stands unmodified.

## 18. VERIFIED
Server-side role enforcement; approval evidence with the real principal; passed and failed revalidation recorded and traced; approval preserved on failure; no execution or recovery change on failure; legacy rows untouched and labelled; ORM append-only guards; no credential in audit or trace output.

## 19. UNKNOWN
Behavior with a real IdP or multiple tenants; DB-level tamper resistance; concurrent approve/execute races; historical approver identities; whether `RECOVERCHAIN_PROTECT_READS` breaks any external consumer.

## 20. BLOCKERS
- Static shared keys, no rotation or expiry, no per-user identity.
- No self-approval segregation (proposer not tracked).
- SINGLE-TENANT / DEVELOPMENT MODE: cases are not isolated per customer or tenant; any authenticated principal can read any case snapshot.
- Immutability is application-level only.
- Public dev reads unless `RECOVERCHAIN_PROTECT_READS=1`.
- With `OPERATOR_API_KEY` unset the dev default gives ADMIN to the API key (a startup warning is logged).

## 21. Production Validation Status
`PRODUCTION_VALIDATION_STATUS = NOT_READY`.

## 22. NBA Status
`NBA_MODEL_STATUS = DATA_BLOCKED` — untouched; no training, no fabricated labels.

## 23. Phase 18 Recommendation
**Conditional GO** for a security-hardening Phase 18: a real identity provider or per-user keys with rotation, a proposer≠approver rule, DB-level append-only triggers, tenant scoping, and `RECOVERCHAIN_PROTECT_READS` on by default outside dev. **NO-GO** for any production-readiness or NBA claim: there is still no real production outcome data and the blockers above are open.
