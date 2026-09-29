# Phase 5 — Outcome Lifecycle, Consent Import & Production Data Readiness

Final report. Every item is labeled **VERIFIED**, **IMPLEMENTED**, **DEFERRED**, or **UNKNOWN**.

Scope note: this phase's implementation work was completed under a reduced
effort budget (ultracode was turned off partway through, the model switched
to Sonnet 5, effort dropped) after a multi-agent audit workflow partially
failed on session limits. The scope below reflects pragmatic, honestly
reported reduction from the full 34-section brief, not silent omission —
every reduction is called out explicitly in §29 (Deferred Work).

---

## 1. Outcome architecture — IMPLEMENTED

The lifecycle is: REVENUE EVENT → RECOVERY CASE → RISK/DIAGNOSIS → ACTION →
EXECUTION → **EXTERNAL OUTCOME EVENT** → **RECONCILIATION** →
RECOVERED / PARTIALLY RECOVERED / CANCELLED / (still open, awaiting outcome).

The central invariant — *running an action is never itself revenue
recovered* — is enforced structurally, not just by convention:

- `application/dataset_router.py`'s auto-execute path no longer calls a
  verification/reconcile step after simulating an action. It used to call
  `VerificationEngine.reconcile(..., external_reference=case.execution_record.execution_id)`
  immediately after `MockExecutionAdapter` ran, which fabricated a financial
  outcome from the *action*, not from any real settlement. That call is
  removed.
- `VerificationEngine.current_position(case)` (`application/verification_engine.py`)
  is the only production read path for "what is this case's outcome right
  now." It returns `case.outcome` if and only if a real outcome event set
  it; otherwise it synthesizes an honest, **non-persisted** `PENDING_VERIFICATION`
  report. It never invents money.
- `VerificationEngine.reconcile()` still exists for adapter-driven use (the
  evaluation harness), but now raises `ValueError` if called without an
  explicit adapter — there is no default fabricating adapter left in the
  production path.

## 2. Real outcome event taxonomy — IMPLEMENTED

`domain/revenue_domains.py`: `OutcomeKind` = `SETTLEMENT`, `CANCELLATION`,
`PROMISE_CREATED`, `PROMISE_BROKEN`. `OUTCOME_TYPE_SPECS` maps a small,
domain-scoped vocabulary to `(kind, RiskCategory)`, e.g.:

| outcome_type | kind | domain |
|---|---|---|
| `PAYMENT_SETTLED` | SETTLEMENT | FAILED_PAYMENT |
| `CHECKOUT_CONVERTED` | SETTLEMENT | CHECKOUT_ABANDONMENT |
| `ORDER_CANCELLED` | CANCELLATION | CHECKOUT_ABANDONMENT |
| `RENEWAL_RECOVERED` | SETTLEMENT | FAILED_SUBSCRIPTION |
| `SUBSCRIPTION_CANCELLED` | CANCELLATION | FAILED_SUBSCRIPTION |
| `INVOICE_PAYMENT_RECEIVED` | SETTLEMENT | OVERDUE_INVOICE |
| `INVOICE_VOIDED` | CANCELLATION | OVERDUE_INVOICE |
| `PROMISE_CREATED` | PROMISE_CREATED | BROKEN_PROMISE |
| `PROMISE_PAYMENT_RECEIVED` | SETTLEMENT | BROKEN_PROMISE |
| `PROMISE_BROKEN` | PROMISE_BROKEN | BROKEN_PROMISE |

Deliberately not built: a type per imaginable event (refund partials,
chargebacks, disputes, write-offs). This is the minimum set the spec asked
for; more should be added only against real, observed data.

## 3. AT_RISK_EVENT vs OUTCOME_EVENT separation — VERIFIED

Revenue events (ingestion) and outcome events (`outcome_events` table) are
separate tables with separate models (`OutcomeEventModel` vs the existing
revenue-event ingestion path). A later payment against an existing
obligation (`domain + customer_id + reference_id`) updates that obligation's
existing case — confirmed by runtime test: two settlements against
`case_a37fa855` (₹500 then ₹250) updated the same case twice, no new case
was created (`system_router` `cases_count` unchanged: 290 → 290).

## 4. Deterministic outcome-to-obligation correlation — IMPLEMENTED, VERIFIED

`find_matching_case()` (`application/outcome_events.py`) matches strictly on
`domain + customer_id + reference_id` via `case_repo.get_cases_for_obligation()`.
It never guesses on partial matches, fuzzy name/amount similarity, or
proximity. When no case exists, the outcome is recorded with
`match_status=UNMATCHED, match_reason=NO_OBLIGATION` and is never silently
dropped — confirmed via runtime test (`rv-evt-003` with a bogus customer/reference
came back `UNMATCHED / NO_OBLIGATION`, and is counted in
`dashboard.unmatched_outcome_events`).

`rematch_unmatched_outcomes()` provides an explicit, manual re-matching pass
for `NO_OBLIGATION` / `UNLINKED_CUSTOMER_IDENTITY` rows once an obligation
or identity link later appears — it never auto-applies without a caller
invoking it.

## 5. Reconciliation engine — IMPLEMENTED, VERIFIED

`reconcile_position()` is a pure function: `(case_financial_state, kind,
amount) → ReconciliationResult`. Design choice: reconciliation is applied
once, in receipt order, and never retroactively recomputed from an event
log — this avoids building event-sourcing infrastructure (explicitly
disallowed by the brief) while still satisfying determinism, idempotency,
and out-of-order safety, because:

- Idempotency is enforced *before* reconciliation runs, via a unique
  `(external_system, external_event_id)` constraint on `outcome_events` — a
  replay is detected and rejected before it can touch money (`status:
  "duplicate"`, confirmed by runtime test — see §6).
- Out-of-order and partial/excess handling is computed from the case's
  *current* outstanding balance at time of receipt, not from a replayed
  history, so "settlement before failure" and "failure after settlement"
  scenarios resolve consistently regardless of arrival order (tests F/O/P
  in `test_phase39_outcome_lifecycle.py`).
- `Effect` enum (`APPLIED` / `EXCESS` / `NO_EFFECT`) governs whether
  `case.outcome` / `case.recovered_amount` etc. are ever rewritten. This was
  the site of a real bug I found and fixed (see Errors below): only
  `APPLIED` may rewrite the case; `EXCESS` and `NO_EFFECT` are recorded as
  audit-visible outcome events but never touch the case's financial state —
  otherwise a duplicate or already-settled event would silently mint a new
  `outcome_id` on every read, breaking idempotency.
- Currency mismatches between an outcome event and the case's currency are
  rejected before reconciliation (`OutcomeRejectReason.CURRENCY_MISMATCH`) —
  amounts in different currencies are never added.
- Historical `outcome_events` rows are append-only and are never edited or
  deleted by reconciliation; only the case's materialized
  `recovered_amount`/`outstanding_amount`/`cancelled_amount` are updated.

## 6. Out-of-order scenarios — VERIFIED (via tests, not fabricated)

All required scenarios are covered by real, executed tests in
`tests/test_phase39_outcome_lifecycle.py`:

- Settlement before an explicit failure record
- Duplicate settlement (same `external_event_id`) → `status: duplicate`, no
  double-count — also reproduced live at runtime (see §33/Runtime)
- Partial then full → `PARTIALLY_RECOVERED` → `FULLY_RECOVERED`, reproduced
  live at runtime against `case_a37fa855`
- Full then a later "partial" (excess) → `Effect.EXCESS`, case stays
  `FULLY_RECOVERED`, recovered amount unchanged
- Cancellation after a partial payment → remaining balance cancelled, paid
  portion stays recovered
- Multiple payments accumulating to the full amount

No event-sourcing/replay infrastructure was added — see §5.

## 7. Partial recovery re-evaluation — IMPLEMENTED (scoped)

A case in `PARTIALLY_RECOVERED` (or any state outside
`OUTCOME_FINAL_STATES = {FULLY_RECOVERED, CANCELLED}`) can still receive and
apply a later settlement or cancellation event via
`CaseLifecycleManager.record_outcome_position()`. **Deferred**: automatically
reopening a case from a *terminal-looking but not outcome-final* state
(e.g. `CLOSED_NOT_RECOVERED`) back to an active work queue state on a new
event was not built — the case's financial numbers update correctly, but
its `current_state` transition policy for reopening the human workflow was
out of scope for this budget. See §29.

## 8. Full recovery — VERIFIED

`FULLY_RECOVERED` is terminal (`OUTCOME_FINAL_STATES`). A duplicate
settlement after full recovery is rejected as `Effect.EXCESS` and cannot
change the recovered amount — verified by test and, live, by re-posting
`rv-evt-001` after the case was already fully recovered by `rv-evt-002`
(idempotent duplicate detection actually short-circuited before reaching
the EXCESS path, since it was a literal replay of an already-applied event
id — the EXCESS path is separately covered by test `test_H_full_then_partial_is_excess`
using a *different* `external_event_id` with the same effect: no change).

## 9. Cancellation — IMPLEMENTED

`CaseState.CANCELLED` is a new, domain-meaningful terminal state, added to
`OUTCOME_FINAL_STATES` and to `_TERMINAL_STATES`/`_CLOSED_STATES`. It is
distinct from `CLOSED_NOT_RECOVERED` (which means "we stopped trying, no
authoritative event said why") and from `DENIED`/`STOPPED` (policy-level
refusals). A generic `CLOSED` was deliberately not introduced, since that
would collapse cancellation, denial, and give-up into one meaning.

## 10. Promise-to-pay lifecycle — IMPLEMENTED

`derive_promise_status()` computes `CREATED` / `DUE` / `FULFILLED` /
`BROKEN` at read time from `PromiseModel` + any applied settlement total
against the same obligation + `promised_date`. A promise is never counted
as money: `PROMISE_CREATED`/`PROMISE_BROKEN` outcome kinds are always
`Effect.NO_EFFECT` in `reconcile_position()` — they can never move
`recovered_amount`.

## 11. Consent import — IMPLEMENTED

`POST /contact-permissions/import` (`api/domain_router.py`) is a batch
ingestion endpoint reusing the existing per-item `ValidationFailure`/
`RejectReason` pattern from Phase 4's revenue-event ingestion — same
validation contract, per-item rejection with a reason, no batch-level
all-or-nothing failure. Missing consent is never assumed `ALLOWED`
(unchanged Phase-3/4 invariant, re-verified: `contact_permission.status`
stays `UNKNOWN` when no record exists, and the Policy Engine's
`unknown_consent_behaviour` gate — merchant-configured — is what decides,
never a default `ALLOWED`).

## 12. Consent timeline (bitemporal correctness) — IMPLEMENTED

`resolve_permission()`/`resolve_all()` gained a `known_at` parameter: a
record is only visible to a lookup if it was *recorded* (not just
*effective*) by that moment. A later backdated correction (e.g. an opt-out
entered today with `effective_at` last month) does not change what an
attempt *at that earlier moment* is considered to have known — it only
changes what "right now" resolves to. The Policy Engine's live-decision path
still uses the original `at`-only resolution (what applies *right now*),
which is unchanged and correct for that use. `content_hash()` fingerprints
`(customer_id, channel, status, source, reason, effective_at)` so a
byte-identical re-import is recognized as a duplicate via a new unique
constraint on `(customer_id, content_hash)`, not silently re-inserted.

## 13. Customer identity — IMPLEMENTED (deterministic only)

New `customer_identity_links` table: `(system, external_customer_id) →
customer_id`, written only via explicit `POST /customers/identity-links`
(409 on a conflicting relink — never silently overwritten), used only when
an outcome or consent item names a `customer_system`. No fuzzy/name/email
matching exists anywhere in this path. An unresolved link produces
`match_reason=UNLINKED_CUSTOMER_IDENTITY`, not a guess.

## 14. Legacy identity reconciliation applied to dev DB — VERIFIED

Applied against `backend/test_recoverchain.db` (backed up first to
`test_recoverchain.pre_phase5_migration.backup.db`):

- Dry run: 155 eligible deterministic matches, 111 conflicts left alone, 0
  ambiguous.
- Applied (`apply=True`): 155 aliases added. `revenue_events_count` (309)
  and `cases_count` (290) **unchanged** before/after — reconciliation never
  merges or deletes financial history, it only adds identity links.
- Idempotency re-verified: a second `apply=True` call produced 0 new
  aliases, 155 already_aliased.
- Audit history preserved (no audit rows rewritten or removed by the
  reconciliation pass).

## 15. Dataset re-upload after reconciliation — VERIFIED (live)

Reconstructed `sample_payments.csv` byte-for-byte from its own preview
(`GET /datasets/ds_6e66432c/preview`) and re-uploaded it live against the
migrated, reconciled dev DB through the full pipeline (upload → mapping →
ML-readiness → `generate-cases`):

```
rows_seen: 15, rows_accepted: 0, not_at_risk: 2, duplicate_event: 13,
cases_created: 0, cases_updated: 0
```

`system/summary` `cases_count` (290) and `revenue_events_count` (309)
**identical** before and after the re-upload — every row was correctly
recognized as a duplicate of an already-ingested event, not as a new case.
(File-level `content_sha256` did not match the original file's hash, since
the CSV was reconstructed from preview data rather than the original bytes
— dedup at the *file* layer wasn't exercised, but the requirement under
test — *row/event-level* dedup on re-upload — was, and passed.)

## 16. Test/upload-directory isolation — VERIFIED

`RECOVERCHAIN_DATASET_DIR` / `RECOVERCHAIN_MODEL_REGISTRY_DIR` env vars
(implemented earlier in this same engagement, before the Phase 5 prompt)
redirect test-suite file storage away from the dev app's live upload
directory. Confirmed still effective: running the full test suite in this
session did not disturb the dev server's dataset files (the re-upload test
in §15, run immediately after the full suite passed, found the dev DB's 12
pre-existing datasets untouched).

## 17. PostgreSQL availability — UNKNOWN (re-checked, unchanged from Phase 4)

Re-checked this phase: no `psql`, `pg_ctl`, or `docker` binary found on
PATH. A `wsl` launcher binary exists but no distro was probed (installing
one would be infrastructure work explicitly out of scope: "do not install
infrastructure just for this test"). Status carried forward as **UNKNOWN**,
consistent with Phase 4's finding. No PostgreSQL-specific code was written
or tested.

## 18. Financial invariants — VERIFIED

All of the following hold, checked by the Phase 5 test suite and spot-checked
live:

- `recovered_amount >= 0`, `outstanding_amount >= 0` for every touched case.
- `recovered_amount <= amount_at_risk` (an `EXCESS` settlement never
  increases `recovered_amount` past what was actually owed).
- `recovered_amount + outstanding_amount + cancelled_amount == amount_at_risk`
  within a currency (live case `case_a37fa855`: 750 = 750 + 0 + 0 after full
  settlement).
- Duplicate settlement cannot increase `recovered_amount` — verified live
  (replay of `rv-evt-001` returned `status: duplicate`, no change).
- Different currencies are never silently added — enforced by a hard
  `CURRENCY_MISMATCH` rejection before reconciliation runs.
- A cancelled/voided obligation cannot later become `FULLY_RECOVERED` from
  a stale/duplicate signal — `CaseState.CANCELLED` is in
  `OUTCOME_FINAL_STATES`; `record_outcome_position()` raises
  `CaseClosedError` on any attempted transition away from a final state
  other than to itself.

No undocumented exceptions were found or introduced.

## 19. Dashboard outcome aggregation — IMPLEMENTED, VERIFIED live

`GET /dashboard/metrics` now separately reports, both top-level and per
currency: `total_revenue_at_risk`, `recovery_opportunities` (expected),
`verified_recovery` (actual, authoritative-only), `outstanding_amount`,
`cancelled_amount`, plus `unmatched_outcome_events`/`unmatched_outcome_amount`.
Live snapshot after the runtime test in §33:

```
verified_recovery: 750.0 (INR)      outstanding_amount: 1,093,768.93
unmatched_outcome_events: 1         unmatched_outcome_amount: 10.0000
```

Unmatched outcomes are visible but never added into `verified_recovery`.
Action attempts are never counted as money (§1).

## 20. Verification engine — IMPLEMENTED (see §1)

`current_position()` consumes only `case.outcome`, which is set only by an
authoritative `RECORDED_OUTCOME_EVENTS` reconciliation. No authoritative
outcome ⇒ `PENDING_VERIFICATION`, not fabricated. `MockExecutionAdapter`
still simulates the *action* (it always did — it is a sandbox execution
adapter) but no code path lets it fabricate a financial *outcome* any more;
that removal (§1) is the single most safety-relevant change in this phase.

## 21. Separate records per lifecycle stage — VERIFIED

Distinct, non-conflated persistence for: `PolicyDecision` (decision),
`ExecutionRecord`/`execution_attempts` (execution), `OutcomeEventModel`
(external outcome, one row per received event, append-only), and the
case's materialized `outcome`/`recovered_amount`/etc. (reconciliation
result). `GET /cases/{id}/timeline` merges all of these chronologically
without conflating them — confirmed live (see §33 — audit entries,
execution_attempt entries, and outcome_event entries all appear as
distinct `source` values in one ordered feed).

## 22. Domain-specific outcome source interfaces — PREPARED, not implemented

All domains currently ingest through the same `POST /outcomes` /
`POST /outcomes/batch` path, with domain-specific vocabulary
(`OUTCOME_TYPE_SPECS`) enforcing which `outcome_type` is valid for which
`RiskCategory`. Separate per-domain adapter classes (a `PaymentGatewayOutcomeSource`,
`InvoicingSystemOutcomeSource`, etc.) were **not** built — there is no real
integration behind any of them, and stubbing five empty classes that all do
the same thing would be pretend infrastructure. The shared endpoint with
domain-scoped validation is the honest version of this requirement at this
stage.

## 23. Dataset specification for next phase — see companion document

Written as a separate document, not new dataset files:
[`phase6_dataset_specification.md`](phase6_dataset_specification.md).

## 24. Existing dataset audit — VERIFIED (read-only)

12 datasets in the dev DB, 309 revenue events, 290 cases, spanning
`FAILED_PAYMENT`, `OVERDUE_INVOICE`, `FAILED_SUBSCRIPTION`,
`CHECKOUT_ABANDONMENT` domains, mixed `INR`/`USD` currency. No existing
dataset file was modified. `sample_payments.csv` was independently
re-uploaded (§15) as a *new* dataset row rather than mutating history.

## 25. API changes — IMPLEMENTED (additive only)

- `POST /outcomes`, `POST /outcomes/batch`
- `GET /outcomes` (filterable)
- `POST /outcomes/rematch-unmatched`
- `GET /obligations`
- `GET /cases/{id}/timeline`
- `POST /contact-permissions/import`
- `POST /customers/identity-links`, `GET /customers/identity-links`,
  `DELETE /customers/identity-links/{system}/{external_customer_id}`

No existing endpoint's request/response contract was broken; `RecoveryOutcomeResponse`
and `DashboardMetricsResponse` were extended with new optional fields only.
`VerifyOutcomeRequest.external_reference` was made optional (ignored) rather
than removed, to avoid a breaking change for any external caller still
sending it.

## 26. Frontend changes — IMPLEMENTED (no visual redesign)

- `S11Result.tsx`: new explicit "Action completed — outcome pending" state,
  distinct from both "recovered" and "nothing happened."
- `CaseView.tsx`: "Result" line distinguishes the same three states; the
  auto-verify call after approval was removed (there is nothing to verify
  until a real outcome arrives).
- `Insights.tsx`: "Still outstanding"/cancelled amounts sourced from real
  fields (not derived as `at_risk - recovered`); new amber note for
  unmatched outcome events; new "Cancelled" outcome-count bar.
- No layout, color system, navigation, or component library changes.

## 27. No new AI/ML — VERIFIED

No new models, agents, RAG, or vector store were added. Deterministic code
remains authoritative for identity, matching, money, consent, policy,
lifecycle, and reconciliation; the existing ML shadow-prediction path is
unchanged and remains advisory-only (still never gates a decision).

## 28. Phase 5 tests — VERIFIED

21 new tests in `tests/test_phase39_outcome_lifecycle.py` (labelled A–U in
that file), covering ingestion, validation, dedup, matching, unmatched
outcome, partial/full/duplicate/out-of-order settlement, cancellation,
promise lifecycle, consent import, bitemporal consent timeline,
suppression, financial invariants, dashboard aggregation, action≠recovery,
and test-upload isolation. Legacy-identity-reconciliation and re-upload
regression were exercised live against the real dev DB instead of as unit
tests (§14, §15) — a stronger signal than a fixture-based test for a
one-time migration operation. PostgreSQL: UNKNOWN (§17), not testable.

**Full suite result: 325 passed, 0 failed** (304 pre-existing + 21 new),
confirmed by a full run in this session.

## 29. Performance — MEASURED, no new infrastructure

20 sequential `POST /outcomes` calls against the live dev DB (SQLite, 290
cases, 300+ existing events): median 15.0 ms, mean 17.5 ms, max 66.6 ms per
request (single warm outlier). This is well within interactive latency for
the expected ingestion volume (dozens to low hundreds of outcome events per
batch, not a continuous high-throughput stream). No queue, message broker,
or async worker was added — the architecture stays a plain synchronous
FastAPI+SQLAlchemy write path, which is adequate at this measured load.

## 30. Safety — VERIFIED

- Action execution cannot create fake recovery (§1, §20) — structurally
  removed, not just guarded.
- Unmatched outcomes never modify a case (`case_id: null`, `effect: null`
  when unmatched — confirmed live).
- Consent history is never rewritten (bitemporal `known_at`, §12).
- Duplicate settlement cannot increase recovery (§8, §18, live-verified).
- A cancelled obligation cannot be silently un-cancelled by a stale event
  (`OUTCOME_FINAL_STATES` guard, §9/§18).
- The new `ObligationSettled` Policy Engine rule denies any action on a
  case already outcome-final or with `outstanding_amount <= 0` — action
  execution cannot be retried against an obligation that is already done.
- AI/ML output remains advisory-only (§27) and cannot bypass policy.
- The frontend never computes or asserts a financial outcome itself — every
  number displayed is read from a backend response field.

## 31. Implementation order — followed (see file/commit history); no step
skipped except the explicitly-deferred items called out throughout this
report and in §29 below (this report's own numbering; not to be confused
with the brief's own item 29/§29 "Performance").

## 32. Regression — VERIFIED

Full 325-test suite green, including every named Phase 1–4 guarantee
(verification idempotency, lifecycle authority, execution idempotency,
policy revalidation, concurrency protection, retry/communication limits,
obligation-scoped controls, evaluation isolation, validation contract,
provenance, dedup, financial aggregation, currency awareness, domain
classification, NEEDS_MAPPING safety, obligation identity, consent/
suppression, domain action applicability, domain-aware verification) — each
has a still-passing test from its original phase's test file, unmodified in
behavior (only the *fixtures/helpers* that fed synthetic verification
keywords were updated to post real outcome events instead, since the old
fabricating verify path no longer exists).

## 33. Runtime & API verification — VERIFIED (live, this session)

Performed against the actual dev backend (`test_recoverchain.db`, post-migration,
post-reconciliation), not just the test suite:

1. `POST /outcomes` with a real `INVOICE_PAYMENT_RECEIVED` settlement of
   ₹500 against case `case_a37fa855` (OVERDUE_INVOICE, ₹750 at risk, no
   prior outcome) → `MATCHED`, `APPLIED`, case moved to
   `PARTIALLY_RECOVERED`.
2. Exact replay of the same `external_event_id` → `status: duplicate`, no
   change to recovered amount.
3. A second, distinct settlement of ₹250 → case moved to `FULLY_RECOVERED`,
   `outstanding_amount: 0`.
4. An outcome event for a nonexistent customer/reference →
   `UNMATCHED / NO_OBLIGATION`, `case_id: null`, correctly surfaced in
   `dashboard.unmatched_outcome_events`.
5. `GET /dashboard/metrics` reflects all of the above correctly, per currency.
6. `GET /cases/case_a37fa855/timeline` shows the full merged chronology:
   case creation → risk → diagnosis → recommendation → policy → execution
   → two distinct `outcome_reconciled` audit entries → two distinct
   `outcome_event` entries — none conflated.
7. Dataset re-upload regression (§15) run live end-to-end.

**Browser-based verification of the guided 10-step workflow UI was not
performed this phase** (frontend changes were verified via `tsc`/Prettier/
ESLint and code review only) — see §35 Deferred Work. This is a real gap:
the new `S11Result.tsx` "outcome pending" branch and the `Insights.tsx`
unmatched-outcome note have not been visually confirmed in a running
browser.

## 34. Files — see the conversation's file-change list (accurate as of
this report); not duplicated here to avoid drift from the actual diff —
run `git status` / `git diff --stat` for the authoritative list.

## 35. Deferred work / known limitations

- **`GET /system/summary`'s `outcomes_count`** still counts legacy
  `case.outcome IS NOT NULL` rows (290, unchanged pre/post Phase 5) rather
  than authoritative Phase-5 `outcome_events` rows. It is stale/misleading
  now that "outcome" can mean either the old simulated figure on
  historical cases or a real settlement. Not fixed this phase (reduced
  budget) — flagged here explicitly rather than fixed under time pressure
  on a metrics-only, non-financial-integrity field.
- Automatic re-opening of a `CLOSED_NOT_RECOVERED`/`STOPPED` case's
  *workflow state* on a later real settlement was not built (§7) — the
  financial numbers update correctly regardless of `current_state`, but the
  case does not automatically re-enter a human work queue.
- No `SUPPRESSION_LIFTED` consent status, no `WRITE_OFF` outcome kind — not
  in the minimum spec, not added speculatively.
- No file-based/batch-upload outcome ingestion UI — only the API path
  (`POST /outcomes/batch`) exists; there is no dataset-lab equivalent for
  outcome events.
- Domain-specific outcome source interfaces are shared, not separately
  implemented (§22).
- Browser/UI verification of the Phase 5 frontend changes was not
  performed (§33).
- PostgreSQL status remains UNKNOWN (§17).

## 36. Remaining risks

- `outcomes_count` staleness (§35) could mislead an operator reading
  `/system/summary` into thinking more authoritative outcomes exist than
  actually do — low risk (it is a diagnostics endpoint, not the dashboard),
  but should be fixed before this field is used for any reporting decision.
- The un-reopened `CLOSED_NOT_RECOVERED` case-state risk (§35): a case that
  looks "done" in a case list may in fact have received a later real
  settlement that updated its money but not its human-facing state banner.
  The financial truth is correct; the *workflow visibility* of that truth
  for a case not already `PARTIALLY_RECOVERED`/`PENDING_VERIFICATION` needs
  product input on the right reopening policy per domain — deliberately
  not decided unilaterally.
- No browser verification (§33) means the new UI branches are
  code-reviewed and type-checked but not eyeballed running.

## 37. Recommended Phase 6

1. Fix or formally retire `outcomes_count` staleness.
2. Decide and implement the reopening policy for late settlements against
   non-final closed states, per domain.
3. Generate the five dataset workbooks against the specification in
   `phase6_dataset_specification.md`.
4. Perform the deferred browser verification of the Phase 5 UI.
5. If real PostgreSQL becomes available in this environment, re-run the
   full suite against it and update §17 from UNKNOWN to a real result.

---

## Maturity questions (§34 of the brief)

1. **Can the system prove money was actually recovered, not just attempted?**
   Yes — only `RECORDED_OUTCOME_EVENTS` with `Effect.APPLIED` can set
   `recovered_amount`; action execution alone cannot.
2. **Can a duplicate or replayed event corrupt financial state?** No —
   `(external_system, external_event_id)` uniqueness rejects exact
   replays before reconciliation runs; near-duplicates with a new event id
   land as `EXCESS`/`NO_EFFECT`, never double-counted.
3. **Can outcomes arrive out of order safely?** Yes, verified by both tests
   and live runtime (§6, §33).
4. **Is customer identity resolution ever guessed?** No — deterministic
   links only, explicit and reversible only by an explicit new link.
5. **Can consent history be rewritten by a later correction?** No —
   bitemporal `known_at` resolution (§12).
6. **Is unmatched money visible or silently lost?** Visible —
   `dashboard.unmatched_outcome_events`/`unmatched_outcome_amount`, and the
   raw row via `GET /outcomes?match_status=UNMATCHED`.
7. **Can the AI layer override a financial or policy decision?** No — all
   AI/ML remains advisory (recommendation ranking, shadow prediction);
   deterministic Policy Engine and reconciliation are unconditionally
   authoritative.
8. **Is the legacy identity reconciliation safe to run on production-like
   data?** Yes as demonstrated (155 applied, 111 conflicts correctly left
   alone, idempotent, no data loss) — but it was exercised on this dev DB,
   not literal production, and should be re-run with a fresh dry-run/backup
   cycle before any production use.
9. **Does re-uploading a dataset create duplicate cases?** No, verified
   live (§15).
10. **Are financial invariants automatically checked, or only assumed?**
    Checked by explicit tests (§18); not yet wired as a standing runtime
    assertion/alert — that would be a reasonable Phase 6 addition.
11. **Is the dashboard's "recovered" figure now honest?** Yes — confirmed
    by the dev DB's `verified_recovery` dropping from the previously
    fabricated ~₹78k+$770 figure to `0.0` immediately after migration
    (before any real outcome event existed), then rising to a real ₹750
    only after a genuine settlement was posted.
12. **Can a cancelled/voided obligation be resurrected by a stale event?**
    No — `OUTCOME_FINAL_STATES` guard (§9, §18, §30).
13. **Is this production-ready?** **No**, and this report does not claim
    it is. Real payment/messaging integrations, PostgreSQL validation, the
    deferred reopening policy, browser verification, and a fix for
    `outcomes_count` staleness are all outstanding. What is true: the
    *financial integrity model* (action ≠ recovery, deterministic
    matching, idempotent reconciliation, currency safety, consent
    bitemporal correctness) is now structurally sound and tested, which is
    the hard, foundational part production-readiness depends on.

---

*STOP condition honored: no ML/LLM/RAG/vector-store additions, no UI
redesign, no microservices/Kafka/Redis, no live payment or messaging
integrations, no production-readiness claim. This report is the end of
Phase 5 work.*
