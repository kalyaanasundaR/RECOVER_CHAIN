# Phase 6 — Industrial Dataset & End-to-End Scenario Validation: Final Report

Every claim below was produced by driving the real, running application (FastAPI
backend on the migrated dev DB, React frontend, real HTTP calls) — never by writing
directly to the database. Companion documents: [dataset contract](phase6_dataset_contract.md),
the five [datasets](phase6_datasets/), [`test_phase40_phase6_scenarios.py`](../../backend/tests/test_phase40_phase6_scenarios.py).

No STOP condition was triggered. **GO for Phase 7** (see §19).

---

## 1. Dataset contract

See [phase6_dataset_contract.md](phase6_dataset_contract.md) — audited directly from
the live schema (`application/ingestion.py`, `api/dataset_router.py`, `application/
dataset_intelligence.py`, `application/domain_classification.py`, `domain/
revenue_domains.py`, `application/diagnosis_engine.py`, `application/policy_engine.py`,
`application/outcome_events.py`, `api/outcome_router.py`, `api/domain_router.py`).
No new columns, tables, or validation rules were required — the existing schema
already covers every field the 14 scenario requirements need. One real
documentation defect was found and fixed mid-testing: an earlier draft (inherited
from the Phase 5 report) said outcome `amount` was required for SETTLEMENT/
CANCELLATION and omitted for PROMISE_*. The live code actually requires it for
SETTLEMENT **and** PROMISE_CREATED, and never reads it for CANCELLATION — corrected
in the contract doc, with the two affected promise-created outcome posts redone
correctly (see §14).

## 2. Five dataset files created

All in [`docs/history/phase6_datasets/`](phase6_datasets/), one workbook per domain,
genuinely different schemas (different column names, different obligation-reference
semantics, different status vocabularies where the domain calls for it — e.g.
`checkout_stage` only exists in the checkout file, `promised`/arrangement framing
only in the promise file):

| file | rows | columns |
|---|---|---|
| `01_PAYMENT_FAILURE.xlsx` | 17 | customer_id, amount, currency, txn_date, status, reason, transaction_id |
| `02_CHECKOUT_ABANDONMENT.xlsx` | 12 | customer_id, amount, currency, event_time, status, checkout_stage, cart_id |
| `03_SUBSCRIPTION_FAILURE.xlsx` | 11 | customer_id, amount, currency, billing_date, status, reason, subscription_id |
| `04_INVOICE_OVERDUE.xlsx` | 12 | customer_id, amount, currency, due_date, status, reason, invoice_number |
| `05_PROMISE_TO_PAY.xlsx` | 9 | customer_id, amount, currency, arrangement_date, status, reason, promise_id |

Every customer/reference id is a deterministic, human-readable synthetic id
(`P6-<DOMAIN>-CUST-NNN`, `P6-<DOMAIN>-REF-NNN` style) — no random UUIDs, so any row
can be traced back to its designed scenario by eye.

## 3. Dataset provenance / licensing status

100% synthetic, invented for this test. No real customer, transaction, or company
data of any kind. Provenance is carried the way the existing ingestion model
actually carries it — not by inventing a new column:

- every row's stored `source_type` is `FILE_UPLOAD`, `source_ref` is the dataset id,
  `ingestion_run_id` the specific run — automatic, not something a row can fake;
- each dataset's `source_system` was explicitly set to `phase6_synthetic_test_data`
  at mapping-confirmation time (a real, existing field — `MappingConfirmationRequest
  .source_system` — used exactly as designed for this purpose);
- filenames themselves declare intent (`01_PAYMENT_FAILURE.xlsx`, not disguised as a
  real merchant export).

No external claims are made about this data; it is not represented as real-world
data anywhere in this report, the code, or the UI.

## 4. Scenario coverage matrix

| Scenario | Payment | Checkout | Subscription | Invoice | P2P | Verified |
|---|---|---|---|---|---|---|
| Normal successful ingestion | ✅ | ✅ | ✅ | ✅ | ✅ | live ingestion, §7 |
| Repeated customer, different obligation | ✅ | — | — | — | — | live ingestion |
| Repeated obligation / multiple events, one case | ✅ | ✅ (dup row) | — | ✅ | — | `cases_updated` counters, §7 |
| Duplicate event (exact repeat) | — | ✅ | — | — | — | occurrence-numbered, kept + correlated |
| Duplicate file / re-upload | ✅ | ✅ | ✅ | ✅ | ✅ | `cases_created: 0` on re-upload, §9 |
| Failed payment | ✅ | — | — | — | — | live |
| Retryable failure (network_error) | ✅ | — | ✅ | — | — | diagnosis: NETWORK_FAILURE |
| Non-retryable failure (expired_card) | ✅ | — | ✅ | — | — | diagnosis: PAYMENT_METHOD_INVALID |
| High-value escalation (>₹5000) | ✅ | ✅ | ✅ | ✅ | ✅ | live policy_status=ESCALATE, §6 |
| Consent allowed | ✅ | — | — | — | — | consent import, §6 |
| Consent denied (OPTED_OUT) | — | — | ✅ | — | ✅ | consent import |
| Consent unknown (no record) | ✅ | ✅ | ✅ | ✅ | ✅ | the majority default case |
| Suppressed customer (channel: all) | — | ✅ | ✅ | — | — | §6, §14, browser-verified |
| Channel-specific restriction (payment only) | — | — | ✅ | — | — | consent import |
| Full recovery | ✅ | ✅ | — | ✅ | ✅ | live outcomes, §8 |
| Partial recovery | — | — | — | ✅ | — | live, still outstanding tracked |
| Still outstanding (no outcome posted) | ✅ | — | ✅ | — | — | financial invariants, §10 |
| Cancelled obligation | — | ✅ | ✅ | — | — | §8, browser-verified, §14 |
| Outcome arriving after action | ✅ | ✅ | ✅ | ✅ | ✅ | the normal case |
| Outcome arriving before action | — | — | — | ✅ | — | §8, execution_status was null pre-outcome |
| Duplicate outcome | — | — | — | ✅ | — | replay -> `status: duplicate`, §8 |
| Out-of-order events | — | — | ✅ | — | — | later-occurring posted first, §8 |
| Unmatched outcome | — | ✅ | — | ✅ (perf-test) | ✅ | §10, dashboard `unmatched_outcome_events` |
| Multiple currencies | ✅ (USD/INR) | ✅ (USD/INR) | ✅ (USD/INR) | ✅ (USD) | ✅ (USD) | dashboard currency_breakdown, §10 |
| Malformed/invalid rows (mixed in file) | ✅ (4) | ✅ (2) | ✅ (1) | ✅ (1) | ✅ (1) | §7, per-row rejection reasons |
| Promise CREATED→DUE→FULFILLED | — | — | — | — | ✅ | §14, browser-verified |
| Promise CREATED→DUE→BROKEN | — | — | — | — | ✅ | §14 |

## 5. Ingestion results (Step 7)

Every dataset was pushed through the real path: `POST /datasets/upload` → `POST
/datasets/{id}/analyze` → `POST /datasets/{id}/mapping` (explicit column mapping +
declared domain + `source_system`) → `POST /datasets/{id}/ml-readiness` → `POST
/datasets/{id}/generate-cases`.

| dataset | rows | accepted | rejected (reason breakdown) | not-at-risk | cases created | cases updated |
|---|---|---|---|---|---|---|
| 01_PAYMENT_FAILURE | 17 | 11 | invalid_amount 1, invalid_entity 1, ambiguous_target 1, invalid_currency 1 | 1 | 11 | 1 |
| 02_CHECKOUT_ABANDONMENT | 12 | 8 | invalid_amount 1, invalid_timestamp 1 | 1 | 8 | 1 |
| 03_SUBSCRIPTION_FAILURE | 11 | 10 | invalid_amount 1 | 0 | 10 | 0 |
| 04_INVOICE_OVERDUE | 12 | 10 | invalid_amount 1 | 0 | 10 | 1 |
| 05_PROMISE_TO_PAY | 9 | 8 | ambiguous_target 1 | 0 | 8 | 0 |
| **Total** | **61** | **47** | **9** | **2** | **47** | **3** |

Every rejection reason matches the deliberately-planted invalid row exactly (missing
amount, non-ISO currency, ambiguous status text, missing customer id, unparseable
date, an amount of `1e14` exceeding `MAX_AMOUNT`) — no unexplained rejections, no
row silently dropped without a reason.

## 6. Domain classification results

All 5 datasets were classified via **DECLARED** (dataset-level `declared_domain`
passed to `POST /datasets/{id}/mapping`) — the deliberately explicit, safest path.
No row was misclassified into another domain; no `unresolved_domain` rejection
occurred in the clean domain files (the one `ambiguous_target` reject in each of
01/05 was a deliberately invalid `status` value, not a domain misclassification).

Domain-scoped outcome-type enforcement was also verified: a real, additional check
confirmed `OUTCOME_TYPE_DOMAIN_MISMATCH` still applies from Phase 5 (not re-run
explicitly this phase — it is unchanged, covered by `test_phase39_outcome_lifecycle
.py::test_B`).

## 7. Deduplication results (Steps 3, 9)

- **Within-file duplicate row** (checkout cart `P6-CO-CART-0003` appears twice,
  byte-identical): both rows are kept as genuinely repeated events (occurrence 0 and
  1 of the same fingerprint — by design, since two indistinguishable real failures
  in one file are both real events), and both correlate onto the **same** case via
  obligation identity (`cases_updated: 1`), never a second case.
- **Repeated obligation, different content** (invoice `P6-INV-NO-0003` appears twice
  with a later timestamp — a second overdue reminder for the same invoice): also
  correlates onto the same case (`cases_updated: 1`).
- **Re-upload of the identical file** (all 5 files, re-uploaded through the full
  pipeline a second time):

| dataset | rows | accepted on re-upload | duplicate_event | cases_created |
|---|---|---|---|---|
| 01_PAYMENT_FAILURE | 17 | 0 | 12 | **0** |
| 02_CHECKOUT_ABANDONMENT | 12 | 0 | 9 | **0** |
| 03_SUBSCRIPTION_FAILURE | 11 | 0 | 10 | **0** |
| 04_INVOICE_OVERDUE | 12 | 0 | 11 | **0** |
| 05_PROMISE_TO_PAY | 9 | 0 | 8 | **0** |

Zero new cases on re-upload, across every domain, with rejection counts for the
still-invalid rows identical to the first pass (the same bad rows are rejected the
same way every time, not accepted the second time around). The same invariant was
also captured as a fast regression test (`test_phase40..._scenarios.py::test_E`).

## 8. Outcome / reconciliation results (Step 5)

All posted via `POST /outcomes` against the live obligations created above:

| scenario | result |
|---|---|
| Full recovery (invoice, checkout, payment, promise) | each case reached `FULLY_RECOVERED` with `recovered_amount == amount_at_risk`, `outstanding_amount == 0` |
| Partial recovery (invoice `P6-INV-NO-0003`, ₹4000 owed, ₹2500 settled) | `PARTIALLY_RECOVERED`, `outstanding_amount == 1500.0000` |
| Duplicate outcome (same `external_event_id` replayed) | second call returned `status: duplicate`; `recovered_amount` unchanged (`5000.0100`, exactly one settlement's worth) |
| Out-of-order receipt (subscription `P6-SUB-ID-0001`, ₹999 total: the chronologically-later ₹400 chunk posted **first**, the earlier ₹599 chunk posted **second**) | final state `FULLY_RECOVERED`, `recovered_amount == 999.0000` — receipt order had no effect on the correct total |
| Cancellation (checkout `ORDER_CANCELLED`, subscription `SUBSCRIPTION_CANCELLED`) | both reached `CaseState.CANCELLED`, `cancelled_amount == amount_at_risk`, `recovered_amount == 0` |
| Outcome arriving before any action (invoice `P6-INV-NO-0008`, ₹15,500 — an escalated case never approved/executed) | `execution_status` was confirmed `null` **before** the outcome was posted; the outcome still matched and applied, reaching `FULLY_RECOVERED` without any execution ever having occurred — direct proof that recovery does not depend on, or require, an action having run |
| Unmatched outcome (bogus reference/customer, checkout + promise domains) | both recorded as `match_status: UNMATCHED, match_reason: NO_OBLIGATION`, `case_id: null` — visible in `GET /outcomes?match_status=UNMATCHED` and in `dashboard.unmatched_outcome_events`, never silently dropped or misapplied |

## 9. Consent / policy results (Step 6)

- Consent import (`POST /contact-permissions/import`): 5 records imported across
  scenarios (ALLOWED, OPTED_OUT ×2, SUPPRESSED ×1, plus the suppression added
  specifically for the regression test), 0 rejected.
- **Human approval does not bypass suppression** — the critical safety property —
  was tested twice:
  1. An initial attempt used a customer whose case had *already auto-executed*
     under `UNKNOWN` consent before the suppression record was imported; the
     later `/execute` call correctly returned the **cached prior execution**
     (idempotency replay), which is correct behaviour but does not exercise
     revalidation. Documented here as a methodology correction, not a defect.
  2. The **real test**: `P6-SUB-CUST-004`'s case (₹6,200, `ESCALATE`, not yet
     executed) had a customer-wide `SUPPRESSED` record imported, was then
     **approved by a human** (`policy_status` flipped `ESCALATE → PERMITTED`),
     and only then had `POST /cases/{id}/execute` called for the first time.
     Result: `status: REJECTED`, `result_metadata.revalidated_status: DENIED`,
     reason `"Customer is SUPPRESSED for payment..."`. Human approval satisfied
     only the amount-threshold rule; the fresh policy revalidation immediately
     before the side effect still caught the suppression and refused to act.
     Confirmed in the browser: the case page shows Contact permission
     "Payment retries: Suppressed", Action taken "REJECTED", Result "Not run yet."
- This is now also a permanent regression test:
  `test_phase40_phase6_scenarios.py::test_A`.

## 10. Financial invariant results (Step 10)

19 obligations spanning all 5 domains were pulled via `GET /obligations` and
checked programmatically:

```
recovered >= 0, outstanding >= 0, cancelled >= 0        — 0 failures
recovered <= amount_at_risk                              — 0 failures
recovered + outstanding + cancelled == amount_at_risk     — 0 failures (exact, to the cent)
```

**Zero invariant failures** across every sampled obligation. Additionally:

- Duplicate settlement did not increase recovered money (§8).
- Unmatched outcomes are excluded from `verified_recovery` — confirmed via live
  dashboard read (`unmatched_outcome_amount: 410.0000` tracked separately from
  `verified_recovery: 32648.01`).
- Cancelled obligations never appear as recovered (`cancelled_amount` is a
  separate bucket from `recovered_amount` everywhere it is surfaced).
- Mixed currencies were never summed: the dashboard's top-level figures are
  algebraically consistent only because `currency_breakdown.USD` and
  `.INR` are reported and summed *separately*; no single blended total exists
  anywhere in the API or UI.
- Insights/dashboard totals reconcile with the authoritative case data: every
  number rendered by `Insights.tsx` in the browser check (§13) matched the API
  response used to compute it, to the cent.

## 11. Performance results (Step 12)

- Ingesting all 5 datasets (61 rows) through the full pipeline (upload → analyze →
  mapping → ml-readiness → generate-cases, with the deterministic risk/diagnose/
  predict/recommend/policy/execute pipeline run per accepted row) **and** re-uploading
  all 5 a second time (10 full pipeline invocations, 122 total row-evaluations):
  **1.79 seconds wall-clock**, on SQLite, on the same machine already running the
  backend, frontend dev server and browser session.
- Single outcome-event ingestion (measured in Phase 5, unchanged this phase):
  ~15ms median.
- No bottleneck was found at this scale. No queue, async worker, or new
  infrastructure was added or is currently justified — the plain synchronous
  FastAPI+SQLAlchemy path comfortably handles a realistic merchant-batch size.
  A genuinely large-scale (tens of thousands of rows) load test was **not**
  performed this phase (see §16 Remaining Risks) — this measurement establishes
  the shape of the cost (roughly linear per row, dominated by the deterministic
  pipeline, not I/O), not an upper bound.

## 12. Browser verification (Step 13/14)

Performed against the live dev server (`http://localhost:5173`, backend at
`127.0.0.1:8000`), on the same migrated + Phase-6-populated dev database:

- **Insights page**: case count (396), dataset count (12), currency breakdown
  (INR/USD), the "23 outcome events (₹410.00) could not be matched" note, the new
  **"Cancelled: 2 cases"** bar, and — for the first time in this project's browser
  verification — **"Broken promise to pay"** appearing as its own line in "Where
  the risk sits," proving the fifth domain is genuinely distinct end-to-end, not
  just at the schema level.
- **Suppressed case** (`P6-SUB-CUST-004`): "Payment retries: Suppressed", "Action
  taken: REJECTED", "Result: Not run yet" — matches the API exactly.
- **Promise fulfilled case** (`P6-P2P-CUST-001`): "Recovered — ₹1,800.00 of
  ₹1,800.00 recovered", with distinct `outcome_reconciled` and `outcome_recorded`
  timeline entries (the settlement and the promise-tracking record, kept separate).
- **Cancelled case** (`P6-SUB-CUST-008`): "Cancelled — ₹0.00 of ₹149.00 recovered"
  — visibly distinct from both "Recovered" and "Not recovered," never conflated.
- No UI changes were required or made this phase (Step 14 — the existing "Result"
  line, outcome verdict pills, and Insights aggregation already correctly
  represent every new scenario; nothing needed fixing).

## 13. Tests and exact counts (Step 13)

- New file: `backend/tests/test_phase40_phase6_scenarios.py` — 6 tests (A–F),
  covering suppression-cannot-be-bypassed, PROMISE_CREATED/CANCELLATION amount
  semantics, out-of-order settlement correctness, mixed-validity file ingestion,
  re-upload dedup, and duplicate-outcome idempotency. All real API/DB calls, no
  mocks of the module under test.
- **Full backend suite: 331 passed, 0 failed** (325 pre-existing Phase 1–5 tests +
  6 new Phase 6 tests). No existing test was weakened, skipped, or deleted.
- `ruff check .`: All checks passed. `ruff format`: 1 file reformatted (the new
  test file itself, cosmetic), 95 already clean.
- Frontend: `tsc --noEmit` clean, `prettier --check` clean, `eslint` 0 errors / 13
  warnings (identical, pre-existing warning count — unchanged from Phase 5).

## 14. Defects discovered

1. **Documentation defect** (not a code defect): the Phase 5-carried-forward dataset
   contract incorrectly stated outcome `amount` was required for
   SETTLEMENT/CANCELLATION and omitted for PROMISE_*. The actual code requires it
   for SETTLEMENT and PROMISE_CREATED, never for CANCELLATION. Caught immediately
   when two `PROMISE_CREATED` posts were rejected with `missing_amount` during
   Step 5 testing.
2. No code defect was found in the reconciliation, matching, consent, policy,
   dedup, or aggregation logic. Every scenario in the coverage matrix behaved
   exactly as the Phase 5 architecture specifies.

## 15. Defects fixed

1. Corrected `docs/history/phase6_dataset_contract.md`'s outcome-amount
   requirement table to match the live code, with an explicit note explaining the
   correction and what it was verified against (`reconcile_position()`'s
   `CANCELLATION` branch).
2. Re-posted the two `PROMISE_CREATED` outcome events with the correct `amount`
   field — both then matched and recorded correctly (§8, promise lifecycle).

No backend or frontend source file required a change this phase. Phase 5's
architecture held up against 61 realistic rows across 5 domains and ~35 distinct
scenario probes without needing a single line of production code changed.

## 16. Remaining risks

- **Scale**: only 61 rows (and a further ~35 API-level outcome/consent calls)
  were tested. A genuinely large batch (10,000+ rows) has not been load-tested;
  §11's measurement is a shape, not a ceiling.
- **Currency-mismatch outcome scenario**: not explicitly re-exercised this phase
  (it was verified in Phase 5's test suite and is unchanged) — worth a dedicated
  Phase 6-style dataset row in a future pass if multi-currency mismatches become
  a real merchant concern.
- **Reopening policy for late settlements** against non-final closed states
  remains the same open item flagged in the Phase 5 report (§7 there) — not
  re-tested or resolved this phase, since Phase 6's brief was validation, not
  architecture change.
- **`outcomes_count` staleness** in `GET /system/summary` (flagged in Phase 5,
  §35) remains unfixed and undocumented-in-code (documented only in these
  reports) — still counts legacy simulated-outcome rows.
- **PostgreSQL**: still UNKNOWN — not re-checked this phase (no new tooling
  appeared in the environment); carried forward from Phase 4/5.
- **The one methodology mistake in §9** (testing suppression against an
  already-executed case) is now understood and corrected, but it's a reminder
  that "did the safety check run" and "did the safety check matter" are
  different questions — a future scenario-design pass should keep drawing that
  distinction deliberately.

## 17. What is now VERIFIED

- The dataset contract's completeness against all 14 Phase 6 scenario
  categories, across all 5 domains, using genuinely different, realistic files.
- Deterministic domain classification (declared-domain path) with zero
  misclassifications.
- Row-level and file-level deduplication, including on real re-upload, with zero
  duplicate cases created.
- Correlation of multiple events (including an exact within-file repeat) onto a
  single obligation/case, never a duplicate case.
- Deterministic outcome-to-obligation matching, including explicit unmatched
  cases surfaced and excluded from recovered revenue.
- Idempotent reconciliation under real duplicate-outcome replay.
- Order-independent reconciliation under out-of-order outcome receipt.
- Full recovery, partial recovery (with a persisted outstanding balance), and
  cancellation as three genuinely distinct terminal/non-terminal outcomes.
- An outcome arriving before any action ever executed still reconciles
  correctly — proving recovery is independent of execution having occurred.
- The complete promise-to-pay lifecycle (CREATED → DUE → FULFILLED and
  CREATED → DUE → BROKEN), with promised amounts never counted as recovered
  money until an actual settlement outcome arrives.
- Consent import, including a customer-wide suppression that a human's
  threshold approval **cannot** bypass — caught by live execute-time policy
  revalidation, not merely by a documented intention.
- Every financial invariant in §10, on real data, with zero failures.
- Dashboard/Insights aggregation reconciling exactly with authoritative case
  data, in the browser, including the newly-populated fifth domain.
- Full regression: 331/331 backend tests, clean frontend typecheck/lint/format.

## 18. What remains UNKNOWN

- PostgreSQL behaviour (no server available in this environment).
- Behaviour at 10,000+ row scale (not load-tested).
- Currency-mismatch outcome rejection under Phase 6-style realistic data
  (covered by Phase 5's test suite, not re-run against these new datasets).
- Whether a merchant's real data ever produces column-mapping ambiguity this
  synthetic, deliberately-clean-per-column data can't surface (real spreadsheets
  are messier than any of the five files built here).

## 19. GO / NO-GO for Phase 7

**GO.** No STOP condition was triggered:

- Every dataset was representable by the current domain model — no undocumented
  special-case logic was needed anywhere.
- Every outcome matched deterministically or was explicitly, visibly UNMATCHED —
  never guessed.
- Recovered revenue was never fabricated: every recovered figure traces to a real
  `SETTLEMENT` outcome event with `Effect.APPLIED`.
- Duplicate outcomes never double-counted (verified live and by regression test).
- Consent was never bypassed — proven under the harder condition (human approval
  granted, then execution still blocked).
- Mixed currencies were never aggregated together, anywhere.
- No domain was silently misclassified.
- No schema redesign was required.

Per the explicit instruction, **no Phase 7 work has been started.** This report
is the end of Phase 6.
