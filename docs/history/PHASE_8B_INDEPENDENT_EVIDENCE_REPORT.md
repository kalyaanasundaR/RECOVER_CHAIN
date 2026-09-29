# Phase 8B — Independent Evidence Acquisition & Data Quality

**No model was trained, fine-tuned, evaluated, or deployed in this phase.**
Companion documents: [PHASE_8B_SOURCE_IMPORT_CONTRACTS.md](PHASE_8B_SOURCE_IMPORT_CONTRACTS.md),
the five [external-source-simulated datasets](phase8b_external_source_datasets/).

## 1. Current evidence architecture (Step 1 audit)

Traced end to end, nothing found broken or in need of redesign:

```
source (file / API caller)
  -> POST /outcomes or /outcomes/batch  (application/outcome_events.py, unchanged since Phase 5)
  -> validate_outcome_item()             (canonical: amount, currency, timestamp, domain, outcome_type)
  -> normalize_provenance()               (application/outcome_provenance.py, Phase 8A/8B)
  -> find_matching_case()                 (deterministic domain+customer+reference match)
  -> reconcile_position()                 (pure function, Phase 5)
  -> OutcomeEventModel row (append-only) + case.recovered/outstanding/cancelled_amount
  -> run_quality_checks() / build_data_readiness_report()   (application/data_readiness.py, Phase 8A)
```

`POST /cases/{id}/observed-cause` is the separate, parallel path for root-cause
evidence (`application/cause_observations.py`), intentionally **not** merged
into the outcome path — a cause observation is not money and must never be
confused with one (unchanged design principle from Phase 5's domain model).

Nothing in this architecture was redesigned. Phase 8B's job was closing three
concrete gaps found in it:

1. **Provenance could be claimed, not verified** — Phase 8A defined the
   `production`/`payment_provider`/etc. vocabulary but never checked whether
   a caller was entitled to claim it. Closed (§5).
2. **No distinct category for "shaped like a real source but not real"** —
   only `test` existed; a genuinely external-source-shaped simulation had
   nowhere honest to live except `test`, which undersells what it's for.
   Closed (§2, §4).
3. **No mechanism to acquire real root-cause evidence at all** — Phase 8A
   built the *schema* (`cause_observations`) but the only usable source was
   `human_confirmed`; other listed sources were accepted without any gate.
   Closed (§6).

## 2. Trusted-source definitions (Step 2)

| Task | Trusted source(s) | Reachable today? |
|---|---|---|
| PAYMENT | payment-provider settlement/webhook export | No — reserved (`payment_provider`), no connector exists |
| CHECKOUT | checkout/order conversion record | No — reserved (`checkout_system`) |
| SUBSCRIPTION | billing/subscription provider event | No — reserved (`subscription_system`) |
| INVOICE | accounting/ERP settlement record | No — reserved (`invoice_system`) |
| PROMISE-TO-PAY | receivables/payment record confirming fulfillment/breach | No — reserved (no dedicated provenance value; would fall under whichever real payment/receivables system eventually reports it) |
| ROOT CAUSE | independent human observation | **Yes** — `human_confirmed` |
| ROOT CAUSE | independent operational classification (a real gateway/checkout/subscription/invoice system's own reason code) | No — reserved (`payment_gateway_response` etc.) |

**A source is never called "production" merely because a user uploaded it.**
This is enforced structurally, not by policy statement: `ACCEPTED_VIA_SHARED_
API_KEY` (`application/outcome_provenance.py`) is the actual gate — see §5.
An uploaded file, however well-formatted, can only ever be accepted as
`test`, `human_confirmed`, or (new this phase) `external_source_simulated`.

## 3. Source ingestion contracts (Step 3)

Full field-by-field mapping for all 5 simulated sources:
[PHASE_8B_SOURCE_IMPORT_CONTRACTS.md](PHASE_8B_SOURCE_IMPORT_CONTRACTS.md).
Every source maps onto the **same** canonical `POST /outcomes` contract —
no new ingestion architecture was built, matching the instruction not to
create five unrelated pipelines. Only the column-name mapping differs per
source, because that's genuinely what each kind of export looks like.

## 4. External-source-simulation datasets (Step 4)

Five CSVs in [`phase8b_external_source_datasets/`](phase8b_external_source_datasets/),
one per source type, each with its own native-looking column names and its
own independently-generated event-id namespace (`psp_sim_*`, `chk_sim_*`,
`sub_sim_*`, `erp_sim_*`, `p2p_sim_*` — never reusing any existing Phase
5/6/8A test id). Each targets real, pre-existing Phase 6 obligations (still
outstanding) plus one deliberately unmatched row.

**Live import result** (`POST /outcomes` per row, real API, real DB):

```
total rows posted:  20
matched:             15  (3 per source x 5 sources)
unmatched:            5  (1 deliberate bad reference per source)
validation errors:    0
```

Every matched row was stored with `provenance = external_source_simulated`
— confirmed never counted toward `production_outcomes_count` (§5, live).

## 5. Provenance integrity results (Step 5)

**Enforced, not just documented.** `application/outcome_provenance
.ACCEPTED_VIA_SHARED_API_KEY = {test, human_confirmed,
external_source_simulated}` — the only three values `POST /outcomes` will
accept from a caller holding the shared dev API key. Verified live:

```
POST /outcomes {"provenance": "production", ...}
  -> 422 "'production' requires a stronger authentication mechanism
     (a registered source, a signed webhook, or a dedicated integration
     credential) that does not exist in this deployment"

POST /outcomes {"provenance": "payment_provider", ...}
  -> 422, same reason
```

This closes a real gap Phase 8A left open: before this phase, any caller
with the shared key could type `"provenance": "production"` and have it
silently accepted and stored. Also verified live (final `GET
/system/data-readiness`, after all Phase 8B imports):

```
production_outcomes_count:               0
test_outcomes_count:                     16
external_source_simulated_outcomes_count: 16
human_confirmed_outcomes_count:           0
provenance_breakdown: {"external_source_simulated": 16, "test": 16}
```

TEST ≠ EXTERNAL_SOURCE_SIMULATED ≠ HUMAN_CONFIRMED ≠ PRODUCTION are four
genuinely distinct, independently-visible buckets everywhere this report is
read (API, and now — for a fresh outcome — the case detail UI, §14).
Regression tests: `test_phase43...::test_1_*`, `test_2_*`.

## 6. Independent root-cause evidence (Step 6)

`cause_observations` (Phase 8A) gains, this phase: `evidence_reference`
(a pointer to what the observation was actually based on — a ticket id, a
call-log reference) and `confidence` (optional, 0.0–1.0). `recorded_by`
already served as observer/integration identity. The source vocabulary is
now gated exactly like outcome provenance: `ACCEPTED_VIA_SHARED_API_KEY =
{human_confirmed, external_source_simulated}`; the four per-integration
sources (`payment_gateway_response`, `checkout_system_event`,
`subscription_billing_event`, `invoice_payment_history`) are real vocabulary
values, reserved, **rejected** until a real connector exists — verified
live and by regression test that `"diagnosis_engine"` and `"rule"` are
rejected identically to the reserved integration sources (never treated as
independent).

Live demonstration:

```
POST /cases/case_a37fa855/observed-cause
  {"observed_cause_category": "NETWORK_FAILURE", "source": "human_confirmed",
   "note": "merchant support confirmed the gateway timeout via call log",
   "evidence_reference": "support-ticket-4711", "confidence": 0.9}
  -> 200, stored as an independent observation

independent_observation_count (live): 1
rule_derived_diagnosis_count (live):   5896   <- must NEVER be used as a label
```

## 7. Temporal integrity results (Step 7)

Every `outcome_events` row distinguishes `occurred_at` (event time),
`ingested_at` (ingestion time) — unchanged from Phase 5 — and every
`cause_observations` row has its own `recorded_at` (observation time),
distinct from both. Tested live and by regression:

- **Duplicate**: exact `external_event_id` replay -> `status: duplicate`,
  not re-applied (test 4).
- **Out-of-order**: a later-occurring settlement posted first, an
  earlier-occurring one posted second -> both apply, correct final total
  regardless of receipt order (test 5).
- **Late**: an `occurred_at` far in the past relative to ingestion is
  accepted (a genuinely late report is not itself invalid) but **flagged**
  if it predates the case's own existence — a real, non-hypothetical
  finding: 30 of the current 32 real outcome rows are flagged
  `outcome_occurred_before_case_existed`, because both the Phase 6 dataset
  dates and this phase's own simulated-source dates were authored relative
  to the dataset's fictional timeline, not to actual ingestion wall-clock
  time. Left exactly as found — nothing was silently repaired or
  regenerated to hide it (test 6).
- **Same obligation, multiple outcomes**: already covered by Phase 5/6/8A's
  partial/full/cancellation scenarios; unchanged, re-verified this phase via
  the multi-provenance settlements posted against Phase 6 obligations.
- **Future information as a feature**: unchanged from Phase 7's leakage
  audit — `FeatureExtractor` still only reads risk/diagnosis/amount/age,
  none of which can carry a later outcome's information; re-verified
  (`test_phase41::test_4` still passes unmodified).

## 8. Training-data eligibility rules (Step 8)

`application/eligibility.py::evaluate_case_eligibility()` — a case with at
least one MATCHED outcome is checked against: provenance (every touching
outcome must be real-world), duplicate/quality-issue status (via the live
quality checks), obligation identity (reference_id present), currency
completeness, financial-position reconciliation, required-feature
availability (diagnosis + risk_assessment present), and terminal-state
label availability. Returns `ELIGIBLE` / `INELIGIBLE` /
`INSUFFICIENT_EVIDENCE` with the concrete reasons.

**Live result** (28 cases evaluated — every case with a real matched
outcome, across the full dev DB including this phase's imports):

```
ELIGIBLE:              0
INELIGIBLE:            28   (all: non-production provenance, mostly also a
                              temporal quality issue)
```

Verified the classifier is not permanently stuck by construction, only by
current evidence: a fresh case settled with `provenance=human_confirmed`,
complete decision lifecycle, and a genuinely current `occurred_at` **is**
reachable as `ELIGIBLE` (`test_phase43::test_7_8`, passing live).

## 9. Dataset versioning (Step 9)

`ml_dataset_snapshots` (new table): `dataset_id`, `dataset_name`+`version`
(unique, never reused), `created_at`, `eligible_case_count`,
`provenance_distribution`, `time_range_start/end`, `feature_schema_version`,
`label_schema_version`, `exclusion_counts`, `eligible_case_ids` (the frozen
list), `checksum` (sha256 over the sorted eligible-id list + schema
versions). `POST /system/dataset-snapshots` creates one; `GET
/system/dataset-snapshots[/{id}]` reads.

Live demonstration of reproducibility:

```
snapshot 1: version=1, eligible_case_count=0, checksum=75e810...3570
snapshot 2 (same dataset_name, re-run): version=2, checksum=75e810...3570 (IDENTICAL)
```

Version always increments, the row is never overwritten, and the checksum
is bit-for-bit identical when the underlying eligible set hasn't changed —
the actual reproducibility guarantee, not merely "a hash exists." Verified
by regression test that this also holds when called twice in the same
process (`test_phase43::test_9_10`).

## 10. Scale benchmark (Step 11)

Two real ingestion paths were measured; **no infrastructure was added, and
no production-scale claim is made.**

**Path A — `POST /datasets/{id}/generate-cases` (the full dataset-lab
pipeline, per-row risk/diagnose/predict/recommend/policy/execute), 5,000-row
file:**

```
upload:            0.067s
analyze:            1.107s
mapping:            0.089s
ml-readiness:       0.661s
generate-cases:     7.375s for 500 rows  <- see finding below
rows/sec:          67.8 (of the rows actually processed)
db growth:          +3.63 MB
```

**Real, measured bottleneck found — not previously known**:
`api/dataset_router.py` hard-caps `generate-cases` at
`bounded_max_cases = min(max(1, req.max_cases), 500)` **regardless of what
`max_cases` the caller requests.** Of the 5,000-row file, only the first
500 rows were processed per call; the response correctly and honestly
reports this (`truncated: true`, `rows_not_processed: 4500`) — it is not
silently hidden, but it does mean a single `generate-cases` call cannot
process more than 500 rows no matter how it's invoked. This pre-dates Phase
8B (not introduced this phase) and was not changed — per the brief's
instruction to report a measured bottleneck rather than prematurely
introducing infrastructure, and because raising an internal-safety cap on a
per-row-simulation-heavy endpoint is a product decision outside this
phase's scope.

**Path B — `POST /events/batch` (the uncapped ingestion path,
`auto_advance=true` running the same deterministic pipeline per case),
5,000 distinct events:**

```
elapsed:           127.77s
throughput:         39.13 rows/sec
ingested:            5000, duplicates: 0, failed: 0
db growth:          +30.9 MB (7.35 MB -> 38.25 MB)
```

**Duplicate-handling re-run (same 5,000 events posted again):**

```
elapsed:             7.86s
ingested:               0, duplicates: 5000, failed: 0
```

Duplicate detection is ~16x faster than first ingestion (7.86s vs 127.77s
for the same 5,000 rows), because a recognised duplicate short-circuits
before the expensive per-case deterministic pipeline runs — confirming
Phase 5's dedup-before-reconciliation design holds up at this scale.

Memory (RSS) was **not measured** — `psutil` is not installed in this
environment and installing a new dependency mid-phase for a benchmark was
judged out of scope; DB file growth is reported instead as the closest
available proxy for storage growth.

## 11. PostgreSQL status (Step 12)

**UNKNOWN**, unchanged. Re-checked: no `psql`, `pg_ctl`, or `docker` on
PATH; WSL is not installed on this machine (`wsl -l -v` reports it must be
installed, not merely "no distro"). No infrastructure was installed to
force a check.

## 12. Phase 8A carry-forward status (Step 13)

- **`outcomes_count` consistency**: re-verified. `GET /system/summary`
  still separates the legacy `outcomes_count` (case-level, includes
  pre-Phase-5 simulated outcomes) from `authoritative_outcomes_count`
  (case-level, real reconciliation only) — unchanged, correct.
- **Provenance counts**: extended this phase from a 2-way (production/test)
  split to the full 4-way breakdown (§5) — a genuine improvement, not a
  defect fix.
- **Independent observation count**: working, now richer (evidence_reference
  + confidence added, §6).
- **Readiness API**: extended (temporal coverage gained an
  `external_source_simulated_only` bucket) — working correctly, verified
  live throughout this report.
- **No mutation endpoint for authoritative outcome provenance**: re-verified
  — `outcome_events` still has no PUT/PATCH/DELETE route
  (`test_phase42::test_outcome_events_has_no_mutation_endpoint`, unmodified,
  still passing). A provenance value, once stored, cannot be edited; a
  correction would have to be a new row.

**One concrete defect found and fixed this phase** (not a Phase 8A carry-
forward item, but discovered while extending the same code):
`normalize_provenance()` previously accepted an *explicit* claim of
`"production"` at face value (Phase 8A's own test asserted this as correct
behaviour). Phase 8B's Step 5 gate makes this the security fix it always
should have been — the old test was updated to assert the new, correct,
stricter behaviour (`test_phase42::test_11`, now passing against the fixed
code, not against the old permissive behaviour).

## 13. Tests and exact counts (Step 15)

New file: `backend/tests/test_phase43_phase8b_evidence.py` — **11 tests**
covering exactly the brief's Step 15 list (provenance cannot be falsely
promoted; external-source-simulation remains non-production; independent
cause observation remains distinct; duplicate external outcomes; out-of-
order outcomes; late outcomes; training-data eligibility; temporal
eligibility; dataset snapshot reproducibility; operational/evaluation
isolation; financial invariants; multi-currency safety — 12 categories
covered across 11 test functions, two combined where the same scenario
naturally covers both). One existing Phase 8A test
(`test_phase42::test_11`) was updated to match the intentionally-tightened
provenance behaviour (§12) — no test was weakened, only corrected to match
a real security improvement.

**Full backend regression: 359 passed, 0 failed** (348 pre-existing Phase
1–8A tests + 11 new Phase 8B tests), clean run this session. `ruff check` /
`ruff format --check`: clean across the whole repo. Frontend `tsc
--noEmit`, `prettier --check`, `eslint`: clean (0 errors, 13 pre-existing
warnings, unchanged). Alembic migration round-trip (`upgrade` ->
`downgrade -1` -> `upgrade` -> `alembic check`): clean, "No new upgrade
operations detected."

## 14. Browser verification (Step 15 checkpoint)

Backend and frontend dev servers restarted fresh this phase and confirmed
working. A fresh outcome posted live with `provenance=
external_source_simulated` against a real Phase 6 obligation
(`case_811c9ac4`) renders in the case detail page as:

> **Result**: Recovered — ₹410.00 of ₹410.00 recovered **· simulated
> external-source data**

confirming the minimal Step 14 UI addition (§below) works end-to-end, not
just at the API layer. The existing advisory framing elsewhere on the page
("Chance of getting it back ~89% (a helper estimate — a person always
decides)") is unchanged.

**UI change made (Step 14)**: one small, additive badge on the existing
"Result" line (`CaseView.tsx`) and the guided workflow's equivalent note
(`S11Result.tsx`), shown only when an outcome's provenance is not
`production` — exactly the brief's stated bar ("only add minimal
provenance/evidence indicators if necessary to prevent users confusing
test/simulated evidence with production evidence"). No layout, navigation,
or visual-design change was made anywhere else. No private outcome data is
exposed — the badge shows only the provenance *category* (a closed
vocabulary label like "simulated external-source data"), never any raw
`external_system`/`external_event_id`/source-file detail.

## 15. VERIFIED

- Provenance cannot be forged: `production` and the four reserved
  integration values are rejected via the shared API key, live and by test.
- External-source-simulated data is accepted, distinctly categorized, and
  structurally excluded from `production_outcomes_count` (never merely
  "not counted" by convention — excluded by the same query set that defines
  `REAL_WORLD_VALUES`).
- Independent root-cause evidence collection works end-to-end
  (`human_confirmed`, with evidence reference + confidence), and the
  deterministic diagnosis engine's output remains structurally ineligible
  as a label regardless of its volume (5,896 rule-derived vs. 1 independent).
- Duplicate, out-of-order, and late-outcome handling all still hold at this
  phase's added scale and with the new provenance dimension.
- The eligibility classifier correctly reaches `ELIGIBLE` when real evidence
  exists, and correctly reaches `INELIGIBLE` for everything that exists
  today — it is not permanently stuck by construction.
- Dataset snapshots are immutable, versioned, and checksum-reproducible.
- Building a snapshot or running eligibility/quality checks never mutates
  an operational case or outcome row (Step 10) — verified by an explicit
  before/after case-count test.
- 5,000-row-scale ingestion is measured (two real paths), with one genuine,
  previously-undocumented bottleneck found and reported (the 500-row
  `generate-cases` cap) rather than silently worked around.
- 359/359 backend tests, clean lint/format/migration, clean frontend,
  browser-verified UI change.

## 16. UNKNOWN

- PostgreSQL behaviour (still no server available).
- Real production-scale performance (deliberately not claimed — see §10;
  this was a controlled benchmark, not a production load test).
- Memory (RSS) behaviour at scale (`psutil` unavailable this session).
- Whether the 500-row `generate-cases` cap is a deliberate product decision
  or simply never revisited — not investigated further, since changing it
  was out of this phase's scope either way.
- What genuine `production` provenance data would actually look like at
  scale — still entirely unobserved, by definition, since none exists.

## 17. BLOCKERS

1. **Zero production-provenance evidence still exists** — unchanged from
   Phase 8A, and cannot change until a real integration or a sustained
   human-confirmation workflow produces some. This phase did not, and could
   not appropriately, manufacture any.
2. **The 500-row-per-call ceiling on `generate-cases`** (§10) is a genuine
   throughput constraint for any future large-scale real ingestion via that
   specific path — `/events/batch` does not share it, but carries its own
   ~39 rows/sec ceiling from the full deterministic pipeline running
   per-case.
3. **The ML native-DLL environment issue** (Phase 7, §13) remains
   unresolved and untouched, as instructed.
4. **PostgreSQL remains UNKNOWN.**

## 18. Production-data readiness

**Not ready.** The *mechanism* for collecting trustworthy production
evidence is ready (provenance gating, quality checks, eligibility, dataset
snapshots — all live and tested); the *evidence itself* does not exist. This
phase deliberately did not, and could not honestly, change that fact — only
real merchant usage or a real integration can.

## 19. ML readiness

**Not ready**, unchanged from Phase 8A's NO-GO, now with more precise
instrumentation confirming exactly why: `production_outcomes_count = 0`,
`independent_observation_count = 1` (up from 0, but still far below any
usable volume), and every model-readiness gate reports `NOT_READY` with
real, queried reasons — no threshold was picked to make the answer come out
differently.

## 20. Final GO/NO-GO Decision for Phase 9

**GO for continued evidence-infrastructure work; NO-GO for any model
development.** This phase's own success criterion (per its explicit final
rule) does not depend on reaching a positive readiness verdict — it depends
on making the system capable of collecting trustworthy evidence without
contamination, which is demonstrated live throughout this report. Per the
explicit instruction, **no Phase 9 work has been started, and no model was
trained or fine-tuned.** This report is the end of Phase 8B.
