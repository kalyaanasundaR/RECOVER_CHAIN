# Phase 6 Dataset Specification (written in Phase 5, per §23/§33 item 29)

This is a specification for the *next* dataset-generation step, not new
dataset files. It exists so five real workbooks can be produced later that
actually exercise the Phase 5 outcome-lifecycle machinery, which the
current 12 existing datasets do not (they only carry revenue events, not
outcome events).

## Format

CSV or XLSX, one row per event. Two workbook *kinds* are needed:

1. **Revenue-event workbooks** (existing format, unchanged) — the
   obligation-creating events. Reuse the existing Phase 1–4 column
   contract (`customer_id`, `amount`, a timestamp column, an
   outcome/status column, a reason column, plus a `reference_id` column —
   several existing sample files lack this, which is why the runtime
   re-upload test in Phase 5 §15 had to use a case that happened to carry
   one).
2. **Outcome-event workbooks** (new) — one row per external outcome,
   matching the `POST /outcomes` contract:

   | column | maps to | required |
   |---|---|---|
   | `domain` | `RiskCategory` value, e.g. `FAILED_PAYMENT` | yes |
   | `outcome_type` | one of `OUTCOME_TYPE_SPECS` keys, e.g. `PAYMENT_SETTLED` | yes |
   | `customer_id` | matches the revenue-event workbook's `customer_id` | yes |
   | `customer_system` | only when identity resolution should be exercised | no |
   | `reference_id` | matches the revenue-event workbook's `reference_id` | yes (else UNMATCHED by design) |
   | `amount` | decimal | yes for SETTLEMENT/CANCELLATION; omit for PROMISE_* |
   | `currency` | ISO code, must match the case's currency for a real match | yes |
   | `external_system` | source label, e.g. `stripe_sandbox` | yes |
   | `external_event_id` | unique per row — duplicate rows test dedup | yes |
   | `occurred_at` | ISO 8601 | yes |
   | `promised_date` | only for `PROMISE_CREATED` rows | conditional |

## The 14 coverage requirements, mapped to concrete rows to generate

1. **Payment failures + settlements** — a `FAILED_PAYMENT` revenue-event
   row, then a matching `PAYMENT_SETTLED` outcome row (full amount).
2. **Checkout abandonment + conversion** — a `CHECKOUT_ABANDONMENT` revenue
   row, then a `CHECKOUT_CONVERTED` outcome row; a second abandoned-cart
   row followed by `ORDER_CANCELLED` instead.
3. **Subscription failure + recovery/cancellation** — a
   `FAILED_SUBSCRIPTION` revenue row, one path ending in
   `RENEWAL_RECOVERED`, a second ending in `SUBSCRIPTION_CANCELLED`.
4. **Invoice overdue + partial/full settlement** — one `OVERDUE_INVOICE`
   row followed by two `INVOICE_PAYMENT_RECEIVED` outcome rows (split
   amounts summing to the invoice total, different `occurred_at`, to
   exercise partial-then-full).
5. **Promise-to-pay lifecycle** — a `BROKEN_PROMISE` revenue row, then a
   `PROMISE_CREATED` outcome row with a `promised_date`; one variant
   followed by `PROMISE_PAYMENT_RECEIVED` before the promised date
   (FULFILLED), one left with no further event past the promised date
   (BROKEN).
6. **Consent/suppression history** — a consent-import workbook (separate
   from revenue/outcome): rows of `(customer_id, channel, status, source,
   reason, effective_at)` including at least one later-dated correction to
   an earlier `effective_at`, to exercise the bitemporal `known_at` path.
7. **Duplicate/replayed events** — deliberately repeat 2–3 outcome rows
   with the identical `external_event_id` (and identical payload) to
   assert dedup; repeat 1–2 with the same reference/amount but a *new*
   `external_event_id` to assert `EXCESS` handling instead.
8. **Multi-currency** — the same domain/outcome-type pairs repeated once in
   `USD` and once in `INR`, including one deliberately mismatched pair
   (outcome currency ≠ case currency) to assert the currency-mismatch
   rejection.
9. **Invalid rows** — missing `customer_id`, missing `amount`, an
   unparseable `occurred_at`, an `outcome_type` that doesn't belong to the
   declared `domain` (e.g. `INVOICE_VOIDED` with `domain: FAILED_PAYMENT`) —
   one row per failure mode, to exercise per-item rejection reasons.
10. **Unmatched outcomes** — 2–3 outcome rows whose `reference_id` or
    `customer_id` matches nothing in the revenue-event workbook.
11. **Out-of-order events** — an outcome row whose `occurred_at` predates
    the revenue event's own timestamp (settlement recorded before the
    failure was even ingested, testing ingestion-order vs event-order
    independence), and a cancellation row arriving after a partial
    settlement row in file order but with an earlier `occurred_at`.
12. **High-value escalation** — at least one row per domain with an amount
    above whatever the current Policy Engine escalation threshold is
    (check `application/policy_engine.py` for the live figure before
    generating, since thresholds may differ by domain/currency).
13. **Retry limits** — repeated revenue events for the same obligation
    (same `reference_id`) with `FAILED_PAYMENT` outcome types occurring
    enough times to exceed the existing retry-limit policy rule, to
    confirm outcome ingestion doesn't bypass it.
14. **Communication limits** — same idea as #13 but for the
    communication-channel contact-frequency limit, using
    `CHECKOUT_ABANDONMENT`/`OVERDUE_INVOICE` rows relying on reminder
    messages.

## Row-count guidance

Small and legible beats large and unreviewable: 15–40 rows per workbook is
enough to hit every branch above at least once with a couple of repeats for
the duplicate/dedup cases specifically. This phase's own runtime
verification (Phase 5 §33) used exactly this scale (single-digit rows) and
it was sufficient to prove every mechanism.

## What NOT to generate

- No synthetic PII beyond what the existing sample datasets already use
  (fictional company names, `CUST-...`/`p2_...` style ids).
- No real external system credentials or live sandbox account references —
  `external_system` values should stay clearly fictional (e.g.
  `stripe_sandbox`, `internal_invoicing_v2`), matching the existing
  convention in this codebase's sample data.
- No attempt to simulate a genuinely high-volume event stream — Phase 5's
  own perf measurement (§29 of the Phase 5 report) showed single-request
  latency around 15 ms on SQLite at the current data scale; a dataset sized
  to "prove throughput" is not what's needed here, correctness coverage is.
