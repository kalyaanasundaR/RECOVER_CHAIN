# Phase 6 — Step 1: Authoritative Dataset Contract (from the live schema)

Audited directly from the running code, not re-invented:
`application/ingestion.py`, `api/dataset_router.py` (generate-cases path),
`application/dataset_intelligence.py` (`CanonicalField`), `application/
domain_classification.py`, `domain/revenue_domains.py`, `application/
diagnosis_engine.py`, `application/policy_engine.py`, `application/
outcome_events.py`, `api/outcome_router.py`, `api/domain_router.py`
(consent), `application/contact_permissions.py`.

## Revenue-event (obligation-creating) contract

A dataset row becomes a `RevenueEvent` only through this path: file → column
mapping to a `CanonicalField` → `generate-cases` validates with the exact
same primitives `POST /events` uses (`application/ingestion.py`).

| field | required? | type | meaning | valid examples | domain(s) |
|---|---|---|---|---|---|
| customer identifier (`CUSTOMER_ID`/`ACCOUNT_ID`/`ENTITY_ID` mapped column) | yes (if row is "at risk") | string | who owes the money | `acme_co`, `CUST-0042` | all |
| amount (`AMOUNT` or `BALANCE` mapped column) | yes | decimal string/number, > 0, < 1e13 | how much is at stake | `240.00`, `15000` | all |
| currency (a column literally named `currency`/`ccy`/`currency_code`, else `DEFAULT_CURRENCY` env, default `INR`) | no (defaults) | ISO 4217 alpha code | settlement currency | `INR`, `USD` | all |
| timestamp (`TIMESTAMP` mapped column) | yes | parseable date/datetime; naive treated as UTC | when the failure/obligation occurred | `2026-08-20`, ISO 8601 | all |
| outcome/status (`TARGET` or `OUTCOME` mapped column) | yes | one of a fixed vocabulary (see below) | did the row itself already succeed or fail | `failed`, `paid` | all — gates whether a case is even created |
| reference id (`TRANSACTION_ID` mapped column) | **no, but required for obligation correlation** | string | the domain's obligation reference (see below) | `INV-1042`, `sub_882`, `cart_77` | all — omitted ⇒ the row can never correlate with another row; each becomes its own case |
| domain (declared per-dataset, or a confirmed `EVENT_CATEGORY` column) | yes, one way or another | one of the 5 `RiskCategory` values | which of the five domains the row belongs to | `FAILED_PAYMENT` | all |
| failure/reason (any column whose name contains `reason`/`cause`/`failure_code`/`decline`/`error_code`) | no | free text, lower-cased on read | why it failed — feeds diagnosis and, combined with the other fields, event identity | `network_error`, `insufficient_funds` | mainly FAILED_PAYMENT/FAILED_SUBSCRIPTION |
| source/provenance | automatic, not a dataset column | — | every ingested row is stamped `source_type=FILE_UPLOAD`, `source_ref=<dataset_id>`, `ingestion_run_id=<run>` | — | all |

**Outcome/status vocabulary actually enforced** (`api/dataset_router.py`,
`_NOT_FAILED` / `_FAILED` sets — anything outside these two sets is
rejected as `ambiguous_target`, not guessed):

- not-at-risk (row skipped, no case): `0`, `false`, `no`, `success`, `paid`, `settled`, `completed`
- at-risk (row becomes a case): `1`, `true`, `yes`, `failed`, `unpaid`, `returned`, `declined`, `error`, `insufficient_funds`

**Obligation identity** (`domain/revenue_domains.py::obligation_key`):
`(RiskCategory, customer_id, reference_id)`. Never customer alone, never a
synthetic/positional reference, never across domains. What `reference_id`
means per domain:

| domain | `reference_id` means |
|---|---|
| FAILED_PAYMENT | the payment / transaction id of the failed charge |
| CHECKOUT_ABANDONMENT | the checkout / cart / order id |
| FAILED_SUBSCRIPTION | the subscription billing attempt / subscription id |
| OVERDUE_INVOICE | the invoice number |
| BROKEN_PROMISE | the promise / arrangement id, or the obligation it covers |

**Event identity / dedup** (`event_fingerprint_v2`): a hash of
`(customer_id, amount, currency, timestamp, reference_id, failure_code)`.
Two rows — in the same file, in a re-upload, or in a different file — that
agree on all six are the same event; a real *n*-th repeat within one file
is kept via an `occurrence` counter. Domain is deliberately **not** part of
identity (so a dataset whose domain is redeclared still recognises its own
rows).

## Outcome-event contract (`POST /outcomes`, `application/outcome_events.py`)

| field | required? | type | meaning | valid examples | domain(s) |
|---|---|---|---|---|---|
| `external_system` | yes | string | source label for this outcome feed | `stripe_sandbox`, `internal_invoicing` | all |
| `external_event_id` | yes | string, unique per `(external_system, external_event_id)` | idempotency key — an exact repeat is `status: duplicate`, never re-applied | `evt_001` | all |
| `domain` | yes | a `RiskCategory` value | which obligation type this outcome is against | `OVERDUE_INVOICE` | all |
| `outcome_type` | yes | one of the fixed per-domain vocabulary below | what happened | `INVOICE_PAYMENT_RECEIVED` | domain-specific, rejected if it belongs to another domain |
| `customer_id` | yes | string | matches the obligation's customer | — | all |
| `reference_id` | no, but required to match | string | matches the obligation's `reference_id` | — | all — omitted/wrong ⇒ `UNMATCHED / NO_OBLIGATION` |
| `customer_system` | no | string | used only when identity resolution via `customer_identity_links` is needed | — | all |
| `amount` | required for SETTLEMENT and PROMISE_CREATED; **not required** for CANCELLATION or PROMISE_BROKEN | decimal | money moved (SETTLEMENT) or the promised amount being recorded (PROMISE_CREATED) — a CANCELLATION always cancels whatever is currently outstanding, regardless of any amount supplied, so none is read | `500.00` | all |
| `currency` | required whenever `amount` is (SETTLEMENT / PROMISE_CREATED) | ISO 4217 | must equal the case's currency or the event is rejected (`CURRENCY_MISMATCH`) | `INR` | all |
| `occurred_at` | yes | ISO 8601 | when the settlement/cancellation/promise event happened at the source | — | all |
| `promised_date` | required for `PROMISE_CREATED` | ISO date | when payment was promised | — | BROKEN_PROMISE only |

Correction made during Step 5 testing: an earlier draft of this contract (carried
over from the Phase 5 report) said amount was required for "SETTLEMENT/CANCELLATION"
and omitted for "PROMISE_*". The live code (`application/outcome_events.py::
validate_outcome_item`) actually requires amount for `SETTLEMENT` **and**
`PROMISE_CREATED` (the promise's own amount), and never reads it for
`CANCELLATION` at all — verified against `reconcile_position()`'s `CANCELLATION`
branch, which cancels the full outstanding balance without referencing `amount`.
The table above reflects the corrected, code-verified contract.

**Outcome-type vocabulary** (`OUTCOME_TYPE_SPECS`):

| outcome_type | kind | domain |
|---|---|---|
| `PAYMENT_SETTLED` | SETTLEMENT | FAILED_PAYMENT |
| `CHECKOUT_CONVERTED` | SETTLEMENT | CHECKOUT_ABANDONMENT |
| `ORDER_CANCELLED` | CANCELLATION | CHECKOUT_ABANDONMENT |
| `RENEWAL_RECOVERED` | SETTLEMENT | FAILED_SUBSCRIPTION |
| `SUBSCRIPTION_CANCELLED` | CANCELLATION | FAILED_SUBSCRIPTION |
| `INVOICE_PAYMENT_RECEIVED` / `INVOICE_SETTLED` / `INVOICE_PARTIALLY_SETTLED` | SETTLEMENT | OVERDUE_INVOICE |
| `INVOICE_VOIDED` | CANCELLATION | OVERDUE_INVOICE |
| `PROMISE_CREATED` | PROMISE_CREATED (never money) | BROKEN_PROMISE |
| `PROMISE_BROKEN` | PROMISE_BROKEN (never money) | BROKEN_PROMISE |
| `PROMISE_PAYMENT_RECEIVED` | SETTLEMENT | BROKEN_PROMISE |

## Consent contract (`POST /contact-permissions/import`, `application/contact_permissions.py`)

| field | required? | type | meaning | valid examples |
|---|---|---|---|---|
| `customer_id` | yes | string | whose permission | — |
| `channel` | yes | `communication` \| `payment` \| `all` | which action channel | — |
| `status` | yes | `ALLOWED` \| `OPTED_OUT` \| `SUPPRESSED` | the recorded state — never defaulted; absent status is rejected, never assumed ALLOWED | — |
| `source` | yes | string | who/what recorded it | `customer_request`, `compliance_hold` |
| `reason` | no | free text | — | — |
| `effective_at` | no (defaults to now) | ISO 8601 | when the permission takes effect *in the real world* — bitemporal: a later-recorded correction to an earlier `effective_at` does not rewrite what an attempt already knew at that moment (`known_at`) | — |
| `source_record_id` | no | string | external record id, for provenance | — |

`SUPPRESSED` with `channel: all` is customer-wide suppression;
`channel: communication` / `channel: payment` alone is channel-specific.
Records are append-only (never edited); a change of mind is a new record.
Dedup is by `content_hash(customer_id, channel, status, source, reason,
effective_at)` — a byte-identical re-import is a duplicate, not a new row.

## Policy context relevant to dataset design

- `financial_max_automated_amount = 5000.0` — any case above this amount
  requires human approval regardless of channel/consent.
- `payment_max_retries = 3`, `payment_retry_cooldown_hours = 24`.
- `communication_max_messages_24h = 2`.
- `unknown_consent_communication` / `unknown_consent_payment` default to
  `ALLOW` (configurable) — a channel with **no permission record at all**
  resolves to `UNKNOWN`, and the *merchant policy setting* (not the absence
  itself) decides whether that's allowed, escalated, or denied.

## Diagnosis-relevant failure codes (feeds `failure_code`/`reason` column)

| failure_code | recognised as | retryable? |
|---|---|---|
| `network_error`, `timeout`, `gateway_timeout`, `processor_unavailable`, `3ds_failed` | NETWORK_FAILURE | yes — retry payment |
| `insufficient_funds`, `nsf`, `balance_too_low`, `not_enough_balance` | INSUFFICIENT_FUNDS | yes, after a delay |
| `expired_card`, `invalid_expiry`, `card_expired` | PAYMENT_METHOD_INVALID | **no** — needs the customer to update the card |
| `dispute`, `disputed`, `chargeback`, `under_review`, `contested` | UNRESOLVED_DISPUTE (invoice domain) | no — needs resolution, not a retry |
| `net_terms_pending`, `net_terms_exceeded`, `awaiting_approval`, `bank_rejected`, ... | MISSED_COMMITMENT (invoice domain) | domain-dependent |

## Summary — what the contract does NOT need invented

Every field above already exists in the live schema. No new columns, new
tables, or new validation rules were required to represent all 14 Phase 6
scenario requirements. The one true gap found: there is no canonical
`CanonicalField` for *reference id* under an obvious "REFERENCE"/"OBLIGATION_ID"
name — it is `TRANSACTION_ID`, which reads oddly for an invoice number or a
subscription id, but semantically it is exactly what obligation identity
needs, so it was used as-is (a Phase 7+ rename would be cosmetic only, not
a schema change).
