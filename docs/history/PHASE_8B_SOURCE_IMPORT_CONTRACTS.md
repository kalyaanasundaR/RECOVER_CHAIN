# Phase 8B — Source Import Contracts (Step 3)

Every source below maps onto the **same** canonical outcome contract
(`POST /outcomes` / `POST /outcomes/batch`, `application/outcome_events.py`,
unchanged since Phase 5) through a small column-mapping adapter. **No new
ingestion architecture was built** — this is the same reuse decision Phase 6
made for revenue-event datasets, applied to outcome evidence. Only the
column names differ per source, because that's genuinely what a payment
provider's export looks like versus a checkout platform's feed; the target
shape and validation are identical everywhere.

## Canonical target (unchanged, Phase 5)

`external_system, external_event_id, domain, outcome_type, customer_id,
reference_id, amount, currency, occurred_at, provenance` — see
`docs/history/phase6_dataset_contract.md`'s outcome-event table for the full
field-by-field contract; Phase 8B adds only `provenance` (Phase 8A) on top
of it.

## Per-source mapping

### 1. Payment-provider settlement export

| source column | -> canonical field |
|---|---|
| `psp_txn_ref` | `reference_id` |
| `merchant_customer_ref` | `customer_id` |
| `settled_amount` | `amount` |
| `settled_currency` | `currency` |
| `settlement_ts` | `occurred_at` |
| `event_uid` | `external_event_id` (prefixed `psp_sim_` for identity, §4) |

`domain = FAILED_PAYMENT`, `outcome_type = PAYMENT_SETTLED`,
`external_system = "payment_provider_simulated"`,
`provenance = external_source_simulated`.

### 2. Checkout conversion feed

| source column | -> canonical field |
|---|---|
| `order_id` | `reference_id` |
| `shopper_ref` | `customer_id` |
| `converted_total` | `amount` |
| `currency_code` | `currency` |
| `conversion_time` | `occurred_at` |
| `feed_event_id` | `external_event_id` (prefixed `chk_sim_`) |

`domain = CHECKOUT_ABANDONMENT`, `outcome_type = CHECKOUT_CONVERTED`,
`external_system = "checkout_platform_simulated"`,
`provenance = external_source_simulated`.

### 3. Subscription billing result

| source column | -> canonical field |
|---|---|
| `subscription_ref` | `reference_id` |
| `account_ref` | `customer_id` |
| `billed_amount` | `amount` |
| `currency` | `currency` |
| `billed_at` | `occurred_at` |
| `billing_event_id` | `external_event_id` (prefixed `sub_sim_`) |

`domain = FAILED_SUBSCRIPTION`, `outcome_type = RENEWAL_RECOVERED`,
`external_system = "subscription_billing_simulated"`,
`provenance = external_source_simulated`.

### 4. Invoice / ERP settlement export

| source column | -> canonical field |
|---|---|
| `invoice_no` | `reference_id` |
| `customer_code` | `customer_id` |
| `amount_paid` | `amount` |
| `currency` | `currency` |
| `paid_date` | `occurred_at` |
| `erp_txn_id` | `external_event_id` (prefixed `erp_sim_`) |

`domain = OVERDUE_INVOICE`, `outcome_type = INVOICE_SETTLED`,
`external_system = "invoice_erp_simulated"`,
`provenance = external_source_simulated`.

### 5. Receivables / promise-to-pay confirmation

| source column | -> canonical field |
|---|---|
| `arrangement_ref` | `reference_id` |
| `debtor_ref` | `customer_id` |
| `confirmed_amount` | `amount` (only when `status=fulfilled`) |
| `currency` | `currency` |
| `confirmation_date` | `occurred_at` |
| `receivables_event_id` | `external_event_id` (prefixed `p2p_sim_`) |
| `status` (`fulfilled` / `broken`) | selects `outcome_type` |

`domain = BROKEN_PROMISE`; `status=fulfilled` -> `outcome_type =
PROMISE_PAYMENT_RECEIVED` (amount required); `status=broken` -> `outcome_type
= PROMISE_BROKEN` (amount omitted — cancellation/breach outcomes never carry
one, application/outcome_events.py). `external_system =
"receivables_simulated"`, `provenance = external_source_simulated`.

## Event identity (Step 4 requirement — independently generated, never reused)

Each source's own event id (`event_uid`, `feed_event_id`,
`billing_event_id`, `erp_txn_id`, `receivables_event_id`) is prefixed by
source (`psp_sim_`, `chk_sim_`, `sub_sim_`, `erp_sim_`, `p2p_sim_`) before
becoming `external_event_id`, paired with a distinct
`external_system` per source. This guarantees these events can never
collide with — or be confused for — any id from the application's own
existing Phase 5/6/8A test fixtures (`p5_out_*`, `p6_out_*`, `p8a_out_*`,
`P6-*-TXN/CART/ID/NO/ARR-*` are *obligation references*, a different
namespace entirely, reused deliberately so these simulated settlements
target real, pre-existing Phase 6 obligations).

## Validation rules

Identical to the canonical contract — nothing source-specific was loosened
or added: `parse_amount`, `normalize_currency`, `parse_timestamp`,
`normalize_identifier`, `outcome_type_spec` domain-matching, and
`normalize_provenance`'s `ACCEPTED_VIA_SHARED_API_KEY` gate (Step 5) all
apply exactly as they do to any other `POST /outcomes` caller. One
deliberately invalid/unmatched row is included per source file (a
nonexistent reference/customer) to exercise the `UNMATCHED` path, exactly as
Phase 6 did for revenue-event ingestion.
