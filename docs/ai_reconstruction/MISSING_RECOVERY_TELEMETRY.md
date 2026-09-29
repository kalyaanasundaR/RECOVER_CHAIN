# Missing Recovery Telemetry (the 9 `DATA_BLOCKED` features)

Nothing below is approximated at runtime. Each row says what would have to be collected.
"Available after" is when the feature could first be computed for *new* cases — history
cannot be back-filled, and any retrain must use real (not synthetic) history.

| Feature | Why missing | Event/telemetry required | Where it would be stored | Available after |
|---|---|---|---|---|
| customer_tenure | No account-open date anywhere | Customer master `account_opened_at` from the merchant's system | new `customers` table (or a column on `customer_identity_links`) | once the merchant supplies it; then immediately for all that customer's cases |
| prior_payment_success_rate | Paid rows are dropped at ingestion (`not_at_risk`) and never stored | Persist successful payment events (customer, amount, timestamp, outcome) | new `payment_events` table fed by upload/API | after ingestion starts persisting paid rows; needs ≥ several cycles of history |
| prior_late_rate | No due-date vs paid-date per payment | Payment events carrying `due_at` and `paid_at` | same `payment_events` table | same |
| invoice_age | Invoice date not stored separately from the failure event | `invoice_date` (and `due_date`) mapped as canonical fields at ingestion | `revenue_events` column or `invoices` table | immediately for uploads that map an invoice date |
| customer_engagement | No engagement measurement | Portal/app/e-mail engagement events | new `customer_engagement_events` | after instrumentation ships |
| recent_login | No login telemetry | Login events from the merchant's app | same | after instrumentation ships |
| previous_contact_success | No reply/response tracking on communications | Delivery/open/reply/payment-after-contact events linked to `execution_attempts` | new `contact_responses` table | after a real messaging adapter reports responses (today's adapter is a simulation) |
| account_status | No account state field | Account status feed (active/dormant/closed) | customer master | same as customer_tenure |
| historical_recovery_behavior | Defined at training from a latent trait, not an empirical rate | Redefine on retrain as a bucket of the observed rolling recovery rate | training contract change (new model version) | after a retrain with the redefined feature |

Notes:
- `payment_method` / `payment_method_valid` / `dispute_indicator` are only
  PARTIALLY_AVAILABLE: they populate when the upload maps a payment-method column or
  the diagnosis engine records positive evidence. A validated payment-method record
  (with issue/expiry dates) and a dispute ledger would make them fully available.
- Contact "success" today would have to come from a real channel; `MockExecutionAdapter`
  produces no genuine response, so no honest value can exist.
