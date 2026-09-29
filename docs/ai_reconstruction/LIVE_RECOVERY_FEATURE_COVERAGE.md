# Live Recovery Feature Coverage

Traced from the ORM (`infrastructure/orm.py`) and implemented in
`ml/inference/recovery_features.py` (`derive_recovery_features`). Verified by
`tests/ml/test_recovery_feature_coverage.py`.

**Result: 11 features are derivable unconditionally, 3 more only when positive
evidence exists on the case, 9 are `DATA_BLOCKED`.**
Observed live on 228 real dev-DB cases: **8–12 of 23 available** (8:4 cases, 9:150, 10:59, 11:11, 12:4), after the vocabulary rules below.
Theoretical maximum with today's data: **14/23 (61%)**. 100% is not reachable
without new telemetry, and is not pursued.

Temporal rule for every historical feature: only records **strictly before** the
prediction timestamp (cutoff) **and** strictly before the case being predicted.
Outcomes are dated by their own business/audit timestamp — a prior case that
recovers *after* the cutoff is not a recovery at prediction time (tested).

| Feature | In model | In DB | At prediction time | Source | Derivable | Missing reason | Status |
|---|:-:|:-:|:-:|---|:-:|---|---|
| amount_at_risk | yes | yes | yes | `CASE` recovery_cases.amount_at_risk | yes | – | AVAILABLE |
| domain | yes | yes | yes | `CASE` risk_category | yes | – | AVAILABLE |
| age_of_obligation | yes | yes | yes | `REVENUE_EVENT` earliest event timestamp of the case → cutoff | yes | – | DERIVABLE |
| prior_recovery_rate | yes | yes | yes | `CASE` history + `OUTCOME_EVENT`(SETTLEMENT, APPLIED) + `AUDIT_RECORD` | yes | cold start → `-1.0` NO_HISTORY sentinel | DERIVABLE |
| number_of_previous_recoveries | yes | yes | yes | same | yes | cold start → 0 | DERIVABLE |
| number_of_previous_failed_attempts | yes | yes | yes | `AUDIT_RECORD` to_state=CLOSED_NOT_RECOVERED | yes | cold start → 0 | DERIVABLE |
| amount_vs_customer_average | yes | yes | yes | mean of the customer's earlier `CASE` amounts | yes | no earlier case → left missing | DERIVABLE |
| average_days_to_payment | yes | yes | yes | prior recovered cases: settlement `occurred_at` − first event timestamp | yes | no prior recovery → left missing | DERIVABLE |
| reminder_history | yes | yes | yes | `EXECUTION_RECORD` (execution_attempts, channel=communication) < cutoff | yes | – (0 is an observed absence) | DERIVABLE |
| consent_status | yes | yes | yes | `CUSTOMER` contact_permissions, `resolve_all` bitemporal at cutoff | yes | UNKNOWN = no record (an observed fact, not consent) | DERIVABLE |
| suppression_status | yes | yes | yes | contact_permissions (any channel SUPPRESSED) | yes | – | DERIVABLE |
| payment_method | yes | partly | if uploaded | `REVENUE_EVENT` raw_payload.payment_method | only if the file has it | column absent in most uploads | PARTIALLY_AVAILABLE |
| payment_method_valid | yes | partly | yes | `CASE` diagnosis.has_expired_card (positive evidence → 0) | only with evidence | absence of evidence ≠ valid, left missing | PARTIALLY_AVAILABLE |
| dispute_indicator | yes | partly | yes | diagnosis.has_dispute (positive evidence → 1) | only with evidence | absence of evidence ≠ no dispute | PARTIALLY_AVAILABLE |
| customer_tenure | yes | no | – | none (first-seen-by-RecoverChain ≠ account age) | no | no account-open date | DATA_BLOCKED |
| prior_payment_success_rate | yes | no | – | none | no | successful payments skipped at ingestion (`not_at_risk`), never persisted | DATA_BLOCKED |
| prior_late_rate | yes | no | – | none | no | no due-date/paid-date history persisted | DATA_BLOCKED |
| invoice_age | yes | no | – | none | no | invoice date not stored apart from the failure event | DATA_BLOCKED |
| customer_engagement | yes | no | – | none | no | no engagement telemetry | DATA_BLOCKED |
| recent_login | yes | no | – | none | no | no login telemetry | DATA_BLOCKED |
| previous_contact_success | yes | no | – | none | no | no reply/response tracking | DATA_BLOCKED |
| account_status | yes | no | – | none | no | no such field in any table | DATA_BLOCKED |
| historical_recovery_behavior | yes | no | – | none | no | defined at training from a latent reliability trait, not an empirical rate — deriving a bucket from `prior_recovery_rate` would silently change its meaning | DATA_BLOCKED |

23/23 accounted for. `amount_vs_customer_average` is a rolling-mean approximation of
the training feature's "customer's typical obligation" and is flagged as such in its
provenance note.

## Cold start
No settled prior obligation → `prior_recovery_rate = -1.0` (the same NO_HISTORY sentinel
the training data uses), counts = 0; `amount_vs_customer_average` and
`average_days_to_payment` stay missing. Provenance status `COLD_START`. This is **not**
a `POPULATION_PRIOR` and not customer history. No population prior is used.

## Live provenance (real case, prediction timestamp 2026-09-24T09:13:58)
`case_a5ad9931`, ₹20,578.20 — coverage 9/23. Each entry returned by
`GET /cases/{id}/recovery-probability` → `feature_provenance`:
`amount_at_risk=20578.2 OBSERVED (recovery_cases, n=1)`, `age_of_obligation=27 OBSERVED
(revenue_events earliest event)`, `prior_recovery_rate=-1.0 COLD_START (n=0 prior cases)`,
`consent_status=UNKNOWN OBSERVED (contact_permissions, bitemporal)`,
`reminder_history=0 OBSERVED`, … blocked: the nine above.

## Not derived on purpose
No random values, no mean imputation shown as history, no runtime synthetic behaviour,
no LLM-inferred values. Missing values go through the trained pipeline's own documented
`SimpleImputer` and are reported as missing in provenance.

## Vocabulary rule (integrity checkpoint)
A live categorical value outside the trained encoder's vocabulary is left missing, not mapped: an unknown `payment_method` string, and a `SUPPRESSED` payment permission for `consent_status` (that state is carried by `suppression_status`). `amount_vs_customer_average` is an approximation - see `FEATURE_CONTRACT.md`.
