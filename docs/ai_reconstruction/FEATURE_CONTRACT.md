# Canonical Recovery Model — Feature Contract

> **Superseded status table.** The authoritative per-feature live status (23/23
> accounted for, 11 derivable + 3 partial + 9 DATA_BLOCKED) is now
> `LIVE_RECOVERY_FEATURE_COVERAGE.md`, implemented in
> `ml/inference/recovery_features.py`. The earlier table below (written when only 6
> features were derived, in the now-deleted `application/recovery_feature_derivation.py`)
> is kept for history; where they disagree, the coverage document wins. Training
> feature list: `ml/training/train_recovery_model.py` (`FEATURES`); parity is enforced by
> `tests/ml/test_recovery_feature_coverage.py::test_training_inference_feature_schema_parity`.

The canonical recovery model (`ml/training/train_recovery_model.py`,
`RecoveryProbabilityPredictor`) declares 23 features
(`application/recovery_feature_derivation.py:CANONICAL_RECOVERY_FEATURES`).
Live inference sources what it genuinely can; everything else is passed as
missing (`None`/NaN), never fabricated.

| feature_name | source (training) | source (live inference) | prediction-time availability | missing-value handling |
|---|---|---|---|---|
| `amount_at_risk` | generator: log-normal draw | `case.amount_at_risk.amount` — real | always available | never missing |
| `domain` | generator: weighted category draw | `case.risk_category.value` — real | always available | never missing |
| `age_of_obligation` | generator: `age_of_obligation` column | `now − case's own CaseModel.created_at` — real | available once the case exists | 0 default if `created_at` unresolvable |
| `prior_recovery_rate` | **rolling**: recoveries / settled obligations, strictly before this one, same customer | real `CaseModel` query: `customer_id` match, `created_at <` this case's, `current_state` in the settled sets — see `recovery_feature_derivation.py` | available once ≥1 prior settled obligation exists for the customer | `-1.0` sentinel (matches training convention) when no prior history |
| `number_of_previous_recoveries` | rolling count, same mechanism | same real DB query | same | `0` when no history |
| `number_of_previous_failed_attempts` | rolling count, same mechanism | same real DB query | same | `0` when no history |
| `prior_payment_success_rate` | generator: static per-customer trait (pre-dataset reputation) | **not derivable today** — no equivalent "onboarding reputation" field exists in the live schema | n/a | `None` → imputer's median |
| `prior_late_rate` | generator: static per-customer trait | **not derivable today** | n/a | `None` → imputer's median |
| `customer_tenure` | generator: random pool per customer | **not derivable today** — no account-open date tracked | n/a | `None` → imputer's median |
| `average_days_to_payment` | generator: reliability-correlated draw | **not derivable today** | n/a | `None` → imputer's median |
| `amount_vs_customer_average` | generator: this amount / customer's historical average | **not derivable today** (would need a customer-average query — not implemented this pass) | n/a | `None` → imputer's median |
| `invoice_age` | generator | **not derivable today** | n/a | `None` → imputer's median |
| `payment_method` / `payment_method_valid` | generator | **not derivable today** — live uploads don't reliably carry a validated payment-method field | n/a | `None` → imputer/`"missing"` category |
| `customer_engagement`, `recent_login` | generator: engagement score | **not derivable today** — no engagement telemetry in this app | n/a | `None` → imputer's median |
| `reminder_history` | generator | **not derivable today** — no reminder-send log queried | n/a | `None` → imputer's median |
| `previous_contact_success` | generator | **not derivable today** | n/a | `None` → imputer's median |
| `dispute_indicator` | generator | **not derivable today** — diagnosis has a dispute signal (`has_dispute`) but it isn't wired into this derivation yet | n/a | `None` → imputer's median |
| `account_status` | generator | **not derivable today** | n/a | `None` → imputer/`"missing"` |
| `consent_status`, `suppression_status` | generator | **derivable, not yet wired** — `application/contact_permissions.py` has real data for this; not connected to this derivation function in this pass | n/a | `None` → imputer/`"missing"` |
| `historical_recovery_behavior` | generator: categorical summary of reliability | **not derivable today** (would need to be computed from the same rolling history as `prior_recovery_rate`, bucketed — not implemented) | n/a | `None` → imputer/`"missing"` |

**Feature coverage today: 6/23 (~26%) at best (a customer with real prior
history), 2/23 (~9%) at worst (a first-time customer).** Every response
from `GET /cases/{id}/recovery-probability` reports `feature_coverage`
explicitly. This is the honest state of live derivation after this pass —
not a claim that the model runs at full fidelity.

**Real, scoped next steps** (not done this pass): wire `consent_status`/
`suppression_status` from `contact_permissions` (real data, just not
connected yet); wire `dispute_indicator` from the diagnosis engine's own
`has_dispute` signal (also real, also not connected); compute
`amount_vs_customer_average` from a customer-average query (straightforward,
not done for time). The remaining ~12 features have no real source
anywhere in this repository and would require new product instrumentation,
not a code change here.


---
# Parity audit (integrity checkpoint)

Verified by `tests/ml/test_training_inference_parity.py` and
`ml_reconstruction/feature_parity_audit.py`:

| Aspect | Result |
|---|---|
| names / order / kinds | identical across `ml/training/train_recovery_model.py`, `ml/inference/model_integrity.py`, registered metadata, the artifact's own `feature_names_in_` and transformer columns |
| numeric / categorical dtypes | numbers-or-None / strings-or-None (tested); order fixed |
| categorical vocabulary | artifact encoder categories == `TRAINED_PAYMENT_METHODS`, `TRAINED_CONSENT_STATUSES`, `RiskCategory` values; out-of-vocabulary live values are left missing, not encoded as unseen |
| preprocessing | median imputer + StandardScaler fitted on the train split (scaler mean for `amount_at_risk` equals the train-split mean); OneHotEncoder(handle_unknown=ignore) |
| missing handling | numeric NaN -> trained median (training itself only had NaN in `average_days_to_payment`); a missing categorical becomes an all-zero one-hot, which the training data never contained - so a missing categorical is *not* the same as any trained category |
| dict -> pipeline path | predictions via `predict()` equal direct pipeline predictions on 25 real training rows to 1e-12 |
| rolling history | counters equal the generator's definition step by step, cold start `-1.0` |
| cutoff | strictly before the prediction timestamp and before the case; boundary, duplicate and future-event tests pass |

**`amount_vs_customer_average` - NOT identical.** Training: `amount_at_risk /
customer_avg_amount_pool[customer]`, a fixed latent per-customer typical amount drawn
independently of the customer's actual amounts, never missing. Live: `amount_at_risk /
mean(amounts of the customer's earlier cases)`, missing at cold start (17% of rows in the
synthetic data). Between the two definitions: Pearson 0.32, Spearman 0.62 on the synthetic
data. On the synthetic validation split the model is insensitive to it (PR-AUC -0.0006 when
permuted, -0.0002 when set missing), but the label generator never uses this feature, so that
insensitivity is by construction and is not evidence about real data. The feature stays
explicitly marked as an approximation (API `approximations`, provenance note, guard test) and is
a documented migration criterion (`ERV_MIGRATION_CRITERIA.md`, item 2). Fixing it needs a new
model version (retrain with the live definition or drop the feature) - not done here.

Other definition differences (documented, not blockers on their own): `age_of_obligation` is
days since the earliest event of the case live vs an independent 0-179 draw in training;
`reminder_history` counts all communication attempts to the customer live vs a per-row Poisson
draw; live `prior_*` counts only *settled* prior cases while training counts every earlier
obligation resolved in the 30-day window; `payment_method_valid`/`dispute_indicator` are emitted
live only from positive evidence, whereas training always had a value.
