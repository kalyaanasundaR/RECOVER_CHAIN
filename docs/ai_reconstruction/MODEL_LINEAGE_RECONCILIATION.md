# Model Lineage Reconciliation

Forensic comparison of the two model lineages, from actual code and actual
datasets — not from naming.

| Dimension | Lineage A — Payment Failure Model | Lineage B — Recovery Model |
|---|---|---|
| Business question | Will this specific payment attempt succeed or fail? | Given an at-risk obligation, will the money actually come back within a defined window? |
| Target | Whatever column an uploaded file's own `OUTCOME` mapping points to (usually `status`) | `recovery_success` |
| Target definition | `_NOT_FAILED`/`_FAILED` vocabulary applied to that column's raw values (`api/dataset_router.py`) | `1` = at-risk amount recovered within the dataset's 30-day observation window (`data/recoverchain/manifests/recovery_outcome_manifest.json`) |
| Observation window | None — the row's own recorded status, already known at upload time | 30 days, fixed and documented |
| Features | Whatever raw columns the uploaded file happens to have (`customer_id, amount, date, status, reason` in practice) | 23 declared features: financial, historical-behavior, governance (consent/suppression), domain (see `FEATURE_CONTRACT.md`) |
| Prediction time | Training uses the WHOLE uploaded file, including rows that already show a known outcome; live inference re-scores an already-failed case's own row | Training is temporally split per customer; each obligation's rolling history uses only strictly earlier obligations of the same customer |
| Dataset | Whatever the merchant just uploaded (demo: `guide_demo.xlsx`, 147 rows) | `recovery_outcome.xlsx`, 5,935 rows, purpose-built and validated (see `RECOVERY_DATASET_VALIDATION.md`) |
| Model | Per-upload LR/XGBoost, trained automatically (`application/ml_training.py`) | Dedicated LR/RF/XGBoost ladder (`ml/training/train_recovery_model.py`), one canonical artifact |
| Calibration | `CalibratedClassifierCV` — real, fixed this session | Same mechanism, real, verified `calibrated: true` |
| Output meaning | `1 − P(payment failure)` — a complement, computed and labeled as such in `application/case_pipeline.py:87-88` | `P(recovery within 30 days | X)` — direct model output, no inversion |
| Can it be used for ERV? | Yes, and is — it's the only signal the live recommendation path has today | Yes, and now is — via the dedicated `GET /cases/{id}/recovery-probability` endpoint |
| Can it support recovery decisions? | As a proxy, with the caveat that "not-yet-failed" and "will never be recovered" are different questions | Yes, directly — but only as well as its live feature coverage allows (see below) |

## Why `1 − P(payment_failure)` is not `P(recovery)`

A payment that is *not currently failing* is not the same claim as an
obligation that *will be recovered*. Concretely: `ar_recovery_v1`'s own
documentation states every invoice in that dataset eventually settles —
there is no such thing as "never recovered" in it, only "settled late."
`billing_recovery_v3`'s target is a noisy per-cycle proxy the dataset's own
metadata admits is dominated by temporary same-cycle noise. Neither
represents a genuine recovered/not-recovered-within-a-window outcome.
Inverting a failure-risk score gives you "the model doesn't currently think
this will fail" — a statement about the payment attempt's own mechanics
(card validity, network conditions, mandate status), not about whether a
merchant will eventually see the money. The two can and do diverge: a
payment can have low failure risk today and still never be recovered (the
customer churns, disputes, or simply never retries), and a payment with
high failure risk today can still be fully recovered later (a retry after
a card update). Demonstrated live, same case, same session:

```
case_0a8a1665 (₹34,112.32, expired_card):
  Lineage A (1 - P(failure)):        recovery_probability = 0.0477
  Lineage B (canonical, coverage 26%): calibrated_recovery_probability = 0.1357
```
Nearly a 3x difference on the identical case — not noise, a genuinely
different question answered by a genuinely different model.

## Which model is canonical?

**Lineage B (`application/recovery_probability_predictor.py`,
`RecoveryProbabilityPredictor`, task `"recovery-probability"`) is the
canonical RecoverChain recovery model.** It is real, trained, calibrated,
loadable, and reachable live via `GET /cases/{id}/recovery-probability`.

**Lineage A remains, unrenamed, as a useful upstream payment-mechanics
signal** (`MLPaymentFailurePredictor`, task `"payment-failure-risk"` /
`"late-settlement-risk"`). It is not deleted, not deprecated, and its
existing behavior is completely unchanged — every one of the 633
pre-existing tests still passes.

## Why the live recommendation/ERV path was NOT fully switched to Lineage B

This is the honest, load-bearing limitation of this reconstruction pass.
Lineage B's model was trained on 23 features; this app's live feature
derivation (`application/recovery_feature_derivation.py`) can genuinely
source only 6 of them from real operational data today (`amount_at_risk`,
`age_of_obligation`, `domain`, and three rolling recovery-history
statistics computed from real `CaseModel` rows). The other 17 —
`customer_engagement`, `reminder_history`, `dispute_indicator`,
`payment_method_valid`, `historical_recovery_behavior`, consent/suppression
detail beyond what `contact_permissions` already tracks, etc. — have no
real source anywhere in this repository's data model. A case with no prior
settled history (the common case for a first-time upload) gets `feature_coverage
≈ 0.09-0.26`.

**Fabricating the other 17 features to hit "full coverage" was explicitly
prohibited by this brief, and would have been dishonest.** Passing them as
missing and letting the trained pipeline's own `SimpleImputer` handle them
(exactly what it was trained to do — 2% of the training data itself has
real missing values) is the correct, documented choice (§10 Option C of
the brief). The `feature_coverage` field on every canonical-endpoint
response makes this limitation visible to every caller, not hidden.

**Given that**, replacing the recommendation engine's primary ERV
calculation — which today runs on every case, with real, if narrow, input
— with a model that has single-digit-to-quarter feature coverage for most
cases would be a net quality regression for the live product, not an
improvement. The canonical model is real, live, and reachable; it is
deliberately not yet the sole driver of the automated recommendation path.
That would require either genuinely tracking the missing 17 features (real
product work: engagement telemetry, reminder logs, dispute records) or
accepting a second, richer upload schema specifically for recovery-history
data — both out of scope for this pass, and both noted as the real next
step rather than worked around.


## Update — live feature coverage (later pass)
The "6 of 23 / 9–26%" figures above are superseded. After auditing the ORM
(`LIVE_RECOVERY_FEATURE_COVERAGE.md`) live derivation now sources 11 features
unconditionally and 3 more when evidence exists: observed **8–12/23 (mean 9.5) on 228
real dev cases**, max 14/23. Nine features are `DATA_BLOCKED`
(`MISSING_RECOVERY_TELEMETRY.md`). The production ERV path is **still** Lineage A;
`GET /cases/{id}/shadow-erv` compares the two read-only, and `ERV_MIGRATION_CRITERIA.md`
lists what must be true before any migration.

Disagreement on real cases (`ml_reconstruction/lineage_disagreement.py`, 228 cases,
neither model ranked): mean |A−B| = 0.352, Pearson r = 0.467, Lineage A mean 0.485
(median 0.53) vs Lineage B mean 0.134 (median 0.14). Caveat: 225/228 stored Lineage A
values were the deterministic baseline, only 3 the trained shadow model — so this
mostly compares the *baseline heuristic* with the canonical model. Live accuracy /
calibration against real outcomes: **DATA_BLOCKED** (0 production-provenance outcomes).
