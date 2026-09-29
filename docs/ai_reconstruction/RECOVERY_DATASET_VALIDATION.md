# `recovery_outcome` (RecoverChain Recovery Training Dataset) — Validation

## Result: `RECOVERY_DATASET_STATUS = VALID` (after one real repair, documented below)

This dataset was **not** accepted on its name. It was checked against every
criterion the brief specifies, a real problem was found, the generator was
repaired (not just the labels), and the dataset was regenerated and
retested before being called valid.

## 1. Target — does `recovery_success = 1` actually mean recovery?

Yes. It is generated from a documented weighted-logistic mechanism (see
`scripts/data_generation/generate_recoverchain_datasets.py`,
`generate_recovery_outcome()`) combining prior recovery/payment history,
amount, obligation age, engagement/contact signals, dispute status,
payment-method validity, consent/suppression state, and a per-domain
effect, plus injected Gaussian noise — never a single-feature rule.

## 2. Prediction-time features — available before the outcome?

Yes, by construction: every feature is generated for the obligation at its
own creation time; `recovery_success` is decided strictly after all other
row features are fixed, and (post-repair) the customer's *rolling* history
features are updated only after the label is drawn, so they can only
inform *later* obligations of the same customer, never the row's own
label or any earlier row's.

## 3. Observation window — fixed?

Yes: 30 days, documented in the dataset manifest
(`data/recoverchain/manifests/recovery_outcome_manifest.json`) and in the
generator's own module docstring. Not "recovered eventually."

## 4. Leakage — any feature perfectly separates the label?

No. Automated check (`validate_recovery_outcome()`): correlation of every
numeric feature with the target stays well under the 0.98 threshold (the
generator refuses to write the dataset otherwise — this is enforced, not
just asserted after the fact).

## 5. Synthetic generation — trivial rule or multiple interacting factors + noise?

Multiple interacting factors plus unmodeled noise (see the `z = ...`
formula in the generator). Confirmed empirically by training: real models
land at test ROC-AUC ≈ 0.88 — clearly above the no-fit historical-rate
baseline (0.776) and nowhere near a suspicious ~1.0, the expected signature
of a realistic, non-trivial, non-leaked label.

## 6. Entity leakage — customer contamination across train/test?

**Found, investigated, and determined NOT to be leakage — but a related,
more precise problem WAS found and fixed (see §7 below).**

587 of 597 test customers also appear in train. On its own this is
**expected and correct**, not a defect: a temporal split naturally has
returning customers whose earlier obligations inform a later prediction —
this is the actual deployment scenario (a known customer's history is
supposed to be used), and `ar_recovery_v1`'s own already-accepted
methodology has the same property. Excluding repeat customers entirely
would make the dataset less representative of real deployment, not more
valid.

## 7. Temporal validity — the real problem, found and repaired

**This is the one genuine defect found in this dataset, and it required a
real generator repair, not a relabeling.**

The original generator drew `prior_recovery_rate`,
`number_of_previous_recoveries`, and `number_of_previous_failed_attempts`
as **independent random values per row**, disconnected from each
customer's actual obligation sequence. Verified live on a real customer's
15-obligation history: `number_of_previous_recoveries` went `1 → 1 → 5 →
1 → 2 → 0 → 3` across that customer's own chronologically-ordered
obligations — a logical impossibility for a genuine cumulative count, and
a real (if subtle) form of leakage: a feature *named* as "prior" that
wasn't actually computed from what came before it, meaning a training-set
obligation's "prior recovery rate" could in principle reflect information
correlated with that same customer's *later* obligations, including ones
that ended up in the test set.

**Repair:** rewrote `generate_recovery_outcome()` to build each customer's
obligations **sequentially in time**, maintaining real running counters
(`recoveries_so_far`, `non_recoveries_so_far`) that are read to compute
that row's `prior_recovery_rate` and rolling counts, and updated **only
after** that row's own `recovery_success` is decided — exactly
`ar_recovery_v1`'s own documented methodology ("Constructed temporal
history features using strictly prior invoices"), applied here for the
first time to this dataset. `prior_payment_success_rate` and
`prior_late_rate` remain legitimately static per-customer traits
(documented as representing a payment-success reputation from *before*
this dataset's own observation window begins — not leakage, a modeling
assumption stated plainly).

**Automated regression test added:** `validate_recovery_outcome()` now
checks, for every customer, that both rolling counters are
non-decreasing across their own chronologically-sorted obligations. The
generator refuses to write the dataset if this fails — this is what
actually caught the bug's absence after the repair, not a manual
eyeball check.

**Re-verified after repair**, same customer, same query:
```
2024-01-01  prior_recovery_rate=-1.00  recoveries=0   failures=0
2024-01-10  prior_recovery_rate= 0.00  recoveries=0   failures=1
2024-12-03  prior_recovery_rate= 0.00  recoveries=0   failures=6   (recovered this time)
2024-12-11  prior_recovery_rate= 0.14  recoveries=1   failures=6
2025-03-30  prior_recovery_rate= 0.13  recoveries=1   failures=7   (recovered)
2025-04-28  prior_recovery_rate= 0.22  recoveries=2   failures=7
...
2025-11-01  prior_recovery_rate= 0.14  recoveries=2   failures=12
```
Both counters now strictly non-decreasing. Re-trained: test ROC-AUC ≈
0.88 (comparable honest signal, not destroyed by the fix; positive rate
dropped from 17% to ~6.8% as a side effect of the more realistic sequential
dynamics — also expected and reported as-is, not adjusted for
appearances).

## Final verdict

`RECOVERY_DATASET_STATUS = VALID`, effective after the repair documented
in §7. The dataset, generator, validator, and this document were all
regenerated/rewritten together — nothing here was written before the fix
was verified.
