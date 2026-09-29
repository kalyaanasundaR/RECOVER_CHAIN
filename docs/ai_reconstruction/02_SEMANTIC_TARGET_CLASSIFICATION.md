# Critical Semantic Correction (§3) — what these targets actually are

RecoverChain being the project name does not make a late-payment model a
recovery model. Classifying every existing target against the brief's
vocabulary: `DIRECT_RECOVERY_TARGET / VALID_PROXY / WEAK_PROXY / UNRELATED / DATA_BLOCKED`.

## `ar_recovery_v1` / `ar_recovery_v2` — `target_late`

- **What it actually measures:** whether an invoice was settled late
  (`DaysLate > 0`) vs on time. Per its own metadata: *"ALL invoices in this
  dataset were eventually settled. There are ZERO write-offs or permanent
  non-payments."*
- **Classification: `VALID_PROXY`, not `DIRECT_RECOVERY_TARGET`.** Late
  settlement is a legitimate, measurable proxy for "this obligation is at
  elevated risk and needs attention" — but it is not recovery, because in
  this dataset *everything* eventually gets paid. There is no "did we get
  the money back or not" outcome to learn here at all.
- **Correct name for what was trained:** **Late Payment Risk Model** —
  already renamed as such throughout `docs/ai_reconstruction/01_TRAINING_RESULTS.md`
  from this session's own work. It should not be called a "Recovery
  Probability Model" anywhere.

## `billing_recovery_v1/v2/v3` — `target_recovered`

- **What it actually measures:** per its own metadata, a **per-cycle**
  proxy — 1 if every row for that subscriber-month was Paid/Paid Late, 0 if
  any row was Failed/Unpaid *in that one billing cycle*. The same file
  documents: *"89% of subscribers with Failed/Unpaid status in any month
  later show successful payment in subsequent months."*
- **Classification: `WEAK_PROXY`, trending toward `DATA_BLOCKED` as a
  usable recovery model.** It is *labeled* "recovered" but the label is
  dominated by cycle-boundary noise rather than genuine non-recovery — and
  this session's real training run confirms it empirically: every model
  (LR/RF/XGBoost) tops out at ROC-AUC 0.54-0.55, barely above random, and
  collapses to "always predict recovered" at the standard threshold. The
  dataset's own documentation predicted this before the run happened.
- **Correct name for what was (attempted to be) trained:** not usable as a
  "Recovery Probability Model" in its current form. If reported at all, it
  should be labeled **Per-Cycle Payment Failure Proxy (weak signal,
  deployment-blocked)** — never "Recovery AI."

## Is there a `DIRECT_RECOVERY_TARGET` anywhere in this repo?

**No.** A genuine recovery target needs: an obligation that was actually
at risk of never being paid, a clearly defined observation window (e.g.
"recovered within 30 days of the risk event"), and both recovered and
genuinely non-recovered examples. Neither `ar_recovery_v1` (nothing is ever
unrecovered) nor `billing_recovery_v3` (recovery is smeared across
resettling within the same noisy monthly cycle) provides that. The live
app's own operational data (`backend/dataset/`, real case outcomes via
`OutcomeEventModel`) is closer in *shape* to what's needed, but it's demo/
test-session volume (dozens to low hundreds of cases), not a trained-model-
grade corpus, and mixing it into training would blur "live operational
data" with "training data" — the brief's own §4 explicitly forbids that.

## `ACTION_CONDITIONED_RECOVERY` — formally `DATA_BLOCKED`

`P(recovery | X, action)` requires historical rows that record: the
specific action taken, when, under what policy/consent state, and the
actual outcome that followed *that specific action* — ideally with enough
action-variety per case profile to separate action effect from case
difficulty. **No dataset in this repository has that shape.** This
confirms and formalizes what Phase 17's own report already concluded:
`NBA_MODEL_STATUS = DATA_BLOCKED`. Nothing in this pass changes that
status — it remains blocked, honestly, until real action/outcome-linked
data exists. The brief's own §8 caution applies in advance: even if such
data existed, it would be `OBSERVATIONAL_DATA`, not causal evidence, unless
the actions were genuinely randomized at collection time.

## Net effect on this project's terminology

No renaming was required in the *live application* — the audit in
`REPOSITORY_FORENSIC_AUDIT.md` §5 found the codebase already avoids
"recovery model" language for these targets. This document exists so the
distinction is stated once, explicitly, in one place, rather than left
implicit across scattered comments.
