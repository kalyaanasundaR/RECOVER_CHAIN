# RecoverChain — Repository Forensic Audit

Real, executed audit — every classification below was checked against actual
imports/references before any file was touched, per the brief's own
dependency-verification rule. Nothing was deleted without confirming zero
references first (`Grep` across the whole repo). This document covers §1-2
and §33 (partial) of the reconstruction brief; §4-16 (new synthetic
datasets, ml/ pipeline restructure) are separate, larger work — not
attempted here, see the closing section.

## Classification vocabulary used

`KEEP` / `REPAIR` / `MERGE` / `REPLACE` / `ARCHIVE` / `DELETE` / `DATA_BLOCKED`

## 1. Actually deleted (verified zero references anywhere in the repo)

| Path | What it was | Why safe |
|---|---|---|
| `audit_report.html` | A generated report artifact (styled HTML, not source) at repo root | No code references it; not gitignored but purely generated |
| `backend/p16_cases.txt` | Stray debug dump (case-id list) | No code references it |
| `backend/bank_flow_1788106527515.csv`, `bank_flow_1788106565512.csv`, `synthetic_bank_1788106487768.csv`, `leak.csv`, `t.csv`, `t_full.csv`, `temp_alpha.csv`, `temp_test.csv` | Ad-hoc scratch CSVs from manual testing sessions at repo/backend root | Already covered by `.gitignore`'s `backend/*.csv`; zero code references; genuine session debris |
| `backend/scripts/_sandbox_output/`, `backend/scripts/_sandbox_sample.csv` | My own smoke-test leftovers from an earlier turn's sandbox-script demo | Served their purpose (proved `train_sandbox.py` works); regenerable in seconds |
| `frontend/dist/` | Stale Vite build output | Gitignored, regenerable via `npm run build`, not used by the dev server (`preview_start` runs `vite` dev, not the built output) |
| `backend/**/__pycache__/`, `backend/.pytest_cache/` | Python bytecode/test caches | Purely generated |

**Verification after deletion:** full backend test suite re-run — 617 passed
(one flake on the first run, `test_generate_cases_risk_category_payload_and_export`,
reproduced as passing in isolation and on a clean rerun — this project's
already-documented intermittent Windows Application Control / scipy
collection issue, unrelated to this cleanup).

## 2. Archived, not deleted (real evidence, kept per the "archive, don't silently destroy" rule)

| Path (new) | What it was | Why archived not deleted |
|---|---|---|
| `backend/archive/db_backups/recoverchain.db.bak_phase11`, `.bak_pre_phase16`, `.stale_pre_phase11` | Manual pre-migration DB snapshots | Not gitignored (`*.db` doesn't match a `.bak_*` suffix), sitting as untracked root clutter — but this project has a standing rule to never delete a DB speculatively. Moved to a clearly-labelled archive folder with a README explaining when it's actually safe to delete for real. |

## 3. Found already correctly archived — no action needed

`scripts/legacy/` **already exists**, with its own README explicitly
stating these are one-off historical scripts, not part of the application,
not run by CI. This is exactly the `ARCHIVE` discipline the brief asks for
— it was already done by a prior phase. Contents include the actual
dataset-construction scripts for `ar_recovery_v1/v2` and
`billing_recovery_v1/2/3` (`construct_training_data.py`,
`phase16c_billing_constructed.py`, `phase16d_construct_v2.py`/`v3.py`,
bias/dedup checks) and a prior training run, `phase17_train.py` (this
folder's own internal numbering — unrelated to the current
`docs/history/PHASE_17_*` authorization work), whose recorded output
(`phase17_train_out.txt`) **already correctly named the target "Payment
Failure Risk," not recovery**, and independently found the same
near-random result (ROC-AUC ≈ 0.545) that this reconstruction's
`ml_reconstruction/train_billing_recovery.py` reproduced from scratch this
turn. One stale cross-reference in that README (pointing only to
`run_phase19_training.py` as "the" training entry point) was corrected to
also point at the newer, fuller `ml_reconstruction/` scripts.

**Verdict: `KEEP` as-is**, with the one README pointer fix applied.

## 4. Model artifact classification (formalizing last turn's audit)

| Path | Classification | Why |
|---|---|---|
| `backend/ml/models/registry/failure_risk_tree.json`, `features.json` | `ARCHIVE` (already gitignored, no action needed) | Real trained XGBoost artifact, but its metadata sidecar is 0 bytes — no accuracy/ROC-AUC/split-count survives. Not loaded by the live serving path. Superseded, with full metrics, by this turn's `train_billing_recovery.py` run. |
| `backend/ml/models/registry/ds_75100ad3_churn_run_metadata.json` (0 bytes) | `DELETE` (candidate) | Empty file, carries no information. Left in place this pass since it's gitignored and harmless, but it should not be treated as evidence of anything. |
| `backend/dataset/` (55+ uploaded files) | `KEEP` | This is the app's live operational upload store, not code cruft — already has its own retention policy (`DATASET_KEEP` env var, `_prune_stale_datasets()` in `dataset_router.py`). Not a cleanup target. |

## 5. LLM / agent code — formal terminology check (§33)

Grepped the whole backend for `causal`, `reinforcement learning`,
`LLM-powered`, `real-time AI`, `autonomous agent`, `LLM agent`. **Every
hit found is already a correct, defensive disclaimer**, not a misleading
claim — e.g. `agent_contract.py:137`: *"MODEL ASSOCIATION only. Never a
causal claim."*; `offline_policy_evaluation.py:10-11`: *"never a causal
claim (Section 7's distinction: PREDICTED OUTCOME vs CAUSAL EFFECT
applies)"*. No instance of "reinforcement learning," "LLM-powered," or
"real-time AI" describing something that isn't, was found anywhere in the
backend. `docs/history/architecture_phase13.md` and `architecture_phase13a.md`
say "LangGraph Orchestration" — confirmed accurate at the time those docs
were written for the mocked orchestrator that exists today
(`langgraph_orchestrator.py:3`: *"Mocking LangGraph StateGraph"* — the docs
correctly describe what's there, a deterministic mock standing in for
LangGraph, not a false "real LangGraph" claim). No fabricated accuracy
figures (98%/99%/95%) were found in application code, frontend, or current
docs — the one 98%+ ROC-AUC number that exists anywhere in this project is
this session's own sandbox smoke-test on synthetic toy data, already
labeled as such in the conversation, not written into any doc as a project
result.

**Verdict: the terminology-hygiene problem the brief anticipates does not
actually exist in this codebase.** This project's prior phases were
already disciplined about this. No rewrite pass was needed.

## 6. What this audit does not cover

Per the brief's own §4-16, building six new "Strong" synthetic XLSX
datasets (Payment Failure, Recovery Outcome, Action Outcome, MultiDomain,
Ethical Edge Cases, Guide Demo) with a documented generator script, a
restructured `ml/training/` pipeline, an ethics policy document, and the
remaining 13 of 15 requested documents are **not attempted in this pass**
— see the closing report below for why, and what real data would be
needed to do the Action-Outcome dataset honestly (§8's own caution:
`OBSERVATIONAL_DATA`, not causal evidence, unless genuinely randomized).
