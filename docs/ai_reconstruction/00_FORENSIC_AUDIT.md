# RecoverChain AI — Forensic Audit (Phase A only)

Scope: this document is the result of actually inspecting the repository — no
training was run, no claim below is inferred or assumed. Everything here is
verifiable at the cited path. This is Phase A of the requested master
reconstruction (sections 2–6 of the brief); sections 7 onward (rebuild,
retraining, DL, calibration, LLM/RAG audit-to-fix, 21 documents, viva pack)
are **not** attempted in this pass — see the note at the end.

## 1. LLM — actual runtime status

- **Live case pipeline (`application/agents.py`, `application/case_pipeline.py`):
  zero LLM references.** Grepping `agents.py` for any LLM adapter symbol
  returns nothing. The live path — diagnosis, recovery prediction,
  recommendation, policy, execution — is 100% deterministic Python. This
  matches every prior phase's stated invariant; today's audit independently
  confirms it from source, not from a prior report.
- **`application/langgraph_orchestrator.py:3`** — literal comment: *"Mocking
  LangGraph StateGraph due to Windows Application Control DLL block on
  xxhash"*. LangGraph is not actually running; it's a mock standing in for it.
  Anything described elsewhere as "LangGraph-orchestrated" is running the mock.
- **`application/real_llm_adapter.py`** — a real, functioning Gemini adapter
  (`RealGeminiAdapter`) exists: it calls
  `https://generativelanguage.googleapis.com/.../gemini-2.5-flash:generateContent`
  over `httpx`, with `responseSchema` constraining the model to structured
  JSON, and `_validate_evidence()` rejects any output whose evidence
  references don't correspond to real `event_id`s on the case (hallucination
  guard). This is real code, not a stub.
- **But it is only ever imported in `evaluation/runner.py:102` and one test**
  (`test_phase50_phase15_multi_agent.py:594`). It is never imported by
  `agents.py`, `case_pipeline.py`, or any API router. It cannot run in the
  live financial path even if a key is set, because nothing on the live path
  calls it.
- **`self.available = bool(os.environ.get("GEMINI_API_KEY"))`** —
  `.env.example:36` ships `GEMINI_API_KEY` commented out. On a stock checkout
  of this repo the adapter reports `available=False` and raises
  `REAL_LLM_UNAVAILABLE` if called.
- **Verdict:** `LLM_STATUS = CODE_EXISTS_NOT_WIRED_TO_LIVE_PATH,
  UNAVAILABLE_BY_DEFAULT (no key)`. Not "dead code" (it's real, tested,
  schema-validated, evidence-guarded) — but it is evaluation-harness-only,
  never a live decision-maker. No RAG/vector-store/embedding code exists
  anywhere in the backend (`faiss`, `chromadb`, `embedding` all return no
  hits outside this grep noise).

## 2. Model artifacts — actual registry inspection

Three **separate, unconnected** training lineages exist in this repo. This
is itself a finding: they do not share a registry.

### 2a. `backend/ml/models/registry/` — the "Phase 19" one-off run
- `failure_risk_tree.json` (131 KB) — a genuine XGBoost model, native
  XGBoost JSON format (not joblib), 16 features matching
  `billing_recovery_v3`'s columns exactly. Internal metadata:
  `best_iteration: 7`, `best_score: 0.2529503949964736` (this is the
  `eval_metric="logloss"` value XGBoost tracked during early stopping — not
  ROC-AUC, not accuracy).
- `features.json` — the 16 feature names, matches the tree file.
- **`ds_75100ad3_churn_run_metadata.json` — 0 bytes. Empty file.** No
  accuracy, no ROC-AUC, no split counts, no run date survive for this
  artifact anywhere I can find.
- Produced by `backend/run_phase19_training.py`, a standalone script (not
  wired to any API) that builds a spec by hand (`target_column:
  "target_recovered"`, features excluding identifiers/target/split) and calls
  `MLTrainingEngine` directly against `evaluation/datasets/billing_recovery_v3.csv`.
- **Verdict: `STATUS = ARTIFACT_EXISTS, PROVENANCE_INCOMPLETE`.** The model
  file is real and loadable (native XGBoost format), but no accuracy/ROC-AUC
  /precision/recall/calibration record survives for it, and it is not loaded
  by the live serving path (`recovery_predictor_ml.py` loads from a
  dataset-scoped `_registry_dir()`, not from `ml/models/registry/`). Treat
  as a research artifact from a prior exploration, not a deployed model.

### 2b. Live shadow model — `dataset_router.py` → `ml_training.py`
- This is the one actually wired to the UI's "recovery probability" number
  (confirmed in an earlier turn of this session, tracing
  `generate_cases_from_dataset()` → `run_ml_training_task()` →
  `MLTrainingEngine.train_and_evaluate()`).
- It retrains **per uploaded dataset**, on whatever the user just uploaded
  (or the demo sample), writing to `RECOVERCHAIN_MODEL_REGISTRY_DIR`
  (test/dev-scoped, not `ml/models/registry/`). There is no single
  "the model" — every dataset gets its own run, gated at ROC-AUC ≥ 0.55
  (`ML_MIN_ROC_AUC`) and ≥ 200 test rows (adaptively relaxed to 15 for small
  real files), else flagged `REJECTED_LOW_QUALITY` and never served.
- **Verdict:** this is legitimate, honestly-gated shadow ML — but it is
  trained on whatever a user uploads (often 15–120 rows in this session's
  own demo/sandbox runs), so its metrics are only as meaningful as that
  file. It is architecturally correct; it is not evidence of a validated
  production model, and was never meant to be (the code says so —
  `lifecycle_status` never reaches `PRODUCTION_VALIDATED` anywhere in this
  pipeline; see `ml_training.py:443-449`'s own comment).

### 2c. `evaluation/` — the offline evaluation harness
- `evaluation/runner.py`'s `EvaluationRunner` explicitly raises
  `OperationalStoreError` if constructed against the live operational DB
  session — it is contractually forbidden from touching production data
  (confirmed by `test_9b_evaluation_cannot_touch_the_operational_store` in
  `test_phase51_phase16_traceability.py`).
- It carries its own datasets (`evaluation/datasets/*.csv`) and its own
  scenario definitions (`evaluation/scenarios.py`), and is the only caller
  of `RealGeminiAdapter`. Its purpose is agent/LLM/policy scenario
  evaluation, not model training — it does not itself call
  `MLTrainingEngine`.

## 3. Dataset catalog — what's actually on disk

| Location | What it is | Real training value |
|---|---|---|
| `backend/dataset/` (55+ files) | The **operational upload store** — every file ever uploaded through the running app across every demo/test session in this project's history (`ds_<id>_<name>.xlsx/csv`), plus stray test dedup copies (`test_dummy_*`). This is app *state*, not a curated corpus. | None as a training set — it's demo/smoke-test exhaust, not a dataset choice. |
| `backend/evaluation/datasets/ar_recovery_v1.csv` / `v2.csv` | 2,467 rows, accounts-receivable invoice lateness. Has a real, already-written provenance doc (see below) — `target_late`, pre-split `train/validation/test` column, engineered `prior_late_rate` etc. | **Real, documented, ready for classical-ML retraining.** |
| `backend/evaluation/datasets/billing_recovery_v1.csv` / `v2.csv` / `v3.csv` | **3.6M rows each**, synthetic-labeled telecom billing. `billing_recovery_v3_metadata.json` (293 lines, already exists — not written by me) documents: training unit (`subscriber_id + year_month`), target definition, an explicit **deduplication + conservative-labeling rule**, and — critically — a documented **caveat**: *"89% of subscribers with Failed/Unpaid status in any month later show successful payment in subsequent months. The target captures per-cycle payment failures, which are mostly temporary."* Provenance is honestly marked `"UNKNOWN — appears synthetic telecom billing data"`. | **Real, large, already-documented — but the dataset's own metadata admits the label is noisy/near-random at the level prior phases found** (matches the master brief's own §69 expectation for `billing_recovery_v1`). This is exactly the kind of "weak-signal dataset, don't manufacture a good result" case the brief calls out. |
| `data/test_datasets/*.xlsx`, `docs/history/phase6_datasets/*`, `docs/history/phase8b_external_source_datasets/*`, `sample_datasets/*` | Small (~hundreds of rows) synthetic fixtures built per-domain (payment failure, checkout abandonment, subscription failure, invoice overdue, promise-to-pay) for earlier phases' end-to-end and domain-classification tests. | Fine for pipeline/integration testing; too small and synthetic for a defensible trained-model claim on their own. |
| `backend/bank_flow_*.csv`, `leak.csv`, `t*.csv`, `temp_*.csv` (repo root) | Untracked scratch files from ad-hoc testing sessions. | None — candidates for cleanup, not data. |

**The two evaluation-dataset families (`ar_recovery_*`, `billing_recovery_*`)
are the only real candidates for the classical-ML retraining ladder** (§17-20
of the brief) — they're the only datasets in this repo with row counts,
documented targets, and an already-written leakage/provenance statement.
Nothing in this repo currently supports a genuine sequence/GRU/LSTM task
(§23-24) — I found no dataset with a per-entity chronological event sequence
of meaningful length; `ar_recovery_v1` and `billing_recovery_v3` are one-row-
per-obligation/cycle, not sequences. A Transformer (§25) has no justified
use case here today for the same reason.

## 4. What this audit does *not* attempt

This is Phase A only. It does **not** include: label/feature rebuild,
leakage re-audit against the pipeline's actual code (only the dataset's own
prior documentation was read), baseline-through-XGBoost training with a
temporal split and held-out test, calibration, DL, error analysis,
financial-cost analysis, an LLM/RAG rebuild, or the 21-document set. Those
are sections 7–90 of the brief and are a materially different scope of work
(real training runs against a 3.6M-row file, multiple model families,
multiple documents) than a single audit pass.

**Recommendation:** scope §7 onward as its own explicit, bounded piece of
work — most defensibly against `ar_recovery_v1/v2` first (small, clean,
already documented, fast to iterate), with `billing_recovery_v3` attempted
second and reported honestly if it stays near-random, exactly as the
brief's own §69 anticipates. Trying to do both at 3.6M-row scale plus DL
plus LLM/RAG plus 21 documents in one continuous pass is how the "no
fabricated ML" rule gets violated by accident — not by intent, but by
running out of room to actually verify each claim before writing it down.
