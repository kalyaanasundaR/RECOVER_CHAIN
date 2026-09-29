# Phase 15 — Multi-Agent Orchestration, Agent Evaluation & Bounded AI

**Scope:** RecoverChain AI only. **Date:** 2026-09-19. **Prior phase:** 14 (not re-run; nothing from it was rebuilt).

## Executive summary

The read-only audit found that the "multi-agent" layer was thinner than its name suggests, and that a framework's presence was misleading:

- The five "agents" in `application/agents.py` are **deterministic execution-preparation classes** (`can_handle(action)` → `prepare_execution` → params). They sit at the *end* of the pipeline, are chosen by **action**, and contain no reasoning.
- The file `application/langgraph_orchestrator.py` is **not LangGraph**. It is a hand-written mock `StateGraph` (its own header says LangGraph was blocked by a Windows Application Control DLL policy), it is not installed as a dependency, it is used **only by the evaluation harness**, and all five "specialist" nodes are the *same function*.
- There is **no LLM in the live path**. `SimulatedLLMAdapter` is hard-coded canned output with `sleep(0.2)`; `RealGeminiAdapter` (httpx, `GEMINI_API_KEY`) is reachable only from `evaluation/runner.py`.

This phase therefore did **not** add an LLM or a framework. It made the existing deterministic layer real, structured, bounded and traceable: a strict agent contract, five domain-bound specialist agents behind a restricted toolbox, deterministic routing with `NEEDS_MAPPING`, a bounded run state machine, structured handoffs, a safe runner (timeout / validation / cross-check → deterministic fallback), audit-based tracing, an evaluation contract with a synthetic benchmark, and a real routing defect fix at the executor. Full backend regression, Ruff and frontend typecheck/lint are clean; 56 new tests.

## 1. Read-only audit

| Item | Verdict | Evidence |
|---|---|---|
| Executor classes (`PaymentRecoveryAgent`…`PromiseRecoveryAgent`) | VERIFIED — deterministic service classes used live at `POST /cases/{id}/execute` | `agents.py` |
| `AgentOrchestrator` (policy gate, mismatch check, slot-claim idempotency, ledger) | VERIFIED — the real authorized-execution adapter boundary | `agents.py:execute` |
| Policy revalidation before execution | VERIFIED (Phase 13) | `api/main.py:918-978` |
| "LangGraph" | **DEAD/UNUSED in live path; mock** — no dependency, eval-only, one function for five nodes; fell back to `payment_agent` for unknown domains | `langgraph_orchestrator.py`, `requirements.txt` |
| LLM adapters | Simulated = canned/demo; Real = eval-only. **Not in live path** | `llm_adapters.py`, `real_llm_adapter.py` |
| Agent state / agent contract / tool boundaries / handoffs | **MISSING** → built | this phase |
| Routing | PARTIALLY IMPLEMENTED: action-based at execution; category-based only in the mock graph; **wrong-specialist defect** (below) | `agents.py` |
| Decision context, candidates, ML prediction, explanation, anomaly | VERIFIED (Phase 12–13) — reused | `decision_context.py`, `action_evaluator.py` |
| Audit trail | VERIFIED — reused for tracing | `AuditModel` |
| Agent tests | PARTIALLY IMPLEMENTED (one execution test) → 56 added | `test_execution.py` |

### Real defect found and fixed — wrong specialist at execution
`REQUEST_PAYMENT_METHOD_UPDATE` is valid for both `FAILED_PAYMENT` and `FAILED_SUBSCRIPTION`. Executor selection took the **first** agent whose `can_handle(action)` was true, so a subscription case's action was prepared by `PaymentRecoveryAgent`. Executors now carry `domains`, selection is by *(domain, action)*, `SubscriptionRecoveryAgent` handles that action for subscriptions, and `PromiseRecoveryAgent` handles `ESCALATE_COLLECTION` (valid for `BROKEN_PROMISE` per `ACTION_SPECS`, previously unowned by any promise executor). `test_4d` asserts every non-internal `(action, domain)` pair in `ACTION_SPECS` has **exactly one** owning executor. Confirmed live: a subscription mandate-failure case executed via `SubscriptionRecoveryAgent`.

## 2. Responsibility boundaries
Specialists (`specialist_agents.py`) are bound one-per-domain: Payment↔`FAILED_PAYMENT`, Checkout↔`CHECKOUT_ABANDONMENT`, Subscription↔`FAILED_SUBSCRIPTION`, Invoice↔`OVERDUE_INVOICE`, PromiseToPay↔`BROKEN_PROMISE`. Their permitted actions are **derived from `ACTION_SPECS`**, not re-listed. They implement no consent, suppression, retry or contact logic; they only *list* the Policy Engine checks that will apply (`policy_prerequisites`). Executors remain a separate layer downstream of policy.

## 3–4. Agent contract and structured output
`agent_contract.py`: `AgentRecommendation` (strict, frozen, `extra="forbid"`) carries agent id/version, case id, domain, candidate actions, recommended action, model reference, recovery probability, ERV, anomaly, explanation (`MODEL_ASSOCIATION` items only), closed-vocabulary `reason_codes`, `policy_prerequisites`, status, timestamp and `trace_id`. It has **no** amount, recovered-amount, approval, policy-status or outcome field. `validate_agent_output` accepts only a model, a dict, or a JSON *object*; prose is rejected, never string-parsed into an action. Unknown fields (`tool_calls`, `amount_at_risk`, `recovered_amount`, `human_approved`, `policy_status`, `outcome_event`), out-of-domain actions, non-candidate recommendations, NaN/out-of-range probabilities and invalid enums are all rejected (tests 2–2f). `rationale` is display-only, capped at 500 chars.

## 5. Tool permissions
Registry of 11 tools classified `READ_ONLY` / `RECOMMENDATION` / `POLICY` / `EXECUTION` / `OUTCOME`. `AgentToolbox` allows only READ_ONLY + RECOMMENDATION, denies POLICY/EXECUTION/OUTCOME and unknown names, and is bound to a **deep copy** of the case (tests 3–3c). An AST test proves neither agent module imports the executor, policy engine, outcome writer, verifier, repositories or adapters.

## 6. Agent → Policy boundary
Verified by tests 7–7g: an agent proposes a suppressed customer's retry and the Policy Engine returns `DENIED`; execution is `REJECTED` with `adapter_used: NONE`. Consent, retry cap and the ₹5,000 approval threshold are unchanged by any agent output; a mismatched action is `REJECTED_MISMATCH`; an agent cannot approve itself (contract rejects `human_approved`; `PolicyContext` is `extra="forbid"`); amounts, recovered amount and state are unchanged after an agent run; a settled obligation yields `NO_ACTION` and policy `DENIED`.

## 7. Routing
`route_case` is a deterministic domain→specialist map. Unknown domain → `None` → `NEEDS_MAPPING` recommendation with no candidates and a terminal `NEEDS_MAPPING` run (test 4b). The mock graph now delegates to the same router and has a `needs_mapping` terminal node instead of defaulting to the payment agent. Specialist and executor domains are asserted consistent (test 4e).

## 8. State machine
`AgentRun`: RECEIVED → CONTEXT_BUILT → ANALYZED → RECOMMENDATION_READY → POLICY_REVIEW → {APPROVAL_REQUIRED | APPROVED | REJECTED} → EXECUTING → VERIFYING → COMPLETED (+ FAILED / NEEDS_MAPPING). APPROVED/APPROVAL_REQUIRED/REJECTED can only be entered through `apply_policy_decision(PolicyDecisionStatus)`; human review re-enters only via `apply_human_review(<fresh revalidation status>)`; `advance()` refuses them and any skip; terminal states stay terminal (tests 5–5d).

## 9. Handoffs
`Handoff` is a schema-validated record (source, destination, case, reason, payload schema, trace id, timestamp) with `grants_authorization: Literal[False]`. The only destination is `CONTEXT_LAYER`; agent→agent handoff, unknown payload schemas and extra payload fields are rejected; each handoff serialises to audit evidence (tests 6–6b). **Status: contract and validator implemented and tested; not yet invoked by a live agent** — no current agent needs cross-context information the decision context does not already supply.

## 10. LLM integration status
**No LLM in the live path; none added.** Adding one would be marketing, not capability: the specialists' job is deterministic selection among already-validated candidates. Existing LLM code remains evaluation-only; its outputs are schema- and evidence-checked (hallucinated evidence references are rejected in `RealGeminiAdapter`). Design note for any future LLM: it would sit behind `validate_agent_output` and the same toolbox, with the same fallback.

## 11. Explainability layers
Kept distinct in the contract/audit/trace: `explanation` = MODEL ASSOCIATION (feature + direction only, no values); `reason_codes` + `rationale` = AGENT REASONING (closed vocabulary derived from real fields); policy decision = Policy Engine (separate stage); outcome = authoritative events only. No causal claims.

## 12–13. Evaluation dataset and benchmark
`AgentEvalRecord` carries case, domain, context version, agent/model version, recommendation, candidates, policy result, execution result, outcome, latency, errors, provenance. `run_agent_benchmark()` (also `GET /system/agent-benchmark`) runs 19 deterministic scenarios across all five domains through the real router, agents, Policy Engine and simulated executor. Expected policy outcomes were written before running and **not tuned afterwards**.

| Metric | Result |
|---|---|
| Scenarios | 19 (payment 5, checkout 3, subscription 3, invoice 4, promise 4) |
| Routing accuracy | 1.00 |
| Valid structured-output rate | 1.00 |
| Policy as expected | 1.00 |
| Unsafe recommendation rate (out-of-domain, or executed without PERMITTED) | 0.00 |
| Execution-block rate | 0.526 (10/19 blocked by policy, as scripted) |
| Outcome correctness | NOT_APPLICABLE_SYNTHETIC |
| Latency | mean 0.68 ms, p95 1.56 ms |

**Caveats, stated plainly:** every record is `provenance = SYNTHETIC_SCENARIO`. This measures wiring under scripted inputs — not production agent quality. 100% rates on scenarios I wrote are a consistency check, not evidence of generalisation. Promise "created/due/fulfilled/broken" are outcome-event kinds rather than distinct case states, so those scenarios approximate them. No recovery was produced.

## 14. Adversarial tests
Covered and passing: bypass policy, unauthorized/mismatched action, ignore suppression, ignore consent, exceed retry cap, self-approval, skip human approval, alter amount, fabricate/mark recovery, create an outcome, invalid action, illegal state transitions, injected tool calls, free-text-as-command. All end in BLOCKED / REJECTED / fallback.

## 15. Idempotency
Real API: 4 identical `/execute` calls → identical `COMPLETED_SIMULATED` replay and exactly **1** `execution_attempts` row (test 9). Orchestrator level: 3 claims on one key → adapter called once (test 9b). The existing slot-claim ledger stays authoritative; nothing was added there.

## 16. Failure recovery
Agent exception, malformed (prose) output, timeout, and an agent disagreeing with the deterministic evaluator → `FALLBACK` recommendation, run `FAILED`, case byte-identical (test 8). Policy-service error → fails closed, no policy decision, execution `REJECTED` (8b). Database failure at slot claim → adapter never called (8c). Adapter error → `FAILED` record, no recovery recorded (8d). LLM unavailable → normal, no effect (8e). Model missing/corrupt/invalid probability: covered by Phase 11–12 tests, unchanged. Stale/duplicate: policy freshness/revalidation and idempotency (above).

## 17. Observability
Each recommendation writes one `agent_recommendation` audit record (trace id, agent id/version, contract version, domain, status, action, probability, ERV, model version, reason codes, prerequisites, run states, latency, whether it agrees with the deterministic evaluator) via the existing audit recorder. `GET /system/cases/{id}/agent-trace` reconstructs CASE → MODEL → RECOMMENDATION → AGENT → POLICY → APPROVAL → EXECUTION → OUTCOME from audit + policy decision id + execution id. **Limitation:** the `trace_id` is stamped on the agent step; later stages are correlated by case id and their own ids, not by `trace_id`.

## 18. Performance (per call, warm)
routing 0.0001 ms · `agent.analyze` 0.082 ms · `run_specialist` (thread + validation + cross-check) 0.352 ms · policy 0.018 ms · total decision (context + agent + policy) 0.393 ms. Model inference 9.2 ms and explanation 84.9 ms are unchanged from Phase 12. LLM latency/timeouts: not applicable.

## 19. Security
No LLM in the live path, so no prompt-injection surface there. Agent output cannot become a tool call (closed toolbox, strict schema, no free-text parsing). Agents never see raw feature values in audit output. `RealGeminiAdapter` sends its key in a header, not the URL (pre-existing). **Gaps, not fixed:** no tenant model exists in this codebase (single-tenant), so tenant isolation is UNKNOWN/N/A; the shared dev API key (`test-api-key`) remains the mutating-endpoint auth and the new read-only endpoints (`agent-trace`, `agent-benchmark`) are unauthenticated like other `/system` reads.

## 20. Cache cleanup
Not required; no stale cache interfered. Backend restarted manually (no `--reload`), frontend dev server restarted after it had stopped. No database touched.

## 21. Tests
- **New:** `tests/test_phase50_phase15_multi_agent.py` — 56 tests (contract, structured output, tools, routing, executor uniqueness, state machine, handoffs, adversarial, failure recovery, idempotency, trace, benchmark, import boundary).
- **Full backend regression:** all pass. `ruff check` and `ruff format --check`: clean. Frontend `typecheck`: clean; `lint`: 0 errors, same 13 pre-existing warnings.
- No existing test was modified or weakened. Note: running `test_evaluation.py` *before* API tests in a hand-picked subset produced "no such table" errors — a pre-existing order dependence, absent in normal suite order. An intermittent Windows Application Control block on a scipy DLL also caused one collection failure that cleared on re-run; environmental, not code.

## 22. Browser verification (real app)
- **A/B/C/D:** `p15_B_high` (₹20,000): "Handled by PaymentAgent (v1.0.0 · proposes only — the Policy Engine decides · trace_36f53a90f506)"; prediction ~85% and ERV ₹17,080 (deterministic baseline; no ML model registered for these events); "Needs approval".
- **E/F:** clicked the real **Approve & run** → "Approved", "Action taken: Done (simulated)".
- **G:** "Result: Awaiting an outcome — … Running the action is not by itself money recovered."
- **H:** trace shows the ordered chain; no OUTCOME stage.
- **Suppressed customer** (`p15_C_supp`): identical recommendation; "Blocked", "Action taken: REJECTED".
Explanations do not appear for these cases because they used the baseline predictor, not an ML shadow model.

## VERIFIED
Agents cannot execute, authorize, create outcomes or change amounts by any tested path; routing is deterministic with `NEEDS_MAPPING`; wrong-specialist defect fixed with a uniqueness test; structured-only output; fallback on every agent failure; idempotent execution; audit-based trace; full regression clean.

## UNKNOWN
Real-world agent quality (no production data); tenant isolation; behaviour under a real LLM (none integrated); trace correlation beyond case/decision/execution ids.

## BLOCKERS
None.

## Production validation status
`NOT_READY` — unchanged. Benchmarks are synthetic.

## NBA status
`DATA_BLOCKED` — unchanged. No labels created.

## Phase 16 recommendation
**GO** for hardening this layer: stamp `trace_id` onto policy/execution audit records; authenticate the read-only trace/benchmark endpoints; add real handoff use only if a concrete need appears.
**NO-GO** for: adding an LLM to the live path without a stated need and an evaluation showing it beats the deterministic path; any claim of agent production performance; NBA training; production-validation claims.
