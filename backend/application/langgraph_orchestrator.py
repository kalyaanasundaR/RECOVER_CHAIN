from typing import Any, TypedDict

# Mocking LangGraph StateGraph due to Windows Application Control DLL block on xxhash
START = "__START__"
END = "__END__"


class StateGraph:
    def __init__(self, state_schema):
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, source, target):
        self.edges[source] = target

    def add_conditional_edges(self, source, func, routing_map):
        self.conditional_edges[source] = (func, routing_map)

    def compile(self):
        return self

    def invoke(self, state):
        current_node = self.edges.get(START)
        while current_node and current_node != END:
            # Execute node
            state = self.nodes[current_node](state)

            # Find next node
            if current_node in self.conditional_edges:
                func, routing_map = self.conditional_edges[current_node]
                route = func(state)
                current_node = routing_map[route]
            else:
                current_node = self.edges.get(current_node)
        return state


import time

from application.action_evaluator import DeterministicActionEvaluator
from application.diagnosis_engine import DeterministicDiagnosisEngine
from domain.lifecycle import CaseLifecycleManager
from domain.models import (
    ActionRecommendation,
    CandidateAction,
    DiagnosisStatus,
    RecommendationStatus,
    RecoveryCase,
    RootCauseDiagnosis,
)


class GraphState(TypedDict):
    case: RecoveryCase
    mode: str
    llm: Any
    det_diag: DeterministicDiagnosisEngine
    det_rec: DeterministicActionEvaluator

    # Track metadata
    llm_latency_ms: int
    llm_tokens_used: int
    prompt_version: str
    model_identifier: str
    error_message: str | None


def load_case(state: GraphState):
    return state


def diagnose(state: GraphState):
    start = time.time()
    if state["mode"] in ["B", "D", "REAL_LLM_DIAGNOSIS", "REAL_LLM_FULL_REASONING"]:
        # LLM Diagnosis
        try:
            llm_out = state["llm"].diagnose(state["case"])
            import uuid

            diag = RootCauseDiagnosis(
                diagnosis_id=f"diag_llm_{uuid.uuid4().hex[:8]}",
                cause_category=llm_out.diagnosis_category,
                confidence=llm_out.confidence,
                status=DiagnosisStatus.CONFIRMED,
                supporting_signals={"reasoning": llm_out.reasoning_summary},
                evidence_references=llm_out.evidence_references,
                diagnostic_method=state["llm"].model_version,
            )
            state["case"].diagnosis = diag
        except Exception as e:
            state["error_message"] = str(e)
            # Fallback
            state["case"].diagnosis = state["det_diag"].diagnose(state["case"])
    else:
        state["case"].diagnosis = state["det_diag"].diagnose(state["case"])

    CaseLifecycleManager.record_diagnosis_result(state["case"])
    state["llm_latency_ms"] += int((time.time() - start) * 1000)
    return state


def predict(state: GraphState):
    # Requirement: "Do NOT allow the LLM to fabricate calibrated recovery probabilities. Keep existing baseline."
    from application.recovery_predictor import DeterministicBaselinePredictor

    pred = DeterministicBaselinePredictor().predict(state["case"])
    state["case"].prediction = pred
    return state


def route_specialist(state: GraphState) -> str:
    """Deterministic domain routing (Phase 15): shares application.specialist_agents's
    router. An unrecognised domain goes to `needs_mapping`, never to a default agent."""
    from application.specialist_agents import route_case

    agent = route_case(state["case"])
    return {
        "PaymentAgent": "payment_agent",
        "CheckoutAgent": "checkout_agent",
        "SubscriptionAgent": "subscription_agent",
        "InvoiceAgent": "receivables_agent",
        "PromiseToPayAgent": "promise_agent",
    }.get(agent.agent_id if agent else "", "needs_mapping")


def needs_mapping(state: GraphState):
    state["error_message"] = "NEEDS_MAPPING: no specialist registered for this domain"
    return state


def specialist_agent(state: GraphState):
    # This acts as the logic for all 5 specialists for simulation
    # In a real setup, each node might have different system prompts
    start = time.time()
    if state["mode"] in ["C", "D", "REAL_LLM_RECOMMENDATION", "REAL_LLM_FULL_REASONING"]:
        try:
            llm_out = state["llm"].recommend(state["case"])
            import uuid

            candidates = []
            for c in llm_out.candidates:
                erv = state["case"].amount_at_risk.amount * c.estimated_probability
                candidates.append(
                    CandidateAction(
                        action_type=c.action_type,
                        estimated_probability=c.estimated_probability,
                        expected_recoverable_value=erv,
                        rationale=c.rationale,
                    )
                )

            if candidates:
                candidates.sort(key=lambda x: -x.expected_recoverable_value)
                top = candidates[0]
            else:
                top = None

            rec = ActionRecommendation(
                recommendation_id=f"rec_llm_{uuid.uuid4().hex[:8]}",
                candidates=candidates,
                top_candidate=top,
                status=RecommendationStatus.RECOMMENDED
                if top
                else RecommendationStatus.INSUFFICIENT_EVIDENCE,
                rationale="LLM specialist derived recommendation",
                engine_version=state["llm"].model_version,
            )
            state["case"].recommendation = rec
        except Exception as e:
            state["error_message"] = str(e)
            state["case"].recommendation = state["det_rec"].evaluate(state["case"])
    else:
        state["case"].recommendation = state["det_rec"].evaluate(state["case"])

    CaseLifecycleManager.record_recommendation_result(state["case"])
    state["llm_latency_ms"] += int((time.time() - start) * 1000)
    return state


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("load_case", load_case)
    workflow.add_node("diagnose", diagnose)
    workflow.add_node("predict", predict)

    # 5 Specialists
    workflow.add_node("payment_agent", specialist_agent)
    workflow.add_node("checkout_agent", specialist_agent)
    workflow.add_node("subscription_agent", specialist_agent)
    workflow.add_node("receivables_agent", specialist_agent)
    workflow.add_node("promise_agent", specialist_agent)
    workflow.add_node("needs_mapping", needs_mapping)

    workflow.add_edge(START, "load_case")
    workflow.add_edge("load_case", "diagnose")
    workflow.add_edge("diagnose", "predict")

    workflow.add_conditional_edges(
        "predict",
        route_specialist,
        {
            "payment_agent": "payment_agent",
            "checkout_agent": "checkout_agent",
            "subscription_agent": "subscription_agent",
            "receivables_agent": "receivables_agent",
            "promise_agent": "promise_agent",
            "needs_mapping": "needs_mapping",
        },
    )

    workflow.add_edge("payment_agent", END)
    workflow.add_edge("checkout_agent", END)
    workflow.add_edge("subscription_agent", END)
    workflow.add_edge("receivables_agent", END)
    workflow.add_edge("promise_agent", END)
    workflow.add_edge("needs_mapping", END)

    return workflow.compile()
